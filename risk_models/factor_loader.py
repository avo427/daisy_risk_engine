import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from pathlib import Path

def load_config(config_path=None):
    if config_path is None:
        # Resolves config.yaml from the project root
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_price_data(tickers, start_date, end_date, use_total_returns=True):
    # Remove threads=True to avoid race conditions and rate limiting issues
    raw = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False, threads=False)

    if raw.empty:
        raise ValueError("Yahoo Finance download returned no data.")

    price_col = 'Adj Close' if use_total_returns else 'Close'
    collected = {}
    skipped = []

    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            try:
                collected[tk] = raw[(tk, price_col)]
            except KeyError:
                skipped.append(tk)
                print(f"WARNING: Skipping {tk}: missing '{price_col}' column.")
    else:
        try:
            collected[tickers[0]] = raw[price_col]
        except KeyError:
            raise KeyError(f"'{price_col}' column not found in single ticker download.")

    if skipped:
        print(f"WARNING: Skipped {len(skipped)} tickers: {', '.join(skipped)}")

    # Validate that we actually collected some data
    if not collected:
        raise ValueError("No price data was successfully collected. Check your ticker symbols and internet connection.")

    df = pd.DataFrame(collected)
    return df.ffill().bfill()

def build_standard_factors(factor_config, price_data):
    factor_returns = {}
    for name, meta in factor_config.items():
        proxy = meta['proxy']
        transform = meta.get('transform', 'log_return')

        if proxy not in price_data.columns:
            print(f"WARNING: Skipping {name}: proxy '{proxy}' not found in downloaded data.")
            continue

        prices = price_data[proxy]

        if transform == 'log_return':
            factor_returns[name] = np.log(prices).diff()
        elif transform == 'pct_change':
            factor_returns[name] = prices.pct_change()
        elif transform == 'zscore':
            rolling = prices.rolling(60)
            rolling_std = rolling.std()
            # Handle division by zero in zscore calculation
            if rolling_std.min() == 0:
                print(f"WARNING: Skipping {name}: constant price series (zero standard deviation)")
                continue
            factor_returns[name] = (prices - rolling.mean()) / rolling_std

    return pd.DataFrame(factor_returns)

def build_thematic_factors(theme_config, price_data, market_caps=None):
    thematic_returns = {}
    for theme, meta in theme_config.items():
        tickers = meta['tickers']
        weights_mode = meta.get('weights', 'equal')

        available = [tk for tk in tickers if tk in price_data.columns]
        if len(available) < 2:
            print(f"WARNING: Skipping {theme}: insufficient valid tickers found.")
            continue

        weights = np.ones(len(available)) / len(available)
        if weights_mode == 'market_cap' and market_caps:
            weights = np.array([market_caps.get(tk, 1) for tk in available])
            weights_sum = weights.sum()
            # Handle division by zero in weight normalization
            if weights_sum > 0:
                weights /= weights_sum
            else:
                print(f"WARNING: Skipping {theme}: zero market cap sum, using equal weights")
                weights = np.ones(len(available)) / len(available)

        returns = price_data[available].pct_change()
        thematic_returns[theme] = (returns @ weights)

    return pd.DataFrame(thematic_returns)

def main():
    config = load_config()
    years = config['user_settings']['years']
    use_total_returns = config['user_settings'].get('total_returns', True)

    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)

    std_factors = config['factors']['standard']
    thematic = config['factors'].get('thematic', {})

    std_tickers = [meta['proxy'] for meta in std_factors.values()]
    thematic_tickers = [tk for meta in thematic.values() for tk in meta['tickers']]
    all_tickers = list(set(std_tickers + thematic_tickers))

    price_data = download_price_data(all_tickers, start_date, end_date, use_total_returns)

    if price_data.empty:
        raise ValueError("No valid price data available for factor construction.")

    std_returns = build_standard_factors(std_factors, price_data)
    thematic_returns = build_thematic_factors(thematic, price_data)
    all_returns = pd.concat([std_returns, thematic_returns], axis=1).dropna()

    # Respect config.yaml path fully, resolved relative to project root
    base_path = Path(__file__).resolve().parent.parent
    output_path = base_path / config["paths"]["factor_returns"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_returns.to_csv(output_path)
    print(f"SUCCESS: Factor returns saved to {output_path}")

