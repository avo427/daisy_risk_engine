import logging
import yaml
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t
from datetime import datetime

# === Load Config ===
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Get Proxy for Ticker ===
def get_proxy(ticker, config):
    theme = config["ticker_themes"].get(ticker)
    proxy = config["theme_proxies"].get(theme)
    return proxy or config["fallback_proxy"]

# === Fit GARCH(1,1)-t and simulate returns ===
def simulate_garch_returns(proxy_returns, n, random_state=None):
    proxy_returns = proxy_returns.dropna()
    if len(proxy_returns) < 100:
        raise ValueError("Not enough data to fit GARCH model")

    model = arch_model(proxy_returns * 100, vol='Garch', p=1, q=1, dist='t')

    try:
        res = model.fit(disp="off")
    except Exception as e:
        raise RuntimeError(f"GARCH model fitting failed: {e}")

    # Set random state for reproducibility
    rng = np.random.default_rng(seed=random_state)

    sim_vol = np.zeros(n)
    sim_ret = np.zeros(n)

    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    nu = res.params['nu']
    cond_var = res.conditional_volatility.iloc[-1] ** 2

    for i in reversed(range(n)):
        z = t.rvs(df=nu, random_state=rng)
        sim_vol[i] = np.sqrt(cond_var)
        sim_ret[i] = sim_vol[i] * z / np.sqrt(nu / (nu - 2))
        cond_var = omega + alpha * (sim_ret[i] ** 2) + beta * cond_var

    return sim_ret / 100  # convert back to returns in decimal

# === Reconstruct Price Series ===
def reconstruct_price_series(ticker, series, proxy_series, random_state=None):
    first_valid = series.first_valid_index()
    if first_valid is None:
        logging.warning(f"Ticker {ticker} has no data at all.")
        return series  # skip

    missing_dates = series.loc[:first_valid].iloc[:-1].index
    n_missing = len(missing_dates)

    if n_missing == 0:
        return series  # nothing to backfill

    try:
        proxy_returns = proxy_series.pct_change().dropna()
        simulated_returns = simulate_garch_returns(proxy_returns, n_missing, random_state=random_state)
    except Exception as e:
        logging.warning(f"⚠️ GARCH simulation failed for {ticker}: {e}")
        return series

    # Walk prices backwards from anchor
    anchor_price = series.loc[first_valid]
    backfilled = pd.Series(index=missing_dates, dtype=float)
    price = anchor_price

    for i, date in enumerate(reversed(missing_dates)):
        price = price / (1 + simulated_returns[i])
        backfilled.loc[date] = price

    # Merge and return full series
    return pd.concat([backfilled, series.loc[first_valid:]])

# === Main Entry Point ===
def reconstruct_missing_prices(raw_prices, portfolio_tickers, config_path="config.yaml"):
    config = load_config(config_path)
    random_state = config.get("user_settings", {}).get("random_state", None)

    prices = raw_prices.copy()
    all_tickers = list(portfolio_tickers)

    for ticker in all_tickers:
        if ticker not in prices.columns:
            logging.warning(f"Ticker {ticker} not in price matrix.")
            continue

        proxy = get_proxy(ticker, config)
        if proxy not in prices.columns:
            logging.warning(f"No proxy data found for {proxy} (used by {ticker})")
            continue

        orig = prices[ticker]
        proxy_series = prices[proxy]
        backfilled = reconstruct_price_series(ticker, orig, proxy_series, random_state=random_state)
        prices[ticker] = backfilled

        logging.info(f"✅ Reconstructed: {ticker} using proxy {proxy}")

    # Save final backfilled price matrix
    output_path = config["paths"]["recon_prices_output"]
    prices.to_csv(output_path, float_format="%.4f")
    logging.info(f"✅ Saved reconstructed prices: {output_path} (shape={prices.shape})")

    return prices
