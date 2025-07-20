import logging
import yaml
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t

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
    res = model.fit(disp="off")

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

    return sim_ret / 100  # convert to decimal

# === Reconstruct Price Series ===
def reconstruct_price_series(ticker, series, proxy_series, proxy_returns, random_state=None):
    first_valid = series.first_valid_index()
    if first_valid is None:
        logging.warning(f"Ticker {ticker} has no data at all.")
        return series, pd.Series(index=series.index, dtype=object)

    missing_dates = series.loc[:first_valid].iloc[:-1].index
    n_missing = len(missing_dates)
    if n_missing <= 1:
        sources = pd.Series("REAL", index=series.loc[first_valid:].index)
        return series, sources

    backfilled = pd.Series(index=missing_dates, dtype=float)
    sources = pd.Series(index=missing_dates, dtype=object)
    anchor_price = series.loc[first_valid]
    price = anchor_price

    # Step 1: Try using proxy returns aligned by date
    usable_dates = []
    for i in range(n_missing - 1):
        date = missing_dates[-(i + 1)]  # reversed order
        if date in proxy_returns.index:
            r = proxy_returns.loc[date]
            price = price / (1 + r)
            backfilled.loc[date] = price
            sources.loc[date] = "PROXY"
            usable_dates.append(date)
        else:
            logging.warning(f"Proxy return missing on {date} for {ticker}, switching to GARCH.")
            usable_dates = []
            break

    # Step 2: If proxy return coverage was incomplete, use GARCH
    if len(usable_dates) < n_missing - 1:
        remaining_dates = missing_dates.difference(pd.Index(usable_dates)).sort_values(ascending=False)[1:]

        try:
            sim_returns = simulate_garch_returns(proxy_returns, len(remaining_dates), random_state=random_state)
        except Exception as e:
            logging.warning(f"⚠️ GARCH failed for {ticker}: {e}")
            return pd.concat([backfilled, series.loc[first_valid:]]), pd.concat([sources, pd.Series("REAL", index=series.loc[first_valid:].index)])

        price = backfilled.dropna().iloc[0] if len(backfilled.dropna()) > 0 else anchor_price
        for i, date in enumerate(remaining_dates):
            r_sim = sim_returns[i]
            price = price / (1 + r_sim)
            backfilled.loc[date] = price
            sources.loc[date] = "GARCH"

    if len(missing_dates) > 0:
        backfilled.loc[missing_dates[0]] = np.nan
        sources.loc[missing_dates[0]] = np.nan

    real_series = series.loc[first_valid:]
    real_sources = pd.Series("REAL", index=real_series.index)

    return pd.concat([backfilled.sort_index(), real_series]), pd.concat([sources.sort_index(), real_sources])

# === Main Entry Point ===
def reconstruct_missing_prices(raw_prices, portfolio_tickers, config_path="config.yaml"):
    config = load_config(config_path)
    random_state = config.get("user_settings", {}).get("random_state", None)

    prices = raw_prices.copy()
    source_matrix = pd.DataFrame(index=prices.index)
    all_tickers = list(portfolio_tickers)

    # Step 1: Precompute all proxy returns
    proxy_returns_dict = {}
    for ticker in all_tickers:
        proxy = get_proxy(ticker, config)
        if proxy in prices.columns:
            proxy_returns_dict[proxy] = prices[proxy].pct_change().dropna()

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
        proxy_returns = proxy_returns_dict.get(proxy)

        try:
            backfilled, sources = reconstruct_price_series(ticker, orig, proxy_series, proxy_returns, random_state=random_state)
            prices[ticker] = backfilled
            source_matrix[ticker] = sources
            logging.info(f"✅ Reconstructed: {ticker} using proxy {proxy}")
        except Exception as e:
            logging.warning(f"❌ Error reconstructing {ticker}: {e}")

    output_path = config["paths"]["recon_prices_output"]
    prices.to_csv(output_path, float_format="%.4f")

    source_output = config["paths"].get("recon_sources_output", "recon_sources.csv")
    source_matrix.to_csv(source_output)
    logging.info(f"✅ Saved reconstruction sources: {source_output} (shape={source_matrix.shape})")

    return prices
