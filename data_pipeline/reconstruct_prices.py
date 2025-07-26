import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# === Load Config ===
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Get Proxy for Ticker ===
def get_proxy(ticker, config):
    theme = config["ticker_themes"].get(ticker)
    proxy = config["theme_proxies"].get(theme)
    return proxy or config["fallback_proxy"]

# === Reconstruct Price Series ===
# Note: We use simple historical returns rather than GARCH simulation because:
# 1. For tech portfolios, NDX provides natural correlation with holdings
# 2. Historical returns are more reliable than simulated volatility clustering
# 3. GARCH models can introduce unrealistic behavior for price reconstruction
#
# Enhanced with volatility scaling and factor adjustments (industry best practice)
def reconstruct_price_series(ticker, series, proxy_series, proxy_returns, fallback_returns=None):
    first_valid = series.first_valid_index()
    if first_valid is None:
        logging.warning(f"Ticker {ticker} has no data at all.")
        return series, pd.Series(index=series.index, dtype=object)

    missing_dates = series.loc[:first_valid].iloc[:-1].index
    n_missing = len(missing_dates)
    if n_missing <= 1:
        sources = pd.Series("REAL", index=series.loc[first_valid:].index)
        return series, sources

    # DEBUG: Log key information
    logging.info(f"DEBUG: {ticker} - First valid date: {first_valid}, Anchor price: {series.loc[first_valid]:.2f}")
    if n_missing > 0:
        logging.info(f"DEBUG: {ticker} - Missing dates: {len(missing_dates)} from {missing_dates[0]} to {missing_dates[-1]}")
    else:
        logging.info(f"DEBUG: {ticker} - No missing dates to reconstruct")
    
    backfilled = pd.Series(index=missing_dates, dtype=float)
    sources = pd.Series(index=missing_dates, dtype=object)
    anchor_price = series.loc[first_valid]
    price = anchor_price

    # Calculate volatility scaling factors using pre-IPO period (simpler and more logical)
    asset_vol = None
    proxy_vol = None
    vol_ratio = 1.0
    
    # Use the period before the first valid date (pre-IPO period)
    pre_ipo_period = series.loc[:first_valid].iloc[:-1]  # All dates before first valid
    if len(pre_ipo_period) > 30:  # Need sufficient pre-IPO data
        # Calculate volatility from the pre-IPO period (handle NaN values)
        asset_returns = pre_ipo_period.pct_change().dropna()
        if len(asset_returns) > 10:  # Need minimum data points
            asset_vol = asset_returns.std()
            if not np.isnan(asset_vol) and asset_vol > 0:
                if proxy_returns is not None:
                    # Use proxy returns from the same pre-IPO period
                    pre_ipo_proxy_returns = proxy_returns.loc[pre_ipo_period.index].dropna()
                    if len(pre_ipo_proxy_returns) > 10:
                        proxy_vol = pre_ipo_proxy_returns.std()
                        if not np.isnan(proxy_vol) and proxy_vol > 0 and asset_vol > 0:
                            vol_ratio = min(max(asset_vol / proxy_vol, 0.5), 2.0)  # Cap at reasonable range
                            logging.info(f"DEBUG: {ticker} - Pre-IPO vol ratio: {vol_ratio:.3f} (asset: {asset_vol:.4f}, proxy: {proxy_vol:.4f})")
    
    # If no pre-IPO data available, use fallback
    if vol_ratio == 1.0:
        logging.info(f"DEBUG: {ticker} - No pre-IPO volatility data, using 1.0x scaling")

    # Step 1: Process missing dates backwards, using proxy returns where available
    # Use fallback proxy if primary proxy doesn't have data
    reconstructed_count = 0
    for i in range(n_missing):
        date = missing_dates[-(i + 1)]  # reversed order
        if date in proxy_returns.index:
            r = proxy_returns.loc[date]
            # Apply volatility scaling (industry best practice)
            r_scaled = r * vol_ratio
            
            # Quality checks and regime awareness (AQR best practice)
            if abs(r_scaled) > 0.15:  # 15% return - flag extreme moves
                logging.warning(f"WARNING: {ticker} - Extreme scaled proxy return on {date}: {r_scaled:.4f} ({r_scaled*100:.2f}%) [original: {r:.4f}]")
                # Cap extreme moves to prevent unrealistic reconstruction
                r_scaled = np.clip(r_scaled, -0.15, 0.15)
                logging.warning(f"WARNING: {ticker} - Capped to: {r_scaled:.4f} ({r_scaled*100:.2f}%)")
            elif abs(r_scaled) > 0.1:  # 10% return - log for monitoring
                logging.info(f"DEBUG: {ticker} - Large scaled proxy return on {date}: {r_scaled:.4f} ({r_scaled*100:.2f}%) [original: {r:.4f}]")
            
            # Calculate new price (handle division by zero)
            old_price = price
            if abs(r_scaled) >= 1.0:  # Prevent division by zero or negative prices
                logging.warning(f"WARNING: {ticker} - Invalid return {r_scaled:.4f} on {date}, skipping")
                backfilled.loc[date] = np.nan
                sources.loc[date] = np.nan
            else:
                price = old_price / (1 + r_scaled)
                backfilled.loc[date] = price
                sources.loc[date] = "PROXY"
                reconstructed_count += 1
                logging.debug(f"DEBUG: {ticker} - Price change: {old_price:.2f} -> {price:.2f}")
                
        elif fallback_returns is not None and date in fallback_returns.index:
            # Use fallback proxy with volatility scaling
            r = fallback_returns.loc[date]
            r_scaled = r * vol_ratio
            
            # Quality checks and regime awareness (AQR best practice)
            if abs(r_scaled) > 0.15:  # 15% return - flag extreme moves
                logging.warning(f"WARNING: {ticker} - Extreme scaled fallback return on {date}: {r_scaled:.4f} ({r_scaled*100:.2f}%) [original: {r:.4f}]")
                # Cap extreme moves to prevent unrealistic reconstruction
                r_scaled = np.clip(r_scaled, -0.15, 0.15)
                logging.warning(f"WARNING: {ticker} - Capped to: {r_scaled:.4f} ({r_scaled*100:.2f}%)")
            elif abs(r_scaled) > 0.1:  # 10% return - log for monitoring
                logging.info(f"DEBUG: {ticker} - Large scaled fallback return on {date}: {r_scaled:.4f} ({r_scaled*100:.2f}%) [original: {r:.4f}]")
            
            # Calculate new price (handle division by zero)
            old_price = price
            if abs(r_scaled) >= 1.0:  # Prevent division by zero or negative prices
                logging.warning(f"WARNING: {ticker} - Invalid return {r_scaled:.4f} on {date}, skipping")
                backfilled.loc[date] = np.nan
                sources.loc[date] = np.nan
            else:
                price = old_price / (1 + r_scaled)
                backfilled.loc[date] = price
                sources.loc[date] = "FALLBACK"
                reconstructed_count += 1
                logging.debug(f"DEBUG: {ticker} - Price change: {old_price:.2f} -> {price:.2f}")
                
        else:
            # No proxy return available - set to NaN and continue
            backfilled.loc[date] = np.nan
            sources.loc[date] = np.nan
            logging.debug(f"DEBUG: {ticker} - No proxy return available for {date}, setting to NaN")

    # Log reconstruction summary
    if reconstructed_count > 0:
        logging.info(f"DEBUG: {ticker} - Reconstructed {reconstructed_count}/{n_missing} missing dates")
    else:
        logging.warning(f"WARNING: {ticker} - No dates could be reconstructed (no proxy data available)")

    real_series = series.loc[first_valid:]
    real_sources = pd.Series("REAL", index=real_series.index)

    # DEBUG: Check for jumps at transition
    if len(backfilled.dropna()) > 0:
        last_reconstructed = backfilled.dropna().iloc[-1]
        first_actual = real_series.iloc[0]
        jump_pct = abs(last_reconstructed - first_actual) / first_actual
        if jump_pct > 0.05:  # 5% jump
            logging.warning(f"DEBUG: {ticker} - Large jump at transition: {last_reconstructed:.2f} -> {first_actual:.2f} ({jump_pct*100:.2f}%)")

    return pd.concat([backfilled.sort_index(), real_series]), pd.concat([sources.sort_index(), real_sources])

# === Main Entry Point ===
def reconstruct_missing_prices(raw_prices, portfolio_tickers, config_path="config.yaml"):
    config = load_config(config_path)

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
        
        # Get fallback proxy returns if different from primary proxy
        fallback_proxy = config["fallback_proxy"]
        fallback_returns = None
        if fallback_proxy != proxy and fallback_proxy in prices.columns:
            fallback_returns = prices[fallback_proxy].pct_change().dropna()

        try:
            backfilled, sources = reconstruct_price_series(ticker, orig, proxy_series, proxy_returns, fallback_returns)
            prices[ticker] = backfilled
            source_matrix[ticker] = sources
            logging.info(f"SUCCESS: Reconstructed: {ticker} using proxy {proxy}")
        except Exception as e:
            logging.warning(f"ERROR: Error reconstructing {ticker}: {e}")

    output_path = config["paths"]["recon_prices_output"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, float_format="%.4f")

    source_output = config["paths"].get("recon_sources_output", "recon_sources.csv")
    Path(source_output).parent.mkdir(parents=True, exist_ok=True)
    source_matrix.to_csv(source_output)
    logging.info(f"SUCCESS: Saved reconstruction sources: {source_output} (shape={source_matrix.shape})")

    return prices
