import os
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime


def download_prices(config: dict) -> pd.DataFrame:
    years = config["user_settings"]["years"]
    total_returns = config["user_settings"]["total_returns"]

    weight_path = config["paths"]["portfolio_weights"]
    raw_output_path = config["paths"]["raw_prices_output"]

    # Read tickers from portfolio CSV and ensure they are uppercase strings
    weights_df = pd.read_csv(weight_path)
    tickers = set(weights_df["Ticker"].astype(str).str.upper().tolist())

    # Include proxy tickers and benchmarks
    theme_proxies = config["theme_proxies"]
    proxy_set = set(v for v in theme_proxies.values() if v is not None)
    benchmark_set = {"^SPX", "^NDX"}
    universe = sorted(tickers | proxy_set | benchmark_set)

    logging.info(f"Downloading Yahoo Finance data for {len(universe)} tickers...")
    logging.debug(f"Final ticker universe: {universe}")

    # Use calendar-based window
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    end = pd.Timestamp.today().normalize()

    all_prices = {}

    for t in universe:
        try:
            assert isinstance(t, str), f"Ticker is not a string: {t}"
            logging.debug(f"FETCHING: Downloading: {t}")
            data = yf.download(t, start=start, end=end, auto_adjust=False, progress=False, threads=True)

            if data.empty:
                logging.warning(f"WARNING: No data for {t}")
                continue

            col = "Adj Close" if total_returns else "Close"

            # Handle both single and multi-indexed DataFrames
            if isinstance(data.columns, pd.MultiIndex):
                series = data.xs(col, level=0, axis=1)[t]
            else:
                series = data[col]

            all_prices[t] = series.rename(t)

        except Exception as e:
            logging.warning(f"WARNING: Failed to download {t}: {e}")

    price_df = pd.DataFrame(all_prices).ffill()

    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    price_df.to_csv(raw_output_path, float_format="%.4f")

    logging.info(f"SUCCESS: Saved raw price matrix: {raw_output_path} (shape={price_df.shape})")
    return price_df
