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

    # First, download ^NDX to determine the common start date
    logging.info("Downloading ^NDX as reference for common start date...")
    try:
        ndx_data = yf.download("^NDX", start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if not ndx_data.empty:
            col = "Adj Close" if total_returns else "Close"
            
            # Handle both single and multi-indexed DataFrames (fixed MultiIndex handling)
            if isinstance(ndx_data.columns, pd.MultiIndex):
                # For MultiIndex, get the column at the specified level
                ndx_series = ndx_data.xs(col, level=0, axis=1)
                # Get the first (and should be only) column from the second level
                if len(ndx_series.columns) > 0:
                    ndx_series = ndx_series.iloc[:, 0]  # Get first column
                else:
                    raise ValueError("No columns found in MultiIndex DataFrame")
            else:
                ndx_series = ndx_data[col]
            
            # Use ^NDX's first valid date as the common start date
            common_start_date = ndx_series.first_valid_index()
            if common_start_date is not None:
                logging.info(f"Using ^NDX start date as reference: {common_start_date.date()}")
            else:
                common_start_date = start
                logging.warning("^NDX has no valid data, using original start date")
        else:
            common_start_date = start
            logging.warning("No ^NDX data available, using original start date")
    except Exception as e:
        common_start_date = start
        logging.warning(f"Error downloading ^NDX, using original start date: {e}")

    successful_downloads = 0
    for t in universe:
        try:
            assert isinstance(t, str), f"Ticker is not a string: {t}"
            logging.debug(f"FETCHING: Downloading: {t}")
            
            # Remove threads=True to avoid race conditions and rate limiting issues
            data = yf.download(t, start=start, end=end, auto_adjust=False, progress=False, threads=False)

            if data.empty:
                logging.warning(f"WARNING: No data for {t}")
                continue

            col = "Adj Close" if total_returns else "Close"

            # Handle both single and multi-indexed DataFrames (fixed MultiIndex handling)
            if isinstance(data.columns, pd.MultiIndex):
                # For MultiIndex, get the column at the specified level
                series = data.xs(col, level=0, axis=1)
                # Get the first (and should be only) column from the second level
                if len(series.columns) > 0:
                    series = series.iloc[:, 0]  # Get first column
                else:
                    logging.warning(f"WARNING: No columns found in MultiIndex DataFrame for {t}")
                    continue
            else:
                series = data[col]

            # Filter to start from the common start date (^NDX's first trading day)
            series = series[series.index >= common_start_date]
            
            # Only add if we have valid data
            if not series.empty and series.notna().any():
                all_prices[t] = series.rename(t)
                successful_downloads += 1
                logging.debug(f"SUCCESS: Downloaded {t} with {len(series.dropna())} data points")
            else:
                logging.warning(f"WARNING: No valid data for {t} after filtering")

        except Exception as e:
            logging.warning(f"WARNING: Failed to download {t}: {e}")

    # Validate that we actually downloaded some data
    if not all_prices:
        raise ValueError("No price data was successfully downloaded. Check your internet connection and ticker symbols.")
    
    if successful_downloads == 0:
        raise ValueError("All downloads failed. Check your internet connection and ticker symbols.")

    price_df = pd.DataFrame(all_prices).ffill()

    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    price_df.to_csv(raw_output_path, float_format="%.4f")

    logging.info(f"SUCCESS: Downloaded {successful_downloads}/{len(universe)} tickers successfully")
    logging.info(f"SUCCESS: Saved raw price matrix: {raw_output_path} (shape={price_df.shape})")
    logging.info(f"Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
    return price_df
