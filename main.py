import logging
import pandas as pd
from pathlib import Path
import yaml

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Import Pipeline Steps ===
try:
    from data_pipeline.load_portfolio import load_portfolio_from_excel
except ImportError:
    logging.warning("⚠️ load_portfolio_from_excel() not yet implemented.")

try:
    from data_pipeline.download_prices import download_prices
except ImportError:
    logging.warning("⚠️ download_prices() not yet implemented.")

try:
    from data_pipeline.reconstruct_prices import reconstruct_missing_prices
except ImportError:
    logging.warning("⚠️ reconstruct_missing_prices() not yet implemented.")

try:
    from risk_models.realized import compute_realized_metrics
except ImportError:
    logging.warning("⚠️ compute_realized_metrics() not yet implemented.")

try:
    from risk_models.forecast import compute_forecast_metrics
except ImportError:
    logging.warning("⚠️ compute_forecast_metrics() not yet implemented.")

# === Load Config (robust to working directory) ===
def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    else:
        path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ config.yaml not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Main Run Function ===
def run():
    logging.info("🐶 Starting Daisy Risk Engine")
    config = load_config()

    # === Step 1: Load Portfolio ===
    try:
        df_weights = load_portfolio_from_excel(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info(f"✅ Loaded portfolio with {len(df_weights)} tickers")
    except Exception as e:
        logging.error(f"❌ Failed to load portfolio: {e}")
        return

    # === Step 2: Download Yahoo Finance Prices ===
    try:
        price_data = download_prices(config)
        logging.info("✅ Downloaded raw price data")

        raw_prices = price_data.dropna(axis=0, how="all")
        portfolio_df = pd.read_csv(config["paths"]["portfolio_weights"])
        portfolio_tickers = set(portfolio_df["Ticker"].str.upper())
    except Exception as e:
        logging.error(f"❌ Failed during price download: {e}")
        return

    # === Step 3: Reconstruct Missing Prices ===
    try:
        reconstructed_prices = reconstruct_missing_prices(raw_prices, portfolio_tickers)
        logging.info("✅ Reconstructed missing price history")
    except Exception as e:
        logging.error(f"❌ Failed during reconstruction: {e}")
        return

    # === Step 4: Compute Realized Metrics ===
    try:
        compute_realized_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed realized metrics")
    except Exception as e:
        logging.error(f"❌ Failed to compute realized metrics: {e}")

    # === Step 5: Compute Forecast Metrics ===
    try:
        compute_forecast_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed forecast metrics")
    except Exception as e:
        logging.error(f"❌ Failed to compute forecast metrics: {e}")

    logging.info("✅ Daisy Risk Engine finished.")

# === CLI Entry Point ===
if __name__ == "__main__":
    run()