import logging
import pandas as pd
from pathlib import Path
import yaml
import time
import json
from datetime import datetime
from utils.config import load_config

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Import Pipeline Steps ===
try:
    from data_pipeline.load_portfolio import load_portfolio_from_excel
    from data_pipeline.download_prices import download_prices
    from data_pipeline.reconstruct_prices import reconstruct_missing_prices
    from risk_models.realized import compute_realized_metrics
    from risk_models.forecast import compute_forecast_metrics
    from risk_models.factor_loader import main as run_factor_loader
    from risk_models.factor_engine import main as build_factors
except ImportError as e:
    logging.error(f"❌ Import error: {e}")
    raise

# === Load Config ===
def load_config(path=None):
    path = Path(path or Path(__file__).resolve().parent / "config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"❌ config.yaml not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Shared Setup Steps ===
def initialize_pipeline():
    config = load_config()
    df_weights = load_portfolio_from_excel(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
    logging.info(f"✅ Loaded portfolio with {len(df_weights)} tickers")

    price_data = download_prices(config)
    logging.info("✅ Downloaded raw price data")

    raw_prices = price_data.dropna(axis=0, how="all")
    portfolio_df = pd.read_csv(config["paths"]["portfolio_weights"])
    portfolio_tickers = set(portfolio_df["Ticker"].str.upper())

    reconstruct_missing_prices(raw_prices, portfolio_tickers)
    logging.info("✅ Reconstructed missing price history")

    return config

# === Function 1: Full Pipeline ===
def run_full_pipeline():
    config = load_config()
    logging.info("🚀 Running Full Pipeline")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        compute_realized_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed realized metrics")

        compute_forecast_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed forecast metrics")

        run_factor_loader()
        logging.info("✅ Ran factor loader to generate factor returns")

        build_factors()
        logging.info("✅ Ran factor engine to process factor exposure")

    except Exception as e:
        logging.error(f"❌ Full pipeline failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Function 2: Risk Analysis ===
def run_risk_analysis():
    logging.info("🧮 Running Risk Analysis (Realized + Forecast)")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        compute_realized_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed realized metrics")

        compute_forecast_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("✅ Computed forecast metrics")

    except Exception as e:
        logging.error(f"❌ Risk analysis failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Function 3: Factor Exposure ===
def run_factor_exposure():
    logging.info("📊 Running Factor Exposure (Construction + Regression)")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        run_factor_loader()
        logging.info("✅ Ran factor loader to generate factor returns")

        build_factors()
        logging.info("✅ Ran factor engine to process factor exposure")

    except Exception as e:
        logging.error(f"❌ Factor exposure failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Utility: Save Runtime Info ===
def save_runtime_info(config, status, start_time):
    runtime_info = {
        "last_run_seconds": round(time.time() - start_time, 2),
        "last_run_timestamp": datetime.now().isoformat(),
        "status": status
    }

    runtime_output_path = Path(config["paths"].get("runtime_info_output", "data/runtime_info.json"))
    runtime_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(runtime_output_path, "w") as f:
            json.dump(runtime_info, f, indent=2)
        logging.info(f"📦 Saved runtime info to {runtime_output_path}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to write runtime info: {e}")

# === Main Function — Controlled by app.py ===
def main(mode="full"):
    if mode == "full":
        run_full_pipeline()
    elif mode == "risk":
        run_risk_analysis()
    elif mode == "factor":
        run_factor_exposure()
    else:
        logging.error(f"❌ Invalid mode: {mode}. Use 'full', 'risk', or 'factor'.")

# === CLI Entry Point (for manual testing) ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "risk", "factor"], default="full", help="Which pipeline to run")
    args = parser.parse_args()
    main(mode=args.mode)
