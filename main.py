import logging
import pandas as pd
from pathlib import Path
import yaml
import time
import json
from datetime import datetime

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
    from risk_models.stress_test import run_stress_test
except ImportError as e:
    logging.error(f"ERROR: Import error: {e}")
    raise

# === Load Config ===
def load_config(path=None):
    path = Path(path or Path(__file__).resolve().parent / "config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"ERROR: config.yaml not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Shared Setup Steps ===
def initialize_pipeline():
    config = load_config()
    df_weights = load_portfolio_from_excel(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
    logging.info(f"SUCCESS: Loaded portfolio with {len(df_weights)} tickers")

    price_data = download_prices(config)
    logging.info("SUCCESS: Downloaded raw price data")

    raw_prices = price_data.dropna(axis=0, how="all")
    portfolio_df = pd.read_csv(config["paths"]["portfolio_weights"])
    portfolio_tickers = set(portfolio_df["Ticker"].str.upper())

    reconstruct_missing_prices(raw_prices, portfolio_tickers)
    logging.info("SUCCESS: Reconstructed missing price history")

    return config

# === Function 1: Full Pipeline ===
def run_full_pipeline():
    logging.info("RUNNING: Full Pipeline")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        compute_realized_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("SUCCESS: Computed realized metrics")

        compute_forecast_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("SUCCESS: Computed forecast metrics")

        run_factor_loader()
        logging.info("SUCCESS: Ran factor loader to generate factor returns")

        build_factors()
        logging.info("SUCCESS: Ran factor engine to process factor exposure")

        # Run stress testing after factor engine (required for factor stress tests)
        output_path = run_stress_test(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info(f"SUCCESS: Stress testing completed, results saved to {output_path}")

    except Exception as e:
        logging.error(f"ERROR: Full pipeline failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Function 2: Risk Analysis ===
def run_risk_analysis():
    logging.info("RUNNING: Risk Analysis (Realized + Forecast Metrics)")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        compute_realized_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("SUCCESS: Computed realized metrics")

        compute_forecast_metrics(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info("SUCCESS: Computed forecast metrics")

    except Exception as e:
        logging.error(f"ERROR: Risk analysis failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Function 3: Factor Exposure ===
def run_factor_exposure():
    logging.info("ANALYSIS: Running Factor Exposure (Construction + Regression)")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        run_factor_loader()
        logging.info("SUCCESS: Ran factor loader to generate factor returns")

        build_factors()
        logging.info("SUCCESS: Ran factor engine to process factor exposure")

    except Exception as e:
        logging.error(f"ERROR: Factor exposure failed: {e}")
        status = "failed"

    save_runtime_info(config, status, start)

# === Function 4: Stress Testing ===
def run_stress_testing():
    logging.info("TESTING: Running Stress Testing Suite (with Factor Exposure)")
    start = time.time()
    status = "success"

    try:
        config = initialize_pipeline()

        # First run factor exposure pipeline (required for stress testing)
        logging.info("STEP 1: Running factor exposure pipeline...")
        run_factor_loader()
        logging.info("SUCCESS: Ran factor loader to generate factor returns")

        build_factors()
        logging.info("SUCCESS: Ran factor engine to process factor exposure")

        # Then run stress testing
        logging.info("STEP 2: Running stress testing...")
        output_path = run_stress_test(config_path=str(Path(__file__).resolve().parent / "config.yaml"))
        logging.info(f"SUCCESS: Stress testing completed, results saved to {output_path}")

    except Exception as e:
        logging.error(f"ERROR: Stress testing failed: {e}")
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
        logging.info(f"SAVED: Runtime info to {runtime_output_path}")
    except Exception as e:
        logging.warning(f"WARNING: Failed to write runtime info: {e}")

# === Main Function - Controlled by app.py ===
def main(mode="full"):
    if mode == "full":
        run_full_pipeline()
    elif mode == "risk":
        run_risk_analysis()
    elif mode == "factor":
        run_factor_exposure()
    elif mode == "stress":
        run_stress_testing()
    else:
        logging.error(f"ERROR: Invalid mode: {mode}. Use 'full', 'risk', 'factor', or 'stress'.")

# === CLI Entry Point (for manual testing) ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "risk", "factor", "stress"], default="full", help="Which pipeline to run")
    args = parser.parse_args()
    main(mode=args.mode)
