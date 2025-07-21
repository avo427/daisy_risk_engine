import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import sys

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config file at {config_path}: {e}")
        sys.exit(1)

def load_portfolio_returns(prices, weights):
    missing_tickers = set(weights["Ticker"]) - set(prices.columns)
    if missing_tickers:
        raise ValueError(f"Tickers missing in price data: {missing_tickers}")

    weights = weights.set_index("Ticker")["Weight"]
    prices = prices[weights.index]
    returns = np.log(prices).diff()

    if returns.isnull().all().all():
        raise ValueError("All returns are NaN after log-diff calculation.")

    weighted_returns = returns @ weights
    return weighted_returns.rename("PORTFOLIO")

def run_rolling_ols(y, X, window):
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"Window must be a positive integer, got {window}")

    y = y.dropna()
    X = X.reindex(y.index).copy()
    X = sm.add_constant(X)

    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]

    if len(y) < window:
        raise ValueError(f"Not enough data points ({len(y)}) for rolling window size {window}")

    model = RollingOLS(endog=y, exog=X, window=window)
    results = model.fit()
    return results.params.drop(columns="const", errors="ignore"), results.rsquared

def main():
    config = load_config()
    paths = config["paths"]
    settings = config["user_settings"]
    base_path = Path(__file__).resolve().parent.parent

    # Load inputs
    try:
        prices = pd.read_csv(base_path / paths["recon_prices_output"], parse_dates=["Date"], index_col="Date")
        weights = pd.read_csv(base_path / paths["portfolio_weights"])
        factors = pd.read_csv(base_path / paths["factor_returns"], parse_dates=["Date"], index_col="Date")
    except Exception as e:
        print(f"❌ Error loading input data: {e}")
        sys.exit(1)

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices file must have 'Date' as a datetime index.")
    if not isinstance(factors.index, pd.DatetimeIndex):
        raise ValueError("Factor file must have 'Date' as a datetime index.")
    if factors.isnull().any().any():
        print("⚠️ Warning: NaNs detected in factor data — these rows will be dropped in regressions.")

    window = settings.get("factor_regression_window", 60)
    if not isinstance(window, int) or window <= 0:
        print(f"❌ Invalid factor_regression_window in config: {window}")
        sys.exit(1)

    targets = settings.get("factor_targets", ["PORTFOLIO"])
    all_returns = {}

    try:
        all_returns["PORTFOLIO"] = load_portfolio_returns(prices, weights)
        if "TICKERS" in targets:
            ticker_rets = np.log(prices).diff()
            for tk in ticker_rets.columns:
                all_returns[tk] = ticker_rets[tk]
    except Exception as e:
        print(f"❌ Error preparing return targets: {e}")
        sys.exit(1)

    all_results = []
    r2_values = []

    for name, y in all_returns.items():
        try:
            betas, r2 = run_rolling_ols(y, factors, window)
        except Exception as e:
            print(f"⚠️ Regression failed for {name}: {e}")
            continue

        # Quality checks
        if betas.drop(columns="Ticker", errors="ignore").var().sum() == 0:
            print(f"⚠️ Warning: All-zero beta series for {name}")
        if r2.mean() < 0.05:
            print(f"⚠️ Low R² ({r2.mean():.3f}) for {name}")

        betas["Ticker"] = name
        all_results.append(betas)

        r2 = r2.to_frame(name="R2")
        r2.index.name = "Date"
        r2["Ticker"] = name
        r2_values.append(r2)

    if not all_results:
        print("❌ No successful regression results to save.")
        sys.exit(1)

    # Save beta exposures (long format)
    all_betas = pd.concat(all_results)
    all_betas.index.name = "Date"
    all_betas = all_betas.reset_index().melt(id_vars=["Date", "Ticker"], var_name="Factor", value_name="Beta")
    all_betas.sort_values(by=["Ticker", "Factor", "Date"], inplace=True)

    rolling_path = base_path / paths["factor_rolling_long"]
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    all_betas.to_csv(rolling_path, index=False)

    # Save latest exposures
    exposures_path = base_path / paths["factor_exposures"]
    exposures_path.parent.mkdir(parents=True, exist_ok=True)
    latest_exposures = all_betas.groupby(["Ticker", "Factor"]).last().reset_index()

    if "Date" in latest_exposures.columns:
        cols = latest_exposures.columns.tolist()
        cols.insert(0, cols.pop(cols.index("Date")))
        latest_exposures = latest_exposures[cols]

    latest_exposures.sort_values(by=["Ticker", "Factor", "Date"], inplace=True)
    latest_exposures.to_csv(exposures_path, index=False)

    # Save rolling R²
    r2_long = pd.concat(r2_values).reset_index()
    r2_long["Factor"] = "R2"
    r2_long = r2_long[["Date", "Ticker", "Factor", "R2"]]
    r2_long.sort_values(by=["Ticker", "Date"], inplace=True)

    r2_path = base_path / paths["r2_rolling_long"]
    r2_path.parent.mkdir(parents=True, exist_ok=True)
    r2_long.to_csv(r2_path, index=False)

    print("✅ Factor regression outputs saved.")

if __name__ == "__main__":
    main()
