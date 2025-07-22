import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Load config ===
def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found at {config_path}")
        sys.exit(1)

# === Rolling OLS regression with aligned data ===
def run_rolling_ols(y, X, window):
    betas = pd.DataFrame(index=y.index, columns=X.columns)
    r2_scores = pd.Series(index=y.index)

    for i in range(window, len(y)):
        y_window = y.iloc[i - window:i]
        X_window = X.iloc[i - window:i]

        try:
            model = LinearRegression().fit(X_window, y_window)
            betas.iloc[i] = model.coef_
            r2_scores.iloc[i] = r2_score(y_window, model.predict(X_window))
        except Exception as e:
            print(f"❌ Fit failed at index {i}: {e}")
            continue

    return betas, r2_scores

# === Portfolio returns ===
def load_portfolio_returns(prices, weights):
    prices = prices[weights["Ticker"].values].dropna(how="all")
    returns = np.log(prices).diff()
    aligned_weights = weights.set_index("Ticker")["Weight"]
    weighted_returns = returns.mul(aligned_weights, axis=1)
    return weighted_returns.sum(axis=1)

# === Main entry point ===
def main():
    config = load_config()
    paths = config["paths"]
    settings = config["user_settings"]
    window = settings.get("factor_regression_window", 60)

    # Load data
    prices = pd.read_csv(paths["recon_prices_output"], index_col=0, parse_dates=True)
    weights = pd.read_csv(paths["portfolio_weights"])
    factors = pd.read_csv(paths["factor_returns"], index_col=0, parse_dates=True).dropna(how="all")

    if factors.empty:
        print("❌ factor_returns.csv is empty — aborting.")
        return

    # Identify active tickers (Weight > 0)
    active_tickers = weights[weights["Weight"] > 0]["Ticker"].str.upper().tolist()

    # Build return matrix
    all_returns = {}
    if "PORTFOLIO" in settings.get("factor_targets", []):
        port_ret = load_portfolio_returns(prices, weights)
        if not port_ret.dropna().empty:
            all_returns["PORTFOLIO"] = port_ret

    if "TICKERS" in settings.get("factor_targets", []):
        ticker_returns = np.log(prices).diff()
        for t in ticker_returns.columns:
            if t.upper() in active_tickers:
                all_returns[t] = ticker_returns[t]

    # Expand targets to only valid tickers
    targets = settings.get("factor_targets", ["PORTFOLIO"])
    if "TICKERS" in targets:
        ticker_list = [t for t in prices.columns if t.upper() in active_tickers]
        targets = [t for t in targets if t != "TICKERS"] + ticker_list

    print(f"📊 Regressing on targets: {targets}")
    print(f"📈 Available return series: {list(all_returns.keys())}")
    print(f"📉 Factor matrix shape: {factors.shape}")
    print(f"📅 Factor returns date range: {factors.index.min()} → {factors.index.max()}")

    exposures_list = []
    latest_expo = []
    r2_list = []

    for name in targets:
        y = all_returns.get(name)
        if y is None or y.dropna().empty:
            print(f"⚠️ Skipping {name} — no valid return series.")
            continue

        # Align y and factors to common index
        common_index = y.index.intersection(factors.index)
        y_aligned = y.loc[common_index].dropna()
        X_aligned = factors.loc[common_index].dropna(how="any")

        # Align both again to final shared index
        final_index = y_aligned.index.intersection(X_aligned.index)
        y_final = y_aligned.loc[final_index]
        X_final = X_aligned.loc[final_index]

        if len(y_final) < window:
            print(f"⚠️ Skipping {name} — not enough data after alignment ({len(y_final)} rows)")
            continue

        print(f"🔄 Running regression for {name}: {len(y_final)} rows aligned from {final_index.min().date()} to {final_index.max().date()}")

        try:
            betas, r2 = run_rolling_ols(y_final, X_final, window)
            betas = betas.dropna(how="all")
            r2 = r2.dropna()

            if not betas.empty:
                betas["Date"] = betas.index
                betas["Ticker"] = name
                exposures_list.append(betas)

                # === Save just the latest exposure for static output
                latest_row = betas.iloc[-1].drop(["Date", "Ticker"])
                for factor in latest_row.index:
                    latest_expo.append({
                        "Date": betas.index[-1],
                        "Ticker": name,
                        "Factor": factor,
                        "Beta": latest_row[factor]
                    })

            if not r2.empty:
                r2_df = r2.reset_index()
                r2_df.columns = ["Date", "R2"]
                r2_df["Ticker"] = name
                r2_list.append(r2_df)
        except Exception as e:
            print(f"❌ Regression failed for {name}: {e}")

    if not exposures_list and not r2_list:
        raise RuntimeError("❌ No factor regressions succeeded. Check return matrix and targets.")

    # === Save static factor exposures (latest only) ===
    if latest_expo:
        df_static = pd.DataFrame(latest_expo)
        Path(paths["factor_exposures"]).parent.mkdir(parents=True, exist_ok=True)
        df_static.to_csv(paths["factor_exposures"], index=False)
        print(f"✅ Saved latest factor exposures to {paths['factor_exposures']}")
    else:
        print("⚠️ No static factor exposures saved.")

    # === Save rolling factor betas in long format ===
    if exposures_list:
        df_rolling = pd.concat(exposures_list).melt(id_vars=["Date", "Ticker"], var_name="Factor", value_name="Beta").dropna()
        df_rolling.to_csv(paths["factor_rolling_long"], index=False)
        print(f"✅ Saved rolling exposures to {paths['factor_rolling_long']}")
    else:
        print("⚠️ No rolling exposures generated.")

    # === Save rolling R² ===
    if r2_list:
        df_r2 = pd.concat(r2_list).dropna()
        df_r2.to_csv(paths["r2_rolling_long"], index=False)
        print(f"✅ Saved rolling R² to {paths['r2_rolling_long']}")
    else:
        print("⚠️ No R² data generated.")

if __name__ == "__main__":
    main()
