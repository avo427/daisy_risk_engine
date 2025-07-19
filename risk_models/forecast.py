import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from arch import arch_model
from scipy.stats import norm

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def directional_floor(x):
    return np.floor(x) if x > 0 else np.ceil(x)

def compute_forecast_metrics(config_path="config.yaml"):
    config = load_config(config_path)
    rf_rate = config["user_settings"].get("risk_free_rate", 0.0)
    annual_factor = config["user_settings"].get("trading_days_per_year", 252)
    random_state = config["user_settings"].get("random_state", 42)
    np.random.seed(random_state)

    prices = pd.read_csv(config["paths"]["recon_prices_output"], index_col=0, parse_dates=True).dropna(how="all")
    returns = prices.pct_change().dropna()

    weights_df = pd.read_csv(config["paths"]["portfolio_weights"])
    weights_df["Ticker"] = weights_df["Ticker"].str.upper()
    weights_df = weights_df[weights_df["MarketValue"] > 0].copy()

    portfolio_tickers = list(weights_df["Ticker"])
    benchmark_tickers = ["^NDX", "^SPX"]
    all_tickers = portfolio_tickers + benchmark_tickers
    relevant_tickers = [t for t in all_tickers if t in returns.columns]
    returns = returns[relevant_tickers]
    tickers = returns.columns

    market_values = weights_df.set_index("Ticker")["MarketValue"].reindex(tickers).fillna(0)
    total_value = market_values.sum()

    ewma_windows = [5, 20]
    metrics = []

    for t in tickers:
        if market_values.get(t, 0) == 0 and t not in benchmark_tickers:
            continue
        r = returns[t].dropna()
        row = {"Ticker": t}

        for w in ewma_windows:
            try:
                vol = r.ewm(span=w, adjust=False).std().iloc[-1] * np.sqrt(annual_factor)
                row[f"EWMA_{w}D"] = vol
            except Exception:
                row[f"EWMA_{w}D"] = np.nan

        try:
            garch = arch_model(r * 100, vol='GARCH', p=1, q=1, dist='t', rescale=True)
            res = garch.fit(disp="off")
            fcast = res.forecast(horizon=1)
            row["GARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
        except Exception:
            row["GARCH_VOL"] = np.nan

        try:
            egarch = arch_model(r * 100, vol='EGARCH', p=1, q=1, dist='t', rescale=True)
            res = egarch.fit(disp="off")
            fcast = res.forecast(horizon=1)
            row["EGARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
        except Exception:
            row["EGARCH_VOL"] = np.nan

        mu, sigma = r.mean(), r.std()
        var_95 = norm.ppf(0.05, mu, sigma)
        cvar_95 = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05
        row["VaR_95"] = var_95
        row["CVaR_95"] = cvar_95

        mv = market_values.get(t, 0)
        row["$VaR_95"] = var_95 * mv
        row["$CVaR_95"] = cvar_95 * mv

        metrics.append(row)

    port_ret = (returns[portfolio_tickers] * market_values[portfolio_tickers] / total_value).sum(axis=1)
    port_row = {"Ticker": "PORTFOLIO"}

    for w in ewma_windows:
        try:
            port_row[f"EWMA_{w}D"] = port_ret.ewm(span=w, adjust=False).std().iloc[-1] * np.sqrt(annual_factor)
        except Exception:
            port_row[f"EWMA_{w}D"] = np.nan

    try:
        garch = arch_model(port_ret * 100, vol='GARCH', p=1, q=1, dist='t', rescale=True)
        res = garch.fit(disp="off")
        fcast = res.forecast(horizon=1)
        port_row["GARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
    except Exception:
        port_row["GARCH_VOL"] = np.nan

    try:
        egarch = arch_model(port_ret * 100, vol='EGARCH', p=1, q=1, dist='t', rescale=True)
        res = egarch.fit(disp="off")
        fcast = res.forecast(horizon=1)
        port_row["EGARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
    except Exception:
        port_row["EGARCH_VOL"] = np.nan

    mu, sigma = port_ret.mean(), port_ret.std()
    port_row["VaR_95"] = norm.ppf(0.05, mu, sigma)
    port_row["CVaR_95"] = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05
    port_row["$VaR_95"] = port_row["VaR_95"] * total_value
    port_row["$CVaR_95"] = port_row["CVaR_95"] * total_value

    df = pd.DataFrame(metrics + [port_row])
    rename_map = {
        "EWMA_5D": "EWMA (5D)",
        "EWMA_20D": "EWMA (20D)",
        "GARCH_VOL": "Garch Volatility",
        "EGARCH_VOL": "E-Garch Volatility",
        "VaR_95": "VaR (95%)",
        "CVaR_95": "CVaR (95%)",
        "$VaR_95": "VaR ($)",
        "$CVaR_95": "CVaR ($)"
    }
    df.rename(columns=rename_map, inplace=True)

    pct_cols = [
        "EWMA (5D)", "EWMA (20D)",
        "Garch Volatility", "E-Garch Volatility",
        "VaR (95%)", "CVaR (95%)"
    ]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "")

    for col in ["VaR ($)", "CVaR ($)"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")

    df.to_csv("data/forecast_metrics.csv", index=False)

    # === Volatility-Based Sizing and Risk Contribution ===
    try:
        forecast_df = pd.read_csv("data/forecast_metrics.csv")
        forecast_df = forecast_df[forecast_df["Ticker"].isin(portfolio_tickers)]
        latest_prices = prices.ffill().iloc[-1].to_dict()
        records = []
        contrib_records = []
        vol_models = ["EWMA (5D)", "EWMA (20D)", "Garch Volatility", "E-Garch Volatility"]

        for model in vol_models:
            sizing_tmp = []
            for _, row in forecast_df.iterrows():
                ticker = row["Ticker"]
                base_weight = market_values[ticker] / total_value
                try:
                    forecast_vol = float(str(row[model]).strip('%')) / 100
                except:
                    forecast_vol = np.nan
                if not np.isfinite(forecast_vol) or forecast_vol <= 0:
                    continue

                vol_adj_weight = base_weight * 0.20 / forecast_vol
                sizing_tmp.append((ticker, base_weight, forecast_vol, vol_adj_weight))

            total_vol_weight = sum([x[3] for x in sizing_tmp])

            for ticker, base_weight, forecast_vol, vol_adj_weight in sizing_tmp:
                weight_norm = vol_adj_weight / total_vol_weight
                price = latest_prices.get(ticker, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                current_dollar = market_values[ticker]
                target_dollar = weight_norm * total_value
                delta_dollar = target_dollar - current_dollar
                delta_shares = directional_floor(delta_dollar / price)

                records.append({
                    "Ticker": ticker,
                    "Model": model,
                    "ForecastVol": forecast_vol,
                    "BaseWeight": base_weight,
                    "VolAdjWeight": weight_norm,
                    "CurrentDollar": current_dollar,
                    "TargetDollar": target_dollar,
                    "DeltaDollar": delta_dollar,
                    "Price": price,
                    "DeltaShares": delta_shares
                })

            # Actual weight-based risk contribution
            actual_contribs = []
            total_risk = 0
            for _, row in forecast_df.iterrows():
                ticker = row["Ticker"]
                try:
                    forecast_vol = float(str(row[model]).strip('%')) / 100
                    weight = market_values[ticker] / total_value
                    contrib = weight * forecast_vol
                    actual_contribs.append((ticker, weight, forecast_vol, contrib))
                    total_risk += contrib
                except:
                    continue

            for ticker, weight, forecast_vol, risk_contribution in actual_contribs:
                contrib_records.append({
                    "Ticker": ticker,
                    "Model": model,
                    "ForecastVol": forecast_vol,
                    "ActualWeight": weight,
                    "Forecast_Risk_Contribution": risk_contribution,
                    "Forecast_Risk_%": 100 * risk_contribution / total_risk if total_risk > 0 else np.nan
                })

        pd.DataFrame(records).to_csv("data/vol_sizing_weights_long.csv", index=False)
        pd.DataFrame(contrib_records).to_csv("data/forecast_risk_contributions.csv", index=False)
        logging.info("✅ Volatility-based sizing and risk contribution saved.")
    except Exception as e:
        logging.warning(f"⚠️ Volatility sizing or risk contribution failed: {e}")

    # === Rolling Volatility ===
    records = []
    window = 21
    for t in list(tickers) + ["PORTFOLIO"]:
        r = port_ret.dropna() if t == "PORTFOLIO" else returns[t].dropna()
        vol = r.rolling(window).std() * np.sqrt(annual_factor)
        for dt in vol.index:
            records.append({"Date": dt, "Ticker": t, "Metric": "rolling_vol_21d", "Value": vol.loc[dt]})

    roll_df = pd.DataFrame(records)
    Path("data/rolling_metrics").mkdir(parents=True, exist_ok=True)
    roll_df.to_csv("data/rolling_metrics/forecast_rolling_long.csv", index=False)

    logging.info("✅ Forecast metrics computed and saved.")

if __name__ == "__main__":
    compute_forecast_metrics()
