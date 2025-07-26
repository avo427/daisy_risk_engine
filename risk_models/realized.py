import pandas as pd
import numpy as np
import yaml
import logging
from scipy.stats import skew, kurtosis
from pathlib import Path
import warnings
from arch.__future__ import reindexing
from arch.univariate.base import DataScaleWarning
import os

warnings.filterwarnings("ignore", category=DataScaleWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*convergence of the optimizer.*")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_realized_metrics(config_path="config.yaml"):
    config = load_config(config_path)
    rf_rate = config["user_settings"].get("risk_free_rate", 0.0)
    annual_factor = config["user_settings"].get("trading_days_per_year", 252)
    rolling_enabled = True

    recon_path = config["paths"]["recon_prices_output"]
    price_df = pd.read_csv(recon_path, index_col=0, parse_dates=True).dropna(how="all")
    returns = price_df.pct_change().dropna()

    weights_df = pd.read_csv(config["paths"]["portfolio_weights"])
    if "Ticker" not in weights_df.columns:
        weights_df.reset_index(inplace=True)
    weights_df["Ticker"] = weights_df["Ticker"].str.upper()
    weights_df = weights_df[weights_df["MarketValue"] > 0]

    portfolio_tickers = list(weights_df["Ticker"])
    benchmark_tickers = ["^NDX", "^SPX"]
    filtered_cols = [t for t in portfolio_tickers + benchmark_tickers if t in returns.columns]
    returns = returns[filtered_cols]
    tickers = returns.columns

    weights = weights_df.set_index("Ticker")["Weight"].reindex(tickers).fillna(0)
    portfolio_tickers = [t for t in portfolio_tickers if t in returns.columns]
    port_ret = (returns[portfolio_tickers] * weights[portfolio_tickers]).sum(axis=1)

    ndx_ret = returns["^NDX"] if "^NDX" in returns else None
    spx_ret = returns["^SPX"] if "^SPX" in returns else None

    metrics = []

    for t in tickers:
        try:
            r = returns[t].dropna()
            ann_return = (1 + r.mean()) ** annual_factor - 1
            ann_vol = r.std() * np.sqrt(annual_factor)
            downside_std = r[r < 0].std()
            sortino = ((r.mean() - rf_rate / annual_factor) / downside_std * np.sqrt(annual_factor)) if downside_std > 0 else np.nan
            sharpe = ((r.mean() - rf_rate / annual_factor) / r.std()) * np.sqrt(annual_factor) if r.std() > 0 else np.nan
            cum = (1 + r).cumprod()
            mdd = (cum / cum.cummax() - 1).min()
            var = np.percentile(r, 5)
            cvar = r[r <= var].mean()
            hit = (r > 0).mean()
            skewness = skew(r)
            kurt_val = kurtosis(r)
            beta_ndx = r.cov(ndx_ret) / ndx_ret.var() if ndx_ret is not None else np.nan
            beta_spx = r.cov(spx_ret) / spx_ret.var() if spx_ret is not None else np.nan
            upside = r[ndx_ret > 0].mean() / ndx_ret[ndx_ret > 0].mean() if ndx_ret is not None and (ndx_ret > 0).any() else np.nan
            downside = r[ndx_ret < 0].mean() / ndx_ret[ndx_ret < 0].mean() if ndx_ret is not None and (ndx_ret < 0).any() else np.nan
            track_err = np.std(r - ndx_ret) if ndx_ret is not None else np.nan
            info_ratio = ((r.mean() - ndx_ret.mean()) / track_err * np.sqrt(annual_factor)) if ndx_ret is not None and track_err != 0 else np.nan

            metrics.append({
                "Ticker": t,
                "AnnReturn": ann_return,
                "AnnVol": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Skew": skewness,
                "Kurtosis": kurt_val,
                "MaxDrawdown": mdd,
                "VaR_95": var,
                "CVaR_95": cvar,
                "HitRatio": hit,
                "Beta_NDX": beta_ndx,
                "Beta_SPX": beta_spx,
                "UpCapture_NDX": upside,
                "DownCapture_NDX": downside,
                "TrackingError": track_err,
                "InformationRatio": info_ratio
            })

        except Exception as e:
            logging.warning(f"WARNING: Skipping metrics for {t}: {e}")

    # Portfolio-level metrics
    try:
        r = port_ret.dropna()
        ann_return = (1 + r.mean()) ** annual_factor - 1
        ann_vol = r.std() * np.sqrt(annual_factor)
        downside_std = r[r < 0].std()
        sortino = ((r.mean() - rf_rate / annual_factor) / downside_std * np.sqrt(annual_factor)) if downside_std > 0 else np.nan
        sharpe = ((r.mean() - rf_rate / annual_factor) / r.std()) * np.sqrt(annual_factor) if r.std() > 0 else np.nan
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        var = np.percentile(r, 5)
        cvar = r[r <= var].mean()
        hit = (r > 0).mean()
        skewness = skew(r)
        kurt_val = kurtosis(r)
        beta_ndx = r.cov(ndx_ret) / ndx_ret.var() if ndx_ret is not None else np.nan
        beta_spx = r.cov(spx_ret) / spx_ret.var() if spx_ret is not None else np.nan
        upside = r[ndx_ret > 0].mean() / ndx_ret[ndx_ret > 0].mean() if ndx_ret is not None and (ndx_ret > 0).any() else np.nan
        downside = r[ndx_ret < 0].mean() / ndx_ret[ndx_ret < 0].mean() if ndx_ret is not None and (ndx_ret < 0).any() else np.nan
        track_err = np.std(r - ndx_ret) if ndx_ret is not None else np.nan
        info_ratio = ((r.mean() - ndx_ret.mean()) / track_err * np.sqrt(annual_factor)) if ndx_ret is not None and track_err != 0 else np.nan

        metrics.insert(0, {
            "Ticker": "PORTFOLIO",
            "AnnReturn": ann_return,
            "AnnVol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Skew": skewness,
            "Kurtosis": kurt_val,
            "MaxDrawdown": mdd,
            "VaR_95": var,
            "CVaR_95": cvar,
            "HitRatio": hit,
            "Beta_NDX": beta_ndx,
            "Beta_SPX": beta_spx,
            "UpCapture_NDX": upside,
            "DownCapture_NDX": downside,
            "TrackingError": track_err,
            "InformationRatio": info_ratio
        })

    except Exception as e:
        logging.warning(f"WARNING: Skipping portfolio metrics: {e}")

    metrics_df = pd.DataFrame(metrics)
    cols = metrics_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Ticker")))
    metrics_df = metrics_df[cols]

    metrics_df.rename(columns={
        "AnnReturn": "Ann. Return",
        "AnnVol": "Ann. Volatility",
        "MaxDrawdown": "Max Drawdown",
        "VaR_95": "VaR (95%)",
        "CVaR_95": "CVaR (95%)",
        "HitRatio": "Hit Ratio",
        "UpCapture_NDX": "Up Capture (NDX)",
        "DownCapture_NDX": "Down Capture (NDX)",
        "TrackingError": "Tracking Error",
        "InformationRatio": "Information Ratio",
        "Beta_NDX": "Beta (NDX)",
        "Beta_SPX": "Beta (SPX)"
    }, inplace=True)

    percent_cols = [
        "Ann. Return", "Ann. Volatility", "Max Drawdown", "VaR (95%)", "CVaR (95%)",
        "Hit Ratio", "Up Capture (NDX)", "Down Capture (NDX)", "Tracking Error"
    ]
    for col in percent_cols:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "")

    decimal_cols = [
        "Sharpe", "Sortino", "Skew", "Kurtosis",
        "Beta (NDX)", "Beta (SPX)", "Information Ratio"
    ]
    for col in decimal_cols:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

    # Ensure data directory exists before saving
    output_dir = Path(config["paths"]["realized_output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config["paths"]["realized_output"], index=False)

    # === Correlation Matrix ===
    corr = returns[portfolio_tickers].corr().round(2)
    corr.to_csv(config["paths"]["correlation_matrix"], index=True)

    # === Volatility Contribution ===
    port_vol = port_ret.std()
    # Handle division by zero in volatility contribution calculation
    if port_vol > 0:
        vol_contrib = (weights[portfolio_tickers] * returns[portfolio_tickers].std() * returns[portfolio_tickers].corrwith(port_ret)) / port_vol
        vol_contrib = vol_contrib.rename("VolatilityContribution")
        vol_contrib.to_csv(config["paths"]["vol_contribution"], header=True)
    else:
        logging.warning("WARNING: Portfolio volatility is zero, skipping volatility contribution calculation")
        # Create empty volatility contribution file
        pd.Series(dtype=float).to_csv(config["paths"]["vol_contribution"], header=True)

    # === Rolling Metrics ===
    if rolling_enabled:
        records = []
        rolling_windows = [21, 60, 126]
        rolling_tickers = portfolio_tickers + ["^NDX", "^SPX", "PORTFOLIO"]

        for window in rolling_windows:
            for t in rolling_tickers:
                if t == "PORTFOLIO":
                    r = port_ret.dropna()
                elif t not in returns:
                    continue
                else:
                    r = returns[t].dropna()

                if len(r) < window:
                    continue

                vol_roll = r.rolling(window).std() * np.sqrt(annual_factor / window)
                ret_roll = r.rolling(window).apply(lambda x: np.exp(np.log1p(x).sum()) - 1, raw=True)
                
                # Handle division by zero in Sharpe ratio calculation
                rolling_std = r.rolling(window).std()
                rolling_mean = r.rolling(window).mean()
                sharpe_roll = pd.Series(index=r.index, dtype=float)
                
                # Only calculate Sharpe where standard deviation is non-zero
                valid_sharpe = rolling_std > 0
                sharpe_roll[valid_sharpe] = (rolling_mean[valid_sharpe] - rf_rate / 252) / rolling_std[valid_sharpe]
                sharpe_roll[~valid_sharpe] = np.nan  # Set to NaN where std is zero

                valid_idx = vol_roll.dropna().index
                for date in valid_idx:
                    records.append({
                        "Date": date,
                        "Ticker": t,
                        "Metric": f"rolling_vol_{window}d",
                        "Value": vol_roll.loc[date]
                    })
                    records.append({
                        "Date": date,
                        "Ticker": t,
                        "Metric": f"rolling_ret_{window}d",
                        "Value": ret_roll.loc[date]
                    })
                    records.append({
                        "Date": date,
                        "Ticker": t,
                        "Metric": f"rolling_sharpe_{window}d",
                        "Value": sharpe_roll.loc[date]
                    })

        roll_df = pd.DataFrame(records)
        realized_roll_path = Path(config["paths"]["realized_rolling_output"])
        realized_roll_path.parent.mkdir(parents=True, exist_ok=True)
        roll_df.to_csv(realized_roll_path, index=False)

    logging.info("SUCCESS: Realized metrics computed and saved.")

if __name__ == "__main__":
    compute_realized_metrics()
