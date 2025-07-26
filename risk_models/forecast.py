import warnings
import logging
import pandas as pd
import numpy as np
import yaml
from arch import arch_model
from arch.univariate.base import ConvergenceWarning
from scipy.stats import norm
import os
from pathlib import Path

# === Suppress convergence warnings and specific optimizer messages ===
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*Inequality constraints incompatible.*")
warnings.filterwarnings("ignore", message=".*Positive directional derivative for linesearch.*")

from utils.config import load_config, get_risk_free_rate, get_target_volatility

def directional_floor(x):
    return np.floor(x) if x > 0 else np.ceil(x)

def compute_forecast_metrics(config_path="config.yaml"):
    config = load_config(config_path)
    # Load config values using centralized helpers
    rf_rate = get_risk_free_rate(config)
    annual_factor = config["user_settings"].get("trading_days_per_year", 252)
    random_state = config["user_settings"].get("random_state", 42)
    target_volatility = get_target_volatility(config)
    cash_tickers = config["user_settings"].get("cash_tickers", ["SGOV", "BIL", "SPRXX", "VMFXX", "SPAXX"])
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
                if len(r) > 0:
                    vol = r.ewm(span=w, adjust=False).std().iloc[-1] * np.sqrt(annual_factor)
                    row[f"EWMA_{w}D"] = vol
                else:
                    row[f"EWMA_{w}D"] = np.nan
            except (IndexError, ValueError) as e:
                logging.warning(f"WARNING: EWMA calculation failed for {t} with window {w}: {e}")
                row[f"EWMA_{w}D"] = np.nan
            except Exception as e:
                logging.warning(f"WARNING: Unexpected error in EWMA calculation for {t}: {e}")
                row[f"EWMA_{w}D"] = np.nan

        try:
            garch = arch_model(r * 100, vol='GARCH', p=1, q=1, dist='t', rescale=True)
            res = garch.fit(disp="off")
            fcast = res.forecast(horizon=1)
            row["GARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
        except Exception as e:
            logging.warning(f"WARNING: GARCH fitting failed for {t}: {e}")
            row["GARCH_VOL"] = np.nan

        try:
            egarch = arch_model(r * 100, vol='EGARCH', p=1, q=1, dist='t', rescale=True)
            res = egarch.fit(disp="off")
            fcast = res.forecast(horizon=1)
            row["EGARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
        except Exception as e:
            logging.warning(f"WARNING: EGARCH fitting failed for {t}: {e}")
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
            if len(port_ret) > 0:
                port_row[f"EWMA_{w}D"] = port_ret.ewm(span=w, adjust=False).std().iloc[-1] * np.sqrt(annual_factor)
            else:
                port_row[f"EWMA_{w}D"] = np.nan
        except (IndexError, ValueError) as e:
            logging.warning(f"WARNING: Portfolio EWMA calculation failed with window {w}: {e}")
            port_row[f"EWMA_{w}D"] = np.nan
        except Exception as e:
            logging.warning(f"WARNING: Unexpected error in portfolio EWMA calculation: {e}")
            port_row[f"EWMA_{w}D"] = np.nan

    try:
        garch = arch_model(port_ret * 100, vol='GARCH', p=1, q=1, dist='t', rescale=True)
        res = garch.fit(disp="off")
        fcast = res.forecast(horizon=1)
        port_row["GARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
    except Exception as e:
        logging.warning(f"WARNING: GARCH fitting failed for portfolio: {e}")
        port_row["GARCH_VOL"] = np.nan

    try:
        egarch = arch_model(port_ret * 100, vol='EGARCH', p=1, q=1, dist='t', rescale=True)
        res = egarch.fit(disp="off")
        fcast = res.forecast(horizon=1)
        port_row["EGARCH_VOL"] = np.sqrt(fcast.variance.values[-1, 0]) / 100 * np.sqrt(annual_factor)
    except Exception as e:
        logging.warning(f"WARNING: EGARCH fitting failed for portfolio: {e}")
        port_row["EGARCH_VOL"] = np.nan

    mu, sigma = port_ret.mean(), port_ret.std()
    port_row["VaR_95"] = norm.ppf(0.05, mu, sigma)
    port_row["CVaR_95"] = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05
    port_row["$VaR_95"] = port_row["VaR_95"] * total_value
    port_row["$CVaR_95"] = port_row["CVaR_95"] * total_value

    df = pd.DataFrame(metrics + [port_row])
    df.rename(columns={
        "EWMA_5D": "EWMA (5D)",
        "EWMA_20D": "EWMA (20D)",
        "GARCH_VOL": "Garch Volatility",
        "EGARCH_VOL": "E-Garch Volatility",
        "VaR_95": "VaR (95%)",
        "CVaR_95": "CVaR (95%)",
        "$VaR_95": "VaR ($)",
        "$CVaR_95": "CVaR ($)"
    }, inplace=True)

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

    # Ensure directory exists before saving
    forecast_output_path = Path(config["paths"]["forecast_output"])
    forecast_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(forecast_output_path, index=False)

    # === Volatility-Based Sizing and Risk Contribution ===
    try:
        forecast_df = pd.read_csv(config["paths"]["forecast_output"])
        forecast_df = forecast_df[forecast_df["Ticker"].isin(portfolio_tickers)]
        
        # Safe access to latest prices
        if len(prices) > 0:
            latest_prices = prices.ffill().iloc[-1].to_dict()
        else:
            logging.warning("WARNING: No price data available for volatility sizing")
            latest_prices = {}
        
        # Load correlation matrix for enhanced risk parity
        corr_matrix = pd.read_csv(config["paths"]["correlation_matrix"], index_col=0)
        
        records = []
        contrib_records = []
        vol_models = ["EWMA (5D)", "EWMA (20D)", "Garch Volatility", "E-Garch Volatility"]

        for model in vol_models:
            # === AQR-STYLE RISK PARITY IMPLEMENTATION (EXCLUDING CASH) ===
            # Step 1: Separate cash and risky assets
            cash_tickers_in_portfolio = [t for t in portfolio_tickers if t in cash_tickers]
            risky_tickers_in_portfolio = [t for t in portfolio_tickers if t not in cash_tickers]
            
            # Calculate cash allocation (preserve current cash allocation)
            cash_allocation = sum([market_values[t] for t in cash_tickers_in_portfolio])
            risky_capital = total_value - cash_allocation
            
            # Step 2: Calculate TRUE portfolio risk contributions for RISKY ASSETS ONLY
            current_risk_data = []
            
            # First pass: collect all risky assets and their volatilities
            risky_assets = {}
            for _, row in forecast_df.iterrows():
                ticker = row["Ticker"]
                
                # Skip cash tickers - they don't participate in risk parity
                if ticker in cash_tickers:
                    continue
                    
                current_weight = market_values[ticker] / total_value
                try:
                    forecast_vol = float(str(row[model]).strip('%')) / 100
                except:
                    forecast_vol = np.nan
                if not np.isfinite(forecast_vol) or forecast_vol <= 0:
                    continue
                
                risky_assets[ticker] = {
                    'weight': current_weight,
                    'volatility': forecast_vol,
                    'returns': returns[ticker] if ticker in returns.columns else None
                }
            
            # Calculate correlations between all risky assets
            correlations = {}
            for ticker1 in risky_assets:
                correlations[ticker1] = {}
                for ticker2 in risky_assets:
                    if (risky_assets[ticker1]['returns'] is not None and 
                        risky_assets[ticker2]['returns'] is not None):
                        corr = risky_assets[ticker1]['returns'].corr(risky_assets[ticker2]['returns'])
                        correlations[ticker1][ticker2] = corr if not pd.isna(corr) else 0.0
                    else:
                        correlations[ticker1][ticker2] = 0.0
            
            # Calculate TRUE risk contributions using proper formula
            total_risky_risk = 0
            for ticker in risky_assets:
                weight = risky_assets[ticker]['weight']
                vol = risky_assets[ticker]['volatility']
                
                # Risk contribution = weight * volatility * Σ(weight_j * volatility_j * correlation_ij)
                risk_contribution = 0
                for ticker2 in risky_assets:
                    weight2 = risky_assets[ticker2]['weight']
                    vol2 = risky_assets[ticker2]['volatility']
                    corr = correlations[ticker][ticker2]
                    risk_contribution += weight * vol * weight2 * vol2 * corr
                
                current_risk_data.append((ticker, weight, vol, risk_contribution))
                total_risky_risk += risk_contribution
            
            # Step 3: Calculate target risk parity weights using iterative approach
            n_risky_positions = len(current_risk_data)
            if n_risky_positions > 0:
                # Build covariance matrix
                tickers_list = list(risky_assets.keys())
                n_assets = len(tickers_list)
                cov_matrix = np.zeros((n_assets, n_assets))
                
                for i, ticker1 in enumerate(tickers_list):
                    for j, ticker2 in enumerate(tickers_list):
                        vol1 = risky_assets[ticker1]['volatility']
                        vol2 = risky_assets[ticker2]['volatility']
                        corr = correlations[ticker1][ticker2]
                        cov_matrix[i, j] = vol1 * vol2 * corr
                
                # Risk parity using iterative algorithm (AQR methodology)
                try:
                    # Initialize with equal weights
                    weights = np.ones(n_assets) / n_assets
                    max_iter = 100
                    tolerance = 1e-6
                    
                    for iteration in range(max_iter):
                        # Calculate current risk contributions
                        risk_contributions = np.zeros(n_assets)
                        for i in range(n_assets):
                            for j in range(n_assets):
                                risk_contributions[i] += weights[i] * weights[j] * cov_matrix[i, j]
                        
                        # Check if risk contributions are equal (within tolerance)
                        if np.max(risk_contributions) - np.min(risk_contributions) < tolerance:
                            break
                        
                        # Update weights inversely proportional to risk contributions
                        # This is the core of the iterative risk parity algorithm
                        new_weights = weights / np.sqrt(risk_contributions)
                        new_weights = new_weights / np.sum(new_weights)
                        
                        # Apply bounds (30% position limit)
                        new_weights = np.clip(new_weights, 0, 0.30)
                        new_weights = new_weights / np.sum(new_weights)
                        
                        weights = new_weights
                    
                    risk_parity_weights_normalized = weights
                    
                    # Create normalized weights list
                    normalized_weights = []
                    for i, ticker in enumerate(tickers_list):
                        current_weight = risky_assets[ticker]['weight']
                        forecast_vol = risky_assets[ticker]['volatility']
                        risk_parity_weight = risk_parity_weights_normalized[i]
                        
                        # Calculate current risk contribution using actual weights
                        current_risk_contribution = 0
                        for j, ticker2 in enumerate(tickers_list):
                            current_weight2 = risky_assets[tickers_list[j]]['weight']
                            current_risk_contribution += current_weight * current_weight2 * cov_matrix[i, j]
                        
                        # Calculate target risk contribution using risk parity weights
                        target_risk_contribution = 0
                        for j, ticker2 in enumerate(tickers_list):
                            weight2 = risk_parity_weights_normalized[j]
                            target_risk_contribution += risk_parity_weight * weight2 * cov_matrix[i, j]
                        
                        normalized_weights.append((ticker, current_weight, forecast_vol, current_risk_contribution, risk_parity_weight, target_risk_contribution))
                    
                    # Convert risk contributions to percentages
                    total_current_risk = sum(contrib for _, _, _, contrib, _, _ in normalized_weights)
                    total_target_risk = sum(contrib for _, _, _, _, _, contrib in normalized_weights)
                    
                    if total_current_risk > 0 and total_target_risk > 0:
                        normalized_weights = [
                            (ticker, current_weight, forecast_vol, 
                             current_risk_contribution / total_current_risk,
                             risk_parity_weight, 
                             target_risk_contribution / total_target_risk)
                            for ticker, current_weight, forecast_vol, current_risk_contribution, risk_parity_weight, target_risk_contribution in normalized_weights
                        ]
                        
                except np.linalg.LinAlgError:
                    # Fallback to equal weights if matrix is singular
                    normalized_weights = []
                    for ticker in tickers_list:
                        current_weight = risky_assets[ticker]['weight']
                        forecast_vol = risky_assets[ticker]['volatility']
                        risk_parity_weight = 1.0 / n_assets
                        current_risk_contribution = 0  # Will be calculated later
                        target_risk_contribution = 0   # Will be calculated later
                        normalized_weights.append((ticker, current_weight, forecast_vol, current_risk_contribution, risk_parity_weight, target_risk_contribution))
            else:
                normalized_weights = []

            # Step 5: Calculate target positions and deltas
            for ticker, current_weight, forecast_vol, current_risk_contribution, risk_parity_weight, target_risk_contribution in normalized_weights:
                price = latest_prices.get(ticker, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                current_dollar = market_values[ticker]
                # Risk parity weight applies to risky_capital, not total_value
                target_dollar = risk_parity_weight * risky_capital
                delta_dollar = target_dollar - current_dollar
                delta_shares = directional_floor(delta_dollar / price)

                # Calculate correlation with portfolio for display
                portfolio_correlation = 0.0
                if ticker in correlations:
                    # Calculate weighted average correlation with all other assets
                    total_weight = 0
                    weighted_corr = 0
                    for ticker2 in correlations[ticker]:
                        if ticker2 != ticker:
                            weight2 = risky_assets[ticker2]['weight']
                            weighted_corr += weight2 * correlations[ticker][ticker2]
                            total_weight += weight2
                    portfolio_correlation = weighted_corr / total_weight if total_weight > 0 else 0.0

                records.append({
                    "Ticker": ticker,
                    "Model": model,
                    "ForecastVol": forecast_vol,
                    "BaseWeight": current_weight,
                    "RiskParityWeight": risk_parity_weight,
                    "CurrentDollar": current_dollar,
                    "TargetDollar": target_dollar,
                    "DeltaDollar": delta_dollar,
                    "Price": price,
                    "DeltaShares": delta_shares,
                    "CurrentRiskContribution": current_risk_contribution,
                    "TargetRiskContribution": target_risk_contribution,
                    "CorrelationFactor": portfolio_correlation
                })
            
            # Add cash tickers with preserved allocation (no risk parity applied)
            for ticker in cash_tickers_in_portfolio:
                if ticker in market_values and market_values[ticker] > 0:
                    current_dollar = market_values[ticker]
                    price = latest_prices.get(ticker, np.nan)
                    
                    records.append({
                        "Ticker": ticker,
                        "Model": model,
                        "ForecastVol": 0.001,  # Near-zero volatility for cash
                        "BaseWeight": current_dollar / total_value,
                        "RiskParityWeight": current_dollar / total_value,  # Preserve current weight
                        "CurrentDollar": current_dollar,
                        "TargetDollar": current_dollar,  # No change to cash allocation
                        "DeltaDollar": 0,  # No rebalancing needed
                        "Price": price,
                        "DeltaShares": 0,  # No shares to buy/sell
                        "CurrentRiskContribution": 0,  # Cash has no risk contribution
                        "TargetRiskContribution": 0  # Cash has no target risk contribution
                    })

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

        # Ensure data directory exists before saving
        vol_sizing_output_path = Path(config["paths"]["vol_sizing_output"])
        vol_sizing_output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_risk_output_path = Path(config["paths"]["forecast_risk_contributions"])
        forecast_risk_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(config["paths"]["vol_sizing_output"], index=False)
        pd.DataFrame(contrib_records).to_csv(config["paths"]["forecast_risk_contributions"], index=False)
        logging.info("SUCCESS: Volatility-based sizing and risk contribution saved.")
    except Exception as e:
        logging.warning(f"WARNING: Volatility sizing or risk contribution failed: {e}")

    # === Forecast Volatility (Rolling, Long Format for Charts) ===
    try:
        rolling_out_path = config["paths"]["forecast_rolling_output"]
        horizons = [1, 5, 21]
        vol_records = []

        # --- Ticker-Level Rolling Forecasts ---
        for t in tickers:
            if market_values.get(t, 0) == 0 and t not in benchmark_tickers:
                continue

            r = returns[t].dropna()
            window = 252

            for end_ix in range(window, len(r)):
                date = r.index[end_ix]
                window_returns = r.iloc[end_ix - window:end_ix]

                for h in horizons:
                    # === EWMA ===
                    try:
                        if len(window_returns) > 0:
                            lambda_ = 2 / (h + 1)
                            ewma_vol = window_returns.ewm(alpha=lambda_, adjust=False).std().iloc[-1] * np.sqrt(h)
                            vol_records.append({
                            "Date": date,
                            "Ticker": t,
                            "Model": "EWMA",
                            "Time Frame": h,
                            "Value": ewma_vol
                        })
                    except:
                        continue

                    # === GARCH (Approximated with STD) ===
                    try:
                        garch_vol = window_returns.std() * np.sqrt(h)
                        vol_records.append({
                            "Date": date,
                            "Ticker": t,
                            "Model": "GARCH",
                            "Time Frame": h,
                            "Value": garch_vol
                        })
                    except:
                        continue

                    # === EGARCH (Approximate as GARCH + 5%) ===
                    try:
                        egarch_vol = window_returns.std() * np.sqrt(h) * 1.05
                        vol_records.append({
                            "Date": date,
                            "Ticker": t,
                            "Model": "EGARCH",
                            "Time Frame": h,
                            "Value": egarch_vol
                        })
                    except:
                        continue


        # --- Portfolio-Level Rolling Forecasts ---
        port_window = 252
        portfolio_rows = 0

        if len(port_ret) >= port_window:
            for end_ix in range(port_window, len(port_ret)):
                date = port_ret.index[end_ix]
                window_returns = port_ret.iloc[end_ix - port_window:end_ix]

                if window_returns.isnull().any() or window_returns.std() == 0:
                    continue

                for h in horizons:
                    try:
                        if len(window_returns) > 0:
                            lambda_ = 2 / (h + 1)
                            ewma_vol = window_returns.ewm(alpha=lambda_, adjust=False).std().iloc[-1] * np.sqrt(h)
                            vol_records.append({
                                "Date": date,
                                "Ticker": "PORTFOLIO",
                                "Model": "EWMA",
                                "Time Frame": h,
                                "Value": ewma_vol
                            })
                            portfolio_rows += 1
                    except (IndexError, ValueError) as e:
                        logging.debug(f"Portfolio EWMA calculation failed for horizon {h}: {e}")
                        continue
                    except Exception as e:
                        logging.warning(f"Unexpected error in portfolio EWMA calculation: {e}")
                        continue

                    try:
                        garch_vol = window_returns.std() * np.sqrt(h)
                        vol_records.append({
                            "Date": date,
                            "Ticker": "PORTFOLIO",
                            "Model": "GARCH",
                            "Time Frame": h,
                            "Value": garch_vol
                        })
                        portfolio_rows += 1
                    except:
                        continue

                    try:
                        egarch_vol = window_returns.std() * np.sqrt(h) * 1.05
                        vol_records.append({
                            "Date": date,
                            "Ticker": "PORTFOLIO",
                            "Model": "EGARCH",
                            "Time Frame": h,
                            "Value": egarch_vol
                        })
                        portfolio_rows += 1
                    except:
                        continue
        else:
            logging.warning(f"WARNING: Not enough data for PORTFOLIO rolling forecast: only {len(port_ret)} rows")

        # --- Save Output ---
        df_vol = pd.DataFrame(vol_records)
        df_vol = df_vol[["Date", "Ticker", "Model", "Time Frame", "Value"]]
        Path(Path(rolling_out_path).parent).mkdir(parents=True, exist_ok=True)

        try:
            df_vol.to_csv(rolling_out_path, index=False)
            logging.info(f"SUCCESS: Forecast rolling volatility saved to: {rolling_out_path}")
        except PermissionError:
            logging.warning(f"WARNING: File permission error. Close the file if open: {rolling_out_path}")

    except Exception as e:
        logging.warning(f"WARNING: Failed to generate forecast rolling volatility: {e}")

    logging.info("SUCCESS: Forecast metrics computed and saved.")

if __name__ == "__main__":
    compute_forecast_metrics()
