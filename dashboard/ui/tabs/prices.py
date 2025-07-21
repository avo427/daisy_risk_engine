import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

def prices_tab(project_root, paths):
    st.subheader("Reconstructed Prices Viewer")
    recon_path   = project_root / paths["recon_prices_output"]
    weights_path = project_root / paths["portfolio_weights"]
    yahoo_path   = project_root / paths["raw_prices_output"]
    if recon_path.exists() and weights_path.exists() and yahoo_path.exists():
        df_prices = pd.read_csv(recon_path, index_col=0, parse_dates=True)
        df_weights = pd.read_csv(weights_path)
        df_yahoo = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)
        ipo_dates = {}
        for ticker in df_yahoo.columns.intersection(df_prices.columns):
            yahoo_series = df_yahoo[ticker].dropna()
            recon_series = df_prices[ticker].dropna()
            if not yahoo_series.empty and not recon_series.empty:
                yahoo_start = yahoo_series.index[0]
                if recon_series.index.min() < yahoo_start:
                    ipo_dates[ticker] = yahoo_start
                else:
                    ipo_dates[ticker] = None
            else:
                ipo_dates[ticker] = None
        ipo_dates = pd.Series(ipo_dates)
        tickers = df_weights[df_weights["Weight"] > 0]["Ticker"].tolist()
        tickers += ["^NDX", "^SPX"]
        tickers = sorted(set(t for t in tickers if t in df_prices.columns))
        if not tickers:
            st.warning("No valid tickers found in reconstructed prices.")
        else:
            selected_ticker = st.selectbox("Select Ticker", tickers)
            price_series = df_prices[selected_ticker].dropna()
            ipo_date = ipo_dates.get(selected_ticker)
            fallback_date = df_yahoo.index[0]
            use_ipo = ipo_date if isinstance(ipo_date, pd.Timestamp) else fallback_date
            recon_part = price_series[price_series.index < use_ipo]
            actual_part = price_series[price_series.index >= use_ipo]
            fig = go.Figure()
            if not recon_part.empty:
                fig.add_trace(go.Scatter(
                    x=recon_part.index,
                    y=recon_part.values,
                    mode='lines',
                    name='Reconstructed',
                    line=dict(color='orange')
                ))
            if not actual_part.empty:
                fig.add_trace(go.Scatter(
                    x=actual_part.index,
                    y=actual_part.values,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='#5dade2')
                ))
            max_date = price_series.idxmax()
            max_value = price_series.max()
            min_date = price_series.idxmin()
            min_value = price_series.min()
            fig.add_trace(go.Scatter(
                x=[max_date], y=[max_value],
                mode='markers+text',
                name='High',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=[f"High: {max_value:.2f}<br>{max_date.date()}"],
                textfont=dict(size=12),
                textposition="top center",
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[min_date], y=[min_value],
                mode='markers+text',
                name='Low',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text=[f"Low: {min_value:.2f}<br>{min_date.date()}"],
                textfont=dict(size=12),
                textposition="bottom center",
                showlegend=False
            ))
            fig.update_layout(
                title=f"{selected_ticker} Reconstructed vs Actual Price",
                xaxis_title="Date",
                yaxis_title="Price",
                height=1040,
                font=dict(size=18),
                legend_title="Segment",
                legend=dict(font=dict(size=14))
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Validation Check: Reconstructed vs Yahoo Prices")
            tickers_to_check = set(df_weights[df_weights["Weight"] > 0]["Ticker"].tolist() + ["^NDX", "^SPX"])
            filtered_tickers = df_yahoo.columns.intersection(df_prices.columns).intersection(tickers_to_check)
            validation_data = []
            for ticker in sorted(filtered_tickers):
                ipo = ipo_dates.get(ticker)
                if ipo is None:
                    recon_slice = df_prices[ticker]
                    yahoo_slice = df_yahoo[ticker]
                else:
                    recon_slice = df_prices.loc[df_prices.index >= ipo, ticker]
                    yahoo_slice = df_yahoo.loc[df_yahoo.index >= ipo, ticker]
                joined = pd.DataFrame({"recon": recon_slice, "yahoo": yahoo_slice}).dropna()
                mismatches = (np.round(joined["recon"], 6) != np.round(joined["yahoo"], 6)).sum()
                if not joined.empty:
                    mae = mean_absolute_error(joined["yahoo"], joined["recon"])
                    r2 = r2_score(joined["yahoo"], joined["recon"])
                else:
                    mae = np.nan
                    r2 = np.nan
                validation_data.append({
                    "Ticker": ticker,
                    "IPO Date": "" if ipo is None else ipo.date(),
                    "Checked Dates": len(joined),
                    "Mismatches": mismatches,
                    "MAE": f"{mae:.2f}" if pd.notnull(mae) else "",
                    "R2 Score": f"{r2:.2f}" if pd.notnull(r2) else "",
                    "Status": "✅" if mismatches == 0 else "❌"
                })
            if validation_data:
                df_valid = pd.DataFrame(validation_data).sort_values("Ticker").set_index("Ticker")
                row_height = 38
                table_height = row_height * len(df_valid)
                st.dataframe(df_valid, use_container_width=True, height=table_height)
            else:
                st.info("No valid tickers found to validate.")
            st.subheader("Source Attribution by Ticker")
            sources_path = project_root / paths["recon_sources_output"]
            if sources_path.exists():
                df_sources = pd.read_csv(sources_path, index_col=0, parse_dates=True)
                valid_tickers = df_weights[df_weights["Weight"] > 0]["Ticker"].unique()
                df_sources = df_sources[df_sources.columns.intersection(valid_tickers)]
                def format_ranges(series):
                    series = series.dropna()
                    if series.empty:
                        return ""
                    result = []
                    prev_type = None
                    range_start = None
                    prev_date = None
                    for date, val in series.items():
                        if val != prev_type:
                            if prev_type is not None:
                                if range_start == prev_date:
                                    result.append(range_start.strftime("%Y-%m-%d"))
                                else:
                                    result.append(f"{range_start.strftime('%Y-%m-%d')} to {prev_date.strftime('%Y-%m-%d')}")
                            range_start = date
                            prev_type = val
                        prev_date = date
                    if prev_type is not None:
                        if range_start == prev_date:
                            result.append(range_start.strftime("%Y-%m-%d"))
                        else:
                            result.append(f"{range_start.strftime('%Y-%m-%d')} to {prev_date.strftime('%Y-%m-%d')}")
                    return ", ".join(result)
                summary_data = []
                for ticker in df_sources.columns:
                    s = df_sources[ticker]
                    usage = {
                        "Ticker": ticker,
                        "REAL": format_ranges(s[s == "REAL"]),
                        "PROXY": format_ranges(s[s == "PROXY"]),
                        "GARCH": format_ranges(s[s == "GARCH"])
                    }
                    summary_data.append(usage)
                df_summary = pd.DataFrame(summary_data).set_index("Ticker")
                summary_height = 38 * len(df_summary)
                st.dataframe(df_summary, use_container_width=True, height=summary_height)
            else:
                st.info("Source attribution file not found. Please run the reconstruction pipeline.")
    else:
        st.info("Missing one or more data files: reconstructed_prices.csv, portfolio_weights.csv, yahoo_prices.csv.") 