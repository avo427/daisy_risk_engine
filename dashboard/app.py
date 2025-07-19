import streamlit as st
import pandas as pd
import subprocess
import yaml
from pathlib import Path
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

# === Load Config ===
def load_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(updated_config):
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(updated_config, f, sort_keys=False)

config = load_config()
user_settings = config.get("user_settings", {})
paths = config.get("paths", {})
theme_proxies = config.get("theme_proxies", {})
ticker_themes = config.get("ticker_themes", {})
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

# === Load CSVs ===
@st.cache_data
def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=None)

def load_all_data():
    return {
        "realized": load_csv(project_root / paths["realized_output"]),
        "roll": load_csv(data_dir / "rolling_metrics/rolling_metrics_long.csv"),
        "corr": load_csv(data_dir / "correlation_matrix.csv"),
        "vol": load_csv(data_dir / "vol_contribution.csv"),
        "forecast": load_csv(project_root / paths["forecast_output"]),
        "forecast_roll": load_csv(data_dir / "rolling_metrics/forecast_rolling_long.csv"),
    }

# === UI Setup ===
st.set_page_config(page_title="Daisy Risk Engine", layout="wide")
st.title("Daisy Risk Engine Dashboard")

# === Sidebar Controls ===
with st.sidebar:
    st.header("Engine Controls")

    st.subheader("User Settings")
    years = st.number_input("Years of History", min_value=1, max_value=50, value=user_settings.get("years", 20))
    risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=0.2,
                                     value=user_settings.get("risk_free_rate", 0.05), step=0.001)
    random_state = st.number_input("Random Seed", min_value=0, value=user_settings.get("random_state", 42), step=1)
    total_returns = st.checkbox("Use Total Returns", value=user_settings.get("total_returns", True))

    if st.button("Save Settings"):
        config["user_settings"]["years"] = years
        config["user_settings"]["total_returns"] = total_returns
        config["user_settings"]["risk_free_rate"] = float(risk_free_rate)
        config["user_settings"]["random_state"] = int(random_state)
        save_config(config)
        st.success("Settings saved to config.yaml")
        st.cache_data.clear()
        st.session_state.update(load_all_data())
        st.rerun()

    if st.button("Run Daisy Risk Engine"):
        with st.spinner("Running full pipeline..."):
            result = subprocess.run(
                [sys.executable, str(project_root / "main.py")],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            if result.returncode == 0:
                st.success("Pipeline complete.")
                st.cache_data.clear()
                st.session_state.update(load_all_data())
                st.rerun()
            else:
                st.error("Pipeline failed.")
                st.text(result.stderr)

    if st.button("Reload Data Only"):
        st.cache_data.clear()
        st.session_state.update(load_all_data())
        st.rerun()

# === Initialize session data ===
if "realized" not in st.session_state:
    st.session_state.update(load_all_data())

df_realized = st.session_state.get("realized", pd.DataFrame())
df_roll = st.session_state.get("roll", pd.DataFrame())
df_corr = st.session_state.get("corr", pd.DataFrame())
df_vol = st.session_state.get("vol", pd.DataFrame())
df_forecast = st.session_state.get("forecast", pd.DataFrame())
df_fore_roll = st.session_state.get("forecast_roll", pd.DataFrame())

# === Tabs ===
tabs = st.tabs(["Realized Risk", "Forecast Risk", "Volatility-Based Sizing", "Themes & Proxies","Reconstructed Prices"])

# === Tab 1: Realized Risk ===
with tabs[0]:
    st.subheader("Realized Metrics")
    if df_realized.empty:
        st.info("No realized metrics available.")
    else:
        df_realized = df_realized.copy()
        percent_cols = [
            "Ann. Return", "Ann. Volatility", "Max Drawdown",
            "VaR (95%)", "CVaR (95%)", "Hit Ratio",
            "Up Capture (NDX)", "Down Capture (NDX)"
        ]
        for col in percent_cols:
            if col in df_realized.columns:
                df_realized[col] = (
                    df_realized[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace('N/A', '', regex=False)
                    .replace('', np.nan)
                    .astype(float) / 100.0
                )

        float_cols = [
            "Sharpe", "Sortino", "Skew", "Kurtosis",
            "Beta (NDX)", "Beta (SPX)", "Information Ratio"
        ]
        for col in float_cols:
            if col in df_realized.columns:
                df_realized[col] = pd.to_numeric(df_realized[col], errors="coerce")

        def realized_style(row):
            return ["color: orange" if row.name == "PORTFOLIO" else "" for _ in row]

        col_widths = {
            col: min(max(80, int(df_realized[col].astype(str).str.len().max() * 7)), 160)
            for col in df_realized.columns if col != "Ticker"
        }
        realized_height = min(600, 35 * (len(df_realized) + 1))
        st.dataframe(
            df_realized.set_index("Ticker").style
                .apply(realized_style, axis=1)
                .set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold"), ("white-space", "normal")]},
                    *[{"selector": f"td.col{i}", "props": [("min-width", f"{w}px")]}
                      for i, w in enumerate(col_widths.values())]
                ])
                .format({
                    "Ann. Return": "{:.2%}",
                    "Ann. Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Skew": "{:.2f}",
                    "Kurtosis": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "VaR (95%)": "{:.2%}",
                    "CVaR (95%)": "{:.2%}",
                    "Hit Ratio": "{:.2%}",
                    "Beta (NDX)": "{:.2f}",
                    "Beta (SPX)": "{:.2f}",
                    "Up Capture (NDX)": "{:.2%}",
                    "Down Capture (NDX)": "{:.2%}",
                    "Information Ratio": "{:.2f}"
                }),
            use_container_width=True,
            height=realized_height
        )

    st.subheader("Rolling Metrics")
    if not df_roll.empty:
        selected_metric = st.selectbox("Choose rolling metric:", df_roll["Metric"].unique())
        chart_data = df_roll[df_roll["Metric"] == selected_metric]
        fig = px.line(chart_data, x="Date", y="Value", color="Ticker", title=selected_metric)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    if not df_corr.empty:
        df_corr = df_corr.copy()

        # Fix "Unnamed: 0" if present
        if df_corr.columns[0].lower().startswith("unnamed"):
            df_corr.set_index(df_corr.columns[0], inplace=True)

        # Fixed width rendering using columns
        col1, _, _ = st.columns([1.5, 0.2, 0.3])  # Lock width
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig, clear_figure=True)

    st.subheader("Volatility Contribution")
    if not df_vol.empty:
        df_vol = df_vol.iloc[:, :2]
        df_vol.columns = ["Ticker", "Vol_Contribution"]
        fig2 = px.pie(df_vol, names="Ticker", values="Vol_Contribution", title="Volatility Contribution")
        st.plotly_chart(fig2, use_container_width=True)

# === Tab 2: Forecast ===
with tabs[1]:
    st.subheader("Forecast Metrics")
    if df_forecast.empty:
        st.info("No forecast metrics available.")
    else:
        df_forecast = df_forecast.copy()
        df_forecast["VaR ($)"] = df_forecast["VaR ($)"].apply(lambda x: f"-${abs(float(str(x).replace(',', '').replace('$', ''))):,.0f}" if pd.notnull(x) and float(str(x).replace(',', '').replace('$', '')) != 0 else "")
        df_forecast["CVaR ($)"] = df_forecast["CVaR ($)"].apply(lambda x: f"-${abs(float(str(x).replace(',', '').replace('$', ''))):,.0f}" if pd.notnull(x) and float(str(x).replace(',', '').replace('$', '')) != 0 else "")

        def forecast_style(row):
            return ["color: orange" if row.name == "PORTFOLIO" else "" for _ in row]

        col_widths_forecast = {
            col: min(max(80, int(df_forecast[col].astype(str).str.len().max() * 7)), 160)
            for col in df_forecast.columns if col != "Ticker"
        }

        forecast_height = min(600, 35 * (len(df_forecast) + 1))
        st.dataframe(
            df_forecast.set_index("Ticker").style
                .apply(forecast_style, axis=1)
                .set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold"), ("white-space", "normal")]},
                    *[{"selector": f"td.col{i}", "props": [("min-width", f"{w}px")]}
                      for i, w in enumerate(col_widths_forecast.values())]
                ]),
            use_container_width=True,
            height=forecast_height
        )

    st.subheader("Rolling Forecast Metrics")
    if not df_fore_roll.empty:
        selected_fore_metric = st.selectbox("Choose forecast rolling metric:", df_fore_roll["Metric"].unique())
        forecast_data = df_fore_roll[df_fore_roll["Metric"] == selected_fore_metric]
        fig = px.line(forecast_data, x="Date", y="Value", color="Ticker", title=selected_fore_metric)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Risk Contribution")
    forecast_contrib_path = data_dir / "forecast_risk_contributions.csv"
    if forecast_contrib_path.exists():
        df_forecast_contrib = pd.read_csv(forecast_contrib_path)
        models = df_forecast_contrib["Model"].unique()
        selected_model = st.selectbox("Select Model for Risk Contribution:", models)
        df_selected = df_forecast_contrib[df_forecast_contrib["Model"] == selected_model].copy()

        df_selected["Forecast_Risk_%"] = df_selected["Forecast_Risk_%"].round(2)
        df_selected["Forecast_Risk_Contribution"] = df_selected["Forecast_Risk_Contribution"] * 100

        fig_bar = px.bar(
            df_selected.sort_values("Forecast_Risk_Contribution", ascending=True),
            x="Forecast_Risk_Contribution", y="Ticker", orientation="h",
            labels={"Forecast_Risk_Contribution": "Marginal Risk Contribution (%)"},
            title="Marginal Risk Contribution by Ticker"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            df_selected.sort_values("Forecast_Risk_Contribution", ascending=False),
            values="Forecast_Risk_Contribution",
            names="Ticker",
            title="Forecast Risk Contribution (Pie)",
            hole=0.35
        )
        fig_pie.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:.2f}%")
        st.plotly_chart(fig_pie, use_container_width=True)


# === Tab 3: Volatility-Based Sizing ===
with tabs[2]:
    st.subheader("Volatility-Based Sizing")

    sizing_path = data_dir / "vol_sizing_weights_long.csv"
    if sizing_path.exists():
        df_sizing = pd.read_csv(sizing_path)
        model_options = df_sizing["Model"].unique().tolist()
        selected_model = st.selectbox("Select Volatility Forecast Model", model_options)

        filtered_df = df_sizing[df_sizing["Model"] == selected_model].copy()

        rename_map = {
            "ForecastVol": "Forecast Volatility",
            "BaseWeight": "Current Weight",
            "VolAdjWeight": "Adj. Volatility Weight",
            "CurrentDollar": "Current MV",
            "TargetDollar": "Target MV",
            "DeltaDollar": "(+/-)",
            "Price": "Last Price",
            "DeltaShares": "Shares Delta"
        }
        filtered_df.rename(columns=rename_map, inplace=True)

        numeric_cols = list(rename_map.values())
        for col in numeric_cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

        # Format Shares Delta with + for positive values
        def format_shares_delta(x):
            try:
                val = float(x)
                return f"+{val:,.0f}" if val > 0 else f"{val:,.0f}"
            except:
                return ""

        # Formatters
        formatters = {
            "Forecast Volatility": "{:.2%}".format,
            "Current Weight": "{:.2%}".format,
            "Adj. Volatility Weight": "{:.2%}".format,
            "Current MV": lambda x: f"${x:,.0f}",
            "Target MV": lambda x: f"${x:,.0f}",
            "(+/-)": lambda x: f"${x:,.0f}",
            "Last Price": lambda x: f"${x:,.2f}",
            "Shares Delta": format_shares_delta
        }

        table_height = min(600, 35 * (len(filtered_df) + 1))
        st.dataframe(
            filtered_df[["Ticker"] + numeric_cols].style.format(formatters),
            use_container_width=True,
            height=table_height
        )

        st.subheader("Volatility-Adjusted Weights (Pie Chart)")
        fig = px.pie(
            filtered_df,
            names="Ticker",
            values="Adj. Volatility Weight",
            title=f"Volatility-Adjusted Weights using {selected_model}",
            hole=0.3
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        # === Volatility Contribution Chart ===
        st.subheader("Expected Volatility Contribution")
        volcontrib_df = filtered_df.copy()
        volcontrib_df["Volatility Contribution"] = (
            volcontrib_df["Forecast Volatility"] * volcontrib_df["Adj. Volatility Weight"]
        )
        total_contrib = volcontrib_df["Volatility Contribution"].sum()
        volcontrib_df["Volatility Contribution"] = (
            volcontrib_df["Volatility Contribution"] / total_contrib * 100
        )

        fig_contrib = px.pie(
            volcontrib_df,
            names="Ticker",
            values="Volatility Contribution",
            title="Expected Volatility Contribution (Post-Sizing)",
            hole=0.3
        )
        fig_contrib.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:.2f}%")
        st.plotly_chart(fig_contrib, use_container_width=True)

    else:
        st.info("No volatility-based sizing output found. Run the engine to generate it.")

# === Tab 4: Theme Mapping Editor ===
with tabs[3]:
    st.subheader("Theme Proxies")
    theme_proxy_df = pd.DataFrame(list(theme_proxies.items()), columns=["Theme", "ETF Proxy"])
    theme_proxy_height = 35 * (len(theme_proxy_df) + 2)
    theme_proxy_edit = st.data_editor(
        theme_proxy_df,
        num_rows="dynamic",
        use_container_width=True,
        height=theme_proxy_height
    )

    st.subheader("Ticker-to-Theme Map")
    ticker_theme_df = pd.DataFrame(list(ticker_themes.items()), columns=["Ticker", "Theme"])
    ticker_theme_height = 35 * (len(ticker_theme_df) + 2)
    ticker_theme_edit = st.data_editor(
        ticker_theme_df,
        num_rows="dynamic",
        use_container_width=True,
        height=ticker_theme_height
    )

    if st.button("Save Theme & Ticker Mappings"):
        config["theme_proxies"] = dict(zip(theme_proxy_edit["Theme"], theme_proxy_edit["ETF Proxy"]))
        config["ticker_themes"] = dict(zip(ticker_theme_edit["Ticker"], ticker_theme_edit["Theme"]))
        save_config(config)
        st.success("Theme and ticker mappings saved.")
        st.rerun()

# === Tab 5: Reconstructed Prices ===
with tabs[4]:
    st.subheader("Reconstructed Prices Viewer")

    # Load required files
    recon_path = data_dir / "reconstructed_prices.csv"
    weights_path = project_root / paths["portfolio_weights"]
    yahoo_path = data_dir / "yahoo_prices.csv"

    if recon_path.exists() and weights_path.exists() and yahoo_path.exists():
        df_prices = pd.read_csv(recon_path, index_col=0, parse_dates=True)
        df_weights = pd.read_csv(weights_path)
        df_yahoo = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)

        # === Robust IPO date detection (None if no reconstruction needed) ===
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

        # Ticker list for dropdown
        tickers = df_weights[df_weights["Weight"] > 0]["Ticker"].tolist()
        tickers += ["^NDX", "^SPX"]
        tickers = sorted(set(t for t in tickers if t in df_prices.columns))

        if not tickers:
            st.warning("No valid tickers found in reconstructed prices.")
        else:
            selected_ticker = st.selectbox("Select Ticker", tickers)

            import plotly.graph_objects as go
            import numpy as np

            price_series = df_prices[selected_ticker].dropna()
            ipo_date = ipo_dates.get(selected_ticker)

            # Fallback: use start of yahoo_prices if ipo_date is None
            fallback_date = df_yahoo.index[0]
            use_ipo = ipo_date if isinstance(ipo_date, pd.Timestamp) else fallback_date

            # Determine segments
            recon_part = price_series[price_series.index < use_ipo]
            actual_part = price_series[price_series.index >= use_ipo]

            fig = go.Figure()

            # Reconstructed segment
            if not recon_part.empty:
                fig.add_trace(go.Scatter(
                    x=recon_part.index,
                    y=recon_part.values,
                    mode='lines',
                    name='Reconstructed',
                    line=dict(color='orange')
                ))

            # Actual segment
            if not actual_part.empty:
                fig.add_trace(go.Scatter(
                    x=actual_part.index,
                    y=actual_part.values,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='#5dade2')  # lighter blue
                ))

            # High / Low markers
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
                textfont=dict(size=18),
                textposition="top center",
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[min_date], y=[min_value],
                mode='markers+text',
                name='Low',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text=[f"Low: {min_value:.2f}<br>{min_date.date()}"],
                textfont=dict(size=18),
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
                legend=dict(font=dict(size=18))
            )
            st.plotly_chart(fig, use_container_width=True)

            # === Validation Table ===
            st.subheader("Validation Check: Reconstructed vs Yahoo Prices")

            tickers_to_check = set(df_weights[df_weights["Weight"] > 0]["Ticker"].tolist() + ["^NDX", "^SPX"])
            filtered_tickers = df_yahoo.columns.intersection(df_prices.columns).intersection(tickers_to_check)

            validation_data = []
            for ticker in sorted(filtered_tickers):
                ipo = ipo_dates.get(ticker)
                if ipo is None:
                    # Compare full history
                    recon_slice = df_prices[ticker]
                    yahoo_slice = df_yahoo[ticker]
                else:
                    # Compare from IPO onward
                    recon_slice = df_prices.loc[df_prices.index >= ipo, ticker]
                    yahoo_slice = df_yahoo.loc[df_yahoo.index >= ipo, ticker]

                joined = pd.DataFrame({"recon": recon_slice, "yahoo": yahoo_slice}).dropna()
                mismatches = (np.round(joined["recon"], 6) != np.round(joined["yahoo"], 6)).sum()

                validation_data.append({
                    "Ticker": ticker,
                    "IPO Date": "" if ipo is None else ipo.date(),
                    "Checked Dates": len(joined),
                    "Mismatches": mismatches,
                    "Status": "✅" if mismatches == 0 else "❌"
                })

            if validation_data:
                df_valid = pd.DataFrame(validation_data).sort_values("Ticker")
                row_height = 35
                table_height = row_height * (len(df_valid) + 1)
                st.dataframe(df_valid, use_container_width=True, height=table_height)
            else:
                st.info("No valid tickers found to validate.")

    else:
        st.info("Missing one or more data files: reconstructed_prices.csv, portfolio_weights.csv, yahoo_prices.csv.")
