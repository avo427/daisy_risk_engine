import streamlit as st
import pandas as pd
import subprocess
import yaml
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import sys  # Needed to modify sys.path for parent-level imports
from sklearn.metrics import mean_absolute_error, r2_score

# === Add parent directory to sys.path so we can import main.py ===
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import functions for pipeline steps
try:
    from main import run_full_pipeline, run_risk_analysis, run_factor_exposure
except ImportError as e:
    st.error(f"❌ Failed to import functions from main.py:\n{e}")
    run_full_pipeline = run_risk_analysis = run_factor_exposure = None

# === Load Config ===
def load_config():
    config_path = project_root / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(updated_config):
    config_path = project_root / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(updated_config, f, sort_keys=False)

config = load_config()
user_settings = config.get("user_settings", {})
paths = config.get("paths", {})
theme_proxies = config.get("theme_proxies", {})
ticker_themes = config.get("ticker_themes", {})
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
        "roll": load_csv(project_root / paths["realized_rolling_output"]),
        "corr": load_csv(project_root / paths["correlation_matrix"]),
        "vol": load_csv(project_root / paths["vol_contribution"]),
        "forecast": load_csv(project_root / paths["forecast_output"]),
        "forecast_roll": load_csv(project_root / paths["forecast_rolling_output"]),
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

    st.markdown("---")  # Horizontal separator before pipeline execution

    st.subheader("Pipeline Configuration")

    # Radio buttons instead of checkboxes (only one can be selected at a time)
    pipeline_option = st.radio(
        "Select the pipeline to run",
        ["Full Pipeline", "Risk Analysis", "Factor Exposure"],
        index=0  # Default is "Full Pipeline"
    )

    # Execute the selected components when the user presses the button
    if st.button("🐶 Run Daisy Risk Engine"):
        import subprocess
        import sys

        mode_map = {
            "Full Pipeline": "full",
            "Risk Analysis": "risk",
            "Factor Exposure": "factor"
        }
        selected_mode = mode_map[pipeline_option]  # Map UI label to CLI mode

        with st.spinner(f"Running {pipeline_option}... Please wait."):
            result = subprocess.run(
                [sys.executable, str(project_root / "main.py"), "--mode", selected_mode],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("✅ Run completed successfully.")  # Show success in UI
            st.cache_data.clear()
            st.session_state.update(load_all_data())  # Reload data
            st.rerun()  # Refresh app with new state
        else:
            st.error("❌ Pipeline run failed. See logs below.")  # Show error
            st.code(result.stdout + "\n" + result.stderr, language="bash")  # Print logs

    # Button to reload data without running the pipeline
    if st.button("Reload Data Only"):
        st.cache_data.clear()
        st.session_state.update(load_all_data())
        st.rerun()

    st.markdown("---")  # Horizontal separator before runtime


# === Initialize session data ===
if "realized" not in st.session_state:
    st.session_state.update(load_all_data())


df_realized = st.session_state.get("realized", pd.DataFrame())
df_roll = st.session_state.get("roll", pd.DataFrame())
df_corr = st.session_state.get("corr", pd.DataFrame())
df_vol = st.session_state.get("vol", pd.DataFrame())
df_forecast = st.session_state.get("forecast", pd.DataFrame())
df_fore_roll = st.session_state.get("forecast_roll", pd.DataFrame())
if not df_fore_roll.empty:
    df_fore_roll["Ticker"] = df_fore_roll["Ticker"].astype(str).str.strip().str.upper()


# === Tabs ===
tabs = st.tabs(["Realized Risk", 
                "Forecast Risk", 
                "Volatility-Based Sizing",
                "Factor Exposure", 
                "Themes & Proxies",
                "Reconstructed Prices"])

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
        metric_map = {
            "Rolling Volatility": "rolling_vol",
            "Rolling Return": "rolling_ret",
            "Rolling Sharpe": "rolling_sharpe"
        }
        window_map = {
            "21D (1M)": "21d",
            "60D (3M)": "60d",
            "126D (6M)": "126d"
        }

        col1, col2 = st.columns(2)
        with col1:
            selected_metric_label = st.selectbox("Select Rolling Metric", list(metric_map.keys()))
        with col2:
            selected_window_label = st.selectbox("Select Time Frame", list(window_map.keys()))

        metric_key = metric_map[selected_metric_label]
        window_key = window_map[selected_window_label]
        full_metric_name = f"{metric_key}_{window_key}"

        chart_data = df_roll[df_roll["Metric"] == full_metric_name]
        if chart_data.empty:
            st.warning(f"No data found for {selected_metric_label} {selected_window_label}")
        else:
            y_axis_label = (
                "Volatility (%)" if "vol" in metric_key
                else "Return (%)" if "ret" in metric_key
                else "Sharpe Ratio"
            )

            all_tickers = chart_data["Ticker"].unique().tolist()
            if "selected_rolling_tickers" not in st.session_state:
                default_selection = ["PORTFOLIO"] if "PORTFOLIO" in all_tickers else all_tickers
                st.session_state["selected_rolling_tickers"] = default_selection

            selected_tickers = st.multiselect(
                "Select Tickers:",
                options=all_tickers,
                default=st.session_state["selected_rolling_tickers"]
            )
            st.session_state["selected_rolling_tickers"] = selected_tickers

            chart_data = chart_data[chart_data["Ticker"].isin(selected_tickers)]
            fig = px.line(
                chart_data, x="Date", y="Value", color="Ticker",
                title=f"{selected_metric_label} - {selected_window_label}",
                labels={"Value": y_axis_label}
            )
            fig.update_layout(height=800)
            fig.update_traces(line=dict(width=1))
            fig.update_yaxes(tickformat=".2%" if "vol" in metric_key or "ret" in metric_key else ".2f")
            st.plotly_chart(fig, use_container_width=True)



    st.subheader("Correlation Matrix")
    if not df_corr.empty:
        df_corr = df_corr.copy()
        if df_corr.columns[0].lower().startswith("unnamed"):
            df_corr.set_index(df_corr.columns[0], inplace=True)
            df_corr.index.name = None

        tickers = df_corr.columns.tolist()
        z = df_corr.values

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, '#2166ac'],
                [0.5, '#f7f7f7'],
                [1.0, '#b2182b']
            ],
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
            showscale=True
        ))

        annotations = []
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                annotations.append(dict(
                    x=tickers[j],
                    y=tickers[i],
                    text=f"{z[i][j]:.2f}",
                    showarrow=False,
                    font=dict(color="black", size=12)
                ))

        fig.update_layout(
            width=900,
            height=900,
            font=dict(size=14),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            margin=dict(t=40, b=40),
            annotations=annotations
        )
        st.plotly_chart(fig, use_container_width=False)

    st.subheader("Volatility Contribution")
    if not df_vol.empty:
        df_vol = df_vol.iloc[:, :2]
        df_vol.columns = ["Ticker", "Vol_Contribution"]

        fig2 = px.pie(
            df_vol,
            names="Ticker",
            values="Vol_Contribution",
            hole=0.3
        )
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

# === Tab 2: Forecast Risk ===
with tabs[1]:
    st.subheader("Forecast Metrics")
    if df_forecast.empty:
        st.info("No forecast metrics available.")
    else:
        df_forecast = df_forecast.copy()

        # Clean and convert percent columns
        percent_cols_forecast = [
            "EWMA (5D)", "EWMA (20D)",
            "Garch Volatility", "E-Garch Volatility",
            "VaR (95%)", "CVaR (95%)"
        ]
        for col in percent_cols_forecast:
            if col in df_forecast.columns:
                df_forecast[col] = (
                    df_forecast[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace('N/A', '', regex=False)
                    .replace('', np.nan)
                    .astype(float) / 100.0
                )

        # Clean and convert dollar columns
        dollar_cols_forecast = ["VaR ($)", "CVaR ($)"]
        for col in dollar_cols_forecast:
            if col in df_forecast.columns:
                df_forecast[col] = (
                    df_forecast[col]
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .replace('', np.nan)
                    .astype(float)
                )

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
                ])
                .format({
                    "EWMA (5D)": "{:.2%}",
                    "EWMA (20D)": "{:.2%}",
                    "Garch Volatility": "{:.2%}",
                    "E-Garch Volatility": "{:.2%}",
                    "VaR (95%)": "{:.2%}",
                    "CVaR (95%)": "{:.2%}",
                    "VaR ($)": lambda x: f"-${abs(x):,.0f}" if pd.notnull(x) else "",
                    "CVaR ($)": lambda x: f"-${abs(x):,.0f}" if pd.notnull(x) else ""
                }),
            use_container_width=True,
            height=forecast_height
        )

    # === Rolling Forecast Metrics ===
    st.subheader("Rolling Forecast Metrics")
    if not df_fore_roll.empty:
        df_fore_roll.rename(columns={"Horizon": "Time Frame", "Metric": "Value"}, inplace=True)
        unique_forecast_tickers = sorted(df_fore_roll["Ticker"].unique().tolist())
        df_fore_roll["Ticker"] = df_fore_roll["Ticker"].astype(str).str.strip().str.upper()

        model_options = sorted(df_fore_roll["Model"].unique())
        time_options = sorted(df_fore_roll["Time Frame"].unique())

        col1, col2 = st.columns(2)
        with col1:
            default_model_index = model_options.index("EGARCH") if "EGARCH" in model_options else 0
            selected_model = st.selectbox("Select Model", model_options, index=default_model_index)
        with col2:
            default_time_index = time_options.index("1D") if "1D" in time_options else 0
            selected_time = st.selectbox("Select Time Frame", time_options, index=default_time_index)

        df_filtered = df_fore_roll[
            (df_fore_roll["Model"] == selected_model) &
            (df_fore_roll["Time Frame"] == selected_time)
        ]

        if df_filtered.empty:
            st.warning("No forecast rolling data for this model and time frame.")
        else:
            all_forecast_tickers = sorted(df_fore_roll["Ticker"].unique().tolist())
            if "selected_forecast_rolling_tickers" not in st.session_state:
                st.session_state["selected_forecast_rolling_tickers"] = (
                    ["PORTFOLIO"] if "PORTFOLIO" in all_forecast_tickers else all_forecast_tickers
                )

            selected_forecast_tickers = st.multiselect(
                "Select Tickers:",
                options=all_forecast_tickers,
                default=st.session_state["selected_forecast_rolling_tickers"]
            )
            st.session_state["selected_forecast_rolling_tickers"] = selected_forecast_tickers

            df_filtered = df_filtered[df_filtered["Ticker"].isin(selected_forecast_tickers)]

            fig = px.line(
                df_filtered,
                x="Date",
                y="Value",
                color="Ticker",
                title=f"{selected_model} - {selected_time}",
                labels={"Value": "Volatility (%)"}
            )
            fig.update_layout(height=800)
            fig.update_traces(line=dict(width=1))
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Risk Contribution")
        forecast_contrib_path = project_root / paths["forecast_risk_contributions"]
        if forecast_contrib_path.exists():
            df_forecast_contrib = pd.read_csv(forecast_contrib_path)
            models = df_forecast_contrib["Model"].unique()

            default_index = list(models).index("E-Garch Volatility") if "E-Garch Volatility" in models else 0
            selected_model = st.selectbox("Select Model for Risk Contribution:", models, index=default_index)

            df_selected = df_forecast_contrib[df_forecast_contrib["Model"] == selected_model].copy()
            df_selected["Forecast_Risk_%"] = df_selected["Forecast_Risk_%"].round(2)
            df_selected["Forecast_Risk_Contribution"] = df_selected["Forecast_Risk_Contribution"] * 100

            fig_bar = px.bar(
                df_selected.sort_values("Forecast_Risk_Contribution", ascending=True),
                x="Forecast_Risk_Contribution",
                y="Ticker",
                orientation="h",
                labels={"Forecast_Risk_Contribution": "Marginal Risk Contribution (%)", "Ticker": ""},
                title="Marginal Risk Contribution"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                df_selected.sort_values("Forecast_Risk_Contribution", ascending=False),
                values="Forecast_Risk_Contribution",
                names="Ticker",
                title="Forecast Risk Contribution",
                hole=0.3
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=0
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No forecast risk contribution file found. Please run the risk engine first.")


# === Tab 3: Volatility-Based Sizing ===
with tabs[2]:
    st.subheader("Volatility-Based Sizing")

    sizing_path = project_root / paths["vol_sizing_output"]
    if sizing_path.exists():
        df_sizing = pd.read_csv(sizing_path)
        model_options = df_sizing["Model"].unique().tolist()
        selected_model = st.selectbox("Select Volatility Forecast Model", model_options)

        filtered_df = df_sizing[df_sizing["Model"] == selected_model].copy()
        filtered_df.drop(columns=["Model"], inplace=True)

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

        percent_cols = ["Forecast Volatility", "Current Weight", "Adj. Volatility Weight"]
        dollar_cols = ["Current MV", "Target MV", "(+/-)", "Last Price"]
        share_cols = ["Shares Delta"]

        for col in percent_cols + dollar_cols + share_cols:
            if col in filtered_df.columns:
                filtered_df[col] = (
                    filtered_df[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .replace('', np.nan)
                    .astype(float)
                )

        def sizing_style(row):
            return ["color: orange" if row.name == "PORTFOLIO" else "" for _ in row]

        col_widths_sizing = {
            col: min(max(80, int(filtered_df[col].astype(str).str.len().max() * 7)), 160)
            for col in filtered_df.columns if col != "Ticker"
        }

        table_height = min(600, 35 * (len(filtered_df) + 1))

        def format_shares(x):
            if pd.isnull(x): return ""
            return f"+{x:,.0f}" if x > 0 else f"{x:,.0f}"

        st.dataframe(
            filtered_df.set_index("Ticker").style
                .apply(sizing_style, axis=1)
                .set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold"), ("white-space", "normal")]},
                    *[{"selector": f"td.col{i}", "props": [("min-width", f"{w}px")]}
                      for i, w in enumerate(col_widths_sizing.values())]
                ])
                .format({
                    "Forecast Volatility": "{:.2%}",
                    "Current Weight": "{:.2%}",
                    "Adj. Volatility Weight": "{:.2%}",
                    "Current MV": "${:,.0f}",
                    "Target MV": "${:,.0f}",
                    "(+/-)": "${:,.0f}",
                    "Last Price": "${:,.2f}",
                    "Shares Delta": format_shares
                }),
            use_container_width=True,
            height=table_height
        )

        # === Pie Chart: Volatility-Adjusted Weights ===
        st.subheader("Volatility-Adjusted Weights")
        fig = px.pie(
            filtered_df,
            names="Ticker",
            values="Adj. Volatility Weight",
            hole=0.3
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # === Pie Chart: Expected Volatility Contribution ===
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
            hole=0.3
        )
        fig_contrib.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:.2f}%")
        fig_contrib.update_layout(
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0
            )
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    else:
        st.info("No volatility-based sizing output found. Run the engine to generate it.")

# === Tab 4: Factor Exposure ===
with tabs[3]:
    st.subheader("Latest Factor Exposures")

    # Load weights and valid tickers
    weights_path = project_root / paths["portfolio_weights"]
    exposures_path = project_root / paths["factor_exposures"]
    df_weights = pd.read_csv(weights_path)
    valid_tickers = set(df_weights[df_weights["Weight"] > 0]["Ticker"].str.upper())
    valid_tickers.update(["^NDX", "^SPX", "PORTFOLIO"])

    # Load and filter factor exposures
    if exposures_path.exists():
        df_expo = pd.read_csv(exposures_path)
        df_expo["Date"] = pd.to_datetime(df_expo["Date"])
        latest_date = df_expo["Date"].max()
        df_latest = df_expo[df_expo["Date"] == latest_date].copy()
        df_latest["Ticker"] = df_latest["Ticker"].str.upper()
        df_latest = df_latest[df_latest["Ticker"].isin(valid_tickers)]

        df_matrix = df_latest.pivot(index="Ticker", columns="Factor", values="Beta").round(2)

        def exposure_style(row):
            return ["color: orange" if row.name == "PORTFOLIO" else "" for _ in row]

        table_height = 38 * len(df_matrix)
        st.caption(f"Exposures as of {latest_date.date()}")
        st.dataframe(
            df_matrix.style
                .apply(exposure_style, axis=1)
                .format("{:.2f}"),
            use_container_width=True,
            height=table_height
        )
    else:
        df_matrix = pd.DataFrame()
        st.info("No factor exposure file found. Please run the risk engine.")

    # === Heatmap of Latest Exposures ===
    st.subheader("Heatmap of Latest Exposures")
    if not df_matrix.empty:
        fig = px.imshow(
            df_matrix,
            labels=dict(color="Beta"),
            x=df_matrix.columns,
            y=df_matrix.index,
            color_continuous_scale=[
                [0.0, '#2166ac'],
                [0.5, '#f7f7f7'],
                [1.0, '#b2182b']
            ],
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    # === Rolling Factor Exposures ===
    st.subheader("Rolling Factor Exposures")
    rolling_path = project_root / paths["factor_rolling_long"]
    if rolling_path.exists():
        df_roll_factors = pd.read_csv(rolling_path)
        df_roll_factors["Date"] = pd.to_datetime(df_roll_factors["Date"])
        df_roll_factors["Ticker"] = df_roll_factors["Ticker"].str.upper()
        df_roll_factors = df_roll_factors[df_roll_factors["Ticker"].isin(valid_tickers)]

        available_factors = sorted(df_roll_factors["Factor"].unique())
        available_tickers = sorted(df_roll_factors["Ticker"].unique())

        col1, col2 = st.columns(2)
        with col1:
            default_factor = ["MARKET"] if "MARKET" in available_factors else available_factors[:1]
            selected_factors = st.multiselect("Select Factors", options=available_factors, default=default_factor)
        with col2:
            default_ticker = ["PORTFOLIO"] if "PORTFOLIO" in available_tickers else available_tickers[:1]
            selected_tickers = st.multiselect("Select Tickers", options=available_tickers, default=default_ticker)

        df_filtered = df_roll_factors[
            df_roll_factors["Factor"].isin(selected_factors) &
            df_roll_factors["Ticker"].isin(selected_tickers)
        ]
        df_filtered["Beta"] = df_filtered["Beta"].round(2)

        if df_filtered.empty:
            st.warning("No data for selected tickers and factors.")
        else:
            fig = px.line(
                df_filtered,
                x="Date", y="Beta", color="Factor", line_dash="Ticker",
                title="Rolling Factor Exposures"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rolling factor exposure file found.")

    # === Rolling R² ===
    st.subheader("Rolling R²")
    r2_path = project_root / paths["r2_rolling_long"]
    if r2_path.exists():
        df_r2 = pd.read_csv(r2_path)
        df_r2["Date"] = pd.to_datetime(df_r2["Date"])
        df_r2["Ticker"] = df_r2["Ticker"].str.upper()
        df_r2 = df_r2[df_r2["Ticker"].isin(valid_tickers)]

        available_r2_tickers = sorted(df_r2["Ticker"].unique())
        default_r2 = ["PORTFOLIO"] if "PORTFOLIO" in available_r2_tickers else available_r2_tickers[:1]
        selected_r2_tickers = st.multiselect(
            "Select Tickers for R²:", options=available_r2_tickers, default=default_r2
        )

        df_r2_filtered = df_r2[df_r2["Ticker"].isin(selected_r2_tickers)]

        if df_r2_filtered.empty:
            st.warning("No R² data for selected tickers.")
        else:
            fig = px.line(
                df_r2_filtered,
                x="Date", y="R2", color="Ticker",
                title="Rolling R²"
            )
            fig.update_layout(height=700, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No R² rolling file found.")


# === Tab 5: Themes & Proxies ===
with tabs[4]:
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

# === Tab 6: Reconstructed Prices ===
with tabs[5]:
    st.subheader("Reconstructed Prices Viewer")

    # Load required files
    recon_path   = project_root / paths["recon_prices_output"]
    weights_path = project_root / paths["portfolio_weights"]
    yahoo_path   = project_root / paths["raw_prices_output"]

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
            from sklearn.metrics import mean_absolute_error, r2_score

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
                    line=dict(color='#5dade2')
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

            # === Validation Table ===
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

            # === Source Summary Table ===
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