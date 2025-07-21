import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def forecast_tab(df_forecast, df_fore_roll, project_root, paths):
    st.subheader("Forecast Metrics")
    if df_forecast.empty:
        st.info("No forecast metrics available.")
    else:
        df_forecast = df_forecast.copy()
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