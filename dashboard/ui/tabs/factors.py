import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path

def factors_tab(project_root, paths):
    st.subheader("Latest Factor Exposures")
    weights_path = project_root / paths["portfolio_weights"]
    df_weights = pd.read_csv(weights_path)
    valid_tickers = set(df_weights[df_weights["Weight"] > 0]["Ticker"].str.upper())
    valid_tickers.update(["^NDX", "^SPX", "PORTFOLIO"])
    
    # Get factor exposures from session state (cached data)
    df_expo = st.session_state.get("factor_exposures", pd.DataFrame())
    if not df_expo.empty:
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
        st.info("No factor exposure data found. Please run the factor exposure pipeline.")
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
    st.subheader("Rolling Factor Exposures")
    # Get rolling factor data from session state (cached data)
    df_roll_factors = st.session_state.get("factor_rolling", pd.DataFrame())
    if not df_roll_factors.empty:
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
        st.info("No rolling factor exposure data found. Please run the factor exposure pipeline.")
    st.subheader("Rolling R2")
    # Get R2 data from session state (cached data)
    df_r2 = st.session_state.get("r2_rolling", pd.DataFrame())
    if not df_r2.empty:
        df_r2["Date"] = pd.to_datetime(df_r2["Date"])
        df_r2["Ticker"] = df_r2["Ticker"].str.upper()
        df_r2 = df_r2[df_r2["Ticker"].isin(valid_tickers)]
        available_r2_tickers = sorted(df_r2["Ticker"].unique())
        default_r2 = ["PORTFOLIO"] if "PORTFOLIO" in available_r2_tickers else available_r2_tickers[:1]
        selected_r2_tickers = st.multiselect(
            "Select Tickers for R2:", options=available_r2_tickers, default=default_r2
        )
        df_r2_filtered = df_r2[df_r2["Ticker"].isin(selected_r2_tickers)]
        if df_r2_filtered.empty:
            st.warning("No R2 data for selected tickers.")
        else:
            fig = px.line(
                df_r2_filtered,
                x="Date", y="R2", color="Ticker",
                title="Rolling R2"
            )
            fig.update_layout(height=700, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No R2 rolling data found. Please run the factor exposure pipeline.") 