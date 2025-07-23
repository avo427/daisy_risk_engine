import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import numpy as np

def sizing_tab(project_root, paths):
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
                    *[{"selector": f"td.col{i}", "props": [("min-width", f"{w}px")]} for i, w in enumerate(col_widths_sizing.values())]
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
        st.info("No volatility-based sizing output found.") 