import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def realized_tab(df_realized, df_roll, df_corr, df_vol):
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
                    *[{"selector": f"td.col{i}", "props": [("min-width", f"{w}px")]} for i, w in enumerate(col_widths.values())]
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