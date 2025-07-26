import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from utils.config import get_target_volatility

def sizing_tab(project_root, paths):
    st.subheader("Risk Parity Sizing")
    
    # Define sizing_path first
    sizing_path = project_root / paths["vol_sizing_output"]
    
    # Load config to get target volatility and cash tickers
    config_path = project_root / "config.yaml"
    target_vol = 0.20
    cash_tickers = ["SGOV", "BIL", "SPRXX", "VMFXX", "SPAXX"]
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                target_vol = get_target_volatility(config)
                cash_tickers = config["user_settings"].get("cash_tickers", ["SGOV", "BIL", "SPRXX", "VMFXX", "SPAXX"])
        except yaml.YAMLError as e:
            st.error(f"Error parsing config file: {e}")
            st.info("Using default values for target volatility and cash tickers.")
        except FileNotFoundError as e:
            st.error(f"Config file not found: {e}")
            st.info("Using default values for target volatility and cash tickers.")
        except KeyError as e:
            st.warning(f"Missing config key: {e}")
            st.info("Using default values for missing configuration.")
        except Exception as e:
            st.error(f"Unexpected error loading config: {e}")
            st.info("Using default values for target volatility and cash tickers.")
    
    st.caption(f"Target Volatility: {target_vol:.1%}")
    
    if sizing_path.exists():
        try:
            df_sizing = pd.read_csv(sizing_path)
            
            # Validate required columns exist
            required_columns = ["Ticker", "Model", "ForecastVol", "RiskParityWeight"]
            missing_columns = [col for col in required_columns if col not in df_sizing.columns]
            if missing_columns:
                st.error(f"Missing required columns in sizing data: {', '.join(missing_columns)}")
                return
            
            # Find which cash positions are actually in the portfolio
            if 'Ticker' in df_sizing.columns:
                portfolio_cash_tickers = [t for t in cash_tickers if t in df_sizing['Ticker'].values]
                if portfolio_cash_tickers:
                    st.info(f"Cash positions excluded from risk parity: {', '.join(portfolio_cash_tickers)}")
            
            model_options = df_sizing["Model"].unique().tolist()
            if not model_options:
                st.error("No volatility models found in sizing data")
                return
                
            selected_model = st.selectbox("Select Volatility Forecast Model", model_options)
            filtered_df = df_sizing[df_sizing["Model"] == selected_model].copy()
            
            if filtered_df.empty:
                st.error(f"No data found for model: {selected_model}")
                return
                
        except pd.errors.EmptyDataError:
            st.error("Sizing data file is empty")
            return
        except pd.errors.ParserError as e:
            st.error(f"Error parsing sizing data file: {e}")
            return
        except FileNotFoundError:
            st.error("Sizing data file not found. Please run the risk analysis pipeline first.")
            return
        except Exception as e:
            st.error(f"Unexpected error loading sizing data: {e}")
            return
            
        filtered_df.drop(columns=["Model"], inplace=True)
        rename_map = {
            "ForecastVol": "Forecast Volatility",
            "BaseWeight": "Current Weight",
            "RiskParityWeight": "Risk Parity Weight",
            "CurrentDollar": "Current MV",
            "TargetDollar": "Target MV",
            "DeltaDollar": "(+/-)",
            "CurrentRiskContribution": "Current Risk Contrib.",
            "TargetRiskContribution": "Target Risk Contrib.",
            "CorrelationFactor": "Portfolio Correlation",
            "Price": "Last Price",
            "DeltaShares": "Shares Delta"
        }
        filtered_df.rename(columns=rename_map, inplace=True)
        
        # Reorder columns to put risk contribution after (+/-)
        column_order = [
            "Ticker", "Forecast Volatility", "Current Weight", "Risk Parity Weight",
            "Current MV", "Target MV", "(+/-)", "Current Risk Contrib.", "Target Risk Contrib.", "Portfolio Correlation",
            "Last Price", "Shares Delta"
        ]
        filtered_df = filtered_df.reindex(columns=column_order)
        
        # Efficient data type conversion with proper error handling
        percent_cols = ["Forecast Volatility", "Current Weight", "Risk Parity Weight", "Current Risk Contrib.", "Target Risk Contrib."]
        decimal_cols = ["Portfolio Correlation"]
        dollar_cols = ["Current MV", "Target MV", "(+/-)", "Last Price"]
        share_cols = ["Shares Delta"]
        
        for col in percent_cols + decimal_cols + dollar_cols + share_cols:
            if col in filtered_df.columns:
                try:
                    # Handle different data types more efficiently
                    if filtered_df[col].dtype == 'object':
                        # Convert string columns with formatting
                        filtered_df[col] = (
                            filtered_df[col]
                            .astype(str)
                            .str.replace('%', '', regex=False)
                            .str.replace('$', '', regex=False)
                            .str.replace(',', '', regex=False)
                            .replace(['', 'nan', 'None'], np.nan)
                        )
                    
                    # Convert to numeric with error handling
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    
                except Exception as e:
                    st.warning(f"Error converting column {col}: {e}")
                    # Keep original data if conversion fails
                    continue
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
                    "Risk Parity Weight": "{:.2%}",
                    "Current Risk Contrib.": "{:.2%}",
                    "Target Risk Contrib.": "{:.2%}",
                    "Portfolio Correlation": "{:.2f}",
                    "Current MV": "${:,.0f}",
                    "Target MV": "${:,.0f}",
                    "(+/-)": "${:,.0f}",
                    "Last Price": "${:,.2f}",
                    "Shares Delta": format_shares
                }),
            use_container_width=True,
            height=table_height
        )
        st.subheader("Risk Parity - Equal Risk Contribution")
        volcontrib_df = filtered_df.copy()
        # Use the actual target risk contributions calculated in the forecast model
        # These already account for correlation and should be equal for all positions
        volcontrib_df["Expected Risk Contribution"] = volcontrib_df["Target Risk Contrib."]
        total_contrib = volcontrib_df["Expected Risk Contribution"].sum()
        if total_contrib > 0:
            volcontrib_df["Risk Contribution %"] = (
                volcontrib_df["Expected Risk Contribution"] / total_contrib * 100
            )
        else:
            volcontrib_df["Risk Contribution %"] = 0
        
        fig = px.pie(
            volcontrib_df,
            names="Ticker",
            values="Risk Contribution %",
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
    else:
        st.info("No volatility-based sizing output found.") 