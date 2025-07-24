import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

def stress_test_tab(project_root, paths):
    """Render the stress testing tab."""
    
    # Check if stress test results exist
    stress_test_file = project_root / "data" / "stress_test_results.json"
    
    if not stress_test_file.exists():
        st.info("No stress test results found. Run the stress test pipeline from the sidebar to generate results.")
        return
    
    # Load and display results
    try:
        with open(stress_test_file, 'r') as f:
            results = json.load(f)
        
        display_comprehensive_results(results)
        
    except Exception as e:
        st.error(f"Error loading stress test results: {str(e)}")
        st.info("Run the stress test pipeline from the sidebar to generate new results.")

def display_comprehensive_results(results):
    """Display comprehensive stress test results with charts."""
    
    # Data availability summary
    data_availability = results.get("data_availability", {})
    available_period = data_availability.get("available_period", ("Unknown", "Unknown"))
    scenarios_included = data_availability.get("scenarios_included", [])
    scenarios_excluded = data_availability.get("scenarios_excluded", [])
    

    
    # Regime analysis with proper visualization - MOVED TO TOP
    regime_analysis = results.get("regime_analysis", {})
    if regime_analysis and "error" not in regime_analysis:
        st.subheader("Market Regime Analysis")
        
        # Current regime
        current_regime = regime_analysis.get("current_regime", "Unknown")
        st.metric("Current Market Regime", current_regime.title())
        
        # Regime indicators
        regime_indicators = regime_analysis.get("regime_indicators", {})
        if regime_indicators:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volatility = regime_indicators.get("volatility", 0)
                st.metric("Volatility", f"{volatility:.2%}")
            
            with col2:
                correlation = regime_indicators.get("correlation", 0)
                st.metric("Correlation", f"{correlation:.3f}")
            
            with col3:
                momentum = regime_indicators.get("momentum", 0)
                st.metric("Momentum", f"{momentum:.2%}")
            
            # Determine regime based on triangle size (sum of normalized values)
            triangle_size = volatility*100 + correlation*100 + momentum*100
            
            # Define regime thresholds and descriptions
            if triangle_size < 60:  # Small triangle
                regime_name = "Normal Regime"
                regime_description = "Stable market conditions with good diversification"
            elif triangle_size < 90:  # Medium triangle
                regime_name = "Stress Regime"
                regime_description = "Moderate volatility with some correlation breakdown"
            else:  # Large triangle
                regime_name = "Crisis Regime"
                regime_description = "Extreme stress with diversification broken down"
            
            # Regime indicators radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[volatility*100, correlation*100, momentum*100],
                theta=['Volatility', 'Correlation', 'Momentum'],
                fill='toself',
                name='Current Regime',
                hovertemplate="<b>Current Regime:</b> " + regime_name + "<br>" +
                            "<b>Regime Meaning:</b> " + regime_description + "<br>" +
                            "<extra></extra>"
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 40]  # Further reduced range to increase triangle area by 25%
                    )),
                showlegend=False,
                title="Market Regime Indicators"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        

    
    # Data coverage summary
    st.subheader("Stress Scenario Tests")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scenarios Analyzed", len(scenarios_included))
    with col2:
        st.metric("Scenarios Excluded", len(scenarios_excluded))
    
    if scenarios_excluded:
        with st.expander("Excluded Scenarios"):
            for scenario in scenarios_excluded:
                # Extract date ranges from the scenario description
                if "outside data range" in scenario:
                    # Format: "scenario_name (period yyyy-mm-dd to yyyy-mm-dd outside data range yyyy-mm-dd to yyyy-mm-dd)"
                    parts = scenario.split(" (period ")
                    if len(parts) > 1:
                        scenario_name = parts[0].replace("_", " ")  # Remove underscores
                        date_part = parts[1].replace(")", "")
                        
                        # Clean up date format to remove time component
                        # Split by "outside data range" to get the two date ranges
                        if "outside data range" in date_part:
                            period_part, range_part = date_part.split(" outside data range ")
                            # Clean up each date range
                            period_clean = period_part.replace(" 00:00:00", "").replace(" 00:00", "")
                            range_clean = range_part.replace(" 00:00:00", "").replace(" 00:00", "")
                            clean_date_part = f"{period_clean} outside data range {range_clean}"
                        else:
                            clean_date_part = date_part.replace(" 00:00:00", "").replace(" 00:00", "")
                        
                        st.write(f"• {scenario_name} ({clean_date_part})")
                    else:
                        scenario_name = scenario.replace("_", " ")
                        st.write(f"• {scenario_name}")
                else:
                    scenario_name = scenario.replace("_", " ")
                    st.write(f"• {scenario_name}")
    
    # Historical scenarios with charts
    historical_scenarios = results.get("historical_scenarios", {})
    if historical_scenarios:
        st.subheader("Historical Scenarios")
        
        # Create summary table with scenario start dates for sorting
        scenario_data = []
        
        for name, scenario in historical_scenarios.items():
            if "error" not in scenario:
                portfolio_metrics = scenario.get("portfolio_metrics", {})
                
                # Get scenario start date for sorting
                start_date = ""
                if "period" in scenario:
                    period = scenario.get("period", [])
                    if isinstance(period, list) and len(period) > 0:
                        start_date = period[0]
                    elif isinstance(period, tuple) and len(period) > 0:
                        start_date = period[0]
                
                scenario_data.append({
                    "Scenario": name,
                    "Start Date": start_date,
                    "Total Return": portfolio_metrics.get('total_return', 'N/A'),
                    "Annual Volatility": portfolio_metrics.get('annual_volatility', 'N/A'),
                    "Max Drawdown": portfolio_metrics.get('max_drawdown', 'N/A'),
                    "Sharpe Ratio": portfolio_metrics.get('sharpe_ratio', 'N/A')
                })
        
        if scenario_data:
            # Sort by start date
            df_scenarios = pd.DataFrame(scenario_data)
            
            # Create scenario descriptions mapping
            scenario_descriptions = {}
            for name, scenario in historical_scenarios.items():
                if "error" not in scenario:
                    description = scenario.get("description", "No description available")
                    scenario_descriptions[name] = description
            
            # Sort by start date
            df_scenarios = df_scenarios.sort_values("Start Date")
            
            # Create a categorical order for plotting based on sorted dates
            scenario_order = df_scenarios["Scenario"].tolist()
            
            # Filter out N/A values for plotting
            plot_data = df_scenarios[df_scenarios["Total Return"] != 'N/A'].copy()
            
            if not plot_data.empty:
                # Clean scenario names (remove underscores) and add descriptions for tooltips
                plot_data["Scenario_Clean"] = plot_data["Scenario"].str.replace("_", " ")
                plot_data["Description"] = plot_data["Scenario"].apply(lambda x: scenario_descriptions.get(x, "No description available"))
                
                # Max Drawdown chart (full width on top)
                fig_dd = px.bar(
                    plot_data,
                    x="Scenario_Clean",
                    y="Max Drawdown",
                    color="Max Drawdown",
                    color_continuous_scale="Reds_r",  # Reversed gradient
                    text=plot_data["Max Drawdown"].apply(lambda x: f"{x:.1%}")
                )
                fig_dd.update_traces(hovertemplate="%{customdata}<extra></extra>", customdata=plot_data["Description"])
                fig_dd.update_xaxes(categoryorder='array', categoryarray=[s.replace("_", " ") for s in scenario_order])
                fig_dd.update_traces(textposition='outside')
                fig_dd.update_layout(
                    height=600,  # Increased height to avoid clipping
                    yaxis=dict(tickformat=".1%"),
                    xaxis_title="",  # Remove x-axis title
                    showlegend=False,
                    coloraxis=dict(
                        colorbar=dict(
                            tickformat=".1%",
                            title="Max Drawdown (%)"
                        )
                    )
                )
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Return and Sharpe Ratio charts (side by side below)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Return chart
                    fig = px.bar(
                        plot_data, 
                        x="Scenario_Clean", 
                        y="Total Return",
                        color="Total Return",
                        color_continuous_scale="RdYlGn",
                        text=plot_data["Total Return"].apply(lambda x: f"{x:.1%}")
                    )
                    fig.update_traces(hovertemplate="%{customdata}<extra></extra>", customdata=plot_data["Description"])
                    fig.update_xaxes(categoryorder='array', categoryarray=[s.replace("_", " ") for s in scenario_order])
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        yaxis_title="Return (%)",
                        height=600,  # Increased height to avoid clipping
                        yaxis=dict(tickformat=".1%"),
                        xaxis_title="",  # Remove x-axis title
                        showlegend=False,
                        coloraxis=dict(
                            colorbar=dict(
                                tickformat=".1%",
                                title="Return (%)"
                            )
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sharpe Ratio chart
                    fig_sharpe = px.bar(
                        plot_data,
                        x="Scenario_Clean",
                        y="Sharpe Ratio",
                        color="Sharpe Ratio",
                        color_continuous_scale="RdYlGn",
                        text=plot_data["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
                    )
                    fig_sharpe.update_traces(hovertemplate="%{customdata}<extra></extra>", customdata=plot_data["Description"])
                    fig_sharpe.update_xaxes(categoryorder='array', categoryarray=[s.replace("_", " ") for s in scenario_order])
                    fig_sharpe.update_traces(textposition='outside')
                    fig_sharpe.update_layout(
                        height=600,  # Increased height to avoid clipping
                        yaxis=dict(tickformat=".2f"),
                        xaxis_title="",  # Remove x-axis title
                        showlegend=False,
                        coloraxis=dict(
                            colorbar=dict(
                                tickformat=".2f",
                                title="Sharpe Ratio"
                            )
                        )
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Display summary table
            st.subheader("Historical Summary")
            
            # Clean scenario names and format data for display
            df_display = df_scenarios.copy()
            df_display["Scenario"] = df_display["Scenario"].str.replace("_", " ")
            
            # Format columns for display
            def format_percentage(value):
                if isinstance(value, (int, float)) and value != 'N/A':
                    return f"{value:.1%}"
                return value
            
            def format_decimal(value):
                if isinstance(value, (int, float)) and value != 'N/A':
                    return f"{value:.1f}"
                return value
            
            df_display["Total Return"] = df_display["Total Return"].apply(format_percentage)
            df_display["Annual Volatility"] = df_display["Annual Volatility"].apply(format_percentage)
            df_display["Max Drawdown"] = df_display["Max Drawdown"].apply(format_percentage)
            df_display["Sharpe Ratio"] = df_display["Sharpe Ratio"].apply(format_decimal)
            
            # Drop the Start Date column for display
            df_display = df_display.drop("Start Date", axis=1)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Monte Carlo results with visualizations
    monte_carlo = results.get("monte_carlo", {})
    if monte_carlo and "error" not in monte_carlo:
        st.subheader("Monte Carlo Results")
        
        simulation_stats = monte_carlo.get("simulation_statistics", {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return", f"{simulation_stats.get('mean', 0):.2%}")
        with col2:
            st.metric("Volatility", f"{simulation_stats.get('std', 0):.2%}")
        with col3:
            st.metric("Skewness", f"{simulation_stats.get('skewness', 0):.2f}")
        with col4:
            st.metric("Kurtosis", f"{simulation_stats.get('kurtosis', 0):.2f}")
        
        # VaR and CVaR visualization
        var_cvar = monte_carlo.get("var_cvar", {})
        if var_cvar:
            st.subheader("Risk Metrics (VaR/CVaR)")
            
            # Create VaR/CVaR chart
            var_data = []
            for key, value in var_cvar.items():
                if isinstance(value, (int, float)):
                    var_data.append({
                        "Metric": key,
                        "Value": value,
                        "Type": "VaR" if "VaR" in key else "CVaR"
                    })
            
            if var_data:
                df_var = pd.DataFrame(var_data)
                
                # Clean up metric names for display
                df_var["Metric_Display"] = df_var["Metric"].apply(lambda x: {
                    "VaR_95": "VaR (95%)",
                    "CVaR_95": "CVaR (95%)", 
                    "VaR_99": "VaR (99%)",
                    "CVaR_99": "CVaR (99%)"
                }.get(x, x))
                
                # Create custom tooltips with explanations
                tooltips = []
                for _, row in df_var.iterrows():
                    metric = row["Metric"]
                    value = row["Value"]
                    if "VaR_95" in metric:
                        tooltip = f"95% confidence that portfolio losses won't exceed {value:.1%}"
                    elif "CVaR_95" in metric:
                        tooltip = f"Average loss when the worst 5% of scenarios occur: {value:.1%}"
                    elif "VaR_99" in metric:
                        tooltip = f"99% confidence that portfolio losses won't exceed {value:.1%}"
                    elif "CVaR_99" in metric:
                        tooltip = f"Average loss when the worst 1% of scenarios occur: {value:.1%}"
                    else:
                        tooltip = f"{row['Metric_Display']}: {value:.1%}"
                    tooltips.append(tooltip)
                
                fig_var = px.bar(
                    df_var,
                    x="Metric_Display",
                    y="Value",
                    title="Value at Risk (VaR) and Conditional VaR (CVaR)",
                    text=df_var["Value"].apply(lambda x: f"{x:.2%}")
                )
                fig_var.update_traces(
                    textposition='outside', 
                    marker_color='#FF6B35',  # Reddish orange color
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=tooltips
                )
                fig_var.update_layout(
                    xaxis_title="Risk Metric",
                    yaxis_title="Loss (%)",
                    height=600,  # Increased height to avoid clipping
                    yaxis=dict(tickformat=".1%"),
                    showlegend=False
                )
                st.plotly_chart(fig_var, use_container_width=True)
    
    # Factor stress tests with visualizations
    factor_stress = results.get("factor_stress", {})
    if factor_stress:
        st.subheader("Factor Stress Tests")
        
        factor_data = []
        for scenario_name, scenario_results in factor_stress.items():
            if "error" not in scenario_results:
                # Extract factor impacts if available
                factor_impact = scenario_results.get("factor_impact", {})
                portfolio_impact = scenario_results.get("portfolio_impact", 0)
                
                factor_data.append({
                    "Scenario": scenario_name,
                    "Portfolio Impact": portfolio_impact,
                    "Factor Count": len(factor_impact)
                })
        
        if factor_data:
            df_factors = pd.DataFrame(factor_data)
            
            # Factor stress impact chart
            # Clean scenario names for display
            df_factors["Scenario_Clean"] = df_factors["Scenario"].str.replace("_", " ")
            
            # Create scenario descriptions for tooltips
            scenario_descriptions = {
                "Tech_Sector_Crash": "Technology sector crash: AI and market factors decline by 25%",
                "Interest_Rate_Spike": "Interest rates spike by 50%, affecting rate-sensitive assets",
                "Volatility_Explosion": "VIX volatility index explodes by 200%, causing market stress",
                "Momentum_Reversal": "Momentum factor reverses by 30%, hurting trend-following strategies",
                "Liquidity_Crisis": "Liquidity crisis: small caps and low vol stocks decline by 20%"
            }
            
            df_factors["Description"] = df_factors["Scenario"].apply(lambda x: scenario_descriptions.get(x, "No description available"))
            
            fig_factors = px.bar(
                df_factors,
                x="Scenario_Clean",
                y="Portfolio Impact",
                title="Factor Stress Test Portfolio Impacts",
                color="Portfolio Impact",
                color_continuous_scale="RdYlGn",
                text=df_factors["Portfolio Impact"].apply(lambda x: f"{x:.1%}")
            )
            fig_factors.update_traces(
                textposition='outside',
                hovertemplate="%{customdata}<extra></extra>",
                customdata=df_factors["Description"]
            )
            fig_factors.update_layout(
                xaxis_title="",
                yaxis_title="Portfolio Impact (%)",
                height=600,
                yaxis=dict(tickformat=".1%"),
                coloraxis=dict(
                    colorbar=dict(
                        tickformat=".1%",
                        title="Portfolio Impact (%)"
                    )
                )
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        # Detailed factor results in expandable sections
        for scenario_name, scenario_results in factor_stress.items():
            if "error" not in scenario_results:
                with st.expander(f"Factor Stress: {scenario_name.replace('_', ' ')}"):
                    # Display factor impacts if available
                    factor_impact = scenario_results.get("factor_impact", {})
                    if factor_impact:
                        st.subheader("Factor Impacts")
                        impact_data = [{"Factor": k, "Impact": v} for k, v in factor_impact.items()]
                        df_impact = pd.DataFrame(impact_data)
                        
                        # Factor impact chart
                        fig_impact = px.bar(
                            df_impact,
                            x="Factor",
                            y="Impact",
                            title=f"Factor Impacts - {scenario_name.replace('_', ' ')}",
                            color="Impact",
                            color_continuous_scale="RdYlGn",
                            text=df_impact["Impact"].apply(lambda x: f"{x:.1%}")
                        )
                        fig_impact.update_traces(
                            textposition='outside',
                            hovertemplate=None,
                            hoverinfo='skip'
                        )
                        
                        # Calculate dynamic Y-axis range to prevent clipping
                        min_impact = df_impact["Impact"].min()
                        max_impact = df_impact["Impact"].max()
                        y_range = max_impact - min_impact
                        y_padding = y_range * 0.1  # 10% padding
                        
                        fig_impact.update_layout(
                            yaxis=dict(
                                tickformat=".1%",
                                range=[min_impact - y_padding, max_impact + y_padding]
                            ),
                            yaxis_title="Impact (%)",
                            height=600
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                    
                    # Display factor composition in a consolidated table
                    if factor_impact:
                        st.subheader("Factor Composition")
                        impact_table_data = []
                        
                        # Add individual factor rows
                        for factor, impact in factor_impact.items():
                            impact_table_data.append({
                                "Factor": factor,
                                "Impact": f"{impact:.2%}",
                                "Shock Applied": f"{scenario_results.get('shocks', {}).get(factor, 0):.1%}"
                            })
                        
                        # Add portfolio total row
                        portfolio_impact = scenario_results.get('portfolio_impact', 0)
                        impact_table_data.append({
                            "Factor": "PORTFOLIO",
                            "Impact": f"{portfolio_impact:.2%}",
                            "Shock Applied": ""
                        })
                        
                        df_impact_table = pd.DataFrame(impact_table_data)
                        st.dataframe(df_impact_table, use_container_width=True, hide_index=True)
    
 