import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import risk scorer with error handling
try:
    from utils.risk_scorer import PortfolioRiskScorer, get_risk_score_color, get_risk_level_description
    RISK_SCORER_AVAILABLE = True
except ImportError:
    RISK_SCORER_AVAILABLE = False
    print("Warning: Risk scorer not available")

def summary_tab(project_root, paths, config):
    """
    Portfolio Summary Dashboard - Highlights potential issues across all metrics
    """
    
    # Load all available data
    data = load_summary_data(project_root, paths)
    
    if not data['has_data']:
        st.warning("No portfolio data available. Please run the risk analysis pipeline first.")
        return
    
    # Create summary metrics and alerts
    summary_metrics, alerts = generate_portfolio_summary(data, config)
    
    # Calculate risk score
    if RISK_SCORER_AVAILABLE:
        risk_scorer = PortfolioRiskScorer(config)
        risk_breakdown = risk_scorer.calculate_overall_risk_score(data)
        
        # Display risk score
        display_risk_score(risk_breakdown)
    else:
        st.warning("Risk scoring not available. Basic summary metrics will be displayed.")
        risk_breakdown = None
    
    # Display summary metrics (Portfolio Overview)
    if risk_breakdown:
        summary_metrics['risk_breakdown'] = risk_breakdown
    display_summary_metrics(summary_metrics, data)
    
    # Display portfolio positions
    display_portfolio_positions(data)
    
    # Display risk contribution breakdown
    display_risk_contribution_breakdown(summary_metrics)
    
    # Display risk alerts
    display_risk_alerts(alerts)
    
    # Display recommendations
    display_recommendations(alerts, summary_metrics)

def load_summary_data(project_root, paths):
    """Load all available data for summary analysis"""
    data = {
        'has_data': False,
        'realized': pd.DataFrame(),
        'forecast': pd.DataFrame(),
        'forecast_risk_contributions': pd.DataFrame(),
        'weights': pd.DataFrame(),
        'correlation': pd.DataFrame(),
        'vol_contribution': pd.DataFrame(),
        'factor_exposures': pd.DataFrame(),
        'stress_test': None,
        'sizing': pd.DataFrame()
    }
    
    try:
        # Load realized metrics
        realized_path = project_root / paths.get("realized_output", "data/realized_metrics.csv")
        if realized_path.exists():
            data['realized'] = pd.read_csv(realized_path)
            data['has_data'] = True
        
        # Load forecast metrics
        forecast_path = project_root / paths.get("forecast_output", "data/forecast_metrics.csv")
        if forecast_path.exists():
            data['forecast'] = pd.read_csv(forecast_path)
        
        # Load forecast risk contributions
        forecast_risk_path = project_root / paths.get("forecast_risk_contributions", "data/forecast_risk_contributions.csv")
        if forecast_risk_path.exists():
            data['forecast_risk_contributions'] = pd.read_csv(forecast_risk_path)
        
        # Load portfolio weights
        weights_path = project_root / paths.get("portfolio_weights", "data/portfolio_weights.csv")
        if weights_path.exists():
            data['weights'] = pd.read_csv(weights_path)
        
        # Load correlation matrix
        corr_path = project_root / paths.get("correlation_matrix", "data/correlation_matrix.csv")
        if corr_path.exists():
            data['correlation'] = pd.read_csv(corr_path, index_col=0)
        
        # Load volatility contribution
        vol_path = project_root / paths.get("vol_contribution", "data/vol_contribution.csv")
        if vol_path.exists():
            data['vol_contribution'] = pd.read_csv(vol_path)
        
        # Load factor exposures
        factor_path = project_root / paths.get("factor_exposures", "data/factor_exposures.csv")
        if factor_path.exists():
            data['factor_exposures'] = pd.read_csv(factor_path)
        
        # Load stress test results
        stress_path = project_root / paths.get("stress_test_output", "data/stress_test_results.json")
        if stress_path.exists():
            try:
                with open(stress_path, 'r') as f:
                    data['stress_test'] = json.load(f)
            except:
                pass
        
        # Load sizing data
        sizing_path = project_root / paths.get("vol_sizing_output", "data/vol_sizing_weights_long.csv")
        if sizing_path.exists():
            data['sizing'] = pd.read_csv(sizing_path)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    return data

def generate_portfolio_summary(data, config):
    """Generate summary metrics and risk alerts"""
    summary = {}
    alerts = []
    
    # Portfolio concentration analysis
    if not data['weights'].empty:
        weights = data['weights']
        active_weights = weights[weights['Weight'] > 0]
        
        summary['total_positions'] = len(active_weights)
        
        # Calculate portfolio value from MarketValue column
        if 'MarketValue' in active_weights.columns:
            summary['portfolio_value'] = active_weights['MarketValue'].sum()
        
        # Get largest position with ticker
        if len(active_weights) > 0:
            largest_idx = active_weights['Weight'].idxmax()
            largest_weight = active_weights.loc[largest_idx, 'Weight']
            largest_ticker = active_weights.loc[largest_idx, 'Ticker']
            summary['largest_position'] = largest_weight
            summary['largest_position_ticker'] = largest_ticker
        
        summary['top_5_concentration'] = active_weights['Weight'].nlargest(5).sum() if len(active_weights) >= 5 else active_weights['Weight'].sum()
        summary['top_3_concentration'] = active_weights['Weight'].nlargest(3).sum() if len(active_weights) >= 3 else active_weights['Weight'].sum()
        
        # Concentration alerts
        if summary['largest_position'] > 0.15:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Concentration Risk',
                'message': f"Largest position ({summary['largest_position']:.1%}) exceeds 15% limit",
                'ticker': active_weights.loc[active_weights['Weight'].idxmax(), 'Ticker'] if len(active_weights) > 0 else 'Unknown'
            })
        
        if summary['top_3_concentration'] > 0.50:
            alerts.append({
                'type': 'HIGH',
                'category': 'Concentration Risk',
                'message': f"Top 3 positions ({summary['top_3_concentration']:.1%}) exceed 50% of portfolio",
                'ticker': 'Multiple'
            })
    
    # Risk metrics analysis
    if not data['realized'].empty:
        portfolio_metrics = data['realized'][data['realized']['Ticker'] == 'PORTFOLIO']
        if not portfolio_metrics.empty:
            portfolio = portfolio_metrics.iloc[0]
            
            # Parse percentage columns
            for col in ['Ann. Return', 'Ann. Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']:
                if col in portfolio.index:
                    try:
                        value_str = str(portfolio[col])
                        if '%' in value_str:
                            summary[col.lower().replace(' ', '_').replace('(', '').replace(')', '')] = float(value_str.replace('%', '')) / 100
                    except:
                        pass
            
            # Parse CVaR specifically
            if 'CVaR (95%)' in portfolio.index:
                try:
                    value_str = str(portfolio['CVaR (95%)'])
                    if '%' in value_str:
                        summary['cvar_95'] = float(value_str.replace('%', '')) / 100
                except:
                    pass
            
            # Parse decimal columns
            for col in ['Sharpe', 'Sortino', 'Beta (NDX)', 'Beta (SPX)']:
                if col in portfolio.index:
                    try:
                        summary[col.lower().replace(' ', '_').replace('(', '').replace(')', '')] = float(portfolio[col])
                    except:
                        pass
            
            # Risk alerts
            if 'ann_volatility' in summary and summary['ann_volatility'] > 0.30:
                alerts.append({
                    'type': 'HIGH',
                    'category': 'Volatility Risk',
                    'message': f"Portfolio volatility ({summary['ann_volatility']:.1%}) is very high",
                    'ticker': 'PORTFOLIO'
                })
            
            if 'max_drawdown' in summary and summary['max_drawdown'] < -0.20:
                alerts.append({
                    'type': 'MEDIUM',
                    'category': 'Drawdown Risk',
                    'message': f"Maximum drawdown ({summary['max_drawdown']:.1%}) is significant",
                    'ticker': 'PORTFOLIO'
                })
            
            if 'sharpe' in summary and summary['sharpe'] < 0.5:
                alerts.append({
                    'type': 'MEDIUM',
                    'category': 'Risk-Adjusted Returns',
                    'message': f"Sharpe ratio ({summary['sharpe']:.2f}) is below target",
                    'ticker': 'PORTFOLIO'
                })
            
            if 'beta_ndx' in summary and summary['beta_ndx'] > 1.5:
                alerts.append({
                    'type': 'MEDIUM',
                    'category': 'Market Risk',
                    'message': f"High beta to NDX ({summary['beta_ndx']:.2f}) - high market sensitivity",
                    'ticker': 'PORTFOLIO'
                })
    
    # Forecast risk analysis
    if not data['forecast'].empty:
        portfolio_forecast = data['forecast'][data['forecast']['Ticker'] == 'PORTFOLIO']
        if not portfolio_forecast.empty:
            forecast = portfolio_forecast.iloc[0]
            
            # Parse forecast metrics
            for col in ['EWMA (5D)', 'EWMA (20D)', 'Garch Volatility', 'E-Garch Volatility', 'VaR (95%)', 'CVaR (95%)', 'VaR ($)', 'CVaR ($)']:
                if col in forecast:
                    try:
                        value_str = str(forecast[col])
                        # Remove quotes first, then handle formatting
                        value_str = value_str.replace('"', '')
                        
                        if '%' in value_str:
                            # Handle percentage values
                            clean_value = value_str.replace('%', '')
                            # Create proper key names that match what display function expects
                            if col == 'E-Garch Volatility':
                                key = 'forecast_egarch_volatility'
                            elif col == 'CVaR (95%)':
                                key = 'forecast_cvar_95'
                            elif col == 'VaR (95%)':
                                key = 'forecast_var_95'
                            else:
                                key = f'forecast_{col.lower().replace(" ", "_").replace("(", "").replace(")", "")}'
                            summary[key] = float(clean_value) / 100
                        elif '$' in value_str:
                            # Handle dollar values - remove $ and commas
                            clean_value = value_str.replace('$', '').replace(',', '')
                            # Create proper key names that match what display function expects
                            if col == 'CVaR ($)':
                                key = 'forecast_cvar_dollar'
                            elif col == 'VaR ($)':
                                key = 'forecast_var_dollar'
                            else:
                                key = f'forecast_{col.lower().replace(" ", "_").replace("(", "").replace(")", "")}'
                            summary[key] = float(clean_value)
                    except Exception as e:
                        pass
            
            # Find top risk contributor from forecast risk contributions
            if not data['forecast_risk_contributions'].empty:
                # Get E-Garch Volatility model data (most recent/accurate)
                egarch_data = data['forecast_risk_contributions'][data['forecast_risk_contributions']['Model'] == 'E-Garch Volatility']
                if not egarch_data.empty:
                    # Find ticker with highest risk contribution percentage
                    top_risk_idx = egarch_data['Forecast_Risk_%'].idxmax()
                    top_risk_ticker = egarch_data.loc[top_risk_idx, 'Ticker']
                    top_risk_contribution = egarch_data.loc[top_risk_idx, 'Forecast_Risk_%']
                    summary['top_risk_contributor'] = top_risk_ticker
                    summary['top_risk_contribution'] = top_risk_contribution
            else:
                # Fallback to CVaR method if forecast risk contributions not available
                if not data['forecast'].empty:
                    individual_forecasts = data['forecast'][data['forecast']['Ticker'] != 'PORTFOLIO']
                    if not individual_forecasts.empty:
                        # Find the ticker with highest CVaR ($)
                        if 'CVaR ($)' in individual_forecasts.columns:
                            # Convert CVaR ($) to numeric, handling the quoted format
                            individual_forecasts['CVaR ($)'] = individual_forecasts['CVaR ($)'].astype(str).str.replace('$', '').str.replace(',', '').str.replace('"', '')
                            individual_forecasts['CVaR ($)'] = pd.to_numeric(individual_forecasts['CVaR ($)'], errors='coerce')
                            
                            top_risk_idx = individual_forecasts['CVaR ($)'].idxmax()
                            top_risk_ticker = individual_forecasts.loc[top_risk_idx, 'Ticker']
                            top_risk_cvar = individual_forecasts.loc[top_risk_idx, 'CVaR ($)']
                            summary['top_risk_contributor'] = top_risk_ticker
                            summary['top_risk_contributor_cvar'] = top_risk_cvar
            
            # Forecast alerts
            if 'forecast_ewma_5d' in summary and summary['forecast_ewma_5d'] > 0.40:
                alerts.append({
                    'type': 'HIGH',
                    'category': 'Forecast Risk',
                    'message': f"5-day forecast volatility ({summary['forecast_ewma_5d']:.1%}) is very high",
                    'ticker': 'PORTFOLIO'
                })
    
    # Factor exposure analysis
    if not data['factor_exposures'].empty:
        factor_data = data['factor_exposures']
        if 'Date' in factor_data.columns and 'Ticker' in factor_data.columns:
            latest_date = factor_data['Date'].max()
            portfolio_factors = factor_data[
                (factor_data['Date'] == latest_date) & 
                (factor_data['Ticker'] == 'PORTFOLIO')
            ]
            
            # Store top factor exposures for display
            if not portfolio_factors.empty:
                # Sort by absolute beta value and get top 3
                portfolio_factors_sorted = portfolio_factors.sort_values('Beta', key=abs, ascending=False)
                top_factors = portfolio_factors_sorted.head(3)['Factor'].tolist()
                summary['top_factors'] = top_factors
                
                # Store individual factor betas
                for _, row in portfolio_factors.iterrows():
                    factor = row['Factor']
                    beta = row['Beta']
                    summary[f'factor_{factor.lower()}'] = beta
            
            for _, row in portfolio_factors.iterrows():
                factor = row['Factor']
                beta = row['Beta']
                
                if abs(beta) > 0.5:
                    alerts.append({
                        'type': 'MEDIUM',
                        'category': 'Factor Exposure',
                        'message': f"High exposure to {factor} factor (beta: {beta:.2f})",
                        'ticker': 'PORTFOLIO'
                    })
    
    # Correlation analysis
    if not data['correlation'].empty:
        corr_matrix = data['correlation']
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.7:
                    high_corr_pairs.append({
                        'ticker1': corr_matrix.columns[i],
                        'ticker2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if len(high_corr_pairs) > 0:
            alerts.append({
                'type': 'MEDIUM',
                'category': 'Correlation Risk',
                'message': f"{len(high_corr_pairs)} pairs with correlation > 0.7",
                'ticker': 'Multiple',
                'details': high_corr_pairs[:3]  # Show top 3
            })
    
    # Stress test analysis
    if data['stress_test']:
        stress_data = data['stress_test']
        
        # Check worst-case scenarios
        if 'worst_case_scenarios' in stress_data:
            for scenario in stress_data['worst_case_scenarios']:
                if scenario.get('return', 0) < -0.25:
                    alerts.append({
                        'type': 'HIGH',
                        'category': 'Stress Test',
                        'message': f"Worst-case scenario: {scenario.get('return', 0):.1%} return",
                        'ticker': 'PORTFOLIO',
                        'details': scenario
                    })
    
    return summary, alerts

def display_risk_score(risk_breakdown):
    """Display overall risk score with visual indicators"""
    st.subheader("Portfolio Risk Score")
    
    overall_score = risk_breakdown['overall_score']
    risk_level = risk_breakdown['risk_level']
    
    # Create columns for score display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Risk score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score (%)", 'font': {'color': '#e2e8f0', 'size': 16}},
            delta = {'reference': 50, 'font': {'color': '#a0aec0'}},
            gauge = {
                'axis': {
                    'range': [None, 100],
                    'tickcolor': '#4a5568',
                    'tickfont': {'color': '#a0aec0', 'size': 12}
                },
                'bar': {'color': get_risk_score_color(overall_score)},
                'bgcolor': '#2d3748',
                'steps': [
                    {'range': [0, 30], 'color': "#68d391"},  # Subtle green
                    {'range': [30, 60], 'color': "#f6ad55"},  # Subtle orange
                    {'range': [60, 80], 'color': "#fc8181"},  # Subtle red
                    {'range': [80, 100], 'color': "#e53e3e"}  # Darker red
                ],
                'threshold': {
                    'line': {'color': "#e53e3e", 'width': 3},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level and description
        risk_colors = {
            "LOW": "green",
            "MEDIUM": "orange", 
            "HIGH": "red",
            "CRITICAL": "darkred"
        }
        
        st.markdown(f"""
        ### Risk Level: <span style='color: {risk_colors.get(risk_level, "black")}'>{risk_level}</span>
        
        **{get_risk_level_description(risk_level)}**
        
        **Overall Score:** {overall_score:.1%}
        """, unsafe_allow_html=True)
    
    with col3:
        # Quick stats
        risk_scores = risk_breakdown['risk_scores']
        
        # Find highest risk component
        max_risk = max(risk_scores.items(), key=lambda x: x[1])
        st.metric("Highest Risk", max_risk[0].replace('_', ' ').title(), f"{max_risk[1]:.1%}")
        
        # Count high risk components and create tooltip
        high_risk_components = [(name, score) for name, score in risk_scores.items() if score > 0.7]
        high_risk_count = len(high_risk_components)
        
        if high_risk_count > 0:
            # Create tooltip content
            tooltip_content = "**High Risk Components:**\n"
            for name, score in high_risk_components:
                tooltip_content += f"• {name.replace('_', ' ').title()}: {score:.1%}\n"
            
            # Display metric with tooltip
            st.metric("High Risk Components", high_risk_count, help=tooltip_content)
        else:
            st.metric("High Risk Components", high_risk_count)

def display_summary_metrics(summary, data):
    """Display key portfolio summary metrics in a 2x4 grid layout"""
    
    # Create custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #4a5568;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.3rem 0;
        color: #f7fafc;
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        color: #a0aec0;
        font-weight: 500;
    }
    .metric-change {
        font-size: 0.75rem;
        opacity: 0.7;
        color: #cbd5e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Portfolio Overview Section
    st.subheader("Portfolio Overview")
    
    # First row - Portfolio metrics from portfolio_weights.csv
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'portfolio_value' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${summary['portfolio_value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'total_positions' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Positions</div>
                <div class="metric-value">{summary['total_positions']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'largest_position' in summary and 'largest_position_ticker' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Largest Position</div>
                <div class="metric-value">{summary['largest_position']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'top_3_concentration' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Top 3 Concentration</div>
                <div class="metric-value">{summary['top_3_concentration']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row - Forecast risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'forecast_egarch_volatility' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">E-Garch Volatility</div>
                <div class="metric-value">{summary['forecast_egarch_volatility']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'forecast_cvar_95' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CVaR (95%)</div>
                <div class="metric-value">{summary['forecast_cvar_95']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'forecast_cvar_dollar' in summary:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CVaR ($)</div>
                <div class="metric-value">${summary['forecast_cvar_dollar']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'top_risk_contributor' in summary:
            risk_contribution_text = summary['top_risk_contributor']
            if 'top_risk_contribution' in summary:
                risk_contribution_text += f" ({summary['top_risk_contribution']:.1f}%)"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Top Risk Contributor</div>
                <div class="metric-value">{risk_contribution_text}</div>
            </div>
            """, unsafe_allow_html=True)
    


def display_risk_contribution_breakdown(summary):
    """Display risk contribution breakdown with pie chart and weights bar chart"""
    if 'risk_breakdown' in summary and summary['risk_breakdown']:
        st.markdown("---")
        st.subheader("Risk Scoring")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Create weights bar chart data
            weights_data = []
            for risk_type, weight in summary['risk_breakdown']['risk_weights'].items():
                weights_data.append({
                    'Risk Type': risk_type.replace('_', ' ').title(),
                    'Weight': weight
                })
            
            weights_df = pd.DataFrame(weights_data)
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            # Create horizontal bar chart for weights
            fig = px.bar(
                weights_df,
                x='Weight',
                y='Risk Type',
                orientation='h',
                title="Risk Score Weights",
                color='Weight',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            # Update x-axis to percentages with no decimal places
            fig.update_xaxes(tickformat='.0%')
            
            # Update hover template to show percentage without decimal places
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>" +
                             "Weight=%{x:.0%}<br>" +
                             "<extra></extra>"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create pie chart data
            risk_data = []
            for risk_type, score in summary['risk_breakdown']['risk_scores'].items():
                weight = summary['risk_breakdown']['risk_weights'][risk_type]
                weighted_score = score * weight
                risk_data.append({
                    'Risk Type': risk_type.replace('_', ' ').title(),
                    'Contribution': weighted_score
                })
            
            risk_df = pd.DataFrame(risk_data)
            risk_df = risk_df.sort_values('Contribution', ascending=False)
            
            # Create pie chart
            fig = px.pie(
                risk_df,
                names='Risk Type',
                values='Contribution',
                title="Risk Contribution by Component"
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            fig.update_traces(
                textposition='outside',
                textinfo='label+percent',
                pull=[0.1] * len(risk_df),  # Slight pull for better label visibility
                hovertemplate="<b>%{label}</b><br>" +
                             "Contribution=%{value:.0%}<br>" +
                             "<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

def display_portfolio_positions(data):
    """Display portfolio positions as a bar chart"""
    if data['weights'].empty:
        return
    
    weights = data['weights'][data['weights']['Weight'] > 0].copy()
    if len(weights) == 0:
        return
    
    # Sort by weight descending
    weights = weights.sort_values('Weight', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        weights, 
        x='Ticker', 
        y='Weight',
        title="Portfolio Positions",
        color='Weight',
        color_continuous_scale='RdYlGn_r'
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None
    )
    
    # Update y-axis to percentages
    fig.update_yaxes(tickformat='.0%')
    
    # Add percentage values above bars
    for i, weight in enumerate(weights['Weight']):
        fig.add_annotation(
            x=i,
            y=weight,
            text=f"{weight:.1%}",
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
    
    # Remove hover template
    fig.update_traces(hovertemplate=None, hoverinfo='skip')
    
    st.plotly_chart(fig, use_container_width=True)



def display_risk_alerts(alerts):
    """Display risk alerts with color coding"""
    st.subheader("Risk Alerts")
    
    if not alerts:
        st.success("No significant risk alerts detected")
        return
    
    # Group alerts by type
    critical_alerts = [a for a in alerts if a['type'] == 'CRITICAL']
    high_alerts = [a for a in alerts if a['type'] == 'HIGH']
    medium_alerts = [a for a in alerts if a['type'] == 'MEDIUM']
    
    # Display critical alerts
    if critical_alerts:
        st.error("CRITICAL ALERTS")
        for alert in critical_alerts:
            with st.expander(f"{alert['category']}: {alert['message']}", expanded=True):
                st.write(f"**Ticker:** {alert['ticker']}")
                if 'details' in alert:
                    display_alert_details(alert['details'], alert['category'])
    
    # Display high alerts
    if high_alerts:
        st.warning("HIGH RISK ALERTS")
        for alert in high_alerts:
            with st.expander(f"{alert['category']}: {alert['message']}"):
                st.write(f"**Ticker:** {alert['ticker']}")
                if 'details' in alert:
                    display_alert_details(alert['details'], alert['category'])
    
    # Display medium alerts
    if medium_alerts:
        st.info("MEDIUM RISK ALERTS")
        for alert in medium_alerts:
            with st.expander(f"{alert['category']}: {alert['message']}"):
                st.write(f"**Ticker:** {alert['ticker']}")
                if 'details' in alert:
                    display_alert_details(alert['details'], alert['category'])

def display_alert_details(details, category):
    """Display alert details in a clean format based on category"""
    if category == 'Correlation Risk' and isinstance(details, list):
        # Display correlation pairs in a clean table format
        st.write("**High Correlation Pairs:**")
        
        # Create a DataFrame for better display
        corr_data = []
        for pair in details:
            corr_data.append({
                'Ticker 1': pair['ticker1'],
                'Ticker 2': pair['ticker2'],
                'Correlation': f"{pair['correlation']:.2f}"
            })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # Add a note about correlation interpretation
            st.info("💡 **Note:** Correlations above 0.7 indicate high dependency between positions, which may reduce diversification benefits.")
    else:
        # For other types of details, display as before
        st.write("**Details:**", details)

def display_detailed_analysis(data, summary):
    """Display detailed portfolio analysis"""
    st.subheader("Detailed Analysis")
    
    # Position concentration chart
    if not data['weights'].empty:
        weights = data['weights'][data['weights']['Weight'] > 0]
        if len(weights) > 0:
            fig = px.bar(
                weights, 
                x='Ticker', 
                y='Weight',
                title="Position Weights",
                color='Weight',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk contribution analysis
    if not data['vol_contribution'].empty:
        vol_contrib = data['vol_contribution']
        if 'Ticker' in vol_contrib.columns and 'Volatility Contribution' in vol_contrib.columns:
            fig = px.pie(
                vol_contrib,
                names='Ticker',
                values='Volatility Contribution',
                title="Volatility Contribution by Position"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    if not data['correlation'].empty:
        corr_matrix = data['correlation']
        fig = px.imshow(
            corr_matrix,
            title="Position Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_recommendations(alerts, summary):
    """Display actionable recommendations based on alerts and metrics"""
    st.subheader("Recommendations")
    
    recommendations = []
    
    # Concentration recommendations
    if 'largest_position' in summary and summary['largest_position'] > 0.15:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Position Sizing',
            'recommendation': f"Consider reducing largest position from {summary['largest_position']:.1%} to below 15%",
            'rationale': 'High concentration increases idiosyncratic risk'
        })
    
    if 'top_3_concentration' in summary and summary['top_3_concentration'] > 0.50:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Diversification',
            'recommendation': f"Consider diversifying top 3 positions (currently {summary['top_3_concentration']:.1%})",
            'rationale': 'High concentration in few positions increases portfolio risk'
        })
    
    # Risk-adjusted return recommendations
    if 'sharpe' in summary and summary['sharpe'] < 0.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Risk Management',
            'recommendation': 'Review position sizing and risk allocation',
            'rationale': f'Low Sharpe ratio ({summary["sharpe"]:.2f}) indicates poor risk-adjusted returns'
        })
    
    # Volatility recommendations
    if 'ann_volatility' in summary and summary['ann_volatility'] > 0.30:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Volatility Management',
            'recommendation': 'Consider volatility-based position sizing',
            'rationale': f'High volatility ({summary["ann_volatility"]:.1%}) may require risk reduction'
        })
    
    # Market risk recommendations
    if 'beta_ndx' in summary and summary['beta_ndx'] > 1.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Market Risk',
            'recommendation': 'Consider hedging market exposure or adding defensive positions',
            'rationale': f'High beta ({summary["beta_ndx"]:.2f}) makes portfolio vulnerable to market downturns'
        })
    
    # Factor exposure recommendations
    factor_alerts = [a for a in alerts if a['category'] == 'Factor Exposure']
    if factor_alerts:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Factor Risk',
            'recommendation': 'Review factor exposures and consider diversification',
            'rationale': 'High factor exposures can lead to concentrated risk'
        })
    
    # Display recommendations
    if not recommendations:
        st.success("Portfolio appears well-balanced. Continue monitoring key metrics.")
        return
    
    for rec in recommendations:
        priority_color = {
            'HIGH': 'HIGH',
            'MEDIUM': 'MEDIUM',
            'LOW': 'LOW'
        }
        
        with st.expander(f"{priority_color[rec['priority']]} {rec['category']}: {rec['recommendation']}"):
            st.write(f"**Rationale:** {rec['rationale']}")

def display_risk_breakdown(risk_breakdown):
    """Display detailed risk breakdown by component"""
    st.subheader("Risk Breakdown by Component")
    
    risk_scores = risk_breakdown['risk_scores']
    risk_weights = risk_breakdown['risk_weights']
    details = risk_breakdown['details']
    
    # Create risk breakdown chart
    risk_data = []
    for risk_type, score in risk_scores.items():
        risk_data.append({
            'Risk Type': risk_type.replace('_', ' ').title(),
            'Score': score,
            'Weight': risk_weights[risk_type],
            'Weighted Score': score * risk_weights[risk_type]
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Sort by weighted score
    risk_df = risk_df.sort_values('Weighted Score', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        risk_df,
        x='Weighted Score',
        y='Risk Type',
        orientation='h',
        color='Score',
        color_continuous_scale='RdYlGn_r',
        title="Risk Contribution by Component"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed breakdown
    st.subheader("Detailed Risk Analysis")
    
    for risk_type in risk_df['Risk Type']:
        risk_key = risk_type.lower().replace(' ', '_')
        score = risk_scores.get(risk_key, 0)
        weight = risk_weights.get(risk_key, 0)
        detail = details.get(risk_key, {})
        
        with st.expander(f"{risk_type}: {score:.1%} (Weight: {weight:.1%})"):
            if detail:
                for key, value in detail.items():
                    if isinstance(value, float):
                        st.write(f"**{key.replace('_', ' ').title()}:** {value:.3f}")
                    elif isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.write("No detailed data available") 