import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

def concentration_tab(project_root, paths):
    """Render the concentration risk analysis tab."""
    
    # Load portfolio data
    portfolio_file = project_root / paths.get("portfolio_weights", "data/portfolio_weights.csv")
    
    if not portfolio_file.exists():
        st.info("No portfolio data found. Please run the pipeline to load portfolio data.")
        return
    
    try:
        # Load portfolio weights
        portfolio_df = pd.read_csv(portfolio_file)
        
        # Filter active positions
        active_portfolio = portfolio_df[portfolio_df['Weight'] > 0].copy()
        
        if active_portfolio.empty:
            st.info("No active positions found in portfolio.")
            return
        
        # Load Yahoo Finance data for concentration analysis
        concentration_data = load_concentration_data(active_portfolio, project_root, paths)
        
        if concentration_data is None:
            st.error("Failed to load concentration data from Yahoo Finance or CSV.")
            return
        
        # Display concentration analysis
        display_concentration_analysis(active_portfolio, concentration_data)
        
    except Exception as e:
        st.error(f"Error loading concentration data: {str(e)}")
        st.info("Please ensure portfolio data is available and Yahoo Finance is accessible.")

def load_concentration_data(portfolio_df, project_root, paths):
    """
    Load concentration-related data from Yahoo Finance or fallback to CSV.
    
    Required Yahoo Finance data:
    - sector: Company sector classification
    - industry: Company industry classification  
    - marketCap: Market capitalization
    - averageVolume: 30-day average trading volume
    - volume: Current day trading volume
    - bid: Current bid price
    - ask: Current ask price
    - previousClose: Previous closing price
    """
    
    concentration_data = {}
    csv_file = project_root / paths.get("concentration_data", "data/concentration_data.csv")
    
    # Try to load from Yahoo Finance first
    try:
        with st.spinner("Loading concentration data from Yahoo Finance..."):
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                
                try:
                    # Get Yahoo Finance data
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    
                    # Extract required data
                    concentration_data[ticker] = {
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'avg_volume': info.get('averageVolume', 0),
                        'current_volume': info.get('volume', 0),
                        'bid': info.get('bid', 0),
                        'ask': info.get('ask', 0),
                        'previous_close': info.get('previousClose', 0),
                        'weight': row['Weight'],
                        'market_value': row['MarketValue']
                    }
                    
                except Exception as e:
                    st.warning(f"Failed to load data for {ticker}: {str(e)}")
                    # Use placeholder data
                    concentration_data[ticker] = {
                        'sector': 'Unknown',
                        'industry': 'Unknown', 
                        'market_cap': 0,
                        'avg_volume': 0,
                        'current_volume': 0,
                        'bid': 0,
                        'ask': 0,
                        'previous_close': 0,
                        'weight': row['Weight'],
                        'market_value': row['MarketValue']
                    }
        
        # Save to CSV for future use
        save_concentration_data_to_csv(concentration_data, csv_file)
        st.success("✅ Concentration data loaded from Yahoo Finance and saved to CSV")
        
    except Exception as e:
        st.warning(f"⚠️ Failed to load from Yahoo Finance: {str(e)}")
        
        # Try to load from CSV fallback
        try:
            if csv_file.exists():
                concentration_data = load_concentration_data_from_csv(portfolio_df, csv_file)
                st.info("📁 Using cached concentration data from CSV file")
            else:
                st.error("❌ No internet connection and no cached data available")
                return None
                
        except Exception as csv_error:
            st.error(f"❌ Failed to load from CSV: {str(csv_error)}")
            return None
    
    return concentration_data

def save_concentration_data_to_csv(concentration_data, csv_file):
    """Save concentration data to CSV file."""
    try:
        # Ensure directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data_list = []
        for ticker, data in concentration_data.items():
            data_list.append({
                'ticker': ticker,
                'sector': data.get('sector', 'Unknown'),
                'industry': data.get('industry', 'Unknown'),
                'market_cap': data.get('market_cap', 0),
                'avg_volume': data.get('avg_volume', 0),
                'current_volume': data.get('current_volume', 0),
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0),
                'previous_close': data.get('previous_close', 0),
                'weight': data.get('weight', 0),
                'market_value': data.get('market_value', 0),
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        df = pd.DataFrame(data_list)
        df.to_csv(csv_file, index=False)
        
    except Exception as e:
        st.warning(f"Failed to save concentration data to CSV: {str(e)}")

def load_concentration_data_from_csv(portfolio_df, csv_file):
    """Load concentration data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        concentration_data = {}
        
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            csv_row = df[df['ticker'] == ticker]
            
            if not csv_row.empty:
                csv_data = csv_row.iloc[0]
                concentration_data[ticker] = {
                    'sector': csv_data.get('sector', 'Unknown'),
                    'industry': csv_data.get('industry', 'Unknown'),
                    'market_cap': csv_data.get('market_cap', 0),
                    'avg_volume': csv_data.get('avg_volume', 0),
                    'current_volume': csv_data.get('current_volume', 0),
                    'bid': csv_data.get('bid', 0),
                    'ask': csv_data.get('ask', 0),
                    'previous_close': csv_data.get('previous_close', 0),
                    'weight': row['Weight'],
                    'market_value': row['MarketValue']
                }
            else:
                # Fallback for missing data
                concentration_data[ticker] = {
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'avg_volume': 0,
                    'current_volume': 0,
                    'bid': 0,
                    'ask': 0,
                    'previous_close': 0,
                    'weight': row['Weight'],
                    'market_value': row['MarketValue']
                }
        
        return concentration_data
        
    except Exception as e:
        raise Exception(f"Failed to load concentration data from CSV: {str(e)}")

def display_concentration_analysis(portfolio_df, concentration_data):
    """Display comprehensive concentration analysis."""
    
    # Create tabs for different concentration views
    conc_tab1, conc_tab2, conc_tab3, conc_tab4 = st.tabs([
        "Position Concentration", 
        "Sector Concentration", 
        "Market Cap Concentration",
        "Concentration Alerts"
    ])
    
    with conc_tab1:
        display_position_concentration(portfolio_df, concentration_data)
    
    with conc_tab2:
        display_sector_concentration(portfolio_df, concentration_data)
    
    with conc_tab3:
        display_market_cap_concentration(portfolio_df, concentration_data)
    
    with conc_tab4:
        display_concentration_alerts(portfolio_df, concentration_data)

def display_position_concentration(portfolio_df, concentration_data):
    """Display position-level concentration analysis."""
    
    st.subheader("Position Concentration Analysis")
    
    # Calculate position concentration metrics
    weights = portfolio_df['Weight'].values
    sorted_weights = np.sort(weights)[::-1]  # Sort descending
    
    # Position concentration metrics
    largest_position = sorted_weights[0]
    top_3_concentration = np.sum(sorted_weights[:3])
    top_5_concentration = np.sum(sorted_weights[:5])
    top_10_concentration = np.sum(sorted_weights[:10])
    herfindahl_index = np.sum(weights ** 2)
    effective_positions = 1 / herfindahl_index
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Largest Position", f"{largest_position:.1%}")
        st.metric("Top 3 Concentration", f"{top_3_concentration:.1%}")
    
    with col2:
        st.metric("Herfindahl Index", f"{herfindahl_index:.3f}")
        st.metric("Top 5 Concentration", f"{top_5_concentration:.1%}")
    
    with col3:
        st.metric("Effective Positions", f"{effective_positions:.1f}")
        st.metric("Top 10 Concentration", f"{top_10_concentration:.1%}")
    
    # Position concentration chart
    st.subheader("Position Weight Distribution")
    
    # Get top 10 positions for chart
    top_positions = portfolio_df.nlargest(10, 'Weight')
    
    fig = px.bar(
        top_positions, 
        x='Ticker', 
        y='Weight',
        title="Top 10 Position Weights",
        labels={'Weight': 'Weight', 'Ticker': ''}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        xaxis_title="",
        yaxis_title="Weight"
    )
    
    # Format y-axis as percentages without decimals
    fig.update_yaxes(tickformat=".0%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Position concentration table
    st.subheader("Position Concentration Details")
    
    concentration_table = []
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        weight = float(row['Weight'])  # Ensure numeric
        market_value = float(row['MarketValue'])  # Ensure numeric
        
        # Get additional data from concentration_data
        ticker_data = concentration_data.get(ticker, {})
        sector = ticker_data.get('sector', '')
        market_cap = ticker_data.get('market_cap', 0)
        
        # Format market cap in billions
        if market_cap > 0:
            market_cap_bn = market_cap / 1e9
            market_cap_formatted = f"${market_cap_bn:,.2f}"
        else:
            market_cap_formatted = ""
        
        concentration_table.append({
            'Ticker': ticker,
            'Weight': weight,  # Keep as numeric for sorting
            'Market Value': market_value,  # Keep as numeric for sorting
            'Sector': sector if sector != 'Unknown' else '',
            'Market Cap ($bn)': market_cap_formatted
        })
    
    concentration_df = pd.DataFrame(concentration_table)
    
    # Sort by weight descending - ensure numeric sorting
    concentration_df = concentration_df.sort_values('Weight', ascending=False)
    
    # Format display columns AFTER sorting
    display_df = concentration_df.copy()
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1%}")
    display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_sector_concentration(portfolio_df, concentration_data):
    """Display sector-level concentration analysis."""
    
    st.subheader("Sector Concentration Analysis")
    
    # Aggregate by sector
    sector_data = {}
    for ticker, data in concentration_data.items():
        sector = data.get('sector', 'Unknown')
        weight = float(data.get('weight', 0))  # Ensure weight is numeric
        
        if sector not in sector_data:
            sector_data[sector] = {'weight': 0.0, 'positions': []}
        
        sector_data[sector]['weight'] += weight
        sector_data[sector]['positions'].append(ticker)
    
    # Create sector summary
    sector_summary = []
    for sector, data in sector_data.items():
        sector_summary.append({
            'Sector': sector,
            'Weight': float(data['weight']),  # Ensure numeric
            'Positions': len(data['positions']),
            'Position List': ', '.join(data['positions'])
        })
    
    sector_df = pd.DataFrame(sector_summary)
    # Ensure Weight column is numeric and sort
    sector_df['Weight'] = pd.to_numeric(sector_df['Weight'], errors='coerce')
    sector_df = sector_df.sort_values('Weight', ascending=False)
    
    # Display sector metrics
    col1, col2 = st.columns(2)
    
    with col1:
        largest_sector = sector_df.iloc[0] if not sector_df.empty else None
        if largest_sector is not None:
            st.metric("Largest Sector", largest_sector['Sector'])
    
    with col2:
        sector_count = len(sector_df)
        st.metric("Number of Sectors", sector_count)
    
    # Sector concentration chart
    fig = px.pie(
        sector_df, 
        values='Weight', 
        names='Sector',
        title="Portfolio Sector Allocation"
    )
    
    # Change tooltip hover to just display the name of the sector
    fig.update_traces(hovertemplate="%{label}<extra></extra>")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector details table
    st.subheader("Sector Breakdown")
    
    # Create display dataframe with numeric sorting first
    display_df = sector_df.copy()
    
    # Sort by numeric weight first
    display_df = display_df.sort_values('Weight', ascending=False)
    
    # Format weights as percentages after sorting
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_market_cap_concentration(portfolio_df, concentration_data):
    """Display market cap concentration analysis."""
    
    st.subheader("Market Cap Concentration Analysis")
    
    # Categorize by market cap
    market_cap_categories = {
        'Mega Cap (>$100B)': {'min': 100e9, 'max': float('inf'), 'weight': 0, 'positions': []},
        'Large Cap ($10B-$100B)': {'min': 10e9, 'max': 100e9, 'weight': 0, 'positions': []},
        'Mid Cap ($2B-$10B)': {'min': 2e9, 'max': 10e9, 'weight': 0, 'positions': []},
        'Small Cap ($500M-$2B)': {'min': 500e6, 'max': 2e9, 'weight': 0, 'positions': []},
        'Micro Cap (<$500M)': {'min': 0, 'max': 500e6, 'weight': 0, 'positions': []}
    }
    
    # Categorize positions
    for ticker, data in concentration_data.items():
        market_cap = data.get('market_cap', 0)
        weight = data.get('weight', 0)
        
        for category, criteria in market_cap_categories.items():
            if criteria['min'] <= market_cap < criteria['max']:
                criteria['weight'] += weight
                criteria['positions'].append(ticker)
                break
    
    # Create market cap summary
    market_cap_summary = []
    for category, data in market_cap_categories.items():
        market_cap_summary.append({
            'Market Cap Category': category,
            'Weight': data['weight'],
            'Positions': len(data['positions']),
            'Position List': ', '.join(data['positions']) if data['positions'] else 'None'
        })
    
    market_cap_df = pd.DataFrame(market_cap_summary)
    market_cap_df = market_cap_df[market_cap_df['Weight'] > 0]  # Only show categories with positions
    
    # Display market cap metrics
    col1, col2 = st.columns(2)
    
    with col1:
        largest_category = market_cap_df.iloc[0] if not market_cap_df.empty else None
        if largest_category is not None:
            st.metric("Largest Market Cap Category", 
                     largest_category['Market Cap Category'])
    
    with col2:
        category_count = len(market_cap_df)
        st.metric("Market Cap Categories", category_count)
    
    # Market cap concentration chart
    if not market_cap_df.empty:
        fig = px.bar(
            market_cap_df,
            x='Market Cap Category',
            y='Weight',
            title="Portfolio Market Cap Distribution",
            labels={'Weight': 'Weight', 'Market Cap Category': ''}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            xaxis_title="",
            yaxis_title="Weight"
        )
        
        # Format y-axis as percentages without decimals
        fig.update_yaxes(tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Market cap details table
    st.subheader("Market Cap Breakdown")
    
    # Create display dataframe with numeric sorting first
    display_df = market_cap_df.copy()
    
    # Sort by numeric weight first
    display_df = display_df.sort_values('Weight', ascending=False)
    
    # Format weights as percentages after sorting
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_concentration_alerts(portfolio_df, concentration_data):
    """Display concentration risk alerts and recommendations."""
    
    st.subheader("Concentration Risk Alerts")
    
    # Define concentration limits
    concentration_limits = {
        'max_single_position': 0.15,  # 15%
        'max_top_3': 0.45,           # 45%
        'max_top_5': 0.60,           # 60%
        'max_tech_sector': 0.80,     # 80%
        'max_single_sector': 0.50    # 50%
    }
    
    # Calculate current metrics
    weights = portfolio_df['Weight'].values
    sorted_weights = np.sort(weights)[::-1]
    
    current_metrics = {
        'largest_position': sorted_weights[0],
        'top_3_concentration': np.sum(sorted_weights[:3]),
        'top_5_concentration': np.sum(sorted_weights[:5])
    }
    
    # Calculate sector concentration
    sector_data = {}
    for ticker, data in concentration_data.items():
        sector = data.get('sector', 'Unknown')
        weight = data.get('weight', 0)
        
        if sector not in sector_data:
            sector_data[sector] = 0
        sector_data[sector] += weight
    
    current_metrics['largest_sector'] = max(sector_data.values()) if sector_data else 0
    
    # Generate alerts
    alerts = []
    
    if current_metrics['largest_position'] > concentration_limits['max_single_position']:
        alerts.append({
            'type': 'High',
            'message': f"Largest position ({current_metrics['largest_position']:.1%}) exceeds {concentration_limits['max_single_position']:.0%} limit",
            'recommendation': "Consider reducing largest position size"
        })
    
    if current_metrics['top_3_concentration'] > concentration_limits['max_top_3']:
        alerts.append({
            'type': 'Medium',
            'message': f"Top 3 concentration ({current_metrics['top_3_concentration']:.1%}) exceeds {concentration_limits['max_top_3']:.0%} limit",
            'recommendation': "Consider diversifying top positions"
        })
    
    if current_metrics['top_5_concentration'] > concentration_limits['max_top_5']:
        alerts.append({
            'type': 'Medium',
            'message': f"Top 5 concentration ({current_metrics['top_5_concentration']:.1%}) exceeds {concentration_limits['max_top_5']:.0%} limit",
            'recommendation': "Consider adding more positions"
        })
    
    if current_metrics['largest_sector'] > concentration_limits['max_single_sector']:
        alerts.append({
            'type': 'High',
            'message': f"Largest sector ({current_metrics['largest_sector']:.1%}) exceeds {concentration_limits['max_single_sector']:.0%} limit",
            'recommendation': "Consider reducing sector concentration"
        })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['type'] == 'High':
                st.error(f"🚨 {alert['message']}")
            else:
                st.warning(f"⚠️ {alert['message']}")
            
            st.info(f"💡 Recommendation: {alert['recommendation']}")
            st.markdown("---")
    else:
        st.success("✅ No concentration risk alerts - portfolio appears well-diversified")
    
    # Display current metrics vs limits
    st.subheader("Current Metrics vs Limits")
    
    metrics_df = pd.DataFrame([
        {
            'Metric': 'Largest Position',
            'Current': f"{current_metrics['largest_position']:.1%}",
            'Limit': f"{concentration_limits['max_single_position']:.0%}",
            'Status': '⚠️ Exceeds' if current_metrics['largest_position'] > concentration_limits['max_single_position'] else '✅ OK'
        },
        {
            'Metric': 'Top 3 Concentration',
            'Current': f"{current_metrics['top_3_concentration']:.1%}",
            'Limit': f"{concentration_limits['max_top_3']:.0%}",
            'Status': '⚠️ Exceeds' if current_metrics['top_3_concentration'] > concentration_limits['max_top_3'] else '✅ OK'
        },
        {
            'Metric': 'Top 5 Concentration',
            'Current': f"{current_metrics['top_5_concentration']:.1%}",
            'Limit': f"{concentration_limits['max_top_5']:.0%}",
            'Status': '⚠️ Exceeds' if current_metrics['top_5_concentration'] > concentration_limits['max_top_5'] else '✅ OK'
        },
        {
            'Metric': 'Largest Sector',
            'Current': f"{current_metrics['largest_sector']:.1%}",
            'Limit': f"{concentration_limits['max_single_sector']:.0%}",
            'Status': '⚠️ Exceeds' if current_metrics['largest_sector'] > concentration_limits['max_single_sector'] else '✅ OK'
        }
    ])
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True) 