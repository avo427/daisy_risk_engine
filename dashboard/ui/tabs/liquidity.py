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

def liquidity_tab(project_root, paths):
    """Render the liquidity risk analysis tab."""
    
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
        
        # Load Yahoo Finance data for liquidity analysis
        liquidity_data = load_liquidity_data(active_portfolio, project_root, paths)
        
        if liquidity_data is None:
            st.error("Failed to load liquidity data from Yahoo Finance or CSV.")
            return
        
        # Display liquidity analysis
        display_liquidity_analysis(active_portfolio, liquidity_data)
        
    except Exception as e:
        st.error(f"Error loading liquidity data: {str(e)}")
        st.info("Please ensure portfolio data is available and Yahoo Finance is accessible.")

def load_liquidity_data(portfolio_df, project_root, paths):
    """
    Load liquidity-related data from Yahoo Finance or fallback to CSV.
    
    Required Yahoo Finance data:
    - bid: Current bid price
    - ask: Current ask price
    - bidSize: Current bid size (shares)
    - askSize: Current ask size (shares)
    - volume: Current day trading volume
    - averageVolume: 30-day average trading volume
    - previousClose: Previous closing price
    - open: Today's opening price
    - dayHigh: Today's high price
    - dayLow: Today's low price
    - marketCap: Market capitalization
    """
    
    liquidity_data = {}
    csv_file = project_root / paths.get("liquidity_data", "data/liquidity_data.csv")
    
    # Try to load from Yahoo Finance first
    try:
        with st.spinner("Loading liquidity data from Yahoo Finance..."):
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                
                try:
                    # Get Yahoo Finance data
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    
                    # Extract required data
                    bid = info.get('bid', 0)
                    ask = info.get('ask', 0)
                    bid_size = info.get('bidSize', 0)
                    ask_size = info.get('askSize', 0)
                    volume = info.get('volume', 0)
                    avg_volume = info.get('averageVolume', 0)
                    prev_close = info.get('previousClose', 0)
                    open_price = info.get('open', 0)
                    day_high = info.get('dayHigh', 0)
                    day_low = info.get('dayLow', 0)
                    market_cap = info.get('marketCap', 0)
                    
                    # Calculate liquidity metrics
                    spread = ask - bid if bid > 0 and ask > 0 else 0
                    spread_pct = (spread / bid) * 100 if bid > 0 else 0
                    mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else prev_close
                    
                    # Get market value from portfolio data
                    market_value = row['MarketValue']
                    
                    # Volume-based liquidity score (relative to position size)
                    position_shares = market_value / mid_price if mid_price > 0 else 0
                    volume_to_position_ratio = avg_volume / position_shares if position_shares > 0 else 0
                    
                    if volume_to_position_ratio > 100:  # Can trade 100x position size daily
                        volume_category = "High"
                        volume_score = 1.0
                    elif volume_to_position_ratio > 50:  # Can trade 50x position size daily
                        volume_category = "Medium"
                        volume_score = 0.8
                    elif volume_to_position_ratio > 20:  # Can trade 20x position size daily
                        volume_category = "Low"
                        volume_score = 0.6
                    else:  # Can trade less than 20x position size daily
                        volume_category = "Very Low"
                        volume_score = 0.4
                    
                    # Spread-based liquidity score
                    if spread_pct < 0.1:  # <0.1%
                        spread_category = "Very Tight"
                        spread_score = 1.0
                    elif spread_pct < 0.25:  # 0.1%-0.25%
                        spread_category = "Tight"
                        spread_score = 0.8
                    elif spread_pct < 0.5:  # 0.25%-0.5%
                        spread_category = "Normal"
                        spread_score = 0.6
                    elif spread_pct < 1.0:  # 0.5%-1.0%
                        spread_category = "Wide"
                        spread_score = 0.4
                    else:  # >1.0%
                        spread_category = "Very Wide"
                        spread_score = 0.2
                    
                    # Combined liquidity score
                    combined_score = (volume_score + spread_score) / 2
                    
                    liquidity_data[ticker] = {
                        'bid': bid,
                        'ask': ask,
                        'bid_size': bid_size,
                        'ask_size': ask_size,
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'spread_category': spread_category,
                        'spread_score': spread_score,
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'volume_category': volume_category,
                        'volume_score': volume_score,
                        'mid_price': mid_price,
                        'prev_close': prev_close,
                        'open': open_price,
                        'day_high': day_high,
                        'day_low': day_low,
                        'market_cap': market_cap,
                        'combined_score': combined_score,
                        'weight': row['Weight'],
                        'market_value': row['MarketValue']
                    }
                    
                except Exception as e:
                    st.warning(f"Failed to load liquidity data for {ticker}: {str(e)}")
                    # Use placeholder data
                    liquidity_data[ticker] = {
                        'bid': 0,
                        'ask': 0,
                        'bid_size': 0,
                        'ask_size': 0,
                        'spread': 0,
                        'spread_pct': 0,
                        'spread_category': 'Unknown',
                        'spread_score': 0,
                        'volume': 0,
                        'avg_volume': 0,
                        'volume_category': 'Unknown',
                        'volume_score': 0,
                        'mid_price': 0,
                        'prev_close': 0,
                        'open': 0,
                        'day_high': 0,
                        'day_low': 0,
                        'market_cap': 0,
                        'combined_score': 0,
                        'weight': row['Weight'],
                        'market_value': row['MarketValue']
                    }
        
        # Save to CSV for future use
        save_liquidity_data_to_csv(liquidity_data, csv_file)
        st.success("✅ Liquidity data loaded from Yahoo Finance and saved to CSV")
        
    except Exception as e:
        st.warning(f"⚠️ Failed to load from Yahoo Finance: {str(e)}")
        
        # Try to load from CSV fallback
        try:
            if csv_file.exists():
                liquidity_data = load_liquidity_data_from_csv(portfolio_df, csv_file)
                st.info("📁 Using cached liquidity data from CSV file")
            else:
                st.error("❌ No internet connection and no cached data available")
                return None
                
        except Exception as csv_error:
            st.error(f"❌ Failed to load from CSV: {str(csv_error)}")
            return None
    
    return liquidity_data

def save_liquidity_data_to_csv(liquidity_data, csv_file):
    """Save liquidity data to CSV file."""
    try:
        # Ensure directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data_list = []
        for ticker, data in liquidity_data.items():
            data_list.append({
                'ticker': ticker,
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0),
                'bid_size': data.get('bid_size', 0),
                'ask_size': data.get('ask_size', 0),
                'spread': data.get('spread', 0),
                'spread_pct': data.get('spread_pct', 0),
                'spread_category': data.get('spread_category', 'Unknown'),
                'spread_score': data.get('spread_score', 0),
                'volume': data.get('volume', 0),
                'avg_volume': data.get('avg_volume', 0),
                'volume_category': data.get('volume_category', 'Unknown'),
                'volume_score': data.get('volume_score', 0),
                'mid_price': data.get('mid_price', 0),
                'prev_close': data.get('prev_close', 0),
                'open': data.get('open', 0),
                'day_high': data.get('day_high', 0),
                'day_low': data.get('day_low', 0),
                'market_cap': data.get('market_cap', 0),
                'combined_score': data.get('combined_score', 0),
                'weight': data.get('weight', 0),
                'market_value': data.get('market_value', 0),
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        df = pd.DataFrame(data_list)
        df.to_csv(csv_file, index=False)
        
    except Exception as e:
        st.warning(f"Failed to save liquidity data to CSV: {str(e)}")

def load_liquidity_data_from_csv(portfolio_df, csv_file):
    """Load liquidity data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        liquidity_data = {}
        
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            csv_row = df[df['ticker'] == ticker]
            
            if not csv_row.empty:
                csv_data = csv_row.iloc[0]
                liquidity_data[ticker] = {
                    'bid': csv_data.get('bid', 0),
                    'ask': csv_data.get('ask', 0),
                    'bid_size': csv_data.get('bid_size', 0),
                    'ask_size': csv_data.get('ask_size', 0),
                    'spread': csv_data.get('spread', 0),
                    'spread_pct': csv_data.get('spread_pct', 0),
                    'spread_category': csv_data.get('spread_category', 'Unknown'),
                    'spread_score': csv_data.get('spread_score', 0),
                    'volume': csv_data.get('volume', 0),
                    'avg_volume': csv_data.get('avg_volume', 0),
                    'volume_category': csv_data.get('volume_category', 'Unknown'),
                    'volume_score': csv_data.get('volume_score', 0),
                    'mid_price': csv_data.get('mid_price', 0),
                    'prev_close': csv_data.get('prev_close', 0),
                    'open': csv_data.get('open', 0),
                    'day_high': csv_data.get('day_high', 0),
                    'day_low': csv_data.get('day_low', 0),
                    'market_cap': csv_data.get('market_cap', 0),
                    'combined_score': csv_data.get('combined_score', 0),
                    'weight': row['Weight'],
                    'market_value': row['MarketValue']
                }
            else:
                # Fallback for missing data
                liquidity_data[ticker] = {
                    'bid': 0,
                    'ask': 0,
                    'bid_size': 0,
                    'ask_size': 0,
                    'spread': 0,
                    'spread_pct': 0,
                    'spread_category': 'Unknown',
                    'spread_score': 0,
                    'volume': 0,
                    'avg_volume': 0,
                    'volume_category': 'Unknown',
                    'volume_score': 0,
                    'mid_price': 0,
                    'prev_close': 0,
                    'open': 0,
                    'day_high': 0,
                    'day_low': 0,
                    'market_cap': 0,
                    'combined_score': 0,
                    'weight': row['Weight'],
                    'market_value': row['MarketValue']
                }
        
        return liquidity_data
        
    except Exception as e:
        raise Exception(f"Failed to load liquidity data from CSV: {str(e)}")

def display_liquidity_analysis(portfolio_df, liquidity_data):
    """Display comprehensive liquidity analysis."""
    
    # Create tabs for different liquidity views
    liq_tab1, liq_tab2, liq_tab3, liq_tab4 = st.tabs([
        "Portfolio Liquidity Overview", 
        "Bid-Ask Spread Analysis", 
        "Volume Analysis",
        "Liquidity Alerts"
    ])
    
    with liq_tab1:
        display_portfolio_liquidity_overview(portfolio_df, liquidity_data)
    
    with liq_tab2:
        display_bid_ask_analysis(portfolio_df, liquidity_data)
    
    with liq_tab3:
        display_volume_analysis(portfolio_df, liquidity_data)
    
    with liq_tab4:
        display_liquidity_alerts(portfolio_df, liquidity_data)

def display_portfolio_liquidity_overview(portfolio_df, liquidity_data):
    """Display overall portfolio liquidity metrics."""
    
    st.subheader("Portfolio Liquidity Overview")
    
    # Calculate portfolio-level liquidity metrics
    total_weight = 0
    weighted_liquidity_score = 0
    total_market_value = 0
    
    for ticker, data in liquidity_data.items():
        weight = data.get('weight', 0)
        market_value = data.get('market_value', 0)
        combined_score = data.get('combined_score', 0)
        
        total_weight += weight
        weighted_liquidity_score += combined_score * weight
        total_market_value += market_value
    
    # Overall portfolio liquidity score (convert from 0-1 to 0-10 scale)
    overall_liquidity_score = weighted_liquidity_score / total_weight if total_weight > 0 else 0
    overall_liquidity_score_10 = overall_liquidity_score * 10  # Convert to 0-10 scale
    
    # Liquidity risk assessment
    if overall_liquidity_score > 0.8:
        risk_assessment = "Low Liquidity"
        liquidation_time = "1-2 days"
        risk_color = "green"
    elif overall_liquidity_score > 0.6:
        risk_assessment = "Medium Liquidity"
        liquidation_time = "2-5 days"
        risk_color = "orange"
    else:
        risk_assessment = "High Liquidity"
        liquidation_time = "5+ days"
        risk_color = "red"
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Liquidity Score", f"{overall_liquidity_score_10:.1f}/10")
    
    with col2:
        st.metric("Estimated Liquidation Time", liquidation_time)
    
    with col3:
        st.metric("Risk Assessment", risk_assessment)
    
    # Portfolio liquidity distribution
    st.subheader("Liquidity Score Distribution")
    
    # Categorize positions by liquidity score
    liquidity_categories = {
        'High Liquidity (8-10)': {'min': 0.8, 'max': 1.0, 'weight': 0, 'positions': []},
        'Medium Liquidity (6-8)': {'min': 0.6, 'max': 0.8, 'weight': 0, 'positions': []},
        'Low Liquidity (4-6)': {'min': 0.4, 'max': 0.6, 'weight': 0, 'positions': []},
        'Very Low Liquidity (0-4)': {'min': 0.0, 'max': 0.4, 'weight': 0, 'positions': []}
    }
    
    for ticker, data in liquidity_data.items():
        score = data.get('combined_score', 0)
        weight = data.get('weight', 0)
        
        for category, criteria in liquidity_categories.items():
            if criteria['min'] <= score < criteria['max']:
                criteria['weight'] += weight
                criteria['positions'].append(ticker)
                break
    
    # Create liquidity distribution summary
    liquidity_summary = []
    for category, data in liquidity_categories.items():
        if data['weight'] > 0:  # Only show categories with positions
            liquidity_summary.append({
                'Liquidity Category': category,
                'Weight': data['weight'],
                'Positions': len(data['positions']),
                'Position List': ', '.join(data['positions'])
            })
    
    liquidity_df = pd.DataFrame(liquidity_summary)
    
    # Display liquidity distribution chart
    if not liquidity_df.empty:
        fig = px.pie(
            liquidity_df,
            values='Weight',
            names='Liquidity Category',
            title="Portfolio Liquidity Distribution"
        )
        
        # Change tooltip hover to just display the liquidity category name
        fig.update_traces(hovertemplate="%{label}<extra></extra>")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Position-level liquidity table
    st.subheader("Position Liquidity Details")
    
    liquidity_table = []
    for ticker, data in liquidity_data.items():
        liquidity_table.append({
            'Ticker': ticker,
            'Weight': f"{data.get('weight', 0):.1%}",
            'Market Value': f"${data.get('market_value', 0):,.0f}",
            'Liquidity Score': f"{data.get('combined_score', 0) * 10:.1f}",
            'Spread': f"{data.get('spread_pct', 0):.2f}%",
            'Avg Volume': f"{data.get('avg_volume', 0):,.0f}",
            'Volume Category': data.get('volume_category', 'Unknown')
        })
    
    liquidity_df = pd.DataFrame(liquidity_table)
    liquidity_df = liquidity_df.sort_values('Liquidity Score', ascending=False)
    
    st.dataframe(liquidity_df, use_container_width=True, hide_index=True)

def display_bid_ask_analysis(portfolio_df, liquidity_data):
    """Display bid-ask spread analysis."""
    
    st.subheader("Bid-Ask Spread Analysis")
    
    # Calculate spread metrics
    spreads = [data.get('spread_pct', 0) for data in liquidity_data.values()]
    avg_spread = np.mean(spreads) if spreads else 0
    min_spread = np.min(spreads) if spreads else 0
    max_spread = np.max(spreads) if spreads else 0
    
    # Display spread metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Spread", f"{avg_spread:.2f}%")
    
    with col2:
        st.metric("Tightest Spread", f"{min_spread:.2f}%")
    
    with col3:
        st.metric("Widest Spread", f"{max_spread:.2f}%")
    
    # Spread distribution chart
    st.subheader("Bid-Ask Spread Distribution")
    
    # Create spread categories
    spread_categories = {
        'Very Tight (<0.1%)': {'min': 0, 'max': 0.1, 'count': 0, 'positions': []},
        'Tight (0.1-0.25%)': {'min': 0.1, 'max': 0.25, 'count': 0, 'positions': []},
        'Normal (0.25-0.5%)': {'min': 0.25, 'max': 0.5, 'count': 0, 'positions': []},
        'Wide (0.5-1.0%)': {'min': 0.5, 'max': 1.0, 'count': 0, 'positions': []},
        'Very Wide (>1.0%)': {'min': 1.0, 'max': float('inf'), 'count': 0, 'positions': []}
    }
    
    for ticker, data in liquidity_data.items():
        spread_pct = data.get('spread_pct', 0)
        
        for category, criteria in spread_categories.items():
            if criteria['min'] <= spread_pct < criteria['max']:
                criteria['count'] += 1
                criteria['positions'].append(ticker)
                break
    
    # Create spread summary
    spread_summary = []
    for category, data in spread_categories.items():
        if data['count'] > 0:  # Only show categories with positions
            spread_summary.append({
                'Spread Category': category,
                'Count': data['count'],
                'Positions': ', '.join(data['positions'])
            })
    
    spread_df = pd.DataFrame(spread_summary)
    
    # Display spread distribution chart
    if not spread_df.empty:
        fig = px.bar(
            spread_df,
            x='Spread Category',
            y='Count',
            title="Bid-Ask Spread Distribution",
            labels={'Count': 'Position Count', 'Spread Category': ''}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            xaxis_title="",
            yaxis_title="Position Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Spread details table
    st.subheader("Bid-Ask Spread Details")
    
    spread_table = []
    for ticker, data in liquidity_data.items():
        spread_table.append({
            'Ticker': ticker,
            'Weight': f"{data.get('weight', 0):.1%}",
            'Bid': f"${data.get('bid', 0):.2f}",
            'Ask': f"${data.get('ask', 0):.2f}",
            'Spread': f"${data.get('spread', 0):.2f}",
            'Spread %': f"{data.get('spread_pct', 0):.2f}%",
            'Category': data.get('spread_category', 'Unknown')
        })
    
    spread_details_df = pd.DataFrame(spread_table)
    spread_details_df = spread_details_df.sort_values('Spread %', ascending=True)
    
    st.dataframe(spread_details_df, use_container_width=True, hide_index=True)

def display_volume_analysis(portfolio_df, liquidity_data):
    """Display volume-based liquidity analysis."""
    
    st.subheader("Volume Analysis")
    
    # Calculate volume metrics
    volumes = [data.get('avg_volume', 0) for data in liquidity_data.values()]
    avg_volume = np.mean(volumes) if volumes else 0
    total_volume = np.sum(volumes) if volumes else 0
    
    # Display volume metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Volume", f"{avg_volume:,.0f}")
    
    with col2:
        st.metric("Total Portfolio Volume", f"{total_volume:,.0f}")
    
    with col3:
        # Calculate volume-weighted average
        volume_weighted_avg = 0
        total_weight = 0
        for data in liquidity_data.values():
            volume = data.get('avg_volume', 0)
            weight = data.get('weight', 0)
            volume_weighted_avg += volume * weight
            total_weight += weight
        
        volume_weighted_avg = volume_weighted_avg / total_weight if total_weight > 0 else 0
        st.metric("Volume-Weighted Average", f"{volume_weighted_avg:,.0f}")
    
    # Volume distribution chart
    st.subheader("Volume Distribution")
    
    # Create volume categories
    volume_categories = {
        'High Volume (>10M)': {'min': 10e6, 'max': float('inf'), 'weight': 0, 'positions': []},
        'Medium Volume (1M-10M)': {'min': 1e6, 'max': 10e6, 'weight': 0, 'positions': []},
        'Low Volume (100K-1M)': {'min': 100e3, 'max': 1e6, 'weight': 0, 'positions': []},
        'Very Low Volume (<100K)': {'min': 0, 'max': 100e3, 'weight': 0, 'positions': []}
    }
    
    for ticker, data in liquidity_data.items():
        volume = data.get('avg_volume', 0)
        weight = data.get('weight', 0)
        
        for category, criteria in volume_categories.items():
            if criteria['min'] <= volume < criteria['max']:
                criteria['weight'] += weight
                criteria['positions'].append(ticker)
                break
    
    # Create volume summary
    volume_summary = []
    for category, data in volume_categories.items():
        if data['weight'] > 0:  # Only show categories with positions
            volume_summary.append({
                'Volume Category': category,
                'Weight': data['weight'],
                'Positions': len(data['positions']),
                'Position List': ', '.join(data['positions'])
            })
    
    volume_df = pd.DataFrame(volume_summary)
    
    # Display volume distribution chart
    if not volume_df.empty:
        fig = px.bar(
            volume_df,
            x='Volume Category',
            y='Weight',
            title="Volume Distribution",
            labels={'Weight': 'Position Percentage', 'Volume Category': ''}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            xaxis_title="",
            yaxis_title="Position Percentage"
        )
        
        # Format y-axis as percentages without decimals
        fig.update_yaxes(tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Volume details table
    st.subheader("Volume Details")
    
    volume_table = []
    for ticker, data in liquidity_data.items():
        volume_table.append({
            'Ticker': ticker,
            'Weight': f"{data.get('weight', 0):.1%}",
            'Market Value': f"${data.get('market_value', 0):,.0f}",
            'Avg Volume': f"{data.get('avg_volume', 0):,.0f}",
            'Current Volume': f"{data.get('volume', 0):,.0f}",
            'Volume Category': data.get('volume_category', 'Unknown'),
            'Volume Score': f"{data.get('volume_score', 0) * 10:.1f}"
        })
    
    volume_details_df = pd.DataFrame(volume_table)
    volume_details_df = volume_details_df.sort_values('Avg Volume', ascending=False)
    
    st.dataframe(volume_details_df, use_container_width=True, hide_index=True)

def display_liquidity_alerts(portfolio_df, liquidity_data):
    """Display liquidity risk alerts and recommendations."""
    
    st.subheader("Liquidity Risk Alerts")
    
    # Define liquidity risk thresholds
    liquidity_thresholds = {
        'min_liquidity_score': 0.6,      # Minimum acceptable liquidity score
        'max_spread_pct': 0.5,           # Maximum acceptable spread
        'min_avg_volume': 1000000,       # Minimum average volume (1M shares)
        'max_position_liquidity_risk': 0.3  # Maximum position with low liquidity
    }
    
    # Calculate portfolio metrics
    total_weight = 0
    low_liquidity_weight = 0
    wide_spread_weight = 0
    low_volume_weight = 0
    
    for ticker, data in liquidity_data.items():
        weight = data.get('weight', 0)
        liquidity_score = data.get('combined_score', 0)
        spread_pct = data.get('spread_pct', 0)
        avg_volume = data.get('avg_volume', 0)
        
        total_weight += weight
        
        if liquidity_score < liquidity_thresholds['min_liquidity_score']:
            low_liquidity_weight += weight
        
        if spread_pct > liquidity_thresholds['max_spread_pct']:
            wide_spread_weight += weight
        
        if avg_volume < liquidity_thresholds['min_avg_volume']:
            low_volume_weight += weight
    
    # Generate alerts
    alerts = []
    
    if low_liquidity_weight > liquidity_thresholds['max_position_liquidity_risk']:
        alerts.append({
            'type': 'High',
            'message': f"Low liquidity positions ({low_liquidity_weight:.1%}) exceed {liquidity_thresholds['max_position_liquidity_risk']:.0%} threshold",
            'recommendation': "Consider reducing positions with low liquidity scores"
        })
    
    if wide_spread_weight > liquidity_thresholds['max_position_liquidity_risk']:
        alerts.append({
            'type': 'Medium',
            'message': f"Wide spread positions ({wide_spread_weight:.1%}) exceed {liquidity_thresholds['max_position_liquidity_risk']:.0%} threshold",
            'recommendation': "Monitor bid-ask spreads and consider alternatives"
        })
    
    if low_volume_weight > liquidity_thresholds['max_position_liquidity_risk']:
        alerts.append({
            'type': 'Medium',
            'message': f"Low volume positions ({low_volume_weight:.1%}) exceed {liquidity_thresholds['max_position_liquidity_risk']:.0%} threshold",
            'recommendation': "Consider positions with higher trading volume"
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
        st.success("✅ No liquidity risk alerts - portfolio appears liquid")
    
    # Display liquidity risk metrics
    st.subheader("Liquidity Risk Metrics")
    
    risk_metrics_df = pd.DataFrame([
        {
            'Metric': 'Low Liquidity Positions',
            'Weight': f"{low_liquidity_weight:.1%}",
            'Threshold': f"{liquidity_thresholds['max_position_liquidity_risk']:.0%}",
            'Status': '⚠️ Exceeds' if low_liquidity_weight > liquidity_thresholds['max_position_liquidity_risk'] else '✅ OK'
        },
        {
            'Metric': 'Wide Spread Positions',
            'Weight': f"{wide_spread_weight:.1%}",
            'Threshold': f"{liquidity_thresholds['max_position_liquidity_risk']:.0%}",
            'Status': '⚠️ Exceeds' if wide_spread_weight > liquidity_thresholds['max_position_liquidity_risk'] else '✅ OK'
        },
        {
            'Metric': 'Low Volume Positions',
            'Weight': f"{low_volume_weight:.1%}",
            'Threshold': f"{liquidity_thresholds['max_position_liquidity_risk']:.0%}",
            'Status': '⚠️ Exceeds' if low_volume_weight > liquidity_thresholds['max_position_liquidity_risk'] else '✅ OK'
        }
    ])
    
    st.dataframe(risk_metrics_df, use_container_width=True, hide_index=True)
    
    # Liquidity stress testing
    st.subheader("Liquidity Stress Testing")
    
    # Estimate liquidation time under different scenarios
    normal_time = estimate_liquidation_time(portfolio_df, liquidity_data, stress_factor=1.0)
    stress_time = estimate_liquidation_time(portfolio_df, liquidity_data, stress_factor=0.5)
    crisis_time = estimate_liquidation_time(portfolio_df, liquidity_data, stress_factor=0.2)
    
    def format_liquidation_time(time_days):
        """Format liquidation time in the most appropriate unit."""
        if time_days < 1/24/60:  # Less than 1 minute
            return f"{time_days * 24 * 60 * 60:.0f} seconds"
        elif time_days < 1/24:  # Less than 1 hour
            return f"{time_days * 24 * 60:.0f} minutes"
        elif time_days < 1:  # Less than 1 day
            return f"{time_days * 24:.1f} hours"
        else:
            return f"{time_days:.1f} days"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Normal Market", format_liquidation_time(normal_time))
    
    with col2:
        st.metric("Stress Market", format_liquidation_time(stress_time))
    
    with col3:
        st.metric("Crisis Market", format_liquidation_time(crisis_time))

def estimate_liquidation_time(portfolio_df, liquidity_data, stress_factor=1.0):
    """Estimate time to liquidate portfolio under stress conditions."""
    
    max_days = 0
    
    for ticker, data in liquidity_data.items():
        avg_volume = data.get('avg_volume', 0)
        market_value = data.get('market_value', 0)
        mid_price = data.get('mid_price', 1)
        
        if mid_price > 0 and avg_volume > 0:
            # Calculate position shares
            position_shares = market_value / mid_price
            
            # Assume we can trade 10% of average volume without significant impact
            daily_capacity = avg_volume * 0.1 * stress_factor
            
            # Estimate days to liquidate this position
            days_to_liquidate = position_shares / daily_capacity
            max_days = max(max_days, days_to_liquidate)
    
    return max_days 