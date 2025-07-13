"""
streamlit_dashboard.py - M.A.N.T.R.A. Main Dashboard
===================================================
The main entry point for the M.A.N.T.R.A. trading intelligence system
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple

# Import all M.A.N.T.R.A. modules
from core_system_foundation import load_data, refresh_data, get_data_summary
from signal_engine import calculate_signals
from decision_engine import make_trading_decisions, get_buy_recommendations
from anomaly_detector import detect_anomalies, get_critical_anomalies
from edge_finder import find_edges, get_stocks_with_edge
from sector_rotation_engine import analyze_sector_rotation, get_sector_leaders
from watchlist_builder import build_watchlists, get_top_opportunities
from alert_engine import generate_alerts, get_critical_alerts, format_alert_summary
from filters import apply_filters, create_streamlit_filters, get_filter_preset
from regime_shifter import detect_market_regime, get_regime_profile
from health_panel import check_system_health, render_health_panel
from ui_components import *

# Import constants
from constants import SIGNAL_LEVELS, FACTOR_WEIGHTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Initialize UI
initialize_ui()

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'stocks_df' not in st.session_state:
        st.session_state.stocks_df = pd.DataFrame()
    if 'sector_df' not in st.session_state:
        st.session_state.sector_df = pd.DataFrame()
    if 'analyzed_df' not in st.session_state:
        st.session_state.analyzed_df = pd.DataFrame()
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Overview"
    if 'filter_values' not in st.session_state:
        st.session_state.filter_values = {}
    if 'user_positions' not in st.session_state:
        st.session_state.user_positions = {}

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_process_data():
    """Load and process all data"""
    try:
        # Start timing
        start_time = time.time()
        
        # Load raw data
        stocks_df, sector_df, health = load_data()
        
        if stocks_df.empty:
            return None, None, None, health
        
        # Calculate signals
        stocks_df = calculate_signals(stocks_df, sector_df)
        
        # Make decisions
        stocks_df = make_trading_decisions(stocks_df)
        
        # Detect anomalies
        stocks_df = detect_anomalies(stocks_df)
        
        # Find edges
        stocks_df = find_edges(stocks_df, sector_df)
        
        # Analyze sectors
        stocks_df, sector_df, sector_analysis = analyze_sector_rotation(stocks_df, sector_df)
        
        # Processing time
        processing_time = time.time() - start_time
        health['processing_times'] = {
            'total_processing': processing_time,
            'data_load': health.get('processing_time_s', 0)
        }
        
        return stocks_df, sector_df, sector_analysis, health
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        st.error(f"Failed to load data: {str(e)}")
        return None, None, None, {'status': 'error', 'error': str(e)}

# ============================================================================
# MAIN DASHBOARD LAYOUT
# ============================================================================

def main():
    """Main dashboard function"""
    # Initialize session state
    initialize_session_state()
    
    # Create header
    market_regime, regime_confidence, regime_analysis = None, 0, {}
    
    if st.session_state.data_loaded:
        market_regime, regime_confidence, regime_analysis = detect_market_regime(
            st.session_state.stocks_df, 
            st.session_state.sector_df
        )
    
    dashboard_header(
        market_status={
            'regime': market_regime.value if market_regime else 'Loading...',
            'confidence': regime_confidence
        }
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        # Data refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.session_state.data_loaded = False
                st.rerun()
        
        with col2:
            if st.session_state.last_refresh:
                time_diff = datetime.now() - st.session_state.last_refresh
                st.caption(f"Updated {time_diff.seconds // 60}m ago")
        
        # Navigation
        st.markdown("---")
        tabs = ["Overview", "Signals", "Opportunities", "Sectors", "Alerts", "Analysis", "Portfolio"]
        selected_tab = st.radio("Navigation", tabs, index=tabs.index(st.session_state.selected_tab))
        st.session_state.selected_tab = selected_tab
        
        # Filters
        st.markdown("---")
        if st.session_state.data_loaded:
            st.session_state.filter_values = create_streamlit_filters(
                st, 
                st.session_state.analyzed_df
            )
        
        # Health Panel
        st.markdown("---")
        if st.session_state.data_loaded:
            health_report = check_system_health(
                st.session_state.stocks_df,
                st.session_state.sector_df,
                processing_times=st.session_state.get('processing_times', {})
            )
            render_health_panel(st, health_report)
    
    # Load data if not loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading market data..."):
            stocks_df, sector_df, sector_analysis, health = load_and_process_data()
            
            if stocks_df is not None:
                st.session_state.stocks_df = stocks_df
                st.session_state.sector_df = sector_df
                st.session_state.analyzed_df = stocks_df.copy()
                st.session_state.sector_analysis = sector_analysis
                st.session_state.data_loaded = True
                st.session_state.last_refresh = datetime.now()
                st.session_state.processing_times = health.get('processing_times', {})
                
                # Generate alerts
                st.session_state.alerts = generate_alerts(stocks_df)
                
                st.success("Data loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load data. Please check your connection and try again.")
                return
    
    # Apply filters
    filtered_df = apply_filters(
        st.session_state.analyzed_df,
        **st.session_state.filter_values
    )
    
    # Display selected tab
    if selected_tab == "Overview":
        show_overview_tab(filtered_df, st.session_state.sector_df, regime_analysis)
    elif selected_tab == "Signals":
        show_signals_tab(filtered_df)
    elif selected_tab == "Opportunities":
        show_opportunities_tab(filtered_df, st.session_state.sector_df)
    elif selected_tab == "Sectors":
        show_sectors_tab(st.session_state.sector_df, filtered_df, st.session_state.sector_analysis)
    elif selected_tab == "Alerts":
        show_alerts_tab(st.session_state.alerts, filtered_df)
    elif selected_tab == "Analysis":
        show_analysis_tab(filtered_df, st.session_state.sector_df, regime_analysis)
    elif selected_tab == "Portfolio":
        show_portfolio_tab(filtered_df)

# ============================================================================
# TAB IMPLEMENTATIONS
# ============================================================================

def show_overview_tab(df: pd.DataFrame, sector_df: pd.DataFrame, regime_analysis: Dict):
    """Display overview dashboard"""
    section_header("Market Overview", "Real-time market intelligence and opportunities")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_stocks = len(df)
        metric_card("Total Stocks", total_stocks, icon="📊")
    
    with col2:
        buy_signals = len(df[df['decision'] == 'BUY'])
        buy_pct = (buy_signals / total_stocks * 100) if total_stocks > 0 else 0
        metric_card("Buy Signals", buy_signals, delta=buy_pct, icon="🟢")
    
    with col3:
        avg_score = df['composite_score'].mean() if 'composite_score' in df.columns else 0
        metric_card("Avg Score", f"{avg_score:.1f}", icon="📈")
    
    with col4:
        critical_alerts = len([a for a in st.session_state.alerts if a.priority.value <= 2])
        metric_card("Alerts", critical_alerts, icon="🔔")
    
    with col5:
        market_breadth = (df['ret_1d'] > 0).sum() / len(df) * 100 if 'ret_1d' in df.columns else 50
        metric_card("Breadth", f"{market_breadth:.0f}%", icon="📊")
    
    # Market regime info
    if regime_analysis:
        st.markdown("---")
        regime_profile = get_regime_profile(regime_analysis.get('regime'))
        if regime_profile:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### 🎯 {regime_profile.name}")
                st.markdown(f"*{regime_profile.description}*")
                st.markdown(f"**Risk Level:** {regime_profile.risk_level}")
                st.markdown(f"**Typical Duration:** {regime_profile.typical_duration}")
            
            with col2:
                st.markdown("**Recommended Strategies:**")
                for strategy in regime_profile.recommended_strategies[:3]:
                    st.markdown(f"• {strategy}")
    
    # Top opportunities
    st.markdown("---")
    section_header("Top Opportunities", "Highest conviction trades right now")
    
    top_buys = get_buy_recommendations(df, min_confidence=75, max_risk=60).head(6)
    
    if not top_buys.empty:
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(top_buys.iterrows()):
            with cols[idx % 3]:
                st.markdown(stock_card(stock), unsafe_allow_html=True)
    else:
        st.info("No high-conviction opportunities found with current filters")
    
    # Sector performance
    st.markdown("---")
    section_header("Sector Performance", "Market rotation and sector strength")
    
    if not sector_df.empty:
        fig = sector_heatmap(sector_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts
    if st.session_state.alerts:
        st.markdown("---")
        section_header("Recent Alerts", "Important market events")
        
        recent_alerts = st.session_state.alerts[:5]
        for alert in recent_alerts:
            alert_banner(
                alert.title,
                alert_type=alert.priority.name.lower()
            )

def show_signals_tab(df: pd.DataFrame):
    """Display signals and recommendations"""
    section_header("Trading Signals", f"Found {len(df)} stocks matching your criteria")
    
    # Signal distribution
    col1, col2, col3, col4 = st.columns(4)
    
    signal_counts = df['decision'].value_counts()
    with col1:
        metric_card("BUY", signal_counts.get('BUY', 0), icon="🟢")
    with col2:
        metric_card("WATCH", signal_counts.get('WATCH', 0), icon="🟡")
    with col3:
        metric_card("NEUTRAL", signal_counts.get('NEUTRAL', 0), icon="⚪")
    with col4:
        metric_card("AVOID", signal_counts.get('AVOID', 0), icon="🔴")
    
    # Filter by signal type
    signal_filter = st.selectbox(
        "Filter by Signal",
        ["All"] + list(df['decision'].unique()),
        key="signal_filter"
    )
    
    if signal_filter != "All":
        display_df = df[df['decision'] == signal_filter]
    else:
        display_df = df
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ['opportunity_score', 'composite_score', 'ret_30d', 'volume_1d', 'market_cap'],
        key="sort_by"
    )
    
    display_df = display_df.sort_values(sort_by, ascending=False)
    
    # Display table
    if not display_df.empty:
        # Select columns to display
        display_columns = [
            'ticker', 'company_name', 'decision', 'composite_score', 
            'price', 'ret_1d', 'ret_30d', 'target_price', 'stop_loss',
            'risk_level', 'reasoning'
        ]
        
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        styled_dataframe(
            display_df[available_columns].head(50),
            highlight_column='composite_score',
            color_map={'decision': SIGNAL_COLORS['BUY']}
        )
        
        # Download button
        csv = display_df[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Signals",
            data=csv,
            file_name=f"mantra_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No stocks found matching the criteria")

def show_opportunities_tab(df: pd.DataFrame, sector_df: pd.DataFrame):
    """Display special opportunities and edges"""
    section_header("Special Opportunities", "High-probability setups and edge trades")
    
    # Build watchlists
    watchlists = build_watchlists(df)
    
    # Watchlist selector
    watchlist_names = list(watchlists.keys())
    selected_watchlist = st.selectbox(
        "Select Watchlist",
        watchlist_names,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_watchlist and selected_watchlist in watchlists:
        watchlist_df = watchlists[selected_watchlist]
        
        if not watchlist_df.empty:
            st.markdown(f"**{len(watchlist_df)} stocks** in {selected_watchlist.replace('_', ' ').title()}")
            
            # Display as cards or table
            view_mode = st.radio("View as", ["Cards", "Table"], horizontal=True)
            
            if view_mode == "Cards":
                cols = st.columns(3)
                for idx, (_, stock) in enumerate(watchlist_df.head(12).iterrows()):
                    with cols[idx % 3]:
                        st.markdown(stock_card(stock), unsafe_allow_html=True)
            else:
                display_columns = [
                    'ticker', 'company_name', 'decision', 'composite_score',
                    'price', 'ret_1d', 'ret_30d', 'edge_score', 'best_edge'
                ]
                available_columns = [col for col in display_columns if col in watchlist_df.columns]
                
                styled_dataframe(
                    watchlist_df[available_columns],
                    highlight_column='composite_score'
                )
        else:
            st.info("No stocks in this watchlist")
    
    # Edge opportunities
    st.markdown("---")
    section_header("Edge Setups", "Statistical edge opportunities")
    
    edge_stocks = get_stocks_with_edge(df, min_edge_score=15).head(10)
    
    if not edge_stocks.empty:
        for _, stock in edge_stocks.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {stock['ticker']} - {stock.get('company_name', 'N/A')}")
                st.markdown(f"**Edge:** {stock.get('best_edge', 'N/A')}")
                st.markdown(f"**Expected Return:** {stock.get('edge_expected_return', 0):.1f}%")
            
            with col2:
                st.markdown(f"**Holding Period:** {stock.get('edge_holding_period', 'N/A')}")
                st.markdown(f"**Risk Level:** {stock.get('edge_risk_level', 'N/A')}")
            
            with col3:
                st.markdown(signal_badge(stock.get('decision', 'NEUTRAL'), stock.get('composite_score', 0)))
            
            st.markdown("---")
    else:
        st.info("No edge setups found with current filters")

def show_sectors_tab(sector_df: pd.DataFrame, stocks_df: pd.DataFrame, sector_analysis: Dict):
    """Display sector analysis and rotation"""
    section_header("Sector Analysis", "Sector rotation and performance insights")
    
    if sector_analysis:
        # Rotation signals
        col1, col2, col3, col4 = st.columns(4)
        
        rotation_signals = sector_analysis.get('rotation_signals', {})
        
        with col1:
            rotate_into = rotation_signals.get('rotate_into', [])
            st.markdown("### 🟢 Rotate Into")
            for sector_info in rotate_into[:3]:
                st.markdown(f"**{sector_info['sector']}**")
                st.caption(f"Score: {sector_info['momentum_score']}")
        
        with col2:
            accumulate = rotation_signals.get('accumulate', [])
            st.markdown("### 🟡 Accumulate")
            for sector_info in accumulate[:3]:
                st.markdown(f"**{sector_info['sector']}**")
                st.caption(f"Score: {sector_info['momentum_score']}")
        
        with col3:
            avoid = rotation_signals.get('avoid', [])
            st.markdown("### 🔴 Avoid")
            for sector_info in avoid[:3]:
                st.markdown(f"**{sector_info['sector']}**")
                st.caption(f"Score: {sector_info['momentum_score']}")
        
        with col4:
            st.markdown("### 📊 Stats")
            stats = sector_analysis.get('statistics', {})
            st.metric("Avg Return", f"{stats.get('average_return_30d', 0):.1f}%")
            st.metric("Best Sector", stats.get('best_performer', 'N/A'))
    
    # Sector performance table
    st.markdown("---")
    
    if not sector_df.empty:
        # Heatmap
        fig = sector_heatmap(sector_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📊 Detailed Sector Metrics"):
            display_columns = [
                'sector', 'sector_ret_30d', 'momentum_score', 
                'rotation_signal', 'sector_category', 'sector_count'
            ]
            available_columns = [col for col in display_columns if col in sector_df.columns]
            
            styled_dataframe(
                sector_df[available_columns].sort_values('momentum_score', ascending=False),
                highlight_column='momentum_score'
            )
    
    # Sector leaders
    st.markdown("---")
    section_header("Sector Leaders", "Top stocks in each sector")
    
    selected_sector = st.selectbox(
        "Select Sector",
        sorted(stocks_df['sector'].unique()) if 'sector' in stocks_df.columns else []
    )
    
    if selected_sector:
        sector_stocks = stocks_df[stocks_df['sector'] == selected_sector]
        sector_leaders = sector_stocks.nlargest(10, 'composite_score')
        
        if not sector_leaders.empty:
            display_columns = [
                'ticker', 'company_name', 'decision', 'composite_score',
                'price', 'ret_1d', 'ret_30d', 'sector_relative_performance'
            ]
            available_columns = [col for col in display_columns if col in sector_leaders.columns]
            
            styled_dataframe(
                sector_leaders[available_columns],
                highlight_column='composite_score'
            )
        else:
            st.info("No stocks found in this sector")

def show_alerts_tab(alerts: List, df: pd.DataFrame):
    """Display alerts and notifications"""
    section_header("Market Alerts", f"{len(alerts)} active alerts")
    
    if not alerts:
        st.info("No active alerts at this time")
        return
    
    # Alert summary
    critical_alerts = get_critical_alerts(alerts)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Critical", len([a for a in alerts if a.priority.value == 1]), icon="🚨")
    with col2:
        metric_card("High", len([a for a in alerts if a.priority.value == 2]), icon="⚠️")
    with col3:
        metric_card("Medium", len([a for a in alerts if a.priority.value == 3]), icon="ℹ️")
    with col4:
        metric_card("Action Required", len([a for a in alerts if a.action_required]), icon="📌")
    
    # Filter alerts
    priority_filter = st.multiselect(
        "Filter by Priority",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"]
    )
    
    # Display alerts
    filtered_alerts = [
        a for a in alerts 
        if a.priority.name.title() in priority_filter
    ]
    
    for alert in filtered_alerts[:20]:
        alert_banner(
            f"{alert.title} - {alert.message}",
            alert_type=alert.priority.name.lower()
        )
    
    # Alert details table
    with st.expander("📋 Alert Details"):
        alert_summary = format_alert_summary(filtered_alerts)
        if not alert_summary.empty:
            st.dataframe(alert_summary, use_container_width=True)

def show_analysis_tab(df: pd.DataFrame, sector_df: pd.DataFrame, regime_analysis: Dict):
    """Display detailed analysis and insights"""
    section_header("Market Analysis", "Deep insights and patterns")
    
    # Market regime details
    if regime_analysis:
        st.markdown("### 🎯 Market Regime Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Indicators:**")
            indicators = regime_analysis.get('indicators', {})
            for key, value in indicators.items():
                st.markdown(f"• {key.replace('_', ' ').title()}: {value}")
        
        with col2:
            st.markdown("**Regime Scores:**")
            scores = regime_analysis.get('regime_scores', {})
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for regime, score in sorted_scores[:5]:
                st.markdown(f"• {regime.replace('_', ' ').title()}: {score:.0f}")
    
    # Factor analysis
    st.markdown("---")
    st.markdown("### 📊 Factor Performance")
    
    factor_cols = [col for col in df.columns if col.endswith('_score')]
    if factor_cols:
        # Calculate average scores
        factor_avgs = {}
        for col in factor_cols:
            factor_name = col.replace('_score', '').replace('_', ' ').title()
            factor_avgs[factor_name] = df[col].mean()
        
        # Display as bar chart
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(factor_avgs.keys()),
                y=list(factor_avgs.values()),
                marker_color=['#00d26a' if v > 60 else '#ffa500' if v > 40 else '#ff4b4b' 
                              for v in factor_avgs.values()]
            )
        ])
        
        fig.update_layout(
            title="Average Factor Scores",
            xaxis_title="Factor",
            yaxis_title="Average Score",
            template='plotly_dark',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly analysis
    st.markdown("---")
    st.markdown("### 🔍 Anomaly Detection")
    
    anomaly_stocks = df[df['anomaly_count'] > 0].nlargest(10, 'anomaly_count')
    
    if not anomaly_stocks.empty:
        for _, stock in anomaly_stocks.iterrows():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{stock['ticker']}** - {stock.get('company_name', 'N/A')}")
                st.caption(f"Anomalies: {stock.get('anomaly_summary', 'N/A')}")
            
            with col2:
                st.markdown(f"Priority: **{stock.get('anomaly_priority', 'N/A')}**")
    else:
        st.info("No significant anomalies detected")
    
    # Distribution analysis
    st.markdown("---")
    st.markdown("### 📈 Score Distribution")
    
    if 'composite_score' in df.columns:
        # Create histogram
        fig = go.Figure(data=[
            go.Histogram(
                x=df['composite_score'],
                nbinsx=20,
                marker_color='#00d26a',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Composite Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            template='plotly_dark',
            showlegend=False,
            height=300
        )
        
        # Add vertical lines for thresholds
        fig.add_vline(x=SIGNAL_LEVELS['BUY'], line_dash="dash", line_color="green", annotation_text="BUY")
        fig.add_vline(x=SIGNAL_LEVELS['WATCH'], line_dash="dash", line_color="orange", annotation_text="WATCH")
        fig.add_vline(x=SIGNAL_LEVELS['AVOID'], line_dash="dash", line_color="red", annotation_text="AVOID")
        
        st.plotly_chart(fig, use_container_width=True)

def show_portfolio_tab(df: pd.DataFrame):
    """Display portfolio tracking and analysis"""
    section_header("Portfolio Tracker", "Monitor your positions and P&L")
    
    # Portfolio input
    st.markdown("### 📝 Add Position")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ticker = st.selectbox(
            "Ticker",
            sorted(df['ticker'].unique()) if 'ticker' in df.columns else [],
            key="portfolio_ticker"
        )
    
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=100, key="portfolio_qty")
    
    with col3:
        buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0, key="portfolio_price")
    
    with col4:
        if st.button("Add Position", use_container_width=True):
            if ticker:
                if 'positions' not in st.session_state:
                    st.session_state.positions = {}
                
                st.session_state.positions[ticker] = {
                    'quantity': quantity,
                    'buy_price': buy_price,
                    'buy_date': datetime.now()
                }
                st.success(f"Added {ticker} to portfolio")
    
    # Display positions
    if hasattr(st.session_state, 'positions') and st.session_state.positions:
        st.markdown("---")
        st.markdown("### 💼 Current Positions")
        
        positions_data = []
        total_investment = 0
        total_current_value = 0
        
        for ticker, position in st.session_state.positions.items():
            stock_data = df[df['ticker'] == ticker].iloc[0] if ticker in df['ticker'].values else None
            
            if stock_data is not None:
                current_price = stock_data.get('price', position['buy_price'])
                quantity = position['quantity']
                buy_price = position['buy_price']
                
                investment = quantity * buy_price
                current_value = quantity * current_price
                pnl = current_value - investment
                pnl_pct = (pnl / investment) * 100 if investment > 0 else 0
                
                total_investment += investment
                total_current_value += current_value
                
                positions_data.append({
                    'Ticker': ticker,
                    'Quantity': quantity,
                    'Buy Price': f"₹{buy_price:,.2f}",
                    'Current Price': f"₹{current_price:,.2f}",
                    'Investment': f"₹{investment:,.0f}",
                    'Current Value': f"₹{current_value:,.0f}",
                    'P&L': f"₹{pnl:+,.0f}",
                    'P&L %': f"{pnl_pct:+.1f}%",
                    'Signal': stock_data.get('decision', 'N/A')
                })
        
        # Summary metrics
        total_pnl = total_current_value - total_investment
        total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card("Total Investment", f"₹{total_investment:,.0f}", icon="💰")
        
        with col2:
            metric_card("Current Value", f"₹{total_current_value:,.0f}", icon="💎")
        
        with col3:
            metric_card("Total P&L", f"₹{total_pnl:+,.0f}", delta=total_pnl_pct, icon="📈")
        
        with col4:
            metric_card("Positions", len(st.session_state.positions), icon="📊")
        
        # Positions table
        if positions_data:
            positions_df = pd.DataFrame(positions_data)
            
            # Style the dataframe
            def color_pnl(val):
                if isinstance(val, str) and '+' in val:
                    return 'color: #00d26a'
                elif isinstance(val, str) and '-' in val:
                    return 'color: #ff4b4b'
                return ''
            
            styled_df = positions_df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Clear portfolio button
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.positions = {}
                st.rerun()
    else:
        st.info("No positions in portfolio. Add stocks above to start tracking.")

# ============================================================================
# RUN DASHBOARD
# ============================================================================

if __name__ == "__main__":
    main()
