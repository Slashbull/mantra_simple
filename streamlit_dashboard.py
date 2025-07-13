"""
streamlit_dashboard.py - M.A.N.T.R.A. Main Dashboard
==================================================
The main application entry point
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import all components
from constants import *
from data_loader import DataLoader
from signal_engine import SignalEngine
from ui_components import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="M.A.N.T.R.A. - Stock Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "📊 Overview"

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
def load_and_analyze_data():
    """Load data and perform analysis"""
    # Load data
    stocks_df, sector_df, health = DataLoader.load_all_data()
    
    if health['status'] != 'success':
        return None, None, None, health
    
    # Calculate signals
    analyzed_df = SignalEngine.calculate_all_signals(stocks_df, sector_df)
    
    return analyzed_df, sector_df, health, None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Refresh button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.session_state.last_refresh:
            mins_ago = (datetime.now() - st.session_state.last_refresh).seconds // 60
            st.caption(f"↻ {mins_ago}m ago")
    
    st.markdown("---")
    
    # Navigation
    tabs = ["📊 Overview", "🎯 Signals", "🔥 Top Picks", "📈 Sectors", "📋 Analysis"]
    st.session_state.selected_tab = st.radio("Navigation", tabs, label_visibility="collapsed")
    
    # Filters
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    
    # Signal filter
    signal_filter = st.multiselect(
        "Signal Type",
        options=['BUY', 'WATCH', 'NEUTRAL', 'AVOID'],
        default=['BUY', 'WATCH']
    )
    
    # Score filter
    min_score = st.slider(
        "Min Score",
        min_value=0,
        max_value=100,
        value=65,
        step=5
    )
    
    # Risk filter
    risk_filter = st.multiselect(
        "Risk Level",
        options=['LOW', 'MEDIUM', 'HIGH'],
        default=['LOW', 'MEDIUM']
    )
    
    # Volume filter
    min_volume = st.number_input(
        "Min Volume",
        min_value=0,
        value=50000,
        step=10000,
        format="%d"
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
dashboard_header()

# Load data
if not st.session_state.data_loaded:
    with st.spinner("🔄 Loading market data..."):
        data = load_and_analyze_data()
        if data[0] is not None:
            st.session_state.stocks_df = data[0]
            st.session_state.sector_df = data[1]
            st.session_state.health = data[2]
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
        else:
            st.error("❌ Failed to load data. Please check your connection and try again.")
            st.stop()

# Apply filters
filtered_df = st.session_state.stocks_df.copy()

# Apply signal filter
if signal_filter:
    filtered_df = filtered_df[filtered_df['decision'].isin(signal_filter)]

# Apply score filter
filtered_df = filtered_df[filtered_df['composite_score'] >= min_score]

# Apply risk filter
if risk_filter:
    filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]

# Apply volume filter
if 'volume_1d' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['volume_1d'] >= min_volume]

# Data quality indicator
if 'data_quality' in st.session_state.health:
    data_quality_badge(st.session_state.health['data_quality'])

# ============================================================================
# TAB CONTENT
# ============================================================================

if st.session_state.selected_tab == "📊 Overview":
    # Market Overview
    section_header("Market Overview", f"Analyzing {len(filtered_df)} stocks")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_stocks = len(st.session_state.stocks_df)
        metric_card("Total Stocks", total_stocks, icon="📊")
    
    with col2:
        buy_signals = len(filtered_df[filtered_df['decision'] == 'BUY'])
        buy_pct = (buy_signals / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        metric_card("Buy Signals", buy_signals, delta=buy_pct, icon="🟢")
    
    with col3:
        avg_score = filtered_df['composite_score'].mean()
        metric_card("Avg Score", f"{avg_score:.1f}", icon="📈")
    
    with col4:
        high_momentum = len(filtered_df[filtered_df['momentum_score'] > 80])
        metric_card("High Momentum", high_momentum, icon="🚀")
    
    with col5:
        market_breadth = (filtered_df['ret_1d'] > 0).sum() / len(filtered_df) * 100
        metric_card("Breadth", f"{market_breadth:.0f}%", icon="📊")
    
    # Top opportunities
    st.markdown("---")
    section_header("🎯 Top Opportunities", "Highest conviction trades")
    
    top_buys = filtered_df[filtered_df['decision'] == 'BUY'].nlargest(6, 'opportunity_score')
    
    if not top_buys.empty:
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(top_buys.iterrows()):
            with cols[idx % 3]:
                stock_card(stock)
    else:
        show_alert("No buy opportunities found with current filters", "info")
    
    # Sector performance
    st.markdown("---")
    section_header("🏭 Sector Performance", "Heat map of sector returns")
    
    if not st.session_state.sector_df.empty:
        fig = sector_heatmap(st.session_state.sector_df)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.selected_tab == "🎯 Signals":
    section_header("Trading Signals", f"Found {len(filtered_df)} stocks matching criteria")
    
    # Signal summary
    col1, col2, col3, col4 = st.columns(4)
    
    signal_counts = filtered_df['decision'].value_counts()
    with col1:
        metric_card("BUY", signal_counts.get('BUY', 0), icon="🟢")
    with col2:
        metric_card("WATCH", signal_counts.get('WATCH', 0), icon="🟡")
    with col3:
        metric_card("NEUTRAL", signal_counts.get('NEUTRAL', 0), icon="⚪")
    with col4:
        metric_card("AVOID", signal_counts.get('AVOID', 0), icon="🔴")
    
    # Detailed table
    st.markdown("---")
    
    # Select columns to display
    display_columns = [
        'ticker', 'company_name', 'sector', 'decision', 'composite_score',
        'price', 'ret_1d', 'ret_30d', 'pe', 'volume_1d',
        'momentum_score', 'value_score', 'risk_level', 'reasoning'
    ]
    
    # Filter available columns
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    # Sort by opportunity score
    display_df = filtered_df.nlargest(100, 'opportunity_score')[available_columns]
    
    display_dataframe(display_df, "Signal Details", height=600)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Signals CSV",
        data=csv,
        file_name=f"mantra_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

elif st.session_state.selected_tab == "🔥 Top Picks":
    section_header("Top Stock Picks", "Best opportunities by category")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Momentum", "💎 Value", "📈 Growth", "🛡️ Safe"])
    
    with tab1:
        # Momentum picks
        momentum_picks = filtered_df[
            (filtered_df['momentum_score'] > 80) & 
            (filtered_df['decision'].isin(['BUY', 'WATCH']))
        ].nlargest(12, 'momentum_score')
        
        if not momentum_picks.empty:
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(momentum_picks.iterrows()):
                with cols[idx % 3]:
                    stock_card(stock)
        else:
            show_alert("No momentum picks found", "info")
    
    with tab2:
        # Value picks
        value_picks = filtered_df[
            (filtered_df['value_score'] > 80) & 
            (filtered_df['pe'] > 0) & 
            (filtered_df['pe'] < 20)
        ].nlargest(12, 'value_score')
        
        if not value_picks.empty:
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(value_picks.iterrows()):
                with cols[idx % 3]:
                    stock_card(stock)
        else:
            show_alert("No value picks found", "info")
    
    with tab3:
        # Growth picks
        growth_picks = filtered_df[
            (filtered_df['fundamental_score'] > 70) &
            (filtered_df['eps_change_pct'] > 20)
        ].nlargest(12, 'composite_score')
        
        if not growth_picks.empty:
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(growth_picks.iterrows()):
                with cols[idx % 3]:
                    stock_card(stock)
        else:
            show_alert("No growth picks found", "info")
    
    with tab4:
        # Safe picks (large cap, low risk)
        safe_picks = filtered_df[
            (filtered_df['risk_level'] == 'LOW') &
            (filtered_df['market_cap'] > 1e11)  # > 100B
        ].nlargest(12, 'composite_score')
        
        if not safe_picks.empty:
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(safe_picks.iterrows()):
                with cols[idx % 3]:
                    stock_card(stock)
        else:
            show_alert("No safe picks found", "info")

elif st.session_state.selected_tab == "📈 Sectors":
    section_header("Sector Analysis", "Performance and rotation insights")
    
    # Sector summary metrics
    if 'sector' in filtered_df.columns:
        sector_stats = filtered_df.groupby('sector').agg({
            'ticker': 'count',
            'ret_30d': 'mean',
            'composite_score': 'mean',
            'decision': lambda x: (x == 'BUY').sum()
        }).round(2)
        
        sector_stats.columns = ['Stocks', 'Avg Return 30D', 'Avg Score', 'Buy Signals']
        sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
        
        # Display sector table
        display_dataframe(sector_stats.reset_index(), "Sector Summary", height=400)
    
    # Sector heatmap
    st.markdown("---")
    if not st.session_state.sector_df.empty:
        fig = sector_heatmap(st.session_state.sector_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top stocks by sector
    st.markdown("---")
    selected_sector = st.selectbox(
        "Select Sector for Details",
        options=sorted(filtered_df['sector'].unique()) if 'sector' in filtered_df.columns else []
    )
    
    if selected_sector:
        sector_stocks = filtered_df[filtered_df['sector'] == selected_sector].nlargest(20, 'composite_score')
        
        display_columns = ['ticker', 'company_name', 'decision', 'composite_score', 
                          'price', 'ret_30d', 'pe', 'risk_level']
        available_columns = [col for col in display_columns if col in sector_stocks.columns]
        
        display_dataframe(sector_stocks[available_columns], f"Top Stocks in {selected_sector}", height=400)

elif st.session_state.selected_tab == "📋 Analysis":
    section_header("Market Analysis", "Deep dive into market metrics")
    
    # Score distributions
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_distribution_chart(filtered_df, 'composite_score', 'Composite Score Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = create_distribution_chart(filtered_df, 'momentum_score', 'Momentum Score Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Factor analysis
    st.markdown("---")
    section_header("Factor Performance", "Average scores by factor")
    
    factor_scores = {
        'Momentum': filtered_df['momentum_score'].mean(),
        'Value': filtered_df['value_score'].mean(),
        'Technical': filtered_df['technical_score'].mean(),
        'Volume': filtered_df['volume_score'].mean(),
        'Fundamental': filtered_df['fundamental_score'].mean()
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for idx, (factor, score) in enumerate(factor_scores.items()):
        with [col1, col2, col3, col4, col5][idx]:
            fig = create_gauge_chart(score, factor)
            st.plotly_chart(fig, use_container_width=True)
    
    # Market regime
    st.markdown("---")
    section_header("Market Regime", "Current market conditions")
    
    # Calculate market breadth
    breadth = (filtered_df['ret_1d'] > 0).sum() / len(filtered_df) * 100
    avg_return = filtered_df['ret_30d'].mean()
    
    # Determine regime
    if breadth > 70 and avg_return > 5:
        regime = "🐂 Bull Market"
        regime_desc = "Strong uptrend with broad participation"
    elif breadth < 30 and avg_return < -5:
        regime = "🐻 Bear Market"
        regime_desc = "Downtrend with widespread weakness"
    elif 45 <= breadth <= 55 and abs(avg_return) < 2:
        regime = "↔️ Sideways Market"
        regime_desc = "Range-bound with no clear direction"
    else:
        regime = "🔄 Transitional"
        regime_desc = "Market searching for direction"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### {regime}")
        st.markdown(f"*{regime_desc}*")
    
    with col2:
        metric_card("Market Breadth", f"{breadth:.0f}%", icon="📊")
    
    with col3:
        metric_card("Avg 30D Return", f"{avg_return:.1f}%", icon="📈")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #8b92a0; padding: 20px;">
        <p>🔱 M.A.N.T.R.A. - Market Analysis Neural Trading Research Assistant</p>
        <p style="font-size: 12px;">All signals are for educational purposes only. Always do your own research.</p>
    </div>
    """,
    unsafe_allow_html=True
)
