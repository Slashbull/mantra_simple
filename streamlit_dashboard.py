#!/usr/bin/env python3
"""
streamlit_dashboard.py - M.A.N.T.R.A. Elite Financial Intelligence Dashboard
Production-grade Streamlit dashboard for Indian stock market intelligence.
Built with world-class UX/UI patterns and bulletproof architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib

# Import core modules
from core_system_foundation import load_and_process
from signal_engine import run_signal_engine
from decision_engine import run_decision_engine
from anomaly_detector import run_anomaly_detector
from sector_mapper import run_sector_mapper
from watchlist_builder import build_watchlist
from edge_finder import compute_edge_signals, find_edges, edge_overview

# ================== PAGE CONFIGURATION ==================
st.set_page_config(
    page_title="M.A.N.T.R.A. Intelligence | Elite Stock Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "M.A.N.T.R.A. - Market Analysis & Trading Intelligence"
    }
)

# ================== CUSTOM CSS & THEMING ==================
def inject_custom_css():
    """Inject custom CSS for professional-grade UI/UX"""
    st.markdown("""
    <style>
    /* Professional color scheme */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
        --dark-bg: #0e1117;
        --light-bg: #ffffff;
    }
    
    /* Enhanced metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.2s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Professional dataframe styling */
    .dataframe {
        font-size: 14px;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(31,119,180,0.2) 0%, transparent 100%);
        border-bottom: 3px solid #1f77b4;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
    
    /* Floating action buttons */
    .floating-actions {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
    }
    
    /* Sparkline containers */
    .sparkline-container {
        display: inline-block;
        width: 100px;
        height: 30px;
        margin-left: 10px;
    }
    
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        text-align: center;
        transition: all 0.3s;
    }
    
    .kpi-card:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        div[data-testid="metric-container"] {
            padding: 15px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 14px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ================== CACHE & PERFORMANCE ==================
@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(regime: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load and process all data with caching for performance.
    TTL: 5 minutes to balance freshness with performance.
    """
    try:
        # Load base data
        stocks_df, sector_df, summary = load_and_process()
        
        # Run complete pipeline
        df = run_signal_engine(stocks_df, sector_df, regime=regime)
        df = run_decision_engine(df)
        df = run_anomaly_detector(df)
        df = compute_edge_signals(df)
        
        # Compute sector scores
        sector_scores = run_sector_mapper(sector_df)
        
        return df, sector_scores, summary
    except Exception as e:
        st.error(f"Data pipeline error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}

# ================== UI COMPONENTS ==================
class DashboardComponents:
    """Reusable UI components for consistent, professional design"""
    
    @staticmethod
    def render_kpi_card(title: str, value: str, delta: str = None, 
                       color: str = "blue", icon: str = "📊") -> None:
        """Render a beautiful KPI card with optional delta"""
        delta_html = f'<p style="color: {"#2ca02c" if delta and delta.startswith("+") else "#d62728"}; margin: 5px 0;">{delta}</p>' if delta else ''
        
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            <h3 style="margin: 0; color: white; font-size: 18px;">{icon} {title}</h3>
            <h1 style="margin: 10px 0; color: white; font-size: 36px;">{value}</h1>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_row(metrics: List[Tuple[str, any, any]], cols: int = 4) -> None:
        """Render a row of metrics with responsive columns"""
        columns = st.columns(cols)
        for i, (label, value, delta) in enumerate(metrics):
            with columns[i % cols]:
                st.metric(label=label, value=value, delta=delta)
    
    @staticmethod
    def create_sparkline(data: pd.Series, color: str = "#1f77b4") -> go.Figure:
        """Create a minimalist sparkline chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data, 
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba(31, 119, 180, 0.1)',
            showlegend=False
        ))
        fig.update_layout(
            height=40,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    @staticmethod
    def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str, 
                      value_col: str, title: str) -> go.Figure:
        """Create an interactive heatmap"""
        pivot = data.pivot(index=y_col, columns=x_col, values=value_col)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=pivot.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            xaxis_nticks=len(pivot.columns),
            yaxis_nticks=len(pivot.index),
            margin=dict(l=100, r=20, t=50, b=50)
        )
        return fig
    
    @staticmethod
    def create_gauge(value: float, title: str, max_value: float = 100) -> go.Figure:
        """Create a gauge chart for scores"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value*0.5], 'color': "lightgray"},
                    {'range': [max_value*0.5, max_value*0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value*0.9
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        return fig

# ================== DATA FILTERING ==================
class SmartFilter:
    """Advanced filtering logic with state management"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply multiple filters with bulletproof logic"""
        if df.empty:
            return df
        
        filtered = df.copy()
        
        # Tag filter
        if filters.get('tag') and 'tag' in filtered.columns:
            filtered = filtered[filtered['tag'] == filters['tag']]
        
        # Score filter
        if filters.get('min_score') is not None and 'final_score' in filtered.columns:
            filtered = filtered[filtered['final_score'] >= filters['min_score']]
        
        # Category filter
        if filters.get('category') != 'All' and 'category' in filtered.columns:
            filtered = filtered[filtered['category'] == filters['category']]
        
        # Sector filter
        if filters.get('sector') != 'All' and 'sector' in filtered.columns:
            filtered = filtered[filtered['sector'] == filters['sector']]
        
        # Search filter
        if filters.get('search'):
            search_term = filters['search'].upper()
            mask = pd.Series([False] * len(filtered))
            
            if 'ticker' in filtered.columns:
                mask |= filtered['ticker'].str.upper().str.contains(search_term, na=False)
            if 'company_name' in filtered.columns:
                mask |= filtered['company_name'].str.upper().str.contains(search_term, na=False)
            
            filtered = filtered[mask]
        
        # Edge filter
        if filters.get('has_edge') and 'has_edge' in filtered.columns:
            filtered = filtered[filtered['has_edge'] == True]
        
        # Anomaly filter
        if filters.get('anomalies_only') and 'anomaly' in filtered.columns:
            filtered = filtered[filtered['anomaly'] == True]
        
        return filtered

# ================== MAIN DASHBOARD ==================
def main():
    """Main dashboard application"""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize components
    components = DashboardComponents()
    
    # ========== HEADER ==========
    col1, col2, col3 = st.columns([2, 6, 2])
    with col2:
        st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; font-size: 48px; margin-bottom: 0;'>
        🚀 M.A.N.T.R.A. Intelligence
        </h1>
        <p style='text-align: center; color: #666; font-size: 18px; margin-top: 0;'>
        Elite Market Analysis & Trading Intelligence Platform
        </p>
        """, unsafe_allow_html=True)
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## 🎛️ Control Center")
        
        # Market Regime Selector
        st.markdown("### 📊 Market Regime")
        regime = st.selectbox(
            "Select Analysis Mode",
            options=["balanced", "momentum", "value", "growth", "volume"],
            index=0,
            help="Choose the market regime for adaptive analysis"
        )
        
        # Load data with selected regime
        with st.spinner("🔄 Loading market intelligence..."):
            df, sector_scores, summary = load_all_data(regime)
        
        # Data Health Monitor
        st.markdown("### 🏥 System Health")
        health_cols = st.columns(2)
        with health_cols[0]:
            st.metric("Stocks", f"{summary.get('total_stocks', 0):,}")
            st.metric("Sectors", summary.get('total_sectors', 0))
        with health_cols[1]:
            st.metric("Coverage", f"{(1 - summary.get('blank_cells', 0) / max(summary.get('total_cells', 1), 1)) * 100:.1f}%")
            st.metric("Quality", "✅ High" if summary.get('duplicates', 0) == 0 else "⚠️ Check")
        
        # Advanced Filters
        st.markdown("### 🔍 Smart Filters")
        
        filters = {}
        
        # Tag filter with color coding
        tag_options = ["Buy", "Watch", "Avoid"]
        tag_colors = {"Buy": "🟢", "Watch": "🟡", "Avoid": "🔴"}
        filters['tag'] = st.selectbox(
            "Signal Tag",
            options=tag_options,
            format_func=lambda x: f"{tag_colors.get(x, '')} {x}"
        )
        
        # Score filter with slider
        filters['min_score'] = st.slider(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Filter stocks by minimum final score"
        )
        
        # Category filter
        if 'category' in df.columns:
            categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            filters['category'] = st.selectbox("Category", categories)
        
        # Sector filter
        if 'sector' in df.columns:
            sectors = ['All'] + sorted(df['sector'].dropna().unique().tolist())
            filters['sector'] = st.selectbox("Sector", sectors)
        
        # Search box
        filters['search'] = st.text_input(
            "🔎 Search",
            placeholder="Ticker or Company name",
            help="Search by ticker symbol or company name"
        )
        
        # Advanced filters (collapsible)
        with st.expander("⚙️ Advanced Filters"):
            filters['has_edge'] = st.checkbox("Edge Signals Only", value=False)
            filters['anomalies_only'] = st.checkbox("Anomalies Only", value=False)
            
            # Volume filter
            if 'vol_ratio_1d_90d' in df.columns:
                vol_range = st.slider(
                    "Volume Ratio Range",
                    min_value=0.0,
                    max_value=5.0,
                    value=(0.5, 3.0),
                    step=0.1
                )
                filters['vol_min'], filters['vol_max'] = vol_range
        
        # Export options
        st.markdown("### 💾 Export Options")
        export_format = st.radio(
            "Format",
            options=["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = SmartFilter.apply_filters(df, filters)
    
    # ========== KEY METRICS OVERVIEW ==========
    st.markdown("## 📈 Market Overview")
    
    # Calculate key metrics
    total_stocks = len(df)
    filtered_stocks = len(filtered_df)
    buy_signals = len(filtered_df[filtered_df['tag'] == 'Buy']) if 'tag' in filtered_df.columns else 0
    avg_score = filtered_df['final_score'].mean() if 'final_score' in filtered_df.columns and not filtered_df.empty else 0
    anomaly_count = filtered_df['anomaly'].sum() if 'anomaly' in filtered_df.columns else 0
    edge_count = filtered_df['has_edge'].sum() if 'has_edge' in filtered_df.columns else 0
    
    # Display metrics
    metric_cols = st.columns(6)
    metrics_data = [
        ("📊 Total Stocks", f"{total_stocks:,}", None),
        ("🎯 Filtered", f"{filtered_stocks:,}", f"{filtered_stocks/total_stocks*100:.1f}%"),
        ("🟢 Buy Signals", buy_signals, f"{buy_signals/max(filtered_stocks,1)*100:.1f}%"),
        ("📈 Avg Score", f"{avg_score:.1f}", None),
        ("🚨 Anomalies", anomaly_count, None),
        ("⚡ Edge Signals", edge_count, None)
    ]
    
    for col, (label, value, delta) in zip(metric_cols, metrics_data):
        with col:
            st.metric(label=label, value=value, delta=delta)
    
    # ========== TOP OPPORTUNITIES SHOWCASE ==========
    st.markdown("## 🏆 Top Opportunities")
    
    if 'tag' in filtered_df.columns and 'final_score' in filtered_df.columns:
        top_opps = filtered_df[filtered_df['tag'] == 'Buy'].nlargest(12, 'final_score')
        
        if not top_opps.empty:
            # Create 3 rows of 4 cards each
            for row in range(0, len(top_opps), 4):
                cols = st.columns(4)
                for col_idx, (_, stock) in enumerate(top_opps.iloc[row:row+4].iterrows()):
                    if col_idx < len(cols):
                        with cols[col_idx]:
                            # Enhanced stock card
                            score_color = "#2ca02c" if stock['final_score'] >= 80 else "#ff7f0e" if stock['final_score'] >= 60 else "#d62728"
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {score_color}22 0%, {score_color}11 100%); 
                                        padding: 15px; border-radius: 10px; border: 1px solid {score_color}44;">
                                <h4 style="margin: 0; color: {score_color};">{stock['ticker']}</h4>
                                <p style="margin: 5px 0; font-size: 12px; color: #666;">{stock.get('company_name', 'N/A')[:25]}...</p>
                                <h2 style="margin: 10px 0; color: {score_color};">{stock['final_score']:.1f}</h2>
                                <p style="margin: 0; font-size: 14px;">
                                    Target: ₹{stock.get('target_price', 0):.0f} 
                                    <span style="color: #2ca02c;">↑{stock.get('upside_pct', 0):.1f}%</span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No buy opportunities found with current filters. Try adjusting your criteria.")
    
    # ========== MAIN CONTENT TABS ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Scanner",
        "🔥 Sector Analysis", 
        "⚡ Edge Finder",
        "🚨 Anomaly Detector",
        "📈 Performance Analytics",
        "🎯 Watchlist Builder"
    ])
    
    # ===== TAB 1: MARKET SCANNER =====
    with tab1:
        st.markdown("### 📊 Complete Market Scanner")
        
        if filtered_df.empty:
            st.warning("No stocks match your current filters. Try adjusting your criteria.")
        else:
            # Display columns selector
            display_cols = ['ticker', 'company_name', 'tag', 'final_score', 'price', 
                          'target_price', 'upside_pct', 'sector', 'edge_types']
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Enhanced dataframe with conditional formatting
            styled_df = filtered_df[available_cols].copy()
            
            # Add color coding for tags
            if 'tag' in styled_df.columns:
                tag_colors = {'Buy': '🟢', 'Watch': '🟡', 'Avoid': '🔴'}
                styled_df['tag'] = styled_df['tag'].map(lambda x: f"{tag_colors.get(x, '')} {x}")
            
            # Display with enhanced configuration
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "company_name": st.column_config.TextColumn("Company", width="large"),
                    "final_score": st.column_config.ProgressColumn(
                        "Score", 
                        min_value=0, 
                        max_value=100,
                        format="%.1f"
                    ),
                    "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                    "target_price": st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "upside_pct": st.column_config.NumberColumn("Upside %", format="%.1f%%"),
                }
            )
            
            # Export section
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if export_format == "CSV":
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"mantra_scan_{datetime.now():%Y%m%d_%H%M}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            with col2:
                if export_format == "Excel":
                    buffer = io.BytesIO()
                    filtered_df.to_excel(buffer, index=False, engine='openpyxl')
                    st.download_button(
                        "📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"mantra_scan_{datetime.now():%Y%m%d_%H%M}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
    
    # ===== TAB 2: SECTOR ANALYSIS =====
    with tab2:
        st.markdown("### 🔥 Sector Rotation & Performance")
        
        if sector_scores.empty:
            st.warning("No sector data available.")
        else:
            # Sector performance metrics
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Top sectors table
                st.markdown("#### 📊 Sector Rankings")
                top_sectors = sector_scores.nlargest(10, 'sector_score')[['sector', 'sector_score', 'companies']]
                st.dataframe(
                    top_sectors,
                    use_container_width=True,
                    column_config={
                        "sector": st.column_config.TextColumn("Sector"),
                        "sector_score": st.column_config.ProgressColumn(
                            "Score",
                            min_value=0,
                            max_value=100,
                            format="%.1f"
                        ),
                        "companies": st.column_config.NumberColumn("Companies", format="%d")
                    }
                )
            
            with col2:
                # Sector performance chart
                st.markdown("#### 📈 Sector Performance Distribution")
                fig = px.bar(
                    sector_scores.nlargest(15, 'sector_score'),
                    x='sector_score',
                    y='sector',
                    orientation='h',
                    color='sector_score',
                    color_continuous_scale='RdYlGn',
                    labels={'sector_score': 'Score', 'sector': 'Sector'},
                    height=400
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sector heatmap
            if len(filtered_df) > 0 and 'sector' in filtered_df.columns:
                st.markdown("#### 🗺️ Sector Opportunity Heatmap")
                
                # Create pivot for heatmap
                sector_pivot = filtered_df.groupby(['sector', 'tag']).size().reset_index(name='count')
                if not sector_pivot.empty:
                    heatmap_fig = components.create_heatmap(
                        sector_pivot,
                        x_col='tag',
                        y_col='sector',
                        value_col='count',
                        title='Opportunities by Sector and Signal'
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # ===== TAB 3: EDGE FINDER =====
    with tab3:
        st.markdown("### ⚡ Alpha Edge Detection")
        
        edge_df = filtered_df[filtered_df['has_edge'] == True] if 'has_edge' in filtered_df.columns else pd.DataFrame()
        
        if edge_df.empty:
            st.info("No edge signals detected in current selection. Edge signals identify stocks with exceptional patterns.")
        else:
            # Edge statistics
            edge_stats = edge_overview(edge_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Edge Signals", edge_stats.get('total_edges', 0))
            with col2:
                st.metric("Unique Stocks", len(edge_df))
            with col3:
                avg_edges = edge_df['edge_count'].mean() if 'edge_count' in edge_df.columns else 0
                st.metric("Avg Edges/Stock", f"{avg_edges:.1f}")
            
            # Edge type distribution
            if 'edge_types' in edge_df.columns:
                st.markdown("#### 📊 Edge Type Distribution")
                edge_types_expanded = edge_df['edge_types'].str.split(', ', expand=True).stack().value_counts()
                
                fig = px.pie(
                    values=edge_types_expanded.values,
                    names=edge_types_expanded.index,
                    title="Edge Signal Distribution",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top edge opportunities
            st.markdown("#### 🎯 Top Edge Opportunities")
            edge_display_cols = ['ticker', 'company_name', 'edge_types', 'edge_count', 
                               'final_score', 'price', 'target_price']
            available_edge_cols = [col for col in edge_display_cols if col in edge_df.columns]
            
            st.dataframe(
                edge_df.nlargest(20, 'edge_count')[available_edge_cols],
                use_container_width=True,
                column_config={
                    "edge_count": st.column_config.NumberColumn("# Edges", format="%d"),
                    "edge_types": st.column_config.TextColumn("Edge Signals", width="large")
                }
            )
    
    # ===== TAB 4: ANOMALY DETECTOR =====
    with tab4:
        st.markdown("### 🚨 Market Anomaly Detection")
        
        anomaly_df = filtered_df[filtered_df['anomaly'] == True] if 'anomaly' in filtered_df.columns else pd.DataFrame()
        
        if anomaly_df.empty:
            st.success("✅ No anomalies detected in current market conditions.")
        else:
            # Anomaly statistics
            st.markdown("#### 📊 Anomaly Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Anomalies", len(anomaly_df))
            with col2:
                anomaly_rate = len(anomaly_df) / max(len(filtered_df), 1) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            with col3:
                if 'anomaly_type' in anomaly_df.columns:
                    unique_types = anomaly_df['anomaly_type'].nunique()
                    st.metric("Anomaly Types", unique_types)
            with col4:
                if 'anomaly_severity' in anomaly_df.columns:
                    avg_severity = anomaly_df['anomaly_severity'].mean()
                    st.metric("Avg Severity", f"{avg_severity:.1f}")
            
            # Anomaly details
            st.markdown("#### 🔍 Anomaly Details")
            anomaly_cols = ['ticker', 'company_name', 'anomaly_reason', 'tag', 
                           'final_score', 'price', 'sector']
            available_anomaly_cols = [col for col in anomaly_cols if col in anomaly_df.columns]
            
            st.dataframe(
                anomaly_df[available_anomaly_cols],
                use_container_width=True,
                column_config={
                    "anomaly_reason": st.column_config.TextColumn("Reason", width="large")
                }
            )
            
            # Anomaly distribution by sector
            if 'sector' in anomaly_df.columns:
                st.markdown("#### 📊 Anomalies by Sector")
                anomaly_by_sector = anomaly_df['sector'].value_counts().head(10)
                
                fig = px.bar(
                    x=anomaly_by_sector.index,
                    y=anomaly_by_sector.values,
                    labels={'x': 'Sector', 'y': 'Anomaly Count'},
                    color=anomaly_by_sector.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 5: PERFORMANCE ANALYTICS =====
    with tab5:
        st.markdown("### 📈 Performance Analytics Dashboard")
        
        if filtered_df.empty:
            st.warning("No data available for analytics.")
        else:
            # Performance overview
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Score distribution
                st.markdown("#### 📊 Score Distribution")
                if 'final_score' in filtered_df.columns:
                    fig = px.histogram(
                        filtered_df,
                        x='final_score',
                        nbins=20,
                        color='tag' if 'tag' in filtered_df.columns else None,
                        title="Distribution of Final Scores",
                        labels={'final_score': 'Score', 'count': 'Number of Stocks'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Tag distribution
                st.markdown("#### 🏷️ Signal Distribution")
                if 'tag' in filtered_df.columns:
                    tag_dist = filtered_df['tag'].value_counts()
                    fig = px.pie(
                        values=tag_dist.values,
                        names=tag_dist.index,
                        title="Signal Tag Distribution",
                        color_discrete_map={'Buy': '#2ca02c', 'Watch': '#ff7f0e', 'Avoid': '#d62728'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics by various dimensions
            st.markdown("#### 📊 Multi-Dimensional Analysis")
            
            analysis_cols = st.columns(2)
            
            with analysis_cols[0]:
                # Top performers
                if 'upside_pct' in filtered_df.columns:
                    st.markdown("##### 🚀 Top Upside Potential")
                    top_upside = filtered_df.nlargest(10, 'upside_pct')[['ticker', 'upside_pct', 'final_score']]
                    
                    fig = px.bar(
                        top_upside,
                        x='ticker',
                        y='upside_pct',
                        color='final_score',
                        color_continuous_scale='Viridis',
                        labels={'upside_pct': 'Upside %'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with analysis_cols[1]:
                # Volume analysis
                if 'vol_ratio_1d_90d' in filtered_df.columns:
                    st.markdown("##### 📊 Volume Activity")
                    high_volume = filtered_df[filtered_df['vol_ratio_1d_90d'] > 1.5].nlargest(10, 'vol_ratio_1d_90d')
                    
                    if not high_volume.empty:
                        fig = px.scatter(
                            high_volume,
                            x='final_score',
                            y='vol_ratio_1d_90d',
                            size='vol_ratio_1d_90d',
                            color='tag' if 'tag' in high_volume.columns else None,
                            text='ticker',
                            labels={'vol_ratio_1d_90d': 'Volume Ratio', 'final_score': 'Score'}
                        )
                        fig.update_traces(textposition='top center')
                        st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 6: WATCHLIST BUILDER =====
    with tab6:
        st.markdown("### 🎯 Smart Watchlist Builder")
        
        # Watchlist creation interface
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("#### 📝 Create Watchlist")
            
            watchlist_name = st.text_input("Watchlist Name", placeholder="My Top Picks")
            
            criteria = st.multiselect(
                "Auto-add stocks matching:",
                options=["Top 10 Scores", "All Buy Signals", "Edge Signals", 
                        "Sector Leaders", "High Volume", "New Highs"],
                default=["Top 10 Scores"]
            )
            
            max_stocks = st.number_input(
                "Maximum Stocks",
                min_value=1,
                max_value=50,
                value=20
            )
            
            if st.button("🚀 Generate Watchlist", use_container_width=True):
                # Build watchlist based on criteria
                watchlist = pd.DataFrame()
                
                if "Top 10 Scores" in criteria and 'final_score' in filtered_df.columns:
                    watchlist = pd.concat([watchlist, filtered_df.nlargest(10, 'final_score')])
                
                if "All Buy Signals" in criteria and 'tag' in filtered_df.columns:
                    buy_stocks = filtered_df[filtered_df['tag'] == 'Buy']
                    watchlist = pd.concat([watchlist, buy_stocks])
                
                if "Edge Signals" in criteria and 'has_edge' in filtered_df.columns:
                    edge_stocks = filtered_df[filtered_df['has_edge'] == True]
                    watchlist = pd.concat([watchlist, edge_stocks])
                
                # Remove duplicates and limit
                watchlist = watchlist.drop_duplicates(subset=['ticker']).head(max_stocks)
                
                if not watchlist.empty:
                    st.success(f"✅ Generated watchlist with {len(watchlist)} stocks!")
                    
                    # Store in session state
                    st.session_state['current_watchlist'] = watchlist
                    st.session_state['watchlist_name'] = watchlist_name or "My Watchlist"
        
        with col2:
            st.markdown("#### 📊 Watchlist Preview")
            
            if 'current_watchlist' in st.session_state and not st.session_state['current_watchlist'].empty:
                watchlist_df = st.session_state['current_watchlist']
                
                # Summary stats
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("Stocks", len(watchlist_df))
                with stats_cols[1]:
                    avg_score = watchlist_df['final_score'].mean() if 'final_score' in watchlist_df.columns else 0
                    st.metric("Avg Score", f"{avg_score:.1f}")
                with stats_cols[2]:
                    buy_count = len(watchlist_df[watchlist_df['tag'] == 'Buy']) if 'tag' in watchlist_df.columns else 0
                    st.metric("Buy Signals", buy_count)
                with stats_cols[3]:
                    sectors = watchlist_df['sector'].nunique() if 'sector' in watchlist_df.columns else 0
                    st.metric("Sectors", sectors)
                
                # Display watchlist
                display_cols = ['ticker', 'company_name', 'tag', 'final_score', 'price', 'target_price']
                available_cols = [col for col in display_cols if col in watchlist_df.columns]
                
                st.dataframe(
                    watchlist_df[available_cols],
                    use_container_width=True,
                    height=300
                )
                
                # Export watchlist
                export_cols = st.columns(2)
                with export_cols[0]:
                    csv = watchlist_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Watchlist (CSV)",
                        data=csv,
                        file_name=f"{st.session_state.get('watchlist_name', 'watchlist')}_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("Generate a watchlist using the criteria on the left.")
    
    # ========== FOOTER ==========
    st.markdown("---")
    
    footer_cols = st.columns([1, 2, 1])
    with footer_cols[1]:
        st.markdown(f"""
        <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>M.A.N.T.R.A. Intelligence Platform | Last Updated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
        <p>All analysis is 100% data-driven | {summary.get('data_hash', 'N/A')[:8]}</p>
        <p style='font-size: 12px; margin-top: 10px;'>
        Built with ❤️ for elite market intelligence
        </p>
        </div>
        """, unsafe_allow_html=True)

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    main()
