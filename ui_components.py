"""
ui_components.py - M.A.N.T.R.A. UI Components
============================================
Beautiful, reusable UI components for the Streamlit dashboard
Provides consistent styling and enhanced user experience
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

# Import from constants
from constants import SIGNAL_COLORS, COLORS, NUMBER_FORMAT

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    """Load custom CSS for beautiful styling"""
    st.markdown("""
    <style>
    /* Main theme */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Signal badges */
    .signal-buy {
        background-color: #00d26a;
        color: #000;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin: 2px;
    }
    
    .signal-watch {
        background-color: #ffa500;
        color: #000;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin: 2px;
    }
    
    .signal-avoid {
        background-color: #ff4b4b;
        color: #fff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin: 2px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #2d3139 100%);
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Stock cards */
    .stock-card {
        background: #1e2329;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s;
    }
    
    .stock-card:hover {
        background: #252a31;
        border-color: #00d26a;
    }
    
    /* Alert banner */
    .alert-banner {
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ffa500 0%, #ffb732 100%);
        color: #000;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #33b5e5 0%, #4fc3f7 100%);
        color: white;
    }
    
    /* Trend indicator */
    .trend-up {
        color: #00d26a;
        font-weight: bold;
    }
    
    .trend-down {
        color: #ff4b4b;
        font-weight: bold;
    }
    
    /* Data quality badge */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .quality-excellent {
        background-color: #00d26a;
        color: #000;
    }
    
    .quality-good {
        background-color: #33b5e5;
        color: white;
    }
    
    .quality-warning {
        background-color: #ffa500;
        color: #000;
    }
    
    .quality-poor {
        background-color: #ff4b4b;
        color: white;
    }
    
    /* Floating summary */
    .floating-summary {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(30, 35, 41, 0.95);
        border: 1px solid #00d26a;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
    
    /* Sparkline container */
    .sparkline-container {
        display: inline-block;
        margin-left: 10px;
        vertical-align: middle;
    }
    
    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #1e2329 0%, transparent 100%);
        padding: 10px 0;
        margin: 20px 0 10px 0;
        border-left: 4px solid #00d26a;
        padding-left: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SIGNAL BADGES
# ============================================================================

def signal_badge(signal: str, score: Optional[float] = None) -> str:
    """Create a beautiful signal badge"""
    signal_upper = signal.upper()
    badge_class = {
        'BUY': 'signal-buy',
        'WATCH': 'signal-watch',
        'AVOID': 'signal-avoid',
        'NEUTRAL': 'signal-watch'
    }.get(signal_upper, 'signal-watch')
    
    if score:
        return f'<span class="{badge_class}">{signal_upper} ({score:.0f})</span>'
    else:
        return f'<span class="{badge_class}">{signal_upper}</span>'

# ============================================================================
# METRIC CARDS
# ============================================================================

def metric_card(
    title: str,
    value: Any,
    delta: Optional[float] = None,
    delta_color: str = "normal",
    subtitle: Optional[str] = None,
    icon: Optional[str] = None
):
    """Create a beautiful metric card"""
    # Format value
    if isinstance(value, (int, float)):
        if value > 1000000:
            formatted_value = f"{value/1000000:.1f}M"
        elif value > 1000:
            formatted_value = f"{value/1000:.1f}K"
        else:
            formatted_value = f"{value:,.0f}"
    else:
        formatted_value = str(value)
    
    # Build HTML
    html = f'<div class="metric-card">'
    
    if icon:
        html += f'<div style="font-size: 24px; margin-bottom: 10px;">{icon}</div>'
    
    html += f'<h4 style="margin: 0; color: #8b92a0;">{title}</h4>'
    html += f'<h2 style="margin: 5px 0; color: #fff;">{formatted_value}</h2>'
    
    if delta is not None:
        delta_icon = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_class = "trend-up" if delta > 0 else "trend-down"
        html += f'<span class="{delta_class}">{delta_icon} {abs(delta):.1f}%</span>'
    
    if subtitle:
        html += f'<p style="margin: 5px 0 0 0; color: #8b92a0; font-size: 12px;">{subtitle}</p>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# STOCK CARDS
# ============================================================================

def stock_card(stock_data: pd.Series, show_details: bool = True) -> str:
    """Create a detailed stock card"""
    ticker = stock_data.get('ticker', 'N/A')
    company = stock_data.get('company_name', ticker)
    price = stock_data.get('price', 0)
    change = stock_data.get('ret_1d', 0)
    signal = stock_data.get('decision', 'NEUTRAL')
    score = stock_data.get('composite_score', 50)
    
    # Build card HTML
    html = f'<div class="stock-card">'
    
    # Header row
    html += f'<div style="display: flex; justify-content: space-between; align-items: center;">'
    html += f'<div>'
    html += f'<h4 style="margin: 0; color: #fff;">{ticker}</h4>'
    html += f'<p style="margin: 0; color: #8b92a0; font-size: 12px;">{company[:30]}...</p>'
    html += f'</div>'
    html += f'<div style="text-align: right;">'
    html += signal_badge(signal, score)
    html += f'</div>'
    html += f'</div>'
    
    if show_details:
        # Price row
        html += f'<div style="display: flex; justify-content: space-between; margin-top: 15px;">'
        html += f'<div>'
        html += f'<span style="font-size: 20px; font-weight: bold;">₹{price:,.2f}</span>'
        change_class = "trend-up" if change > 0 else "trend-down"
        html += f'<span class="{change_class}" style="margin-left: 10px;">{change:+.1f}%</span>'
        html += f'</div>'
        html += f'</div>'
        
        # Metrics row
        html += f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px;">'
        
        # Volume
        volume = stock_data.get('volume_1d', 0)
        rvol = stock_data.get('rvol', 1)
        html += f'<div>'
        html += f'<p style="margin: 0; color: #8b92a0; font-size: 11px;">Volume</p>'
        html += f'<p style="margin: 0; color: #fff; font-size: 13px;">{volume/1000:.0f}K</p>'
        html += f'<p style="margin: 0; color: #00d26a; font-size: 11px;">{rvol:.1f}x</p>'
        html += f'</div>'
        
        # PE
        pe = stock_data.get('pe', 0)
        html += f'<div>'
        html += f'<p style="margin: 0; color: #8b92a0; font-size: 11px;">P/E</p>'
        html += f'<p style="margin: 0; color: #fff; font-size: 13px;">{pe:.1f}</p>'
        html += f'</div>'
        
        # Momentum
        momentum = stock_data.get('momentum_score', 50)
        html += f'<div>'
        html += f'<p style="margin: 0; color: #8b92a0; font-size: 11px;">Momentum</p>'
        html += f'<p style="margin: 0; color: #fff; font-size: 13px;">{momentum:.0f}</p>'
        html += f'</div>'
        
        html += f'</div>'
    
    html += f'</div>'
    
    return html

# ============================================================================
# CHARTS
# ============================================================================

def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """Create an interactive sector performance heatmap"""
    # Prepare data
    sectors = sector_df['sector'].tolist()
    
    # Create matrix of performance metrics
    metrics = ['1D', '7D', '30D', '3M', '6M', '1Y']
    metric_cols = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 
                   'sector_ret_3m', 'sector_ret_6m', 'sector_ret_1y']
    
    # Build matrix
    z_values = []
    for _, row in sector_df.iterrows():
        row_values = []
        for col in metric_cols:
            if col in sector_df.columns:
                val = row[col]
                if isinstance(val, str):
                    val = float(val.replace('%', ''))
                row_values.append(val)
            else:
                row_values.append(0)
        z_values.append(row_values)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=metrics,
        y=sectors,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{val:.1f}%' for val in row] for row in z_values],
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Sector Performance Heatmap',
        xaxis_title='Time Period',
        yaxis_title='Sector',
        height=max(400, len(sectors) * 25),
        template='plotly_dark',
        margin=dict(l=200, r=20, t=50, b=50)
    )
    
    return fig

def trend_sparkline(data: List[float], width: int = 100, height: int = 30) -> str:
    """Create a mini sparkline chart"""
    if not data or len(data) < 2:
        return ""
    
    # Normalize data
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    normalized = [(v - min_val) / range_val * height for v in data]
    
    # Create SVG path
    points = []
    for i, val in enumerate(normalized):
        x = i * (width / (len(data) - 1))
        y = height - val
        points.append(f"{x},{y}")
    
    path = "M" + " L".join(points)
    
    # Determine color
    color = SIGNAL_COLORS['BUY'] if data[-1] > data[0] else SIGNAL_COLORS['AVOID']
    
    svg = f"""
    <svg width="{width}" height="{height}" style="display: inline-block; vertical-align: middle;">
        <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
    </svg>
    """
    
    return svg

def performance_gauge(score: float, title: str = "Score") -> go.Figure:
    """Create a performance gauge chart"""
    # Determine color based on score
    if score >= 80:
        color = SIGNAL_COLORS['BUY']
    elif score >= 60:
        color = SIGNAL_COLORS['WATCH']
    else:
        color = SIGNAL_COLORS['AVOID']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 75, 75, 0.2)"},
                {'range': [50, 70], 'color': "rgba(255, 165, 0, 0.2)"},
                {'range': [70, 100], 'color': "rgba(0, 210, 106, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ============================================================================
# ALERT COMPONENTS
# ============================================================================

def alert_banner(
    message: str,
    alert_type: str = "info",
    icon: Optional[str] = None,
    dismissible: bool = True
):
    """Create an alert banner"""
    # Determine styling based on type
    type_config = {
        'critical': ('alert-critical', '🚨'),
        'high': ('alert-high', '⚠️'),
        'medium': ('alert-medium', 'ℹ️'),
        'success': ('alert-medium', '✅'),
        'info': ('alert-medium', 'ℹ️')
    }
    
    css_class, default_icon = type_config.get(alert_type, ('alert-medium', 'ℹ️'))
    display_icon = icon or default_icon
    
    # Build HTML
    html = f'<div class="alert-banner {css_class}">'
    html += f'<span style="margin-right: 10px; font-size: 18px;">{display_icon}</span>'
    html += f'<span>{message}</span>'
    
    if dismissible:
        html += '<span style="float: right; cursor: pointer;" onclick="this.parentElement.style.display=\'none\'">✕</span>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# DATA DISPLAY COMPONENTS
# ============================================================================

def styled_dataframe(
    df: pd.DataFrame,
    highlight_column: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None
) -> None:
    """Display a beautifully styled dataframe"""
    # Apply styling
    styled_df = df.style
    
    # Highlight specific column
    if highlight_column and highlight_column in df.columns:
        def highlight_col(s):
            return ['background-color: rgba(0, 210, 106, 0.1)' if s.name == highlight_column else '' for _ in s]
        styled_df = styled_df.apply(highlight_col, axis=0)
    
    # Apply color mapping
    if color_map:
        for col, color in color_map.items():
            if col in df.columns:
                styled_df = styled_df.applymap(
                    lambda x: f'color: {color}' if pd.notna(x) else '',
                    subset=[col]
                )
    
    # Format numbers
    format_dict = {}
    for col in df.columns:
        if 'price' in col.lower() or col == 'target_price' or col == 'stop_loss':
            format_dict[col] = '₹{:,.2f}'
        elif 'ret_' in col or '_pct' in col or 'score' in col:
            format_dict[col] = '{:+.1f}%'
        elif 'volume' in col:
            format_dict[col] = '{:,.0f}'
    
    if format_dict:
        styled_df = styled_df.format(format_dict)
    
    # Display
    st.dataframe(styled_df, use_container_width=True)

# ============================================================================
# SUMMARY COMPONENTS
# ============================================================================

def floating_summary(
    metrics: Dict[str, Any],
    position: str = "bottom-right"
):
    """Create a floating summary panel"""
    # Position styles
    position_styles = {
        "bottom-right": "bottom: 20px; right: 20px;",
        "bottom-left": "bottom: 20px; left: 20px;",
        "top-right": "top: 80px; right: 20px;",
        "top-left": "top: 80px; left: 20px;"
    }
    
    style = position_styles.get(position, position_styles["bottom-right"])
    
    # Build HTML
    html = f'<div class="floating-summary" style="{style}">'
    html += '<h4 style="margin: 0 0 10px 0; color: #00d26a;">Quick Summary</h4>'
    
    for key, value in metrics.items():
        html += f'<div style="margin: 5px 0;">'
        html += f'<span style="color: #8b92a0;">{key}:</span> '
        html += f'<span style="color: #fff; font-weight: bold;">{value}</span>'
        html += '</div>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def quality_badge(quality_score: float) -> str:
    """Create a data quality badge"""
    if quality_score >= 90:
        css_class = "quality-excellent"
        text = "Excellent"
    elif quality_score >= 75:
        css_class = "quality-good"
        text = "Good"
    elif quality_score >= 60:
        css_class = "quality-warning"
        text = "Fair"
    else:
        css_class = "quality-poor"
        text = "Poor"
    
    return f'<span class="quality-badge {css_class}">{text} ({quality_score:.0f}%)</span>'

# ============================================================================
# LAYOUT HELPERS
# ============================================================================

def section_header(title: str, subtitle: Optional[str] = None):
    """Create a section header"""
    html = f'<div class="section-header">'
    html += f'<h2 style="margin: 0; color: #fff;">{title}</h2>'
    if subtitle:
        html += f'<p style="margin: 5px 0 0 0; color: #8b92a0;">{subtitle}</p>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def create_columns_adaptive(num_items: int, max_cols: int = 4) -> List:
    """Create adaptive column layout based on number of items"""
    if num_items <= 2:
        return st.columns(num_items)
    elif num_items <= 4:
        return st.columns(min(num_items, max_cols))
    else:
        # For many items, create rows
        cols = []
        items_per_row = min(max_cols, num_items)
        for i in range(0, num_items, items_per_row):
            cols.extend(st.columns(min(items_per_row, num_items - i)))
        return cols

# ============================================================================
# MAIN DASHBOARD HEADER
# ============================================================================

def dashboard_header(
    title: str = "M.A.N.T.R.A.",
    subtitle: str = "Market Analysis Neural Trading Research Assistant",
    market_status: Optional[Dict] = None
):
    """Create main dashboard header"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"# 🔱 {title}")
        st.markdown(f"*{subtitle}*")
    
    with col2:
        if market_status:
            regime = market_status.get('regime', 'Unknown')
            confidence = market_status.get('confidence', 0)
            
            if confidence > 70:
                regime_color = "#00d26a"
            elif confidence > 50:
                regime_color = "#ffa500"
            else:
                regime_color = "#ff4b4b"
            
            st.markdown(
                f'<div style="text-align: center; padding: 10px; background: rgba(30, 35, 41, 0.5); border-radius: 10px;">'
                f'<h3 style="margin: 0; color: {regime_color};">Market Regime: {regime}</h3>'
                f'<p style="margin: 5px 0 0 0; color: #8b92a0;">Confidence: {confidence:.0f}%</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f'<div style="text-align: right; padding: 10px;">'
            f'<p style="margin: 0; color: #8b92a0;">Last Update</p>'
            f'<h4 style="margin: 0; color: #fff;">{current_time}</h4>'
            f'</div>',
            unsafe_allow_html=True
        )

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_ui():
    """Initialize UI components and styling"""
    # Set page config
    st.set_page_config(
        page_title="M.A.N.T.R.A. - Stock Intelligence",
        page_icon="🔱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Hide streamlit branding
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. UI Components")
    print("="*60)
    print("\nBeautiful, reusable UI components for Streamlit")
    print("\nComponents:")
    print("  - signal_badge() - Buy/Watch/Avoid badges")
    print("  - metric_card() - Beautiful metric displays")
    print("  - stock_card() - Detailed stock information")
    print("  - sector_heatmap() - Interactive sector visualization")
    print("  - alert_banner() - Alert notifications")
    print("  - styled_dataframe() - Enhanced data tables")
    print("  - dashboard_header() - Main header")
    print("\nUse initialize_ui() to setup styling")
    print("="*60)
