"""
ui_components.py - M.A.N.T.R.A. UI Components
============================================
Beautiful, reusable UI components for the dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from constants import SIGNAL_COLORS, CHART_COLORS

# ============================================================================
# CUSTOM CSS
# ============================================================================

def load_custom_css():
    """Load custom CSS for beautiful styling"""
    st.markdown("""
    <style>
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #2d3139 100%);
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d26a;
    }
    
    /* Signal badges */
    .signal-badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin: 2px;
    }
    
    .signal-buy {
        background-color: #00d26a;
        color: #000;
    }
    
    .signal-watch {
        background-color: #ffa500;
        color: #000;
    }
    
    .signal-avoid {
        background-color: #ff4b4b;
        color: #fff;
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
        transform: translateY(-1px);
    }
    
    /* Alert styles */
    .alert-info {
        background: linear-gradient(135deg, #33b5e5 0%, #4fc3f7 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffa500 0%, #ffb732 100%);
        color: #000;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Section headers */
    .section-header {
        border-left: 4px solid #00d26a;
        padding-left: 15px;
        margin: 20px 0;
    }
    
    /* Data quality badge */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .quality-good {
        background-color: #00d26a;
        color: #000;
    }
    
    .quality-warning {
        background-color: #ffa500;
        color: #000;
    }
    
    .quality-poor {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# METRIC COMPONENTS
# ============================================================================

def metric_card(label: str, value: Any, delta: Optional[float] = None, icon: str = ""):
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
    html = f"""
    <div class="metric-card">
        {f'<div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>' if icon else ''}
        <div style="color: #8b92a0; font-size: 14px;">{label}</div>
        <div style="color: #fff; font-size: 28px; font-weight: bold; margin: 8px 0;">{formatted_value}</div>
    """
    
    if delta is not None:
        color = SIGNAL_COLORS['BUY'] if delta > 0 else SIGNAL_COLORS['AVOID']
        arrow = "↑" if delta > 0 else "↓"
        html += f'<div style="color: {color}; font-size: 14px;">{arrow} {abs(delta):.1f}%</div>'
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

def signal_badge(signal: str, score: Optional[float] = None):
    """Create a signal badge"""
    signal_class = {
        'BUY': 'signal-buy',
        'WATCH': 'signal-watch',
        'AVOID': 'signal-avoid',
        'NEUTRAL': 'signal-watch'
    }.get(signal, 'signal-watch')
    
    text = signal
    if score is not None:
        text += f" ({score:.0f})"
    
    html = f'<span class="signal-badge {signal_class}">{text}</span>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# STOCK CARDS
# ============================================================================

def stock_card(stock: pd.Series):
    """Create a detailed stock card"""
    ticker = stock.get('ticker', 'N/A')
    company = stock.get('company_name', ticker)[:30]
    price = stock.get('price', 0)
    change = stock.get('ret_1d', 0)
    signal = stock.get('decision', 'NEUTRAL')
    score = stock.get('composite_score', 50)
    volume = stock.get('volume_1d', 0)
    pe = stock.get('pe', 0)
    
    # Signal badge HTML
    signal_color = SIGNAL_COLORS.get(signal, SIGNAL_COLORS['NEUTRAL'])
    
    html = f"""
    <div class="stock-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4 style="margin: 0; color: #fff;">{ticker}</h4>
                <p style="margin: 0; color: #8b92a0; font-size: 12px;">{company}...</p>
            </div>
            <span class="signal-badge signal-{signal.lower()}">{signal} ({score:.0f})</span>
        </div>
        
        <div style="margin-top: 15px;">
            <div style="font-size: 24px; font-weight: bold; color: #fff;">
                ₹{price:,.2f}
                <span style="font-size: 16px; color: {'#00d26a' if change > 0 else '#ff4b4b'}; margin-left: 10px;">
                    {'↑' if change > 0 else '↓'} {abs(change):.1f}%
                </span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 15px;">
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px;">Volume</p>
                <p style="margin: 0; color: #fff; font-size: 13px;">{volume/1000:.0f}K</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px;">P/E</p>
                <p style="margin: 0; color: #fff; font-size: 13px;">{pe:.1f}</p>
            </div>
            <div>
                <p style="margin: 0; color: #8b92a0; font-size: 11px;">Risk</p>
                <p style="margin: 0; color: #fff; font-size: 13px;">{stock.get('risk_level', 'N/A')}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# CHARTS
# ============================================================================

def sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """Create an interactive sector performance heatmap"""
    # Prepare data
    sectors = sector_df['sector'].tolist()
    
    # Time periods
    periods = ['1D', '7D', '30D', '3M', '6M', '1Y']
    period_cols = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 
                   'sector_ret_3m', 'sector_ret_6m', 'sector_ret_1y']
    
    # Build matrix
    z_values = []
    for _, row in sector_df.iterrows():
        row_values = []
        for col in period_cols:
            if col in sector_df.columns:
                val = row[col]
                if isinstance(val, str):
                    val = float(val.replace('%', ''))
                row_values.append(val)
            else:
                row_values.append(0)
        z_values.append(row_values[:len(periods)])  # Ensure same length
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=periods,
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
        margin=dict(l=150, r=20, t=50, b=50)
    )
    
    return fig

def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for scores"""
    # Determine color
    if value >= 80:
        color = SIGNAL_COLORS['BUY']
    elif value >= 65:
        color = SIGNAL_COLORS['WATCH']
    else:
        color = SIGNAL_COLORS['AVOID']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 35], 'color': "rgba(255, 75, 75, 0.2)"},
                {'range': [35, 65], 'color': "rgba(255, 165, 0, 0.2)"},
                {'range': [65, 100], 'color': "rgba(0, 210, 106, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_distribution_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Create a distribution histogram"""
    fig = go.Figure(data=[
        go.Histogram(
            x=df[column],
            nbinsx=30,
            marker_color=SIGNAL_COLORS['BUY'],
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title='Count',
        template='plotly_dark',
        showlegend=False,
        height=300
    )
    
    # Add mean line
    mean_val = df[column].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="white", 
                  annotation_text=f"Mean: {mean_val:.1f}")
    
    return fig

# ============================================================================
# ALERTS AND MESSAGES
# ============================================================================

def show_alert(message: str, alert_type: str = "info"):
    """Show an alert message"""
    alert_class = {
        'info': 'alert-info',
        'warning': 'alert-warning',
        'danger': 'alert-danger',
        'success': 'alert-info'
    }.get(alert_type, 'alert-info')
    
    icon = {
        'info': 'ℹ️',
        'warning': '⚠️',
        'danger': '🚨',
        'success': '✅'
    }.get(alert_type, 'ℹ️')
    
    html = f"""
    <div class="{alert_class}">
        <span style="margin-right: 10px;">{icon}</span>
        <span>{message}</span>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# DATA DISPLAY
# ============================================================================

def display_dataframe(df: pd.DataFrame, title: str = "", height: int = 400):
    """Display a styled dataframe"""
    if title:
        st.markdown(f"### {title}")
    
    # Format numeric columns
    format_dict = {}
    for col in df.columns:
        if 'price' in col.lower():
            format_dict[col] = '₹{:,.2f}'
        elif 'ret_' in col or '_pct' in col or 'score' in col:
            format_dict[col] = '{:+.1f}%'
        elif 'volume' in col:
            format_dict[col] = '{:,.0f}'
    
    # Style the dataframe
    styled_df = df.style.format(format_dict)
    
    # Highlight decision column
    if 'decision' in df.columns:
        def color_decision(val):
            colors = {
                'BUY': f'background-color: {SIGNAL_COLORS["BUY"]}; color: black;',
                'WATCH': f'background-color: {SIGNAL_COLORS["WATCH"]}; color: black;',
                'AVOID': f'background-color: {SIGNAL_COLORS["AVOID"]}; color: white;'
            }
            return colors.get(val, '')
        
        styled_df = styled_df.applymap(color_decision, subset=['decision'])
    
    # Display with scrolling
    st.dataframe(styled_df, height=height, use_container_width=True)

# ============================================================================
# LAYOUT HELPERS
# ============================================================================

def section_header(title: str, subtitle: str = ""):
    """Create a section header"""
    html = f"""
    <div class="section-header">
        <h2 style="margin: 0; color: #fff;">{title}</h2>
        {f'<p style="margin: 5px 0 0 0; color: #8b92a0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def data_quality_badge(quality_score: float):
    """Show data quality badge"""
    if quality_score >= 80:
        badge_class = "quality-good"
        text = "Good"
    elif quality_score >= 60:
        badge_class = "quality-warning" 
        text = "Fair"
    else:
        badge_class = "quality-poor"
        text = "Poor"
    
    html = f'<span class="quality-badge {badge_class}">Data Quality: {text} ({quality_score:.0f}%)</span>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================

def dashboard_header():
    """Create main dashboard header"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("# 🔱 M.A.N.T.R.A.")
        st.markdown("*Market Analysis Neural Trading Research Assistant*")
    
    with col2:
        # Empty for spacing
        pass
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 20px;">
                <p style="margin: 0; color: #8b92a0;">Last Update</p>
                <p style="margin: 0; color: #fff; font-weight: bold;">{current_time}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
