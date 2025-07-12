"""
health_panel.py - Elite Data Health Monitoring System for M.A.N.T.R.A.

Production-grade health diagnostics with intelligent insights and actionable recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings


def calculate_data_health(df: pd.DataFrame, load_summary: Dict) -> Dict:
    """
    Core health calculation logic - can be tested independently of Streamlit.
    
    Args:
        df: The main dataframe
        load_summary: Basic stats from loader
        
    Returns:
        Comprehensive health assessment dictionary
    """
    health = {
        'score': 100,
        'status': 'healthy',
        'issues': [],
        'warnings': [],
        'tips': [],
        'metrics': {}
    }
    
    # Basic counts
    total_rows = len(df) if df is not None else 0
    total_cols = len(df.columns) if df is not None else 0
    
    health['metrics']['total_rows'] = total_rows
    health['metrics']['total_cols'] = total_cols
    health['metrics']['total_stocks'] = load_summary.get('total_stocks', 0)
    health['metrics']['total_sectors'] = load_summary.get('total_sectors', 0)
    health['metrics']['duplicates'] = load_summary.get('duplicates', 0)
    
    # Critical checks
    if df is None or total_rows == 0:
        health['score'] = 0
        health['status'] = 'critical'
        health['issues'].append("No data loaded")
        return health
    
    # Data completeness analysis
    total_cells = total_rows * total_cols
    nan_count = df.isna().sum().sum()
    nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
    
    health['metrics']['nan_count'] = int(nan_count)
    health['metrics']['nan_percentage'] = round(nan_pct, 2)
    health['metrics']['total_cells'] = total_cells
    
    # Column-specific health
    critical_cols = ['ticker', 'price', 'final_score', 'sector']
    missing_critical = [col for col in critical_cols if col not in df.columns]
    
    if missing_critical:
        health['score'] -= 30
        health['issues'].append(f"Missing critical columns: {', '.join(missing_critical)}")
        health['tips'].append("Ensure your data source has all required columns")
    
    # Analyze each column
    column_health = {}
    for col in df.columns:
        col_stats = {
            'nulls': int(df[col].isna().sum()),
            'null_pct': round(df[col].isna().sum() / len(df) * 100, 2),
            'unique': int(df[col].nunique()),
            'dtype': str(df[col].dtype)
        }
        
        # Check for problematic columns
        if col_stats['null_pct'] > 80:
            health['warnings'].append(f"{col}: {col_stats['null_pct']}% missing")
            health['score'] -= 2
        elif col_stats['null_pct'] > 50:
            health['warnings'].append(f"{col}: {col_stats['null_pct']}% missing")
            health['score'] -= 1
            
        # Check for all-zero numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] == 0).all():
                health['warnings'].append(f"{col}: all zeros")
                health['score'] -= 3
                
        column_health[col] = col_stats
    
    health['metrics']['column_health'] = column_health
    
    # Data freshness check
    if 'last_updated' in load_summary:
        last_update = load_summary['last_updated']
        if isinstance(last_update, str):
            try:
                last_update = datetime.fromisoformat(last_update)
                age_hours = (datetime.now() - last_update).total_seconds() / 3600
                health['metrics']['data_age_hours'] = round(age_hours, 1)
                
                if age_hours > 72:
                    health['warnings'].append(f"Data is {age_hours:.0f} hours old")
                    health['score'] -= 10
                    health['tips'].append("Consider refreshing data for latest market conditions")
            except:
                pass
    
    # Price data validation
    if 'price' in df.columns:
        price_stats = df['price'].describe()
        zero_prices = (df['price'] == 0).sum()
        negative_prices = (df['price'] < 0).sum()
        
        if zero_prices > 0:
            health['warnings'].append(f"{zero_prices} stocks with zero price")
            health['score'] -= 5
        if negative_prices > 0:
            health['issues'].append(f"{negative_prices} stocks with negative price")
            health['score'] -= 10
            
        health['metrics']['price_range'] = f"₹{price_stats['min']:.2f} - ₹{price_stats['max']:.2f}"
    
    # Score validation
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    for score_col in score_cols:
        if score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
            out_of_range = ((df[score_col] < 0) | (df[score_col] > 100)).sum()
            if out_of_range > 0:
                health['warnings'].append(f"{score_col}: {out_of_range} values out of 0-100 range")
                health['score'] -= 3
    
    # Duplicate analysis
    dup_count = load_summary.get('duplicates', 0)
    if dup_count > 0:
        health['warnings'].append(f"{dup_count} duplicate tickers found")
        health['score'] -= 5
        health['tips'].append("Remove duplicate entries for accurate analysis")
    
    # Sector distribution
    if 'sector' in df.columns:
        sector_dist = df['sector'].value_counts()
        health['metrics']['sector_distribution'] = sector_dist.to_dict()
        
        # Check for imbalanced sectors
        if len(sector_dist) > 0:
            max_sector_pct = (sector_dist.iloc[0] / len(df) * 100)
            if max_sector_pct > 50:
                health['warnings'].append(f"Sector imbalance: {sector_dist.index[0]} has {max_sector_pct:.1f}% of stocks")
                health['tips'].append("Consider diversifying sector coverage")
    
    # Volume data check
    vol_cols = [col for col in df.columns if 'vol' in col.lower() and 'ratio' not in col.lower()]
    for vol_col in vol_cols:
        if vol_col in df.columns and pd.api.types.is_numeric_dtype(df[vol_col]):
            zero_vol = (df[vol_col] == 0).sum()
            if zero_vol > len(df) * 0.1:  # More than 10% zero volume
                health['warnings'].append(f"{vol_col}: {zero_vol} stocks with zero volume")
                health['tips'].append("Check for trading halts or data feed issues")
    
    # Final score adjustment and status
    health['score'] = max(0, min(100, health['score']))
    
    if health['score'] >= 90:
        health['status'] = 'excellent'
    elif health['score'] >= 75:
        health['status'] = 'good'
    elif health['score'] >= 60:
        health['status'] = 'fair'
    elif health['score'] >= 40:
        health['status'] = 'poor'
    else:
        health['status'] = 'critical'
    
    # Add final tips based on score
    if health['score'] < 75 and not health['tips']:
        health['tips'].append("Review warnings and fix data quality issues")
    elif health['score'] >= 90:
        health['tips'].append("Data quality is excellent - ready for analysis")
    
    return health


def render_health_panel(df: pd.DataFrame, load_summary: Dict):
    """
    Render the health panel in Streamlit sidebar.
    
    Args:
        df: Main dataframe
        load_summary: Loading summary from data loader
    """
    st.sidebar.markdown("## 🏥 Data Health Monitor")
    
    # Calculate health
    health = calculate_data_health(df, load_summary)
    
    # Display health score with color coding
    score = health['score']
    if score >= 90:
        color = "🟢"
        score_color = "#28a745"
    elif score >= 75:
        color = "🟡"
        score_color = "#ffc107"
    elif score >= 60:
        color = "🟠"
        score_color = "#fd7e14"
    else:
        color = "🔴"
        score_color = "#dc3545"
    
    # Main health score display
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.markdown(f"### {color} Health Score")
        st.markdown(f"<h1 style='color: {score_color}; margin: 0;'>{score}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("### Status")
        st.markdown(f"**{health['status'].upper()}**")
    
    # Quick metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Stocks", health['metrics']['total_stocks'])
        st.metric("NaN %", f"{health['metrics']['nan_percentage']:.1f}%")
    with col2:
        st.metric("Total Sectors", health['metrics']['total_sectors'])
        st.metric("Duplicates", health['metrics']['duplicates'])
    
    # Data freshness
    st.sidebar.markdown("### 🕐 Data Freshness")
    source = load_summary.get('source', 'Unknown')
    load_time = load_summary.get('load_time', datetime.now())
    
    if isinstance(load_time, str):
        st.sidebar.caption(f"Source: `{source}`")
        st.sidebar.caption(f"Loaded: {load_time}")
    else:
        time_ago = datetime.now() - load_time
        if time_ago.seconds < 60:
            time_str = "Just now"
        elif time_ago.seconds < 3600:
            time_str = f"{time_ago.seconds // 60} min ago"
        else:
            time_str = f"{time_ago.seconds // 3600} hours ago"
        
        st.sidebar.caption(f"Source: `{source}`")
        st.sidebar.caption(f"Loaded: {time_str}")
    
    if 'data_age_hours' in health['metrics']:
        age = health['metrics']['data_age_hours']
        if age > 24:
            st.sidebar.warning(f"⚠️ Data is {age:.0f} hours old")
    
    # Critical issues
    if health['issues']:
        st.sidebar.markdown("### 🚨 Critical Issues")
        for issue in health['issues']:
            st.sidebar.error(f"❌ {issue}")
    
    # Warnings
    if health['warnings']:
        with st.sidebar.expander(f"⚠️ Warnings ({len(health['warnings'])})"):
            for warning in health['warnings']:
                st.warning(warning)
    
    # Actionable tips
    if health['tips']:
        st.sidebar.markdown("### 💡 Recommendations")
        for tip in health['tips']:
            st.sidebar.info(f"→ {tip}")
    
    # Detailed diagnostics (collapsible)
    with st.sidebar.expander("🔬 Detailed Diagnostics"):
        st.markdown("**Data Shape**")
        st.text(f"Rows: {health['metrics']['total_rows']:,}")
        st.text(f"Columns: {health['metrics']['total_cols']}")
        st.text(f"Total Cells: {health['metrics']['total_cells']:,}")
        
        if 'price_range' in health['metrics']:
            st.markdown("**Price Range**")
            st.text(health['metrics']['price_range'])
        
        # Top missing columns
        if 'column_health' in health['metrics']:
            col_health = health['metrics']['column_health']
            missing_cols = [(col, stats['null_pct']) 
                          for col, stats in col_health.items() 
                          if stats['null_pct'] > 10]
            missing_cols.sort(key=lambda x: x[1], reverse=True)
            
            if missing_cols:
                st.markdown("**Columns with >10% missing**")
                for col, pct in missing_cols[:5]:
                    st.text(f"{col}: {pct}%")
    
    # Reload button
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reload Data", help="Clear cache and reload all data"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("📋 Copy Report", help="Copy health report to clipboard"):
            report = generate_health_report(health, load_summary)
            st.sidebar.code(report, language=None)
            st.sidebar.success("Report ready to copy!")


def generate_health_report(health: Dict, load_summary: Dict) -> str:
    """Generate a text report of data health for sharing."""
    report = f"""DATA HEALTH REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

HEALTH SCORE: {health['score']}/100 ({health['status'].upper()})
Source: {load_summary.get('source', 'Unknown')}

SUMMARY:
- Total Stocks: {health['metrics']['total_stocks']}
- Total Sectors: {health['metrics']['total_sectors']}
- Missing Data: {health['metrics']['nan_percentage']:.1f}%
- Duplicates: {health['metrics']['duplicates']}

"""
    
    if health['issues']:
        report += "CRITICAL ISSUES:\n"
        for issue in health['issues']:
            report += f"- {issue}\n"
        report += "\n"
    
    if health['warnings']:
        report += "WARNINGS:\n"
        for warning in health['warnings'][:5]:
            report += f"- {warning}\n"
        if len(health['warnings']) > 5:
            report += f"... and {len(health['warnings']) - 5} more\n"
        report += "\n"
    
    if health['tips']:
        report += "RECOMMENDATIONS:\n"
        for tip in health['tips']:
            report += f"- {tip}\n"
    
    return report


# Standalone function for testing
def assess_data_health(df: pd.DataFrame, source: str = "Unknown") -> Dict:
    """
    Standalone health assessment function for testing outside Streamlit.
    
    Args:
        df: DataFrame to assess
        source: Data source name
        
    Returns:
        Complete health assessment
    """
    load_summary = {
        'total_stocks': len(df) if df is not None else 0,
        'total_sectors': df['sector'].nunique() if df is not None and 'sector' in df.columns else 0,
        'duplicates': df.duplicated(subset=['ticker']).sum() if df is not None and 'ticker' in df.columns else 0,
        'source': source,
        'load_time': datetime.now()
    }
    
    return calculate_data_health(df, load_summary)


# Example usage for testing
if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'ticker': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI'],
        'price': [2500, 3500, 1500, 1800, 900],
        'sector': ['Energy', 'IT', 'IT', 'Banking', 'Banking'],
        'final_score': [85, 90, 88, 82, 79]
    })
    
    health_report = assess_data_health(test_df, "Test Data")
    print(f"Health Score: {health_report['score']}")
    print(f"Status: {health_report['status']}")
    print(f"Issues: {health_report['issues']}")
    print(f"Tips: {health_report['tips']}")
