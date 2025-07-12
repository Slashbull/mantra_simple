"""
sector_mapper.py - Elite Sector Analysis & Rotation Intelligence Engine

Production-grade sector mapping with advanced analytics, anomaly detection, and edge signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class SectorStatus(Enum):
   """Sector rotation status classifications"""
   BREAKOUT = "Breakout"
   HOT = "Hot"
   WARMING = "Warming"
   NEUTRAL = "Neutral"
   COOLING = "Cooling"
   COLD = "Cold"
   BREAKDOWN = "Breakdown"


@dataclass
class SectorMetrics:
   """Comprehensive sector performance metrics"""
   avg_return_1m: float
   avg_return_3m: float
   avg_return_6m: float
   volatility: float
   momentum_score: float
   relative_strength: float
   breadth: float
   volume_surge: float
   quality_score: float
   risk_adjusted_return: float


def run_sector_mapper(
   sector_df: pd.DataFrame,
   stock_df: Optional[pd.DataFrame] = None,
   lookback_periods: Dict[str, int] = None
) -> pd.DataFrame:
   """
   Ultimate sector rotation and strength analyzer with edge detection.
   
   Args:
       sector_df: Basic sector data with at minimum 'sector' column
       stock_df: Optional full stock data for advanced analytics
       lookback_periods: Custom lookback periods for calculations
       
   Returns:
       Elite sector analysis DataFrame ready for dashboard visualization
   """
   # Initialize with defaults
   if lookback_periods is None:
       lookback_periods = {"short": 30, "medium": 90, "long": 180}
   
   # Data validation and normalization
   df = _normalize_sector_data(sector_df)
   
   # Core sector metrics
   df = _calculate_base_metrics(df, stock_df)
   
   # Advanced analytics
   df = _calculate_momentum_scores(df)
   df = _calculate_relative_strength(df)
   df = _calculate_risk_metrics(df)
   df = _detect_sector_anomalies(df)
   df = _calculate_rotation_signals(df)
   
   # Composite scoring
   df = _calculate_composite_scores(df)
   
   # Final rankings and status
   df = _assign_rotation_status(df)
   df = _generate_edge_signals(df)
   
   # Dashboard-ready formatting
   df = _format_for_dashboard(df)
   
   return df


def _normalize_sector_data(sector_df: pd.DataFrame) -> pd.DataFrame:
   """Normalize and clean sector data with bulletproof handling."""
   df = sector_df.copy()
   
   # Standardize column names
   df.columns = (
       df.columns.str.strip()
                 .str.lower()
                 .str.replace(r'[^\w\s]', '', regex=True)
                 .str.replace(r'\s+', '_', regex=True)
   )
   
   # Ensure sector column exists and is clean
   if 'sector' not in df.columns:
       raise ValueError("'sector' column is required")
   
   df['sector'] = (
       df['sector'].astype(str)
                  .str.strip()
                  .str.title()
                  .replace('Nan', 'Unknown')
   )
   
   # Remove invalid sectors
   df = df[~df['sector'].isin(['Unknown', '', 'None'])].copy()
   
   # Convert numeric columns safely
   numeric_cols = [col for col in df.columns if any(
       metric in col for metric in ['avg', 'return', 'count', 'score', 'vol']
   )]
   
   for col in numeric_cols:
       if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
   
   return df


def _calculate_base_metrics(df: pd.DataFrame, stock_df: Optional[pd.DataFrame]) -> pd.DataFrame:
   """Calculate fundamental sector metrics."""
   # Use provided metrics or calculate from stock data
   if stock_df is not None and 'sector' in stock_df.columns:
       # Aggregate from stock-level data
       metrics = stock_df.groupby('sector').agg({
           'ret_1m': ['mean', 'std', 'count'],
           'ret_3m': ['mean', 'std'],
           'ret_6m': ['mean', 'std'],
           'final_score': 'mean',
           'vol_ratio_1d_90d': 'mean',
           'pe': lambda x: x[x > 0].median() if len(x[x > 0]) > 0 else np.nan,
           'market_cap': 'sum'
       }).reset_index()
       
       # Flatten column names
       metrics.columns = ['_'.join(col).strip('_') for col in metrics.columns]
       metrics = metrics.rename(columns={
           'ret_1m_mean': 'avg_return_1m',
           'ret_3m_mean': 'avg_return_3m',
           'ret_6m_mean': 'avg_return_6m',
           'ret_1m_std': 'volatility_1m',
           'ret_1m_count': 'sector_count',
           'final_score_mean': 'avg_quality_score',
           'vol_ratio_1d_90d_mean': 'avg_volume_surge',
           'pe_<lambda>': 'median_pe',
           'market_cap_sum': 'sector_market_cap'
       })
       
       # Merge with existing data
       df = df.merge(metrics, on='sector', how='left')
   
   # Ensure essential columns exist
   essential_cols = {
       'avg_return_3m': 'sector_avg_3m',
       'avg_return_1m': 'sector_avg_1m',
       'sector_count': 'sector_count'
   }
   
   for new_col, old_col in essential_cols.items():
       if new_col not in df.columns and old_col in df.columns:
           df[new_col] = df[old_col]
       elif new_col not in df.columns:
           df[new_col] = 0
   
   # Fill missing values intelligently
   df['avg_return_3m'] = df['avg_return_3m'].fillna(0)
   df['avg_return_1m'] = df['avg_return_1m'].fillna(0)
   df['sector_count'] = df['sector_count'].fillna(0).astype(int)
   
   return df


def _calculate_momentum_scores(df: pd.DataFrame) -> pd.DataFrame:
   """Calculate multi-timeframe momentum scores."""
   # Short-term momentum (1M)
   df['momentum_1m'] = df['avg_return_1m'].rank(pct=True) * 100
   
   # Medium-term momentum (3M)
   df['momentum_3m'] = df['avg_return_3m'].rank(pct=True) * 100
   
   # Momentum acceleration
   if 'avg_return_1m' in df.columns and 'avg_return_3m' in df.columns:
       df['momentum_acceleration'] = (
           df['avg_return_1m'] - (df['avg_return_3m'] / 3)
       ).rank(pct=True) * 100
   else:
       df['momentum_acceleration'] = 50
   
   # Composite momentum score
   df['momentum_score'] = (
       df['momentum_1m'] * 0.4 +
       df['momentum_3m'] * 0.4 +
       df['momentum_acceleration'] * 0.2
   )
   
   return df


def _calculate_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
   """Calculate sector relative strength vs market."""
   # Market average
   if 'sector_market_cap' in df.columns:
       # Weighted average by market cap
       total_mcap = df['sector_market_cap'].sum()
       weights = df['sector_market_cap'] / total_mcap
       market_avg_3m = (df['avg_return_3m'] * weights).sum()
   else:
       # Simple average
       market_avg_3m = df['avg_return_3m'].mean()
   
   # Relative strength
   df['relative_strength'] = df['avg_return_3m'] - market_avg_3m
   df['relative_strength_score'] = df['relative_strength'].rank(pct=True) * 100
   
   # Outperformance flag
   df['outperforming'] = df['relative_strength'] > 0
   
   return df


def _calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
   """Calculate risk-adjusted performance metrics."""
   # Volatility scoring (lower is better)
   if 'volatility_1m' in df.columns:
       df['volatility_score'] = (1 - df['volatility_1m'].rank(pct=True)) * 100
   else:
       # Estimate from return dispersion
       df['volatility_score'] = 50
   
   # Risk-adjusted returns (Sharpe-like ratio)
   df['risk_adjusted_return'] = np.where(
       df.get('volatility_1m', 1) > 0,
       df['avg_return_3m'] / df.get('volatility_1m', 1),
       df['avg_return_3m']
   )
   df['risk_adjusted_score'] = df['risk_adjusted_return'].rank(pct=True) * 100
   
   # Downside risk indicator
   df['downside_risk'] = np.where(
       (df['avg_return_1m'] < 0) & (df.get('volatility_1m', 0) > df.get('volatility_1m', 0).median()),
       1, 0
   )
   
   return df


def _detect_sector_anomalies(df: pd.DataFrame) -> pd.DataFrame:
   """Detect unusual sector behavior and anomalies."""
   anomalies = []
   
   for idx, row in df.iterrows():
       sector_anomalies = []
       
       # Extreme momentum divergence
       if abs(row.get('momentum_1m', 50) - row.get('momentum_3m', 50)) > 40:
           sector_anomalies.append('momentum_divergence')
       
       # Volume surge
       if row.get('avg_volume_surge', 1) > 2:
           sector_anomalies.append('volume_surge')
       
       # Extreme relative performance
       if abs(row.get('relative_strength', 0)) > df['avg_return_3m'].std() * 2:
           sector_anomalies.append('extreme_performance')
       
       # Sudden reversal
       if row.get('avg_return_1m', 0) * row.get('avg_return_3m', 0) < 0:
           if abs(row.get('avg_return_1m', 0)) > 5:
               sector_anomalies.append('trend_reversal')
       
       anomalies.append(sector_anomalies)
   
   df['anomalies'] = anomalies
   df['anomaly_count'] = df['anomalies'].apply(len)
   df['has_anomaly'] = df['anomaly_count'] > 0
   
   return df


def _calculate_rotation_signals(df: pd.DataFrame) -> pd.DataFrame:
   """Generate sector rotation signals and indicators."""
   # Rotation momentum (sectors gaining strength)
   df['rotation_momentum'] = (
       df['momentum_acceleration'] * 0.5 +
       df['relative_strength_score'] * 0.3 +
       df.get('avg_volume_surge', 1).rank(pct=True) * 100 * 0.2
   )
   
   # Entry signal (oversold + turning up)
   df['entry_signal'] = (
       (df['momentum_3m'] < 30) &
       (df['momentum_1m'] > df['momentum_3m']) &
       (df['momentum_acceleration'] > 60)
   ).astype(int)
   
   # Exit signal (overbought + turning down)
   df['exit_signal'] = (
       (df['momentum_3m'] > 70) &
       (df['momentum_1m'] < df['momentum_3m']) &
       (df['momentum_acceleration'] < 40)
   ).astype(int)
   
   # Rotation phase
   df['rotation_phase'] = df.apply(_determine_rotation_phase, axis=1)
   
   return df


def _calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
   """Calculate final composite sector scores."""
   # Component weights
   weights = {
       'momentum': 0.30,
       'quality': 0.25,
       'risk_adjusted': 0.20,
       'relative_strength': 0.15,
       'rotation': 0.10
   }
   
   # Calculate composite score
   df['sector_score'] = (
       df['momentum_score'] * weights['momentum'] +
       df.get('avg_quality_score', 50) * weights['quality'] +
       df['risk_adjusted_score'] * weights['risk_adjusted'] +
       df['relative_strength_score'] * weights['relative_strength'] +
       df['rotation_momentum'] * weights['rotation']
   )
   
   # Final ranking
   df['sector_rank'] = df['sector_score'].rank(ascending=False, method='min').astype(int)
   
   # Percentile ranks for heatmap
   df['percentile_rank'] = df['sector_score'].rank(pct=True) * 100
   
   return df


def _assign_rotation_status(df: pd.DataFrame) -> pd.DataFrame:
   """Assign intelligent rotation status based on multiple factors."""
   def get_status(row):
       score = row['sector_score']
       momentum = row['momentum_score']
       rs = row['relative_strength_score']
       
       # Breakout/Breakdown detection
       if row.get('entry_signal', 0) and score > 60:
           return SectorStatus.BREAKOUT.value
       elif row.get('exit_signal', 0) and score < 40:
           return SectorStatus.BREAKDOWN.value
       
       # Regular classification
       if score >= 80 and momentum >= 70:
           return SectorStatus.HOT.value
       elif score >= 65 and rs >= 60:
           return SectorStatus.WARMING.value
       elif score >= 35 and score < 65:
           return SectorStatus.NEUTRAL.value
       elif score >= 20 and momentum < 40:
           return SectorStatus.COOLING.value
       else:
           return SectorStatus.COLD.value
   
   df['rotation_status'] = df.apply(get_status, axis=1)
   
   # Status score for sorting
   status_scores = {
       SectorStatus.BREAKOUT.value: 7,
       SectorStatus.HOT.value: 6,
       SectorStatus.WARMING.value: 5,
       SectorStatus.NEUTRAL.value: 4,
       SectorStatus.COOLING.value: 3,
       SectorStatus.COLD.value: 2,
       SectorStatus.BREAKDOWN.value: 1
   }
   df['status_score'] = df['rotation_status'].map(status_scores)
   
   return df


def _generate_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
   """Generate actionable edge signals for trading."""
   edge_signals = []
   
   for idx, row in df.iterrows():
       signals = []
       
       # Sector rotation plays
       if row['rotation_status'] == SectorStatus.BREAKOUT.value:
           signals.append(f"BREAKOUT: Enter {row['sector']} longs")
       elif row['rotation_status'] == SectorStatus.BREAKDOWN.value:
           signals.append(f"BREAKDOWN: Exit {row['sector']} positions")
       
       # Momentum plays
       if row['momentum_acceleration'] > 80 and row['momentum_score'] < 70:
           signals.append(f"MOMENTUM: {row['sector']} accelerating")
       
       # Mean reversion plays
       if row['momentum_3m'] < 20 and row.get('avg_quality_score', 50) > 70:
           signals.append(f"VALUE: {row['sector']} oversold quality")
       
       # Relative strength plays
       if row['relative_strength'] > 10 and row['momentum_score'] > 60:
           signals.append(f"STRENGTH: {row['sector']} leading market")
       
       # Anomaly plays
       if 'volume_surge' in row.get('anomalies', []):
           signals.append(f"VOLUME: Unusual activity in {row['sector']}")
       
       edge_signals.append(signals)
   
   df['edge_signals'] = edge_signals
   df['signal_count'] = df['edge_signals'].apply(len)
   
   # Priority score for sorting
   df['edge_priority'] = (
       df['signal_count'] * 10 +
       df['status_score'] +
       df['anomaly_count'] * 5
   )
   
   return df


def _determine_rotation_phase(row: pd.Series) -> str:
   """Determine sector rotation phase."""
   m1 = row.get('momentum_1m', 50)
   m3 = row.get('momentum_3m', 50)
   rs = row.get('relative_strength', 0)
   
   if m3 < 30 and m1 > m3:
       return "Accumulation"
   elif m3 > 30 and m3 < 70 and rs > 0:
       return "Advancing"
   elif m3 > 70 and m1 < m3:
       return "Distribution"
   elif m3 > 50 and rs < 0:
       return "Declining"
   else:
       return "Neutral"


def _format_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
   """Format data for optimal dashboard display."""
   # Select and order columns
   display_columns = [
       'sector',
       'sector_score',
       'sector_rank',
       'rotation_status',
       'rotation_phase',
       'momentum_score',
       'relative_strength',
       'risk_adjusted_score',
       'sector_count',
       'edge_signals',
       'anomaly_count'
   ]
   
   # Only include columns that exist
   final_columns = [col for col in display_columns if col in df.columns]
   
   # Sort by priority
   df = df.sort_values(
       ['edge_priority', 'sector_score'],
       ascending=[False, False]
   ).reset_index(drop=True)
   
   # Round numeric columns
   numeric_columns = df.select_dtypes(include=[np.number]).columns
   df[numeric_columns] = df[numeric_columns].round(2)
   
   # Format edge signals for display
   if 'edge_signals' in df.columns:
       df['edge_summary'] = df['edge_signals'].apply(
           lambda x: ' | '.join(x[:2]) if x else ''
       )
       final_columns.append('edge_summary')
   
   return df[final_columns]


# Dashboard helper functions

def get_sector_heatmap_data(sector_df: pd.DataFrame) -> pd.DataFrame:
   """Prepare data for sector heatmap visualization."""
   metrics = [
       'momentum_score', 'relative_strength_score', 'risk_adjusted_score',
       'volatility_score', 'rotation_momentum'
   ]
   
   heatmap_data = sector_df[['sector'] + [m for m in metrics if m in sector_df.columns]]
   return heatmap_data.set_index('sector')


def get_rotation_summary(sector_df: pd.DataFrame) -> Dict[str, Any]:
   """Get sector rotation summary for dashboard header."""
   return {
       'hot_sectors': sector_df[sector_df['rotation_status'] == 'Hot']['sector'].tolist(),
       'cold_sectors': sector_df[sector_df['rotation_status'] == 'Cold']['sector'].tolist(),
       'breakout_sectors': sector_df[sector_df['rotation_status'] == 'Breakout']['sector'].tolist(),
       'sectors_with_signals': sector_df[sector_df['signal_count'] > 0]['sector'].tolist(),
       'market_breadth': (sector_df['relative_strength'] > 0).mean(),
       'rotation_intensity': sector_df['momentum_acceleration'].std()
   }


def get_sector_recommendations(sector_df: pd.DataFrame, top_n: int = 3) -> List[Dict[str, str]]:
   """Get top sector recommendations with rationale."""
   recommendations = []
   
   # Top momentum plays
   top_momentum = sector_df.nlargest(top_n, 'momentum_score')
   for _, sector in top_momentum.iterrows():
       recommendations.append({
           'sector': sector['sector'],
           'action': 'BUY',
           'rationale': f"Strong momentum ({sector['momentum_score']:.0f}), {sector['rotation_phase']} phase",
           'confidence': 'High' if sector['sector_score'] > 75 else 'Medium'
       })
   
   # Value plays
   oversold = sector_df[
       (sector_df['momentum_3m'] < 30) &
       (sector_df.get('avg_quality_score', 50) > 60)
   ].nlargest(1, 'avg_quality_score')
   
   for _, sector in oversold.iterrows():
       recommendations.append({
           'sector': sector['sector'],
           'action': 'ACCUMULATE',
           'rationale': f"Oversold quality sector, potential reversal",
           'confidence': 'Medium'
       })
   
   return recommendations
