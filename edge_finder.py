"""
edge_finder.py - Ultimate Edge Detection Module for M.A.N.T.R.A. Stock Intelligence System

Production-ready, data-driven edge detection with adaptive thresholds and comprehensive coverage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EdgeType(Enum):
    """Edge categories for classification"""
    MOMENTUM_BREAKOUT = "Momentum Breakout"
    VALUE_ANOMALY = "Value Anomaly"
    VOLUME_SURGE = "Volume Surge"
    SECTOR_LEADER = "Sector Leader"
    VOLATILITY_SQUEEZE = "Volatility Squeeze"
    TREND_REVERSAL = "Trend Reversal"
    NEW_HIGH = "New 52W High"
    NEW_LOW = "New 52W Low"
    ACCUMULATION = "Accumulation Pattern"
    DISTRIBUTION = "Distribution Warning"
    MEAN_REVERSION = "Mean Reversion Setup"
    RELATIVE_STRENGTH = "Relative Strength"
    BREAKOUT_PULLBACK = "Breakout Pullback"
    OVERSOLD_BOUNCE = "Oversold Bounce"
    OVERBOUGHT_REVERSAL = "Overbought Reversal"


@dataclass
class MarketRegime:
    """Market regime detection for adaptive thresholds"""
    volatility_percentile: float
    trend_strength: float
    breadth: float
    is_trending: bool
    is_volatile: bool


def compute_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main edge detection pipeline - adds all edge signals with adaptive thresholds.
    
    Args:
        df: DataFrame with stock data including price, volume, returns, scores
        
    Returns:
        DataFrame with edge flags, scores, explanations, and summary columns
    """
    df = df.copy()
    
    # Detect market regime for adaptive thresholds
    regime = _detect_market_regime(df)
    
    # Compute percentile-based thresholds
    thresholds = _compute_adaptive_thresholds(df, regime)
    
    # Initialize edge tracking
    edge_flags = {}
    edge_scores = {}
    edge_explanations = {}
    
    # 1. Momentum Breakout Edge
    momentum_edge, momentum_score, momentum_exp = _detect_momentum_breakout(df, thresholds, regime)
    edge_flags['edge_momentum_breakout'] = momentum_edge
    edge_scores['score_momentum_breakout'] = momentum_score
    edge_explanations['exp_momentum_breakout'] = momentum_exp
    
    # 2. Value Anomaly Edge
    value_edge, value_score, value_exp = _detect_value_anomaly(df, thresholds)
    edge_flags['edge_value_anomaly'] = value_edge
    edge_scores['score_value_anomaly'] = value_score
    edge_explanations['exp_value_anomaly'] = value_exp
    
    # 3. Volume Surge Edge
    volume_edge, volume_score, volume_exp = _detect_volume_surge(df, thresholds)
    edge_flags['edge_volume_surge'] = volume_edge
    edge_scores['score_volume_surge'] = volume_score
    edge_explanations['exp_volume_surge'] = volume_exp
    
    # 4. Sector Leadership Edge
    sector_edge, sector_score, sector_exp = _detect_sector_leadership(df, thresholds)
    edge_flags['edge_sector_leader'] = sector_edge
    edge_scores['score_sector_leader'] = sector_score
    edge_explanations['exp_sector_leader'] = sector_exp
    
    # 5. Volatility Squeeze Edge
    vol_edge, vol_score, vol_exp = _detect_volatility_squeeze(df, thresholds, regime)
    edge_flags['edge_volatility_squeeze'] = vol_edge
    edge_scores['score_volatility_squeeze'] = vol_score
    edge_explanations['exp_volatility_squeeze'] = vol_exp
    
    # 6. Trend Reversal Edge
    reversal_edge, reversal_score, reversal_exp = _detect_trend_reversal(df, thresholds)
    edge_flags['edge_trend_reversal'] = reversal_edge
    edge_scores['score_trend_reversal'] = reversal_score
    edge_explanations['exp_trend_reversal'] = reversal_exp
    
    # 7. New High/Low Edges
    high_edge, high_score, high_exp = _detect_new_high(df)
    edge_flags['edge_new_high'] = high_edge
    edge_scores['score_new_high'] = high_score
    edge_explanations['exp_new_high'] = high_exp
    
    low_edge, low_score, low_exp = _detect_new_low(df)
    edge_flags['edge_new_low'] = low_edge
    edge_scores['score_new_low'] = low_score
    edge_explanations['exp_new_low'] = low_exp
    
    # 8. Accumulation Pattern Edge
    acc_edge, acc_score, acc_exp = _detect_accumulation(df, thresholds)
    edge_flags['edge_accumulation'] = acc_edge
    edge_scores['score_accumulation'] = acc_score
    edge_explanations['exp_accumulation'] = acc_exp
    
    # 9. Distribution Warning Edge
    dist_edge, dist_score, dist_exp = _detect_distribution(df, thresholds)
    edge_flags['edge_distribution'] = dist_edge
    edge_scores['score_distribution'] = dist_score
    edge_explanations['exp_distribution'] = dist_exp
    
    # 10. Mean Reversion Setup Edge
    mean_edge, mean_score, mean_exp = _detect_mean_reversion(df, thresholds)
    edge_flags['edge_mean_reversion'] = mean_edge
    edge_scores['score_mean_reversion'] = mean_score
    edge_explanations['exp_mean_reversion'] = mean_exp
    
    # 11. Relative Strength Edge
    rs_edge, rs_score, rs_exp = _detect_relative_strength(df, thresholds)
    edge_flags['edge_relative_strength'] = rs_edge
    edge_scores['score_relative_strength'] = rs_score
    edge_explanations['exp_relative_strength'] = rs_exp
    
    # 12. Breakout Pullback Edge
    pullback_edge, pullback_score, pullback_exp = _detect_breakout_pullback(df, thresholds)
    edge_flags['edge_breakout_pullback'] = pullback_edge
    edge_scores['score_breakout_pullback'] = pullback_score
    edge_explanations['exp_breakout_pullback'] = pullback_exp
    
    # 13. Oversold Bounce Edge
    oversold_edge, oversold_score, oversold_exp = _detect_oversold_bounce(df, thresholds)
    edge_flags['edge_oversold_bounce'] = oversold_edge
    edge_scores['score_oversold_bounce'] = oversold_score
    edge_explanations['exp_oversold_bounce'] = oversold_exp
    
    # 14. Overbought Reversal Edge
    overbought_edge, overbought_score, overbought_exp = _detect_overbought_reversal(df, thresholds)
    edge_flags['edge_overbought_reversal'] = overbought_edge
    edge_scores['score_overbought_reversal'] = overbought_score
    edge_explanations['exp_overbought_reversal'] = overbought_exp
    
    # Add all edge columns to dataframe
    for col, values in edge_flags.items():
        df[col] = values
    for col, values in edge_scores.items():
        df[col] = values
    for col, values in edge_explanations.items():
        df[col] = values
    
    # Create summary columns
    edge_cols = list(edge_flags.keys())
    score_cols = list(edge_scores.keys())
    
    # Edge types list for UI display
    edge_type_map = {
        'edge_momentum_breakout': EdgeType.MOMENTUM_BREAKOUT.value,
        'edge_value_anomaly': EdgeType.VALUE_ANOMALY.value,
        'edge_volume_surge': EdgeType.VOLUME_SURGE.value,
        'edge_sector_leader': EdgeType.SECTOR_LEADER.value,
        'edge_volatility_squeeze': EdgeType.VOLATILITY_SQUEEZE.value,
        'edge_trend_reversal': EdgeType.TREND_REVERSAL.value,
        'edge_new_high': EdgeType.NEW_HIGH.value,
        'edge_new_low': EdgeType.NEW_LOW.value,
        'edge_accumulation': EdgeType.ACCUMULATION.value,
        'edge_distribution': EdgeType.DISTRIBUTION.value,
        'edge_mean_reversion': EdgeType.MEAN_REVERSION.value,
        'edge_relative_strength': EdgeType.RELATIVE_STRENGTH.value,
        'edge_breakout_pullback': EdgeType.BREAKOUT_PULLBACK.value,
        'edge_oversold_bounce': EdgeType.OVERSOLD_BOUNCE.value,
        'edge_overbought_reversal': EdgeType.OVERBOUGHT_REVERSAL.value
    }
    
    # Aggregate edge information
    df['edge_types'] = df.apply(
        lambda r: ', '.join([edge_type_map[col] for col in edge_cols if r.get(col, False)]),
        axis=1
    )
    
    df['has_edge'] = df[edge_cols].any(axis=1)
    df['edge_count'] = df[edge_cols].sum(axis=1)
    
    # Calculate overall edge score (weighted average of individual scores)
    df['edge_score'] = df[score_cols].mean(axis=1, skipna=True).fillna(0)
    
    # Create master explanation combining top edges
    df['edge_explanation'] = df.apply(
        lambda r: _create_master_explanation(r, edge_cols, edge_explanations.keys()),
        axis=1
    )
    
    # Add edge freshness indicator
    df['edge_fresh'] = _detect_edge_freshness(df, edge_cols)
    
    # Add regime context
    df['market_regime'] = _describe_regime(regime)
    
    return df


def find_edges(df: pd.DataFrame, min_edges: int = 1, min_score: float = 0.0) -> pd.DataFrame:
    """
    Filter stocks with significant edges.
    
    Args:
        df: DataFrame with computed edge signals
        min_edges: Minimum number of edges required
        min_score: Minimum edge score required
        
    Returns:
        Filtered DataFrame with edge stocks only
    """
    if 'edge_count' not in df.columns:
        df = compute_edge_signals(df)
    
    mask = (df['edge_count'] >= min_edges) & (df['edge_score'] >= min_score)
    return df[mask].sort_values('edge_score', ascending=False).copy()


def edge_overview(df: pd.DataFrame) -> Dict:
    """
    Generate dashboard-friendly edge summary.
    
    Args:
        df: DataFrame with edge signals
        
    Returns:
        Dictionary with edge statistics and top opportunities
    """
    if 'edge_count' not in df.columns:
        df = compute_edge_signals(df)
    
    # Count edge types
    edge_cols = [col for col in df.columns if col.startswith('edge_') and not col in ['edge_types', 'edge_count', 'edge_score', 'edge_explanation', 'edge_fresh']]
    edge_counts = {col.replace('edge_', ''): df[col].sum() for col in edge_cols}
    
    # Top multi-edge stocks
    top_multi = df[df['edge_count'] >= 2].nlargest(10, 'edge_score')[
        ['ticker', 'edge_types', 'edge_score', 'edge_explanation']
    ].to_dict('records') if len(df[df['edge_count'] >= 2]) > 0 else []
    
    # Edge distribution
    edge_distribution = df['edge_count'].value_counts().to_dict()
    
    # Regime-specific insights
    regime_edges = _get_regime_insights(df)
    
    return {
        'total_edges': int(df['has_edge'].sum()),
        'avg_edge_score': float(df[df['has_edge']]['edge_score'].mean()) if df['has_edge'].any() else 0,
        'edge_type_counts': edge_counts,
        'top_multi_edge': top_multi,
        'edge_distribution': edge_distribution,
        'market_regime': df['market_regime'].iloc[0] if 'market_regime' in df.columns else 'Unknown',
        'regime_insights': regime_edges,
        'fresh_edges': int(df['edge_fresh'].sum()) if 'edge_fresh' in df.columns else 0
    }


def edge_audit(df: pd.DataFrame) -> Dict:
    """
    Audit edge quality and distribution for tuning.
    
    Args:
        df: DataFrame with edge signals
        
    Returns:
        Dictionary with detailed edge analytics
    """
    if 'edge_count' not in df.columns:
        df = compute_edge_signals(df)
    
    edge_cols = [col for col in df.columns if col.startswith('edge_') and not col in ['edge_types', 'edge_count', 'edge_score', 'edge_explanation', 'edge_fresh']]
    
    audit_results = {
        'total_stocks': len(df),
        'stocks_with_edges': int(df['has_edge'].sum()),
        'edge_coverage': float(df['has_edge'].mean()),
        'edge_type_analysis': {},
        'score_distribution': {
            'mean': float(df['edge_score'].mean()),
            'std': float(df['edge_score'].std()),
            'percentiles': {
                '25%': float(df['edge_score'].quantile(0.25)),
                '50%': float(df['edge_score'].quantile(0.50)),
                '75%': float(df['edge_score'].quantile(0.75)),
                '90%': float(df['edge_score'].quantile(0.90)),
                '95%': float(df['edge_score'].quantile(0.95))
            }
        },
        'correlation_matrix': {},
        'sector_edge_distribution': {}
    }
    
    # Analyze each edge type
    for edge in edge_cols:
        edge_name = edge.replace('edge_', '')
        score_col = f'score_{edge_name}'
        
        audit_results['edge_type_analysis'][edge_name] = {
            'count': int(df[edge].sum()),
            'percentage': float(df[edge].mean()),
            'avg_score': float(df[df[edge]][score_col].mean()) if score_col in df.columns and df[edge].any() else 0,
            'with_other_edges': int(df[df[edge] & (df['edge_count'] > 1)].shape[0])
        }
    
    # Edge correlation analysis
    edge_corr = df[edge_cols].corr()
    for i, edge1 in enumerate(edge_cols):
        for j, edge2 in enumerate(edge_cols):
            if i < j:
                corr_val = edge_corr.loc[edge1, edge2]
                if abs(corr_val) > 0.3:  # Only significant correlations
                    key = f"{edge1.replace('edge_', '')}_{edge2.replace('edge_', '')}"
                    audit_results['correlation_matrix'][key] = float(corr_val)
    
    # Sector distribution
    if 'sector' in df.columns:
        sector_edges = df.groupby('sector').agg({
            'has_edge': ['sum', 'mean'],
            'edge_score': 'mean'
        }).round(3)
        audit_results['sector_edge_distribution'] = sector_edges.to_dict()
    
    return audit_results


# Helper Functions

def _detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """Detect current market regime from data"""
    # Calculate market-wide metrics
    avg_volatility = df['ret_7d'].std() if 'ret_7d' in df.columns else 10
    historical_volatility = df['ret_30d'].std() if 'ret_30d' in df.columns else 15
    
    # Trend detection
    advancing = (df.get('ret_7d', 0) > 0).sum() / len(df) if 'ret_7d' in df.columns else 0.5
    strong_trends = (df.get('ret_30d', 0).abs() > 10).sum() / len(df) if 'ret_30d' in df.columns else 0.2
    
    # Volatility percentile
    vol_percentile = min(avg_volatility / historical_volatility, 2.0) if historical_volatility > 0 else 1.0
    
    return MarketRegime(
        volatility_percentile=vol_percentile,
        trend_strength=strong_trends,
        breadth=advancing,
        is_trending=strong_trends > 0.3,
        is_volatile=vol_percentile > 1.2
    )


def _compute_adaptive_thresholds(df: pd.DataFrame, regime: MarketRegime) -> Dict:
    """Compute data-driven adaptive thresholds"""
    thresholds = {}
    
    # Return thresholds
    if 'ret_7d' in df.columns:
        ret_7d_abs = df['ret_7d'].abs()
        thresholds['ret_7d_high'] = ret_7d_abs.quantile(0.9)
        thresholds['ret_7d_extreme'] = ret_7d_abs.quantile(0.95)
    
    if 'ret_30d' in df.columns:
        thresholds['ret_30d_high'] = df['ret_30d'].abs().quantile(0.9)
    
    # Volume thresholds
    if 'vol_ratio_1d_90d' in df.columns:
        thresholds['volume_surge'] = df['vol_ratio_1d_90d'].quantile(0.9)
        thresholds['volume_extreme'] = df['vol_ratio_1d_90d'].quantile(0.95)
    
    # Score thresholds
    if 'final_score' in df.columns:
        thresholds['score_high'] = df['final_score'].quantile(0.8)
        thresholds['score_extreme'] = df['final_score'].quantile(0.95)
    
    # PE thresholds
    if 'pe' in df.columns:
        pe_positive = df[df['pe'] > 0]['pe']
        if len(pe_positive) > 0:
            thresholds['pe_low'] = pe_positive.quantile(0.2)
            thresholds['pe_high'] = pe_positive.quantile(0.8)
    
    # Adjust for regime
    if regime.is_volatile:
        for key in ['ret_7d_high', 'ret_7d_extreme', 'ret_30d_high']:
            if key in thresholds:
                thresholds[key] *= 1.2
    
    return thresholds


def _detect_momentum_breakout(df: pd.DataFrame, thresholds: Dict, regime: MarketRegime) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect momentum breakout patterns"""
    # Multi-timeframe momentum
    short_momentum = df.get('ret_3d', 0) > thresholds.get('ret_7d_high', 5) * 0.5
    medium_momentum = df.get('ret_7d', 0) > thresholds.get('ret_7d_high', 5)
    
    # Price position
    near_high = df.get('price', 0) >= 0.95 * df.get('high_52w', 1)
    above_avg = df.get('price', 0) > df.get('ma_50', 0)
    
    # Volume confirmation
    volume_confirm = df.get('vol_ratio_1d_90d', 0) > 1.5
    
    # Combine signals
    edge = short_momentum & medium_momentum & (near_high | above_avg) & volume_confirm
    
    # Score based on strength
    score = (
        0.3 * (df.get('ret_3d', 0) / thresholds.get('ret_7d_high', 5)).clip(0, 1) +
        0.3 * (df.get('ret_7d', 0) / thresholds.get('ret_7d_high', 5)).clip(0, 1) +
        0.2 * (df.get('price', 0) / df.get('high_52w', 1)).clip(0, 1) +
        0.2 * (df.get('vol_ratio_1d_90d', 0) / 3).clip(0, 1)
    )
    
    # Explanation
    explanation = df.apply(
        lambda r: f"Breaking out: +{r.get('ret_7d', 0):.1f}% weekly on {r.get('vol_ratio_1d_90d', 0):.1f}x volume"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_value_anomaly(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect value anomalies"""
    # Value metrics
    low_pe = (df.get('pe', 99) > 0) & (df.get('pe', 99) < thresholds.get('pe_low', 15))
    high_eps_score = df.get('eps_score', 0) > 80
    strong_fundamentals = df.get('final_score', 0) > thresholds.get('score_high', 70)
    
    # Recent performance not terrible
    not_crashing = df.get('ret_30d', 0) > -20
    
    edge = low_pe & high_eps_score & strong_fundamentals & not_crashing
    
    score = (
        0.4 * (1 - (df.get('pe', 99) / thresholds.get('pe_low', 15))).clip(0, 1) +
        0.3 * (df.get('eps_score', 0) / 100).clip(0, 1) +
        0.3 * (df.get('final_score', 0) / 100).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Value play: PE {r.get('pe', 0):.1f} with EPS score {r.get('eps_score', 0):.0f}"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_volume_surge(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect unusual volume patterns"""
    # Volume spike
    volume_spike = df.get('vol_ratio_1d_90d', 0) > thresholds.get('volume_surge', 3)
    
    # With price movement
    price_move = df.get('ret_1d', 0).abs() > 2
    
    # Not at extremes
    not_extreme = (df.get('rsi', 50) > 20) & (df.get('rsi', 50) < 80)
    
    edge = volume_spike & price_move & not_extreme
    
    score = (
        0.6 * (df.get('vol_ratio_1d_90d', 0) / thresholds.get('volume_extreme', 5)).clip(0, 1) +
        0.4 * (df.get('ret_1d', 0).abs() / 5).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Volume surge: {r.get('vol_ratio_1d_90d', 0):.1f}x avg with {r.get('ret_1d', 0):.1f}% move"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_sector_leadership(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect sector leaders"""
    # High sector score
    sector_leader = df.get('sector_score', 0) > 85
    
    # Strong overall performance
    strong_overall = df.get('final_score', 0) > thresholds.get('score_high', 75)
    
    # Outperforming market
    outperforming = df.get('ret_30d', 0) > df.get('ret_30d', 0).median() + 5
    
    edge = sector_leader & strong_overall & outperforming
    
    score = (
        0.5 * (df.get('sector_score', 0) / 100).clip(0, 1) +
        0.3 * (df.get('final_score', 0) / 100).clip(0, 1) +
        0.2 * ((df.get('ret_30d', 0) - df.get('ret_30d', 0).median()) / 10).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Sector leader: #{r.get('sector_rank', 0):.0f} in sector, score {r.get('sector_score', 0):.0f}"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_volatility_squeeze(df: pd.DataFrame, thresholds: Dict, regime: MarketRegime) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect volatility compression patterns"""
    # Low recent volatility
    low_volatility = df.get('ret_7d', 0).abs() < thresholds.get('ret_7d_high', 5) * 0.3
    tight_range = df.get('ret_30d', 0).abs() < 5
    
    # But increasing volume
    volume_building = df.get('vol_ratio_1d_90d', 0) > 1.2
    
    # Near important levels
    near_resistance = (df.get('price', 0) > 0.95 * df.get('high_52w', 1)) | (df.get('price', 0) > 0.95 * df.get('ma_200', 0))
    
    edge = low_volatility & tight_range & volume_building & near_resistance
    
    # Adjust for regime
    if regime.is_volatile:
        edge = edge & (df.get('vol_ratio_1d_90d', 0) > 1.5)
    
    score = (
        0.4 * (1 - df.get('ret_7d', 0).abs() / thresholds.get('ret_7d_high', 5)).clip(0, 1) +
        0.3 * (df.get('vol_ratio_1d_90d', 0) / 2).clip(0, 1) +
        0.3 * (df.get('price', 0) / df.get('high_52w', 1)).clip(0.9, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Coiling: {r.get('ret_7d', 0):.1f}% weekly range with {r.get('vol_ratio_1d_90d', 0):.1f}x volume"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_trend_reversal(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect potential trend reversals"""
    # Was in downtrend
    was_down = df.get('ret_30d', 0) < -10
    
    # Recent reversal
    recent_up = df.get('ret_3d', 0) > 3
    
    # Volume confirmation
    volume_confirm = df.get('vol_ratio_1d_90d', 0) > 2
    
    # RSI reversal
    rsi_reversal = (df.get('rsi', 50) > 30) & (df.get('rsi', 50) < 50)
    
    edge = was_down & recent_up & volume_confirm & rsi_reversal
    
    score = (
        0.3 * (df.get('ret_3d', 0) / 5).clip(0, 1) +
        0.3 * (df.get('vol_ratio_1d_90d', 0) / 3).clip(0, 1) +
        0.4 * ((df.get('rsi', 50) - 30) / 20).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Reversal: From {r.get('ret_30d', 0):.1f}% monthly to +{r.get('ret_3d', 0):.1f}% recent"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_new_high(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect new 52-week highs"""
    edge = df.get('price', 0) >= df.get('high_52w', 0)
    
    # Score based on breakout strength
    score = pd.Series(1.0, index=df.index) * edge
    
    explanation = df.apply(
        lambda r: f"New 52W high at {r.get('price', 0):.2f}"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_new_low(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect new 52-week lows"""
    edge = df.get('price', 0) <= df.get('low_52w', 999999)
    
    score = pd.Series(0.8, index=df.index) * edge
    
    explanation = df.apply(
        lambda r: f"New 52W low at {r.get('price', 0):.2f}"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_accumulation(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect accumulation patterns"""
    # Steady buying pressure
    positive_days = df.get('ret_1d', 0) > 0
    rising_volume = df.get('vol_ratio_1d_90d', 0) > 1.1
    
    # Price holding well
    above_support = df.get('price', 0) > df.get('ma_50', 0)
    limited_drawdown = df.get('ret_7d', 0) > -3
    
    # Institutional interest
    high_score = df.get('inst_holding_score', 0) > 70
    
    edge = positive_days & rising_volume & above_support & limited_drawdown & high_score
    
    score = (
        0.3 * (df.get('vol_ratio_1d_90d', 0) / 2).clip(0, 1) +
        0.3 * (df.get('inst_holding_score', 0) / 100).clip(0, 1) +
        0.4 * ((df.get('price', 0) - df.get('ma_50', 0)) / df.get('ma_50', 1)).clip(0, 0.1) * 10
    )
    
    explanation = df.apply(
        lambda r: f"Accumulation: Institutional score {r.get('inst_holding_score', 0):.0f} with steady buying"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_distribution(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect distribution patterns"""
    # Selling pressure
    negative_days = df.get('ret_1d', 0) < -1
    high_volume = df.get('vol_ratio_1d_90d', 0) > 2
    
    # Price weakness
    below_avg = df.get('price', 0) < df.get('ma_50', 0)
    downtrend = df.get('ret_7d', 0) < -5
    
    # After run-up
    was_high = df.get('ret_90d', 0) > 20
    
    edge = negative_days & high_volume & below_avg & downtrend & was_high
    
    score = (
        0.4 * (df.get('vol_ratio_1d_90d', 0) / 3).clip(0, 1) +
        0.3 * (abs(df.get('ret_7d', 0)) / 10).clip(0, 1) +
        0.3 * (1 - df.get('price', 0) / df.get('ma_50', 1)).clip(0, 0.2) * 5
    )
    
    explanation = df.apply(
        lambda r: f"Distribution: -{abs(r.get('ret_7d', 0)):.1f}% on {r.get('vol_ratio_1d_90d', 0):.1f}x volume"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_mean_reversion(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect mean reversion setups"""
    # Extended from mean
    far_from_ma = abs(df.get('price', 0) - df.get('ma_50', 0)) / df.get('ma_50', 1) > 0.15
    
    # RSI extreme
    rsi_extreme = (df.get('rsi', 50) < 30) | (df.get('rsi', 50) > 70)
    
    # But fundamentally sound
    good_fundamentals = df.get('final_score', 0) > 60
    
    edge = far_from_ma & rsi_extreme & good_fundamentals
    
    score = (
        0.4 * (abs(df.get('price', 0) - df.get('ma_50', 0)) / df.get('ma_50', 1) / 0.3).clip(0, 1) +
        0.3 * (abs(df.get('rsi', 50) - 50) / 50).clip(0, 1) +
        0.3 * (df.get('final_score', 0) / 100).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Mean reversion: {((r.get('price', 0) - r.get('ma_50', 0)) / r.get('ma_50', 1) * 100):.1f}% from MA50, RSI {r.get('rsi', 50):.0f}"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_relative_strength(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect relative strength patterns"""
    # Outperforming market
    market_median_ret = df.get('ret_30d', 0).median()
    outperforming = df.get('ret_30d', 0) > market_median_ret + 10
    
    # Consistent strength
    positive_momentum = (df.get('ret_7d', 0) > 0) & (df.get('ret_30d', 0) > 0)
    
    # High scores
    strong_scores = df.get('final_score', 0) > thresholds.get('score_high', 75)
    
    edge = outperforming & positive_momentum & strong_scores
    
    score = (
        0.5 * ((df.get('ret_30d', 0) - market_median_ret) / 20).clip(0, 1) +
        0.3 * (df.get('ret_7d', 0) / 10).clip(0, 1) +
        0.2 * (df.get('final_score', 0) / 100).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Rel strength: +{(r.get('ret_30d', 0) - market_median_ret):.1f}% vs market"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_breakout_pullback(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect pullbacks to breakout levels"""
    # Recent breakout
    was_near_high = df.get('ret_30d', 0) > 15
    
    # Pullback
    recent_pullback = (df.get('ret_7d', 0) < 0) & (df.get('ret_7d', 0) > -5)
    
    # At support
    at_support = (df.get('price', 0) > df.get('ma_50', 0) * 0.98) & (df.get('price', 0) < df.get('ma_50', 0) * 1.02)
    
    # Volume drying up
    low_volume = df.get('vol_ratio_1d_90d', 0) < 0.8
    
    edge = was_near_high & recent_pullback & at_support & low_volume
    
    score = (
        0.3 * (df.get('ret_30d', 0) / 30).clip(0, 1) +
        0.3 * (1 - abs(df.get('ret_7d', 0)) / 5).clip(0, 1) +
        0.4 * (1 - abs(df.get('price', 0) - df.get('ma_50', 0)) / df.get('ma_50', 1) / 0.02).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Pullback: Retesting support after {r.get('ret_30d', 0):.1f}% run"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_oversold_bounce(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect oversold bounce setups"""
    # Oversold
    oversold = df.get('rsi', 50) < 30
    
    # But turning
    recent_bounce = df.get('ret_1d', 0) > 1
    
    # Not broken
    above_major_support = df.get('price', 0) > df.get('low_52w', 0) * 1.1
    decent_fundamentals = df.get('final_score', 0) > 50
    
    edge = oversold & recent_bounce & above_major_support & decent_fundamentals
    
    score = (
        0.4 * ((30 - df.get('rsi', 50)) / 30).clip(0, 1) +
        0.3 * (df.get('ret_1d', 0) / 3).clip(0, 1) +
        0.3 * (df.get('final_score', 0) / 100).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Oversold bounce: RSI {r.get('rsi', 50):.0f} with +{r.get('ret_1d', 0):.1f}% turn"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_overbought_reversal(df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Detect overbought reversal patterns"""
    # Overbought
    overbought = df.get('rsi', 50) > 70
    
    # Extreme run
    extreme_run = df.get('ret_7d', 0) > thresholds.get('ret_7d_extreme', 15)
    
    # Showing weakness
    recent_weakness = df.get('ret_1d', 0) < -1
    high_volume = df.get('vol_ratio_1d_90d', 0) > 2
    
    edge = overbought & extreme_run & recent_weakness & high_volume
    
    score = (
        0.3 * ((df.get('rsi', 50) - 70) / 30).clip(0, 1) +
        0.3 * (df.get('ret_7d', 0) / thresholds.get('ret_7d_extreme', 15)).clip(0, 1) +
        0.4 * (abs(df.get('ret_1d', 0)) / 3).clip(0, 1)
    )
    
    explanation = df.apply(
        lambda r: f"Overbought reversal: RSI {r.get('rsi', 50):.0f} after +{r.get('ret_7d', 0):.1f}% run"
        if edge.loc[r.name] else "", axis=1
    )
    
    return edge, score, explanation


def _detect_edge_freshness(df: pd.DataFrame, edge_cols: List[str]) -> pd.Series:
    """Detect if edges are fresh (first occurrence)"""
    # This is a simplified version - in production you'd compare with previous day's data
    # For now, mark edges with extreme scores as "fresh"
    fresh = pd.Series(False, index=df.index)
    
    for col in edge_cols:
        if col in df.columns and col.replace('edge_', 'score_') in df.columns:
            score_col = col.replace('edge_', 'score_')
            fresh |= (df[col] & (df[score_col] > 0.8))
    
    return fresh


def _create_master_explanation(row: pd.Series, edge_cols: List[str], exp_cols: List[str]) -> str:
    """Create combined explanation for multiple edges"""
    explanations = []
    
    for edge_col in edge_cols:
        exp_col = edge_col.replace('edge_', 'exp_')
        if row.get(edge_col, False) and exp_col in row and row[exp_col]:
            explanations.append(row[exp_col])
    
    # Return top 2 explanations
    return ' | '.join(explanations[:2]) if explanations else ''


def _describe_regime(regime: MarketRegime) -> str:
    """Describe market regime in human-readable format"""
    if regime.is_volatile and regime.is_trending:
        return "Volatile Trending"
    elif regime.is_volatile:
        return "High Volatility"
    elif regime.is_trending:
        return "Trending"
    else:
        return "Range-Bound"


def _get_regime_insights(df: pd.DataFrame) -> Dict:
    """Get regime-specific edge insights"""
    insights = {
        'recommended_edges': [],
        'caution_edges': []
    }
    
    if 'market_regime' not in df.columns:
        return insights
    
    regime = df['market_regime'].iloc[0] if len(df) > 0 else 'Unknown'
    
    if regime == "Volatile Trending":
        insights['recommended_edges'] = ['Momentum Breakout', 'Trend Following', 'Relative Strength']
        insights['caution_edges'] = ['Mean Reversion', 'Volatility Squeeze']
    elif regime == "Range-Bound":
        insights['recommended_edges'] = ['Mean Reversion', 'Oversold Bounce', 'Overbought Reversal']
        insights['caution_edges'] = ['Momentum Breakout', 'New High']
    
    return insights


# Example usage
if __name__ == "__main__":
    # Example: Load your data
    # df = pd.read_csv('stock_data.csv')
    
    # Add edge signals
    # df = compute_edge_signals(df)
    
    # Find multi-edge opportunities
    # top_edges = find_edges(df, min_edges=2, min_score=0.7)
    
    # Get dashboard summary
    # summary = edge_overview(df)
    # print(summary)
    
    # Audit edge distribution
    # audit = edge_audit(df)
    # print(audit)
    
    pass
