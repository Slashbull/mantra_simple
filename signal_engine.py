"""
signal_engine.py - M.A.N.T.R.A. Signal Engine
============================================
Fresh, clean implementation of multi-factor scoring
Built from scratch for Indian stock market analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Import constants
from constants import (
    SCORE_WEIGHTS, MOMENTUM_THRESHOLDS, VALUE_THRESHOLDS,
    TECHNICAL_THRESHOLDS, VOLUME_THRESHOLDS, SIGNAL_THRESHOLDS
)

logger = logging.getLogger(__name__)

# ============================================================================
# SIGNAL ENGINE
# ============================================================================

class SignalEngine:
    """
    Multi-factor signal generation engine
    Combines momentum, value, technical, volume, and sector signals
    """
    
    def __init__(self):
        self.scores = {}
        self.explanations = {}
        
    def calculate_all_signals(
        self, 
        stocks_df: pd.DataFrame, 
        sector_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Main method to calculate all signals and composite score
        
        Args:
            stocks_df: Stock data
            sector_df: Sector performance data
            
        Returns:
            DataFrame with all scores and signals
        """
        if stocks_df.empty:
            logger.warning("Empty dataframe provided")
            return stocks_df
            
        df = stocks_df.copy()
        
        # Calculate individual factor scores
        logger.info("Calculating factor scores...")
        
        # 1. Momentum Score
        df['momentum_score'] = self._calculate_momentum_score(df)
        
        # 2. Value Score
        df['value_score'] = self._calculate_value_score(df)
        
        # 3. Technical Score
        df['technical_score'] = self._calculate_technical_score(df)
        
        # 4. Volume Score
        df['volume_score'] = self._calculate_volume_score(df)
        
        # 5. Sector Score
        df['sector_score'] = self._calculate_sector_score(df, sector_df)
        
        # Calculate composite score
        df['composite_score'] = self._calculate_composite_score(df)
        
        # Add signal categories
        df['signal'] = self._categorize_signals(df['composite_score'])
        
        # Add rankings
        df['rank'] = df['composite_score'].rank(ascending=False, method='min')
        df['percentile'] = df['composite_score'].rank(pct=True) * 100
        
        logger.info(f"Signal calculation complete for {len(df)} stocks")
        
        return df
    
    # ========================================================================
    # MOMENTUM SCORING
    # ========================================================================
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum score based on returns over multiple periods
        Higher recent returns = higher score
        """
        score = pd.Series(50.0, index=df.index)
        
        # Define periods and their weights (recent periods weighted more)
        periods = {
            'ret_1d': 0.10,
            'ret_7d': 0.20,
            'ret_30d': 0.30,
            'ret_3m': 0.40
        }
        
        # Calculate weighted momentum
        for period, weight in periods.items():
            if period in df.columns:
                # Normalize returns to 0-100 scale
                returns = df[period].fillna(0)
                
                # Use thresholds from constants
                thresholds = MOMENTUM_THRESHOLDS['BULLISH']
                period_key = period.replace('ret_', '').upper()
                
                # Score based on return strength
                period_score = pd.Series(50.0, index=df.index)
                
                if period_key in thresholds:
                    threshold = thresholds[period_key]
                    # Linear scaling with caps
                    period_score = 50 + (returns / threshold) * 25
                    period_score = period_score.clip(0, 100)
                else:
                    # Fallback: simple percentile ranking
                    period_score = returns.rank(pct=True) * 100
                
                # Add weighted contribution
                score += (period_score - 50) * weight
        
        return score.clip(0, 100)
    
    # ========================================================================
    # VALUE SCORING
    # ========================================================================
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate value score based on PE ratio and EPS growth
        Lower PE + Higher EPS growth = higher score
        """
        score = pd.Series(50.0, index=df.index)
        
        # PE Ratio Score (40% weight)
        if 'pe' in df.columns:
            pe = df['pe'].fillna(df['pe'].median())
            
            # Use thresholds from constants
            pe_thresholds = VALUE_THRESHOLDS['PE_RATIO']
            
            # Invert PE for scoring (lower is better)
            pe_score = pd.Series(50.0, index=df.index)
            
            # Score based on PE ranges
            pe_score[pe < pe_thresholds['DEEP_VALUE']] = 100
            pe_score[pe.between(pe_thresholds['DEEP_VALUE'], pe_thresholds['VALUE'])] = 85
            pe_score[pe.between(pe_thresholds['VALUE'], pe_thresholds['FAIR'])] = 70
            pe_score[pe.between(pe_thresholds['FAIR'], pe_thresholds['EXPENSIVE'])] = 50
            pe_score[pe > pe_thresholds['EXPENSIVE']] = 30
            pe_score[pe < 0] = 20  # Negative PE is bad
            
            score = pe_score * 0.4
        
        # EPS Growth Score (40% weight)
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            
            # Use thresholds from constants
            eps_thresholds = VALUE_THRESHOLDS['EPS_GROWTH']
            
            eps_score = pd.Series(50.0, index=df.index)
            eps_score[eps_growth > eps_thresholds['HYPER']] = 100
            eps_score[eps_growth.between(eps_thresholds['HIGH'], eps_thresholds['HYPER'])] = 85
            eps_score[eps_growth.between(eps_thresholds['MODERATE'], eps_thresholds['HIGH'])] = 70
            eps_score[eps_growth.between(eps_thresholds['LOW'], eps_thresholds['MODERATE'])] = 60
            eps_score[eps_growth < eps_thresholds['NEGATIVE']] = 30
            
            score += eps_score * 0.4
        
        # Price to 52W Range Score (20% weight)
        if 'from_low_pct' in df.columns and 'from_high_pct' in df.columns:
            # Calculate position in 52W range
            position = df['from_low_pct'] / (df['from_low_pct'] + abs(df['from_high_pct']))
            position = position.fillna(0.5) * 100
            
            # Sweet spot is 30-70% of range
            position_score = pd.Series(50.0, index=df.index)
            position_score[position.between(30, 70)] = 70
            position_score[position.between(20, 30) | position.between(70, 80)] = 60
            position_score[position < 20] = 40  # Too close to low
            position_score[position > 80] = 40  # Too close to high
            
            score += position_score * 0.2
        
        return score.clip(0, 100)
    
    # ========================================================================
    # TECHNICAL SCORING
    # ========================================================================
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate technical score based on SMA positions and trends
        """
        score = pd.Series(50.0, index=df.index)
        
        # Check if we have required columns
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        if 'price' in df.columns and all(col in df.columns for col in sma_cols):
            
            # Price vs SMAs (60% weight)
            price_vs_sma20 = ((df['price'] - df['sma_20d']) / df['sma_20d'] * 100).fillna(0)
            price_vs_sma50 = ((df['price'] - df['sma_50d']) / df['sma_50d'] * 100).fillna(0)
            price_vs_sma200 = ((df['price'] - df['sma_200d']) / df['sma_200d'] * 100).fillna(0)
            
            # Score each SMA position
            sma20_score = self._score_sma_position(price_vs_sma20, weight=20)
            sma50_score = self._score_sma_position(price_vs_sma50, weight=20)
            sma200_score = self._score_sma_position(price_vs_sma200, weight=20)
            
            score = sma20_score + sma50_score + sma200_score
            
            # SMA Alignment (40% weight)
            # Perfect bullish: SMA20 > SMA50 > SMA200
            bullish_alignment = (
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            ).astype(float) * 20
            
            # Partial alignment
            partial_alignment = (
                (df['sma_20d'] > df['sma_200d']) |
                (df['sma_50d'] > df['sma_200d'])
            ).astype(float) * 10
            
            # Add alignment bonus
            alignment_score = bullish_alignment + partial_alignment
            score += alignment_score
        
        # Trading position penalty
        if 'trading_under' in df.columns:
            # Penalty for trading under averages
            trading_under_penalty = pd.Series(0.0, index=df.index)
            trading_under_penalty[df['trading_under'].notna()] = -10
            score += trading_under_penalty
        
        return score.clip(0, 100)
    
    def _score_sma_position(self, distance_pct: pd.Series, weight: float) -> pd.Series:
        """Helper to score price distance from SMA"""
        score = pd.Series(0.0, index=distance_pct.index)
        
        # Best: slightly above SMA (2-5%)
        score[distance_pct.between(2, 5)] = weight
        
        # Good: moderately above (5-10%) or just above (0-2%)
        score[distance_pct.between(5, 10) | distance_pct.between(0, 2)] = weight * 0.8
        
        # Neutral: far above (>10%) or slightly below (-2 to 0%)
        score[(distance_pct > 10) | distance_pct.between(-2, 0)] = weight * 0.5
        
        # Bad: below SMA
        score[distance_pct < -2] = weight * 0.2
        
        return score
    
    # ========================================================================
    # VOLUME SCORING
    # ========================================================================
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume score based on relative volume and trends
        """
        score = pd.Series(50.0, index=df.index)
        
        # Relative Volume (50% weight)
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            # Use thresholds from constants
            rvol_thresholds = VOLUME_THRESHOLDS['RVOL_THRESHOLDS']
            
            rvol_score = pd.Series(50.0, index=df.index)
            
            # High volume is generally good (shows interest)
            rvol_score[rvol > rvol_thresholds['EXTREME']] = 90
            rvol_score[rvol.between(rvol_thresholds['HIGH'], rvol_thresholds['EXTREME'])] = 80
            rvol_score[rvol.between(rvol_thresholds['NORMAL'], rvol_thresholds['HIGH'])] = 70
            rvol_score[rvol.between(rvol_thresholds['LOW'], rvol_thresholds['NORMAL'])] = 50
            rvol_score[rvol < rvol_thresholds['LOW']] = 30
            
            score = rvol_score * 0.5
        
        # Volume Trend (50% weight)
        vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        vol_weights = [0.5, 0.3, 0.2]
        
        vol_trend_score = pd.Series(0.0, index=df.index)
        for col, weight in zip(vol_cols, vol_weights):
            if col in df.columns:
                # Parse percentage if needed
                ratio = df[col]
                if ratio.dtype == 'object':
                    ratio = ratio.str.replace('%', '').astype(float)
                
                # Score volume expansion
                col_score = pd.Series(50.0, index=df.index)
                col_score[ratio > 50] = 80  # >50% increase
                col_score[ratio.between(0, 50)] = 60  # Positive
                col_score[ratio.between(-20, 0)] = 50  # Slight decrease
                col_score[ratio < -20] = 30  # Major decrease
                
                vol_trend_score += col_score * weight * 0.5
        
        score += vol_trend_score
        
        return score.clip(0, 100)
    
    # ========================================================================
    # SECTOR SCORING
    # ========================================================================
    
    def _calculate_sector_score(
        self, 
        df: pd.DataFrame, 
        sector_df: Optional[pd.DataFrame]
    ) -> pd.Series:
        """
        Calculate sector strength score
        """
        if sector_df is None or 'sector' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        # Find best available sector metric
        sector_metrics = ['sector_avg_3m', 'sector_avg_6m', 'sector_avg_1y', 'sector_avg_30d']
        metric_col = None
        
        for col in sector_metrics:
            if col in sector_df.columns:
                metric_col = col
                break
        
        if metric_col is None:
            return pd.Series(50.0, index=df.index)
        
        # Create sector performance mapping
        sector_perf = sector_df.set_index('sector')[metric_col]
        
        # Clean percentage values if needed
        if sector_perf.dtype == 'object':
            sector_perf = sector_perf.str.replace('%', '').astype(float)
        
        # Map to stocks
        stock_sector_perf = df['sector'].map(sector_perf).fillna(0)
        
        # Convert to score (percentile ranking)
        ranks = stock_sector_perf.rank(pct=True)
        sector_score = ranks * 100
        
        return sector_score.fillna(50).clip(0, 100)
    
    # ========================================================================
    # COMPOSITE SCORING
    # ========================================================================
    
    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score from all factors
        """
        # Get weights from constants
        weights = SCORE_WEIGHTS.copy()
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        # Calculate weighted sum
        composite = pd.Series(0.0, index=df.index)
        
        factor_mapping = {
            'MOMENTUM': 'momentum_score',
            'VALUE': 'value_score',
            'TECHNICAL': 'technical_score',
            'VOLUME': 'volume_score',
            'SECTOR': 'sector_score'
        }
        
        for factor, col in factor_mapping.items():
            if col in df.columns and factor in weights:
                composite += df[col].fillna(50) * weights[factor]
        
        return composite.round(2)
    
    def _categorize_signals(self, scores: pd.Series) -> pd.Series:
        """
        Categorize stocks into BUY/WATCH/AVOID based on composite score
        """
        conditions = [
            scores >= SIGNAL_THRESHOLDS['BUY'],
            scores >= SIGNAL_THRESHOLDS['WATCH'],
            scores >= SIGNAL_THRESHOLDS['AVOID']
        ]
        
        choices = ['BUY', 'WATCH', 'NEUTRAL']
        
        return pd.Series(
            np.select(conditions, choices, default='AVOID'),
            index=scores.index
        )

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_signals(
    stocks_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate all signals for stocks
    
    Args:
        stocks_df: Stock data
        sector_df: Sector data (optional)
        
    Returns:
        DataFrame with signals and scores
    """
    engine = SignalEngine()
    return engine.calculate_all_signals(stocks_df, sector_df)

def get_top_opportunities(
    scored_df: pd.DataFrame,
    n: int = 20,
    min_score: float = 70.0
) -> pd.DataFrame:
    """
    Get top opportunities based on composite score
    
    Args:
        scored_df: DataFrame with scores
        n: Number of top stocks to return
        min_score: Minimum score threshold
        
    Returns:
        Top opportunities sorted by score
    """
    if 'composite_score' not in scored_df.columns:
        logger.error("No composite_score found in dataframe")
        return pd.DataFrame()
    
    # Filter by minimum score
    filtered = scored_df[scored_df['composite_score'] >= min_score]
    
    # Sort and return top N
    return filtered.nlargest(n, 'composite_score')

def get_signals_by_category(
    scored_df: pd.DataFrame,
    category: str = 'BUY'
) -> pd.DataFrame:
    """
    Get all stocks with a specific signal category
    
    Args:
        scored_df: DataFrame with signals
        category: Signal category (BUY/WATCH/NEUTRAL/AVOID)
        
    Returns:
        Filtered DataFrame
    """
    if 'signal' not in scored_df.columns:
        logger.error("No signal column found in dataframe")
        return pd.DataFrame()
    
    return scored_df[scored_df['signal'] == category].sort_values(
        'composite_score', 
        ascending=False
    )

def get_factor_analysis(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all factor scores
    
    Args:
        scored_df: DataFrame with factor scores
        
    Returns:
        Summary statistics
    """
    factor_cols = [
        'momentum_score', 'value_score', 'technical_score',
        'volume_score', 'sector_score', 'composite_score'
    ]
    
    available_cols = [col for col in factor_cols if col in scored_df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    return scored_df[available_cols].describe().round(2)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Signal Engine")
    print("="*60)
    print("\nMulti-factor scoring system for Indian stocks")
    print("\nFactors:")
    print("- Momentum: Price trends across timeframes")
    print("- Value: PE ratio and EPS growth")
    print("- Technical: SMA positions and alignment")
    print("- Volume: Relative volume and trends")
    print("- Sector: Relative sector performance")
    print("\nUse calculate_signals() to score your stocks")
    print("="*60)
