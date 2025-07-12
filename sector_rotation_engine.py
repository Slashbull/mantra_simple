"""
sector_rotation_engine.py - M.A.N.T.R.A. Sector Rotation Engine
==============================================================
Analyzes sector performance, identifies rotation patterns,
and finds sector-based opportunities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

# Import from constants
from constants import SECTOR_GROUPS, MARKET_REGIMES

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ============================================================================
# SECTOR ANALYSIS ENGINE
# ============================================================================

class SectorRotationEngine:
    """
    Analyzes sector rotation patterns and identifies opportunities
    """
    
    def __init__(self):
        self.sector_scores = {}
        self.rotation_signals = {}
        self.market_regime = None
        
    def analyze_sectors(
        self, 
        stocks_df: pd.DataFrame, 
        sector_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Comprehensive sector analysis
        
        Args:
            stocks_df: Stock data with sector information
            sector_df: Sector performance data
            
        Returns:
            Tuple of (enhanced_stocks_df, enhanced_sector_df, analysis_dict)
        """
        if stocks_df.empty or sector_df.empty:
            logger.warning("Empty dataframes provided")
            return stocks_df, sector_df, {}
            
        logger.info("Starting sector rotation analysis...")
        
        # Enhance sector data
        sector_df = self._enhance_sector_data(sector_df)
        
        # Calculate sector momentum
        sector_df = self._calculate_sector_momentum(sector_df)
        
        # Identify rotation patterns
        sector_df = self._identify_rotation_patterns(sector_df)
        
        # Rank sectors
        sector_df = self._rank_sectors(sector_df)
        
        # Detect market regime
        self.market_regime = self._detect_market_regime(stocks_df, sector_df)
        
        # Add sector analysis to stocks
        stocks_df = self._add_sector_analysis_to_stocks(stocks_df, sector_df)
        
        # Generate rotation signals
        rotation_signals = self._generate_rotation_signals(sector_df)
        
        # Create analysis summary
        analysis = self._create_analysis_summary(sector_df, rotation_signals)
        
        logger.info(f"Sector analysis complete. Market regime: {self.market_regime}")
        
        return stocks_df, sector_df, analysis
    
    # ========================================================================
    # SECTOR DATA ENHANCEMENT
    # ========================================================================
    
    def _enhance_sector_data(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated metrics to sector data"""
        df = sector_df.copy()
        
        # Clean percentage columns
        for col in df.columns:
            if 'ret_' in col or 'avg_' in col:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('%', '').astype(float)
        
        # Calculate average returns across timeframes
        timeframes = {
            'short_term': ['sector_ret_1d', 'sector_ret_3d', 'sector_ret_7d'],
            'medium_term': ['sector_ret_30d', 'sector_ret_3m'],
            'long_term': ['sector_ret_6m', 'sector_ret_1y']
        }
        
        for term, cols in timeframes.items():
            available_cols = [col for col in cols if col in df.columns]
            if available_cols:
                df[f'{term}_avg'] = df[available_cols].mean(axis=1)
        
        # Calculate volatility (using available returns)
        return_cols = [col for col in df.columns if 'sector_ret_' in col]
        if return_cols:
            df['volatility'] = df[return_cols].std(axis=1)
        
        # Risk-adjusted returns (simplified Sharpe-like)
        if 'medium_term_avg' in df.columns and 'volatility' in df.columns:
            df['risk_adjusted_return'] = df['medium_term_avg'] / (df['volatility'] + 1)
        
        return df
    
    # ========================================================================
    # MOMENTUM CALCULATION
    # ========================================================================
    
    def _calculate_sector_momentum(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sector momentum scores"""
        df = sector_df.copy()
        
        # Momentum components with weights
        momentum_components = {
            'sector_ret_1d': 0.05,
            'sector_ret_7d': 0.15,
            'sector_ret_30d': 0.30,
            'sector_ret_3m': 0.35,
            'sector_ret_6m': 0.15
        }
        
        # Calculate weighted momentum
        df['momentum_score'] = 0
        total_weight = 0
        
        for col, weight in momentum_components.items():
            if col in df.columns:
                df['momentum_score'] += df[col] * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            df['momentum_score'] /= total_weight
        
        # Momentum acceleration (is momentum increasing?)
        if all(col in df.columns for col in ['sector_ret_7d', 'sector_ret_30d']):
            df['momentum_acceleration'] = (
                df['sector_ret_7d'] / 7 > df['sector_ret_30d'] / 30
            )
        
        # Momentum consistency (all positive or all negative)
        momentum_cols = ['sector_ret_7d', 'sector_ret_30d', 'sector_ret_3m']
        available_mom_cols = [col for col in momentum_cols if col in df.columns]
        if available_mom_cols:
            df['momentum_consistency'] = (
                (df[available_mom_cols] > 0).all(axis=1) |
                (df[available_mom_cols] < 0).all(axis=1)
            )
        
        return df
    
    # ========================================================================
    # ROTATION PATTERN IDENTIFICATION
    # ========================================================================
    
    def _identify_rotation_patterns(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Identify sector rotation patterns"""
        df = sector_df.copy()
        
        # Pattern flags
        df['emerging_strength'] = False
        df['losing_momentum'] = False
        df['rotation_in'] = False
        df['rotation_out'] = False
        df['consolidating'] = False
        
        # Emerging strength: Poor long-term but strong recent
        if all(col in df.columns for col in ['sector_ret_6m', 'sector_ret_30d', 'sector_ret_7d']):
            df['emerging_strength'] = (
                (df['sector_ret_6m'] < 0) &
                (df['sector_ret_30d'] > 5) &
                (df['sector_ret_7d'] > 2)
            )
        
        # Losing momentum: Strong long-term but weak recent
        if all(col in df.columns for col in ['sector_ret_6m', 'sector_ret_30d', 'sector_ret_7d']):
            df['losing_momentum'] = (
                (df['sector_ret_6m'] > 20) &
                (df['sector_ret_30d'] < 0) &
                (df['sector_ret_7d'] < -2)
            )
        
        # Rotation in: Improving relative performance
        if 'momentum_score' in df.columns:
            sector_median = df['momentum_score'].median()
            df['rotation_in'] = (
                (df['momentum_score'] > sector_median) &
                df.get('momentum_acceleration', False)
            )
        
        # Rotation out: Deteriorating relative performance
        if 'momentum_score' in df.columns:
            df['rotation_out'] = (
                (df['momentum_score'] < sector_median) &
                ~df.get('momentum_acceleration', True)
            )
        
        # Consolidating: Low volatility, flat returns
        if all(col in df.columns for col in ['volatility', 'sector_ret_30d']):
            df['consolidating'] = (
                (df['volatility'] < df['volatility'].quantile(0.25)) &
                (df['sector_ret_30d'].abs() < 5)
            )
        
        # Overall rotation signal
        df['rotation_signal'] = 'NEUTRAL'
        df.loc[df['rotation_in'] | df['emerging_strength'], 'rotation_signal'] = 'BUY'
        df.loc[df['rotation_out'] | df['losing_momentum'], 'rotation_signal'] = 'SELL'
        df.loc[df['consolidating'], 'rotation_signal'] = 'HOLD'
        
        return df
    
    # ========================================================================
    # SECTOR RANKING
    # ========================================================================
    
    def _rank_sectors(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Rank sectors by multiple criteria"""
        df = sector_df.copy()
        
        # Ranking criteria
        ranking_metrics = {
            'momentum_rank': ('momentum_score', True),
            'return_rank': ('sector_ret_30d', True),
            'risk_adj_rank': ('risk_adjusted_return', True),
            'volume_rank': ('sector_count', True),  # More stocks = more opportunities
            'volatility_rank': ('volatility', False)  # Lower is better
        }
        
        # Calculate ranks
        for rank_col, (metric_col, ascending) in ranking_metrics.items():
            if metric_col in df.columns:
                df[rank_col] = df[metric_col].rank(ascending=ascending, method='min')
        
        # Composite rank (average of all ranks)
        rank_cols = [col for col in df.columns if col.endswith('_rank')]
        if rank_cols:
            df['composite_rank'] = df[rank_cols].mean(axis=1)
            df['final_rank'] = df['composite_rank'].rank(method='min')
        
        # Categorize sectors
        if 'final_rank' in df.columns:
            total_sectors = len(df)
            df['sector_category'] = pd.cut(
                df['final_rank'],
                bins=[0, total_sectors*0.2, total_sectors*0.4, 
                      total_sectors*0.6, total_sectors*0.8, total_sectors+1],
                labels=['Leaders', 'Strong', 'Neutral', 'Weak', 'Laggards']
            )
        
        return df
    
    # ========================================================================
    # MARKET REGIME DETECTION
    # ========================================================================
    
    def _detect_market_regime(
        self, 
        stocks_df: pd.DataFrame, 
        sector_df: pd.DataFrame
    ) -> str:
        """Detect overall market regime based on sector behavior"""
        
        # Calculate market breadth
        if 'ret_30d' in stocks_df.columns:
            advancing_pct = (stocks_df['ret_30d'] > 0).sum() / len(stocks_df) * 100
            avg_return = stocks_df['ret_30d'].mean()
        else:
            advancing_pct = 50
            avg_return = 0
        
        # Calculate sector dispersion
        if 'sector_ret_30d' in sector_df.columns:
            sector_dispersion = sector_df['sector_ret_30d'].std()
        else:
            sector_dispersion = 0
        
        # Determine regime
        if advancing_pct > 70 and avg_return > 5:
            regime = "Bull Market"
        elif advancing_pct < 30 and avg_return < -5:
            regime = "Bear Market"
        elif sector_dispersion > 10:
            regime = "Rotation Market"
        elif 45 <= advancing_pct <= 55 and abs(avg_return) < 2:
            regime = "Sideways Market"
        else:
            regime = "Transitional"
        
        return regime
    
    # ========================================================================
    # STOCK-LEVEL SECTOR ANALYSIS
    # ========================================================================
    
    def _add_sector_analysis_to_stocks(
        self, 
        stocks_df: pd.DataFrame, 
        sector_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add sector analysis metrics to individual stocks"""
        df = stocks_df.copy()
        
        # Create sector lookup dictionaries
        sector_metrics = {}
        for col in ['momentum_score', 'rotation_signal', 'final_rank', 
                    'sector_category', 'risk_adjusted_return']:
            if col in sector_df.columns:
                sector_metrics[col] = sector_df.set_index('sector')[col].to_dict()
        
        # Map sector metrics to stocks
        if 'sector' in df.columns:
            for metric, mapping in sector_metrics.items():
                df[f'sector_{metric}'] = df['sector'].map(mapping)
        
        # Calculate relative performance within sector
        if 'sector' in df.columns and 'ret_30d' in df.columns:
            df['sector_relative_performance'] = df.groupby('sector')['ret_30d'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
            
            # Flag sector leaders/laggards
            df['is_sector_leader'] = df.groupby('sector')['ret_30d'].transform(
                lambda x: x >= x.quantile(0.8)
            )
            df['is_sector_laggard'] = df.groupby('sector')['ret_30d'].transform(
                lambda x: x <= x.quantile(0.2)
            )
        
        return df
    
    # ========================================================================
    # ROTATION SIGNALS
    # ========================================================================
    
    def _generate_rotation_signals(self, sector_df: pd.DataFrame) -> Dict:
        """Generate actionable rotation signals"""
        signals = {
            'rotate_into': [],
            'rotate_out_of': [],
            'accumulate': [],
            'avoid': []
        }
        
        for _, sector in sector_df.iterrows():
            sector_name = sector['sector']
            signal = sector.get('rotation_signal', 'NEUTRAL')
            
            signal_info = {
                'sector': sector_name,
                'momentum_score': round(sector.get('momentum_score', 0), 2),
                'return_30d': round(sector.get('sector_ret_30d', 0), 2),
                'rank': int(sector.get('final_rank', 0)),
                'category': sector.get('sector_category', 'Unknown')
            }
            
            if signal == 'BUY' and sector.get('emerging_strength', False):
                signals['rotate_into'].append(signal_info)
            elif signal == 'BUY':
                signals['accumulate'].append(signal_info)
            elif signal == 'SELL':
                signals['rotate_out_of'].append(signal_info)
            elif sector.get('sector_category') == 'Laggards':
                signals['avoid'].append(signal_info)
        
        # Sort by momentum score
        for key in signals:
            signals[key] = sorted(
                signals[key], 
                key=lambda x: x['momentum_score'], 
                reverse=True
            )[:5]  # Top 5 only
        
        return signals
    
    # ========================================================================
    # ANALYSIS SUMMARY
    # ========================================================================
    
    def _create_analysis_summary(
        self, 
        sector_df: pd.DataFrame, 
        rotation_signals: Dict
    ) -> Dict:
        """Create comprehensive sector analysis summary"""
        
        # Top performing sectors
        top_sectors = sector_df.nsmallest(5, 'final_rank')[
            ['sector', 'sector_ret_30d', 'momentum_score', 'sector_category']
        ].to_dict('records')
        
        # Bottom performing sectors
        bottom_sectors = sector_df.nlargest(5, 'final_rank')[
            ['sector', 'sector_ret_30d', 'momentum_score', 'sector_category']
        ].to_dict('records')
        
        # Sector statistics
        stats = {
            'total_sectors': len(sector_df),
            'average_return_30d': round(sector_df['sector_ret_30d'].mean(), 2),
            'best_performer': sector_df.nlargest(1, 'sector_ret_30d').iloc[0]['sector'],
            'worst_performer': sector_df.nsmallest(1, 'sector_ret_30d').iloc[0]['sector'],
            'highest_momentum': sector_df.nlargest(1, 'momentum_score').iloc[0]['sector'],
            'most_stocks': sector_df.nlargest(1, 'sector_count').iloc[0]['sector']
        }
        
        # Pattern summary
        patterns = {
            'emerging_strength': sector_df['emerging_strength'].sum(),
            'losing_momentum': sector_df['losing_momentum'].sum(),
            'consolidating': sector_df['consolidating'].sum()
        }
        
        return {
            'market_regime': self.market_regime,
            'rotation_signals': rotation_signals,
            'top_sectors': top_sectors,
            'bottom_sectors': bottom_sectors,
            'statistics': stats,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_sector_rotation(
    stocks_df: pd.DataFrame,
    sector_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main function for sector rotation analysis
    
    Args:
        stocks_df: Stock data
        sector_df: Sector data
        
    Returns:
        Tuple of (enhanced_stocks_df, enhanced_sector_df, analysis)
    """
    engine = SectorRotationEngine()
    return engine.analyze_sectors(stocks_df, sector_df)

def get_sector_leaders(
    stocks_df: pd.DataFrame,
    top_n_sectors: int = 3
) -> pd.DataFrame:
    """
    Get stocks from leading sectors
    
    Args:
        stocks_df: DataFrame with sector analysis
        top_n_sectors: Number of top sectors to include
        
    Returns:
        Filtered DataFrame
    """
    if 'sector_category' not in stocks_df.columns:
        logger.error("No sector analysis found")
        return pd.DataFrame()
    
    # Get stocks from leader sectors
    leaders = stocks_df[
        (stocks_df['sector_category'] == 'Leaders') |
        (stocks_df['sector_category'] == 'Strong')
    ]
    
    # Further filter to sector leaders within those sectors
    if 'is_sector_leader' in leaders.columns:
        leaders = leaders[leaders['is_sector_leader']]
    
    return leaders.sort_values('composite_score', ascending=False)

def get_rotation_opportunities(
    stocks_df: pd.DataFrame,
    signal_type: str = 'rotate_into'
) -> pd.DataFrame:
    """
    Get stocks from sectors with specific rotation signals
    
    Args:
        stocks_df: DataFrame with sector analysis
        signal_type: Type of rotation signal
        
    Returns:
        Filtered DataFrame
    """
    valid_signals = ['rotate_into', 'rotate_out_of', 'accumulate', 'avoid']
    
    if signal_type not in valid_signals:
        logger.error(f"Invalid signal type. Choose from {valid_signals}")
        return pd.DataFrame()
    
    if 'sector_rotation_signal' not in stocks_df.columns:
        logger.error("No rotation signals found")
        return pd.DataFrame()
    
    # Map signal types to rotation signals
    signal_mapping = {
        'rotate_into': 'BUY',
        'rotate_out_of': 'SELL',
        'accumulate': 'BUY',
        'avoid': 'SELL'
    }
    
    target_signal = signal_mapping.get(signal_type, 'NEUTRAL')
    
    return stocks_df[
        stocks_df['sector_rotation_signal'] == target_signal
    ].sort_values('sector_momentum_score', ascending=False)

def get_sector_summary(sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get clean sector summary for display
    
    Args:
        sector_df: Enhanced sector DataFrame
        
    Returns:
        Summary DataFrame
    """
    summary_cols = [
        'sector', 'sector_ret_30d', 'momentum_score', 
        'rotation_signal', 'sector_category', 'final_rank'
    ]
    
    available_cols = [col for col in summary_cols if col in sector_df.columns]
    
    return sector_df[available_cols].sort_values('final_rank')

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Sector Rotation Engine")
    print("="*60)
    print("\nAnalyzes sector performance and rotation patterns")
    print("\nFeatures:")
    print("- Sector momentum scoring")
    print("- Rotation pattern detection")
    print("- Market regime identification")
    print("- Sector ranking and categorization")
    print("- Stock-level sector analysis")
    print("\nUse analyze_sector_rotation() to start analysis")
    print("="*60)
