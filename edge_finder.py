"""
edge_finder.py - M.A.N.T.R.A. Edge Finder
=========================================
Identifies special trading setups and high-probability opportunities
Combines multiple factors to find stocks with statistical edge
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime

# Import from constants
from constants import (
    MOMENTUM_LEVELS, VALUE_THRESHOLDS, VOLUME_LEVELS,
    SECTOR_GROUPS, MARKET_CAP_RANGES
)

logger = logging.getLogger(__name__)

# ============================================================================
# EDGE SETUP DEFINITIONS
# ============================================================================

@dataclass
class EdgeSetup:
    """Definition of a trading edge setup"""
    name: str
    description: str
    conditions: Dict[str, any]
    expected_return: float
    holding_period: str
    success_rate: float
    risk_level: str

# Predefined edge setups
EDGE_SETUPS = {
    "momentum_breakout": EdgeSetup(
        name="Momentum Breakout",
        description="Strong momentum with volume on technical breakout",
        conditions={
            "min_momentum_score": 80,
            "min_volume_spike": 2.0,
            "sma_breakout": True,
            "near_52w_high": True
        },
        expected_return=15.0,
        holding_period="2-4 weeks",
        success_rate=65.0,
        risk_level="Medium"
    ),
    
    "value_reversal": EdgeSetup(
        name="Value Reversal",
        description="Oversold value stock showing reversal signs",
        conditions={
            "max_pe": 20,
            "min_position_52w": 0,
            "max_position_52w": 30,
            "positive_reversal": True
        },
        expected_return=20.0,
        holding_period="1-3 months",
        success_rate=60.0,
        risk_level="Medium-High"
    ),
    
    "earnings_surprise": EdgeSetup(
        name="Earnings Surprise Play",
        description="Strong earnings beat with positive price reaction",
        conditions={
            "min_eps_change": 25,
            "positive_price_reaction": True,
            "volume_expansion": True
        },
        expected_return=12.0,
        holding_period="1-2 weeks",
        success_rate=70.0,
        risk_level="Low-Medium"
    ),
    
    "sector_leader": EdgeSetup(
        name="Sector Leadership",
        description="Leading stock in hot sector",
        conditions={
            "top_sector_performer": True,
            "sector_outperformance": 5.0,
            "relative_strength": 80
        },
        expected_return=18.0,
        holding_period="1-2 months",
        success_rate=62.0,
        risk_level="Medium"
    ),
    
    "institutional_accumulation": EdgeSetup(
        name="Institutional Accumulation",
        description="Large cap with steady accumulation pattern",
        conditions={
            "min_market_cap": 10000,  # 10,000 Cr
            "volume_trend_positive": True,
            "price_stability": True,
            "higher_lows": True
        },
        expected_return=25.0,
        holding_period="3-6 months",
        success_rate=58.0,
        risk_level="Low"
    ),
    
    "small_cap_rocket": EdgeSetup(
        name="Small Cap Rocket",
        description="Small cap with explosive momentum",
        conditions={
            "max_market_cap": 2000,  # 2,000 Cr
            "min_ret_30d": 20,
            "min_volume_increase": 100,
            "technical_strength": True
        },
        expected_return=30.0,
        holding_period="2-4 weeks",
        success_rate=45.0,
        risk_level="High"
    ),
    
    "turnaround_play": EdgeSetup(
        name="Turnaround Story",
        description="Beaten down stock showing recovery signs",
        conditions={
            "min_decline_from_high": 50,
            "positive_eps_trend": True,
            "volume_returning": True,
            "breaking_downtrend": True
        },
        expected_return=40.0,
        holding_period="3-12 months",
        success_rate=40.0,
        risk_level="High"
    ),
    
    "defensive_value": EdgeSetup(
        name="Defensive Value",
        description="Stable dividend stock at attractive valuation",
        conditions={
            "defensive_sector": True,
            "max_pe": 18,
            "min_market_cap": 5000,
            "low_volatility": True
        },
        expected_return=12.0,
        holding_period="6-12 months",
        success_rate=75.0,
        risk_level="Low"
    )
}

# ============================================================================
# EDGE FINDER ENGINE
# ============================================================================

class EdgeFinder:
    """
    Finds stocks matching predefined edge setups
    """
    
    def __init__(self, setups: Optional[Dict[str, EdgeSetup]] = None):
        self.setups = setups or EDGE_SETUPS
        self.edges_found = []
        
    def find_all_edges(self, df: pd.DataFrame, sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Find all edge setups in the data
        
        Args:
            df: DataFrame with analyzed stock data
            sector_df: Sector performance data
            
        Returns:
            DataFrame with edge setup identification
        """
        if df.empty:
            logger.warning("Empty dataframe provided to edge finder")
            return df
            
        df = df.copy()
        logger.info(f"Searching for edge setups in {len(df)} stocks...")
        
        # Prepare additional metrics needed
        self._prepare_edge_metrics(df, sector_df)
        
        # Check each setup
        for setup_name, setup in self.setups.items():
            logger.debug(f"Checking {setup_name} setup...")
            df[f'edge_{setup_name}'] = df.apply(
                lambda row: self._check_setup(row, setup),
                axis=1
            )
        
        # Aggregate edge information
        edge_columns = [col for col in df.columns if col.startswith('edge_')]
        df['edge_count'] = df[edge_columns].sum(axis=1)
        df['edge_setups'] = df.apply(
            lambda row: self._get_edge_names(row, edge_columns),
            axis=1
        )
        
        # Calculate edge score
        df['edge_score'] = df.apply(self._calculate_edge_score, axis=1)
        
        # Add best edge details
        df['best_edge'] = df.apply(self._get_best_edge, axis=1)
        df['edge_expected_return'] = df.apply(self._get_expected_return, axis=1)
        df['edge_holding_period'] = df.apply(self._get_holding_period, axis=1)
        df['edge_risk_level'] = df.apply(self._get_risk_level, axis=1)
        
        # Log summary
        total_edges = (df['edge_count'] > 0).sum()
        logger.info(f"Found {total_edges} stocks with edge setups")
        
        return df
    
    # ========================================================================
    # METRIC PREPARATION
    # ========================================================================
    
    def _prepare_edge_metrics(self, df: pd.DataFrame, sector_df: Optional[pd.DataFrame]):
        """Prepare additional metrics needed for edge detection"""
        
        # Market cap in Crores
        if 'market_cap' in df.columns:
            df['market_cap_cr'] = df['market_cap'] / 1e7  # Convert to Crores
        
        # Volume trend
        if all(col in df.columns for col in ['volume_1d', 'volume_30d']):
            df['volume_trend_positive'] = df['volume_1d'] > df['volume_30d'] * 1.2
        
        # Price stability (low daily volatility)
        if 'ret_1d' in df.columns:
            df['price_stability'] = df['ret_1d'].abs() < 2
        
        # Higher lows pattern (simplified)
        if all(col in df.columns for col in ['low_52w', 'price']):
            df['higher_lows'] = df['price'] > df['low_52w'] * 1.1
        
        # Technical strength
        if 'technical_score' in df.columns:
            df['technical_strength'] = df['technical_score'] > 70
        
        # Positive EPS trend
        if 'eps_change_pct' in df.columns:
            df['positive_eps_trend'] = df['eps_change_pct'] > 0
        
        # Volume returning (for turnarounds)
        if 'rvol' in df.columns:
            df['volume_returning'] = df['rvol'] > 0.8
        
        # Breaking downtrend (simplified)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['breaking_downtrend'] = (df['ret_30d'] < 0) & (df['ret_7d'] > 3)
        
        # Defensive sectors
        if 'sector' in df.columns:
            defensive_sectors = SECTOR_GROUPS.get('Defensive', [])
            df['defensive_sector'] = df['sector'].isin(defensive_sectors)
        
        # Low volatility (using return std as proxy)
        return_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        if all(col in df.columns for col in return_cols):
            df['volatility'] = df[return_cols].std(axis=1)
            df['low_volatility'] = df['volatility'] < 5
        
        # Sector outperformance
        if sector_df is not None and 'sector' in df.columns:
            sector_perf = sector_df.set_index('sector')['sector_ret_30d'].to_dict()
            df['sector_performance'] = df['sector'].map(sector_perf).fillna(0)
            market_avg = df['ret_30d'].mean() if 'ret_30d' in df.columns else 0
            df['sector_outperformance'] = df['sector_performance'] - market_avg
        
        # Top sector performer
        if 'sector' in df.columns and 'ret_30d' in df.columns:
            df['sector_rank'] = df.groupby('sector')['ret_30d'].rank(ascending=False)
            df['top_sector_performer'] = df['sector_rank'] <= 3
        
        # Relative strength
        if 'percentile' in df.columns:
            df['relative_strength'] = df['percentile']
        elif 'composite_score' in df.columns:
            df['relative_strength'] = df['composite_score']
        
        # Decline from high
        if 'from_high_pct' in df.columns:
            df['min_decline_from_high'] = abs(df['from_high_pct'])
        
        # Volume increase
        if all(col in df.columns for col in ['volume_1d', 'volume_30d']):
            df['min_volume_increase'] = (
                (df['volume_1d'] - df['volume_30d']) / df['volume_30d'] * 100
            ).fillna(0)
        
        # Positive reversal
        if all(col in df.columns for col in ['position_52w', 'ret_7d']):
            df['positive_reversal'] = (df['position_52w'] < 30) & (df['ret_7d'] > 3)
        
        # SMA breakout
        if 'distance_from_sma_200d' in df.columns:
            df['sma_breakout'] = df['distance_from_sma_200d'] > 2
        
        # Near 52w high
        if 'position_52w' in df.columns:
            df['near_52w_high'] = df['position_52w'] > 85
        
        # Positive price reaction
        if 'ret_1d' in df.columns:
            df['positive_price_reaction'] = df['ret_1d'] > 1
        
        # Volume expansion
        if 'rvol' in df.columns:
            df['volume_expansion'] = df['rvol'] > 1.5
    
    # ========================================================================
    # SETUP CHECKING
    # ========================================================================
    
    def _check_setup(self, row: pd.Series, setup: EdgeSetup) -> bool:
        """Check if a stock matches a specific edge setup"""
        for condition, value in setup.conditions.items():
            if condition.startswith('min_'):
                # Minimum value check
                col = condition.replace('min_', '')
                if col not in row or pd.isna(row[col]) or row[col] < value:
                    return False
                    
            elif condition.startswith('max_'):
                # Maximum value check
                col = condition.replace('max_', '')
                if col not in row or pd.isna(row[col]) or row[col] > value:
                    return False
                    
            else:
                # Boolean or equality check
                if condition not in row or row[condition] != value:
                    return False
        
        return True
    
    def _get_edge_names(self, row: pd.Series, edge_columns: List[str]) -> str:
        """Get names of all edge setups matched"""
        edges = []
        for col in edge_columns:
            if row[col]:
                edge_name = col.replace('edge_', '').replace('_', ' ').title()
                edges.append(edge_name)
        
        return " | ".join(edges) if edges else "None"
    
    # ========================================================================
    # SCORING AND DETAILS
    # ========================================================================
    
    def _calculate_edge_score(self, row: pd.Series) -> float:
        """Calculate overall edge score based on setups matched"""
        score = 0
        edge_columns = [col for col in row.index if col.startswith('edge_')]
        
        for col in edge_columns:
            if row[col]:
                setup_name = col.replace('edge_', '')
                if setup_name in self.setups:
                    setup = self.setups[setup_name]
                    # Weight by success rate and expected return
                    score += (setup.success_rate * setup.expected_return) / 100
        
        return round(score, 1)
    
    def _get_best_edge(self, row: pd.Series) -> str:
        """Get the best edge setup for a stock"""
        best_edge = None
        best_score = 0
        
        edge_columns = [col for col in row.index if col.startswith('edge_')]
        
        for col in edge_columns:
            if row[col]:
                setup_name = col.replace('edge_', '')
                if setup_name in self.setups:
                    setup = self.setups[setup_name]
                    score = setup.success_rate * setup.expected_return
                    if score > best_score:
                        best_score = score
                        best_edge = setup.name
        
        return best_edge or "None"
    
    def _get_expected_return(self, row: pd.Series) -> float:
        """Get expected return for best edge setup"""
        best_edge = row.get('best_edge', 'None')
        
        for setup in self.setups.values():
            if setup.name == best_edge:
                return setup.expected_return
        
        return 0.0
    
    def _get_holding_period(self, row: pd.Series) -> str:
        """Get recommended holding period for best edge setup"""
        best_edge = row.get('best_edge', 'None')
        
        for setup in self.setups.values():
            if setup.name == best_edge:
                return setup.holding_period
        
        return "N/A"
    
    def _get_risk_level(self, row: pd.Series) -> str:
        """Get risk level for best edge setup"""
        best_edge = row.get('best_edge', 'None')
        
        for setup in self.setups.values():
            if setup.name == best_edge:
                return setup.risk_level
        
        return "Unknown"

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def find_edges(
    df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    custom_setups: Optional[Dict[str, EdgeSetup]] = None
) -> pd.DataFrame:
    """
    Find all edge setups in stock data
    
    Args:
        df: DataFrame with analyzed stock data
        sector_df: Sector performance data
        custom_setups: Custom edge setups (optional)
        
    Returns:
        DataFrame with edge setup identification
    """
    finder = EdgeFinder(custom_setups)
    return finder.find_all_edges(df, sector_df)

def get_stocks_with_edge(
    df: pd.DataFrame,
    min_edge_score: float = 10.0,
    setup_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get stocks with specific edge setups
    
    Args:
        df: DataFrame with edge analysis
        min_edge_score: Minimum edge score
        setup_names: Specific setups to filter
        
    Returns:
        Filtered DataFrame
    """
    if 'edge_score' not in df.columns:
        logger.error("No edge_score column found")
        return pd.DataFrame()
    
    # Filter by score
    filtered = df[df['edge_score'] >= min_edge_score]
    
    # Filter by specific setups if provided
    if setup_names:
        mask = filtered['edge_setups'].str.contains('|'.join(setup_names), case=False)
        filtered = filtered[mask]
    
    return filtered.sort_values('edge_score', ascending=False)

def get_edge_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about edge setups found
    
    Args:
        df: DataFrame with edge analysis
        
    Returns:
        Dictionary with statistics
    """
    if 'edge_count' not in df.columns:
        return {'error': 'No edge analysis found'}
    
    # Count each setup type
    setup_counts = {}
    for setup_name in EDGE_SETUPS.keys():
        col = f'edge_{setup_name}'
        if col in df.columns:
            setup_counts[setup_name] = df[col].sum()
    
    stats = {
        'total_stocks': len(df),
        'stocks_with_edges': (df['edge_count'] > 0).sum(),
        'avg_edge_score': df[df['edge_count'] > 0]['edge_score'].mean(),
        'setup_counts': setup_counts,
        'risk_distribution': df[df['edge_count'] > 0]['edge_risk_level'].value_counts().to_dict(),
        'top_edges': df.nlargest(10, 'edge_score')[
            ['ticker', 'edge_score', 'best_edge', 'edge_expected_return']
        ].to_dict('records')
    }
    
    return stats

def get_edge_by_risk(
    df: pd.DataFrame,
    risk_levels: List[str] = ['Low', 'Low-Medium']
) -> pd.DataFrame:
    """
    Get edge setups filtered by risk level
    
    Args:
        df: DataFrame with edge analysis
        risk_levels: Acceptable risk levels
        
    Returns:
        Filtered DataFrame
    """
    if 'edge_risk_level' not in df.columns:
        logger.error("No edge_risk_level column found")
        return pd.DataFrame()
    
    return df[
        (df['edge_count'] > 0) & 
        (df['edge_risk_level'].isin(risk_levels))
    ].sort_values('edge_score', ascending=False)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Edge Finder")
    print("="*60)
    print("\nIdentifies high-probability trading setups")
    print("\nEdge Setups:")
    for setup_name, setup in EDGE_SETUPS.items():
        print(f"\n{setup.name}:")
        print(f"  - {setup.description}")
        print(f"  - Expected Return: {setup.expected_return}%")
        print(f"  - Success Rate: {setup.success_rate}%")
        print(f"  - Risk Level: {setup.risk_level}")
    print("\nUse find_edges() to identify opportunities")
    print("="*60)
