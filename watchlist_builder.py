"""
watchlist_builder.py - M.A.N.T.R.A. Watchlist Builder
====================================================
Creates prioritized watchlists based on various criteria
Generates top opportunities across different strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import from constants
from constants import (
    SIGNAL_LEVELS, MARKET_CAP_RANGES, SECTOR_GROUPS,
    PRICE_RANGE_FILTERS, VOLUME_LEVELS
)

logger = logging.getLogger(__name__)

# ============================================================================
# WATCHLIST CONFIGURATIONS
# ============================================================================

@dataclass
class WatchlistConfig:
    """Configuration for watchlist generation"""
    name: str
    description: str
    max_stocks: int = 20
    min_score: float = 70
    max_risk: float = 70
    min_volume: int = 50000
    filters: Dict = None
    sort_by: str = 'opportunity_score'
    sort_ascending: bool = False

# Predefined watchlist templates
WATCHLIST_TEMPLATES = {
    "top_buys": WatchlistConfig(
        name="Top Buy Signals",
        description="Highest conviction buy opportunities",
        max_stocks=20,
        min_score=80,
        max_risk=60,
        min_volume=100000,
        filters={'decision': 'BUY'},
        sort_by='opportunity_score'
    ),
    
    "momentum_leaders": WatchlistConfig(
        name="Momentum Leaders",
        description="Stocks with strongest momentum",
        max_stocks=15,
        min_score=75,
        filters={'momentum_score': (80, 100)},
        sort_by='momentum_score'
    ),
    
    "value_picks": WatchlistConfig(
        name="Value Opportunities",
        description="Undervalued stocks with potential",
        max_stocks=15,
        min_score=70,
        filters={'value_score': (80, 100), 'pe': (0, 20)},
        sort_by='value_score'
    ),
    
    "breakout_alerts": WatchlistConfig(
        name="Breakout Alerts",
        description="Stocks breaking out with volume",
        max_stocks=10,
        filters={'edge_momentum_breakout': True, 'volume_spike': True},
        sort_by='composite_score'
    ),
    
    "sector_leaders": WatchlistConfig(
        name="Sector Leaders",
        description="Best stocks in top sectors",
        max_stocks=20,
        filters={'is_sector_leader': True, 'sector_category': ['Leaders', 'Strong']},
        sort_by='sector_relative_performance'
    ),
    
    "small_cap_gems": WatchlistConfig(
        name="Small Cap Opportunities",
        description="High potential small caps",
        max_stocks=15,
        min_score=75,
        filters={'market_cap': (0, 5000)},  # Under 5000 Cr
        sort_by='opportunity_score'
    ),
    
    "recovery_plays": WatchlistConfig(
        name="Recovery Candidates",
        description="Oversold stocks showing recovery",
        max_stocks=10,
        filters={'reversal_pattern': True, 'position_52w': (0, 30)},
        sort_by='ret_7d'
    ),
    
    "quality_growth": WatchlistConfig(
        name="Quality Growth",
        description="High quality growth stocks",
        max_stocks=15,
        min_score=80,
        filters={'eps_change_pct': (20, 1000), 'pe': (10, 35)},
        sort_by='composite_score'
    ),
    
    "dividend_aristocrats": WatchlistConfig(
        name="Dividend Plays",
        description="Stable large caps for income",
        max_stocks=10,
        filters={'market_cap': (20000, 1000000), 'defensive_sector': True},
        sort_by='value_score'
    ),
    
    "high_volume_movers": WatchlistConfig(
        name="Volume Surge",
        description="Stocks with unusual volume activity",
        max_stocks=10,
        filters={'rvol': (2, 100)},
        sort_by='volume_1d',
        sort_ascending=False
    )
}

# ============================================================================
# WATCHLIST BUILDER
# ============================================================================

class WatchlistBuilder:
    """
    Builds customized watchlists based on various criteria
    """
    
    def __init__(self):
        self.watchlists = {}
        self.build_time = None
        
    def build_all_watchlists(
        self, 
        df: pd.DataFrame,
        templates: Optional[Dict[str, WatchlistConfig]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Build all predefined watchlists
        
        Args:
            df: DataFrame with complete analysis
            templates: Watchlist templates to use
            
        Returns:
            Dictionary of watchlists
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return {}
            
        templates = templates or WATCHLIST_TEMPLATES
        self.build_time = datetime.now()
        
        logger.info(f"Building {len(templates)} watchlists...")
        
        for template_name, config in templates.items():
            try:
                watchlist = self._build_single_watchlist(df, config)
                self.watchlists[template_name] = watchlist
                logger.info(f"Built {config.name}: {len(watchlist)} stocks")
            except Exception as e:
                logger.error(f"Failed to build {config.name}: {e}")
                self.watchlists[template_name] = pd.DataFrame()
        
        return self.watchlists
    
    def _build_single_watchlist(
        self, 
        df: pd.DataFrame, 
        config: WatchlistConfig
    ) -> pd.DataFrame:
        """Build a single watchlist based on configuration"""
        filtered_df = df.copy()
        
        # Apply basic filters
        if config.min_score and 'composite_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['composite_score'] >= config.min_score]
        
        if config.max_risk and 'risk_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['risk_score'] <= config.max_risk]
        
        if config.min_volume and 'volume_1d' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['volume_1d'] >= config.min_volume]
        
        # Apply custom filters
        if config.filters:
            filtered_df = self._apply_filters(filtered_df, config.filters)
        
        # Sort
        if config.sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                config.sort_by, 
                ascending=config.sort_ascending
            )
        
        # Limit to max stocks
        filtered_df = filtered_df.head(config.max_stocks)
        
        # Add watchlist metadata
        filtered_df['watchlist'] = config.name
        filtered_df['watchlist_rank'] = range(1, len(filtered_df) + 1)
        
        return filtered_df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply custom filters to dataframe"""
        filtered = df.copy()
        
        for column, condition in filters.items():
            if column not in filtered.columns:
                continue
                
            if isinstance(condition, bool):
                # Boolean filter
                filtered = filtered[filtered[column] == condition]
                
            elif isinstance(condition, (list, tuple)) and len(condition) == 2:
                # Range filter
                min_val, max_val = condition
                if min_val is not None:
                    filtered = filtered[filtered[column] >= min_val]
                if max_val is not None:
                    filtered = filtered[filtered[column] <= max_val]
                    
            elif isinstance(condition, list):
                # List filter (isin)
                filtered = filtered[filtered[column].isin(condition)]
                
            elif isinstance(condition, (int, float, str)):
                # Exact match
                filtered = filtered[filtered[column] == condition]
        
        return filtered
    
    # ========================================================================
    # SPECIALIZED WATCHLISTS
    # ========================================================================
    
    def build_custom_watchlist(
        self,
        df: pd.DataFrame,
        name: str,
        filters: Dict,
        max_stocks: int = 20,
        sort_by: str = 'composite_score'
    ) -> pd.DataFrame:
        """
        Build a custom watchlist with user-defined criteria
        
        Args:
            df: DataFrame with analysis
            name: Watchlist name
            filters: Dictionary of filters
            max_stocks: Maximum stocks
            sort_by: Column to sort by
            
        Returns:
            Custom watchlist DataFrame
        """
        config = WatchlistConfig(
            name=name,
            description="Custom watchlist",
            max_stocks=max_stocks,
            filters=filters,
            sort_by=sort_by
        )
        
        return self._build_single_watchlist(df, config)
    
    def build_sector_watchlist(
        self,
        df: pd.DataFrame,
        sectors: List[str],
        max_per_sector: int = 5
    ) -> pd.DataFrame:
        """Build watchlist with top stocks from specific sectors"""
        if 'sector' not in df.columns:
            logger.error("No sector column found")
            return pd.DataFrame()
        
        sector_stocks = []
        
        for sector in sectors:
            sector_df = df[df['sector'] == sector]
            if not sector_df.empty:
                top_stocks = sector_df.nlargest(max_per_sector, 'composite_score')
                sector_stocks.append(top_stocks)
        
        if sector_stocks:
            watchlist = pd.concat(sector_stocks, ignore_index=True)
            watchlist['watchlist'] = 'Sector Focus'
            return watchlist.sort_values('composite_score', ascending=False)
        
        return pd.DataFrame()
    
    def build_market_cap_watchlist(
        self,
        df: pd.DataFrame,
        cap_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, pd.DataFrame]:
        """Build watchlists segmented by market cap"""
        watchlists = {}
        
        if 'market_cap' not in df.columns:
            logger.error("No market_cap column found")
            return watchlists
        
        for cap_name, (min_cap, max_cap) in cap_ranges.items():
            cap_df = df[
                (df['market_cap'] >= min_cap) & 
                (df['market_cap'] <= max_cap)
            ]
            
            if not cap_df.empty:
                watchlist = cap_df.nlargest(20, 'composite_score')
                watchlist['watchlist'] = f'{cap_name} Cap Focus'
                watchlists[cap_name] = watchlist
        
        return watchlists
    
    # ========================================================================
    # WATCHLIST ANALYTICS
    # ========================================================================
    
    def get_watchlist_summary(self) -> pd.DataFrame:
        """Get summary of all watchlists"""
        if not self.watchlists:
            return pd.DataFrame()
        
        summary_data = []
        
        for name, watchlist in self.watchlists.items():
            if watchlist.empty:
                continue
                
            summary = {
                'watchlist': name,
                'stocks_count': len(watchlist),
                'avg_score': watchlist['composite_score'].mean() if 'composite_score' in watchlist.columns else 0,
                'avg_return_30d': watchlist['ret_30d'].mean() if 'ret_30d' in watchlist.columns else 0,
                'top_stock': watchlist.iloc[0]['ticker'] if len(watchlist) > 0 else '',
                'sectors': watchlist['sector'].nunique() if 'sector' in watchlist.columns else 0
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def get_overlap_analysis(self) -> pd.DataFrame:
        """Analyze overlap between watchlists"""
        all_tickers = {}
        
        for name, watchlist in self.watchlists.items():
            if 'ticker' in watchlist.columns:
                all_tickers[name] = set(watchlist['ticker'])
        
        overlap_data = []
        watchlist_names = list(all_tickers.keys())
        
        for i, name1 in enumerate(watchlist_names):
            for name2 in watchlist_names[i+1:]:
                overlap = len(all_tickers[name1] & all_tickers[name2])
                if overlap > 0:
                    overlap_data.append({
                        'watchlist_1': name1,
                        'watchlist_2': name2,
                        'overlap_count': overlap,
                        'overlap_pct': overlap / min(len(all_tickers[name1]), len(all_tickers[name2])) * 100
                    })
        
        return pd.DataFrame(overlap_data)
    
    def export_watchlists(self, format: str = 'dict') -> Dict:
        """Export all watchlists in specified format"""
        if format == 'dict':
            return {
                name: watchlist.to_dict('records')
                for name, watchlist in self.watchlists.items()
            }
        elif format == 'summary':
            return {
                name: {
                    'stocks': list(watchlist['ticker']) if 'ticker' in watchlist.columns else [],
                    'count': len(watchlist),
                    'description': WATCHLIST_TEMPLATES.get(name, WatchlistConfig(name=name, description="")).description
                }
                for name, watchlist in self.watchlists.items()
            }
        else:
            return self.watchlists

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_watchlists(
    df: pd.DataFrame,
    templates: Optional[Dict[str, WatchlistConfig]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build all watchlists from analyzed data
    
    Args:
        df: DataFrame with complete analysis
        templates: Custom templates (optional)
        
    Returns:
        Dictionary of watchlists
    """
    builder = WatchlistBuilder()
    return builder.build_all_watchlists(df, templates)

def get_top_opportunities(
    df: pd.DataFrame,
    n: int = 50,
    min_score: float = 70
) -> pd.DataFrame:
    """
    Get top N opportunities across all criteria
    
    Args:
        df: DataFrame with analysis
        n: Number of stocks
        min_score: Minimum composite score
        
    Returns:
        Top opportunities DataFrame
    """
    if 'composite_score' not in df.columns:
        logger.error("No composite_score found")
        return pd.DataFrame()
    
    filtered = df[df['composite_score'] >= min_score]
    
    # Prioritize by opportunity score if available
    sort_column = 'opportunity_score' if 'opportunity_score' in filtered.columns else 'composite_score'
    
    return filtered.nlargest(n, sort_column)

def get_quick_picks(
    df: pd.DataFrame,
    strategy: str = 'balanced'
) -> pd.DataFrame:
    """
    Get quick stock picks based on strategy
    
    Args:
        df: DataFrame with analysis
        strategy: Investment strategy
        
    Returns:
        Quick picks DataFrame
    """
    strategies = {
        'aggressive': {
            'min_score': 80,
            'filters': {'momentum_score': (80, 100), 'risk_score': (0, 80)},
            'max_stocks': 10
        },
        'conservative': {
            'min_score': 70,
            'filters': {'risk_score': (0, 40), 'market_cap': (10000, 1000000)},
            'max_stocks': 10
        },
        'balanced': {
            'min_score': 75,
            'filters': {'risk_score': (0, 60)},
            'max_stocks': 15
        },
        'income': {
            'min_score': 65,
            'filters': {'defensive_sector': True, 'pe': (0, 25)},
            'max_stocks': 10
        }
    }
    
    if strategy not in strategies:
        logger.error(f"Unknown strategy: {strategy}")
        return pd.DataFrame()
    
    config = strategies[strategy]
    builder = WatchlistBuilder()
    
    watchlist_config = WatchlistConfig(
        name=f"{strategy.title()} Picks",
        description=f"Quick picks for {strategy} investors",
        max_stocks=config['max_stocks'],
        min_score=config['min_score'],
        filters=config.get('filters', {})
    )
    
    return builder._build_single_watchlist(df, watchlist_config)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Watchlist Builder")
    print("="*60)
    print("\nCreates prioritized watchlists for different strategies")
    print("\nAvailable Watchlists:")
    for name, config in WATCHLIST_TEMPLATES.items():
        print(f"\n{config.name}:")
        print(f"  {config.description}")
        print(f"  Max stocks: {config.max_stocks}")
    print("\nUse build_watchlists() to create all watchlists")
    print("="*60)
