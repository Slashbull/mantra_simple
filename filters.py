"""
filters.py - M.A.N.T.R.A. Dashboard Filters
==========================================
Flexible filtering system for stock screening
Provides UI-ready filter components and logic
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Import from constants
from constants import (
    SIGNAL_LEVELS, MARKET_CAP_RANGES, SECTOR_GROUPS,
    PRICE_RANGE_FILTERS, PE_RANGES, RISK_SCORES
)

logger = logging.getLogger(__name__)

# ============================================================================
# FILTER TYPES
# ============================================================================

class FilterType(Enum):
    """Types of filters available"""
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    RANGE = "range"
    BOOLEAN = "boolean"
    TEXT = "text"
    NUMERIC = "numeric"

@dataclass
class FilterDefinition:
    """Definition of a single filter"""
    name: str
    display_name: str
    column: str
    filter_type: FilterType
    options: Optional[List] = None
    range: Optional[Tuple[float, float]] = None
    default: Any = None
    description: str = ""
    format_func: Optional[callable] = None

# ============================================================================
# PREDEFINED FILTERS
# ============================================================================

class FilterLibrary:
    """Library of all available filters"""
    
    @staticmethod
    def get_all_filters() -> Dict[str, FilterDefinition]:
        """Get all filter definitions"""
        return {
            # Signal filters
            "decision": FilterDefinition(
                name="decision",
                display_name="Signal",
                column="decision",
                filter_type=FilterType.MULTI_SELECT,
                options=["BUY", "WATCH", "NEUTRAL", "AVOID"],
                default=["BUY", "WATCH"],
                description="Trading decision signal"
            ),
            
            "composite_score": FilterDefinition(
                name="composite_score",
                display_name="Composite Score",
                column="composite_score",
                filter_type=FilterType.RANGE,
                range=(0, 100),
                default=(70, 100),
                description="Overall signal strength"
            ),
            
            # Market cap filters
            "market_cap_category": FilterDefinition(
                name="market_cap_category",
                display_name="Market Cap",
                column="category",
                filter_type=FilterType.MULTI_SELECT,
                options=["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"],
                default=["Large Cap", "Mid Cap", "Small Cap"],
                description="Market capitalization category"
            ),
            
            "market_cap_range": FilterDefinition(
                name="market_cap_range",
                display_name="Market Cap (Cr)",
                column="market_cap",
                filter_type=FilterType.RANGE,
                range=(0, 1000000),
                default=(100, 1000000),
                description="Market cap in Crores",
                format_func=lambda x: f"₹{x:,.0f} Cr"
            ),
            
            # Sector filters
            "sector": FilterDefinition(
                name="sector",
                display_name="Sector",
                column="sector",
                filter_type=FilterType.MULTI_SELECT,
                options=[],  # Will be populated dynamically
                default=[],
                description="Business sector"
            ),
            
            "sector_group": FilterDefinition(
                name="sector_group",
                display_name="Sector Group",
                column="sector",
                filter_type=FilterType.MULTI_SELECT,
                options=list(SECTOR_GROUPS.keys()),
                default=[],
                description="Sector grouping (Defensive, Cyclical, etc.)"
            ),
            
            # Price filters
            "price_range": FilterDefinition(
                name="price_range",
                display_name="Price Range",
                column="price",
                filter_type=FilterType.RANGE,
                range=(0, 100000),
                default=(10, 100000),
                description="Stock price range",
                format_func=lambda x: f"₹{x:,.0f}"
            ),
            
            "price_tier": FilterDefinition(
                name="price_tier",
                display_name="Price Category",
                column="price_tier",
                filter_type=FilterType.MULTI_SELECT,
                options=["<50", "50-100", "100-250", "250-500", "500-1K", "1K-2K", "2K-5K", "5K-10K", ">10K"],
                default=[],
                description="Price tier categories"
            ),
            
            # Performance filters
            "return_30d": FilterDefinition(
                name="return_30d",
                display_name="30-Day Return (%)",
                column="ret_30d",
                filter_type=FilterType.RANGE,
                range=(-50, 200),
                default=(-10, 200),
                description="30-day price performance",
                format_func=lambda x: f"{x:+.1f}%"
            ),
            
            "momentum_score": FilterDefinition(
                name="momentum_score",
                display_name="Momentum Score",
                column="momentum_score",
                filter_type=FilterType.RANGE,
                range=(0, 100),
                default=(0, 100),
                description="Momentum strength indicator"
            ),
            
            # Value filters
            "pe_ratio": FilterDefinition(
                name="pe_ratio",
                display_name="P/E Ratio",
                column="pe",
                filter_type=FilterType.RANGE,
                range=(-100, 200),
                default=(0, 50),
                description="Price to Earnings ratio"
            ),
            
            "value_score": FilterDefinition(
                name="value_score",
                display_name="Value Score",
                column="value_score",
                filter_type=FilterType.RANGE,
                range=(0, 100),
                default=(0, 100),
                description="Value attractiveness score"
            ),
            
            # Volume filters
            "volume_min": FilterDefinition(
                name="volume_min",
                display_name="Min Daily Volume",
                column="volume_1d",
                filter_type=FilterType.NUMERIC,
                default=50000,
                description="Minimum daily trading volume"
            ),
            
            "relative_volume": FilterDefinition(
                name="relative_volume",
                display_name="Relative Volume",
                column="rvol",
                filter_type=FilterType.RANGE,
                range=(0, 10),
                default=(0.5, 10),
                description="Volume vs average",
                format_func=lambda x: f"{x:.1f}x"
            ),
            
            # Risk filters
            "risk_level": FilterDefinition(
                name="risk_level",
                display_name="Risk Level",
                column="risk_level",
                filter_type=FilterType.MULTI_SELECT,
                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                default=["Very Low", "Low", "Moderate"],
                description="Risk assessment level"
            ),
            
            "risk_score": FilterDefinition(
                name="risk_score",
                display_name="Risk Score",
                column="risk_score",
                filter_type=FilterType.RANGE,
                range=(0, 100),
                default=(0, 70),
                description="Numerical risk score"
            ),
            
            # Technical filters
            "52w_position": FilterDefinition(
                name="52w_position",
                display_name="52-Week Position (%)",
                column="position_52w",
                filter_type=FilterType.RANGE,
                range=(0, 100),
                default=(0, 100),
                description="Position in 52-week range"
            ),
            
            "above_200_dma": FilterDefinition(
                name="above_200_dma",
                display_name="Above 200 DMA",
                column="trading_under",
                filter_type=FilterType.BOOLEAN,
                default=None,
                description="Trading above 200-day moving average"
            ),
            
            # Pattern filters
            "has_edge": FilterDefinition(
                name="has_edge",
                display_name="Has Edge Setup",
                column="edge_count",
                filter_type=FilterType.BOOLEAN,
                default=None,
                description="Has identified edge setup"
            ),
            
            "anomaly_detected": FilterDefinition(
                name="anomaly_detected",
                display_name="Anomaly Detected",
                column="anomaly_count",
                filter_type=FilterType.BOOLEAN,
                default=None,
                description="Has detected anomalies"
            ),
            
            # EPS filters
            "eps_growth": FilterDefinition(
                name="eps_growth",
                display_name="EPS Growth (%)",
                column="eps_change_pct",
                filter_type=FilterType.RANGE,
                range=(-100, 500),
                default=(-20, 500),
                description="Earnings per share growth",
                format_func=lambda x: f"{x:+.1f}%"
            ),
            
            # Search filter
            "search": FilterDefinition(
                name="search",
                display_name="Search",
                column="ticker",
                filter_type=FilterType.TEXT,
                default="",
                description="Search by ticker or company name"
            )
        }

# ============================================================================
# FILTER ENGINE
# ============================================================================

class FilterEngine:
    """
    Engine for applying filters to dataframes
    """
    
    def __init__(self):
        self.filters = FilterLibrary.get_all_filters()
        self.active_filters = {}
        
    def apply_filters(
        self,
        df: pd.DataFrame,
        filter_values: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply multiple filters to dataframe
        
        Args:
            df: DataFrame to filter
            filter_values: Dictionary of filter name -> value
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
            
        filtered_df = df.copy()
        self.active_filters = filter_values.copy()
        
        for filter_name, filter_value in filter_values.items():
            if filter_value is None or filter_value == "" or filter_value == []:
                continue
                
            if filter_name not in self.filters:
                logger.warning(f"Unknown filter: {filter_name}")
                continue
                
            filter_def = self.filters[filter_name]
            filtered_df = self._apply_single_filter(filtered_df, filter_def, filter_value)
        
        logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} rows")
        
        return filtered_df
    
    def _apply_single_filter(
        self,
        df: pd.DataFrame,
        filter_def: FilterDefinition,
        value: Any
    ) -> pd.DataFrame:
        """Apply a single filter based on its type"""
        
        column = filter_def.column
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df
        
        if filter_def.filter_type == FilterType.MULTI_SELECT:
            if filter_def.name == "sector_group":
                # Special handling for sector groups
                all_sectors = []
                for group in value:
                    if group in SECTOR_GROUPS:
                        all_sectors.extend(SECTOR_GROUPS[group])
                return df[df[column].isin(all_sectors)]
            else:
                return df[df[column].isin(value)]
        
        elif filter_def.filter_type == FilterType.SINGLE_SELECT:
            return df[df[column] == value]
        
        elif filter_def.filter_type == FilterType.RANGE:
            min_val, max_val = value
            mask = pd.Series(True, index=df.index)
            if min_val is not None:
                mask &= df[column] >= min_val
            if max_val is not None:
                mask &= df[column] <= max_val
            return df[mask]
        
        elif filter_def.filter_type == FilterType.NUMERIC:
            return df[df[column] >= value]
        
        elif filter_def.filter_type == FilterType.BOOLEAN:
            if filter_def.name == "above_200_dma":
                # Special handling for DMA filter
                if value:
                    return df[df[column].isna() | (df[column] == "")]
                else:
                    return df[df[column].notna() & (df[column] != "")]
            else:
                # Generic boolean handling
                if value:
                    return df[df[column] > 0]
                else:
                    return df[df[column] == 0]
        
        elif filter_def.filter_type == FilterType.TEXT:
            # Case-insensitive search in ticker and company name
            mask = df[column].str.contains(value, case=False, na=False)
            if 'company_name' in df.columns:
                mask |= df['company_name'].str.contains(value, case=False, na=False)
            return df[mask]
        
        return df
    
    def get_filter_options(self, df: pd.DataFrame, filter_name: str) -> List:
        """Get available options for a filter based on current data"""
        if filter_name not in self.filters:
            return []
        
        filter_def = self.filters[filter_name]
        column = filter_def.column
        
        if column not in df.columns:
            return []
        
        if filter_def.filter_type in [FilterType.MULTI_SELECT, FilterType.SINGLE_SELECT]:
            # Get unique values from data
            unique_values = df[column].dropna().unique()
            return sorted(unique_values)
        
        return filter_def.options or []
    
    def get_filter_stats(self, df: pd.DataFrame, filter_name: str) -> Dict:
        """Get statistics for a filter"""
        if filter_name not in self.filters:
            return {}
        
        filter_def = self.filters[filter_name]
        column = filter_def.column
        
        if column not in df.columns:
            return {}
        
        stats = {
            'total_count': len(df),
            'non_null_count': df[column].notna().sum()
        }
        
        if filter_def.filter_type == FilterType.RANGE:
            stats.update({
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'median': df[column].median()
            })
        
        elif filter_def.filter_type in [FilterType.MULTI_SELECT, FilterType.SINGLE_SELECT]:
            value_counts = df[column].value_counts()
            stats['value_distribution'] = value_counts.to_dict()
        
        return stats
    
    def create_filter_summary(self) -> pd.DataFrame:
        """Create summary of active filters"""
        if not self.active_filters:
            return pd.DataFrame()
        
        summary_data = []
        for filter_name, value in self.active_filters.items():
            if filter_name in self.filters:
                filter_def = self.filters[filter_name]
                summary_data.append({
                    'Filter': filter_def.display_name,
                    'Value': self._format_filter_value(filter_def, value),
                    'Type': filter_def.filter_type.value
                })
        
        return pd.DataFrame(summary_data)
    
    def _format_filter_value(self, filter_def: FilterDefinition, value: Any) -> str:
        """Format filter value for display"""
        if filter_def.format_func:
            if isinstance(value, (list, tuple)):
                return f"{filter_def.format_func(value[0])} - {filter_def.format_func(value[1])}"
            else:
                return filter_def.format_func(value)
        
        if isinstance(value, list):
            return ", ".join(str(v) for v in value[:3]) + ("..." if len(value) > 3 else "")
        elif isinstance(value, tuple):
            return f"{value[0]} - {value[1]}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        else:
            return str(value)

# ============================================================================
# PRESET FILTERS
# ============================================================================

class FilterPresets:
    """Predefined filter combinations"""
    
    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get all filter presets"""
        return {
            "high_conviction_buys": {
                "name": "High Conviction Buys",
                "description": "Strong buy signals with low risk",
                "filters": {
                    "decision": ["BUY"],
                    "composite_score": (80, 100),
                    "risk_score": (0, 60),
                    "volume_min": 100000
                }
            },
            
            "momentum_plays": {
                "name": "Momentum Plays",
                "description": "Stocks with strong momentum",
                "filters": {
                    "momentum_score": (75, 100),
                    "return_30d": (10, 200),
                    "above_200_dma": True
                }
            },
            
            "value_picks": {
                "name": "Value Picks",
                "description": "Undervalued opportunities",
                "filters": {
                    "pe_ratio": (0, 20),
                    "value_score": (70, 100),
                    "eps_growth": (0, 500)
                }
            },
            
            "large_cap_safe": {
                "name": "Large Cap Safety",
                "description": "Stable large cap stocks",
                "filters": {
                    "market_cap_category": ["Large Cap"],
                    "risk_level": ["Very Low", "Low"],
                    "above_200_dma": True
                }
            },
            
            "small_cap_growth": {
                "name": "Small Cap Growth",
                "description": "High growth small caps",
                "filters": {
                    "market_cap_category": ["Small Cap"],
                    "eps_growth": (20, 500),
                    "return_30d": (5, 200)
                }
            },
            
            "breakout_candidates": {
                "name": "Breakout Candidates",
                "description": "Potential breakout stocks",
                "filters": {
                    "52w_position": (70, 100),
                    "relative_volume": (1.5, 10),
                    "has_edge": True
                }
            },
            
            "turnaround_stories": {
                "name": "Turnaround Stories",
                "description": "Recovering from lows",
                "filters": {
                    "52w_position": (0, 30),
                    "return_30d": (0, 200),
                    "anomaly_detected": True
                }
            },
            
            "defensive_income": {
                "name": "Defensive Income",
                "description": "Stable dividend plays",
                "filters": {
                    "sector_group": ["Defensive"],
                    "market_cap_range": (10000, 1000000),
                    "risk_level": ["Very Low", "Low", "Moderate"]
                }
            }
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def apply_filters(
    df: pd.DataFrame,
    **filter_kwargs
) -> pd.DataFrame:
    """
    Apply filters to dataframe using kwargs
    
    Example:
        filtered = apply_filters(df, decision=['BUY'], pe_ratio=(0, 25))
    """
    engine = FilterEngine()
    return engine.apply_filters(df, filter_kwargs)

def get_filter_preset(
    df: pd.DataFrame,
    preset_name: str
) -> pd.DataFrame:
    """Apply a preset filter combination"""
    presets = FilterPresets.get_presets()
    
    if preset_name not in presets:
        logger.error(f"Unknown preset: {preset_name}")
        return df
    
    preset = presets[preset_name]
    engine = FilterEngine()
    
    return engine.apply_filters(df, preset['filters'])

def create_streamlit_filters(
    st,
    df: pd.DataFrame,
    sidebar: bool = True
) -> Dict[str, Any]:
    """
    Create Streamlit filter widgets
    
    Args:
        st: Streamlit module
        df: DataFrame to filter
        sidebar: Whether to use sidebar
        
    Returns:
        Dictionary of filter values
    """
    container = st.sidebar if sidebar else st
    engine = FilterEngine()
    filter_values = {}
    
    # Add filter presets
    with container.expander("Quick Filters", expanded=False):
        preset = st.selectbox(
            "Select Preset",
            ["None"] + list(FilterPresets.get_presets().keys()),
            format_func=lambda x: x if x == "None" else FilterPresets.get_presets()[x]['name']
        )
        
        if preset != "None":
            return FilterPresets.get_presets()[preset]['filters']
    
    # Individual filters
    with container.expander("Custom Filters", expanded=True):
        # Decision filter
        filter_values['decision'] = st.multiselect(
            "Signal",
            options=["BUY", "WATCH", "NEUTRAL", "AVOID"],
            default=["BUY", "WATCH"]
        )
        
        # Score filter
        score_range = st.slider(
            "Composite Score",
            min_value=0,
            max_value=100,
            value=(70, 100),
            step=5
        )
        filter_values['composite_score'] = score_range
        
        # Market cap filter
        filter_values['market_cap_category'] = st.multiselect(
            "Market Cap",
            options=["Large Cap", "Mid Cap", "Small Cap"],
            default=["Large Cap", "Mid Cap", "Small Cap"]
        )
        
        # Sector filter
        available_sectors = sorted(df['sector'].unique()) if 'sector' in df.columns else []
        filter_values['sector'] = st.multiselect(
            "Sectors",
            options=available_sectors,
            default=[]
        )
        
        # Risk filter
        risk_range = st.slider(
            "Risk Score",
            min_value=0,
            max_value=100,
            value=(0, 70),
            step=5
        )
        filter_values['risk_score'] = risk_range
        
        # Volume filter
        filter_values['volume_min'] = st.number_input(
            "Min Daily Volume",
            min_value=0,
            value=50000,
            step=10000
        )
        
    # Search box (always visible)
    filter_values['search'] = container.text_input(
        "Search Ticker/Company",
        value="",
        placeholder="TCS, Infosys..."
    )
    
    # Remove empty filters
    filter_values = {k: v for k, v in filter_values.items() if v}
    
    return filter_values

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Dashboard Filters")
    print("="*60)
    print("\nFlexible filtering system for stock screening")
    print("\nFilter Types:")
    for filter_type in FilterType:
        print(f"  - {filter_type.value}")
    print("\nPreset Filters:")
    for preset_name, preset in FilterPresets.get_presets().items():
        print(f"  - {preset['name']}: {preset['description']}")
    print("\nUse apply_filters() or create_streamlit_filters()")
    print("="*60)
