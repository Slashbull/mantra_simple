"""
watchlist_builder.py - Elite Watchlist Construction Engine for M.A.N.T.R.A.

World-class watchlist builder with bulletproof edge detection and infinite extensibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum


class WatchlistMode(Enum):
   """Standard watchlist modes"""
   TOP_OVERALL = "top_overall"
   TOP_BY_TAG = "top_by_tag"
   SECTOR_LEADERS = "sector_leaders"
   MULTI_SPIKE = "multi_spike"
   LAGGARD_REVERSAL = "laggard_reversal"
   LONG_TERM_WINNERS = "long_term_winners"
   LOW_VOLATILITY = "low_volatility"
   FRESH_52W_HIGH = "fresh_52w_high"
   VALUE_OUTLIERS = "value_outliers"
   PRICE_TIER = "price_tier"
   CUSTOM = "custom"


@dataclass
class WatchlistConfig:
   """Configuration for watchlist generation"""
   n: int = 20
   by: str = "final_score"
   tag: Optional[str] = None
   sector: Optional[str] = None
   price_tier: Optional[str] = None
   min_score: Optional[float] = None
   max_pe: Optional[float] = None
   min_eps: Optional[float] = None
   exclude_near_high: bool = False
   custom_filter: Optional[Callable] = None


class WatchlistBuilder:
   """Elite watchlist construction engine with maximum flexibility"""
   
   def __init__(self, df: pd.DataFrame):
       """
       Initialize with normalized dataframe.
       
       Args:
           df: Raw dataframe from M.A.N.T.R.A. universe
       """
       self.df = self._normalize_data(df)
       self._validate_data()
   
   def build_all(self) -> Dict[str, pd.DataFrame]:
       """
       Build all standard watchlists at once.
       
       Returns:
           Dictionary of watchlist_name -> DataFrame
       """
       return {
           "top_20_overall": self.top_n(20),
           "top_10_buy": self.top_n(10, tag="Buy"),
           "sector_leaders_buy": self.sector_leaders(n_per_sector=2, tag="Buy"),
           "sector_leaders_all": self.sector_leaders(n_per_sector=3),
           "multi_spike_anomalies": self.multi_spike_anomalies(),
           "laggard_reversals": self.laggard_reversal(),
           "long_term_winners": self.long_term_winners(),
           "low_volatility_gems": self.low_volatility(),
           "fresh_52w_highs": self.fresh_52w_high(),
           "value_outliers": self.value_outliers(),
           "momentum_value": self.momentum_value_combo(),
           "quality_growth": self.quality_growth(),
           "sector_rotation": self.sector_rotation_plays(),
           **self._build_price_tier_watchlists()
       }
   
   def build(self, mode: str, **kwargs) -> pd.DataFrame:
       """
       Build specific watchlist by mode.
       
       Args:
           mode: Watchlist mode name
           **kwargs: Additional parameters for the mode
           
       Returns:
           Filtered and sorted DataFrame
       """
       mode_map = {
           "top_overall": lambda: self.top_n(**kwargs),
           "top_by_tag": lambda: self.top_n(**kwargs),
           "sector_leaders": lambda: self.sector_leaders(**kwargs),
           "multi_spike": lambda: self.multi_spike_anomalies(**kwargs),
           "laggard_reversal": lambda: self.laggard_reversal(**kwargs),
           "long_term_winners": lambda: self.long_term_winners(**kwargs),
           "low_volatility": lambda: self.low_volatility(**kwargs),
           "fresh_52w_high": lambda: self.fresh_52w_high(**kwargs),
           "value_outliers": lambda: self.value_outliers(**kwargs),
           "momentum_value": lambda: self.momentum_value_combo(**kwargs),
           "quality_growth": lambda: self.quality_growth(**kwargs),
           "sector_rotation": lambda: self.sector_rotation_plays(**kwargs),
           "custom": lambda: self.custom(**kwargs)
       }
       
       if mode not in mode_map:
           raise ValueError(f"Unknown mode: {mode}. Available: {list(mode_map.keys())}")
       
       return mode_map[mode]()
   
   def top_n(self, n: int = 20, by: str = "final_score", tag: Optional[str] = None) -> pd.DataFrame:
       """
       Get top N stocks by any metric.
       
       Args:
           n: Number of stocks to return
           by: Column to sort by
           tag: Optional tag filter (Buy/Watch/Avoid)
           
       Returns:
           Top N stocks DataFrame
       """
       df = self._apply_tag_filter(self.df, tag)
       
       if by not in df.columns:
           raise ValueError(f"Column '{by}' not found. Available: {list(df.columns)}")
       
       return (df.nlargest(n, by, keep='first')
                 .reset_index(drop=True))
   
   def sector_leaders(self, n_per_sector: int = 2, by: str = "final_score", 
                     tag: Optional[str] = None, min_sector_stocks: int = 3) -> pd.DataFrame:
       """
       Get top N stocks from each sector.
       
       Args:
           n_per_sector: Number of stocks per sector
           by: Column to sort by
           tag: Optional tag filter
           min_sector_stocks: Minimum stocks required in sector
           
       Returns:
           Sector leaders DataFrame
       """
       df = self._apply_tag_filter(self.df, tag)
       
       # Filter sectors with minimum stocks
       sector_counts = df['sector'].value_counts()
       valid_sectors = sector_counts[sector_counts >= min_sector_stocks].index
       df = df[df['sector'].isin(valid_sectors)]
       
       return (df.sort_values(['sector', by], ascending=[True, False])
                 .groupby('sector')
                 .head(n_per_sector)
                 .sort_values(by, ascending=False)
                 .reset_index(drop=True))
   
   def multi_spike_anomalies(self, min_spike: float = 3.0, min_score: float = 50) -> pd.DataFrame:
       """
       Find stocks with multiple anomaly spikes.
       
       Args:
           min_spike: Minimum spike score
           min_score: Minimum final score
           
       Returns:
           Multi-spike anomaly stocks
       """
       spike_col = self._find_column(['spike_score', 'spike', 'anomaly_score'])
       if not spike_col:
           return pd.DataFrame()
       
       mask = (self.df[spike_col] >= min_spike) & (self.df['final_score'] >= min_score)
       
       return (self.df[mask]
               .sort_values([spike_col, 'final_score'], ascending=[False, False])
               .reset_index(drop=True))
   
   def laggard_reversal(self, min_reversal_1m: float = 5.0, max_1y_return: float = 0.0,
                       tag: str = "Buy") -> pd.DataFrame:
       """
       Find beaten-down stocks showing reversal signs.
       
       Args:
           min_reversal_1m: Minimum 1-month return for reversal
           max_1y_return: Maximum 1-year return (negative = losers)
           tag: Tag filter
           
       Returns:
           Reversal candidates DataFrame
       """
       # Find return columns
       ret_1y = self._find_column(['ret_1y', 'return_1y', 'ret_12m'])
       ret_1m = self._find_column(['ret_30d', 'ret_1m', 'return_1m'])
       
       if not ret_1y or not ret_1m:
           return pd.DataFrame()
       
       mask = (
           (self.df[ret_1y] < max_1y_return) &
           (self.df[ret_1m] > min_reversal_1m) &
           (self.df['tag'] == tag)
       )
       
       # Add reversal strength score
       result = self.df[mask].copy()
       result['reversal_strength'] = result[ret_1m] - (result[ret_1y] / 12)
       
       return (result.sort_values('reversal_strength', ascending=False)
                    .reset_index(drop=True))
   
   def long_term_winners(self, min_years: int = 5, min_annual_return: float = 15.0,
                        max_from_high: float = 20.0) -> pd.DataFrame:
       """
       Find consistent long-term performers not at peaks.
       
       Args:
           min_years: Minimum years of data (3 or 5)
           min_annual_return: Minimum annualized return
           max_from_high: Maximum % from 52w high
           
       Returns:
           Long-term winners DataFrame
       """
       # Determine return columns
       if min_years >= 5:
           long_ret = self._find_column(['ret_5y', 'return_5y'])
           med_ret = self._find_column(['ret_3y', 'return_3y'])
       else:
           long_ret = self._find_column(['ret_3y', 'return_3y'])
           med_ret = self._find_column(['ret_1y', 'return_1y'])
       
       from_high = self._find_column(['from_high_pct', 'from_high', 'pct_from_high'])
       
       if not long_ret:
           return pd.DataFrame()
       
       # Build filter
       mask = self.df[long_ret] > min_annual_return
       
       if med_ret:
           mask &= self.df[med_ret] > (min_annual_return / 2)
       
       if from_high:
           mask &= self.df[from_high] < max_from_high
       
       # Calculate consistency score
       result = self.df[mask].copy()
       if med_ret and long_ret:
           result['consistency_score'] = (
               result[med_ret] / (min_years - 2) / 
               (result[long_ret] / min_years)
           ).clip(0, 2) * 50
       
       return (result.sort_values(long_ret, ascending=False)
                    .reset_index(drop=True))
   
   def low_volatility(self, max_std: float = 2.5, min_score: float = 60.0,
                     lookback_days: List[str] = None) -> pd.DataFrame:
       """
       Find low-volatility outperformers.
       
       Args:
           max_std: Maximum standard deviation of returns
           min_score: Minimum final score
           lookback_days: Return columns to analyze
           
       Returns:
           Low volatility stocks DataFrame
       """
       if lookback_days is None:
           lookback_days = ['ret_3d', 'ret_7d', 'ret_30d']
       
       # Find available return columns
       ret_cols = []
       for col in lookback_days:
           found = self._find_column([col, col.replace('ret_', 'return_')])
           if found:
               ret_cols.append(found)
       
       if len(ret_cols) < 2:
           return pd.DataFrame()
       
       # Calculate rolling volatility
       returns_df = self.df[ret_cols].fillna(0)
       stds = returns_df.std(axis=1)
       
       # Filter
       mask = (stds < max_std) & (self.df['final_score'] > min_score)
       
       result = self.df[mask].copy()
       result['volatility'] = stds[mask]
       result['risk_adjusted_score'] = result['final_score'] / (1 + result['volatility'])
       
       return (result.sort_values('risk_adjusted_score', ascending=False)
                    .reset_index(drop=True))
   
   def fresh_52w_high(self, tolerance: float = 0.02, min_score: float = 50.0) -> pd.DataFrame:
       """
       Find stocks at or near 52-week highs.
       
       Args:
           tolerance: Percentage tolerance from high (0.02 = 2%)
           min_score: Minimum final score
           
       Returns:
           Stocks near 52w high DataFrame
       """
       price_col = self._find_column(['price', 'close', 'last_price'])
       high_col = self._find_column(['high_52w', '52w_high', 'high_1y'])
       
       if not price_col or not high_col:
           return pd.DataFrame()
       
       # Calculate distance from high
       result = self.df.copy()
       result['pct_from_52w_high'] = (
           (result[high_col] - result[price_col]) / result[high_col] * 100
       )
       
       # Filter
       mask = (
           (result['pct_from_52w_high'] <= tolerance * 100) &
           (result['final_score'] >= min_score)
       )
       
       return (result[mask]
               .sort_values(['pct_from_52w_high', 'final_score'], ascending=[True, False])
               .reset_index(drop=True))
   
   def value_outliers(self, max_pe: float = 15.0, min_eps_score: float = 75.0,
                     min_final_score: float = 60.0) -> pd.DataFrame:
       """
       Find undervalued stocks with strong fundamentals.
       
       Args:
           max_pe: Maximum P/E ratio
           min_eps_score: Minimum EPS score
           min_final_score: Minimum final score
           
       Returns:
           Value outliers DataFrame
       """
       pe_col = self._find_column(['pe', 'pe_ratio', 'p_e'])
       eps_col = self._find_column(['eps_score', 'eps', 'earnings_score'])
       
       if not pe_col:
           return pd.DataFrame()
       
       # Build filter
       mask = (
           (self.df[pe_col] > 0) &  # Valid PE
           (self.df[pe_col] < max_pe) &
           (self.df['final_score'] >= min_final_score)
       )
       
       if eps_col:
           mask &= self.df[eps_col] >= min_eps_score
       
       # Calculate value score
       result = self.df[mask].copy()
       result['value_score'] = (
           (1 - result[pe_col] / max_pe) * 50 +
           result['final_score'] / 2
       )
       
       if eps_col:
           result['value_score'] += result[eps_col] / 4
       
       return (result.sort_values('value_score', ascending=False)
                    .reset_index(drop=True))
   
   def momentum_value_combo(self, min_momentum_score: float = 70.0,
                          max_pe: float = 20.0, min_score: float = 65.0) -> pd.DataFrame:
       """
       Find stocks with both momentum and value characteristics.
       
       Args:
           min_momentum_score: Minimum momentum score
           max_pe: Maximum P/E ratio
           min_score: Minimum final score
           
       Returns:
           Momentum-value combination stocks
       """
       momentum_col = self._find_column(['momentum_score', 'mom_score', 'momentum'])
       pe_col = self._find_column(['pe', 'pe_ratio'])
       
       # Calculate momentum if not available
       if not momentum_col:
           ret_1m = self._find_column(['ret_30d', 'ret_1m'])
           ret_3m = self._find_column(['ret_3m', 'ret_90d'])
           
           if ret_1m and ret_3m:
               self.df['momentum_score'] = (
                   self.df[ret_1m].rank(pct=True) * 40 +
                   self.df[ret_3m].rank(pct=True) * 60
               )
               momentum_col = 'momentum_score'
       
       if not momentum_col or not pe_col:
           return pd.DataFrame()
       
       mask = (
           (self.df[momentum_col] >= min_momentum_score) &
           (self.df[pe_col] > 0) &
           (self.df[pe_col] <= max_pe) &
           (self.df['final_score'] >= min_score)
       )
       
       result = self.df[mask].copy()
       result['combo_score'] = (
           result[momentum_col] * 0.4 +
           (100 - result[pe_col] * 2) * 0.3 +
           result['final_score'] * 0.3
       )
       
       return (result.sort_values('combo_score', ascending=False)
                    .reset_index(drop=True))
   
   def quality_growth(self, min_eps_growth: float = 15.0, min_roe: float = 15.0,
                     min_score: float = 70.0) -> pd.DataFrame:
       """
       Find high-quality growth stocks.
       
       Args:
           min_eps_growth: Minimum EPS growth %
           min_roe: Minimum ROE %
           min_score: Minimum final score
           
       Returns:
           Quality growth stocks DataFrame
       """
       eps_growth_col = self._find_column(['eps_growth', 'eps_change_pct', 'earnings_growth'])
       roe_col = self._find_column(['roe', 'return_on_equity'])
       
       mask = self.df['final_score'] >= min_score
       
       if eps_growth_col:
           mask &= self.df[eps_growth_col] >= min_eps_growth
       
       if roe_col:
           mask &= self.df[roe_col] >= min_roe
       
       result = self.df[mask].copy()
       
       # Quality score
       quality_components = ['final_score']
       if eps_growth_col:
           quality_components.append(eps_growth_col)
       if roe_col:
           quality_components.append(roe_col)
       
       if len(quality_components) > 1:
           result['quality_score'] = result[quality_components].mean(axis=1)
       else:
           result['quality_score'] = result['final_score']
       
       return (result.sort_values('quality_score', ascending=False)
                    .reset_index(drop=True))
   
   def sector_rotation_plays(self, top_sectors: int = 3, stocks_per_sector: int = 2) -> pd.DataFrame:
       """
       Find best stocks in top-performing sectors.
       
       Args:
           top_sectors: Number of top sectors to consider
           stocks_per_sector: Stocks to pick from each sector
           
       Returns:
           Sector rotation plays DataFrame
       """
       # Calculate sector performance
       ret_1m = self._find_column(['ret_30d', 'ret_1m'])
       if not ret_1m:
           ret_1m = 'final_score'  # Fallback
       
       sector_perf = (self.df.groupby('sector')[ret_1m]
                          .agg(['mean', 'count'])
                          .query('count >= 3')  # Min 3 stocks
                          .sort_values('mean', ascending=False)
                          .head(top_sectors))
       
       top_sector_names = sector_perf.index.tolist()
       
       # Get best stocks from top sectors
       result = (self.df[self.df['sector'].isin(top_sector_names)]
                     .sort_values(['sector', 'final_score'], ascending=[True, False])
                     .groupby('sector')
                     .head(stocks_per_sector))
       
       # Add sector rank
       sector_rank_map = {s: i+1 for i, s in enumerate(top_sector_names)}
       result['sector_rank'] = result['sector'].map(sector_rank_map)
       
       return (result.sort_values(['sector_rank', 'final_score'], ascending=[True, False])
                    .reset_index(drop=True))
   
   def custom(self, config: Optional[WatchlistConfig] = None, **kwargs) -> pd.DataFrame:
       """
       Ultimate custom watchlist builder with any combination of filters.
       
       Args:
           config: WatchlistConfig object
           **kwargs: Override any config parameter
           
       Returns:
           Custom filtered DataFrame
       """
       # Merge config and kwargs
       if config:
           params = {
               'n': config.n,
               'by': config.by,
               'tag': config.tag,
               'sector': config.sector,
               'price_tier': config.price_tier,
               'min_score': config.min_score,
               'max_pe': config.max_pe,
               'min_eps': config.min_eps,
               'exclude_near_high': config.exclude_near_high,
               'custom_filter': config.custom_filter
           }
           params.update(kwargs)
       else:
           params = kwargs
       
       # Start with full dataset
       result = self.df.copy()
       
       # Apply filters
       if params.get('tag'):
           result = result[result['tag'] == params['tag']]
       
       if params.get('sector'):
           result = result[result['sector'] == params['sector']]
       
       if params.get('price_tier'):
           tier_col = self._find_column(['price_tier', 'tier', 'price_category'])
           if tier_col:
               result = result[result[tier_col] == params['price_tier']]
       
       if params.get('min_score') is not None:
           result = result[result['final_score'] >= params['min_score']]
       
       if params.get('max_pe') is not None:
           pe_col = self._find_column(['pe', 'pe_ratio'])
           if pe_col:
               result = result[(result[pe_col] > 0) & (result[pe_col] <= params['max_pe'])]
       
       if params.get('min_eps') is not None:
           eps_col = self._find_column(['eps_score', 'eps'])
           if eps_col:
               result = result[result[eps_col] >= params['min_eps']]
       
       if params.get('exclude_near_high'):
           from_high = self._find_column(['from_high_pct', 'from_high'])
           if from_high:
               result = result[result[from_high] > 5]
       
       # Apply custom filter function
       if params.get('custom_filter'):
           result = result[result.apply(params['custom_filter'], axis=1)]
       
       # Sort and limit
       by_col = params.get('by', 'final_score')
       if by_col in result.columns:
           result = result.sort_values(by_col, ascending=False)
       
       n = params.get('n', 20)
       if n and n > 0:
           result = result.head(n)
       
       return result.reset_index(drop=True)
   
   # Private helper methods
   
   def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
       """Normalize and clean input data."""
       df = df.copy()
       
       # Standardize column names
       df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r'[^\w\s]', '', regex=True)
                    .str.replace(r'\s+', '_', regex=True))
       
       # Clean text columns
       text_cols = ['ticker', 'sector', 'tag', 'price_tier']
       for col in text_cols:
           if col in df.columns:
               df[col] = df[col].astype(str).str.strip()
               if col == 'tag':
                   df[col] = df[col].str.title()  # Buy/Watch/Avoid
       
       # Ensure numeric columns
       numeric_patterns = ['score', 'ret', 'return', 'pe', 'pb', 'eps', 'roe', 'spike', 'vol']
       for col in df.columns:
           if any(pattern in col for pattern in numeric_patterns):
               df[col] = pd.to_numeric(df[col], errors='coerce')
       
       # Add final_score if missing
       if 'final_score' not in df.columns:
           score_col = self._find_column(['score', 'total_score', 'composite_score'])
           if score_col:
               df['final_score'] = df[score_col]
           else:
               df['final_score'] = 50  # Default
       
       # Remove duplicate tickers
       if 'ticker' in df.columns:
           df = df.drop_duplicates(subset='ticker', keep='first')
       
       return df
   
   def _validate_data(self):
       """Validate data has minimum required columns."""
       required = ['ticker', 'final_score']
       missing = [col for col in required if col not in self.df.columns]
       
       if missing:
           available = list(self.df.columns)
           raise ValueError(f"Missing required columns: {missing}. Available: {available}")
   
   def _find_column(self, candidates: List[str]) -> Optional[str]:
       """Find first matching column from candidates."""
       for candidate in candidates:
           if candidate in self.df.columns:
               return candidate
       return None
   
   def _apply_tag_filter(self, df: pd.DataFrame, tag: Optional[str]) -> pd.DataFrame:
       """Apply tag filter if specified."""
       if tag and 'tag' in df.columns:
           return df[df['tag'] == tag]
       return df
   
   def _build_price_tier_watchlists(self) -> Dict[str, pd.DataFrame]:
       """Build watchlists for each price tier."""
       tier_col = self._find_column(['price_tier', 'tier', 'price_category'])
       if not tier_col:
           return {}
       
       result = {}
       for tier in self.df[tier_col].dropna().unique():
           tier_df = self.df[self.df[tier_col] == tier]
           if len(tier_df) >= 3:  # Minimum stocks for a tier
               key = f"tier_{str(tier).lower().replace(' ', '_')}"
               result[key] = (tier_df.sort_values('final_score', ascending=False)
                                    .head(10)
                                    .reset_index(drop=True))
       
       return result


# Convenience functions for backward compatibility

def build_all_watchlists(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
   """Build all standard watchlists."""
   builder = WatchlistBuilder(df)
   return builder.build_all()


def build_watchlist(df: pd.DataFrame, mode: str = "top_overall", **kwargs) -> pd.DataFrame:
   """Build specific watchlist by mode."""
   builder = WatchlistBuilder(df)
   return builder.build(mode, **kwargs)


def custom_watchlist(df: pd.DataFrame, **filters) -> pd.DataFrame:
   """Build custom watchlist with any filters."""
   builder = WatchlistBuilder(df)
   return builder.custom(**filters)
