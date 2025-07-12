# anomaly_detector.py - M.A.N.T.R.A. Ultimate Anomaly Detection Engine v2.0

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
   PRICE_BREAKOUT = "Price Breakout"
   PRICE_BREAKDOWN = "Price Breakdown"
   VOLUME_SURGE = "Volume Surge"
   VOLUME_DRY = "Volume Dryup"
   MOMENTUM_BURST = "Momentum Burst"
   MOMENTUM_REVERSAL = "Momentum Reversal"
   EARNINGS_EXPLOSION = "Earnings Explosion"
   EARNINGS_COLLAPSE = "Earnings Collapse"
   TECHNICAL_BREAKOUT = "Technical Breakout"
   TECHNICAL_BREAKDOWN = "Technical Breakdown"
   SECTOR_OUTLIER = "Sector Outlier"
   VOLATILITY_SPIKE = "Volatility Spike"
   LIQUIDITY_SURGE = "Liquidity Surge"
   MEAN_REVERSION = "Mean Reversion"
   TREND_ACCELERATION = "Trend Acceleration"
   DISTRIBUTION_SHIFT = "Distribution Shift"
   MULTI_SIGNAL = "Multi-Signal Anomaly"


class AnomalySeverity(Enum):
   MILD = ("mild", 1)
   MODERATE = ("moderate", 2)
   MAJOR = ("major", 3)
   EXTREME = ("extreme", 4)
   CRITICAL = ("critical", 5)


@dataclass
class AnomalyConfig:
   # Statistical thresholds
   z_score_threshold: float = 2.5
   mad_multiplier: float = 3.0
   iqr_multiplier: float = 2.5
   
   # Adaptive percentile thresholds
   price_spike_percentile: float = 0.95
   volume_spike_percentile: float = 0.90
   momentum_burst_percentile: float = 0.93
   
   # Minimum thresholds (fallback)
   min_price_change: float = 3.0
   min_volume_ratio: float = 2.0
   min_eps_change: float = 25.0
   
   # Technical thresholds
   breakout_proximity: float = 0.02
   breakdown_proximity: float = 0.05
   trend_lookback_days: int = 20
   
   # Severity scoring weights
   severity_weights: Dict[str, float] = field(default_factory=lambda: {
       "statistical_rarity": 0.3,
       "magnitude": 0.3,
       "consistency": 0.2,
       "context": 0.2
   })
   
   # Regime adaptation
   adapt_to_market_volatility: bool = True
   adapt_to_sector_behavior: bool = True
   
   # Multi-anomaly detection
   multi_anomaly_boost: float = 1.5
   correlation_window: int = 5
   
   # Data quality
   min_data_points: int = 20
   outlier_cap_percentile: float = 0.995


class QuantAnomalyDetector:
   
   def __init__(self, config: Optional[AnomalyConfig] = None):
       self.config = config or AnomalyConfig()
       self.market_context = {}
       self.thresholds = {}
       self.explanations = []
       
   def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       if df.empty:
           logger.warning("Empty dataframe provided")
           return df
       
       df = df.copy()
       
       # Initialize columns
       df['anomaly'] = False
       df['anomaly_type'] = ''
       df['anomaly_severity'] = ''
       df['anomaly_score'] = 0.0
       df['anomaly_explanation'] = ''
       
       # Calculate market context
       self._calculate_market_context(df)
       
       # Calculate adaptive thresholds
       self._calculate_adaptive_thresholds(df)
       
       # Run detection modules
       anomalies = []
       
       # 1. Price anomalies
       price_anomalies = self._detect_price_anomalies(df)
       anomalies.append(price_anomalies)
       
       # 2. Volume anomalies
       volume_anomalies = self._detect_volume_anomalies(df)
       anomalies.append(volume_anomalies)
       
       # 3. Momentum anomalies
       momentum_anomalies = self._detect_momentum_anomalies(df)
       anomalies.append(momentum_anomalies)
       
       # 4. Earnings anomalies
       earnings_anomalies = self._detect_earnings_anomalies(df)
       anomalies.append(earnings_anomalies)
       
       # 5. Technical anomalies
       technical_anomalies = self._detect_technical_anomalies(df)
       anomalies.append(technical_anomalies)
       
       # 6. Statistical distribution anomalies
       distribution_anomalies = self._detect_distribution_anomalies(df)
       anomalies.append(distribution_anomalies)
       
       # 7. Sector-relative anomalies
       sector_anomalies = self._detect_sector_anomalies(df)
       anomalies.append(sector_anomalies)
       
       # Combine all anomalies
       all_anomalies = pd.concat(anomalies, ignore_index=True)
       
       # Process multi-anomalies
       df = self._process_multi_anomalies(df, all_anomalies)
       
       # Calculate final scores and severity
       df = self._calculate_anomaly_scores(df, all_anomalies)
       
       return df
   
   def _calculate_market_context(self, df: pd.DataFrame) -> None:
       self.market_context = {
           'volatility': df['ret_1d'].std() if 'ret_1d' in df else 1.0,
           'volume_baseline': df['volume_1d'].median() if 'volume_1d' in df else 1.0,
           'price_dispersion': df['price'].std() / df['price'].mean() if 'price' in df else 0.1,
           'sector_count': df['sector'].nunique() if 'sector' in df else 1,
           'market_trend': self._calculate_market_trend(df),
           'volatility_regime': self._classify_volatility_regime(df)
       }
   
   def _calculate_market_trend(self, df: pd.DataFrame) -> str:
       if 'ret_30d' not in df:
           return 'neutral'
       
       avg_return = df['ret_30d'].mean()
       positive_pct = (df['ret_30d'] > 0).mean()
       
       if avg_return > 5 and positive_pct > 0.6:
           return 'bullish'
       elif avg_return < -5 and positive_pct < 0.4:
           return 'bearish'
       else:
           return 'neutral'
   
   def _classify_volatility_regime(self, df: pd.DataFrame) -> str:
       if 'ret_1d' not in df:
           return 'normal'
       
       vol = df['ret_1d'].std()
       
       if vol > 3:
           return 'high'
       elif vol < 1:
           return 'low'
       else:
           return 'normal'
   
   def _calculate_adaptive_thresholds(self, df: pd.DataFrame) -> None:
       # Price thresholds
       if 'ret_1d' in df:
           ret_1d_clean = self._remove_outliers(df['ret_1d'])
           self.thresholds['price_spike'] = max(
               ret_1d_clean.quantile(self.config.price_spike_percentile),
               self.config.min_price_change
           )
           self.thresholds['price_crash'] = min(
               ret_1d_clean.quantile(1 - self.config.price_spike_percentile),
               -self.config.min_price_change
           )
       
       # Volume thresholds
       if 'vol_ratio_1d_90d' in df:
           vol_clean = self._remove_outliers(df['vol_ratio_1d_90d'])
           self.thresholds['volume_spike'] = max(
               vol_clean.quantile(self.config.volume_spike_percentile),
               self.config.min_volume_ratio
           )
           self.thresholds['volume_dry'] = vol_clean.quantile(0.10)
       
       # Momentum thresholds
       if all(col in df for col in ['ret_3d', 'ret_7d', 'ret_30d']):
           momentum_composite = (df['ret_3d'] + df['ret_7d'] + df['ret_30d']) / 3
           momentum_clean = self._remove_outliers(momentum_composite)
           self.thresholds['momentum_burst'] = momentum_clean.quantile(
               self.config.momentum_burst_percentile
           )
       
       # EPS thresholds
       if 'eps_change_pct' in df:
           eps_clean = self._remove_outliers(df['eps_change_pct'])
           self.thresholds['eps_explosion'] = max(
               eps_clean.quantile(0.90),
               self.config.min_eps_change
           )
           self.thresholds['eps_collapse'] = eps_clean.quantile(0.10)
   
   def _remove_outliers(self, series: pd.Series, method: str = 'iqr') -> pd.Series:
       if method == 'iqr':
           Q1 = series.quantile(0.25)
           Q3 = series.quantile(0.75)
           IQR = Q3 - Q1
           lower = Q1 - self.config.iqr_multiplier * IQR
           upper = Q3 + self.config.iqr_multiplier * IQR
           return series[(series >= lower) & (series <= upper)]
       elif method == 'zscore':
           z_scores = np.abs(stats.zscore(series.dropna()))
           return series[z_scores < self.config.z_score_threshold]
       else:
           return series
   
   def _detect_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       if 'ret_1d' not in df:
           return pd.DataFrame()
       
       # Price spikes
       spike_threshold = self.thresholds.get('price_spike', 5.0)
       spike_mask = df['ret_1d'] > spike_threshold
       
       for idx in df[spike_mask].index:
           magnitude = df.loc[idx, 'ret_1d']
           z_score = self._calculate_zscore(df['ret_1d'], magnitude)
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.PRICE_BREAKOUT.value,
               'magnitude': magnitude,
               'z_score': z_score,
               'explanation': f"Price surged {magnitude:.1f}% (Z={z_score:.1f})"
           })
       
       # Price crashes
       crash_threshold = self.thresholds.get('price_crash', -5.0)
       crash_mask = df['ret_1d'] < crash_threshold
       
       for idx in df[crash_mask].index:
           magnitude = abs(df.loc[idx, 'ret_1d'])
           z_score = abs(self._calculate_zscore(df['ret_1d'], df.loc[idx, 'ret_1d']))
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.PRICE_BREAKDOWN.value,
               'magnitude': magnitude,
               'z_score': z_score,
               'explanation': f"Price crashed {magnitude:.1f}% (Z={z_score:.1f})"
           })
       
       return pd.DataFrame(anomalies)
   
   def _detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       if 'vol_ratio_1d_90d' not in df:
           return pd.DataFrame()
       
       # Volume surges
       surge_threshold = self.thresholds.get('volume_spike', 3.0)
       surge_mask = df['vol_ratio_1d_90d'] > surge_threshold
       
       for idx in df[surge_mask].index:
           ratio = df.loc[idx, 'vol_ratio_1d_90d']
           
           # Check if accompanied by price movement
           price_move = df.loc[idx, 'ret_1d'] if 'ret_1d' in df else 0
           context = "with price up" if price_move > 1 else "with price down" if price_move < -1 else "sideways"
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.VOLUME_SURGE.value,
               'magnitude': ratio,
               'z_score': self._calculate_zscore(df['vol_ratio_1d_90d'], ratio),
               'explanation': f"Volume {ratio:.1f}x normal {context}"
           })
       
       # Volume dryups
       dry_threshold = self.thresholds.get('volume_dry', 0.3)
       dry_mask = df['vol_ratio_1d_90d'] < dry_threshold
       
       for idx in df[dry_mask].index:
           ratio = df.loc[idx, 'vol_ratio_1d_90d']
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.VOLUME_DRY.value,
               'magnitude': 1/ratio if ratio > 0 else 10,
               'z_score': abs(self._calculate_zscore(df['vol_ratio_1d_90d'], ratio)),
               'explanation': f"Volume dried up to {ratio:.1%} of normal"
           })
       
       return pd.DataFrame(anomalies)
   
   def _detect_momentum_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       required_cols = ['ret_3d', 'ret_7d', 'ret_30d']
       if not all(col in df for col in required_cols):
           return pd.DataFrame()
       
       # Calculate momentum metrics
       df['momentum_score'] = (df['ret_3d'] * 0.5 + df['ret_7d'] * 0.3 + df['ret_30d'] * 0.2)
       df['momentum_consistency'] = df[required_cols].apply(
           lambda x: 1 if (x > 0).all() or (x < 0).all() else 0, axis=1
       )
       
       # Momentum bursts
       burst_threshold = self.thresholds.get('momentum_burst', 10.0)
       burst_mask = (df['momentum_score'] > burst_threshold) & (df['momentum_consistency'] == 1)
       
       for idx in df[burst_mask].index:
           score = df.loc[idx, 'momentum_score']
           returns = [df.loc[idx, col] for col in required_cols]
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.MOMENTUM_BURST.value,
               'magnitude': score,
               'z_score': self._calculate_zscore(df['momentum_score'], score),
               'explanation': f"Strong momentum: {'/'.join([f'{r:.1f}%' for r in returns])}"
           })
       
       # Momentum reversals
       if 'ret_30d' in df and 'ret_3d' in df:
           reversal_mask = (
               ((df['ret_30d'] > 10) & (df['ret_3d'] < -3)) |
               ((df['ret_30d'] < -10) & (df['ret_3d'] > 3))
           )
           
           for idx in df[reversal_mask].index:
               long_term = df.loc[idx, 'ret_30d']
               short_term = df.loc[idx, 'ret_3d']
               
               anomalies.append({
                   'index': idx,
                   'type': AnomalyType.MOMENTUM_REVERSAL.value,
                   'magnitude': abs(long_term - short_term),
                   'z_score': 2.0,
                   'explanation': f"Momentum reversal: 30d={long_term:.1f}% vs 3d={short_term:.1f}%"
               })
       
       return pd.DataFrame(anomalies)
   
   def _detect_earnings_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       if 'eps_change_pct' not in df:
           return pd.DataFrame()
       
       # EPS explosions
       explosion_threshold = self.thresholds.get('eps_explosion', 40.0)
       explosion_mask = df['eps_change_pct'] > explosion_threshold
       
       for idx in df[explosion_mask].index:
           change = df.loc[idx, 'eps_change_pct']
           
           # Check if accompanied by volume
           volume_context = ""
           if 'vol_ratio_1d_90d' in df:
               vol_ratio = df.loc[idx, 'vol_ratio_1d_90d']
               if vol_ratio > 2:
                   volume_context = " with high volume"
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.EARNINGS_EXPLOSION.value,
               'magnitude': change,
               'z_score': self._calculate_zscore(df['eps_change_pct'].dropna(), change),
               'explanation': f"EPS surged {change:.0f}%{volume_context}"
           })
       
       # EPS collapses
       collapse_threshold = self.thresholds.get('eps_collapse', -30.0)
       collapse_mask = df['eps_change_pct'] < collapse_threshold
       
       for idx in df[collapse_mask].index:
           change = df.loc[idx, 'eps_change_pct']
           
           anomalies.append({
               'index': idx,
               'type': AnomalyType.EARNINGS_COLLAPSE.value,
               'magnitude': abs(change),
               'z_score': abs(self._calculate_zscore(df['eps_change_pct'].dropna(), change)),
               'explanation': f"EPS collapsed {abs(change):.0f}%"
           })
       
       return pd.DataFrame(anomalies)
   
   def _detect_technical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       # 52-week breakouts
       if all(col in df for col in ['price', 'high_52w', 'from_high_pct']):
           breakout_mask = (
               (df['price'] >= df['high_52w'] * 0.98) |
               (df['from_high_pct'] < self.config.breakout_proximity * 100)
           )
           
           for idx in df[breakout_mask].index:
               price = df.loc[idx, 'price']
               high_52w = df.loc[idx, 'high_52w']
               
               if price > high_52w:
                   anomalies.append({
                       'index': idx,
                       'type': AnomalyType.TECHNICAL_BREAKOUT.value,
                       'magnitude': ((price - high_52w) / high_52w) * 100,
                       'z_score': 2.5,
                       'explanation': f"New 52w high at ₹{price:.2f}"
                   })
               else:
                   anomalies.append({
                       'index': idx,
                       'type': AnomalyType.TECHNICAL_BREAKOUT.value,
                       'magnitude': df.loc[idx, 'from_high_pct'],
                       'z_score': 2.0,
                       'explanation': f"Near 52w high ({df.loc[idx, 'from_high_pct']:.1f}% away)"
                   })
       
       # 52-week breakdowns
       if all(col in df for col in ['price', 'low_52w', 'from_low_pct']):
           breakdown_mask = (
               (df['price'] <= df['low_52w'] * 1.02) |
               (df['from_low_pct'] < self.config.breakdown_proximity * 100)
           )
           
           for idx in df[breakdown_mask].index:
               price = df.loc[idx, 'price']
               low_52w = df.loc[idx, 'low_52w']
               
               if price < low_52w:
                   anomalies.append({
                       'index': idx,
                       'type': AnomalyType.TECHNICAL_BREAKDOWN.value,
                       'magnitude': ((low_52w - price) / low_52w) * 100,
                       'z_score': 2.5,
                       'explanation': f"New 52w low at ₹{price:.2f}"
                   })
       
       # Moving average crossovers
       if all(col in df for col in ['price', 'sma_50d', 'sma_200d']):
           golden_cross_mask = (
               (df['sma_50d'] > df['sma_200d']) &
               (df['price'] > df['sma_50d'])
           )
           
           for idx in df[golden_cross_mask].index:
               if idx > 0 and idx - 1 in df.index:
                   prev_50 = df.loc[idx - 1, 'sma_50d'] if idx - 1 in df.index else 0
                   prev_200 = df.loc[idx - 1, 'sma_200d'] if idx - 1 in df.index else 0
                   
                   if prev_50 <= prev_200:
                       anomalies.append({
                           'index': idx,
                           'type': AnomalyType.TECHNICAL_BREAKOUT.value,
                           'magnitude': 3.0,
                           'z_score': 2.0,
                           'explanation': "Golden cross detected"
                       })
       
       return pd.DataFrame(anomalies)
   
   def _detect_distribution_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       # Detect statistical outliers using multiple methods
       numeric_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'volume_1d', 'pe', 'eps_change_pct']
       available_cols = [col for col in numeric_cols if col in df]
       
       for col in available_cols:
           if df[col].notna().sum() < self.config.min_data_points:
               continue
           
           # Z-score method
           z_scores = np.abs(stats.zscore(df[col].dropna()))
           z_outliers = df.index[z_scores > self.config.z_score_threshold]
           
           # MAD method
           mad = stats.median_abs_deviation(df[col].dropna())
           median = df[col].median()
           mad_scores = np.abs((df[col] - median) / (mad * 1.4826))
           mad_outliers = df.index[mad_scores > self.config.mad_multiplier]
           
           # Combine outliers
           outliers = set(z_outliers) | set(mad_outliers)
           
           for idx in outliers:
               if idx in df.index:
                   value = df.loc[idx, col]
                   z_score = z_scores[df.index.get_loc(idx)] if idx in z_outliers else 0
                   
                   anomalies.append({
                       'index': idx,
                       'type': AnomalyType.DISTRIBUTION_SHIFT.value,
                       'magnitude': abs(value),
                       'z_score': z_score,
                       'explanation': f"Statistical outlier in {col}: {value:.2f}"
                   })
       
       return pd.DataFrame(anomalies)
   
   def _detect_sector_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
       anomalies = []
       
       if 'sector' not in df or 'ret_30d' not in df:
           return pd.DataFrame()
       
       # Calculate sector statistics
       sector_stats = df.groupby('sector')['ret_30d'].agg(['mean', 'std', 'count'])
       sector_stats = sector_stats[sector_stats['count'] >= 3]
       
       for idx, row in df.iterrows():
           sector = row['sector']
           if sector not in sector_stats.index:
               continue
           
           sector_mean = sector_stats.loc[sector, 'mean']
           sector_std = sector_stats.loc[sector, 'std']
           
           if sector_std > 0:
               z_score = (row['ret_30d'] - sector_mean) / sector_std
               
               if abs(z_score) > 2:
                   anomalies.append({
                       'index': idx,
                       'type': AnomalyType.SECTOR_OUTLIER.value,
                       'magnitude': abs(row['ret_30d'] - sector_mean),
                       'z_score': abs(z_score),
                       'explanation': f"{'Outperforming' if z_score > 0 else 'Underperforming'} sector by {abs(z_score):.1f}σ"
                   })
       
       return pd.DataFrame(anomalies)
   
   def _process_multi_anomalies(self, df: pd.DataFrame, all_anomalies: pd.DataFrame) -> pd.DataFrame:
       if all_anomalies.empty:
           return df
       
       # Count anomalies per stock
       anomaly_counts = all_anomalies.groupby('index').size()
       multi_anomaly_indices = anomaly_counts[anomaly_counts > 1].index
       
       # Get primary anomaly for each stock
       for idx in all_anomalies['index'].unique():
           stock_anomalies = all_anomalies[all_anomalies['index'] == idx]
           
           # Sort by z-score to get most significant
           stock_anomalies = stock_anomalies.sort_values('z_score', ascending=False)
           primary = stock_anomalies.iloc[0]
           
           df.loc[idx, 'anomaly'] = True
           df.loc[idx, 'anomaly_type'] = primary['type']
           df.loc[idx, 'anomaly_explanation'] = primary['explanation']
           
           # Mark multi-anomalies
           if idx in multi_anomaly_indices:
               count = len(stock_anomalies)
               types = stock_anomalies['type'].unique()[:3]
               df.loc[idx, 'anomaly_type'] = AnomalyType.MULTI_SIGNAL.value
               df.loc[idx, 'anomaly_explanation'] = f"{count} anomalies: {', '.join(types)}"
       
       return df
   
   def _calculate_anomaly_scores(self, df: pd.DataFrame, all_anomalies: pd.DataFrame) -> pd.DataFrame:
       if all_anomalies.empty:
           return df
       
       for idx in all_anomalies['index'].unique():
           stock_anomalies = all_anomalies[all_anomalies['index'] == idx]
           
           # Calculate composite score
           max_z_score = stock_anomalies['z_score'].max()
           max_magnitude = stock_anomalies['magnitude'].max()
           anomaly_count = len(stock_anomalies)
           
           # Statistical rarity score
           rarity_score = min(max_z_score / self.config.z_score_threshold * 100, 100)
           
           # Magnitude score (normalized)
           magnitude_score = min(max_magnitude / 10 * 100, 100)
           
           # Multi-anomaly bonus
           consistency_score = min(anomaly_count * 20, 100)
           
           # Context score (market regime adjustment)
           context_score = 50
           if self.market_context['volatility_regime'] == 'low' and max_magnitude > 5:
               context_score = 80
           elif self.market_context['volatility_regime'] == 'high' and max_magnitude < 10:
               context_score = 20
           
           # Weighted composite
           weights = self.config.severity_weights
           composite_score = (
               rarity_score * weights['statistical_rarity'] +
               magnitude_score * weights['magnitude'] +
               consistency_score * weights['consistency'] +
               context_score * weights['context']
           )
           
           # Apply multi-anomaly boost
           if anomaly_count > 1:
               composite_score *= self.config.multi_anomaly_boost
           
           composite_score = min(composite_score, 100)
           
           # Determine severity
           if composite_score >= 80:
               severity = AnomalySeverity.CRITICAL
           elif composite_score >= 70:
               severity = AnomalySeverity.EXTREME
           elif composite_score >= 60:
               severity = AnomalySeverity.MAJOR
           elif composite_score >= 40:
               severity = AnomalySeverity.MODERATE
           else:
               severity = AnomalySeverity.MILD
           
           df.loc[idx, 'anomaly_score'] = round(composite_score, 1)
           df.loc[idx, 'anomaly_severity'] = severity.value[0]
       
       return df
   
   def _calculate_zscore(self, series: pd.Series, value: float) -> float:
       mean = series.mean()
       std = series.std()
       
       if std == 0:
           return 0
       
       return abs((value - mean) / std)


def run_anomaly_detector(
   df: pd.DataFrame,
   config: Optional[AnomalyConfig] = None,
   return_details: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
   
   detector = QuantAnomalyDetector(config)
   result_df = detector.detect_anomalies(df)
   
   # Log summary
   anomaly_count = result_df['anomaly'].sum()
   logger.info(f"Detected {anomaly_count} anomalies out of {len(df)} stocks")
   
   if return_details:
       details = {
           'market_context': detector.market_context,
           'thresholds': detector.thresholds,
           'summary': {
               'total_anomalies': anomaly_count,
               'by_type': result_df[result_df['anomaly']]['anomaly_type'].value_counts().to_dict(),
               'by_severity': result_df[result_df['anomaly']]['anomaly_severity'].value_counts().to_dict()
           }
       }
       return result_df, details
   
   return result_df


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
   if 'anomaly' not in df:
       return pd.DataFrame()
   
   anomaly_df = df[df['anomaly']].copy()
   
   if anomaly_df.empty:
       return pd.DataFrame()
   
   summary_cols = ['ticker', 'company_name', 'anomaly_type', 'anomaly_severity', 
                   'anomaly_score', 'anomaly_explanation', 'price', 'ret_1d', 
                   'ret_30d', 'volume_1d', 'eps_change_pct']
   
   available_cols = [col for col in summary_cols if col in anomaly_df.columns]
   
   return anomaly_df[available_cols].sort_values('anomaly_score', ascending=False)
