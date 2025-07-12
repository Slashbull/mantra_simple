"""
anomaly_detector.py - M.A.N.T.R.A. Anomaly Detection
===================================================
Detects unusual patterns in price, volume, and fundamentals
Identifies potential breakouts, reversals, and risks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import from constants
from constants import (
    VOLUME_LEVELS, MOMENTUM_LEVELS, ALERT_PRIORITY,
    PRICE_POSITION, PE_RANGES
)

logger = logging.getLogger(__name__)

# ============================================================================
# ANOMALY TYPES
# ============================================================================

@dataclass
class AnomalyThresholds:
    """Thresholds for detecting various anomalies"""
    
    # Price anomalies
    price_spike_1d: float = 5.0      # >5% in a day
    price_crash_1d: float = -5.0     # <-5% in a day
    price_spike_7d: float = 15.0     # >15% in a week
    price_crash_7d: float = -15.0    # <-15% in a week
    
    # Volume anomalies
    volume_spike_multiplier: float = 3.0   # 3x normal volume
    volume_dry_multiplier: float = 0.3     # 30% of normal
    sustained_volume_days: int = 3         # High volume for 3+ days
    
    # Fundamental anomalies
    eps_surprise_positive: float = 20.0    # EPS beat by >20%
    eps_surprise_negative: float = -20.0   # EPS miss by >20%
    pe_extreme_high: float = 50.0          # PE > 50
    pe_extreme_low: float = 5.0            # PE < 5
    
    # Technical anomalies
    sma_breakout_distance: float = 3.0     # >3% above SMA
    near_52w_high_pct: float = 95.0        # Within 5% of 52w high
    near_52w_low_pct: float = 5.0          # Within 5% of 52w low
    
    # Pattern thresholds
    reversal_min_decline: float = -20.0    # 20% decline before reversal
    breakout_consolidation_days: int = 20  # 20 days of sideways

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """
    Detects various types of anomalies in stock data
    """
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        self.thresholds = thresholds or AnomalyThresholds()
        self.anomalies_found = []
        
    def detect_all_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all types of anomalies in the data
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with anomaly flags and details
        """
        if df.empty:
            logger.warning("Empty dataframe provided to anomaly detector")
            return df
            
        df = df.copy()
        logger.info(f"Detecting anomalies for {len(df)} stocks...")
        
        # Price anomalies
        self._detect_price_anomalies(df)
        
        # Volume anomalies
        self._detect_volume_anomalies(df)
        
        # Fundamental anomalies
        self._detect_fundamental_anomalies(df)
        
        # Technical anomalies
        self._detect_technical_anomalies(df)
        
        # Pattern-based anomalies
        self._detect_pattern_anomalies(df)
        
        # Aggregate anomaly information
        df['anomaly_count'] = (
            df['has_price_anomaly'].astype(int) +
            df['has_volume_anomaly'].astype(int) +
            df['has_fundamental_anomaly'].astype(int) +
            df['has_technical_anomaly'].astype(int) +
            df['has_pattern_anomaly'].astype(int)
        )
        
        # Create anomaly summary
        df['anomaly_summary'] = df.apply(self._create_anomaly_summary, axis=1)
        
        # Assign anomaly priority
        df['anomaly_priority'] = df.apply(self._assign_anomaly_priority, axis=1)
        
        # Log summary
        total_anomalies = (df['anomaly_count'] > 0).sum()
        logger.info(f"Found anomalies in {total_anomalies} stocks")
        
        return df
    
    # ========================================================================
    # PRICE ANOMALIES
    # ========================================================================
    
    def _detect_price_anomalies(self, df: pd.DataFrame):
        """Detect unusual price movements"""
        # Daily spikes/crashes
        df['price_spike_1d'] = False
        df['price_crash_1d'] = False
        
        if 'ret_1d' in df.columns:
            df['price_spike_1d'] = df['ret_1d'] > self.thresholds.price_spike_1d
            df['price_crash_1d'] = df['ret_1d'] < self.thresholds.price_crash_1d
        
        # Weekly spikes/crashes
        df['price_spike_7d'] = False
        df['price_crash_7d'] = False
        
        if 'ret_7d' in df.columns:
            df['price_spike_7d'] = df['ret_7d'] > self.thresholds.price_spike_7d
            df['price_crash_7d'] = df['ret_7d'] < self.thresholds.price_crash_7d
        
        # Gap detection (large single-day moves)
        df['price_gap'] = False
        if 'ret_1d' in df.columns:
            df['price_gap'] = df['ret_1d'].abs() > 10
        
        # Unusual momentum (accelerating moves)
        df['momentum_acceleration'] = False
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            # Each period stronger than the last
            df['momentum_acceleration'] = (
                (df['ret_1d'] > 2) & 
                (df['ret_7d'] > df['ret_1d'] * 3) &
                (df['ret_30d'] > df['ret_7d'] * 2)
            )
        
        # Aggregate price anomalies
        df['has_price_anomaly'] = (
            df['price_spike_1d'] | df['price_crash_1d'] |
            df['price_spike_7d'] | df['price_crash_7d'] |
            df['price_gap'] | df['momentum_acceleration']
        )
    
    # ========================================================================
    # VOLUME ANOMALIES
    # ========================================================================
    
    def _detect_volume_anomalies(self, df: pd.DataFrame):
        """Detect unusual volume patterns"""
        # Volume spike
        df['volume_spike'] = False
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > self.thresholds.volume_spike_multiplier
        
        # Volume dry-up
        df['volume_dryup'] = False
        if 'rvol' in df.columns:
            df['volume_dryup'] = df['rvol'] < self.thresholds.volume_dry_multiplier
        
        # Volume expansion trend
        df['volume_expansion'] = False
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d']):
            # Convert percentage strings if needed
            vol_1d = df['vol_ratio_1d_90d']
            vol_7d = df['vol_ratio_7d_90d']
            
            if vol_1d.dtype == 'object':
                vol_1d = pd.to_numeric(vol_1d.str.replace('%', ''), errors='coerce')
            if vol_7d.dtype == 'object':
                vol_7d = pd.to_numeric(vol_7d.str.replace('%', ''), errors='coerce')
            
            df['volume_expansion'] = (vol_1d > 50) & (vol_7d > 30)
        
        # Price-volume divergence
        df['price_volume_divergence'] = False
        if 'ret_1d' in df.columns and 'rvol' in df.columns:
            # High volume but flat price
            df['price_volume_divergence'] = (
                (df['rvol'] > 2) & (df['ret_1d'].abs() < 1)
            )
        
        # Aggregate volume anomalies
        df['has_volume_anomaly'] = (
            df['volume_spike'] | df['volume_dryup'] |
            df['volume_expansion'] | df['price_volume_divergence']
        )
    
    # ========================================================================
    # FUNDAMENTAL ANOMALIES
    # ========================================================================
    
    def _detect_fundamental_anomalies(self, df: pd.DataFrame):
        """Detect unusual fundamental metrics"""
        # EPS surprises
        df['eps_surprise_positive'] = False
        df['eps_surprise_negative'] = False
        
        if 'eps_change_pct' in df.columns:
            df['eps_surprise_positive'] = df['eps_change_pct'] > self.thresholds.eps_surprise_positive
            df['eps_surprise_negative'] = df['eps_change_pct'] < self.thresholds.eps_surprise_negative
        
        # PE extremes
        df['pe_extreme_high'] = False
        df['pe_extreme_low'] = False
        
        if 'pe' in df.columns:
            df['pe_extreme_high'] = df['pe'] > self.thresholds.pe_extreme_high
            df['pe_extreme_low'] = (df['pe'] > 0) & (df['pe'] < self.thresholds.pe_extreme_low)
        
        # Value anomaly (low PE + high growth)
        df['value_anomaly'] = False
        if 'pe' in df.columns and 'eps_change_pct' in df.columns:
            df['value_anomaly'] = (
                (df['pe'] > 0) & (df['pe'] < 15) & 
                (df['eps_change_pct'] > 25)
            )
        
        # Growth anomaly (high growth at reasonable PE)
        df['growth_anomaly'] = False
        if 'pe' in df.columns and 'eps_change_pct' in df.columns:
            df['growth_anomaly'] = (
                (df['pe'] > 0) & (df['pe'] < 30) & 
                (df['eps_change_pct'] > 50)
            )
        
        # Aggregate fundamental anomalies
        df['has_fundamental_anomaly'] = (
            df['eps_surprise_positive'] | df['eps_surprise_negative'] |
            df['pe_extreme_high'] | df['pe_extreme_low'] |
            df['value_anomaly'] | df['growth_anomaly']
        )
    
    # ========================================================================
    # TECHNICAL ANOMALIES
    # ========================================================================
    
    def _detect_technical_anomalies(self, df: pd.DataFrame):
        """Detect unusual technical patterns"""
        # SMA breakouts
        df['sma_breakout'] = False
        for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
            if sma in df.columns and 'price' in df.columns:
                distance = ((df['price'] - df[sma]) / df[sma] * 100)
                df[f'{sma}_breakout'] = distance > self.thresholds.sma_breakout_distance
                df['sma_breakout'] |= df[f'{sma}_breakout']
        
        # Near 52-week extremes
        df['near_52w_high'] = False
        df['near_52w_low'] = False
        
        if 'position_52w' in df.columns:
            df['near_52w_high'] = df['position_52w'] > self.thresholds.near_52w_high_pct
            df['near_52w_low'] = df['position_52w'] < self.thresholds.near_52w_low_pct
        
        # Technical strength (all SMAs aligned)
        df['technical_strength'] = False
        if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d', 'price']):
            df['technical_strength'] = (
                (df['price'] > df['sma_20d']) &
                (df['sma_20d'] > df['sma_50d']) &
                (df['sma_50d'] > df['sma_200d'])
            )
        
        # Support/resistance test
        df['support_test'] = False
        df['resistance_test'] = False
        
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            # Testing support (within 5% of 52w low)
            df['support_test'] = (
                (df['price'] - df['low_52w']) / df['low_52w'] * 100 < 5
            )
            # Testing resistance (within 5% of 52w high)
            df['resistance_test'] = (
                (df['high_52w'] - df['price']) / df['price'] * 100 < 5
            )
        
        # Aggregate technical anomalies
        df['has_technical_anomaly'] = (
            df['sma_breakout'] | df['near_52w_high'] | 
            df['near_52w_low'] | df['technical_strength'] |
            df['support_test'] | df['resistance_test']
        )
    
    # ========================================================================
    # PATTERN ANOMALIES
    # ========================================================================
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame):
        """Detect complex pattern-based anomalies"""
        # Reversal pattern (oversold bounce)
        df['reversal_pattern'] = False
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'position_52w']):
            df['reversal_pattern'] = (
                (df['ret_30d'] < self.thresholds.reversal_min_decline) &
                (df['ret_7d'] > 5) &
                (df['position_52w'] < 30)
            )
        
        # Breakout pattern (consolidation breakout)
        df['breakout_pattern'] = False
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'rvol']):
            # Low 30d return (consolidation) but strong recent move with volume
            df['breakout_pattern'] = (
                (df['ret_30d'].abs() < 10) &
                (df['ret_7d'] > 5) &
                (df['rvol'] > 2)
            )
        
        # Momentum continuation
        df['momentum_continuation'] = False
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'ret_3m']):
            # All timeframes positive and increasing
            df['momentum_continuation'] = (
                (df['ret_7d'] > 5) &
                (df['ret_30d'] > 10) &
                (df['ret_3m'] > 20) &
                (df['ret_7d'] > df['ret_30d'] / 4)  # Accelerating
            )
        
        # Distribution pattern (topping)
        df['distribution_pattern'] = False
        if all(col in df.columns for col in ['position_52w', 'rvol', 'ret_7d']):
            # Near highs but weakening
            df['distribution_pattern'] = (
                (df['position_52w'] > 80) &
                (df['rvol'] < 0.7) &
                (df['ret_7d'] < 0)
            )
        
        # Aggregate pattern anomalies
        df['has_pattern_anomaly'] = (
            df['reversal_pattern'] | df['breakout_pattern'] |
            df['momentum_continuation'] | df['distribution_pattern']
        )
    
    # ========================================================================
    # SUMMARY AND PRIORITY
    # ========================================================================
    
    def _create_anomaly_summary(self, row: pd.Series) -> str:
        """Create summary of all anomalies for a stock"""
        anomalies = []
        
        # Price anomalies
        if row.get('price_spike_1d', False):
            anomalies.append(f"Price spike {row.get('ret_1d', 0):.1f}%")
        if row.get('price_crash_1d', False):
            anomalies.append(f"Price crash {row.get('ret_1d', 0):.1f}%")
        if row.get('momentum_acceleration', False):
            anomalies.append("Momentum accelerating")
        
        # Volume anomalies
        if row.get('volume_spike', False):
            anomalies.append(f"Volume spike {row.get('rvol', 0):.1f}x")
        if row.get('volume_dryup', False):
            anomalies.append("Volume dried up")
        if row.get('price_volume_divergence', False):
            anomalies.append("Price-volume divergence")
        
        # Fundamental anomalies
        if row.get('eps_surprise_positive', False):
            anomalies.append(f"EPS beat {row.get('eps_change_pct', 0):.0f}%")
        if row.get('value_anomaly', False):
            anomalies.append("Value opportunity")
        if row.get('growth_anomaly', False):
            anomalies.append("High growth")
        
        # Technical anomalies
        if row.get('sma_breakout', False):
            anomalies.append("SMA breakout")
        if row.get('near_52w_high', False):
            anomalies.append("Near 52W high")
        if row.get('near_52w_low', False):
            anomalies.append("Near 52W low")
        
        # Pattern anomalies
        if row.get('reversal_pattern', False):
            anomalies.append("Reversal pattern")
        if row.get('breakout_pattern', False):
            anomalies.append("Breakout pattern")
        if row.get('momentum_continuation', False):
            anomalies.append("Momentum continuation")
        if row.get('distribution_pattern', False):
            anomalies.append("Distribution pattern")
        
        return " | ".join(anomalies[:3]) if anomalies else "None"
    
    def _assign_anomaly_priority(self, row: pd.Series) -> str:
        """Assign priority to anomalies"""
        priority_score = 0
        
        # High priority patterns
        if row.get('breakout_pattern', False):
            priority_score += 3
        if row.get('reversal_pattern', False):
            priority_score += 3
        if row.get('value_anomaly', False):
            priority_score += 3
        
        # Medium priority patterns
        if row.get('volume_spike', False) and row.get('price_spike_1d', False):
            priority_score += 2
        if row.get('momentum_continuation', False):
            priority_score += 2
        if row.get('eps_surprise_positive', False):
            priority_score += 2
        
        # Low priority patterns
        if row.get('near_52w_high', False) or row.get('near_52w_low', False):
            priority_score += 1
        if row.get('sma_breakout', False):
            priority_score += 1
        
        # Negative patterns reduce priority
        if row.get('distribution_pattern', False):
            priority_score -= 2
        if row.get('price_crash_1d', False):
            priority_score -= 1
        
        # Convert to priority level
        if priority_score >= 5:
            return "Critical"
        elif priority_score >= 3:
            return "High"
        elif priority_score >= 1:
            return "Medium"
        else:
            return "Low"

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_anomalies(
    df: pd.DataFrame,
    thresholds: Optional[AnomalyThresholds] = None
) -> pd.DataFrame:
    """
    Detect all anomalies in stock data
    
    Args:
        df: DataFrame with stock data
        thresholds: Custom anomaly thresholds
        
    Returns:
        DataFrame with anomaly flags and details
    """
    detector = AnomalyDetector(thresholds)
    return detector.detect_all_anomalies(df)

def get_critical_anomalies(
    df: pd.DataFrame,
    priority_levels: List[str] = ['Critical', 'High']
) -> pd.DataFrame:
    """
    Get stocks with critical anomalies
    
    Args:
        df: DataFrame with anomaly detection results
        priority_levels: Priority levels to include
        
    Returns:
        Filtered DataFrame
    """
    if 'anomaly_priority' not in df.columns:
        logger.error("No anomaly_priority column found")
        return pd.DataFrame()
    
    return df[df['anomaly_priority'].isin(priority_levels)].sort_values(
        'anomaly_count', ascending=False
    )

def get_anomaly_statistics(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of anomalies
    
    Args:
        df: DataFrame with anomaly detection results
        
    Returns:
        Dictionary with statistics
    """
    if 'anomaly_count' not in df.columns:
        return {'error': 'No anomaly data found'}
    
    stats = {
        'total_stocks': len(df),
        'stocks_with_anomalies': (df['anomaly_count'] > 0).sum(),
        'anomaly_types': {
            'price': df['has_price_anomaly'].sum(),
            'volume': df['has_volume_anomaly'].sum(),
            'fundamental': df['has_fundamental_anomaly'].sum(),
            'technical': df['has_technical_anomaly'].sum(),
            'pattern': df['has_pattern_anomaly'].sum()
        },
        'priority_distribution': df['anomaly_priority'].value_counts().to_dict(),
        'top_anomalies': df.nlargest(5, 'anomaly_count')[
            ['ticker', 'anomaly_count', 'anomaly_summary', 'anomaly_priority']
        ].to_dict('records')
    }
    
    return stats

def get_specific_anomaly(
    df: pd.DataFrame,
    anomaly_type: str
) -> pd.DataFrame:
    """
    Get stocks with specific type of anomaly
    
    Args:
        df: DataFrame with anomaly detection results
        anomaly_type: Type of anomaly to filter
        
    Returns:
        Filtered DataFrame
    """
    anomaly_columns = {
        'breakout': 'breakout_pattern',
        'reversal': 'reversal_pattern',
        'volume_spike': 'volume_spike',
        'value': 'value_anomaly',
        'growth': 'growth_anomaly',
        'momentum': 'momentum_continuation',
        'crash': 'price_crash_1d',
        'spike': 'price_spike_1d'
    }
    
    if anomaly_type not in anomaly_columns:
        logger.error(f"Unknown anomaly type: {anomaly_type}")
        return pd.DataFrame()
    
    column = anomaly_columns[anomaly_type]
    if column not in df.columns:
        logger.error(f"Column {column} not found")
        return pd.DataFrame()
    
    return df[df[column] == True].sort_values('anomaly_priority')

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Anomaly Detector")
    print("="*60)
    print("\nDetects unusual patterns and opportunities")
    print("\nAnomaly Types:")
    print("- Price: Spikes, crashes, gaps, acceleration")
    print("- Volume: Spikes, dry-ups, divergences")
    print("- Fundamental: EPS surprises, PE extremes, value plays")
    print("- Technical: Breakouts, 52W extremes, SMA violations")
    print("- Patterns: Reversals, breakouts, continuation, distribution")
    print("\nUse detect_anomalies() to scan for opportunities")
    print("="*60)
