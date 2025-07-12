"""
signal_engine.py - M.A.N.T.R.A. Multi-Factor Scoring Engine
==========================================================
FINAL PRODUCTION VERSION
Calculates composite scores using momentum, value, technical, volume, and sector factors.
Handles all edge cases, Indian market specifics, and is 100% Streamlit-ready.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from collections import deque
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Import constants
try:
    from constants import (
        SCORE_WEIGHTS, MOMENTUM_THRESHOLDS, VALUE_THRESHOLDS,
        TECHNICAL_THRESHOLDS, VOLUME_THRESHOLDS, SIGNAL_THRESHOLDS
    )
except ImportError:
    # Fallback defaults
    SCORE_WEIGHTS = {
        'MOMENTUM': 0.30,
        'VALUE': 0.25,
        'TECHNICAL': 0.20,
        'VOLUME': 0.15,
        'SECTOR': 0.10
    }
    SIGNAL_THRESHOLDS = {'BUY': 85, 'WATCH': 70, 'AVOID': 50}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================================
# ENUMS & CONFIGURATION
# ============================================================================

class ScoringMethod(Enum):
    """Available scoring methods"""
    PERCENTILE = "percentile"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    LINEAR = "linear"
    CUSTOM = "custom"

class RegimeType(Enum):
    """Market regime types affecting factor weights"""
    BALANCED = "balanced"
    MOMENTUM = "momentum"
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    RECOVERY = "recovery"
    VOLATILITY = "volatility"

@dataclass
class SignalConfig:
    """Configuration for signal engine"""
    # Score boundaries
    default_score: float = 50.0
    min_score: float = 0.0
    max_score: float = 100.0
    
    # Data quality
    min_valid_data_pct: float = 0.3
    outlier_clip_pct: float = 1.0
    
    # Features
    use_sector_relative: bool = True
    track_explanations: bool = True
    enable_advanced_factors: bool = True
    
    # Performance
    cache_factor_scores: bool = True
    parallel_processing: bool = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Indian number format handling
_INDIAN_UNITS = {
    "CR": 1e7,     # Crore
    "L": 1e5,      # Lakh  
    "LAC": 1e5,    # Lakh alternate
    "K": 1e3,      # Thousand
    "M": 1e6,      # Million
    "B": 1e9,      # Billion
}

def parse_indian_number(val: Any) -> float:
    """Convert Indian format numbers like '₹3.4 Cr' to float"""
    if pd.isna(val) or val is None:
        return np.nan
    
    if isinstance(val, (int, float)):
        return float(val)
    
    # Clean string
    s = str(val).upper().strip()
    s = s.replace("₹", "").replace("$", "").replace(",", "").strip()
    
    # Check for units
    for unit, factor in _INDIAN_UNITS.items():
        if s.endswith(unit):
            try:
                num_part = s.replace(unit, "").strip()
                return float(num_part) * factor
            except ValueError:
                return np.nan
    
    # Try direct conversion
    try:
        return float(s)
    except ValueError:
        return np.nan

def clean_percentage(val: Any) -> float:
    """Clean percentage values, handling % symbol"""
    if pd.isna(val):
        return np.nan
    
    s = str(val).strip()
    if s.endswith('%'):
        try:
            return float(s.replace('%', ''))
        except ValueError:
            return np.nan
    
    try:
        val_float = float(val)
        # If value is between -1 and 1, likely a decimal percentage
        if -1 <= val_float <= 1 and val_float != 0:
            return val_float * 100
        return val_float
    except ValueError:
        return np.nan

# ============================================================================
# SIGNAL ENGINE CLASS
# ============================================================================

class SignalEngine:
    """
    Multi-factor scoring engine for M.A.N.T.R.A.
    Combines multiple signals to generate composite scores.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.factor_cache = {}
        self.explanations = {}
        self.audit_log = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Define factor specifications
        self.factor_definitions = self._define_factors()
        self.regime_weights = self._define_regime_weights()
        
        logger.info("Signal Engine initialized")
    
    def _define_factors(self) -> Dict[str, Dict[str, Any]]:
        """Define all factor specifications"""
        return {
            # Core factors
            "momentum": {
                "columns": ["ret_1d", "ret_3d", "ret_7d", "ret_30d", "ret_3m"],
                "weights": [0.05, 0.10, 0.20, 0.35, 0.30],
                "method": ScoringMethod.PERCENTILE,
                "higher_better": True,
                "description": "Price momentum across multiple timeframes"
            },
            
            "value": {
                "columns": ["pe", "eps_current", "from_low_pct"],
                "method": ScoringMethod.CUSTOM,
                "higher_better": False,  # Lower PE is better
                "description": "Valuation attractiveness"
            },
            
            "technical": {
                "columns": ["price", "sma_20d", "sma_50d", "sma_200d"],
                "method": ScoringMethod.CUSTOM,
                "description": "Technical indicator alignment"
            },
            
            "volume": {
                "columns": ["rvol", "vol_ratio_1d_90d", "vol_ratio_7d_90d"],
                "weights": [0.50, 0.30, 0.20],
                "method": ScoringMethod.ROBUST,
                "higher_better": True,
                "description": "Volume patterns and liquidity"
            },
            
            "sector_strength": {
                "columns": ["sector"],
                "method": ScoringMethod.CUSTOM,
                "description": "Sector relative performance"
            },
            
            # Advanced factors (if enabled)
            "momentum_quality": {
                "columns": ["ret_3d", "ret_7d", "ret_30d", "ret_3m"],
                "method": ScoringMethod.CUSTOM,
                "description": "Consistency of momentum"
            },
            
            "earnings_growth": {
                "columns": ["eps_change_pct", "eps_current", "eps_last_qtr"],
                "method": ScoringMethod.ROBUST,
                "higher_better": True,
                "description": "Earnings growth momentum"
            },
            
            "risk_adjusted_return": {
                "columns": ["ret_30d", "ret_7d", "ret_3d"],
                "method": ScoringMethod.CUSTOM,
                "description": "Return adjusted for volatility"
            },
            
            "trend_strength": {
                "columns": ["price", "sma_20d", "sma_50d", "sma_200d", "trading_under"],
                "method": ScoringMethod.CUSTOM,
                "description": "Strength of price trend"
            },
            
            "relative_strength": {
                "columns": ["from_high_pct", "from_low_pct", "position_52w"],
                "method": ScoringMethod.CUSTOM,
                "description": "Position within 52-week range"
            }
        }
    
    def _define_regime_weights(self) -> Dict[RegimeType, Dict[str, float]]:
        """Define factor weights for different market regimes"""
        
        # Base balanced weights
        base_weights = {
            "momentum": 0.30,
            "value": 0.25,
            "technical": 0.20,
            "volume": 0.15,
            "sector_strength": 0.10
        }
        
        # Advanced factor weights (distributed when enabled)
        advanced_weights = {
            "momentum_quality": 0.05,
            "earnings_growth": 0.05,
            "risk_adjusted_return": 0.05,
            "trend_strength": 0.03,
            "relative_strength": 0.02
        }
        
        # Regime-specific adjustments
        regimes = {
            RegimeType.BALANCED: {**base_weights},
            
            RegimeType.MOMENTUM: {
                "momentum": 0.40,
                "value": 0.15,
                "technical": 0.25,
                "volume": 0.15,
                "sector_strength": 0.05
            },
            
            RegimeType.VALUE: {
                "momentum": 0.15,
                "value": 0.40,
                "technical": 0.15,
                "volume": 0.15,
                "sector_strength": 0.15
            },
            
            RegimeType.GROWTH: {
                "momentum": 0.25,
                "value": 0.20,
                "technical": 0.20,
                "volume": 0.15,
                "sector_strength": 0.20
            },
            
            RegimeType.QUALITY: {
                "momentum": 0.20,
                "value": 0.30,
                "technical": 0.25,
                "volume": 0.10,
                "sector_strength": 0.15
            },
            
            RegimeType.RECOVERY: {
                "momentum": 0.35,
                "value": 0.30,
                "technical": 0.15,
                "volume": 0.15,
                "sector_strength": 0.05
            },
            
            RegimeType.VOLATILITY: {
                "momentum": 0.20,
                "value": 0.35,
                "technical": 0.20,
                "volume": 0.20,
                "sector_strength": 0.05
            }
        }
        
        # Add advanced weights if enabled
        if self.config.enable_advanced_factors:
            for regime_type, weights in regimes.items():
                # Reduce core weights proportionally
                total_advanced = sum(advanced_weights.values())
                for key in weights:
                    weights[key] *= (1 - total_advanced)
                
                # Add advanced weights
                weights.update(advanced_weights)
        
        # Normalize to ensure sum = 1
        for weights in regimes.values():
            total = sum(weights.values())
            for key in weights:
                weights[key] /= total
        
        return regimes
    
    # ========================================================================
    # SCORING METHODS
    # ========================================================================
    
    def _score_percentile(self, series: pd.Series, higher_better: bool = True) -> pd.Series:
        """Score based on percentile ranking"""
        # Handle all NaN case
        if series.isna().all():
            return pd.Series(self.config.default_score, index=series.index)
        
        # Rank with NaN handling
        ranks = series.rank(method='average', ascending=higher_better, pct=True)
        scores = ranks * 100
        
        return scores.fillna(self.config.default_score).clip(
            self.config.min_score, 
            self.config.max_score
        )
    
    def _score_z_score(self, series: pd.Series, higher_better: bool = True) -> pd.Series:
        """Score based on z-score normalization"""
        if series.isna().all() or series.std() == 0:
            return pd.Series(self.config.default_score, index=series.index)
        
        z_scores = (series - series.mean()) / series.std()
        if not higher_better:
            z_scores = -z_scores
        
        # Convert to 0-100 scale using normal CDF
        scores = stats.norm.cdf(z_scores) * 100
        
        return scores.fillna(self.config.default_score).clip(
            self.config.min_score,
            self.config.max_score
        )
    
    def _score_robust(self, series: pd.Series, higher_better: bool = True) -> pd.Series:
        """Robust scoring using median and IQR"""
        if series.isna().all():
            return pd.Series(self.config.default_score, index=series.index)
        
        # Use median and IQR for robustness
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(self.config.default_score, index=series.index)
        
        # Robust z-score
        robust_z = (series - series.median()) / (1.4826 * iqr)
        if not higher_better:
            robust_z = -robust_z
        
        # Clip extreme values
        robust_z = robust_z.clip(-3, 3)
        
        # Convert to 0-100 scale
        scores = (robust_z + 3) / 6 * 100
        
        return scores.fillna(self.config.default_score).clip(
            self.config.min_score,
            self.config.max_score
        )
    
    # ========================================================================
    # FACTOR CALCULATIONS
    # ========================================================================
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor score"""
        mom_cols = ["ret_1d", "ret_3d", "ret_7d", "ret_30d", "ret_3m"]
        weights = [0.05, 0.10, 0.20, 0.35, 0.30]
        
        # Ensure columns exist
        for col in mom_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Calculate weighted momentum
        mom_df = df[mom_cols].copy()
        
        # Clip extreme values
        for col in mom_cols:
            mom_df[col] = mom_df[col].clip(-50, 200)
        
        # Calculate weighted score
        weighted_sum = pd.Series(0.0, index=df.index)
        for col, weight in zip(mom_cols, weights):
            col_score = self._score_percentile(mom_df[col], higher_better=True)
            weighted_sum += col_score * weight
        
        return weighted_sum
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate value factor score"""
        scores = pd.Series(self.config.default_score, index=df.index)
        
        # PE ratio scoring (lower is better)
        if 'pe' in df.columns:
            pe = df['pe'].copy()
            # Handle negative PE
            pe_score = pd.Series(50.0, index=df.index)
            
            # Best: PE between 10-20
            mask_best = pe.between(10, 20)
            pe_score[mask_best] = 100
            
            # Good: PE between 5-10 or 20-30
            mask_good = pe.between(5, 10) | pe.between(20, 30)
            pe_score[mask_good] = 80
            
            # Fair: PE between 30-50
            mask_fair = pe.between(30, 50)
            pe_score[mask_fair] = 60
            
            # Poor: PE > 50 or < 5
            mask_poor = (pe > 50) | (pe < 5)
            pe_score[mask_poor] = 30
            
            # Negative PE
            pe_score[pe < 0] = 20
            
            scores = pe_score * 0.5
        
        # EPS scoring
        if 'eps_current' in df.columns and 'eps_change_pct' in df.columns:
            # EPS growth
            eps_growth = df['eps_change_pct'].clip(-100, 200)
            eps_score = self._score_percentile(eps_growth, higher_better=True)
            
            # Positive EPS bonus
            positive_eps_bonus = (df['eps_current'] > 0).astype(float) * 10
            
            scores += (eps_score * 0.4 + positive_eps_bonus * 0.1)
        
        return scores.clip(self.config.min_score, self.config.max_score)
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate technical factor score"""
        score = pd.Series(50.0, index=df.index)
        
        # SMA alignment scoring
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            # Price above SMAs
            above_20 = (df['price'] > df['sma_20d']).astype(float) * 25
            above_50 = (df['price'] > df['sma_50d']).astype(float) * 25
            above_200 = (df['price'] > df['sma_200d']).astype(float) * 30
            
            # SMA alignment (20 > 50 > 200)
            sma_aligned = (
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            ).astype(float) * 20
            
            score = above_20 + above_50 + above_200 + sma_aligned
        
        # Trading position bonus
        if 'trading_under' in df.columns:
            # Bonus for not trading under any average
            not_under_any = df['trading_under'].isna() | (df['trading_under'] == '')
            score += not_under_any.astype(float) * 10
        
        return score.clip(self.config.min_score, self.config.max_score)
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume factor score"""
        vol_score = pd.Series(50.0, index=df.index)
        
        # Relative volume
        if 'rvol' in df.columns:
            rvol = df['rvol'].clip(0, 10)
            # High volume is good (but not extreme)
            rvol_score = pd.Series(50.0, index=df.index)
            rvol_score[rvol.between(1.5, 3)] = 100
            rvol_score[rvol.between(1, 1.5) | rvol.between(3, 5)] = 80
            rvol_score[rvol.between(0.5, 1) | rvol.between(5, 10)] = 60
            rvol_score[rvol < 0.5] = 30
            
            vol_score = rvol_score * 0.5
        
        # Volume ratios
        if 'vol_ratio_1d_90d' in df.columns:
            # Parse percentage strings if present
            vol_ratio = df['vol_ratio_1d_90d'].apply(clean_percentage)
            
            # Positive volume expansion is good
            ratio_score = self._score_percentile(vol_ratio, higher_better=True)
            vol_score += ratio_score * 0.5
        
        return vol_score.clip(self.config.min_score, self.config.max_score)
    
    def _calculate_sector_score(self, df: pd.DataFrame, sector_df: Optional[pd.DataFrame]) -> pd.Series:
        """Calculate sector strength score"""
        if sector_df is None or sector_df.empty:
            return pd.Series(self.config.default_score, index=df.index)
        
        if 'sector' not in df.columns:
            return pd.Series(self.config.default_score, index=df.index)
        
        # Use sector average returns
        score_col = 'sector_avg_3m'  # Default to 3-month average
        
        # Try different time horizons
        for col in ['sector_avg_3m', 'sector_avg_6m', 'sector_avg_1y', 'sector_avg_30d']:
            if col in sector_df.columns:
                score_col = col
                break
        
        if score_col not in sector_df.columns:
            return pd.Series(self.config.default_score, index=df.index)
        
        # Create sector score mapping
        sector_scores = sector_df.set_index('sector')[score_col]
        
        # Clean percentage values
        if sector_scores.dtype == 'object':
            sector_scores = sector_scores.apply(clean_percentage)
        
        # Map to stocks
        stock_sector_values = df['sector'].map(sector_scores)
        
        # Convert to percentile scores
        return self._score_percentile(stock_sector_values, higher_better=True)
    
    # ========================================================================
    # ADVANCED FACTORS
    # ========================================================================
    
    def _calculate_momentum_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum quality (consistency) score"""
        mom_cols = ['ret_3d', 'ret_7d', 'ret_30d', 'ret_3m']
        
        if not all(col in df.columns for col in mom_cols):
            return pd.Series(self.config.default_score, index=df.index)
        
        mom_df = df[mom_cols]
        
        # Check consistency (all positive or all negative)
        all_positive = (mom_df > 0).all(axis=1).astype(float) * 30
        all_negative = (mom_df < 0).all(axis=1).astype(float) * 10
        
        # Check if momentum is accelerating
        mom_values = mom_df.values
        accelerating = pd.Series(0.0, index=df.index)
        
        for i in range(len(mom_values)):
            row = mom_values[i]
            if not np.isnan(row).any():
                # Check if each period is stronger than the previous
                if all(row[j] >= row[j-1] for j in range(1, len(row))):
                    accelerating.iloc[i] = 40
                elif all(row[j] <= row[j-1] for j in range(1, len(row))):
                    accelerating.iloc[i] = 20
        
        # Low volatility in returns is good
        volatility = mom_df.std(axis=1)
        vol_score = self._score_percentile(volatility, higher_better=False) * 0.3
        
        return (all_positive + all_negative + accelerating + vol_score).clip(
            self.config.min_score, 
            self.config.max_score
        )
    
    def _calculate_earnings_growth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate earnings growth score"""
        if 'eps_change_pct' not in df.columns:
            return pd.Series(self.config.default_score, index=df.index)
        
        # Clean and clip EPS change
        eps_change = df['eps_change_pct'].apply(clean_percentage).clip(-100, 500)
        
        # Score based on growth tiers
        score = pd.Series(50.0, index=df.index)
        score[eps_change > 50] = 100
        score[eps_change.between(25, 50)] = 85
        score[eps_change.between(10, 25)] = 70
        score[eps_change.between(0, 10)] = 60
        score[eps_change.between(-10, 0)] = 40
        score[eps_change < -10] = 20
        
        return score
    
    def _calculate_risk_adjusted_return(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk-adjusted return (Sharpe-like) score"""
        ret_cols = ['ret_3d', 'ret_7d', 'ret_30d']
        
        if not all(col in df.columns for col in ret_cols):
            return pd.Series(self.config.default_score, index=df.index)
        
        returns = df[ret_cols]
        
        # Calculate mean return and volatility
        mean_return = returns.mean(axis=1)
        volatility = returns.std(axis=1)
        
        # Sharpe-like ratio (avoid division by zero)
        sharpe = mean_return / (volatility + 1e-6)
        
        # Convert to score
        return self._score_percentile(sharpe, higher_better=True)
    
    # ========================================================================
    # MAIN CALCULATION METHODS
    # ========================================================================
    
    def calculate_factor_scores(
        self, 
        df: pd.DataFrame, 
        sector_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Calculate all factor scores
        
        Returns:
            Tuple of (dataframe with scores, dict of individual factor scores)
        """
        if df.empty:
            logger.warning("Empty dataframe provided to calculate_factor_scores")
            return df, {}
        
        df = df.copy()
        factor_scores = {}
        
        # Calculate each factor
        with self._lock:
            # Core factors
            factor_scores['momentum'] = self._calculate_momentum_score(df)
            factor_scores['value'] = self._calculate_value_score(df)
            factor_scores['technical'] = self._calculate_technical_score(df)
            factor_scores['volume'] = self._calculate_volume_score(df)
            factor_scores['sector_strength'] = self._calculate_sector_score(df, sector_df)
            
            # Advanced factors (if enabled)
            if self.config.enable_advanced_factors:
                factor_scores['momentum_quality'] = self._calculate_momentum_quality(df)
                factor_scores['earnings_growth'] = self._calculate_earnings_growth(df)
                factor_scores['risk_adjusted_return'] = self._calculate_risk_adjusted_return(df)
            
            # Add factor scores to dataframe
            for name, scores in factor_scores.items():
                df[f'{name}_score'] = scores.round(2)
            
            # Log calculation
            self._log_event({
                'action': 'factor_calculation',
                'timestamp': datetime.now(),
                'factors_calculated': list(factor_scores.keys()),
                'rows': len(df)
            })
        
        return df, factor_scores
    
    def calculate_composite_score(
        self,
        df: pd.DataFrame,
        factor_scores: Dict[str, pd.Series],
        regime: RegimeType = RegimeType.BALANCED
    ) -> pd.DataFrame:
        """
        Calculate final composite score based on regime weights
        
        Args:
            df: DataFrame with stocks
            factor_scores: Dictionary of factor scores
            regime: Market regime for weighting
            
        Returns:
            DataFrame with composite scores added
        """
        df = df.copy()
        
        # Get regime weights
        weights = self.regime_weights.get(regime, self.regime_weights[RegimeType.BALANCED])
        
        # Calculate weighted composite
        composite = pd.Series(0.0, index=df.index)
        applied_weights = {}
        
        for factor_name, weight in weights.items():
            if factor_name in factor_scores:
                score = factor_scores[factor_name].fillna(self.config.default_score)
                composite += score * weight
                applied_weights[factor_name] = weight
            else:
                logger.debug(f"Factor '{factor_name}' not found in scores")
        
        # Normalize if weights don't sum to 1
        weight_sum = sum(applied_weights.values())
        if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
            composite = composite / weight_sum
        
        # Add to dataframe
        df['composite_score'] = composite.round(2)
        df['signal_strength'] = pd.cut(
            df['composite_score'],
            bins=[0, 30, 50, 70, 85, 100],
            labels=['Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong']
        )
        
        # Add percentile rank
        df['score_rank'] = df['composite_score'].rank(method='min', ascending=False)
        df['score_percentile'] = df['composite_score'].rank(pct=True) * 100
        
        # Store metadata
        df['scoring_regime'] = regime.value
        df['factors_used'] = ', '.join(applied_weights.keys())
        
        # Generate explanations if enabled
        if self.config.track_explanations:
            df['score_explanation'] = df.apply(
                lambda row: self._generate_explanation(row, factor_scores, applied_weights),
                axis=1
            )
        
        # Log scoring
        self._log_event({
            'action': 'composite_scoring',
            'timestamp': datetime.now(),
            'regime': regime.value,
            'avg_score': df['composite_score'].mean(),
            'score_distribution': df['signal_strength'].value_counts().to_dict()
        })
        
        return df
    
    def _generate_explanation(
        self, 
        row: pd.Series, 
        factor_scores: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation for score"""
        explanations = []
        
        # Get top contributing factors
        contributions = []
        for factor, weight in weights.items():
            if factor in factor_scores and row.name in factor_scores[factor].index:
                score = factor_scores[factor].loc[row.name]
                contribution = score * weight
                contributions.append((factor, score, contribution))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[2], reverse=True)
        
        # Build explanation
        for factor, score, contrib in contributions[:3]:  # Top 3 factors
            if score >= 80:
                explanations.append(f"Strong {factor.replace('_', ' ').title()}")
            elif score >= 60:
                explanations.append(f"Good {factor.replace('_', ' ').title()}")
            elif score <= 30:
                explanations.append(f"Weak {factor.replace('_', ' ').title()}")
        
        return " | ".join(explanations) if explanations else "Moderate across all factors"
    
    def _log_event(self, event: Dict[str, Any]):
        """Log event to audit trail"""
        with self._lock:
            self.audit_log.append(event)
    
    def get_factor_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for all factors"""
        factor_cols = [col for col in df.columns if col.endswith('_score')]
        
        if not factor_cols:
            return pd.DataFrame()
        
        summary = df[factor_cols].describe().round(2)
        summary.loc['non_null_count'] = df[factor_cols].notna().sum()
        
        return summary.T

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_signals(
    stocks_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    regime: Union[str, RegimeType] = RegimeType.BALANCED,
    config: Optional[SignalConfig] = None
) -> pd.DataFrame:
    """
    Main entry point for signal calculation
    
    Args:
        stocks_df: DataFrame with stock data
        sector_df: DataFrame with sector data (optional)
        regime: Market regime or regime name
        config: Signal configuration (optional)
        
    Returns:
        DataFrame with all scores calculated
    """
    if stocks_df.empty:
        logger.warning("Empty dataframe provided to calculate_signals")
        return stocks_df
    
    # Convert regime string to enum if needed
    if isinstance(regime, str):
        try:
            regime = RegimeType(regime.lower())
        except ValueError:
            logger.warning(f"Unknown regime '{regime}', using BALANCED")
            regime = RegimeType.BALANCED
    
    # Initialize engine
    engine = SignalEngine(config)
    
    # Calculate factor scores
    scored_df, factor_scores = engine.calculate_factor_scores(stocks_df, sector_df)
    
    # Calculate composite score
    final_df = engine.calculate_composite_score(scored_df, factor_scores, regime)
    
    # Add signal classification based on thresholds
    conditions = [
        final_df['composite_score'] >= SIGNAL_THRESHOLDS.get('BUY', 85),
        final_df['composite_score'] >= SIGNAL_THRESHOLDS.get('WATCH', 70),
        final_df['composite_score'] >= SIGNAL_THRESHOLDS.get('AVOID', 50)
    ]
    choices = ['BUY', 'WATCH', 'NEUTRAL']
    final_df['signal_category'] = np.select(conditions, choices, default='AVOID')
    
    logger.info(f"Signal calculation complete for {len(final_df)} stocks")
    logger.info(f"Average composite score: {final_df['composite_score'].mean():.2f}")
    
    return final_df

def get_top_signals(
    df: pd.DataFrame, 
    n: int = 20, 
    signal_type: Optional[str] = None
) -> pd.DataFrame:
    """Get top N stocks by composite score"""
    if 'composite_score' not in df.columns:
        logger.error("No composite_score column found")
        return pd.DataFrame()
    
    sorted_df = df.sort_values('composite_score', ascending=False)
    
    if signal_type and 'signal_category' in df.columns:
        sorted_df = sorted_df[sorted_df['signal_category'] == signal_type]
    
    return sorted_df.head(n)

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Signal Engine - Ready")
    print("="*60)
    print("\nThis module calculates multi-factor scores for stocks.")
    print("Import and use calculate_signals() function with your data.")
    print("\nExample usage:")
    print("  from signal_engine import calculate_signals")
    print("  scored_df = calculate_signals(stocks_df, sector_df)")
    print("="*60)
