"""
regime_shifter.py - Ultimate Market Regime Detection & Weight Optimization Engine

Production-grade regime detection with statistical robustness and full transparency.
"""

from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class RegimeType(Enum):
   """Market regime classifications"""
   BALANCED = "balanced"
   MOMENTUM = "momentum"
   VALUE = "value"
   GROWTH = "growth"
   VOLATILITY = "volatility"
   DEFENSIVE = "defensive"
   RECOVERY = "recovery"


@dataclass
class RegimeMetrics:
   """Metrics used for regime detection"""
   momentum_strength: float
   value_opportunity: float
   growth_prevalence: float
   volatility_level: float
   market_breadth: float
   volume_surge: float
   defensive_need: float
   recovery_potential: float


@dataclass
class RegimeDetectionResult:
   """Complete regime detection output"""
   regime: str
   confidence: float
   weights: Dict[str, float]
   metrics: RegimeMetrics
   explanation: str
   recommendations: List[str]
   timestamp: datetime


# Optimized regime weight configurations
REGIME_WEIGHTS = {
   "balanced": {
       "momentum": 0.20,
       "value": 0.20,
       "eps": 0.20,
       "volume": 0.20,
       "sector": 0.20
   },
   "momentum": {
       "momentum": 0.35,
       "value": 0.10,
       "eps": 0.15,
       "volume": 0.25,
       "sector": 0.15
   },
   "value": {
       "momentum": 0.10,
       "value": 0.40,
       "eps": 0.20,
       "volume": 0.15,
       "sector": 0.15
   },
   "growth": {
       "momentum": 0.15,
       "value": 0.10,
       "eps": 0.35,
       "volume": 0.20,
       "sector": 0.20
   },
   "volatility": {
       "momentum": 0.15,
       "value": 0.15,
       "eps": 0.15,
       "volume": 0.35,
       "sector": 0.20
   },
   "defensive": {
       "momentum": 0.10,
       "value": 0.30,
       "eps": 0.25,
       "volume": 0.10,
       "sector": 0.25
   },
   "recovery": {
       "momentum": 0.25,
       "value": 0.25,
       "eps": 0.20,
       "volume": 0.20,
       "sector": 0.10
   }
}


def get_regime_weights(regime: str = "balanced") -> Dict[str, float]:
   """
   Get weight configuration for specified regime.
   
   Args:
       regime: Regime name (string or RegimeType)
       
   Returns:
       Dictionary of factor weights
   """
   if isinstance(regime, RegimeType):
       regime = regime.value
   return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["balanced"]).copy()


def calculate_regime_metrics(df: pd.DataFrame) -> RegimeMetrics:
   """
   Calculate comprehensive market metrics for regime detection.
   
   Args:
       df: DataFrame with market data
       
   Returns:
       RegimeMetrics with calculated values
   """
   metrics = RegimeMetrics(
       momentum_strength=0.0,
       value_opportunity=0.0,
       growth_prevalence=0.0,
       volatility_level=0.0,
       market_breadth=0.0,
       volume_surge=0.0,
       defensive_need=0.0,
       recovery_potential=0.0
   )
   
   # Momentum strength (multi-timeframe)
   if "ret_1m" in df.columns and "ret_3m" in df.columns:
       short_momentum = _safe_percentile(df["ret_1m"] > 5, 0.5)
       medium_momentum = _safe_percentile(df["ret_3m"] > 10, 0.5)
       trend_consistency = _safe_correlation(df.get("ret_1m"), df.get("ret_3m"))
       metrics.momentum_strength = (short_momentum + medium_momentum + max(0, trend_consistency)) / 3
   
   # Value opportunity
   if "pe" in df.columns and "pb" in df.columns:
       pe_valid = df["pe"][(df["pe"] > 0) & (df["pe"] < 100)]
       low_pe = _safe_percentile(pe_valid < pe_valid.quantile(0.3), 0.5) if len(pe_valid) > 0 else 0
       pb_valid = df["pb"][(df["pb"] > 0) & (df["pb"] < 10)]
       low_pb = _safe_percentile(pb_valid < pb_valid.quantile(0.3), 0.5) if len(pb_valid) > 0 else 0
       metrics.value_opportunity = (low_pe + low_pb) / 2
   
   # Growth prevalence
   if "eps_change_pct" in df.columns and "revenue_growth" in df.columns:
       eps_growth = _safe_percentile(df["eps_change_pct"] > 15, 0.5)
       revenue_growth = _safe_percentile(df["revenue_growth"] > 10, 0.5)
       metrics.growth_prevalence = (eps_growth + revenue_growth) / 2
   elif "eps_change_pct" in df.columns:
       metrics.growth_prevalence = _safe_percentile(df["eps_change_pct"] > 15, 0.5)
   
   # Volatility level
   if "ret_1d" in df.columns:
       daily_vol = df["ret_1d"].std() if len(df) > 1 else 0
       extreme_moves = _safe_percentile(df["ret_1d"].abs() > 3, 0.5)
       metrics.volatility_level = min(1.0, (daily_vol / 2 + extreme_moves) / 2)
   
   # Market breadth
   if "ret_1w" in df.columns:
       advancing = _safe_percentile(df["ret_1w"] > 0, 0.5)
       strong_advance = _safe_percentile(df["ret_1w"] > 3, 0.5)
       metrics.market_breadth = (advancing + strong_advance) / 2
   
   # Volume surge
   if "vol_ratio_1d_90d" in df.columns:
       high_volume = _safe_percentile(df["vol_ratio_1d_90d"] > 2, 0.5)
       extreme_volume = _safe_percentile(df["vol_ratio_1d_90d"] > 3, 0.5)
       metrics.volume_surge = (high_volume + extreme_volume) / 2
   
   # Defensive need (market stress indicators)
   if "ret_1m" in df.columns and "rsi" in df.columns:
       declining = _safe_percentile(df["ret_1m"] < -5, 0.5)
       oversold = _safe_percentile(df["rsi"] < 30, 0.5)
       metrics.defensive_need = (declining + oversold) / 2
   
   # Recovery potential
   if "ret_3m" in df.columns and "ret_1w" in df.columns:
       beaten_down = _safe_percentile(df["ret_3m"] < -10, 0.5)
       recent_turn = _safe_percentile(df["ret_1w"] > 2, 0.5)
       metrics.recovery_potential = (beaten_down * recent_turn) ** 0.5
   
   return metrics


def auto_detect_regime(df: pd.DataFrame, min_confidence: float = 0.6) -> RegimeDetectionResult:
   """
   Intelligently detect optimal market regime using statistical analysis.
   
   Args:
       df: DataFrame with market data
       min_confidence: Minimum confidence threshold for regime selection
       
   Returns:
       Complete regime detection result with explanation
   """
   # Calculate metrics
   metrics = calculate_regime_metrics(df)
   
   # Score each regime
   regime_scores = {
       "momentum": _score_momentum_regime(metrics),
       "value": _score_value_regime(metrics),
       "growth": _score_growth_regime(metrics),
       "volatility": _score_volatility_regime(metrics),
       "defensive": _score_defensive_regime(metrics),
       "recovery": _score_recovery_regime(metrics),
       "balanced": 0.5  # Default baseline
   }
   
   # Select best regime
   best_regime = max(regime_scores.items(), key=lambda x: x[1])
   selected_regime = best_regime[0]
   confidence = best_regime[1]
   
   # Fall back to balanced if confidence too low
   if confidence < min_confidence:
       selected_regime = "balanced"
       confidence = 0.5
   
   # Get weights
   weights = get_regime_weights(selected_regime)
   
   # Generate explanation
   explanation = _generate_regime_explanation(selected_regime, metrics, confidence)
   
   # Generate recommendations
   recommendations = _generate_regime_recommendations(selected_regime, metrics)
   
   return RegimeDetectionResult(
       regime=selected_regime,
       confidence=confidence,
       weights=weights,
       metrics=metrics,
       explanation=explanation,
       recommendations=recommendations,
       timestamp=datetime.now()
   )


def get_regime_info(regime: str) -> Dict[str, Any]:
   """
   Get detailed information about a specific regime.
   
   Args:
       regime: Regime name
       
   Returns:
       Dictionary with regime details
   """
   descriptions = {
       "balanced": {
           "name": "Balanced",
           "description": "Equal focus across all factors for stable markets",
           "best_for": "Normal market conditions with no clear trend",
           "risk_level": "Medium",
           "key_factors": ["Diversified approach", "No factor bias", "Stable returns"]
       },
       "momentum": {
           "name": "Momentum",
           "description": "Focus on trending stocks with strong price action",
           "best_for": "Bull markets and strong uptrends",
           "risk_level": "High",
           "key_factors": ["Price trends", "Volume confirmation", "Relative strength"]
       },
       "value": {
           "name": "Value",
           "description": "Hunt for undervalued stocks with strong fundamentals",
           "best_for": "Market bottoms and oversold conditions",
           "risk_level": "Medium",
           "key_factors": ["Low valuations", "Strong fundamentals", "Margin of safety"]
       },
       "growth": {
           "name": "Growth",
           "description": "Target high-growth companies with expanding earnings",
           "best_for": "Economic expansion and earnings growth cycles",
           "risk_level": "Medium-High",
           "key_factors": ["EPS growth", "Revenue expansion", "Market leadership"]
       },
       "volatility": {
           "name": "Volatility",
           "description": "Capitalize on high volume and price swings",
           "best_for": "Volatile markets with trading opportunities",
           "risk_level": "Very High",
           "key_factors": ["Volume surges", "Price volatility", "Quick moves"]
       },
       "defensive": {
           "name": "Defensive",
           "description": "Preserve capital with quality and stability focus",
           "best_for": "Bear markets and uncertain conditions",
           "risk_level": "Low",
           "key_factors": ["Quality metrics", "Low volatility", "Sector safety"]
       },
       "recovery": {
           "name": "Recovery",
           "description": "Position for market recovery and mean reversion",
           "best_for": "Post-correction bounce opportunities",
           "risk_level": "Medium-High",
           "key_factors": ["Oversold bounces", "Value emergence", "Trend reversals"]
       }
   }
   
   info = descriptions.get(regime, descriptions["balanced"]).copy()
   info["weights"] = get_regime_weights(regime)
   return info


def validate_weights(weights: Dict[str, float]) -> Tuple[bool, Optional[str]]:
   """
   Validate regime weights for downstream compatibility.
   
   Args:
       weights: Dictionary of factor weights
       
   Returns:
       Tuple of (is_valid, error_message)
   """
   required_factors = {"momentum", "value", "eps", "volume", "sector"}
   
   # Check all factors present
   if not all(factor in weights for factor in required_factors):
       missing = required_factors - set(weights.keys())
       return False, f"Missing factors: {missing}"
   
   # Check weights sum to 1.0
   total = sum(weights[f] for f in required_factors)
   if abs(total - 1.0) > 0.01:
       return False, f"Weights sum to {total}, not 1.0"
   
   # Check all weights positive
   negative = [f for f in required_factors if weights[f] < 0]
   if negative:
       return False, f"Negative weights: {negative}"
   
   return True, None


# Helper functions

def _safe_percentile(series: pd.Series, default: float = 0.0) -> float:
   """Calculate percentage of True values safely."""
   try:
       if len(series) == 0:
           return default
       return float(series.sum() / len(series))
   except:
       return default


def _safe_correlation(s1: pd.Series, s2: pd.Series) -> float:
   """Calculate correlation safely."""
   try:
       if s1 is None or s2 is None or len(s1) < 2:
           return 0.0
       return float(s1.corr(s2))
   except:
       return 0.0


def _score_momentum_regime(metrics: RegimeMetrics) -> float:
   """Score momentum regime suitability."""
   return (
       metrics.momentum_strength * 0.4 +
       metrics.market_breadth * 0.3 +
       metrics.volume_surge * 0.2 +
       (1 - metrics.defensive_need) * 0.1
   )


def _score_value_regime(metrics: RegimeMetrics) -> float:
   """Score value regime suitability."""
   return (
       metrics.value_opportunity * 0.4 +
       metrics.defensive_need * 0.2 +
       (1 - metrics.momentum_strength) * 0.2 +
       (1 - metrics.volatility_level) * 0.2
   )


def _score_growth_regime(metrics: RegimeMetrics) -> float:
   """Score growth regime suitability."""
   return (
       metrics.growth_prevalence * 0.5 +
       metrics.momentum_strength * 0.2 +
       metrics.market_breadth * 0.2 +
       (1 - metrics.defensive_need) * 0.1
   )


def _score_volatility_regime(metrics: RegimeMetrics) -> float:
   """Score volatility regime suitability."""
   return (
       metrics.volatility_level * 0.4 +
       metrics.volume_surge * 0.4 +
       (1 - metrics.market_breadth + 1) / 2 * 0.2  # Works in choppy markets
   )


def _score_defensive_regime(metrics: RegimeMetrics) -> float:
   """Score defensive regime suitability."""
   return (
       metrics.defensive_need * 0.4 +
       metrics.value_opportunity * 0.3 +
       (1 - metrics.volatility_level) * 0.2 +
       (1 - metrics.momentum_strength) * 0.1
   )


def _score_recovery_regime(metrics: RegimeMetrics) -> float:
   """Score recovery regime suitability."""
   return (
       metrics.recovery_potential * 0.4 +
       metrics.value_opportunity * 0.3 +
       metrics.momentum_strength * 0.2 +
       (1 - metrics.defensive_need) * 0.1
   )


def _generate_regime_explanation(regime: str, metrics: RegimeMetrics, confidence: float) -> str:
   """Generate human-readable explanation for regime selection."""
   explanations = {
       "momentum": f"Strong momentum detected with {metrics.momentum_strength:.0%} of stocks trending up and {metrics.market_breadth:.0%} market breadth.",
       "value": f"Value opportunities present with {metrics.value_opportunity:.0%} of stocks showing attractive valuations.",
       "growth": f"Growth environment with {metrics.growth_prevalence:.0%} of companies showing strong earnings expansion.",
       "volatility": f"High volatility regime with {metrics.volatility_level:.0%} volatility and {metrics.volume_surge:.0%} volume surge activity.",
       "defensive": f"Defensive positioning recommended with {metrics.defensive_need:.0%} market stress indicators active.",
       "recovery": f"Recovery setup detected with {metrics.recovery_potential:.0%} of stocks showing reversal potential.",
       "balanced": "No dominant market theme detected, balanced approach recommended."
   }
   
   base = explanations.get(regime, "Regime selected based on current market conditions.")
   return f"{base} Confidence: {confidence:.0%}"


def _generate_regime_recommendations(regime: str, metrics: RegimeMetrics) -> List[str]:
   """Generate actionable recommendations for current regime."""
   recommendations = []
   
   if regime == "momentum":
       recommendations.append("Focus on stocks with strong relative strength")
       recommendations.append("Use trailing stops to protect profits")
       if metrics.volatility_level > 0.7:
           recommendations.append("Consider reducing position sizes due to high volatility")
   
   elif regime == "value":
       recommendations.append("Screen for low PE/PB ratios with strong fundamentals")
       recommendations.append("Be patient - value investing requires time")
       if metrics.recovery_potential > 0.5:
           recommendations.append("Look for turnaround stories in beaten-down sectors")
   
   elif regime == "growth":
       recommendations.append("Prioritize earnings growth and revenue expansion")
       recommendations.append("Monitor valuations to avoid overpaying")
   
   elif regime == "volatility":
       recommendations.append("Consider shorter holding periods")
       recommendations.append("Focus on liquid stocks with tight spreads")
       recommendations.append("Use strict risk management")
   
   elif regime == "defensive":
       recommendations.append("Prioritize capital preservation over returns")
       recommendations.append("Focus on quality companies with strong balance sheets")
       recommendations.append("Consider defensive sectors")
   
   elif regime == "recovery":
       recommendations.append("Look for oversold quality stocks")
       recommendations.append("Monitor for trend reversal confirmations")
       recommendations.append("Scale into positions gradually")
   
   else:  # balanced
       recommendations.append("Maintain diversified exposure across factors")
       recommendations.append("No specific factor bias recommended")
   
   return recommendations


# Streamlit UI helper functions

def render_regime_selector(current_regime: str = "balanced") -> str:
   """
   Render regime selector for Streamlit UI.
   Returns selected regime.
   """
   import streamlit as st
   
   regimes = list(REGIME_WEIGHTS.keys())
   regime_names = [get_regime_info(r)["name"] for r in regimes]
   
   selected_idx = regimes.index(current_regime) if current_regime in regimes else 0
   
   selected_name = st.selectbox(
       "Market Regime",
       regime_names,
       index=selected_idx,
       help="Select market regime manually or use auto-detect"
   )
   
   return regimes[regime_names.index(selected_name)]


def render_regime_info(regime_result: RegimeDetectionResult) -> None:
   """Render regime information in Streamlit UI."""
   import streamlit as st
   
   # Display current regime
   info = get_regime_info(regime_result.regime)
   
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("Regime", info["name"])
   with col2:
       st.metric("Confidence", f"{regime_result.confidence:.0%}")
   with col3:
       st.metric("Risk Level", info["risk_level"])
   
   # Explanation
   st.info(regime_result.explanation)
   
   # Recommendations
   if regime_result.recommendations:
       with st.expander("📊 Recommendations"):
           for rec in regime_result.recommendations:
               st.write(f"• {rec}")
   
   # Weights visualization
   with st.expander("⚖️ Factor Weights"):
       weights_df = pd.DataFrame(
           list(regime_result.weights.items()),
           columns=["Factor", "Weight"]
       )
       st.dataframe(weights_df.style.format({"Weight": "{:.0%}"}))
