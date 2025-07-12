"""
regime_shifter.py - M.A.N.T.R.A. Market Regime Detection
=======================================================
Identifies market conditions and adjusts strategies accordingly
Provides dynamic weighting and recommendations based on regime
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import from constants
from constants import MARKET_REGIMES, FACTOR_WEIGHTS

logger = logging.getLogger(__name__)

# ============================================================================
# REGIME TYPES
# ============================================================================

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    MOMENTUM = "momentum"
    VALUE = "value"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    SECTOR_ROTATION = "sector_rotation"
    TRANSITIONAL = "transitional"

@dataclass
class RegimeIndicators:
    """Indicators used for regime detection"""
    breadth: float  # % of advancing stocks
    average_return: float  # Market average return
    volatility: float  # Market volatility
    volume_ratio: float  # Current vs average volume
    sector_dispersion: float  # Sector return dispersion
    momentum_strength: float  # Overall momentum
    value_spread: float  # Value vs growth spread
    risk_appetite: float  # Risk-on vs risk-off score

@dataclass
class RegimeProfile:
    """Profile of a market regime"""
    name: str
    regime_type: MarketRegime
    description: str
    characteristics: List[str]
    recommended_strategies: List[str]
    factor_weights: Dict[str, float]
    risk_level: str
    typical_duration: str

# ============================================================================
# REGIME PROFILES
# ============================================================================

REGIME_PROFILES = {
    MarketRegime.BULL_MARKET: RegimeProfile(
        name="Bull Market",
        regime_type=MarketRegime.BULL_MARKET,
        description="Strong uptrend with broad participation",
        characteristics=[
            "70%+ stocks advancing",
            "Rising average returns",
            "Expanding volume",
            "Low volatility",
            "Risk-on sentiment"
        ],
        recommended_strategies=[
            "Buy the dips",
            "Focus on momentum leaders",
            "Increase position sizes",
            "Use leveraged positions"
        ],
        factor_weights={
            "momentum": 0.40,
            "technical": 0.25,
            "volume": 0.15,
            "value": 0.10,
            "fundamentals": 0.10
        },
        risk_level="Low",
        typical_duration="6-18 months"
    ),
    
    MarketRegime.BEAR_MARKET: RegimeProfile(
        name="Bear Market",
        regime_type=MarketRegime.BEAR_MARKET,
        description="Sustained downtrend with fear dominant",
        characteristics=[
            "<30% stocks advancing",
            "Declining average returns",
            "High volatility",
            "Panic selling episodes",
            "Risk-off sentiment"
        ],
        recommended_strategies=[
            "Preserve capital",
            "Focus on quality defensive stocks",
            "Build cash reserves",
            "Look for oversold bounces"
        ],
        factor_weights={
            "value": 0.35,
            "fundamentals": 0.30,
            "technical": 0.15,
            "momentum": 0.10,
            "volume": 0.10
        },
        risk_level="High",
        typical_duration="3-12 months"
    ),
    
    MarketRegime.SIDEWAYS: RegimeProfile(
        name="Sideways/Range-bound",
        regime_type=MarketRegime.SIDEWAYS,
        description="Trendless market with range trading",
        characteristics=[
            "45-55% stocks advancing",
            "Near-zero average returns",
            "Low volatility",
            "Sector rotation active",
            "Mixed signals"
        ],
        recommended_strategies=[
            "Trade the range",
            "Focus on stock picking",
            "Sell options premium",
            "Sector rotation plays"
        ],
        factor_weights={
            "value": 0.25,
            "technical": 0.25,
            "fundamentals": 0.20,
            "momentum": 0.15,
            "volume": 0.15
        },
        risk_level="Medium",
        typical_duration="3-9 months"
    ),
    
    MarketRegime.VOLATILE: RegimeProfile(
        name="High Volatility",
        regime_type=MarketRegime.VOLATILE,
        description="Extreme price swings and uncertainty",
        characteristics=[
            "Large daily swings",
            "High VIX equivalent",
            "Gap opens common",
            "News-driven moves",
            "Emotional extremes"
        ],
        recommended_strategies=[
            "Reduce position sizes",
            "Use tight stops",
            "Trade volatility",
            "Focus on quality"
        ],
        factor_weights={
            "technical": 0.30,
            "volume": 0.25,
            "momentum": 0.20,
            "value": 0.15,
            "fundamentals": 0.10
        },
        risk_level="Very High",
        typical_duration="1-6 months"
    ),
    
    MarketRegime.MOMENTUM: RegimeProfile(
        name="Momentum Regime",
        regime_type=MarketRegime.MOMENTUM,
        description="Trend-following environment",
        characteristics=[
            "Strong trends persist",
            "Winners keep winning",
            "Clear sector leaders",
            "FOMO psychology",
            "Low volatility trends"
        ],
        recommended_strategies=[
            "Ride the trends",
            "Buy strength",
            "Use trailing stops",
            "Avoid value traps"
        ],
        factor_weights={
            "momentum": 0.45,
            "technical": 0.25,
            "volume": 0.15,
            "fundamentals": 0.10,
            "value": 0.05
        },
        risk_level="Medium-Low",
        typical_duration="6-12 months"
    ),
    
    MarketRegime.VALUE: RegimeProfile(
        name="Value Regime",
        regime_type=MarketRegime.VALUE,
        description="Fundamentals and value matter",
        characteristics=[
            "Rotation to quality",
            "P/E compression",
            "Dividend focus",
            "Skeptical sentiment",
            "Selective buying"
        ],
        recommended_strategies=[
            "Buy undervalued quality",
            "Focus on fundamentals",
            "Dividend aristocrats",
            "Contrarian plays"
        ],
        factor_weights={
            "value": 0.40,
            "fundamentals": 0.30,
            "technical": 0.15,
            "momentum": 0.10,
            "volume": 0.05
        },
        risk_level="Medium",
        typical_duration="6-18 months"
    )
}

# ============================================================================
# REGIME DETECTION ENGINE
# ============================================================================

class RegimeShifter:
    """
    Detects market regime and provides adaptive strategies
    """
    
    def __init__(self):
        self.current_regime = None
        self.regime_confidence = 0
        self.regime_history = []
        self.indicators = None
        
    def detect_regime(
        self,
        stocks_df: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None,
        lookback_days: int = 30
    ) -> Tuple[MarketRegime, float, Dict]:
        """
        Detect current market regime
        
        Args:
            stocks_df: Stock market data
            sector_df: Sector performance data
            lookback_days: Days to analyze
            
        Returns:
            Tuple of (regime, confidence, analysis)
        """
        logger.info("Detecting market regime...")
        
        # Calculate regime indicators
        self.indicators = self._calculate_indicators(stocks_df, sector_df, lookback_days)
        
        # Score each regime
        regime_scores = self._score_regimes(self.indicators)
        
        # Determine primary regime
        self.current_regime, self.regime_confidence = self._determine_regime(regime_scores)
        
        # Generate analysis
        analysis = self._generate_analysis(regime_scores)
        
        # Update history
        self._update_history(self.current_regime, self.regime_confidence)
        
        logger.info(f"Detected regime: {self.current_regime.value} (confidence: {self.regime_confidence:.1f}%)")
        
        return self.current_regime, self.regime_confidence, analysis
    
    # ========================================================================
    # INDICATOR CALCULATION
    # ========================================================================
    
    def _calculate_indicators(
        self,
        stocks_df: pd.DataFrame,
        sector_df: Optional[pd.DataFrame],
        lookback_days: int
    ) -> RegimeIndicators:
        """Calculate all regime indicators"""
        
        # Market breadth
        if 'ret_30d' in stocks_df.columns:
            breadth = (stocks_df['ret_30d'] > 0).sum() / len(stocks_df) * 100
            average_return = stocks_df['ret_30d'].mean()
        else:
            breadth = 50.0
            average_return = 0.0
        
        # Volatility (using return std as proxy)
        return_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in return_cols if col in stocks_df.columns]
        if available_cols:
            volatility = stocks_df[available_cols].std().mean()
        else:
            volatility = 10.0
        
        # Volume ratio
        if 'rvol' in stocks_df.columns:
            volume_ratio = stocks_df['rvol'].mean()
        else:
            volume_ratio = 1.0
        
        # Sector dispersion
        if sector_df is not None and 'sector_ret_30d' in sector_df.columns:
            sector_dispersion = sector_df['sector_ret_30d'].std()
        else:
            sector_dispersion = 5.0
        
        # Momentum strength
        if 'momentum_score' in stocks_df.columns:
            momentum_strength = stocks_df['momentum_score'].mean()
        else:
            momentum_strength = 50.0
        
        # Value spread (simplified)
        if 'pe' in stocks_df.columns:
            value_stocks = stocks_df[stocks_df['pe'].between(5, 20)]
            growth_stocks = stocks_df[stocks_df['pe'] > 30]
            
            if len(value_stocks) > 0 and len(growth_stocks) > 0:
                value_return = value_stocks['ret_30d'].mean() if 'ret_30d' in value_stocks.columns else 0
                growth_return = growth_stocks['ret_30d'].mean() if 'ret_30d' in growth_stocks.columns else 0
                value_spread = value_return - growth_return
            else:
                value_spread = 0.0
        else:
            value_spread = 0.0
        
        # Risk appetite (simplified)
        if all(col in stocks_df.columns for col in ['market_cap', 'ret_30d']):
            large_caps = stocks_df[stocks_df['market_cap'] > 20000]
            small_caps = stocks_df[stocks_df['market_cap'] < 5000]
            
            if len(large_caps) > 0 and len(small_caps) > 0:
                large_cap_return = large_caps['ret_30d'].mean()
                small_cap_return = small_caps['ret_30d'].mean()
                risk_appetite = 50 + (small_cap_return - large_cap_return)
            else:
                risk_appetite = 50.0
        else:
            risk_appetite = 50.0
        
        return RegimeIndicators(
            breadth=breadth,
            average_return=average_return,
            volatility=volatility,
            volume_ratio=volume_ratio,
            sector_dispersion=sector_dispersion,
            momentum_strength=momentum_strength,
            value_spread=value_spread,
            risk_appetite=risk_appetite
        )
    
    # ========================================================================
    # REGIME SCORING
    # ========================================================================
    
    def _score_regimes(self, indicators: RegimeIndicators) -> Dict[MarketRegime, float]:
        """Score each regime based on indicators"""
        scores = {}
        
        # Bull Market Score
        bull_score = 0
        if indicators.breadth > 70:
            bull_score += 30
        if indicators.average_return > 5:
            bull_score += 25
        if indicators.volatility < 15:
            bull_score += 20
        if indicators.volume_ratio > 1.2:
            bull_score += 15
        if indicators.risk_appetite > 60:
            bull_score += 10
        scores[MarketRegime.BULL_MARKET] = bull_score
        
        # Bear Market Score
        bear_score = 0
        if indicators.breadth < 30:
            bear_score += 30
        if indicators.average_return < -5:
            bear_score += 25
        if indicators.volatility > 20:
            bear_score += 20
        if indicators.risk_appetite < 40:
            bear_score += 15
        if indicators.value_spread > 5:
            bear_score += 10
        scores[MarketRegime.BEAR_MARKET] = bear_score
        
        # Sideways Score
        sideways_score = 0
        if 45 <= indicators.breadth <= 55:
            sideways_score += 30
        if -2 <= indicators.average_return <= 2:
            sideways_score += 25
        if indicators.volatility < 12:
            sideways_score += 20
        if indicators.sector_dispersion > 8:
            sideways_score += 15
        if 0.8 <= indicators.volume_ratio <= 1.2:
            sideways_score += 10
        scores[MarketRegime.SIDEWAYS] = sideways_score
        
        # Volatile Score
        volatile_score = 0
        if indicators.volatility > 25:
            volatile_score += 40
        if indicators.sector_dispersion > 15:
            volatile_score += 20
        if abs(indicators.average_return) > 10:
            volatile_score += 20
        if indicators.volume_ratio > 2:
            volatile_score += 20
        scores[MarketRegime.VOLATILE] = volatile_score
        
        # Momentum Score
        momentum_score = 0
        if indicators.momentum_strength > 70:
            momentum_score += 35
        if indicators.breadth > 60:
            momentum_score += 25
        if indicators.average_return > 3:
            momentum_score += 20
        if indicators.value_spread < -5:
            momentum_score += 20
        scores[MarketRegime.MOMENTUM] = momentum_score
        
        # Value Score
        value_score = 0
        if indicators.value_spread > 10:
            value_score += 35
        if indicators.momentum_strength < 40:
            value_score += 25
        if indicators.risk_appetite < 45:
            value_score += 20
        if indicators.volatility > 15:
            value_score += 20
        scores[MarketRegime.VALUE] = value_score
        
        return scores
    
    def _determine_regime(
        self,
        regime_scores: Dict[MarketRegime, float]
    ) -> Tuple[MarketRegime, float]:
        """Determine primary regime and confidence"""
        # Sort regimes by score
        sorted_regimes = sorted(
            regime_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if not sorted_regimes:
            return MarketRegime.TRANSITIONAL, 0.0
        
        # Get top regime
        top_regime, top_score = sorted_regimes[0]
        
        # Calculate confidence
        if len(sorted_regimes) > 1:
            second_score = sorted_regimes[1][1]
            confidence = min(100, (top_score - second_score) / max(top_score, 1) * 100)
        else:
            confidence = min(100, top_score)
        
        # If confidence is too low, it's transitional
        if confidence < 30:
            return MarketRegime.TRANSITIONAL, confidence
        
        return top_regime, confidence
    
    # ========================================================================
    # ANALYSIS GENERATION
    # ========================================================================
    
    def _generate_analysis(self, regime_scores: Dict[MarketRegime, float]) -> Dict:
        """Generate comprehensive regime analysis"""
        analysis = {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'indicators': {
                'breadth': f"{self.indicators.breadth:.1f}%",
                'average_return': f"{self.indicators.average_return:.1f}%",
                'volatility': f"{self.indicators.volatility:.1f}",
                'volume_ratio': f"{self.indicators.volume_ratio:.2f}x",
                'sector_dispersion': f"{self.indicators.sector_dispersion:.1f}",
                'momentum_strength': f"{self.indicators.momentum_strength:.1f}",
                'value_spread': f"{self.indicators.value_spread:.1f}%",
                'risk_appetite': f"{self.indicators.risk_appetite:.1f}"
            },
            'regime_scores': {
                regime.value: score
                for regime, score in regime_scores.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add regime-specific details
        if self.current_regime in REGIME_PROFILES:
            profile = REGIME_PROFILES[self.current_regime]
            analysis.update({
                'description': profile.description,
                'characteristics': profile.characteristics,
                'recommended_strategies': profile.recommended_strategies,
                'factor_weights': profile.factor_weights,
                'risk_level': profile.risk_level,
                'typical_duration': profile.typical_duration
            })
        
        return analysis
    
    def _update_history(self, regime: MarketRegime, confidence: float):
        """Update regime history"""
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'confidence': confidence
        })
        
        # Keep only last 100 entries
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    # ========================================================================
    # ADAPTIVE STRATEGIES
    # ========================================================================
    
    def get_adapted_weights(self) -> Dict[str, float]:
        """Get factor weights adapted to current regime"""
        if self.current_regime is None:
            return FACTOR_WEIGHTS
        
        if self.current_regime in REGIME_PROFILES:
            return REGIME_PROFILES[self.current_regime].factor_weights
        
        return FACTOR_WEIGHTS
    
    def get_regime_recommendations(self) -> Dict[str, Any]:
        """Get specific recommendations for current regime"""
        if self.current_regime is None:
            return {}
        
        recommendations = {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'position_sizing': self._get_position_sizing(),
            'sector_focus': self._get_sector_focus(),
            'risk_management': self._get_risk_management(),
            'opportunities': self._get_opportunities()
        }
        
        return recommendations
    
    def _get_position_sizing(self) -> Dict[str, Any]:
        """Get position sizing recommendations"""
        if self.current_regime == MarketRegime.BULL_MARKET:
            return {
                'recommendation': 'Increase position sizes',
                'max_position': '10%',
                'cash_level': '5-10%',
                'leverage': 'Moderate leverage acceptable'
            }
        elif self.current_regime == MarketRegime.BEAR_MARKET:
            return {
                'recommendation': 'Reduce position sizes',
                'max_position': '5%',
                'cash_level': '30-50%',
                'leverage': 'Avoid leverage'
            }
        elif self.current_regime == MarketRegime.VOLATILE:
            return {
                'recommendation': 'Small positions only',
                'max_position': '3%',
                'cash_level': '20-30%',
                'leverage': 'No leverage'
            }
        else:
            return {
                'recommendation': 'Normal position sizes',
                'max_position': '7%',
                'cash_level': '10-20%',
                'leverage': 'Minimal leverage'
            }
    
    def _get_sector_focus(self) -> List[str]:
        """Get recommended sectors for regime"""
        if self.current_regime == MarketRegime.BULL_MARKET:
            return ['Technology', 'Consumer Discretionary', 'Financials']
        elif self.current_regime == MarketRegime.BEAR_MARKET:
            return ['Consumer Staples', 'Healthcare', 'Utilities']
        elif self.current_regime == MarketRegime.VALUE:
            return ['Financials', 'Energy', 'Industrials']
        elif self.current_regime == MarketRegime.MOMENTUM:
            return ['Technology', 'Communications', 'Consumer Discretionary']
        else:
            return ['Diversified across sectors']
    
    def _get_risk_management(self) -> Dict[str, str]:
        """Get risk management guidelines"""
        if self.current_regime == MarketRegime.VOLATILE:
            return {
                'stop_loss': 'Tight stops (3-5%)',
                'volatility_adjustment': 'Reduce position size by 50%',
                'hedging': 'Consider protective puts'
            }
        elif self.current_regime == MarketRegime.BEAR_MARKET:
            return {
                'stop_loss': 'Strict stops (5-7%)',
                'volatility_adjustment': 'Focus on low-beta stocks',
                'hedging': 'Maintain defensive positions'
            }
        else:
            return {
                'stop_loss': 'Normal stops (7-10%)',
                'volatility_adjustment': 'Standard position sizing',
                'hedging': 'Optional hedging'
            }
    
    def _get_opportunities(self) -> List[str]:
        """Get regime-specific opportunities"""
        opportunities = {
            MarketRegime.BULL_MARKET: [
                'Buy pullbacks in leaders',
                'Momentum breakouts',
                'Growth stocks',
                'Small cap outperformance'
            ],
            MarketRegime.BEAR_MARKET: [
                'Quality dividend stocks',
                'Oversold bounces',
                'Defensive sectors',
                'Cash generation'
            ],
            MarketRegime.SIDEWAYS: [
                'Range trading',
                'Sector rotation',
                'Mean reversion',
                'Options strategies'
            ],
            MarketRegime.VOLATILE: [
                'Volatility trades',
                'Quick scalps',
                'Quality flight',
                'Cash preservation'
            ],
            MarketRegime.MOMENTUM: [
                'Trend following',
                'Strength buying',
                'Sector leaders',
                'Growth stories'
            ],
            MarketRegime.VALUE: [
                'Deep value plays',
                'Turnaround stories',
                'Dividend aristocrats',
                'Contrarian bets'
            ]
        }
        
        return opportunities.get(self.current_regime, ['General stock picking'])

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_market_regime(
    stocks_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None
) -> Tuple[MarketRegime, float, Dict]:
    """
    Detect current market regime
    
    Args:
        stocks_df: Stock market data
        sector_df: Sector data
        
    Returns:
        Tuple of (regime, confidence, analysis)
    """
    shifter = RegimeShifter()
    return shifter.detect_regime(stocks_df, sector_df)

def get_regime_profile(regime: MarketRegime) -> RegimeProfile:
    """Get profile for a specific regime"""
    return REGIME_PROFILES.get(regime)

def adapt_strategy_to_regime(
    stocks_df: pd.DataFrame,
    current_regime: MarketRegime
) -> pd.DataFrame:
    """
    Adapt stock scores based on regime
    
    Args:
        stocks_df: Stock data with scores
        current_regime: Current market regime
        
    Returns:
        DataFrame with regime-adjusted scores
    """
    if current_regime not in REGIME_PROFILES:
        return stocks_df
    
    df = stocks_df.copy()
    weights = REGIME_PROFILES[current_regime].factor_weights
    
    # Recalculate composite score with regime weights
    score_components = []
    weight_sum = 0
    
    for factor, weight in weights.items():
        col_name = f"{factor}_score"
        if col_name in df.columns:
            score_components.append(df[col_name] * weight)
            weight_sum += weight
    
    if score_components and weight_sum > 0:
        df['regime_adjusted_score'] = sum(score_components) / weight_sum
        df['scoring_regime'] = current_regime.value
    
    return df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Regime Shifter")
    print("="*60)
    print("\nDetects market regimes and adapts strategies")
    print("\nMarket Regimes:")
    for regime in MarketRegime:
        print(f"  - {regime.value}")
    print("\nRegime Profiles:")
    for regime, profile in REGIME_PROFILES.items():
        print(f"\n{profile.name}:")
        print(f"  Risk Level: {profile.risk_level}")
        print(f"  Duration: {profile.typical_duration}")
    print("\nUse detect_market_regime() to analyze market")
    print("="*60)
