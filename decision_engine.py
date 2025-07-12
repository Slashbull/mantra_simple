"""
decision_engine.py - M.A.N.T.R.A. Decision Engine
================================================
Makes final Buy/Watch/Avoid decisions with clear reasoning
Calculates targets, risks, and provides actionable insights
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Import from other modules
from constants import (
    SIGNAL_LEVELS, RISK_FACTORS, PE_RANGES, 
    MOMENTUM_LEVELS, VOLUME_LEVELS, PRICE_POSITION
)

logger = logging.getLogger(__name__)

# ============================================================================
# DECISION CRITERIA
# ============================================================================

@dataclass
class DecisionCriteria:
    """Criteria for making trading decisions"""
    
    # Minimum scores for each decision
    min_score_buy: float = 80
    min_score_watch: float = 65
    
    # Risk limits
    max_risk_buy: float = 60
    max_risk_watch: float = 80
    
    # Volume requirements
    min_volume_buy: int = 50000
    min_volume_watch: int = 10000
    
    # Target calculations
    conservative_target_pct: float = 10
    moderate_target_pct: float = 20
    aggressive_target_pct: float = 30
    
    # Stop loss levels
    tight_stop_pct: float = 5
    normal_stop_pct: float = 8
    wide_stop_pct: float = 12

# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """
    Makes trading decisions based on signals and risk assessment
    """
    
    def __init__(self, criteria: Optional[DecisionCriteria] = None):
        self.criteria = criteria or DecisionCriteria()
        self.decisions_made = 0
        
    def make_decisions(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make Buy/Watch/Avoid decisions for all stocks
        
        Args:
            scored_df: DataFrame with signal scores
            
        Returns:
            DataFrame with decisions, targets, and reasoning
        """
        if scored_df.empty:
            logger.warning("Empty dataframe provided to decision engine")
            return scored_df
            
        df = scored_df.copy()
        
        # Calculate risk scores
        logger.info("Calculating risk scores...")
        df['risk_score'] = df.apply(self._calculate_risk_score, axis=1)
        df['risk_level'] = df['risk_score'].apply(self._get_risk_level)
        
        # Make decisions
        logger.info("Making trading decisions...")
        decisions = df.apply(self._make_decision, axis=1)
        df['decision'] = decisions['decision']
        df['confidence'] = decisions['confidence']
        
        # Calculate targets and stops
        logger.info("Calculating targets and stop losses...")
        targets = df.apply(self._calculate_targets, axis=1)
        df['target_price'] = targets['target']
        df['stop_loss'] = targets['stop_loss']
        df['risk_reward_ratio'] = targets['risk_reward']
        
        # Generate reasoning
        logger.info("Generating decision reasoning...")
        df['reasoning'] = df.apply(self._generate_reasoning, axis=1)
        
        # Add opportunity score
        df['opportunity_score'] = self._calculate_opportunity_score(df)
        
        # Add decision timestamp
        df['decision_time'] = datetime.now()
        
        self.decisions_made = len(df)
        logger.info(f"Decisions made for {self.decisions_made} stocks")
        
        # Log decision summary
        decision_counts = df['decision'].value_counts()
        logger.info(f"Summary - Buy: {decision_counts.get('BUY', 0)}, "
                   f"Watch: {decision_counts.get('WATCH', 0)}, "
                   f"Avoid: {decision_counts.get('AVOID', 0)}")
        
        return df
    
    # ========================================================================
    # RISK CALCULATION
    # ========================================================================
    
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score for a stock (0-100)"""
        risk_score = 0
        
        # PE Risk
        if 'pe' in row and pd.notna(row['pe']):
            if row['pe'] > 40:
                risk_score += RISK_FACTORS['high_pe']
            elif row['pe'] < 0:
                risk_score += RISK_FACTORS['negative_eps']
        
        # Volume Risk
        if 'volume_1d' in row and pd.notna(row['volume_1d']):
            if row['volume_1d'] < 50000:
                risk_score += RISK_FACTORS['low_volume']
        
        # Price Risk
        if 'price' in row and pd.notna(row['price']):
            if row['price'] < 50:
                risk_score += RISK_FACTORS['penny_stock']
        
        # Position Risk
        if 'position_52w' in row and pd.notna(row['position_52w']):
            if row['position_52w'] < 10:
                risk_score += RISK_FACTORS['near_52w_low']
        
        # Volatility Risk (using return variations as proxy)
        return_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        if all(col in row for col in return_cols):
            returns = [row[col] for col in return_cols if pd.notna(row[col])]
            if returns:
                volatility = np.std(returns)
                if volatility > 10:
                    risk_score += RISK_FACTORS['high_volatility']
        
        return min(risk_score, 100)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 20:
            return "Very Low"
        elif risk_score < 40:
            return "Low"
        elif risk_score < 60:
            return "Moderate"
        elif risk_score < 80:
            return "High"
        else:
            return "Very High"
    
    # ========================================================================
    # DECISION MAKING
    # ========================================================================
    
    def _make_decision(self, row: pd.Series) -> pd.Series:
        """Make trading decision for a single stock"""
        decision = "AVOID"
        confidence = 0
        
        # Get composite score
        score = row.get('composite_score', 50)
        risk = row.get('risk_score', 50)
        volume = row.get('volume_1d', 0)
        
        # Decision logic
        if score >= self.criteria.min_score_buy and risk <= self.criteria.max_risk_buy:
            if volume >= self.criteria.min_volume_buy:
                decision = "BUY"
                confidence = min(95, score * (100 - risk) / 100)
            else:
                decision = "WATCH"  # Good score but low volume
                confidence = 70
                
        elif score >= self.criteria.min_score_watch and risk <= self.criteria.max_risk_watch:
            if volume >= self.criteria.min_volume_watch:
                decision = "WATCH"
                confidence = min(80, score * (100 - risk) / 100)
            else:
                decision = "AVOID"  # Okay score but too low volume
                confidence = 40
        
        # Special conditions that override
        if self._has_red_flags(row):
            decision = "AVOID"
            confidence = min(confidence, 30)
        
        if self._has_green_flags(row):
            if decision == "WATCH":
                decision = "BUY"
                confidence = min(95, confidence + 10)
            elif decision == "AVOID" and score > 60:
                decision = "WATCH"
                confidence = 60
        
        return pd.Series({
            'decision': decision,
            'confidence': round(confidence, 1)
        })
    
    def _has_red_flags(self, row: pd.Series) -> bool:
        """Check for red flags that should prevent buying"""
        red_flags = []
        
        # Consistent negative momentum
        if all(row.get(f'ret_{period}', 0) < -5 for period in ['7d', '30d', '3m']):
            red_flags.append("Consistent downtrend")
        
        # Breaking down from SMAs
        if row.get('trading_under', '') == '200 Day Avg':
            red_flags.append("Below 200 DMA")
        
        # Very high PE
        if row.get('pe', 0) > 60:
            red_flags.append("Extremely high PE")
        
        # Declining EPS
        if row.get('eps_change_pct', 0) < -30:
            red_flags.append("Sharp EPS decline")
        
        return len(red_flags) > 0
    
    def _has_green_flags(self, row: pd.Series) -> bool:
        """Check for green flags that support buying"""
        green_flags = []
        
        # Strong momentum with volume
        if row.get('ret_7d', 0) > 5 and row.get('rvol', 1) > 2:
            green_flags.append("Momentum breakout")
        
        # Value play
        if 0 < row.get('pe', 100) < 15 and row.get('eps_change_pct', 0) > 20:
            green_flags.append("Undervalued growth")
        
        # Technical breakout
        if row.get('position_52w', 50) > 80 and row.get('ret_30d', 0) > 10:
            green_flags.append("52-week high breakout")
        
        # Oversold bounce
        if row.get('position_52w', 50) < 20 and row.get('ret_7d', 0) > 5:
            green_flags.append("Oversold reversal")
        
        return len(green_flags) > 0
    
    # ========================================================================
    # TARGET & STOP LOSS
    # ========================================================================
    
    def _calculate_targets(self, row: pd.Series) -> pd.Series:
        """Calculate target price and stop loss"""
        current_price = row.get('price', 0)
        if current_price <= 0:
            return pd.Series({
                'target': 0,
                'stop_loss': 0,
                'risk_reward': 0
            })
        
        # Determine target based on score and momentum
        score = row.get('composite_score', 50)
        momentum = row.get('momentum_score', 50)
        
        # Target calculation
        if score >= 85 and momentum >= 70:
            target_pct = self.criteria.aggressive_target_pct
        elif score >= 75:
            target_pct = self.criteria.moderate_target_pct
        else:
            target_pct = self.criteria.conservative_target_pct
        
        # Adjust target based on volatility
        if row.get('risk_score', 0) > 60:
            target_pct *= 1.5  # Higher risk = higher target needed
        
        target_price = current_price * (1 + target_pct / 100)
        
        # Stop loss calculation
        risk_level = row.get('risk_level', 'Moderate')
        if risk_level in ['Very Low', 'Low']:
            stop_pct = self.criteria.tight_stop_pct
        elif risk_level == 'Moderate':
            stop_pct = self.criteria.normal_stop_pct
        else:
            stop_pct = self.criteria.wide_stop_pct
        
        stop_loss = current_price * (1 - stop_pct / 100)
        
        # Risk-reward ratio
        potential_gain = target_price - current_price
        potential_loss = current_price - stop_loss
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        return pd.Series({
            'target': round(target_price, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward': round(risk_reward, 2)
        })
    
    # ========================================================================
    # REASONING GENERATION
    # ========================================================================
    
    def _generate_reasoning(self, row: pd.Series) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []
        decision = row.get('decision', 'AVOID')
        
        # Score-based reasoning
        score = row.get('composite_score', 50)
        if score >= 85:
            reasons.append("Excellent composite score")
        elif score >= 75:
            reasons.append("Strong composite score")
        elif score >= 65:
            reasons.append("Good composite score")
        elif score < 50:
            reasons.append("Weak composite score")
        
        # Factor-specific reasoning
        if row.get('momentum_score', 0) > 80:
            reasons.append("Strong momentum")
        elif row.get('momentum_score', 0) < 30:
            reasons.append("Poor momentum")
        
        if row.get('value_score', 0) > 80:
            reasons.append("Attractive valuation")
        
        if row.get('technical_score', 0) > 80:
            reasons.append("Positive technicals")
        
        if row.get('volume_score', 0) > 80:
            reasons.append("High volume activity")
        
        # Risk reasoning
        risk_level = row.get('risk_level', 'Unknown')
        if risk_level in ['High', 'Very High']:
            reasons.append(f"{risk_level} risk")
        
        # Special conditions
        if row.get('volume_spike', False):
            reasons.append("Volume spike detected")
        
        if row.get('has_momentum', False):
            reasons.append("Positive trend continuation")
        
        if row.get('is_value_stock', False):
            reasons.append("Value opportunity")
        
        # Decision-specific suffix
        if decision == "BUY":
            suffix = "Ready for entry"
        elif decision == "WATCH":
            suffix = "Monitor for better entry"
        else:
            suffix = "Better opportunities available"
        
        # Combine reasons
        reasoning = " | ".join(reasons[:3])  # Top 3 reasons
        if reasoning:
            reasoning += f" → {suffix}"
        else:
            reasoning = suffix
        
        return reasoning
    
    # ========================================================================
    # OPPORTUNITY SCORING
    # ========================================================================
    
    def _calculate_opportunity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate opportunity score combining upside potential and probability
        """
        # Components of opportunity
        upside = ((df['target_price'] - df['price']) / df['price'] * 100).fillna(0)
        probability = df['confidence'].fillna(0) / 100
        risk_adjustment = (100 - df['risk_score'].fillna(50)) / 100
        
        # Opportunity = Upside * Probability * Risk Adjustment
        opportunity = upside * probability * risk_adjustment
        
        # Normalize to 0-100 scale
        max_opportunity = opportunity.max() if opportunity.max() > 0 else 1
        normalized = (opportunity / max_opportunity * 100).round(1)
        
        return normalized

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def make_trading_decisions(
    scored_df: pd.DataFrame,
    criteria: Optional[DecisionCriteria] = None
) -> pd.DataFrame:
    """
    Make trading decisions for all stocks
    
    Args:
        scored_df: DataFrame with signal scores
        criteria: Decision criteria (optional)
        
    Returns:
        DataFrame with decisions, targets, and reasoning
    """
    engine = DecisionEngine(criteria)
    return engine.make_decisions(scored_df)

def get_buy_recommendations(
    decided_df: pd.DataFrame,
    min_confidence: float = 70,
    max_risk: float = 60
) -> pd.DataFrame:
    """
    Get filtered buy recommendations
    
    Args:
        decided_df: DataFrame with decisions
        min_confidence: Minimum confidence level
        max_risk: Maximum risk score
        
    Returns:
        Filtered DataFrame of buy recommendations
    """
    if 'decision' not in decided_df.columns:
        logger.error("No decision column found")
        return pd.DataFrame()
    
    buys = decided_df[
        (decided_df['decision'] == 'BUY') &
        (decided_df['confidence'] >= min_confidence) &
        (decided_df['risk_score'] <= max_risk)
    ]
    
    return buys.sort_values('opportunity_score', ascending=False)

def get_watch_list(
    decided_df: pd.DataFrame,
    include_reasons: List[str] = None
) -> pd.DataFrame:
    """
    Get watch list with optional filtering by reasons
    
    Args:
        decided_df: DataFrame with decisions
        include_reasons: List of reasons to filter by
        
    Returns:
        Watch list DataFrame
    """
    watch = decided_df[decided_df['decision'] == 'WATCH']
    
    if include_reasons:
        # Filter by reasoning
        mask = watch['reasoning'].str.contains('|'.join(include_reasons), case=False)
        watch = watch[mask]
    
    return watch.sort_values('composite_score', ascending=False)

def get_decision_summary(decided_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of decisions
    
    Args:
        decided_df: DataFrame with decisions
        
    Returns:
        Dictionary with summary stats
    """
    summary = {
        'total_stocks': len(decided_df),
        'decisions': decided_df['decision'].value_counts().to_dict(),
        'avg_confidence': {
            decision: decided_df[decided_df['decision'] == decision]['confidence'].mean()
            for decision in ['BUY', 'WATCH', 'AVOID']
        },
        'risk_distribution': decided_df['risk_level'].value_counts().to_dict(),
        'avg_risk_reward': decided_df[decided_df['decision'] == 'BUY']['risk_reward_ratio'].mean(),
        'top_opportunities': decided_df.nlargest(5, 'opportunity_score')[
            ['ticker', 'opportunity_score', 'decision', 'reasoning']
        ].to_dict('records')
    }
    
    return summary

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Decision Engine")
    print("="*60)
    print("\nTransforms signals into actionable trading decisions")
    print("\nFeatures:")
    print("- Risk-adjusted decision making")
    print("- Target price calculation")
    print("- Stop loss levels")
    print("- Clear reasoning for each decision")
    print("- Opportunity scoring")
    print("\nUse make_trading_decisions() with scored data")
    print("="*60)
