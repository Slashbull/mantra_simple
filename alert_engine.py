"""
alert_engine.py - M.A.N.T.R.A. Alert Engine
==========================================
Real-time monitoring and alert generation for trading opportunities
Tracks price movements, volume spikes, and signal changes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import from constants
from constants import (
    ALERT_PRIORITY, ALERT_CONDITIONS, MOMENTUM_LEVELS,
    VOLUME_LEVELS, PRICE_POSITION
)

logger = logging.getLogger(__name__)

# ============================================================================
# ALERT TYPES AND CONFIGURATION
# ============================================================================

class AlertType(Enum):
    """Types of alerts"""
    PRICE_BREAKOUT = "price_breakout"
    VOLUME_SPIKE = "volume_spike"
    SIGNAL_CHANGE = "signal_change"
    MOMENTUM_SHIFT = "momentum_shift"
    PATTERN_DETECTED = "pattern_detected"
    RISK_WARNING = "risk_warning"
    SECTOR_ROTATION = "sector_rotation"
    TARGET_REACHED = "target_reached"
    STOP_LOSS_WARNING = "stop_loss_warning"
    OPPORTUNITY = "opportunity"

class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

@dataclass
class Alert:
    """Individual alert object"""
    alert_id: str
    timestamp: datetime
    ticker: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    metrics: Dict = field(default_factory=dict)
    action_required: bool = False
    expires_at: Optional[datetime] = None

@dataclass
class AlertConfig:
    """Configuration for alert generation"""
    # Price alerts
    price_spike_pct: float = 5.0
    price_crash_pct: float = -5.0
    breakout_threshold: float = 3.0
    
    # Volume alerts
    volume_spike_multiplier: float = 3.0
    volume_dry_multiplier: float = 0.3
    
    # Signal alerts
    score_improvement: float = 10.0
    signal_upgrade: bool = True
    
    # Risk alerts
    stop_loss_buffer: float = 2.0
    target_buffer: float = 2.0
    risk_increase: float = 20.0
    
    # Timing
    alert_cooldown_hours: int = 24
    max_alerts_per_stock: int = 3

# ============================================================================
# ALERT ENGINE
# ============================================================================

class AlertEngine:
    """
    Generates and manages trading alerts
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.alerts: List[Alert] = []
        self.alert_history: Dict[str, List[datetime]] = {}
        self.alert_counter = 0
        
    def generate_alerts(
        self,
        current_df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame] = None,
        user_positions: Optional[Dict[str, Dict]] = None
    ) -> List[Alert]:
        """
        Generate all alerts based on current market data
        
        Args:
            current_df: Current market data
            previous_df: Previous market data for comparison
            user_positions: User's current positions
            
        Returns:
            List of generated alerts
        """
        self.alerts = []
        timestamp = datetime.now()
        
        logger.info("Generating alerts...")
        
        # Price-based alerts
        self._check_price_alerts(current_df)
        
        # Volume-based alerts
        self._check_volume_alerts(current_df)
        
        # Pattern-based alerts
        self._check_pattern_alerts(current_df)
        
        # Signal changes (if previous data available)
        if previous_df is not None:
            self._check_signal_changes(current_df, previous_df)
        
        # Position-based alerts (if positions provided)
        if user_positions:
            self._check_position_alerts(current_df, user_positions)
        
        # Opportunity alerts
        self._check_opportunity_alerts(current_df)
        
        # Sector rotation alerts
        self._check_sector_alerts(current_df)
        
        # Filter alerts by cooldown
        self.alerts = self._apply_cooldown(self.alerts)
        
        # Sort by priority
        self.alerts.sort(key=lambda x: (x.priority.value, x.timestamp))
        
        logger.info(f"Generated {len(self.alerts)} alerts")
        
        return self.alerts
    
    # ========================================================================
    # PRICE ALERTS
    # ========================================================================
    
    def _check_price_alerts(self, df: pd.DataFrame):
        """Check for price-based alerts"""
        for _, stock in df.iterrows():
            ticker = stock.get('ticker', '')
            
            # Price spike alert
            if 'ret_1d' in stock and stock['ret_1d'] > self.config.price_spike_pct:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.PRICE_BREAKOUT,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Price Surge +{stock['ret_1d']:.1f}%",
                    message=f"Strong upward movement detected. Volume: {stock.get('rvol', 1):.1f}x normal",
                    metrics={
                        'price': stock.get('price', 0),
                        'change_pct': stock['ret_1d'],
                        'volume': stock.get('volume_1d', 0)
                    }
                )
            
            # Price crash alert
            elif 'ret_1d' in stock and stock['ret_1d'] < self.config.price_crash_pct:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.RISK_WARNING,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Price Drop {stock['ret_1d']:.1f}%",
                    message="Significant decline detected. Review position",
                    metrics={
                        'price': stock.get('price', 0),
                        'change_pct': stock['ret_1d']
                    },
                    action_required=True
                )
            
            # 52-week high alert
            if 'position_52w' in stock and stock['position_52w'] > 95:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.PRICE_BREAKOUT,
                    priority=AlertPriority.MEDIUM,
                    title=f"{ticker}: Near 52-Week High",
                    message="Trading within 5% of 52-week high",
                    metrics={
                        'position_52w': stock['position_52w'],
                        'high_52w': stock.get('high_52w', 0)
                    }
                )
            
            # SMA breakout alert
            if 'distance_from_sma_200d' in stock and stock['distance_from_sma_200d'] > self.config.breakout_threshold:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.PRICE_BREAKOUT,
                    priority=AlertPriority.MEDIUM,
                    title=f"{ticker}: 200-DMA Breakout",
                    message=f"Price {stock['distance_from_sma_200d']:.1f}% above 200-DMA",
                    metrics={
                        'price': stock.get('price', 0),
                        'sma_200d': stock.get('sma_200d', 0)
                    }
                )
    
    # ========================================================================
    # VOLUME ALERTS
    # ========================================================================
    
    def _check_volume_alerts(self, df: pd.DataFrame):
        """Check for volume-based alerts"""
        for _, stock in df.iterrows():
            ticker = stock.get('ticker', '')
            
            # Volume spike alert
            if 'rvol' in stock and stock['rvol'] > self.config.volume_spike_multiplier:
                # Check if accompanied by price movement
                price_move = stock.get('ret_1d', 0)
                
                if abs(price_move) > 2:
                    self._create_alert(
                        ticker=ticker,
                        alert_type=AlertType.VOLUME_SPIKE,
                        priority=AlertPriority.HIGH,
                        title=f"{ticker}: Volume Explosion {stock['rvol']:.1f}x",
                        message=f"Massive volume with {price_move:+.1f}% price move",
                        metrics={
                            'rvol': stock['rvol'],
                            'volume': stock.get('volume_1d', 0),
                            'price_change': price_move
                        }
                    )
                else:
                    self._create_alert(
                        ticker=ticker,
                        alert_type=AlertType.VOLUME_SPIKE,
                        priority=AlertPriority.MEDIUM,
                        title=f"{ticker}: High Volume Alert",
                        message="Unusual volume without significant price movement",
                        metrics={
                            'rvol': stock['rvol'],
                            'volume': stock.get('volume_1d', 0)
                        }
                    )
    
    # ========================================================================
    # PATTERN ALERTS
    # ========================================================================
    
    def _check_pattern_alerts(self, df: pd.DataFrame):
        """Check for pattern-based alerts"""
        for _, stock in df.iterrows():
            ticker = stock.get('ticker', '')
            
            # Breakout pattern
            if stock.get('breakout_pattern', False):
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.PATTERN_DETECTED,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Breakout Pattern Detected",
                    message="Technical breakout with volume confirmation",
                    metrics={
                        'pattern': 'breakout',
                        'confidence': stock.get('edge_score', 0)
                    }
                )
            
            # Reversal pattern
            if stock.get('reversal_pattern', False):
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.PATTERN_DETECTED,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Reversal Pattern Forming",
                    message="Potential trend reversal detected",
                    metrics={
                        'pattern': 'reversal',
                        'position_52w': stock.get('position_52w', 0)
                    }
                )
            
            # Momentum continuation
            if stock.get('momentum_continuation', False):
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.MOMENTUM_SHIFT,
                    priority=AlertPriority.MEDIUM,
                    title=f"{ticker}: Strong Momentum Continues",
                    message="Trend acceleration detected",
                    metrics={
                        'momentum_score': stock.get('momentum_score', 0),
                        'trend_strength': stock.get('trend_score', 0)
                    }
                )
    
    # ========================================================================
    # SIGNAL CHANGE ALERTS
    # ========================================================================
    
    def _check_signal_changes(self, current_df: pd.DataFrame, previous_df: pd.DataFrame):
        """Check for changes in signals between periods"""
        # Merge on ticker to compare
        comparison = pd.merge(
            current_df[['ticker', 'decision', 'composite_score', 'risk_score']],
            previous_df[['ticker', 'decision', 'composite_score', 'risk_score']],
            on='ticker',
            suffixes=('_now', '_prev')
        )
        
        for _, row in comparison.iterrows():
            ticker = row['ticker']
            
            # Decision upgrade
            if row['decision_now'] == 'BUY' and row['decision_prev'] in ['WATCH', 'AVOID']:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.SIGNAL_CHANGE,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Upgraded to BUY",
                    message=f"Signal improved from {row['decision_prev']} to BUY",
                    metrics={
                        'score_change': row['composite_score_now'] - row['composite_score_prev'],
                        'new_score': row['composite_score_now']
                    },
                    action_required=True
                )
            
            # Decision downgrade
            elif row['decision_now'] == 'AVOID' and row['decision_prev'] in ['BUY', 'WATCH']:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.RISK_WARNING,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Downgraded to AVOID",
                    message=f"Signal deteriorated from {row['decision_prev']} to AVOID",
                    metrics={
                        'score_change': row['composite_score_now'] - row['composite_score_prev'],
                        'risk_increase': row['risk_score_now'] - row['risk_score_prev']
                    },
                    action_required=True
                )
            
            # Significant score improvement
            score_change = row['composite_score_now'] - row['composite_score_prev']
            if score_change > self.config.score_improvement:
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.SIGNAL_CHANGE,
                    priority=AlertPriority.MEDIUM,
                    title=f"{ticker}: Score Improved +{score_change:.1f}",
                    message="Significant improvement in composite score",
                    metrics={
                        'old_score': row['composite_score_prev'],
                        'new_score': row['composite_score_now']
                    }
                )
    
    # ========================================================================
    # POSITION ALERTS
    # ========================================================================
    
    def _check_position_alerts(self, df: pd.DataFrame, positions: Dict[str, Dict]):
        """Check alerts for user positions"""
        for ticker, position in positions.items():
            if ticker not in df['ticker'].values:
                continue
                
            stock = df[df['ticker'] == ticker].iloc[0]
            current_price = stock.get('price', 0)
            buy_price = position.get('buy_price', current_price)
            
            # Calculate gain/loss
            gain_pct = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
            
            # Target reached
            target = position.get('target_price', buy_price * 1.15)
            if current_price >= target * (1 - self.config.target_buffer / 100):
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.TARGET_REACHED,
                    priority=AlertPriority.HIGH,
                    title=f"{ticker}: Approaching Target",
                    message=f"Price near target ₹{target:.2f} (Gain: {gain_pct:+.1f}%)",
                    metrics={
                        'current_price': current_price,
                        'target_price': target,
                        'gain_pct': gain_pct
                    },
                    action_required=True
                )
            
            # Stop loss warning
            stop_loss = position.get('stop_loss', buy_price * 0.92)
            if current_price <= stop_loss * (1 + self.config.stop_loss_buffer / 100):
                self._create_alert(
                    ticker=ticker,
                    alert_type=AlertType.STOP_LOSS_WARNING,
                    priority=AlertPriority.CRITICAL,
                    title=f"{ticker}: Near Stop Loss",
                    message=f"Price approaching stop loss ₹{stop_loss:.2f} (Loss: {gain_pct:.1f}%)",
                    metrics={
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'loss_pct': gain_pct
                    },
                    action_required=True
                )
    
    # ========================================================================
    # OPPORTUNITY ALERTS
    # ========================================================================
    
    def _check_opportunity_alerts(self, df: pd.DataFrame):
        """Check for new opportunities"""
        # Top opportunities
        top_opps = df.nlargest(5, 'opportunity_score') if 'opportunity_score' in df.columns else df.nlargest(5, 'composite_score')
        
        for _, stock in top_opps.iterrows():
            if stock.get('decision') == 'BUY' and stock.get('composite_score', 0) > 85:
                self._create_alert(
                    ticker=stock['ticker'],
                    alert_type=AlertType.OPPORTUNITY,
                    priority=AlertPriority.HIGH,
                    title=f"{stock['ticker']}: High Conviction Opportunity",
                    message=f"{stock.get('reasoning', 'Strong buy signal detected')}",
                    metrics={
                        'score': stock.get('composite_score', 0),
                        'expected_return': stock.get('edge_expected_return', 0),
                        'risk_level': stock.get('risk_level', 'Unknown')
                    }
                )
    
    # ========================================================================
    # SECTOR ALERTS
    # ========================================================================
    
    def _check_sector_alerts(self, df: pd.DataFrame):
        """Check for sector rotation alerts"""
        if 'sector_rotation_signal' not in df.columns:
            return
            
        # Find sectors with strong rotation signals
        sector_signals = df.groupby('sector')['sector_rotation_signal'].agg(
            lambda x: x.mode()[0] if len(x) > 0 else 'NEUTRAL'
        )
        
        for sector, signal in sector_signals.items():
            if signal == 'BUY':
                # Count stocks in sector with buy signals
                sector_buys = df[(df['sector'] == sector) & (df['decision'] == 'BUY')]
                
                if len(sector_buys) >= 3:
                    self._create_alert(
                        ticker=f"SECTOR:{sector[:10]}",
                        alert_type=AlertType.SECTOR_ROTATION,
                        priority=AlertPriority.MEDIUM,
                        title=f"Sector Alert: {sector}",
                        message=f"Strong rotation into {sector} sector ({len(sector_buys)} buy signals)",
                        metrics={
                            'sector': sector,
                            'buy_count': len(sector_buys),
                            'top_stock': sector_buys.iloc[0]['ticker'] if len(sector_buys) > 0 else ''
                        }
                    )
    
    # ========================================================================
    # ALERT MANAGEMENT
    # ========================================================================
    
    def _create_alert(
        self,
        ticker: str,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        metrics: Dict = None,
        action_required: bool = False
    ):
        """Create and add an alert"""
        self.alert_counter += 1
        
        alert = Alert(
            alert_id=f"ALERT_{self.alert_counter:05d}",
            timestamp=datetime.now(),
            ticker=ticker,
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            metrics=metrics or {},
            action_required=action_required,
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        self.alerts.append(alert)
        
        # Update history
        if ticker not in self.alert_history:
            self.alert_history[ticker] = []
        self.alert_history[ticker].append(alert.timestamp)
    
    def _apply_cooldown(self, alerts: List[Alert]) -> List[Alert]:
        """Apply cooldown period to prevent alert spam"""
        filtered_alerts = []
        current_time = datetime.now()
        
        for alert in alerts:
            ticker = alert.ticker
            
            # Check cooldown
            if ticker in self.alert_history:
                recent_alerts = [
                    t for t in self.alert_history[ticker]
                    if current_time - t < timedelta(hours=self.config.alert_cooldown_hours)
                ]
                
                if len(recent_alerts) >= self.config.max_alerts_per_stock:
                    continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_alerts_by_priority(
        self,
        priorities: List[AlertPriority]
    ) -> List[Alert]:
        """Get alerts filtered by priority"""
        return [
            alert for alert in self.alerts
            if alert.priority in priorities
        ]
    
    def get_alerts_by_type(
        self,
        alert_types: List[AlertType]
    ) -> List[Alert]:
        """Get alerts filtered by type"""
        return [
            alert for alert in self.alerts
            if alert.alert_type in alert_types
        ]
    
    def get_action_required_alerts(self) -> List[Alert]:
        """Get alerts that require immediate action"""
        return [
            alert for alert in self.alerts
            if alert.action_required
        ]

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_alerts(
    current_data: pd.DataFrame,
    previous_data: Optional[pd.DataFrame] = None,
    positions: Optional[Dict[str, Dict]] = None,
    config: Optional[AlertConfig] = None
) -> List[Alert]:
    """
    Generate trading alerts
    
    Args:
        current_data: Current market data
        previous_data: Previous data for comparison
        positions: User positions
        config: Alert configuration
        
    Returns:
        List of alerts
    """
    engine = AlertEngine(config)
    return engine.generate_alerts(current_data, previous_data, positions)

def get_critical_alerts(alerts: List[Alert]) -> List[Alert]:
    """Get only critical and high priority alerts"""
    return [
        alert for alert in alerts
        if alert.priority in [AlertPriority.CRITICAL, AlertPriority.HIGH]
    ]

def format_alert_summary(alerts: List[Alert]) -> pd.DataFrame:
    """Format alerts as a summary DataFrame"""
    if not alerts:
        return pd.DataFrame()
    
    summary_data = []
    for alert in alerts:
        summary_data.append({
            'Time': alert.timestamp.strftime('%H:%M'),
            'Ticker': alert.ticker,
            'Type': alert.alert_type.value,
            'Priority': alert.priority.name,
            'Title': alert.title,
            'Action': '🚨' if alert.action_required else ''
        })
    
    return pd.DataFrame(summary_data)

def get_alert_statistics(alerts: List[Alert]) -> Dict:
    """Get statistics about alerts"""
    if not alerts:
        return {'total': 0}
    
    stats = {
        'total': len(alerts),
        'by_priority': {},
        'by_type': {},
        'action_required': len([a for a in alerts if a.action_required]),
        'unique_tickers': len(set(a.ticker for a in alerts))
    }
    
    # Count by priority
    for priority in AlertPriority:
        count = len([a for a in alerts if a.priority == priority])
        if count > 0:
            stats['by_priority'][priority.name] = count
    
    # Count by type
    for alert_type in AlertType:
        count = len([a for a in alerts if a.alert_type == alert_type])
        if count > 0:
            stats['by_type'][alert_type.value] = count
    
    return stats

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Alert Engine")
    print("="*60)
    print("\nReal-time monitoring and alert generation")
    print("\nAlert Types:")
    for alert_type in AlertType:
        print(f"  - {alert_type.value}")
    print("\nPriority Levels:")
    for priority in AlertPriority:
        print(f"  - {priority.name}: {priority.value}")
    print("\nUse generate_alerts() to monitor market")
    print("="*60)
