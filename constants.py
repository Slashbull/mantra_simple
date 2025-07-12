"""
constants.py - M.A.N.T.R.A. Configuration
========================================
All configuration, thresholds, and settings in one place
Simple, clear, and easy to modify
"""

# ============================================================================
# DATA SOURCES
# ============================================================================

# Google Sheets Configuration
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

SHEET_CONFIGS = {
    "watchlist": {
        "gid": "2026492216",
        "name": "ALL STOCKS 2025 Watchlist"
    },
    "sector": {
        "gid": "140104095",
        "name": "ALL STOCKS 2025 Sector Analysis"
    },
    "returns": {
        "gid": "100734077",
        "name": "Stock Return Analysis"
    }
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================

# Composite score thresholds for Buy/Watch/Avoid signals
SIGNAL_LEVELS = {
    "STRONG_BUY": 90,
    "BUY": 80,
    "WATCH": 65,
    "NEUTRAL": 50,
    "AVOID": 35,
    "STRONG_AVOID": 20
}

# Factor weights for composite scoring
FACTOR_WEIGHTS = {
    "momentum": 0.30,      # 30% - Recent price performance
    "value": 0.25,         # 25% - Valuation metrics
    "technical": 0.20,     # 20% - Technical indicators
    "volume": 0.15,        # 15% - Volume activity
    "fundamentals": 0.10   # 10% - EPS, sector strength
}

# ============================================================================
# MOMENTUM SETTINGS
# ============================================================================

# What constitutes strong/weak momentum
MOMENTUM_LEVELS = {
    "strong_positive": {
        "1d": 3.0,     # >3% daily
        "7d": 7.0,     # >7% weekly
        "30d": 15.0,   # >15% monthly
        "3m": 30.0     # >30% quarterly
    },
    "positive": {
        "1d": 1.0,
        "7d": 2.0,
        "30d": 5.0,
        "3m": 10.0
    },
    "neutral": {
        "1d": -1.0,
        "7d": -2.0,
        "30d": -5.0,
        "3m": -10.0
    },
    "negative": {
        "1d": -3.0,
        "7d": -7.0,
        "30d": -15.0,
        "3m": -30.0
    }
}

# ============================================================================
# VALUE SETTINGS
# ============================================================================

# PE Ratio ranges
PE_RANGES = {
    "undervalued": (0, 15),
    "fairly_valued": (15, 25),
    "overvalued": (25, 40),
    "expensive": (40, float('inf')),
    "negative": (float('-inf'), 0)  # Loss-making
}

# EPS Growth ranges
EPS_GROWTH_RANGES = {
    "high_growth": 30,      # >30% growth
    "growth": 15,           # 15-30%
    "stable": 0,            # 0-15%
    "declining": -15,       # -15% to 0%
    "sharp_decline": -30    # < -30%
}

# ============================================================================
# TECHNICAL SETTINGS
# ============================================================================

# Moving average periods
MA_PERIODS = {
    "short": 20,
    "medium": 50,
    "long": 200
}

# Price position thresholds
PRICE_POSITION = {
    "overbought": 90,       # >90% of 52w range
    "extended": 75,         # 75-90%
    "neutral": 50,          # 25-75%
    "oversold": 25,         # 10-25%
    "extremely_oversold": 10  # <10%
}

# SMA distance thresholds (%)
SMA_DISTANCE = {
    "far_above": 10,        # >10% above SMA
    "above": 5,             # 5-10% above
    "near": 2,              # Within ±2%
    "below": -5,            # 5-10% below
    "far_below": -10        # >10% below
}

# ============================================================================
# VOLUME SETTINGS
# ============================================================================

# Relative volume thresholds
VOLUME_LEVELS = {
    "extreme_spike": 5.0,   # 5x normal volume
    "high_spike": 3.0,      # 3x normal
    "elevated": 1.5,        # 1.5x normal
    "normal": 1.0,          # Normal volume
    "low": 0.5,             # Half normal
    "dry": 0.2              # Very low volume
}

# Volume trend thresholds (% change)
VOLUME_TREND = {
    "surging": 100,         # >100% increase
    "increasing": 50,       # 50-100%
    "stable": 0,            # -20% to +50%
    "decreasing": -20,      # -50% to -20%
    "drying_up": -50        # < -50%
}

# ============================================================================
# MARKET CAP CATEGORIES
# ============================================================================

# In Crores (Cr)
MARKET_CAP_RANGES = {
    "mega_cap": 100000,     # > ₹1,00,000 Cr
    "large_cap": 20000,     # ₹20,000 - ₹1,00,000 Cr
    "mid_cap": 5000,        # ₹5,000 - ₹20,000 Cr
    "small_cap": 500,       # ₹500 - ₹5,000 Cr
    "micro_cap": 0          # < ₹500 Cr
}

# ============================================================================
# RISK LEVELS
# ============================================================================

RISK_SCORES = {
    "very_low": (0, 20),
    "low": (20, 40),
    "moderate": (40, 60),
    "high": (60, 80),
    "very_high": (80, 100)
}

# Factors that increase risk
RISK_FACTORS = {
    "high_pe": 10,          # PE > 40
    "negative_eps": 20,     # Loss-making
    "low_volume": 15,       # Illiquid
    "high_volatility": 15,  # Large price swings
    "penny_stock": 20,      # Price < ₹50
    "near_52w_low": 10,     # Within 10% of 52w low
}

# ============================================================================
# ALERTS AND NOTIFICATIONS
# ============================================================================

# Alert priority levels
ALERT_PRIORITY = {
    "critical": 1,          # Immediate action needed
    "high": 2,              # Important
    "medium": 3,            # Worth noting
    "low": 4,               # Informational
}

# Alert conditions
ALERT_CONDITIONS = {
    "strong_buy_signal": {
        "condition": "composite_score > 90",
        "priority": "critical",
        "message": "Strong Buy opportunity detected"
    },
    "volume_breakout": {
        "condition": "rvol > 3 and ret_1d > 2",
        "priority": "high",
        "message": "Volume breakout with positive price action"
    },
    "oversold_bounce": {
        "condition": "position_52w < 20 and ret_1d > 3",
        "priority": "high",
        "message": "Potential reversal from oversold levels"
    },
    "momentum_surge": {
        "condition": "ret_7d > 10 and volume_spike",
        "priority": "medium",
        "message": "Strong momentum with volume support"
    }
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Colors for UI (hex codes)
COLORS = {
    "buy": "#00C851",       # Green
    "watch": "#FF8800",     # Orange
    "neutral": "#33B5E5",   # Blue
    "avoid": "#FF4444",     # Red
    "background": "#1A1A1A", # Dark
    "text": "#FFFFFF",      # White
    "muted": "#888888"      # Gray
}

# Number formatting
NUMBER_FORMAT = {
    "price": "{:,.2f}",
    "percentage": "{:+.2f}%",
    "volume": "{:,.0f}",
    "market_cap": "{:,.0f} Cr",
    "score": "{:.1f}"
}

# Table display settings
TABLE_SETTINGS = {
    "rows_per_page": 50,
    "max_rows": 500,
    "decimal_places": 2,
    "show_rank": True
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Cache durations (minutes)
CACHE_DURATION = {
    "stock_data": 15,       # 15 minutes
    "sector_data": 30,      # 30 minutes
    "calculations": 5,      # 5 minutes
}

# Processing limits
PROCESSING_LIMITS = {
    "max_stocks": 5000,
    "batch_size": 100,
    "timeout_seconds": 30
}

# ============================================================================
# QUALITY CHECKS
# ============================================================================

# Data quality thresholds
DATA_QUALITY = {
    "min_valid_rows": 100,
    "max_null_percent": 30,
    "min_price": 0.01,
    "max_price": 1000000,
    "min_volume": 0,
    "max_pe": 1000
}

# Required columns for each dataset
REQUIRED_COLUMNS = {
    "stocks": [
        "ticker", "price", "sector", "market_cap",
        "pe", "eps_current", "ret_1d", "ret_7d", 
        "ret_30d", "volume_1d", "rvol"
    ],
    "sector": [
        "sector", "sector_ret_1d", "sector_ret_7d",
        "sector_ret_30d", "sector_count"
    ]
}

# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

# Market regime detection thresholds
MARKET_REGIMES = {
    "bull_market": {
        "advancing_percent": 70,    # >70% stocks up
        "avg_return_30d": 5,        # >5% average return
        "volume_expansion": 1.2     # 20% volume increase
    },
    "bear_market": {
        "advancing_percent": 30,    # <30% stocks up
        "avg_return_30d": -5,       # <-5% average return
        "volume_expansion": 0.8     # 20% volume decrease
    },
    "sideways": {
        "advancing_percent": 50,    # ~50% stocks up
        "avg_return_30d": 0,        # -2% to +2%
        "volume_expansion": 1.0     # Normal volume
    }
}

# ============================================================================
# SECTOR MAPPINGS
# ============================================================================

# Sector groupings for analysis
SECTOR_GROUPS = {
    "Defensive": [
        "FMCG", "Pharmaceuticals", "Utilities", "Telecommunications"
    ],
    "Cyclical": [
        "Automobiles & Auto Parts", "Banks", "Real Estate", 
        "Machinery, Equipment & Components"
    ],
    "Growth": [
        "Software & IT Services", "Internet & E-Commerce", 
        "Biotechnology", "Renewable Energy"
    ],
    "Commodity": [
        "Metals & Mining", "Oil & Gas", "Chemicals", 
        "Agriculture & Allied"
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_signal_category(score: float) -> str:
    """Get signal category based on composite score"""
    if score >= SIGNAL_LEVELS["STRONG_BUY"]:
        return "STRONG BUY"
    elif score >= SIGNAL_LEVELS["BUY"]:
        return "BUY"
    elif score >= SIGNAL_LEVELS["WATCH"]:
        return "WATCH"
    elif score >= SIGNAL_LEVELS["NEUTRAL"]:
        return "NEUTRAL"
    elif score >= SIGNAL_LEVELS["AVOID"]:
        return "AVOID"
    else:
        return "STRONG AVOID"

def get_risk_category(risk_score: float) -> str:
    """Get risk category based on risk score"""
    for category, (min_val, max_val) in RISK_SCORES.items():
        if min_val <= risk_score < max_val:
            return category.replace("_", " ").title()
    return "Unknown"

def format_number(value: float, format_type: str) -> str:
    """Format number based on type"""
    if format_type in NUMBER_FORMAT:
        return NUMBER_FORMAT[format_type].format(value)
    return str(value)

# ============================================================================
# VALIDATION
# ============================================================================

# Sanity check on load
def validate_constants():
    """Validate that constants are properly configured"""
    # Check factor weights sum to 1
    weight_sum = sum(FACTOR_WEIGHTS.values())
    assert abs(weight_sum - 1.0) < 0.01, f"Factor weights must sum to 1, got {weight_sum}"
    
    # Check signal levels are in order
    levels = list(SIGNAL_LEVELS.values())
    assert levels == sorted(levels, reverse=True), "Signal levels must be in descending order"
    
    print("✅ Constants validated successfully")

# Run validation when module loads
if __name__ == "__main__":
    validate_constants()
    print("\nM.A.N.T.R.A. Constants loaded")
    print(f"Factors: {list(FACTOR_WEIGHTS.keys())}")
    print(f"Signal levels: {list(SIGNAL_LEVELS.keys())}")
