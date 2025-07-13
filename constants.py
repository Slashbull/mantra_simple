"""
constants.py - M.A.N.T.R.A. Configuration
========================================
FINAL VERSION - All configuration, thresholds, and settings in one place
Simple, clear, and easy to modify
"""

# ============================================================================
# DATA SOURCES - UPDATE THESE WITH YOUR ACTUAL URLS!
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

# Complete URLs for data sources (constructed from above)
DATA_SOURCES = {
    'WATCHLIST': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['watchlist']['gid']}",
    'SECTOR_ANALYSIS': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['sector']['gid']}",
    'STOCK_RETURNS': f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['returns']['gid']}"
}

# Cache settings
CACHE_DURATION_MINUTES = 15  # Refresh data every 15 minutes
ENABLE_CACHING = True

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

# For backward compatibility
SIGNAL_THRESHOLDS = {
    'BUY': SIGNAL_LEVELS['BUY'],
    'WATCH': SIGNAL_LEVELS['WATCH'],
    'AVOID': SIGNAL_LEVELS['AVOID']
}

# Factor weights for composite scoring (must sum to 1.0)
FACTOR_WEIGHTS = {
    "momentum": 0.30,      # 30% - Recent price performance
    "value": 0.25,         # 25% - Valuation metrics
    "technical": 0.20,     # 20% - Technical indicators
    "volume": 0.15,        # 15% - Volume activity
    "fundamentals": 0.10   # 10% - EPS, sector strength
}

# Alternative name for compatibility
SCORE_WEIGHTS = FACTOR_WEIGHTS.copy()

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

# Alternative format for compatibility
MOMENTUM_THRESHOLDS = {
    'EXTREME_BULLISH': {
        '1D': 5.0,    # 1-day return > 5%
        '7D': 10.0,   # 7-day return > 10%
        '30D': 15.0,  # 30-day return > 15%
        '3M': 25.0,   # 3-month return > 25%
    },
    'BULLISH': {
        '1D': 2.0,
        '7D': 5.0,
        '30D': 8.0,
        '3M': 15.0,
    },
    'NEUTRAL': {
        '1D': -1.0,
        '7D': -2.0,
        '30D': -3.0,
        '3M': -5.0,
    },
    'BEARISH': {
        '1D': -3.0,
        '7D': -7.0,
        '30D': -10.0,
        '3M': -15.0,
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

# Value thresholds
VALUE_THRESHOLDS = {
    'PE_RATIO': {
        'DEEP_VALUE': 12,      # PE < 12 = Deep value
        'VALUE': 18,           # PE < 18 = Value
        'FAIR': 25,            # PE < 25 = Fair
        'EXPENSIVE': 35,       # PE < 35 = Expensive
        'BUBBLE': 50,          # PE > 50 = Bubble territory
    },
    'EPS_GROWTH': {
        'HYPER': 50,          # EPS growth > 50% = Hyper growth
        'HIGH': 25,           # EPS growth > 25% = High growth
        'MODERATE': 15,       # EPS growth > 15% = Moderate
        'LOW': 5,             # EPS growth > 5% = Low growth
        'NEGATIVE': 0,        # EPS growth < 0 = Declining
    },
    'PRICE_TO_52W': {
        'NEAR_LOW': 10,       # Within 10% of 52w low = Oversold
        'CHEAP': 25,          # Within 25% of 52w low = Cheap
        'MIDDLE': 50,         # Middle of range
        'EXPENSIVE': 75,      # Within 25% of 52w high
        'NEAR_HIGH': 90,      # Within 10% of 52w high = Overbought
    }
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

# Technical thresholds
TECHNICAL_THRESHOLDS = {
    'SMA_POSITIONS': {
        'STRONG_BULLISH': 'ABOVE_ALL',     # Price > SMA20 > SMA50 > SMA200
        'BULLISH': 'ABOVE_200',            # Price > SMA200
        'NEUTRAL': 'BETWEEN',              # Mixed signals
        'BEARISH': 'BELOW_200',            # Price < SMA200
        'STRONG_BEARISH': 'BELOW_ALL',     # Price < all SMAs
    },
    'SMA_DISTANCE': {
        'FAR_ABOVE': 10,      # Price > 10% above SMA
        'ABOVE': 5,           # Price > 5% above SMA
        'NEAR': 2,            # Price within 2% of SMA
        'BELOW': -5,          # Price > 5% below SMA
        'FAR_BELOW': -10,     # Price > 10% below SMA
    }
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

# Volume thresholds for alerts
VOLUME_THRESHOLDS = {
    'SPIKE_MULTIPLIER': 3.0,         # Volume > 3x average = Spike
    'HIGH_MULTIPLIER': 2.0,          # Volume > 2x average = High
    'NORMAL_RANGE': (0.7, 1.3),      # 70% to 130% of average
    'LOW_MULTIPLIER': 0.5,           # Volume < 50% average = Low
    'DRY_MULTIPLIER': 0.3,           # Volume < 30% average = Dry
    
    'VOLUME_TREND': {
        'ACCUMULATION': 1.5,         # Rising volume > 1.5x trend
        'DISTRIBUTION': 0.7,         # Falling volume < 0.7x trend
    },
    
    'RVOL_THRESHOLDS': {
        'EXTREME': 5.0,    # RVOL > 5 = Extreme activity
        'HIGH': 2.0,       # RVOL > 2 = High activity
        'NORMAL': 1.0,     # RVOL ~1 = Normal
        'LOW': 0.5,        # RVOL < 0.5 = Low activity
    }
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

# Signal colors for UI
SIGNAL_COLORS = {
    'BUY': '#00d26a',       # Green
    'WATCH': '#ffa500',     # Orange
    'AVOID': '#ff4b4b',     # Red
    'NEUTRAL': '#808080',   # Gray
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
        "FMCG", "Pharmaceuticals", "Utilities", "Telecommunications",
        "Consumer Goods", "Healthcare Services"
    ],
    "Cyclical": [
        "Automobiles & Auto Parts", "Banks", "Real Estate", 
        "Machinery, Equipment & Components", "Construction", "Hotels & Tourism"
    ],
    "Growth": [
        "Software & IT Services", "Internet & E-Commerce", 
        "Biotechnology", "Renewable Energy", "Technology Hardware"
    ],
    "Commodity": [
        "Metals & Mining", "Oil & Gas", "Chemicals", 
        "Agriculture & Allied", "Paper & Forest Products"
    ]
}

# ============================================================================
# FILTERS AND RANGES
# ============================================================================

# Price range filters
PRICE_RANGE_FILTERS = {
    'ALL': (0, float('inf')),
    'PENNY': (0, 50),
    'LOW': (50, 250),
    'MID': (250, 1000),
    'HIGH': (1000, 5000),
    'PREMIUM': (5000, float('inf')),
}

# Default filter settings
DEFAULT_FILTERS = {
    'SIGNAL': 'ALL',
    'SECTOR': 'ALL',
    'MARKET_CAP': 'ALL',
    'PRICE_RANGE': 'ALL',
    'MIN_VOLUME': 10000,
    'SHOW_ONLY_ALERTS': False,
}

# ============================================================================
# COLUMN MAPPINGS (for handling different column names)
# ============================================================================

COLUMN_MAPPINGS = {
    'PRICE': ['price', 'close', 'last_price', 'current_price'],
    'VOLUME': ['volume_1d', 'volume', 'vol', 'daily_volume'],
    'PE': ['pe', 'pe_ratio', 'price_earnings'],
    'EPS': ['eps_current', 'eps', 'earnings_per_share'],
    'MARKET_CAP': ['market_cap', 'mcap', 'mkt_cap'],
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    'PRICE': {'min': 0, 'max': 1000000},
    'PE': {'min': -100, 'max': 1000},
    'VOLUME': {'min': 0, 'max': 1e12},
    'RETURNS': {'min': -100, 'max': 1000},
    'EPS': {'min': -1000, 'max': 1000},
}

# ============================================================================
# EPS TIERS CONFIGURATION
# ============================================================================

EPS_TIERS = {
    '95↑': {'min': 95, 'label': 'Elite', 'color': '#00ff00'},
    '75↑': {'min': 75, 'label': 'Excellent', 'color': '#32cd32'},
    '55↑': {'min': 55, 'label': 'Strong', 'color': '#90ee90'},
    '35↑': {'min': 35, 'label': 'Good', 'color': '#98fb98'},
    '15↑': {'min': 15, 'label': 'Above Avg', 'color': '#f0e68c'},
    '5↑': {'min': 5, 'label': 'Average', 'color': '#ffffe0'},
    '0': {'min': 0, 'label': 'Neutral', 'color': '#ffffff'},
    '5↓': {'min': -5, 'label': 'Below Avg', 'color': '#ffd700'},
    '15↓': {'min': -15, 'label': 'Weak', 'color': '#ffa500'},
    '35↓': {'min': -35, 'label': 'Poor', 'color': '#ff8c00'},
    '55↓': {'min': -55, 'label': 'Very Poor', 'color': '#ff6347'},
}

# ============================================================================
# PRICE TIERS CONFIGURATION
# ============================================================================

PRICE_TIERS = {
    '10K↑': {'min': 10000, 'label': 'Ultra Premium'},
    '5K↑': {'min': 5000, 'label': 'Premium'},
    '2K↑': {'min': 2000, 'label': 'High'},
    '1K↑': {'min': 1000, 'label': 'Mid-High'},
    '500↑': {'min': 500, 'label': 'Mid'},
    '250↑': {'min': 250, 'label': 'Mid-Low'},
    '100↑': {'min': 100, 'label': 'Low'},
    '50↑': {'min': 50, 'label': 'Micro'},
    '25↑': {'min': 25, 'label': 'Penny'},
    '10↑': {'min': 10, 'label': 'Ultra Penny'},
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    'THEME': 'dark',
    'REFRESH_INTERVAL': 300,        # Auto-refresh every 5 minutes
    'MAX_ROWS_DISPLAY': 100,        # Max rows in main table
    'CHART_HEIGHT': 400,            # Default chart height in pixels
    'SHOW_TOOLTIPS': True,
    'ANIMATE_CHARTS': True,
    'DEFAULT_TAB': 'overview',
}

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

PERFORMANCE_BENCHMARKS = {
    'TARGET_LOAD_TIME': 3.0,        # Seconds
    'TARGET_REFRESH_TIME': 2.0,     # Seconds
    'TARGET_CALCULATION_TIME': 1.0, # Seconds
    'MIN_DATA_QUALITY': 0.95,       # 95% data completeness required
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'DATA_LOAD_FAILED': "Unable to load data. Please check your internet connection.",
    'INVALID_DATA': "Data validation failed. Some information may be incorrect.",
    'CALCULATION_ERROR': "Error in calculations. Showing cached results.",
    'NO_DATA': "No data available for the selected filters.",
    'SHEET_NOT_FOUND': "Data source not found. Please check configuration.",
}

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_MESSAGES = {
    'DATA_LOADED': "Data loaded successfully",
    'FILTERS_APPLIED': "Filters applied",
    'WATCHLIST_CREATED': "Watchlist created successfully",
    'ALERT_TRIGGERED': "New alert triggered",
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
    
    # Check data sources are configured
    assert GOOGLE_SHEET_ID != "YOUR_SHEET_ID_HERE", "Please update GOOGLE_SHEET_ID with your actual sheet ID"
    
    print("✅ Constants validated successfully")

# Run validation when module loads
if __name__ == "__main__":
    validate_constants()
    print("\nM.A.N.T.R.A. Constants loaded")
    print(f"Data Source: Google Sheets ID {GOOGLE_SHEET_ID[:10]}...")
    print(f"Factors: {list(FACTOR_WEIGHTS.keys())}")
    print(f"Signal levels: {list(SIGNAL_LEVELS.keys())}")
    print("\n⚠️  IMPORTANT: Update GOOGLE_SHEET_ID with your actual Google Sheets ID!")
