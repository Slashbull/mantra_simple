"""
constants.py - M.A.N.T.R.A. Configuration & Thresholds
======================================================
All magic numbers, thresholds, and configuration in ONE place.
This is the control center - tune the system by adjusting these values.
FINAL VERSION - Production Ready
"""

from typing import Dict, List, Tuple

# ==============================================================================
# DATA SOURCE CONFIGURATION
# ==============================================================================

# Google Sheets URLs or CSV URLs (replace with your actual URLs)
DATA_SOURCES = {
    'WATCHLIST': 'YOUR_WATCHLIST_GOOGLE_SHEET_URL_OR_CSV_URL',
    'SECTOR_ANALYSIS': 'YOUR_SECTOR_ANALYSIS_GOOGLE_SHEET_URL_OR_CSV_URL',
    'STOCK_RETURNS': 'YOUR_STOCK_RETURNS_GOOGLE_SHEET_URL_OR_CSV_URL',  # If separate
}

# For Google Sheets, you can use:
# https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=SHEET_GID

# Cache settings
CACHE_DURATION_MINUTES = 15  # Refresh data every 15 minutes
ENABLE_CACHING = True

# ==============================================================================
# SIGNAL THRESHOLDS - Core Decision Making
# ==============================================================================

# Signal assignment thresholds (composite score based)
SIGNAL_THRESHOLDS = {
    'BUY': 85,      # Score >= 85 = Strong Buy
    'WATCH': 70,    # Score >= 70 = Watch
    'AVOID': 50,    # Score < 50 = Avoid
}

# Individual component score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    'MOMENTUM': 0.30,    # 30% - Recent performance
    'VALUE': 0.25,       # 25% - Valuation metrics
    'TECHNICAL': 0.20,   # 20% - Technical indicators
    'VOLUME': 0.15,      # 15% - Volume patterns
    'SECTOR': 0.10,      # 10% - Sector strength
}

# ==============================================================================
# MOMENTUM THRESHOLDS
# ==============================================================================

# Return thresholds for momentum scoring (in %)
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

# Momentum time period weights
MOMENTUM_PERIOD_WEIGHTS = {
    '1D': 0.10,   # 10% weight to 1-day
    '3D': 0.15,   # 15% weight to 3-day
    '7D': 0.20,   # 20% weight to 7-day
    '30D': 0.25,  # 25% weight to 30-day
    '3M': 0.30,   # 30% weight to 3-month
}

# ==============================================================================
# VALUE THRESHOLDS
# ==============================================================================

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

# ==============================================================================
# TECHNICAL THRESHOLDS
# ==============================================================================

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

# ==============================================================================
# VOLUME THRESHOLDS
# ==============================================================================

VOLUME_THRESHOLDS = {
    'SPIKE_MULTIPLIER': 3.0,         # Volume > 3x average = Spike
    'HIGH_MULTIPLIER': 2.0,          # Volume > 2x average = High
    'NORMAL_RANGE': (0.7, 1.3),      # 70% to 130% of average
    'LOW_MULTIPLIER': 0.5,           # Volume < 50% average = Low
    'DRY_MULTIPLIER': 0.3,           # Volume < 30% average = Dry
    
    'VOLUME_TREND': {
        'ACCUMULATION': 1.5,         # Rising volume > 1.5x trend
        'DISTRIBUTION': 0.7,         # Falling volume < 0.7x trend
    }
}

# Relative Volume (RVOL) thresholds
RVOL_THRESHOLDS = {
    'EXTREME': 5.0,    # RVOL > 5 = Extreme activity
    'HIGH': 2.0,       # RVOL > 2 = High activity
    'NORMAL': 1.0,     # RVOL ~1 = Normal
    'LOW': 0.5,        # RVOL < 0.5 = Low activity
}

# ==============================================================================
# ANOMALY DETECTION THRESHOLDS
# ==============================================================================

ANOMALY_THRESHOLDS = {
    'PRICE_SPIKE': {
        'INTRADAY': 7,              # >7% move in a day = spike
        'MULTI_DAY': 15,            # >15% move in 3 days
        'WEEKLY': 20,               # >20% move in a week
    },
    'VOLUME_SPIKE': {
        'MULTIPLIER': 5,            # >5x average volume
        'SUSTAINED_DAYS': 3,        # High volume for 3+ days
    },
    'EPS_SURPRISE': {
        'POSITIVE': 20,             # EPS beat by >20%
        'NEGATIVE': -20,            # EPS miss by >20%
    },
    '52W_PROXIMITY': {
        'HIGH_THRESHOLD': 95,       # Within 5% of 52w high
        'LOW_THRESHOLD': 5,         # Within 5% of 52w low
    },
    'STD_DEV_MULTIPLIER': 2.5,      # 2.5 standard deviations
}

# ==============================================================================
# SECTOR ANALYSIS THRESHOLDS
# ==============================================================================

SECTOR_THRESHOLDS = {
    'ROTATION_SIGNAL': {
        'STRONG_INFLOW': 10,        # Sector gain >10% vs market
        'INFLOW': 5,                # Sector gain >5% vs market
        'NEUTRAL': (-2, 2),         # Within 2% of market
        'OUTFLOW': -5,              # Sector loss >5% vs market
        'STRONG_OUTFLOW': -10,      # Sector loss >10% vs market
    },
    'MOMENTUM_RANK': {
        'TOP_QUINTILE': 80,         # Top 20% sectors
        'SECOND_QUINTILE': 60,      # Next 20%
        'MIDDLE': 40,               # Middle 20%
        'FOURTH_QUINTILE': 20,      # Bottom 40%
        'BOTTOM_QUINTILE': 0,       # Bottom 20%
    },
    'MIN_STOCKS_FOR_SIGNAL': 5,     # Need 5+ stocks for valid sector signal
}

# ==============================================================================
# MARKET REGIME THRESHOLDS
# ==============================================================================

REGIME_THRESHOLDS = {
    'BREADTH': {
        'STRONG_BULLISH': 70,       # >70% stocks advancing
        'BULLISH': 55,              # >55% stocks advancing
        'NEUTRAL': 45,              # 45-55% advancing
        'BEARISH': 30,              # <45% advancing
        'STRONG_BEARISH': 20,       # <30% advancing
    },
    'VOLATILITY': {
        'LOW': 15,                  # VIX equivalent <15
        'NORMAL': 25,               # VIX equivalent 15-25
        'HIGH': 35,                 # VIX equivalent 25-35
        'EXTREME': 50,              # VIX equivalent >35
    },
    'TREND_STRENGTH': {
        'STRONG_UP': 70,            # Trend score >70
        'UP': 55,                   # Trend score >55
        'SIDEWAYS': 45,             # Trend score 45-55
        'DOWN': 30,                 # Trend score <45
        'STRONG_DOWN': 15,          # Trend score <30
    }
}

# Regime-based weight adjustments
REGIME_WEIGHTS = {
    'MOMENTUM_MARKET': {
        'MOMENTUM': 0.40,
        'VALUE': 0.15,
        'TECHNICAL': 0.25,
        'VOLUME': 0.15,
        'SECTOR': 0.05,
    },
    'VALUE_MARKET': {
        'MOMENTUM': 0.15,
        'VALUE': 0.40,
        'TECHNICAL': 0.20,
        'VOLUME': 0.15,
        'SECTOR': 0.10,
    },
    'VOLATILE_MARKET': {
        'MOMENTUM': 0.20,
        'VALUE': 0.20,
        'TECHNICAL': 0.30,
        'VOLUME': 0.20,
        'SECTOR': 0.10,
    },
    'BALANCED_MARKET': SCORE_WEIGHTS,  # Use default weights
}

# ==============================================================================
# RISK ASSESSMENT THRESHOLDS
# ==============================================================================

RISK_THRESHOLDS = {
    'VOLATILITY': {
        'LOW': 20,                  # <20% annualized vol
        'MEDIUM': 35,               # 20-35% vol
        'HIGH': 50,                 # 35-50% vol
        'EXTREME': 70,              # >50% vol
    },
    'DRAWDOWN': {
        'ACCEPTABLE': 10,           # <10% from high
        'MODERATE': 20,             # 10-20% from high
        'SIGNIFICANT': 30,          # 20-30% from high
        'SEVERE': 40,               # >30% from high
    },
    'LIQUIDITY': {
        'HIGH': 1000000,           # >10L daily volume
        'MEDIUM': 100000,          # 1L-10L daily volume
        'LOW': 10000,              # 10K-1L daily volume
        'ILLIQUID': 0,             # <10K daily volume
    }
}

# ==============================================================================
# ALERT PRIORITIES
# ==============================================================================

ALERT_PRIORITIES = {
    'CRITICAL': 1,    # Immediate action required
    'HIGH': 2,        # Important, check soon
    'MEDIUM': 3,      # Noteworthy
    'LOW': 4,         # Informational
}

ALERT_TYPES = {
    'NEW_BUY_SIGNAL': 'CRITICAL',
    'BREAKOUT': 'HIGH',
    'VOLUME_SPIKE': 'HIGH',
    'SECTOR_ROTATION': 'MEDIUM',
    'EPS_SURPRISE': 'MEDIUM',
    'APPROACHING_TARGET': 'LOW',
    'RISK_WARNING': 'HIGH',
}

# ==============================================================================
# EPS TIERS CONFIGURATION
# ==============================================================================

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

# ==============================================================================
# PRICE TIERS CONFIGURATION
# ==============================================================================

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

# ==============================================================================
# MARKET CAP CATEGORIES
# ==============================================================================

MARKET_CAP_CATEGORIES = {
    'LARGE_CAP': {'min': 20000, 'label': 'Large Cap', 'color': '#0066cc'},
    'MID_CAP': {'min': 5000, 'label': 'Mid Cap', 'color': '#ff9900'},
    'SMALL_CAP': {'min': 500, 'label': 'Small Cap', 'color': '#339966'},
    'MICRO_CAP': {'min': 0, 'label': 'Micro Cap', 'color': '#cc3366'},
}

# ==============================================================================
# UI CONFIGURATION
# ==============================================================================

UI_CONFIG = {
    'THEME': 'dark',
    'REFRESH_INTERVAL': 300,        # Auto-refresh every 5 minutes
    'MAX_ROWS_DISPLAY': 100,        # Max rows in main table
    'CHART_HEIGHT': 400,            # Default chart height in pixels
    'SHOW_TOOLTIPS': True,
    'ANIMATE_CHARTS': True,
    'DEFAULT_TAB': 'overview',
}

# Signal colors for UI
SIGNAL_COLORS = {
    'BUY': '#00d26a',       # Green
    'WATCH': '#ffa500',     # Orange
    'AVOID': '#ff4b4b',     # Red
    'NEUTRAL': '#808080',   # Gray
}

# ==============================================================================
# FILTERS CONFIGURATION
# ==============================================================================

DEFAULT_FILTERS = {
    'SIGNAL': 'ALL',
    'SECTOR': 'ALL',
    'MARKET_CAP': 'ALL',
    'PRICE_RANGE': 'ALL',
    'MIN_VOLUME': 10000,
    'SHOW_ONLY_ALERTS': False,
}

PRICE_RANGE_FILTERS = {
    'ALL': (0, float('inf')),
    'PENNY': (0, 50),
    'LOW': (50, 250),
    'MID': (250, 1000),
    'HIGH': (1000, 5000),
    'PREMIUM': (5000, float('inf')),
}

# ==============================================================================
# PERFORMANCE BENCHMARKS
# ==============================================================================

PERFORMANCE_BENCHMARKS = {
    'TARGET_LOAD_TIME': 3.0,        # Seconds
    'TARGET_REFRESH_TIME': 2.0,     # Seconds
    'TARGET_CALCULATION_TIME': 1.0, # Seconds
    'MIN_DATA_QUALITY': 0.95,       # 95% data completeness required
}

# ==============================================================================
# ERROR MESSAGES
# ==============================================================================

ERROR_MESSAGES = {
    'DATA_LOAD_FAILED': "Unable to load data. Please check your internet connection.",
    'INVALID_DATA': "Data validation failed. Some information may be incorrect.",
    'CALCULATION_ERROR': "Error in calculations. Showing cached results.",
    'NO_DATA': "No data available for the selected filters.",
    'SHEET_NOT_FOUND': "Data source not found. Please check configuration.",
}

# ==============================================================================
# SUCCESS MESSAGES
# ==============================================================================

SUCCESS_MESSAGES = {
    'DATA_LOADED': "Data loaded successfully",
    'FILTERS_APPLIED': "Filters applied",
    'WATCHLIST_CREATED': "Watchlist created successfully",
    'ALERT_TRIGGERED': "New alert triggered",
}

# ==============================================================================
# COLUMN MAPPINGS (for handling different column names)
# ==============================================================================

COLUMN_MAPPINGS = {
    'PRICE': ['price', 'close', 'last_price', 'current_price'],
    'VOLUME': ['volume_1d', 'volume', 'vol', 'daily_volume'],
    'PE': ['pe', 'pe_ratio', 'price_earnings'],
    'EPS': ['eps_current', 'eps', 'earnings_per_share'],
    'MARKET_CAP': ['market_cap', 'mcap', 'mkt_cap'],
}

# ==============================================================================
# VALIDATION RULES
# ==============================================================================

VALIDATION_RULES = {
    'PRICE': {'min': 0, 'max': 1000000},
    'PE': {'min': -100, 'max': 1000},
    'VOLUME': {'min': 0, 'max': 1e12},
    'RETURNS': {'min': -100, 'max': 1000},
    'EPS': {'min': -1000, 'max': 1000},
}

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================
