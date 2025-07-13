"""
constants.py - M.A.N.T.R.A. Configuration Hub
===========================================
All configuration in one place - simple, clear, maintainable
"""

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================

# IMPORTANT: Update this with your actual Google Sheets ID!
GOOGLE_SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"

# Sheet GIDs (tab IDs from Google Sheets)
SHEET_CONFIGS = {
    "watchlist": {
        "gid": "2026492216",
        "name": "ALL STOCKS 2025 Watchlist"
    },
    "sector": {
        "gid": "140104095", 
        "name": "ALL STOCKS 2025 Sector Analysis"
    }
}

# ============================================================================
# TRADING SIGNALS
# ============================================================================

# Signal thresholds (composite score ranges)
SIGNAL_LEVELS = {
    "BUY": 80,      # Score >= 80
    "WATCH": 65,    # Score >= 65
    "AVOID": 35     # Score < 35
}

# Factor weights for composite scoring (must sum to 1.0)
FACTOR_WEIGHTS = {
    "momentum": 0.30,      # 30% - Recent price performance
    "value": 0.25,         # 25% - Valuation metrics
    "technical": 0.20,     # 20% - Technical indicators
    "volume": 0.15,        # 15% - Volume activity
    "fundamentals": 0.10   # 10% - EPS, growth
}

# ============================================================================
# THRESHOLDS
# ============================================================================

# Momentum thresholds
MOMENTUM_THRESHOLDS = {
    "STRONG": {"1d": 3, "7d": 7, "30d": 15},
    "MODERATE": {"1d": 1, "7d": 3, "30d": 5},
    "WEAK": {"1d": -1, "7d": -3, "30d": -5}
}

# Volume thresholds
VOLUME_THRESHOLDS = {
    "SPIKE": 3.0,      # 3x average volume
    "HIGH": 1.5,       # 1.5x average
    "NORMAL": 1.0,     # Normal volume
    "LOW": 0.5         # Half average
}

# Risk levels
RISK_LEVELS = {
    "LOW": (0, 40),
    "MEDIUM": (40, 70),
    "HIGH": (70, 100)
}

# PE ranges
PE_RANGES = {
    "UNDERVALUED": (0, 15),
    "FAIR": (15, 25),
    "OVERVALUED": (25, 40),
    "EXPENSIVE": (40, float('inf'))
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Color scheme
SIGNAL_COLORS = {
    'BUY': '#00d26a',       # Green
    'WATCH': '#ffa500',     # Orange
    'AVOID': '#ff4b4b',     # Red
    'NEUTRAL': '#808080'    # Gray
}

# Chart colors
CHART_COLORS = {
    'positive': '#00d26a',
    'negative': '#ff4b4b',
    'neutral': '#808080',
    'background': '#0e1117',
    'grid': '#1e2329'
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Cache settings
CACHE_DURATION_MINUTES = 5  # Refresh data every 5 minutes

# Display limits
MAX_DISPLAY_ROWS = 100
MAX_WATCHLIST_SIZE = 50
MAX_ALERTS = 20

# ============================================================================
# DATA QUALITY
# ============================================================================

# Minimum data requirements
MIN_REQUIRED_COLUMNS = [
    'ticker', 'price', 'company_name', 'sector',
    'ret_1d', 'ret_7d', 'ret_30d', 'volume_1d'
]

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "MIN_ROWS": 100,
    "MAX_NULL_PERCENT": 30,
    "MIN_PRICE": 0.01,
    "MAX_PRICE": 1000000
}

# ============================================================================
# MARKET REGIMES
# ============================================================================

MARKET_REGIMES = {
    "BULL": {
        "breadth": 70,      # >70% advancing
        "return": 5,        # >5% average return
        "weights": {"momentum": 0.4, "technical": 0.3, "volume": 0.2, "value": 0.1}
    },
    "BEAR": {
        "breadth": 30,      # <30% advancing
        "return": -5,       # <-5% average return
        "weights": {"value": 0.4, "fundamentals": 0.3, "technical": 0.2, "momentum": 0.1}
    },
    "SIDEWAYS": {
        "breadth": 50,      # ~50% advancing
        "return": 0,        # -2% to +2%
        "weights": {"value": 0.3, "technical": 0.3, "fundamentals": 0.2, "momentum": 0.2}
    }
}
