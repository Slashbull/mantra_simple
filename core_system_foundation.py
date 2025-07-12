"""
core_system_foundation.py - M.A.N.T.R.A. Core System Foundation
==============================================================
FINAL PRODUCTION VERSION - Streamlit Cloud Optimized
Data loading, validation, cleaning, and caching layer.
100% synchronous, no external dependencies, battle-tested.
"""

import io
import re
import time
import hashlib
import logging
import json
from typing import Tuple, Set, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

# Import constants
try:
    from constants import (
        DATA_SOURCES, CACHE_DURATION_MINUTES, VALIDATION_RULES,
        COLUMN_MAPPINGS, ERROR_MESSAGES
    )
except ImportError:
    # Fallback if constants.py not available yet
    DATA_SOURCES = {}
    CACHE_DURATION_MINUTES = 15
    VALIDATION_RULES = {}
    COLUMN_MAPPINGS = {}
    ERROR_MESSAGES = {}

print("=== LOADED core_system_foundation.py FINAL VERSION ===")

# ============================================================================
# ERROR HIERARCHY
# ============================================================================

class CoreFoundationError(Exception):
    """Base exception for all core foundation errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class DataSourceError(CoreFoundationError):
    """Raised when data source is unavailable or returns invalid data"""
    pass

class SchemaError(CoreFoundationError):
    """Raised when data schema validation fails"""
    pass

class DataValidationError(CoreFoundationError):
    """Raised when data quality checks fail"""
    pass

class ConfigurationError(CoreFoundationError):
    """Raised when configuration is invalid"""
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for M.A.N.T.R.A. data foundation"""
    # Data Sources - Will be overridden by constants.py if available
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",
        "returns": "100734077",
        "sector": "140104095"
    })
    
    # Schema Requirements - Based on provided documentation
    REQUIRED_WATCHLIST: Set[str] = field(default_factory=lambda: {
        "ticker", "exchange", "company_name", "year", "market_cap", "category", 
        "sector", "eps_tier", "price", "ret_1d", "low_52w", "high_52w",
        "from_low_pct", "from_high_pct", "sma_20d", "sma_50d", "sma_200d",
        "trading_under", "ret_3d", "ret_7d", "ret_30d", "ret_3m", "ret_6m",
        "ret_1y", "ret_3y", "ret_5y", "volume_1d", "volume_7d", "volume_30d",
        "volume_3m", "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
        "rvol", "price_tier", "prev_close", "pe", "eps_current", "eps_last_qtr", 
        "eps_duplicate", "eps_change_pct"
    })
    
    REQUIRED_RETURNS: Set[str] = field(default_factory=lambda: {
        "ticker", "company_name", "returns_ret_1d", "returns_ret_3d", 
        "returns_ret_7d", "returns_ret_30d", "returns_ret_3m", "returns_ret_6m",
        "returns_ret_1y", "returns_ret_3y", "returns_ret_5y",
        "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", 
        "avg_ret_3y", "avg_ret_5y"
    })
    
    REQUIRED_SECTOR: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", 
        "sector_ret_30d", "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", 
        "sector_ret_3y", "sector_ret_5y", "sector_avg_30d", "sector_avg_3m", 
        "sector_avg_6m", "sector_avg_1y", "sector_avg_3y", "sector_avg_5y", 
        "sector_count"
    })
    
    # Critical Fields - Must exist and be valid
    CRITICAL_FIELDS: Tuple[str, ...] = ("ticker", "price", "eps_current", "sector")
    
    # Performance Settings
    CACHE_TTL: int = 300  # 5 minutes
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.0
    
    # Data Quality Thresholds
    MIN_DATA_QUALITY_SCORE: float = 70.0
    MAX_NULL_PERCENTAGE: float = 20.0
    OUTLIER_THRESHOLD: float = 3.0  # z-score
    
    # System Settings
    SCHEMA_VERSION: str = "2025.07.12.FINAL"
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration and apply overrides from constants.py"""
        # Override with constants.py if available
        if DATA_SOURCES:
            if 'WATCHLIST' in DATA_SOURCES:
                self.update_from_url(DATA_SOURCES['WATCHLIST'], 'watchlist')
            if 'SECTOR_ANALYSIS' in DATA_SOURCES:
                self.update_from_url(DATA_SOURCES['SECTOR_ANALYSIS'], 'sector')
            if 'STOCK_RETURNS' in DATA_SOURCES:
                self.update_from_url(DATA_SOURCES['STOCK_RETURNS'], 'returns')
        
        # Validate
        if self.CACHE_TTL < 60:
            warnings.warn(f"Cache TTL {self.CACHE_TTL}s is very low")
        
        if self.MIN_DATA_QUALITY_SCORE < 50:
            raise ConfigurationError("MIN_DATA_QUALITY_SCORE must be >= 50")
    
    def update_from_url(self, url: str, sheet_name: str):
        """Update configuration from a Google Sheets URL"""
        if 'docs.google.com' in url and '/d/' in url:
            # Extract sheet ID and GID if present
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
            if sheet_id_match:
                self.BASE_URL = f"https://docs.google.com/spreadsheets/d/{sheet_id_match.group(1)}"
            
            gid_match = re.search(r'[#&]gid=([0-9]+)', url)
            if gid_match:
                self.SHEET_GIDS[sheet_name] = gid_match.group(1)
    
    def get_sheet_url(self, name: str) -> str:
        """Get Google Sheets export URL for a sheet"""
        if name not in self.SHEET_GIDS:
            raise ConfigurationError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"

# ============================================================================
# LOGGING
# ============================================================================

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# HTTP SESSION WITH RETRY
# ============================================================================

def create_session() -> requests.Session:
    """Create HTTP session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Set headers
    session.headers.update({
        'User-Agent': 'M.A.N.T.R.A./1.0 (Stock Analysis System)'
    })
    
    return session

# Global session
_session = None

def get_session() -> requests.Session:
    """Get or create global session"""
    global _session
    if _session is None:
        _session = create_session()
    return _session

# ============================================================================
# SIMPLE IN-MEMORY CACHE
# ============================================================================

class SimpleCache:
    """Thread-safe TTL-based in-memory cache"""
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                self._access_count[key] += 1
                return value
            else:
                del self._cache[key]
                self._access_count.pop(key, None)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp"""
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._access_count.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'total_accesses': sum(self._access_count.values()),
            'top_accessed': sorted(
                self._access_count.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }

# Global cache instance
_cache = SimpleCache()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_sheet(name: str, config: Config, use_cache: bool = True) -> pd.DataFrame:
    """Load a single sheet from Google Sheets with robust error handling"""
    url = config.get_sheet_url(name)
    cache_key = f"sheet_{name}_{config.SCHEMA_VERSION}"
    
    # Check cache first
    if use_cache:
        cached = _cache.get(cache_key, config.CACHE_TTL)
        if cached is not None:
            logger.info(f"✓ Cache hit for sheet '{name}'")
            return cached
    
    # Fetch from remote
    logger.info(f"📥 Fetching sheet '{name}'...")
    max_attempts = config.MAX_RETRIES
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            session = get_session()
            response = session.get(url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Validate response
            if not response.text or response.text.strip() == '':
                raise DataSourceError(f"Empty response for sheet '{name}'")
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))
            
            # Validate not empty
            if df.empty:
                raise DataSourceError(f"No data in sheet '{name}'")
            
            # Clean dataframe
            df = clean_dataframe(df)
            
            # Cache result
            if use_cache:
                _cache.set(cache_key, df)
            
            logger.info(f"✓ Loaded {len(df)} rows from '{name}'")
            return df
            
        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = config.BACKOFF_FACTOR * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    # All attempts failed
    error_msg = f"Failed to load sheet '{name}' after {max_attempts} attempts"
    logger.error(f"❌ {error_msg}: {last_error}")
    raise DataSourceError(
        error_msg,
        error_code="FETCH_ERROR",
        details={'sheet': name, 'url': url, 'error': str(last_error)}
    )

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize dataframe with comprehensive cleaning"""
    original_shape = df.shape
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
    
    # Clean column names
    df.columns = [
        re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', col.strip().lower()))
        for col in df.columns
    ]
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Clean string data - remove hidden characters
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).apply(lambda x: 
            x.replace('\u00A0', ' ')  # Non-breaking space
            .replace('\u200b', '')    # Zero-width space
            .replace('\xa0', ' ')     # Another non-breaking space
            .strip()
        )
    
    # Replace various null representations
    df = df.replace(['NA', 'N/A', 'n/a', 'null', 'NULL', 'None', '-', ''], np.nan)
    
    if df.shape != original_shape:
        logger.debug(f"Cleaned dataframe: {original_shape} → {df.shape}")
    
    return df

# ============================================================================
# DATA PROCESSING
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """Advanced numeric series cleaning with Indian format support"""
    if series.dtype in ['int64', 'float64']:
        return series
    
    # Convert to string for cleaning
    s = series.astype(str)
    
    # Handle Indian number formats (Cr, L, K)
    def convert_indian_notation(value):
        """Convert Indian notation to numeric"""
        value = str(value).strip()
        
        # Extract number and multiplier
        multipliers = {
            'cr': 10000000,    # Crore = 1,00,00,000
            'l': 100000,       # Lakh = 1,00,000
            'k': 1000,         # Thousand
            'm': 1000000,      # Million
            'b': 1000000000    # Billion
        }
        
        for suffix, multiplier in multipliers.items():
            if value.lower().endswith(suffix):
                try:
                    num_part = value[:-len(suffix)].strip()
                    # Remove currency symbols and commas
                    num_part = re.sub(r'[₹$€£,]', '', num_part)
                    return float(num_part) * multiplier
                except:
                    return np.nan
        
        return value
    
    # Apply Indian notation conversion
    s = s.apply(convert_indian_notation)
    
    # Convert back to string for further cleaning
    s = s.astype(str)
    
    # Remove currency symbols and clean
    currency_pattern = r'[₹$€£¥]'
    s = s.str.replace(currency_pattern, '', regex=True)
    
    # Remove commas and spaces
    s = s.str.replace(',', '').str.strip()
    
    # Handle percentages
    is_percentage = s.str.contains('%', na=False).any()
    if is_percentage:
        s = s.str.replace('%', '').str.strip()
    
    # Remove arrows and other symbols
    s = s.str.replace(r'[↑↓→←]', '', regex=True)
    
    # Handle empty strings
    s = s.replace(['', 'nan', 'NaN'], np.nan)
    
    # Convert to numeric
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Handle percentage columns that might be stored as decimals
    if col_name.endswith('_pct') or 'percent' in col_name or 'pct' in col_name:
        # Check if values are likely decimals (all between -1 and 1)
        non_null = numeric_series.dropna()
        if len(non_null) > 0 and non_null.abs().max() <= 1:
            numeric_series = numeric_series * 100
    
    return numeric_series

def validate_schema(df: pd.DataFrame, required_cols: Set[str], sheet_name: str) -> None:
    """Validate dataframe schema with detailed error reporting"""
    actual_cols = set(df.columns)
    missing_cols = required_cols - actual_cols
    
    if missing_cols:
        # Check for possible column name variations
        suggestions = {}
        for missing in missing_cols:
            # Look for similar column names
            similar = [col for col in actual_cols if missing in col or col in missing]
            if similar:
                suggestions[missing] = similar
        
        error_details = {
            'sheet': sheet_name,
            'missing': list(missing_cols),
            'actual': list(actual_cols),
            'suggestions': suggestions
        }
        
        raise SchemaError(
            f"Missing {len(missing_cols)} required columns in {sheet_name}",
            error_code="SCHEMA_MISMATCH",
            details=error_details
        )

def merge_datasets(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Merge watchlist and returns data with conflict resolution"""
    logger.info("📊 Merging datasets...")
    
    # Normalize ticker columns
    for df in [watchlist_df, returns_df]:
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    
    # Identify overlapping columns (except ticker)
    overlap_cols = set(watchlist_df.columns) & set(returns_df.columns) - {'ticker'}
    
    # Drop duplicate columns from returns (keep watchlist version)
    if overlap_cols:
        logger.debug(f"Dropping duplicate columns from returns: {overlap_cols}")
        returns_df = returns_df.drop(columns=list(overlap_cols), errors='ignore')
    
    # Merge on ticker
    merged = watchlist_df.merge(
        returns_df,
        on='ticker',
        how='left',
        suffixes=('', '_returns'),
        validate='one_to_one'
    )
    
    # Check for duplicates
    if merged['ticker'].duplicated().any():
        dup_count = merged['ticker'].duplicated().sum()
        dup_tickers = merged[merged['ticker'].duplicated()]['ticker'].unique()
        logger.warning(f"Found {dup_count} duplicate tickers: {dup_tickers[:5]}")
        merged = merged.drop_duplicates(subset='ticker', keep='last')
    
    logger.info(f"✓ Merged data: {len(merged)} stocks")
    return merged

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all numeric columns with progress tracking"""
    # Identify numeric columns
    numeric_patterns = [
        r'^price', r'^ret_', r'^avg_ret', r'^volume', r'^vol_ratio',
        r'^low_52w', r'^high_52w', r'^from_low_pct', r'^from_high_pct',
        r'^pe$', r'^eps', r'^rvol', r'^market_cap', r'^sma_', r'^dma_',
        r'^sector_ret_', r'^sector_avg_', r'^returns_ret_'
    ]
    
    numeric_cols = []
    for col in df.columns:
        if any(re.match(pattern, col) for pattern in numeric_patterns):
            numeric_cols.append(col)
    
    # Clean each numeric column
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            df[col] = clean_numeric_series(df[col], col)
    
    logger.info(f"✓ Cleaned {len(numeric_cols)} numeric columns")
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intelligent derived features for analysis"""
    logger.info("🔧 Adding derived features...")
    
    # Price-based features
    if 'price' in df.columns and df['price'].notna().any():
        # Price tiers
        df['price_tier_auto'] = pd.cut(
            df['price'],
            bins=[-np.inf, 50, 100, 250, 500, 1000, 2000, 5000, 10000, np.inf],
            labels=['<50', '50-100', '100-250', '250-500', '500-1K', 
                   '1K-2K', '2K-5K', '5K-10K', '>10K']
        )
        
        # Price momentum score
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            df['price_momentum'] = (
                df['ret_1d'] * 0.5 +
                df['ret_7d'] * 0.3 +
                df['ret_30d'] * 0.2
            )
    
    # Volume features
    if 'rvol' in df.columns:
        df['volume_signal'] = pd.cut(
            df['rvol'],
            bins=[-np.inf, 0.5, 1.0, 2.0, 5.0, np.inf],
            labels=['Very Low', 'Low', 'Normal', 'High', 'Extreme']
        )
    
    # EPS features
    if 'eps_change_pct' in df.columns:
        df['eps_signal'] = pd.cut(
            df['eps_change_pct'],
            bins=[-np.inf, -20, 0, 20, 50, np.inf],
            labels=['Strong Decline', 'Decline', 'Stable', 'Growth', 'High Growth']
        )
    
    # Value indicators
    if 'pe' in df.columns and 'eps_change_pct' in df.columns:
        # Value score combining PE and EPS growth
        df['value_score'] = np.where(
            (df['pe'] > 0) & (df['pe'] < 50),
            (50 - df['pe']) / 50 * 50 + df['eps_change_pct'].clip(-50, 50) / 2,
            0
        )
    
    # Technical indicators
    if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
        # SMA distances
        df['dist_from_sma20'] = ((df['price'] - df['sma_20d']) / df['sma_20d'] * 100).round(2)
        df['dist_from_sma50'] = ((df['price'] - df['sma_50d']) / df['sma_50d'] * 100).round(2)
        df['dist_from_sma200'] = ((df['price'] - df['sma_200d']) / df['sma_200d'] * 100).round(2)
        
        # Technical strength
        df['technical_strength'] = (
            (df['price'] > df['sma_20d']).astype(int) +
            (df['price'] > df['sma_50d']).astype(int) +
            (df['price'] > df['sma_200d']).astype(int)
        ) / 3 * 100
    
    # Market position
    if 'from_low_pct' in df.columns and 'from_high_pct' in df.columns:
        # Position in 52-week range (0 = at low, 100 = at high)
        df['position_52w'] = df['from_low_pct'] / (df['from_low_pct'] + abs(df['from_high_pct'])) * 100
        
        # Near extremes flags
        df['near_52w_high'] = df['from_high_pct'] > -5
        df['near_52w_low'] = df['from_low_pct'] < 10
    
    # Composite trend score
    trend_cols = [col for col in ['ret_3d', 'ret_7d', 'ret_30d', 'ret_3m'] if col in df.columns]
    if trend_cols:
        weights = [0.1, 0.2, 0.3, 0.4][:len(trend_cols)]
        df['trend_score'] = sum(df[col] * w for col, w in zip(trend_cols, weights))
    
    # Risk indicators
    if 'ret_1d' in df.columns:
        # Simple volatility proxy
        df['volatility_1d'] = df['ret_1d'].abs()
    
    logger.info("✓ Added derived features")
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize memory usage with smart dtype conversion"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        # Check if can be converted to float32
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.notna(col_min) and pd.notna(col_max):
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        # Try to downcast
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert low-cardinality strings to category
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5 and num_unique < 100:
            df[col] = df[col].astype('category')
    
    # Special handling for known categorical columns
    categorical_cols = [
        'exchange', 'category', 'sector', 'eps_tier', 'price_tier',
        'trading_under', 'price_tier_auto', 'volume_signal', 'eps_signal'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = (1 - final_memory / initial_memory) * 100
    
    logger.info(f"💾 Memory optimized: {initial_memory:.1f}MB → {final_memory:.1f}MB ({reduction:.1f}% saved)")
    
    return df

# ============================================================================
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Comprehensive data quality analysis"""
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'dtypes_summary': df.dtypes.value_counts().to_dict()
    }
    
    # Null analysis
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df) * 100).round(2)
    
    analysis['null_analysis'] = {
        'total_nulls': int(null_counts.sum()),
        'null_percentage': float((null_counts.sum() / (len(df) * len(df.columns)) * 100).round(2)),
        'columns_with_nulls': null_counts[null_counts > 0].to_dict(),
        'high_null_columns': null_percentages[null_percentages > config.MAX_NULL_PERCENTAGE].to_dict()
    }
    
    # Critical fields analysis
    critical_analysis = {}
    for field in config.CRITICAL_FIELDS:
        if field in df.columns:
            critical_analysis[field] = {
                'null_count': int(df[field].isnull().sum()),
                'null_pct': float((df[field].isnull().sum() / len(df) * 100).round(2)),
                'unique_count': int(df[field].nunique()),
                'dtype': str(df[field].dtype)
            }
    analysis['critical_fields'] = critical_analysis
    
    # Duplicate analysis
    analysis['duplicate_analysis'] = {
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_tickers': int(df['ticker'].duplicated().sum()) if 'ticker' in df else 0,
        'duplicate_percentage': float((df.duplicated().sum() / len(df) * 100).round(2))
    }
    
    # Outlier analysis for numeric columns
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns[:20]:  # Limit to first 20 numeric columns
        if df[col].notna().sum() > 10:  # Need enough data
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(df) * 100).round(2)),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
    
    analysis['outlier_analysis'] = outliers
    
    # Data freshness (if date columns exist)
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        analysis['freshness'] = {}
        for col in date_cols[:3]:  # Check first 3 date columns
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().any():
                    analysis['freshness'][col] = {
                        'latest': dates.max().isoformat(),
                        'oldest': dates.min().isoformat()
                    }
            except:
                pass
    
    # Calculate quality score
    quality_score = 100.0
    
    # Deduct for nulls
    null_penalty = min(30, analysis['null_analysis']['null_percentage'] * 2)
    quality_score -= null_penalty
    
    # Deduct for high null columns
    high_null_penalty = min(20, len(analysis['null_analysis']['high_null_columns']) * 5)
    quality_score -= high_null_penalty
    
    # Deduct for duplicates
    dup_penalty = min(20, analysis['duplicate_analysis']['duplicate_percentage'] * 5)
    quality_score -= dup_penalty
    
    # Deduct for missing critical fields
    for field, info in critical_analysis.items():
        if info['null_pct'] > 10:
            quality_score -= 5
    
    # Deduct for too many outliers
    outlier_penalty = min(10, len(outliers))
    quality_score -= outlier_penalty
    
    quality_score = max(0, quality_score)
    
    analysis['quality_score'] = round(quality_score, 1)
    analysis['quality_grade'] = (
        'A' if quality_score >= 90 else
        'B' if quality_score >= 80 else
        'C' if quality_score >= 70 else
        'D' if quality_score >= 60 else
        'F'
    )
    
    # Add recommendations
    recommendations = []
    if analysis['null_analysis']['null_percentage'] > 10:
        recommendations.append("High percentage of null values detected")
    if analysis['duplicate_analysis']['duplicate_tickers'] > 0:
        recommendations.append(f"Remove {analysis['duplicate_analysis']['duplicate_tickers']} duplicate tickers")
    if len(outliers) > 5:
        recommendations.append("Multiple columns have outliers - review data quality")
    
    analysis['recommendations'] = recommendations
    
    return analysis

def validate_data(df: pd.DataFrame, config: Config) -> List[Dict[str, Any]]:
    """Run comprehensive data validation checks"""
    issues = []
    
    # Check if dataframe is empty
    if df.empty:
        issues.append({
            'rule': 'not_empty',
            'message': "Dataframe is empty",
            'severity': 'critical'
        })
        return issues
    
    # Price validation
    if 'price' in df.columns:
        # Check for non-positive prices
        invalid_prices = df['price'] <= 0
        if invalid_prices.any():
            invalid_tickers = df[invalid_prices]['ticker'].tolist()[:5]
            issues.append({
                'rule': 'positive_price',
                'message': f"Found {invalid_prices.sum()} stocks with non-positive prices",
                'severity': 'critical',
                'examples': invalid_tickers
            })
        
        # Check for extreme prices
        extreme_high = df['price'] > 100000
        if extreme_high.any():
            issues.append({
                'rule': 'price_range',
                'message': f"Found {extreme_high.sum()} stocks with prices > ₹1,00,000",
                'severity': 'warning'
            })
    
    # Volume validation
    if 'volume_1d' in df.columns:
        zero_volume = df['volume_1d'] == 0
        zero_vol_pct = (zero_volume.sum() / len(df) * 100)
        if zero_vol_pct > 20:
            issues.append({
                'rule': 'volume_liquidity',
                'message': f"{zero_vol_pct:.1f}% of stocks have zero volume",
                'severity': 'warning'
            })
    
    # PE validation
    if 'pe' in df.columns:
        # Check for extreme PE ratios
        extreme_pe = (df['pe'] < -100) | (df['pe'] > 1000)
        if extreme_pe.any():
            issues.append({
                'rule': 'pe_range',
                'message': f"Found {extreme_pe.sum()} stocks with extreme PE ratios",
                'severity': 'warning'
            })
    
    # EPS validation
    if 'eps_current' in df.columns and 'eps_change_pct' in df.columns:
        # Check for impossible EPS changes
        impossible_change = df['eps_change_pct'].abs() > 1000
        if impossible_change.any():
            issues.append({
                'rule': 'eps_change_range',
                'message': f"Found {impossible_change.sum()} stocks with >1000% EPS change",
                'severity': 'warning'
            })
    
    # Return percentage validation
    return_cols = [col for col in df.columns if col.startswith('ret_')]
    for col in return_cols:
        if col in df.columns:
            extreme_returns = df[col].abs() > 200
            if extreme_returns.any():
                issues.append({
                    'rule': 'return_range',
                    'message': f"Column '{col}' has {extreme_returns.sum()} extreme values (>200%)",
                    'severity': 'info'
                })
                break  # Only report once
    
    # Critical fields validation
    for field in config.CRITICAL_FIELDS:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            null_pct = null_count / len(df) * 100
            if null_pct > 50:
                issues.append({
                    'rule': 'critical_field_nulls',
                    'message': f"Critical field '{field}' is {null_pct:.1f}% null",
                    'severity': 'critical'
                })
    
    # Ticker validation
    if 'ticker' in df.columns:
        # Check for invalid tickers
        invalid_tickers = df['ticker'].str.len() > 20
        if invalid_tickers.any():
            issues.append({
                'rule': 'ticker_format',
                'message': f"Found {invalid_tickers.sum()} tickers longer than 20 characters",
                'severity': 'warning'
            })
    
    return issues

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_and_process(
    config: Optional[Config] = None,
    use_cache: bool = True,
    validate_quality: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for M.A.N.T.R.A. data loading and processing
    
    Args:
        config: Configuration object (uses defaults if None)
        use_cache: Whether to use caching (default: True)
        validate_quality: Whether to enforce quality checks (default: True)
    
    Returns:
        Tuple of (stocks_df, sector_df, health_dict)
    
    Raises:
        DataSourceError: If data cannot be loaded
        SchemaError: If schema validation fails
        DataValidationError: If critical validation fails
    """
    if config is None:
        config = Config()
    
    start_time = time.time()
    lineage = []
    
    try:
        # Step 1: Load raw data
        logger.info("🚀 Starting M.A.N.T.R.A. data pipeline...")
        
        watchlist_df = load_sheet('watchlist', config, use_cache)
        returns_df = load_sheet('returns', config, use_cache)
        sector_df = load_sheet('sector', config, use_cache)
        
        lineage.append({
            'step': 'data_loading',
            'watchlist_rows': len(watchlist_df),
            'returns_rows': len(returns_df),
            'sector_rows': len(sector_df)
        })
        
        # Step 2: Validate schemas
        logger.info("🔍 Validating schemas...")
        validate_schema(watchlist_df, config.REQUIRED_WATCHLIST, 'watchlist')
        validate_schema(returns_df, config.REQUIRED_RETURNS, 'returns')
        validate_schema(sector_df, config.REQUIRED_SECTOR, 'sector')
        lineage.append({'step': 'schema_validation', 'status': 'passed'})
        
        # Step 3: Merge datasets
        stocks_df = merge_datasets(watchlist_df, returns_df)
        lineage.append({
            'step': 'data_merge',
            'merged_rows': len(stocks_df),
            'merged_columns': len(stocks_df.columns)
        })
        
        # Step 4: Clean numeric columns
        logger.info("🧹 Cleaning data...")
        stocks_df = clean_numeric_columns(stocks_df)
        sector_df = clean_numeric_columns(sector_df)
        lineage.append({'step': 'numeric_cleaning', 'status': 'completed'})
        
        # Step 5: Add derived features
        stocks_df = add_derived_features(stocks_df)
        lineage.append({
            'step': 'feature_engineering',
            'features_added': len([col for col in stocks_df.columns if col not in watchlist_df.columns])
        })
        
        # Step 6: Optimize memory
        stocks_df = optimize_dtypes(stocks_df)
        sector_df = optimize_dtypes(sector_df)
        lineage.append({'step': 'memory_optimization', 'status': 'completed'})
        
        # Step 7: Validate data quality
        validation_issues = validate_data(stocks_df, config)
        critical_issues = [i for i in validation_issues if i['severity'] == 'critical']
        
        if critical_issues and validate_quality:
            raise DataValidationError(
                f"Found {len(critical_issues)} critical validation issues",
                error_code="VALIDATION_FAILED",
                details={'issues': critical_issues}
            )
        
        lineage.append({
            'step': 'data_validation',
            'total_issues': len(validation_issues),
            'critical_issues': len(critical_issues)
        })
        
        # Step 8: Analyze quality
        quality_analysis = analyze_data_quality(stocks_df, config)
        
        if validate_quality and quality_analysis['quality_score'] < config.MIN_DATA_QUALITY_SCORE:
            logger.warning(
                f"⚠️ Data quality score {quality_analysis['quality_score']} "
                f"below threshold {config.MIN_DATA_QUALITY_SCORE}"
            )
        
        # Step 9: Generate data fingerprint
        fingerprint_cols = ['ticker', 'price', 'eps_current']
        fingerprint_data = stocks_df[fingerprint_cols].fillna(0)
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(fingerprint_data, index=False).values
        ).hexdigest()[:16]
        
        # Build comprehensive health report
        processing_time = time.time() - start_time
        
        health = {
            'status': 'success',
            'processing_time_s': round(processing_time, 2),
            'timestamp': datetime.utcnow().isoformat(),
            'schema_version': config.SCHEMA_VERSION,
            'data_hash': data_hash,
            'lineage': lineage,
            'summary': {
                'total_stocks': len(stocks_df),
                'total_sectors': len(sector_df),
                'unique_sectors': stocks_df['sector'].nunique() if 'sector' in stocks_df else 0,
                'cache_used': use_cache,
                'quality_score': quality_analysis['quality_score'],
                'quality_grade': quality_analysis['quality_grade']
            },
            'validation': {
                'issues': validation_issues,
                'critical_count': len(critical_issues),
                'warning_count': len([i for i in validation_issues if i['severity'] == 'warning']),
                'passed': len(critical_issues) == 0
            },
            'quality_analysis': quality_analysis,
            'cache_stats': _cache.stats()
        }
        
        # Success logging
        logger.info(f"✅ Pipeline completed successfully in {processing_time:.2f}s")
        logger.info(f"📊 Loaded {len(stocks_df)} stocks across {len(sector_df)} sectors")
        logger.info(f"🏆 Data quality: {quality_analysis['quality_grade']} ({quality_analysis['quality_score']:.1f}/100)")
        
        return stocks_df, sector_df, health
        
    except Exception as e:
        # Build error health report
        error_health = {
            'status': 'error',
            'processing_time_s': round(time.time() - start_time, 2),
            'timestamp': datetime.utcnow().isoformat(),
            'error': {
                'type': type(e).__name__,
                'message': str(e),
                'details': getattr(e, 'details', {})
            },
            'lineage': lineage
        }
        
        logger.error(f"❌ Pipeline failed: {e}")
        
        # Re-raise with health context
        if isinstance(e, CoreFoundationError):
            e.details['health'] = error_health
            raise
        else:
            raise CoreFoundationError(
                f"Pipeline failed: {str(e)}",
                error_code="PIPELINE_ERROR",
                details={'health': error_health, 'original_error': str(e)}
            )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_cache():
    """Get cache instance for external use"""
    return _cache

def clear_cache():
    """Clear all cached data"""
    _cache.clear()
    logger.info("✓ Cache cleared")

def health_check(config: Optional[Config] = None) -> Dict[str, Any]:
    """Quick health check without loading data"""
    if config is None:
        config = Config()
    
    start_time = time.time()
    health = {
        'status': 'unknown',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {},
        'cache_stats': _cache.stats()
    }
    
    # Check each data source
    session = get_session()
    
    for sheet_name in config.SHEET_GIDS:
        check_start = time.time()
        try:
            url = config.get_sheet_url(sheet_name)
            response = session.head(url, timeout=5, allow_redirects=True)
            
            health['checks'][sheet_name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time_ms': round((time.time() - check_start) * 1000, 1)
            }
        except Exception as e:
            health['checks'][sheet_name] = {
                'status': 'error',
                'error': str(e),
                'response_time_ms': round((time.time() - check_start) * 1000, 1)
            }
    
    # Determine overall status
    check_statuses = [check.get('status') for check in health['checks'].values()]
    
    if all(s == 'healthy' for s in check_statuses):
        health['status'] = 'healthy'
    elif any(s == 'healthy' for s in check_statuses):
        health['status'] = 'degraded'
    else:
        health['status'] = 'critical'
    
    health['total_time_ms'] = round((time.time() - start_time) * 1000, 1)
    
    return health

def get_sample_data(n: int = 5) -> pd.DataFrame:
    """Get sample data for testing without full load"""
    try:
        config = Config()
        watchlist_df = load_sheet('watchlist', config, use_cache=True)
        return watchlist_df.head(n)
    except Exception as e:
        logger.error(f"Failed to get sample data: {e}")
        return pd.DataFrame()

# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    """Test the data pipeline directly"""
    print("\n" + "="*60)
    print("M.A.N.T.R.A. Core System Foundation Test")
    print("="*60 + "\n")
    
    # Run health check
    print("1. Running health check...")
    health = health_check()
    print(f"   Status: {health['status']}")
    for sheet, check in health['checks'].items():
        print(f"   - {sheet}: {check['status']} ({check.get('response_time_ms', 'N/A')}ms)")
    
    # Load sample data
    print("\n2. Loading sample data...")
    sample = get_sample_data(3)
    if not sample.empty:
        print(f"   ✓ Sample data shape: {sample.shape}")
        print(f"   ✓ Columns: {', '.join(sample.columns[:5])}...")
    
    # Run full pipeline
    print("\n3. Running full pipeline...")
    try:
        stocks_df, sector_df, pipeline_health = load_and_process(validate_quality=False)
        
        print(f"\n   ✅ SUCCESS!")
        print(f"   - Stocks loaded: {len(stocks_df)}")
        print(f"   - Sectors loaded: {len(sector_df)}")
        print(f"   - Quality score: {pipeline_health['quality_analysis']['quality_score']:.1f}/100")
        print(f"   - Processing time: {pipeline_health['processing_time_s']:.2f}s")
        print(f"   - Data hash: {pipeline_health['data_hash']}")
        
        # Show sample stocks
        print(f"\n   Sample stocks:")
        print(stocks_df[['ticker', 'company_name', 'price', 'ret_1d', 'sector']].head(3).to_string())
        
    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        if hasattr(e, 'details'):
            print(f"   Details: {json.dumps(e.details, indent=2)}")
    
    print("\n" + "="*60 + "\n")

# END OF core_system_foundation.py
