# core_system_foundation.py v5.0 (STREAMLIT CLOUD PRODUCTION READY)
"""
M.A.N.T.R.A. Core System Foundation - Streamlit Cloud Edition
=============================================================
Optimized for Streamlit Community Cloud deployment
- 100% synchronous (no async/await)
- In-memory caching with TTL
- Streamlit cache integration ready
- No external dependencies (Redis, etc.)
- Fast, reliable, production-ready
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

print("=== LOADED core_system_foundation.py v5.0 (STREAMLIT CLOUD EDITION) ===")

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
    """Configuration optimized for Streamlit Cloud"""
    # Data Sources
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",
        "returns": "100734077",
        "sector": "140104095"
    })
    
    # Schema Requirements
    REQUIRED_WATCHLIST: Set[str] = field(default_factory=lambda: {
        "ticker", "exchange", "company_name", "year", "market_cap", "category", "sector",
        "eps_tier", "price", "prev_close", "ret_1d", "low_52w", "high_52w",
        "from_low_pct", "from_high_pct", "sma_20d", "sma_50d", "sma_200d",
        "trading_under", "ret_3d", "ret_7d", "ret_30d", "ret_3m", "ret_6m",
        "ret_1y", "ret_3y", "ret_5y", "volume_1d", "volume_7d", "volume_30d",
        "volume_3m", "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
        "rvol", "price_tier", "eps_current", "eps_last_qtr", "eps_duplicate", "eps_change_pct"
    })
    REQUIRED_RETURNS: Set[str] = field(default_factory=lambda: {
        "ticker", "company_name",
        "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", "avg_ret_3y", "avg_ret_5y"
    })
    REQUIRED_SECTOR: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", "sector_ret_30d",
        "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", "sector_ret_3y", "sector_ret_5y",
        "sector_avg_30d", "sector_avg_3m", "sector_avg_6m", "sector_avg_1y",
        "sector_avg_3y", "sector_avg_5y", "sector_count"
    })
    
    # Critical Fields
    CRITICAL_FIELDS: Tuple[str, ...] = ("ticker", "price", "eps_current", "sector")
    
    # Performance Settings
    CACHE_TTL: int = 300  # seconds (5 minutes)
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.0
    
    # Data Quality Thresholds
    MIN_DATA_QUALITY_SCORE: float = 70.0
    MAX_NULL_PERCENTAGE: float = 20.0
    OUTLIER_THRESHOLD: float = 3.0  # z-score
    
    # System Settings
    SCHEMA_VERSION: str = "2025.07.11"
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.CACHE_TTL < 60:
            warnings.warn(f"Cache TTL {self.CACHE_TTL}s is very low, consider increasing")
        
        if self.MIN_DATA_QUALITY_SCORE < 50:
            raise ConfigurationError("MIN_DATA_QUALITY_SCORE must be >= 50")
    
    def get_sheet_url(self, name: str) -> str:
        """Get Google Sheets export URL for a sheet"""
        if name not in self.SHEET_GIDS:
            raise ConfigurationError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"

# ============================================================================
# LOGGING
# ============================================================================

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

# Global session (reused for efficiency)
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
    """Simple TTL-based in-memory cache"""
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp"""
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()

# Global cache instance
_cache = SimpleCache()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_sheet(name: str, config: Config, use_cache: bool = True) -> pd.DataFrame:
    """Load a single sheet from Google Sheets"""
    url = config.get_sheet_url(name)
    cache_key = f"sheet_{name}_{config.SCHEMA_VERSION}"
    
    # Check cache
    if use_cache:
        cached = _cache.get(cache_key, config.CACHE_TTL)
        if cached is not None:
            logger.info(f"Cache hit for sheet '{name}'")
            return cached
    
    # Fetch from remote
    logger.info(f"Fetching sheet '{name}' from {url}")
    try:
        session = get_session()
        response = session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean dataframe
        df = clean_dataframe(df)
        
        # Cache result
        if use_cache:
            _cache.set(cache_key, df)
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch sheet '{name}': {e}")
        raise DataSourceError(f"Cannot load sheet '{name}' from Google Sheets", 
                            error_code="FETCH_ERROR",
                            details={'sheet': name, 'error': str(e)})

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize dataframe columns"""
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    
    # Clean column names
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    
    # Remove hidden characters
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace("\u00A0", " ", regex=False)
    
    return df

# ============================================================================
# DATA PROCESSING
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """Clean and convert a numeric series"""
    # Convert to string and clean
    s = series.astype(str)
    
    # Remove currency symbols and units
    for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Remove non-ASCII characters
    s = s.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    
    # Handle empty strings
    s = s.replace('', 'NaN')
    
    # Convert to numeric
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Handle percentage columns
    if col_name.endswith('_pct') or '%' in series.astype(str).str.cat():
        return numeric_series
    
    # Auto-detect if values need scaling
    non_null = numeric_series.dropna()
    if len(non_null) > 0 and non_null.max() < 1 and non_null.min() >= 0:
        # Likely percentages stored as decimals
        return numeric_series * 100
    
    return numeric_series

def validate_schema(df: pd.DataFrame, required_cols: Set[str], sheet_name: str) -> None:
    """Validate dataframe has required columns"""
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise SchemaError(
            f"Missing columns in {sheet_name}: {missing_cols}",
            error_code="SCHEMA_MISMATCH",
            details={'sheet': sheet_name, 'missing': list(missing_cols)}
        )

def merge_datasets(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Merge watchlist and returns data"""
    # Drop duplicate columns
    returns_df = returns_df.drop(columns=['company_name'], errors='ignore')
    
    # Merge on ticker
    merged = watchlist_df.merge(
        returns_df,
        on='ticker',
        how='left',
        validate='one_to_one'
    )
    
    # Handle duplicates
    if merged['ticker'].duplicated().any():
        dup_count = merged['ticker'].duplicated().sum()
        logger.warning(f"Found {dup_count} duplicate tickers, keeping last")
        merged = merged.drop_duplicates(subset='ticker', keep='last')
    
    # Normalize ticker
    merged['ticker'] = merged['ticker'].astype(str).str.upper().str.strip()
    
    return merged

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all numeric columns in dataframe"""
    numeric_pattern = re.compile(
        r"^(price|prev_close|ret_|avg_ret|volume|vol_ratio|"
        r"low_52w|high_52w|from_low_pct|from_high_pct|pe|eps|rvol|"
        r"market_cap|sma_|dma_)"
    )
    
    for col in df.columns:
        if numeric_pattern.match(col):
            df[col] = clean_numeric_series(df[col], col)
    
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and indicators"""
    # Price tiers
    if 'price' in df and df['price'].notna().any():
        df['price_tier'] = pd.cut(
            df['price'],
            bins=[-np.inf, 100, 200, 500, 1000, 2000, 5000, np.inf],
            labels=['<100', '100-200', '200-500', '500-1K', '1K-2K', '2K-5K', '>5K']
        )
    
    # EPS tiers
    if 'eps_current' in df and df['eps_current'].notna().any():
        df['eps_tier'] = pd.cut(
            df['eps_current'],
            bins=[-np.inf, 5, 15, 35, 55, 75, 95, np.inf],
            labels=['<5', '5-15', '15-35', '35-55', '55-75', '75-95', '>95']
        )
    
    # Value indicators
    if 'pe' in df and 'eps_change_pct' in df:
        df['is_undervalued'] = (df['pe'] < 20) & (df['eps_change_pct'] > 25)
    
    # Volume indicators
    if 'rvol' in df:
        df['is_volume_spike'] = df['rvol'] > 2
    
    # Technical indicators
    if 'from_low_pct' in df:
        df['is_near_low'] = df['from_low_pct'] < 15
    
    if 'from_high_pct' in df:
        df['is_near_high'] = df['from_high_pct'] < 5
    
    # Trend strength
    if all(col in df for col in ['ret_3d', 'ret_7d', 'ret_30d']):
        df['trend_strength'] = (
            df['ret_3d'] * 0.5 +
            df['ret_7d'] * 0.3 +
            df['ret_30d'] * 0.2
        )
    
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert low-cardinality object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    memory_reduction = (1 - final_memory / initial_memory) * 100
    
    logger.info(f"Memory optimization: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                f"({memory_reduction:.1f}% reduction)")
    
    return df

# ============================================================================
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Analyze data quality and generate metrics"""
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
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
    
    # Duplicate analysis
    analysis['duplicate_analysis'] = {
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_tickers': int(df['ticker'].duplicated().sum()) if 'ticker' in df else 0
    }
    
    # Outlier analysis
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df and df[col].notna().sum() > 0:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outlier_count = (z_scores > config.OUTLIER_THRESHOLD).sum()
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(df) * 100).round(2)),
                        'mean': float(mean),
                        'std': float(std)
                    }
    analysis['outlier_analysis'] = outliers
    
    # Calculate overall quality score
    quality_score = 100.0
    quality_score -= min(50, analysis['null_analysis']['null_percentage'] * 2)
    quality_score -= min(20, len(analysis['null_analysis']['high_null_columns']) * 5)
    quality_score -= min(20, analysis['duplicate_analysis']['duplicate_tickers'] * 0.5)
    quality_score -= min(10, len(outliers) * 2)
    
    analysis['quality_score'] = max(0, quality_score)
    analysis['quality_grade'] = (
        'A' if quality_score >= 90 else
        'B' if quality_score >= 80 else
        'C' if quality_score >= 70 else
        'D' if quality_score >= 60 else
        'F'
    )
    
    return analysis

def validate_data(df: pd.DataFrame, config: Config) -> List[Dict[str, Any]]:
    """Run data validation checks"""
    issues = []
    
    # Price sanity check
    if 'price' in df:
        invalid_prices = (df['price'] <= 0) | (df['price'] > 1000000)
        if invalid_prices.any():
            issues.append({
                'rule': 'price_sanity',
                'message': f"Found {invalid_prices.sum()} stocks with invalid prices",
                'severity': 'critical'
            })
    
    # Volume check
    if 'volume_1d' in df:
        zero_volume = (df['volume_1d'] == 0).sum()
        zero_volume_pct = zero_volume / len(df) * 100
        if zero_volume_pct > 20:
            issues.append({
                'rule': 'volume_non_zero',
                'message': f"{zero_volume_pct:.1f}% of stocks have zero volume",
                'severity': 'warning'
            })
    
    # Critical fields check
    for field in config.CRITICAL_FIELDS:
        if field in df and df[field].isna().all():
            issues.append({
                'rule': 'critical_field_empty',
                'message': f"Critical field '{field}' is completely empty",
                'severity': 'critical'
            })
    
    return issues

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_and_process(
    config: Optional[Config] = None,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for data loading and processing
    
    Args:
        config: Configuration object (optional)
        use_cache: Whether to use caching (default: True)
    
    Returns:
        Tuple of (stocks_df, sector_df, health_dict)
    """
    if config is None:
        config = Config()
    
    start_time = time.time()
    lineage = []
    
    try:
        # Load sheets
        logger.info("Loading data sheets...")
        watchlist_df = load_sheet('watchlist', config, use_cache)
        returns_df = load_sheet('returns', config, use_cache)
        sector_df = load_sheet('sector', config, use_cache)
        lineage.append(f"Loaded sheets: watchlist={len(watchlist_df)}, returns={len(returns_df)}, sector={len(sector_df)}")
        
        # Validate schemas
        logger.info("Validating schemas...")
        validate_schema(watchlist_df, config.REQUIRED_WATCHLIST, 'watchlist')
        validate_schema(returns_df, config.REQUIRED_RETURNS, 'returns')
        validate_schema(sector_df, config.REQUIRED_SECTOR, 'sector')
        lineage.append("Schema validation passed")
        
        # Merge datasets
        logger.info("Merging datasets...")
        stocks_df = merge_datasets(watchlist_df, returns_df)
        lineage.append(f"Merged data: {len(stocks_df)} stocks")
        
        # Clean numeric columns
        logger.info("Cleaning numeric columns...")
        stocks_df = clean_numeric_columns(stocks_df)
        sector_df = clean_numeric_columns(sector_df)
        lineage.append("Numeric columns cleaned")
        
        # Add derived features
        logger.info("Adding derived features...")
        stocks_df = add_derived_features(stocks_df)
        lineage.append("Derived features added")
        
        # Optimize memory
        logger.info("Optimizing memory usage...")
        stocks_df = optimize_dtypes(stocks_df)
        sector_df = optimize_dtypes(sector_df)
        lineage.append("Memory optimized")
        
        # Validate data
        logger.info("Validating data quality...")
        validation_issues = validate_data(stocks_df, config)
        
        # Check for critical issues
        critical_issues = [i for i in validation_issues if i['severity'] == 'critical']
        if critical_issues:
            raise DataValidationError(
                "Critical validation failures detected",
                details={'issues': critical_issues}
            )
        
        # Analyze data quality
        quality_analysis = analyze_data_quality(stocks_df, config)
        
        # Check quality score
        if quality_analysis['quality_score'] < config.MIN_DATA_QUALITY_SCORE:
            logger.warning(
                f"Data quality score {quality_analysis['quality_score']:.1f} "
                f"below threshold {config.MIN_DATA_QUALITY_SCORE}"
            )
        
        # Generate data hash
        hash_input = pd.util.hash_pandas_object(stocks_df[['ticker', 'price']], index=False).values
        data_hash = hashlib.sha256(hash_input).hexdigest()[:16]
        
        # Build health report
        health = {
            'processing_time_s': time.time() - start_time,
            'schema_version': config.SCHEMA_VERSION,
            'timestamp': datetime.utcnow().isoformat(),
            'data_hash': data_hash,
            'lineage': lineage,
            'validation_issues': validation_issues,
            'quality_analysis': quality_analysis,
            'total_stocks': len(stocks_df),
            'total_sectors': len(sector_df),
            'source': 'google_sheets',
            'cache_used': use_cache
        }
        
        logger.info(f"✅ Pipeline completed in {health['processing_time_s']:.2f}s")
        logger.info(f"📊 Loaded {len(stocks_df)} stocks, quality score: {quality_analysis['quality_score']:.1f}")
        
        return stocks_df, sector_df, health
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# ============================================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================================

def get_cache():
    """Get cache instance (for Streamlit integration)"""
    return _cache

def clear_cache():
    """Clear all cached data"""
    _cache.clear()
    logger.info("Cache cleared")

def health_check(config: Optional[Config] = None) -> Dict[str, Any]:
    """Quick health check without loading data"""
    if config is None:
        config = Config()
    
    health = {
        'status': 'unknown',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Check each sheet URL
    session = get_session()
    for sheet_name in config.SHEET_GIDS:
        try:
            url = config.get_sheet_url(sheet_name)
            response = session.head(url, timeout=5)
            health['checks'][sheet_name] = {
                'status': 'ok' if response.status_code == 200 else 'error',
                'status_code': response.status_code
            }
        except Exception as e:
            health['checks'][sheet_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Determine overall status
    statuses = [check.get('status') for check in health['checks'].values()]
    if all(s == 'ok' for s in statuses):
        health['status'] = 'healthy'
    elif any(s == 'ok' for s in statuses):
        health['status'] = 'degraded'
    else:
        health['status'] = 'critical'
    
    return health

# ============================================================================
# For Streamlit: Use these decorators on your functions
# ============================================================================
# Example:
# @st.cache_data(ttl=300)
# def load_data():
#     return load_and_process()

if __name__ == "__main__":
    # Simple test
    try:
        print("Running health check...")
        health = health_check()
        print(f"Health status: {health['status']}")
        print(json.dumps(health, indent=2))
        
        print("\nLoading data...")
        stocks, sectors, load_health = load_and_process()
        print(f"✅ Loaded {len(stocks)} stocks, {len(sectors)} sectors")
        print(f"📊 Quality score: {load_health['quality_analysis']['quality_score']:.1f}")
        print(f"🔑 Data hash: {load_health['data_hash']}")
    except Exception as e:
        print(f"❌ Error: {e}")

# End of core_system_foundation.py v5.0
