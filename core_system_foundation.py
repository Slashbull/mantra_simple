# core_system_foundation.py v5.1 (STREAMLIT CLOUD PRODUCTION READY)
"""
M.A.N.T.R.A. Core System Foundation - Streamlit Cloud Edition
Optimized for Streamlit Community Cloud deployment.
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
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

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
    pass
class SchemaError(CoreFoundationError):
    pass
class DataValidationError(CoreFoundationError):
    pass
class ConfigurationError(CoreFoundationError):
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",
        "returns": "100734077",
        "sector": "140104095"
    })
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
        "ticker", "company_name", "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", "avg_ret_3y", "avg_ret_5y"
    })
    REQUIRED_SECTOR: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", "sector_ret_30d",
        "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", "sector_ret_3y", "sector_ret_5y",
        "sector_avg_30d", "sector_avg_3m", "sector_avg_6m", "sector_avg_1y",
        "sector_avg_3y", "sector_avg_5y", "sector_count"
    })
    CRITICAL_FIELDS: Tuple[str, ...] = ("ticker", "price", "eps_current", "sector")
    CACHE_TTL: int = 300
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.0
    MIN_DATA_QUALITY_SCORE: float = 70.0
    MAX_NULL_PERCENTAGE: float = 20.0
    OUTLIER_THRESHOLD: float = 3.0
    SCHEMA_VERSION: str = "2025.07.11"
    LOG_LEVEL: str = "INFO"
    def get_sheet_url(self, name: str) -> str:
        if name not in self.SHEET_GIDS:
            raise ConfigurationError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"

# ============================================================================
# LOGGING
# ============================================================================

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
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session
_session = None
def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = create_session()
    return _session

# ============================================================================
# SIMPLE IN-MEMORY CACHE
# ============================================================================

class SimpleCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                del self._cache[key]
        return None
    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())
    def clear(self) -> None:
        self._cache.clear()
_cache = SimpleCache()

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_sheet(name: str, config: Config, use_cache: bool = True) -> pd.DataFrame:
    url = config.get_sheet_url(name)
    cache_key = f"sheet_{name}_{config.SCHEMA_VERSION}"
    if use_cache:
        cached = _cache.get(cache_key, config.CACHE_TTL)
        if cached is not None:
            logger.info(f"Cache hit for sheet '{name}'")
            return cached
    logger.info(f"Fetching sheet '{name}' from {url}")
    try:
        session = get_session()
        response = session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = clean_dataframe(df)
        if use_cache:
            _cache.set(cache_key, df)
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch sheet '{name}': {e}")
        raise DataSourceError(f"Cannot load sheet '{name}' from Google Sheets", error_code="FETCH_ERROR", details={'sheet': name, 'error': str(e)})

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace("\u00A0", " ", regex=False)
    return df

# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    s = series.astype(str)
    for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓']:
        s = s.str.replace(symbol, '', regex=False)
    s = s.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    s = s.replace('', 'NaN')
    numeric_series = pd.to_numeric(s, errors='coerce')
    if col_name.endswith('_pct') or '%' in series.astype(str).str.cat():
        return numeric_series
    non_null = numeric_series.dropna()
    if len(non_null) > 0 and non_null.max() < 1 and non_null.min() >= 0:
        return numeric_series * 100
    return numeric_series

def validate_schema(df: pd.DataFrame, required_cols: Set[str], sheet_name: str) -> None:
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise SchemaError(
            f"Missing columns in {sheet_name}: {missing_cols}",
            error_code="SCHEMA_MISMATCH",
            details={'sheet': sheet_name, 'missing': list(missing_cols)}
        )

def merge_datasets(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    returns_df = returns_df.drop(columns=['company_name'], errors='ignore')
    merged = watchlist_df.merge(
        returns_df,
        on='ticker',
        how='left',
        validate='one_to_one'
    )
    if merged['ticker'].duplicated().any():
        dup_count = merged['ticker'].duplicated().sum()
        logger.warning(f"Found {dup_count} duplicate tickers, keeping last")
        merged = merged.drop_duplicates(subset='ticker', keep='last')
    merged['ticker'] = merged['ticker'].astype(str).str.upper().str.strip()
    return merged

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    if 'price' in df and df['price'].notna().any():
        df['price_tier'] = pd.cut(
            df['price'],
            bins=[-np.inf, 100, 200, 500, 1000, 2000, 5000, np.inf],
            labels=['<100', '100-200', '200-500', '500-1K', '1K-2K', '2K-5K', '>5K']
        )
    if 'eps_current' in df and df['eps_current'].notna().any():
        df['eps_tier'] = pd.cut(
            df['eps_current'],
            bins=[-np.inf, 5, 15, 35, 55, 75, 95, np.inf],
            labels=['<5', '5-15', '15-35', '35-55', '55-75', '75-95', '>95']
        )
    if 'pe' in df and 'eps_change_pct' in df:
        df['is_undervalued'] = (df['pe'] < 20) & (df['eps_change_pct'] > 25)
    if 'rvol' in df:
        df['is_volume_spike'] = df['rvol'] > 2
    if 'from_low_pct' in df:
        df['is_near_low'] = df['from_low_pct'] < 15
    if 'from_high_pct' in df:
        df['is_near_high'] = df['from_high_pct'] < 5
    if all(col in df for col in ['ret_3d', 'ret_7d', 'ret_30d']):
        df['trend_strength'] = (
            df['ret_3d'] * 0.5 +
            df['ret_7d'] * 0.3 +
            df['ret_30d'] * 0.2
        )
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_total > 0 and (num_unique / num_total) < 0.5:
            df[col] = df[col].astype('category')
    return df

# ============================================================================
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df) * 100).round(2)
    analysis['null_analysis'] = {
        'total_nulls': int(null_counts.sum()),
        'null_percentage': float((null_counts.sum() / (len(df) * len(df.columns)) * 100).round(2)),
        'columns_with_nulls': null_counts[null_counts > 0].to_dict(),
        'high_null_columns': null_percentages[null_percentages > config.MAX_NULL_PERCENTAGE].to_dict()
    }
    analysis['duplicate_analysis'] = {
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_tickers': int(df['ticker'].duplicated().sum()) if 'ticker' in df else 0
    }
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
    issues = []
    if 'price' in df:
        invalid_prices = (df['price'] <= 0) | (df['price'] > 1000000)
        if invalid_prices.any():
            issues.append({
                'rule': 'price_sanity',
                'message': f"Found {invalid_prices.sum()} stocks with invalid prices",
                'severity': 'critical'
            })
    if 'volume_1d' in df:
        zero_volume = (df['volume_1d'] == 0).sum()
        zero_volume_pct = zero_volume / len(df) * 100
        if zero_volume_pct > 20:
            issues.append({
                'rule': 'volume_non_zero',
                'message': f"{zero_volume_pct:.1f}% of stocks have zero volume",
                'severity': 'warning'
            })
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
    if config is None:
        config = Config()
    start_time = time.time()
    lineage = []
    try:
        watchlist_df = load_sheet('watchlist', config, use_cache)
        returns_df = load_sheet('returns', config, use_cache)
        sector_df = load_sheet('sector', config, use_cache)
        lineage.append(f"Loaded sheets: watchlist={len(watchlist_df)}, returns={len(returns_df)}, sector={len(sector_df)}")
        validate_schema(watchlist_df, config.REQUIRED_WATCHLIST, 'watchlist')
        validate_schema(returns_df, config.REQUIRED_RETURNS, 'returns')
        validate_schema(sector_df, config.REQUIRED_SECTOR, 'sector')
        lineage.append("Schema validation passed")
        stocks_df = merge_datasets(watchlist_df, returns_df)
        lineage.append(f"Merged data: {len(stocks_df)} stocks")
        stocks_df = clean_numeric_columns(stocks_df)
        sector_df = clean_numeric_columns(sector_df)
        lineage.append("Numeric columns cleaned")
        stocks_df = add_derived_features(stocks_df)
        lineage.append("Derived features added")
        stocks_df = optimize_dtypes(stocks_df)
        sector_df = optimize_dtypes(sector_df)
        lineage.append("Memory optimized")
        validation_issues = validate_data(stocks_df, config)
        critical_issues = [i for i in validation_issues if i['severity'] == 'critical']
        if critical_issues:
            raise DataValidationError(
                "Critical validation failures detected",
                details={'issues': critical_issues}
            )
        quality_analysis = analyze_data_quality(stocks_df, config)
        if quality_analysis['quality_score'] < config.MIN_DATA_QUALITY_SCORE:
            logger.warning(
                f"Data quality score {quality_analysis['quality_score']:.1f} below threshold {config.MIN_DATA_QUALITY_SCORE}"
            )
        hash_input = pd.util.hash_pandas_object(stocks_df[['ticker', 'price']], index=False).values
        data_hash = hashlib.sha256(hash_input).hexdigest()[:16]
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
        # --- DOUBLE SAFETY: always return health as dict! ---
        if not isinstance(health, dict):
            try:
                health = dict(health)
            except Exception:
                health = {"health": str(health)}
        return stocks_df, sector_df, health
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# ============================================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================================

def get_cache():
    return _cache
def clear_cache():
    _cache.clear()
    logger.info("Cache cleared")
def health_check(config: Optional[Config] = None) -> Dict[str, Any]:
    if config is None:
        config = Config()
    health = {
        'status': 'unknown',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
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
    statuses = [check.get('status') for check in health['checks'].values()]
    if all(s == 'ok' for s in statuses):
        health['status'] = 'healthy'
    elif any(s == 'ok' for s in statuses):
        health['status'] = 'degraded'
    else:
        health['status'] = 'critical'
    return health

# ============================================================================
# CLI TEST HARNESS
# ============================================================================
if __name__ == "__main__":
    try:
        print("Running health check...")
        health = health_check()
        print(f"Health status: {health['status']}")
        print(json.dumps(health, indent=2))
        print("\nLoading data...")
        stocks, sectors, summary = load_and_process()
        if not isinstance(summary, dict):
            summary = dict(summary)
        print(f"✅ Loaded {len(stocks)} stocks, {len(sectors)} sectors")
        print(f"📊 Quality score: {summary.get('quality_analysis',{}).get('quality_score','NA')}")
        print(f"🔑 Data hash: {summary.get('data_hash')}")
    except Exception as e:
        print(f"❌ Error: {e}")
