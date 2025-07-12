"""
core_system_foundation.py - M.A.N.T.R.A. Data Foundation
======================================================
Clean, simple data loading and processing layer
Designed for reliability and ease of use
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class DataConfig:
    """Simple configuration for data sources"""
    
    # Google Sheets base URL
    SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    
    # Sheet GIDs
    SHEETS = {
        "watchlist": "2026492216",
        "returns": "100734077", 
        "sector": "140104095"
    }
    
    # Data quality thresholds
    MIN_ROWS = 100
    MAX_NULL_PERCENT = 30
    
    # Cache settings
    CACHE_MINUTES = 15
    
    @classmethod
    def get_sheet_url(cls, sheet_name: str) -> str:
        """Get Google Sheets CSV export URL"""
        gid = cls.SHEETS.get(sheet_name)
        if not gid:
            raise ValueError(f"Unknown sheet: {sheet_name}")
        return f"https://docs.google.com/spreadsheets/d/{cls.SHEET_ID}/export?format=csv&gid={gid}"

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Simple, reliable data loader for M.A.N.T.R.A."""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all required data with proper error handling
        
        Returns:
            Tuple of (stocks_df, sector_df, health_report)
        """
        start_time = time.time()
        health = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'warnings': []
        }
        
        try:
            # Load individual sheets
            logger.info("Loading data sheets...")
            watchlist_df = self._load_sheet('watchlist')
            returns_df = self._load_sheet('returns')
            sector_df = self._load_sheet('sector')
            
            # Basic validation
            if watchlist_df.empty or len(watchlist_df) < DataConfig.MIN_ROWS:
                raise ValueError(f"Watchlist has insufficient data: {len(watchlist_df)} rows")
            
            # Merge stock data
            logger.info("Merging stock data...")
            stocks_df = self._merge_stock_data(watchlist_df, returns_df)
            
            # Clean data
            logger.info("Cleaning data...")
            stocks_df = self._clean_stock_data(stocks_df)
            sector_df = self._clean_sector_data(sector_df)
            
            # Add calculated fields
            logger.info("Adding calculated fields...")
            stocks_df = self._add_calculated_fields(stocks_df)
            
            # Final validation
            validation_issues = self._validate_data(stocks_df, sector_df)
            health['warnings'] = validation_issues
            
            # Summary statistics
            health['summary'] = {
                'total_stocks': len(stocks_df),
                'total_sectors': len(sector_df),
                'unique_sectors': stocks_df['sector'].nunique() if 'sector' in stocks_df else 0,
                'processing_time': round(time.time() - start_time, 2)
            }
            
            logger.info(f"✅ Data loaded successfully: {len(stocks_df)} stocks, {len(sector_df)} sectors")
            
            return stocks_df, sector_df, health
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {str(e)}")
            health['status'] = 'error'
            health['errors'].append(str(e))
            
            # Return empty dataframes on error
            return pd.DataFrame(), pd.DataFrame(), health
    
    def _load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load a single sheet with caching"""
        # Check cache
        cache_key = f"sheet_{sheet_name}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {sheet_name}")
            return self.cache[cache_key]
        
        # Load from URL
        url = DataConfig.get_sheet_url(sheet_name)
        logger.info(f"Downloading {sheet_name} from Google Sheets...")
        
        try:
            df = pd.read_csv(url)
            
            # Basic cleaning
            df = self._basic_clean(df)
            
            # Cache the result
            self.cache[cache_key] = df
            self.cache_time[cache_key] = time.time()
            
            logger.info(f"Loaded {len(df)} rows from {sheet_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {sheet_name}: {str(e)}")
            raise
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cache_age = (time.time() - self.cache_time.get(key, 0)) / 60
        return cache_age < DataConfig.CACHE_MINUTES
    
    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic dataframe cleaning"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        return df
    
    def _merge_stock_data(self, watchlist: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Merge watchlist and returns data"""
        # Ensure ticker columns are clean
        for df in [watchlist, returns]:
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
        
        # Remove duplicate columns from returns (except ticker)
        common_cols = set(watchlist.columns) & set(returns.columns) - {'ticker'}
        if common_cols:
            returns = returns.drop(columns=list(common_cols))
        
        # Merge on ticker
        merged = pd.merge(watchlist, returns, on='ticker', how='left')
        
        # Remove any duplicate tickers
        if merged['ticker'].duplicated().any():
            logger.warning(f"Found {merged['ticker'].duplicated().sum()} duplicate tickers")
            merged = merged.drop_duplicates(subset=['ticker'], keep='first')
        
        return merged
    
    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean stock data with Indian market conventions"""
        # Price columns
        price_columns = ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
        for col in price_columns:
            if col in df.columns:
                df[col] = self._clean_price_column(df[col])
        
        # Return/percentage columns
        return_columns = [col for col in df.columns if 'ret_' in col or '_pct' in col]
        for col in return_columns:
            if col in df.columns:
                df[col] = self._clean_percentage_column(df[col])
        
        # Volume columns
        volume_columns = [col for col in df.columns if 'volume' in col or 'vol_' in col]
        for col in volume_columns:
            if col in df.columns and 'ratio' not in col:
                df[col] = self._clean_volume_column(df[col])
        
        # Market cap
        if 'market_cap' in df.columns:
            df['market_cap'] = self._clean_market_cap(df['market_cap'])
        
        # Numeric columns
        numeric_columns = ['pe', 'eps_current', 'eps_last_qtr', 'rvol', 'year']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_price_column(self, series: pd.Series) -> pd.Series:
        """Clean price columns removing ₹ symbol"""
        if series.dtype == 'object':
            series = series.astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
        return pd.to_numeric(series, errors='coerce')
    
    def _clean_percentage_column(self, series: pd.Series) -> pd.Series:
        """Clean percentage columns"""
        if series.dtype == 'object':
            series = series.astype(str).str.replace('%', '').str.strip()
        return pd.to_numeric(series, errors='coerce')
    
    def _clean_volume_column(self, series: pd.Series) -> pd.Series:
        """Clean volume columns"""
        if series.dtype == 'object':
            series = series.astype(str).str.replace(',', '').str.strip()
        return pd.to_numeric(series, errors='coerce')
    
    def _clean_market_cap(self, series: pd.Series) -> pd.Series:
        """Clean market cap with Cr/Lakh conversion"""
        def parse_mcap(val):
            if pd.isna(val):
                return np.nan
            
            val = str(val).upper().replace('₹', '').replace(',', '').strip()
            
            # Handle Crores
            if 'CR' in val:
                number = val.replace('CR', '').strip()
                try:
                    return float(number) * 1e7
                except:
                    return np.nan
            
            # Handle Lakhs
            if 'L' in val or 'LAC' in val:
                number = val.replace('L', '').replace('LAC', '').strip()
                try:
                    return float(number) * 1e5
                except:
                    return np.nan
            
            # Try direct conversion
            try:
                return float(val)
            except:
                return np.nan
        
        return series.apply(parse_mcap)
    
    def _clean_sector_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sector data"""
        # Clean all return columns
        for col in df.columns:
            if 'ret_' in col or 'avg_' in col:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('%', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure sector_count is numeric
        if 'sector_count' in df.columns:
            df['sector_count'] = pd.to_numeric(df['sector_count'], errors='coerce')
        
        return df
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful calculated fields"""
        # Price position in 52-week range
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            range_size = df['high_52w'] - df['low_52w']
            df['position_52w'] = np.where(
                range_size > 0,
                ((df['price'] - df['low_52w']) / range_size * 100).round(2),
                50.0
            )
        
        # Distance from key SMAs
        if 'price' in df.columns:
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    df[f'distance_from_{sma}'] = ((df['price'] - df[sma]) / df[sma] * 100).round(2)
        
        # Volume spike indicator
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > 2.0
        
        # Value indicator
        if 'pe' in df.columns:
            df['is_value_stock'] = (df['pe'] > 0) & (df['pe'] < 20)
        
        # Momentum indicator
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['has_momentum'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
        
        return df
    
    def _validate_data(self, stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> List[str]:
        """Validate data and return list of warnings"""
        warnings = []
        
        # Check for critical columns
        critical_columns = ['ticker', 'price', 'sector']
        for col in critical_columns:
            if col not in stocks_df.columns:
                warnings.append(f"Missing critical column: {col}")
            elif stocks_df[col].isna().sum() > len(stocks_df) * 0.5:
                warnings.append(f"Too many null values in {col}")
        
        # Check data quality
        null_percent = stocks_df.isna().sum().sum() / (len(stocks_df) * len(stocks_df.columns)) * 100
        if null_percent > DataConfig.MAX_NULL_PERCENT:
            warnings.append(f"High null percentage: {null_percent:.1f}%")
        
        # Check for reasonable values
        if 'price' in stocks_df.columns:
            negative_prices = (stocks_df['price'] < 0).sum()
            if negative_prices > 0:
                warnings.append(f"{negative_prices} stocks have negative prices")
        
        # Check sector data
        if sector_df.empty:
            warnings.append("Sector data is empty")
        
        return warnings
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_time.clear()
        logger.info("Cache cleared")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global loader instance
_loader = DataLoader()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main function to load all M.A.N.T.R.A. data
    
    Returns:
        Tuple of (stocks_df, sector_df, health_report)
    """
    return _loader.load_all_data()

def get_sample_data(n: int = 10) -> pd.DataFrame:
    """Get sample stock data for testing"""
    stocks_df, _, _ = load_data()
    return stocks_df.head(n) if not stocks_df.empty else pd.DataFrame()

def refresh_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Force refresh of all data (bypass cache)"""
    _loader.clear_cache()
    return load_data()

def get_data_summary() -> Dict:
    """Get summary statistics about loaded data"""
    stocks_df, sector_df, health = load_data()
    
    if stocks_df.empty:
        return {'error': 'No data loaded'}
    
    return {
        'total_stocks': len(stocks_df),
        'sectors': stocks_df['sector'].nunique() if 'sector' in stocks_df else 0,
        'avg_pe': stocks_df['pe'].mean() if 'pe' in stocks_df else None,
        'data_quality': 100 - (stocks_df.isna().sum().sum() / (len(stocks_df) * len(stocks_df.columns)) * 100),
        'last_updated': health.get('timestamp', 'Unknown')
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Core System Foundation")
    print("="*60)
    
    print("\nLoading data...")
    stocks_df, sector_df, health = load_data()
    
    if health['status'] == 'success':
        print(f"\n✅ Success!")
        print(f"Loaded {len(stocks_df)} stocks across {len(sector_df)} sectors")
        print(f"Processing time: {health['summary']['processing_time']}s")
        
        if health['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in health['warnings']:
                print(f"  - {warning}")
        
        print("\nSample data:")
        print(stocks_df[['ticker', 'company_name', 'price', 'sector']].head())
        
        print("\nData quality summary:")
        summary = get_data_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n❌ Error: {health['errors']}")
    
    print("="*60)
