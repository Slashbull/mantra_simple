"""
data_loader.py - M.A.N.T.R.A. Data Foundation
============================================
Clean, simple, reliable data loading from Google Sheets
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime
from constants import (
    GOOGLE_SHEET_ID, SHEET_CONFIGS, CACHE_DURATION_MINUTES,
    MIN_REQUIRED_COLUMNS, DATA_QUALITY_THRESHOLDS
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Simple and bulletproof data loader"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_DURATION_MINUTES*60, show_spinner=False)
    def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all data from Google Sheets with proper error handling
        Returns: (stocks_df, sector_df, health_report)
        """
        health = {
            'status': 'loading',
            'timestamp': datetime.now(),
            'errors': [],
            'warnings': []
        }
        
        try:
            # Build URLs
            watchlist_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['watchlist']['gid']}"
            sector_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={SHEET_CONFIGS['sector']['gid']}"
            
            # Load dataframes
            stocks_df = pd.read_csv(watchlist_url)
            sector_df = pd.read_csv(sector_url)
            
            # Basic cleaning
            stocks_df = DataLoader._basic_clean(stocks_df)
            sector_df = DataLoader._basic_clean(sector_df)
            
            # Type-specific cleaning
            stocks_df = DataLoader._clean_stocks_data(stocks_df)
            sector_df = DataLoader._clean_sector_data(sector_df)
            
            # Add calculated fields
            stocks_df = DataLoader._add_calculated_fields(stocks_df)
            
            # Validate data
            validation_issues = DataLoader._validate_data(stocks_df, sector_df)
            health['warnings'].extend(validation_issues)
            
            # Update health status
            health['status'] = 'success'
            health['stocks_count'] = len(stocks_df)
            health['sectors_count'] = len(sector_df)
            health['data_quality'] = DataLoader._calculate_data_quality(stocks_df)
            
            return stocks_df, sector_df, health
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            health['status'] = 'error'
            health['errors'].append(str(e))
            return pd.DataFrame(), pd.DataFrame(), health
    
    @staticmethod
    def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Basic dataframe cleaning"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    
    @staticmethod
    def _clean_stocks_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean stock-specific data"""
        # Price columns - remove ₹ symbol and convert
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 
                      'sma_20d', 'sma_50d', 'sma_200d']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('₹', '').str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Percentage columns - remove % and convert
        pct_cols = [col for col in df.columns if 'ret_' in col or '_pct' in col]
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', ''), 
                    errors='coerce'
                )
        
        # Volume columns - remove commas and convert
        vol_cols = [col for col in df.columns if 'volume' in col.lower()]
        for col in vol_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Market cap - handle Cr/Lakh notations
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].apply(DataLoader._parse_market_cap)
        
        # Numeric columns
        numeric_cols = ['pe', 'eps_current', 'rvol', 'position_52w']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure ticker is uppercase
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        return df
    
    @staticmethod
    def _clean_sector_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean sector-specific data"""
        # Clean all return columns
        for col in df.columns:
            if 'ret_' in col or 'avg_' in col:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(
                        df[col].str.replace('%', ''), 
                        errors='coerce'
                    )
        
        # Ensure sector_count is numeric
        if 'sector_count' in df.columns:
            df['sector_count'] = pd.to_numeric(df['sector_count'], errors='coerce')
        
        return df
    
    @staticmethod
    def _parse_market_cap(val) -> float:
        """Parse market cap with Cr/Lakh conversion"""
        if pd.isna(val):
            return np.nan
        
        val = str(val).upper().replace('₹', '').replace(',', '').strip()
        
        # Handle Crores (1 Cr = 10 million)
        if 'CR' in val:
            number = val.replace('CR', '').strip()
            try:
                return float(number) * 1e7
            except:
                return np.nan
        
        # Handle Lakhs (1 Lakh = 100,000)
        if 'L' in val or 'LAC' in val or 'LAKH' in val:
            number = val.replace('L', '').replace('LAC', '').replace('LAKH', '').strip()
            try:
                return float(number) * 1e5
            except:
                return np.nan
        
        # Try direct conversion
        try:
            return float(val)
        except:
            return np.nan
    
    @staticmethod
    def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add useful calculated fields"""
        # Price position in 52-week range
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            range_size = df['high_52w'] - df['low_52w']
            df['position_52w'] = np.where(
                range_size > 0,
                ((df['price'] - df['low_52w']) / range_size * 100).round(2),
                50.0
            )
        
        # Distance from SMAs
        if 'price' in df.columns:
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    df[f'distance_from_{sma}'] = (
                        (df['price'] - df[sma]) / df[sma] * 100
                    ).round(2)
        
        # Volume spike indicator
        if 'rvol' in df.columns:
            df['volume_spike'] = df['rvol'] > 2.0
        
        # Value indicator
        if 'pe' in df.columns:
            df['is_value_stock'] = (df['pe'] > 0) & (df['pe'] < 20)
        
        # Momentum indicator
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['has_momentum'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
        
        # Market cap category
        if 'market_cap' in df.columns:
            df['market_cap_category'] = pd.cut(
                df['market_cap'],
                bins=[0, 5e9, 2e10, 1e11, float('inf')],
                labels=['Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']
            )
        
        return df
    
    @staticmethod
    def _validate_data(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> list:
        """Validate data quality and return warnings"""
        warnings = []
        
        # Check for minimum required columns
        missing_cols = set(MIN_REQUIRED_COLUMNS) - set(stocks_df.columns)
        if missing_cols:
            warnings.append(f"Missing columns: {', '.join(missing_cols)}")
        
        # Check data size
        if len(stocks_df) < DATA_QUALITY_THRESHOLDS['MIN_ROWS']:
            warnings.append(f"Low data count: {len(stocks_df)} stocks")
        
        # Check null percentage
        if not stocks_df.empty:
            null_pct = stocks_df.isna().sum().sum() / (len(stocks_df) * len(stocks_df.columns)) * 100
            if null_pct > DATA_QUALITY_THRESHOLDS['MAX_NULL_PERCENT']:
                warnings.append(f"High null percentage: {null_pct:.1f}%")
        
        # Check price validity
        if 'price' in stocks_df.columns:
            invalid_prices = (
                (stocks_df['price'] < DATA_QUALITY_THRESHOLDS['MIN_PRICE']) |
                (stocks_df['price'] > DATA_QUALITY_THRESHOLDS['MAX_PRICE'])
            ).sum()
            if invalid_prices > 0:
                warnings.append(f"{invalid_prices} stocks have invalid prices")
        
        # Check sector data
        if sector_df.empty:
            warnings.append("No sector data available")
        
        return warnings
    
    @staticmethod
    def _calculate_data_quality(df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        if df.empty:
            return 0.0
        
        scores = []
        
        # Completeness score
        completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        scores.append(completeness)
        
        # Column coverage score
        required_present = len(set(MIN_REQUIRED_COLUMNS) & set(df.columns))
        column_score = (required_present / len(MIN_REQUIRED_COLUMNS)) * 100
        scores.append(column_score)
        
        # Data validity score (price ranges)
        if 'price' in df.columns:
            valid_prices = (
                (df['price'] >= DATA_QUALITY_THRESHOLDS['MIN_PRICE']) &
                (df['price'] <= DATA_QUALITY_THRESHOLDS['MAX_PRICE'])
            ).sum()
            validity_score = (valid_prices / len(df)) * 100
            scores.append(validity_score)
        
        return np.mean(scores)
