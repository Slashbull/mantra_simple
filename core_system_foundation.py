# core_system_foundation.py (FINAL, PRODUCTION-GRADE, FOR M.A.N.T.R.A.)
"""
Loads, cleans, and prepares all sheets for the M.A.N.T.R.A. stock intelligence engine.
Handles percent/₹/float conversion, NA, and column normalization.
"""

import pandas as pd
import numpy as np
import re
import io
import requests

# =========================
# --- Cleaning helpers ---
# =========================

def _clean_percent(val):
    if pd.isnull(val): return np.nan
    if isinstance(val, str) and "%" in val:
        try: return float(val.replace('%','').replace(',',''))
        except: return np.nan
    try: return float(val)
    except: return np.nan

def _clean_rupee(val):
    if pd.isnull(val): return np.nan
    if isinstance(val, str):
        cleaned = val.replace("₹", "").replace(",", "").strip()
        match = re.search(r'([\d.]+)', cleaned)
        if not match: return np.nan
        num = float(match.group(1))
        if 'Cr' in cleaned or 'cr' in cleaned:
            return num
        elif 'Lac' in cleaned or 'lac' in cleaned:
            return num / 100.0
        else:
            return num
    try: return float(val)
    except: return np.nan

def _fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ======================================
# --- Main loader for all sheet types ---
# ======================================

def load_clean_watchlist(
    source: str, is_url: bool = True
) -> pd.DataFrame:
    # Confirmed columns as per your final schema
    percent_cols = [
        "ret_1d","ret_3d","ret_7d","ret_30d","ret_3m","ret_6m",
        "ret_1y","ret_3y","ret_5y","from_low_pct","from_high_pct","eps_change_pct"
    ]
    rupee_cols = ["market_cap"]
    numeric_cols = [
        "price","prev_close","low_52w","high_52w",
        "sma_20d","sma_50d","sma_200d",
        "volume_1d","volume_7d","volume_30d","volume_3m",
        "vol_ratio_1d_90d","vol_ratio_7d_90d","vol_ratio_30d_90d","rvol",
        "pe","eps_current","eps_last_qtr","eps_duplicate"
    ]
    key_cols = ["ticker","company_name","sector"]

    if is_url:
        df = _fetch_csv(source)
    else:
        df = pd.read_csv(source)
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    # Normalize headers
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    # Cleaning
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_percent)
    for col in rupee_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_rupee)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ticker, company_name, sector cleanup
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].str.upper()
    # Drop dups, drop rows missing criticals
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df = df.dropna(subset=["ticker","price","final_score"], how="any")
    # Ensure final_score >= 0
    if "final_score" in df.columns:
        df["final_score"] = df["final_score"].clip(lower=0)
    return df

def load_clean_industry(
    source: str, is_url: bool = True
) -> pd.DataFrame:
    # Confirmed columns: ticker, company_name, returns_ret_1d...returns_ret_5y, avg_ret_30d, ...
    percent_cols = [
        "returns_ret_1d","returns_ret_3d","returns_ret_7d","returns_ret_30d",
        "returns_ret_3m","returns_ret_6m","returns_ret_1y","returns_ret_3y","returns_ret_5y",
        "avg_ret_30d","avg_ret_3m","avg_ret_6m","avg_ret_1y","avg_ret_3y","avg_ret_5y"
    ]
    key_cols = ["ticker","company_name"]
    if is_url:
        df = _fetch_csv(source)
    else:
        df = pd.read_csv(source)
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_percent)
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].str.upper()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df = df.dropna(subset=["ticker"], how="any")
    return df

def load_clean_sector(
    source: str, is_url: bool = True
) -> pd.DataFrame:
    # Confirmed columns: sector, sector_ret_1d...sector_ret_5y, sector_avg_30d, ..., sector_count
    percent_cols = [
        "sector_ret_1d","sector_ret_3d","sector_ret_7d","sector_ret_30d",
        "sector_ret_3m","sector_ret_6m","sector_ret_1y","sector_ret_3y","sector_ret_5y",
        "sector_avg_30d","sector_avg_3m","sector_avg_6m","sector_avg_1y","sector_avg_3y","sector_avg_5y"
    ]
    if is_url:
        df = _fetch_csv(source)
    else:
        df = pd.read_csv(source)
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_percent)
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()
    if "sector_count" in df.columns:
        df["sector_count"] = pd.to_numeric(df["sector_count"], errors="coerce")
    df = df.drop_duplicates(subset=["sector"]).reset_index(drop=True)
    df = df.dropna(subset=["sector"], how="any")
    return df

# =================================================
# Example usage:
# Replace these URLs/gids with your own
WATCHLIST_URL = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=2026492216"
INDUSTRY_URL  = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=100734077"
SECTOR_URL    = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=140104095"

# To load in your app:
# wl_df = load_clean_watchlist(WATCHLIST_URL)
# ind_df = load_clean_industry(INDUSTRY_URL)
# sec_df = load_clean_sector(SECTOR_URL)
