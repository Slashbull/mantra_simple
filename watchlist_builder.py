# watchlist_builder.py (ALL-TIME BEST, ALL-EDGE, SUPER CLEAN, DATA-DRIVEN)

import pandas as pd
from typing import List, Dict, Any, Optional

# Core scenario filters
def top_n(df: pd.DataFrame, n=20, by="final_score", tag=None):
    filt = df.copy()
    if tag: filt = filt[filt["tag"] == tag]
    return filt.sort_values(by=by, ascending=False).head(n).reset_index(drop=True)

def sector_leaders(df: pd.DataFrame, n_per_sector=2, by="final_score", tag=None):
    filt = df.copy()
    if tag: filt = filt[filt["tag"] == tag]
    return (filt.sort_values([ "sector", by], ascending=[True, False])
                .groupby("sector")
                .head(n_per_sector)
                .reset_index(drop=True))

def multi_spike_anomalies(df: pd.DataFrame, min_spike=3):
    return df[df.get("spike_score", 0) >= min_spike].sort_values("spike_score", ascending=False).reset_index(drop=True)

def laggard_reversal(df: pd.DataFrame):
    return df[(df.get("ret_1y", 0) < 0) & (df.get("ret_30d", 0) > 5) & (df.get("tag", "") == "Buy")].reset_index(drop=True)

def long_term_winners(df: pd.DataFrame, min_yrs=5, min_ret=15):
    # Strong 5Y return, consistent, not at high
    cols = ["ret_3y", "ret_5y", "from_high_pct"]
    for c in cols: df[c] = pd.to_numeric(df.get(c, 0), errors="coerce")
    f = (df.get("ret_5y", 0) > min_ret) & (df.get("ret_3y", 0) > min_ret/2) & (df.get("from_high_pct", 0) > 10)
    return df[f].sort_values("ret_5y", ascending=False).reset_index(drop=True)

def low_volatility(df: pd.DataFrame, max_std=2.5):
    # Low std-dev of short-term returns, still scoring well
    short_rets = df[["ret_3d", "ret_7d", "ret_30d"]].fillna(0)
    stds = short_rets.std(axis=1)
    f = (stds < max_std) & (df["final_score"] > 60)
    return df[f].assign(ret_std=stds).sort_values("final_score", ascending=False).reset_index(drop=True)

def fresh_52w_high(df: pd.DataFrame):
    return df[df.get("price", 0) >= df.get("high_52w", 0)].sort_values("final_score", ascending=False).reset_index(drop=True)

def value_outliers(df: pd.DataFrame, pe_max=15, eps_min=75):
    f = (df.get("pe", 99) < pe_max) & (df.get("eps_score", 0) > eps_min)
    return df[f].sort_values("final_score", ascending=False).reset_index(drop=True)

def price_tier_watchlists(df: pd.DataFrame, tiers=None, tag=None):
    if "price_tier" not in df: return {}
    result = {}
    for t in (tiers or sorted(df["price_tier"].dropna().unique())):
        q = df[df["price_tier"] == t]
        if tag: q = q[q["tag"] == tag]
        if not q.empty:
            result[t] = q.sort_values("final_score", ascending=False).reset_index(drop=True)
    return result

def custom(df: pd.DataFrame, price_tier=None, eps_min=None, pe_max=None, sector=None, tag=None, exclude_near_high=True):
    f = pd.Series([True] * len(df), index=df.index)
    if price_tier: f &= (df.get("price_tier", "") == price_tier)
    if eps_min is not None: f &= (df.get("eps_score", 0) >= eps_min)
    if pe_max is not None: f &= (df.get("pe", 0) <= pe_max)
    if sector: f &= (df.get("sector", "") == sector)
    if tag: f &= (df.get("tag", "") == tag)
    if exclude_near_high: f &= (df.get("from_high_pct", 0) > 5)
    filt = df[f]
    return filt.sort_values("final_score", ascending=False).reset_index(drop=True)

# Build all watchlists (extendable: just add to this dict!)
def build_all_watchlists(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "top_20_overall": top_n(df, 20),
        "top_10_buy": top_n(df, 10, tag="Buy"),
        "sector_leaders": sector_leaders(df, 2, tag="Buy"),
        "multi_spike_anomalies": multi_spike_anomalies(df),
        "laggard_reversal": laggard_reversal(df),
        "long_term_winners": long_term_winners(df),
        "low_volatility": low_volatility(df),
        "fresh_52w_high": fresh_52w_high(df),
        "value_outliers": value_outliers(df),
        "by_price_tier": price_tier_watchlists(df, tag="Buy"),
        # Example custom: Large Cap, Buy, EPS>75, not near high
        "custom": custom(df, eps_min=75, tag="Buy", exclude_near_high=True),
    }

def build_watchlist(df: pd.DataFrame, mode: str = "top_20_overall") -> pd.DataFrame:
    """Returns DataFrame for the selected mode (see keys above)."""
    return build_all_watchlists(df).get(mode, pd.DataFrame())
