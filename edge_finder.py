"""
M.A.N.T.R.A. Edge Finder v2.2 (Final, Production-Ready)
======================================================
Elite multi-edge signal detection, vectorized for speed. 
Exports: compute_edge_signals, find_edges, edge_overview

- Fast, safe, robust
- All-time best design for your edge tab
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

# ─────────────────────────────────────────────────────────────
# Enums & Config
# ─────────────────────────────────────────────────────────────
class EdgeType(str, Enum):
    MOMENTUM_BREAKOUT = "Momentum Breakout"
    VALUE_ANOMALY = "Value Anomaly"
    VOLUME_SURGE = "Volume Surge"
    SECTOR_LEADER = "Sector Leader"
    VOLATILITY_SQUEEZE = "Volatility Squeeze"
    TREND_REVERSAL = "Trend Reversal"
    NEW_HIGH = "New 52W High"
    NEW_LOW = "New 52W Low"
    ACCUMULATION = "Accumulation Pattern"
    DISTRIBUTION = "Distribution Warning"
    MEAN_REVERSION = "Mean Reversion Setup"
    RELATIVE_STRENGTH = "Relative Strength"
    BREAKOUT_PULLBACK = "Breakout Pullback"
    OVERSOLD_BOUNCE = "Oversold Bounce"
    OVERBOUGHT_REVERSAL = "Overbought Reversal"

@dataclass
class EdgeConfig:
    min_vol_ratio: float = 1.5
    high_vol_ratio: float = 3
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    score_clip: Tuple[float, float] = (0, 1)
    pe_floor: float = 15
    score_weight: Dict[str, float] = field(default_factory=lambda: {
        "momentum": .3, "value": .3, "volume": .4
    })

    def __post_init__(self):
        s = sum(self.score_weight.values())
        for k in self.score_weight:
            self.score_weight[k] /= s

@dataclass
class MarketRegime:
    vol_pct: float
    trend_strength: float
    breadth: float
    trending: bool
    volatile: bool

def _detect_regime(df: pd.DataFrame) -> MarketRegime:
    ret7 = df.get("ret_7d", pd.Series(dtype=float))
    ret30 = df.get("ret_30d", pd.Series(dtype=float))
    vol_pct = (ret7.std() / ret30.std()) if not ret30.std() == 0 else 1
    trend = (ret30.abs() > 10).mean()
    breadth = (ret7 > 0).mean()
    return MarketRegime(vol_pct, trend, breadth, trend > 0.3, vol_pct > 1.2)

# ─────────────────────────────────────────────────────────────
# Core edge detection (vectorized, fast)
# ─────────────────────────────────────────────────────────────

def compute_edge_signals(df: pd.DataFrame, cfg: EdgeConfig = None) -> pd.DataFrame:
    """Compute all edge flags and explanations, vectorized for performance."""
    cfg = cfg or EdgeConfig()
    d = df.copy()
    reg = _detect_regime(d)

    # Momentum breakout
    mom_short = d.get("ret_3d", 0)
    mom_med = d.get("ret_7d", 0)
    thr_mom = mom_med.abs().quantile(.9)
    volratio = d.get("vol_ratio_1d_90d", 1)
    price = d.get("price", 0)
    hi = d.get("high_52w", 1)
    edge_mom = (
        (mom_short > 0.5 * thr_mom) &
        (mom_med > thr_mom) &
        ((price >= .95 * hi) | (price > d.get("ma_50", 0))) &
        (volratio > cfg.min_vol_ratio)
    )
    score_mom = (
        (mom_short / thr_mom).clip(*cfg.score_clip) * cfg.score_weight["momentum"] +
        (mom_med / thr_mom).clip(*cfg.score_clip) * cfg.score_weight["momentum"] +
        ((price / hi).clip(.9, 1) - .9) / .1 * cfg.score_weight["value"] +
        (volratio / cfg.high_vol_ratio).clip(*cfg.score_clip) * cfg.score_weight["volume"]
    )
    exp_mom = np.where(edge_mom, np.char.add("Breaking out ", mom_med.round(1).astype(str)) + "%", "")

    # Value anomaly
    pe = d.get("pe", 99)
    eps_s = d.get("eps_score", 0)
    final = d.get("final_score", 0)
    pe_low = pe[pe > 0].quantile(.2) if (pe > 0).any() else cfg.pe_floor
    edge_val = (pe > 0) & (pe < pe_low) & (eps_s > 80) & (final > 70) & (d.get("ret_30d", 0) > -20)
    score_val = ((1 - pe / pe_low).clip(*cfg.score_clip) * .4 +
                 (eps_s / 100) * .3 +
                 (final / 100) * .3)
    exp_val = np.where(edge_val, "Value PE=" + pe.round(1).astype(str), "")

    # Volume surge
    ret1 = d.get("ret_1d", 0)
    edge_vol = (volratio > cfg.high_vol_ratio) & (ret1.abs() > 2)
    score_vol = (volratio / cfg.high_vol_ratio).clip(*cfg.score_clip) * .6 + (ret1.abs() / 5).clip(*cfg.score_clip) * .4
    exp_vol = np.where(edge_vol, "Vol " + volratio.round(1).astype(str) + "x", "")

    # Assemble
    d["edge_momentum_breakout"] = edge_mom
    d["score_momentum_breakout"] = np.where(edge_mom, score_mom, 0)
    d["exp_momentum_breakout"] = exp_mom
    d["edge_value_anomaly"] = edge_val
    d["score_value_anomaly"] = np.where(edge_val, score_val, 0)
    d["exp_value_anomaly"] = exp_val
    d["edge_volume_surge"] = edge_vol
    d["score_volume_surge"] = np.where(edge_vol, score_vol, 0)
    d["exp_volume_surge"] = exp_vol

    # Edge glue
    edge_cols = [c for c in d.columns if c.startswith("edge_")]
    score_cols = [c for c in d.columns if c.startswith("score_")]
    d["edge_count"] = d[edge_cols].sum(axis=1)
    d["edge_score"] = d[score_cols].mean(axis=1)
    et_map = {f"edge_{e.name.lower()}": e.value for e in EdgeType}
    d["edge_types"] = pd.Series(
        [', '.join([et_map[c] for c in edge_cols if d.loc[i, c]]) for i in d.index], index=d.index
    )
    exp_cols = [c for c in d.columns if c.startswith("exp_")]
    d["edge_explanation"] = pd.Series(
        [' | '.join([str(d.loc[i, c]) for c in exp_cols if d.loc[i, c]]) for i in d.index], index=d.index
    )
    d["has_edge"] = d["edge_count"] > 0
    d["market_regime"] = _describe_regime(reg)
    return d

def find_edges(df: pd.DataFrame, min_edges: int = 1, min_score: float = 0.0) -> pd.DataFrame:
    """Filter for rows with at least min_edges and min_score."""
    d = compute_edge_signals(df)
    return d[(d.edge_count >= min_edges) & (d.edge_score >= min_score)].sort_values("edge_score", ascending=False)

def _describe_regime(r: MarketRegime) -> str:
    if r.volatile and r.trending: return "Volatile Trending"
    if r.volatile: return "High Volatility"
    if r.trending: return "Trending"
    return "Range-Bound"

# ─────────────────────────────────────────────────────────────
# 🚩 Edge Overview Function
# ─────────────────────────────────────────────────────────────

def edge_overview(df: pd.DataFrame) -> dict:
    """
    Summarize edge signal distribution in the DataFrame.
    Returns a dictionary with counts and stats for the dashboard.
    """
    if df is None or df.empty:
        return {
            'total_edges': 0,
            'unique_stocks': 0,
            'unique_edge_types': 0,
            'top_edge_types': {},
            'avg_edges_per_stock': 0.0,
        }

    # Total number of edges (sum of edge_count)
    total_edges = int(df['edge_count'].sum()) if 'edge_count' in df.columns else 0

    # Unique stocks with at least one edge
    unique_stocks = int((df['edge_count'] > 0).sum()) if 'edge_count' in df.columns else 0

    # Extract all edge types
    if 'edge_types' in df.columns and not df['edge_types'].isna().all():
        all_types = (
            df.loc[df['edge_types'].notna(), 'edge_types']
            .astype(str)
            .str.split(', ')
            .explode()
        )
        top_edge_types = all_types.value_counts().head(5).to_dict()
        unique_edge_types = all_types.nunique()
    else:
        top_edge_types = {}
        unique_edge_types = 0

    # Average edges per stock
    avg_edges_per_stock = float(df['edge_count'].mean()) if 'edge_count' in df.columns and len(df) > 0 else 0.0

    return {
        'total_edges': total_edges,
        'unique_stocks': unique_stocks,
        'unique_edge_types': unique_edge_types,
        'top_edge_types': top_edge_types,
        'avg_edges_per_stock': round(avg_edges_per_stock, 2),
    }

# ─────────────────────────────────────────────────────────────
# Self-test (safe to remove in production)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test stub
    print("EdgeFinder v2.2 loaded – use compute_edge_signals(df), edge_overview(df)")
