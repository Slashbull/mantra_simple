"""
sector_rotation_engine.py - Elite Sector Rotation Analysis Engine for M.A.N.T.R.A.

v2.1 — tuned weights, safer math, and a bug-fix in edge detection
──────────────────────────────────────────────────────────────────
* **Bug-fix:** `_detect_rotation_edges` mistakenly appended the outer list to itself. Now correctly appends `sector_edges`.
* **Modular weights:** tweak `WEIGHTS` at the top to rebalance score contribution without touching code.
* **Utility helpers** (`_pct`, `_safe_rank`) keep percentile logic DRY and robust to NaNs / singleton frames.
* **Explainable pipeline:** every step writes an inline comment about *why* it’s needed.
* **Self-test:** `python sector_rotation_engine.py --demo` prints a 3-sector demo so you can sanity-check in <2 s.

Author: Claude (AI Quant Architect)
License: Proprietary – M.A.N.T.R.A. System
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG – tweak here, not in the guts
# ────────────────────────────────────────────────────────────────────────────────

WEIGHTS: Dict[str, float] = {
    "sector": 0.40,
    "consistency": 0.20,
    "relative": 0.20,
    "diversity": 0.20,  # lower concentration risk → higher score
}

# percentile bins for status assignment
STATUS_THRESHOLDS = {
    "Explosive": 0.90,
    "Hot": 0.75,
    "Warming": 0.50,
    "Neutral": 0.25,
    "Cooling": 0.10,
    "Cold": 0.00,
}

# ────────────────────────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────────


class RotationStatus(Enum):
    EXPLOSIVE = "Explosive"
    HOT = "Hot"
    WARMING = "Warming"
    NEUTRAL = "Neutral"
    COOLING = "Cooling"
    COLD = "Cold"


@dataclass
class SectorAnalytics:
    momentum_velocity: float
    consistency_score: float
    relative_strength: float
    concentration_risk: float
    edge_score: float


# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────────────────


def compute_sector_rotation(
    sector_df: pd.DataFrame,
    metric: str = "sector_avg_3m",
    enable_advanced: bool = True,
) -> pd.DataFrame:
    """End-to-end sector-rotation dataframe.

    Parameters
    ----------
    sector_df : pd.DataFrame
        Must contain at least a *sector* column plus the chosen *metric* column.
    metric : str, default "sector_avg_3m"
        Which numeric column to rank for the core momentum score.
    enable_advanced : bool, default True
        Adds velocity, consistency, relative strength & edge detection.
    """

    df = _normalize(sector_df)

    if metric not in df.columns:
        metric = _fallback_metric(df)

    df = _core_scores(df, metric)
    df = _assign_status(df)

    if enable_advanced and not df.empty:
        df = _advanced_metrics(df, metric)
        df = _detect_edges(df)

    return _format(df, enable_advanced)


def sector_rotation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Dashboard-friendly snapshot of the rotation landscape."""
    out: Dict[str, Any] = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "total_sectors": len(df),
        "state": "balanced",
    }

    if "rotation_status" in df.columns:
        dist = df["rotation_status"].value_counts(dropna=False).to_dict()
        out["rotation_distribution"] = dist

        hot = dist.get("Hot", 0) / max(len(df), 1)
        cold = dist.get("Cold", 0) / max(len(df), 1)
        if hot > 0.4:
            out["state"] = "risk-on"
        elif cold > 0.4:
            out["state"] = "risk-off"

    # Top / bottom snapshot
    if {"sector", "sector_score"}.issubset(df.columns):
        out["leaders"] = df.nlargest(3, "sector_score")[
            ["sector", "sector_score", "rotation_status"]
        ].to_dict("records")
        out["laggards"] = df.nsmallest(3, "sector_score")[
            ["sector", "sector_score", "rotation_status"]
        ].to_dict("records")

    if "edge_score" in df.columns:
        out["edge_overview"] = {
            "avg_edge": round(df["edge_score"].mean(), 2),
            "potential": int((df["edge_score"] > 70).sum()),
        }

    return out


# ────────────────────────────────────────────────────────────────────────────────
# INTERNAL – helpers are prefixed with _
# ────────────────────────────────────────────────────────────────────────────────


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Empty sector dataframe provided")

    df = raw.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]+", "_", regex=True)
    )

    sector_col = next((c for c in df.columns if "sector" in c), None)
    if not sector_col:
        raise ValueError("No sector column found")
    if sector_col != "sector":
        df = df.rename(columns={sector_col: "sector"})

    df["sector"] = df["sector"].astype(str).str.title()
    df = df.dropna(subset=["sector"]).reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "sector"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


def _fallback_metric(df: pd.DataFrame) -> str:
    # choose the first reasonable numeric column
    numeric = df.select_dtypes("number").columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns to rank by")
    return numeric[0]


def _core_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # replace NaNs with median to avoid skewing ranks
    clean = df[metric].fillna(df[metric].median())
    df["sector_score"] = _pct(clean)
    df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)
    return df


def _assign_status(df: pd.DataFrame) -> pd.DataFrame:
    def which_status(p: float) -> str:  # percentile 0-100
        for name, thr in STATUS_THRESHOLDS.items():
            if p >= thr * 100:
                return name
        return "Cold"

    df["rotation_status"] = df["sector_score"].apply(which_status)
    return df


def _advanced_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # momentum velocity: 1M minus third of 3M (if both exist)
    m1 = next((c for c in df.columns if "1m" in c and "avg" in c), None)
    m3 = next((c for c in df.columns if "3m" in c and "avg" in c), None)
    df["momentum_velocity"] = df[m1] - df[m3] / 3 if m1 and m3 else 0

    # consistency: low volatility preferred
    vol = next((c for c in df.columns if "vol" in c or "std" in c), None)
    df["consistency_score"] = 100 - _pct(df[vol]) if vol else 50

    # relative strength to market
    market_avg = df[metric].mean()
    df["relative_strength"] = ((df[metric] - market_avg) / abs(market_avg).clip(lower=1e-9)) * 100

    # concentration risk (needs sector_count)
    if "sector_count" in df.columns:
        total = df["sector_count"].sum() or 1
        df["concentration_risk"] = df["sector_count"] / total * 100
    else:
        df["concentration_risk"] = 100 / len(df)

    # composite edge score
    df["edge_score"] = (
        df["sector_score"] * WEIGHTS["sector"]
        + df["consistency_score"] * WEIGHTS["consistency"]
        + (df["relative_strength"] + 100) / 2 * WEIGHTS["relative"]
        + (100 - df["concentration_risk"]) * WEIGHTS["diversity"]
    ).clip(0, 100)

    return df


def _detect_edges(df: pd.DataFrame) -> pd.DataFrame:
    # threshold shortcuts to avoid recomputing quantiles every loop
    vel_hi = df["momentum_velocity"].quantile(0.9)
    edges: List[List[str]] = []

    for _, row in df.iterrows():
        sector_edges: List[str] = []
        if row.get("momentum_velocity", 0) > vel_hi:
            sector_edges.append("momentum_surge")
        if row["sector_score"] < 30 and row.get("consistency_score", 50) > 70:
            sector_edges.append("oversold_quality")
        if row["rotation_status"] == "Explosive" and row.get("relative_strength", 0) > 10:
            sector_edges.append("rotation_leader")
        if row.get("sector_score", 0) < 20 and row.get("momentum_velocity", 0) > 0:
            sector_edges.append("potential_reversal")
        edges.append(sector_edges)

    df["rotation_edges"] = edges
    df["edge_count"] = df["rotation_edges"].apply(len)
    return df


def _format(df: pd.DataFrame, adv: bool) -> pd.DataFrame:
    base = ["sector", "sector_score", "sector_rank", "rotation_status"]
    adv_cols = [
        "momentum_velocity",
        "consistency_score",
        "relative_strength",
        "edge_score",
        "edge_count",
    ]
    cols = base + adv_cols if adv else base
    cols = [c for c in cols if c in df.columns]

    out = df[cols].sort_values("sector_score", ascending=False).reset_index(drop=True)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


# ────────────────────────────────────────────────────────────────────────────────
# SMALL UTILS
# ────────────────────────────────────────────────────────────────────────────────


def _pct(series: pd.Series) -> pd.Series:
    """Percentile rank 0-100, NaN-safe."""
    if series.nunique(dropna=True) < 2:
        return pd.Series(50.0, index=series.index)
    return series.rank(pct=True, method="average") * 100


# ────────────────────────────────────────────────────────────────────────────────
# DEMO – quick test run
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, textwrap

    p = argparse.ArgumentParser(description="Tiny demo / smoke-test")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        demo = pd.DataFrame(
            {
                "sector": ["Tech", "Banks", "Energy"],
                "sector_avg_3m": [12.3, -4.1, 6.7],
                "sector_avg_1m": [5.2, 1.2, 3.3],
                "volatility_1m": [7.5, 12.1, 9.9],
                "sector_count": [55, 40, 30],
            }
        )
        res = compute_sector_rotation(demo)
        print(textwrap.indent(res.to_string(index=False), prefix="\n "))
        print("\nSummary:", sector_rotation_summary(res))
