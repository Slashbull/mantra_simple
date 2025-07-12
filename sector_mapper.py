"""
sector_mapper.py - Elite Sector Analysis & Rotation Intelligence Engine

Production-grade sector mapping with advanced analytics, anomaly detection, and edge signals.
Updated by ChatGPT‑o3 – v2.1 (2025‑07‑12)

Key enhancements in v2.1
-----------------------
* **Cleaner helpers** – dedicated _pct, _rank utilities, NaN‑safe math.
* **Faster aggregation** – pandas.named_agg + categorical dtypes.
* **Config knobs** – WEIGHTS, STATUS_SCORES exposed at top for 1‑line tuning.
* **Extensive type hints** – for static checkers / IDEs.
* **Self‑test** – lightweight example in __main__ guarded by –demo flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Config – tweak without digging into the code body
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "momentum": 0.30,
    "quality": 0.25,
    "risk_adjusted": 0.20,
    "relative_strength": 0.15,
    "rotation": 0.10,
}

STATUS_SCORES: Dict[str, int] = {
    "Breakout": 7,
    "Hot": 6,
    "Warming": 5,
    "Neutral": 4,
    "Cooling": 3,
    "Cold": 2,
    "Breakdown": 1,
}


class SectorStatus(str, Enum):
    BREAKOUT = "Breakout"
    HOT = "Hot"
    WARMING = "Warming"
    NEUTRAL = "Neutral"
    COOLING = "Cooling"
    COLD = "Cold"
    BREAKDOWN = "Breakdown"


@dataclass(slots=True)
class SectorMetrics:
    avg_return_1m: float
    avg_return_3m: float
    avg_return_6m: float
    volatility: float
    momentum_score: float
    relative_strength: float
    breadth: float
    volume_surge: float
    quality_score: float
    risk_adjusted_return: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "run_sector_mapper",
    "get_sector_heatmap_data",
    "get_rotation_summary",
    "get_sector_recommendations",
]


def run_sector_mapper(
    sector_df: pd.DataFrame,
    stock_df: Optional[pd.DataFrame] = None,
    *,
    lookback_periods: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Main entry – returns dashboard‑ready DataFrame."""

    periods = lookback_periods or {"short": 30, "medium": 90, "long": 180}

    df = _normalize_sector_data(sector_df)
    df = _calculate_base_metrics(df, stock_df)
    df = _calculate_momentum_scores(df)
    df = _calculate_relative_strength(df)
    df = _calculate_risk_metrics(df)
    df = _detect_sector_anomalies(df)
    df = _calculate_rotation_signals(df)
    df = _calculate_composite_scores(df)
    df = _assign_rotation_status(df)
    df = _generate_edge_signals(df)
    df = _format_for_dashboard(df)

    df.attrs["generated"] = datetime.now().isoformat(timespec="seconds")
    df.attrs["lookback"] = periods

    return df

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(series: pd.Series) -> pd.Series:
    """Return percentile rank (0‑100) NaN‑safe."""
    return series.rank(pct=True, method="average") * 100


def _rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False, method="min")


def _normalize_sector_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    if "sector" not in df.columns:
        raise KeyError("'sector' column is required in sector_df")

    df["sector"] = (
        df["sector"].astype("string").str.strip().str.title().fillna("Unknown")
    )
    df = df[df["sector"] != "Unknown"].reset_index(drop=True)

    # make sector categorical for memory/perf
    df["sector"] = df["sector"].astype("category")

    # convert obvious numeric cols
    for col in df.columns:
        if any(k in col for k in ("return", "avg", "score", "vol", "count")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _calculate_base_metrics(df: pd.DataFrame, stock_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if stock_df is None or "sector" not in stock_df.columns:
        # ensure essential placeholders
        for col in (
            "avg_return_1m",
            "avg_return_3m",
            "avg_return_6m",
            "volatility_1m",
            "sector_count",
            "avg_quality_score",
            "avg_volume_surge",
            "sector_market_cap",
        ):
            df[col] = df.get(col, 0)
        return df

    grp = stock_df.groupby("sector", observed=True).agg(
        avg_return_1m=("ret_1m", "mean"),
        avg_return_3m=("ret_3m", "mean"),
        avg_return_6m=("ret_6m", "mean"),
        volatility_1m=("ret_1m", "std"),
        sector_count=("ret_1m", "size"),
        avg_quality_score=("final_score", "mean"),
        avg_volume_surge=("vol_ratio_1d_90d", "mean"),
        median_pe=("pe", lambda s: s.loc[s > 0].median()),
        sector_market_cap=("market_cap", "sum"),
    )
    grp.reset_index(inplace=True)

    df = df.merge(grp, on="sector", how="left")

    df[[
        "avg_return_1m",
        "avg_return_3m",
        "avg_return_6m",
        "volatility_1m",
    ]] = df[[
        "avg_return_1m",
        "avg_return_3m",
        "avg_return_6m",
        "volatility_1m",
    ]].fillna(0)
    df["sector_count"] = df["sector_count"].fillna(0).astype(int)

    return df


def _calculate_momentum_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["momentum_1m"] = _pct(df["avg_return_1m"])
    df["momentum_3m"] = _pct(df["avg_return_3m"])
    df["momentum_acceleration"] = _pct(
        df["avg_return_1m"] - df["avg_return_3m"].div(3)
    )
    df["momentum_score"] = (
        df["momentum_1m"] * 0.4 + df["momentum_3m"] * 0.4 + df["momentum_acceleration"] * 0.2
    )
    return df


def _calculate_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    market_avg = (
        (df["avg_return_3m"] * df.get("sector_market_cap", 1).fillna(1)).sum()
        / df.get("sector_market_cap", 1).fillna(1).sum()
    )
    df["relative_strength"] = df["avg_return_3m"] - market_avg
    df["relative_strength_score"] = _pct(df["relative_strength"])
    df["outperforming"] = df["relative_strength"] > 0
    return df


def _calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["volatility_score"] = 100 - _pct(df.get("volatility_1m", 0))
    df["risk_adjusted_return"] = np.where(
        df.get("volatility_1m", 0) > 0,
        df["avg_return_3m"] / df["volatility_1m"],
        df["avg_return_3m"],
    )
    df["risk_adjusted_score"] = _pct(df["risk_adjusted_return"])
    df["downside_risk"] = (
        (df["avg_return_1m"] < 0)
        & (df.get("volatility_1m", 0) > df.get("volatility_1m", 0).median())
    ).astype(int)
    return df


def _detect_sector_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    def _row_anomalies(r: pd.Series) -> List[str]:
        anomalies: List[str] = []
        if abs(r["momentum_1m"] - r["momentum_3m"]) > 40:
            anomalies.append("momentum_divergence")
        if r.get("avg_volume_surge", 1) > 2:
            anomalies.append("volume_surge")
        if abs(r["relative_strength"]) > df["avg_return_3m"].std() * 2:
            anomalies.append("extreme_performance")
        if r["avg_return_1m"] * r["avg_return_3m"] < 0 and abs(r["avg_return_1m"]) > 5:
            anomalies.append("trend_reversal")
        return anomalies

    df["anomalies"] = df.apply(_row_anomalies, axis=1)
    df["anomaly_count"] = df["anomalies"].str.len()
    df["has_anomaly"] = df["anomaly_count"] > 0
    return df


def _calculate_rotation_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["rotation_momentum"] = (
        df["momentum_acceleration"] * 0.5
        + df["relative_strength_score"] * 0.3
        + _pct(df.get("avg_volume_surge", 1)) * 0.2
    )
    df["entry_signal"] = (
        (df["momentum_3m"] < 30)
        & (df["momentum_1m"] > df["momentum_3m"])
        & (df["momentum_acceleration"] > 60)
    ).astype(int)
    df["exit_signal"] = (
        (df["momentum_3m"] > 70)
        & (df["momentum_1m"] < df["momentum_3m"])
        & (df["momentum_acceleration"] < 40)
    ).astype(int)

    df["rotation_phase"] = df.apply(_determine_rotation_phase, axis=1)
    return df


def _calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["sector_score"] = (
        df["momentum_score"] * WEIGHTS["momentum"]
        + df.get("avg_quality_score", 50) * WEIGHTS["quality"]
        + df["risk_adjusted_score"] * WEIGHTS["risk_adjusted"]
        + df["relative_strength_score"] * WEIGHTS["relative_strength"]
        + df["rotation_momentum"] * WEIGHTS["rotation"]
    )
    df["sector_rank"] = _rank_desc(df["sector_score"]).astype(int)
    df["percentile_rank"] = _pct(df["sector_score"])
    return df


def _assign_rotation_status(df: pd.DataFrame) -> pd.DataFrame:
    def _status(r: pd.Series) -> str:
        sc = r["sector_score"]
        m = r["momentum_score"]
        rs = r["relative_strength_score"]
        if r["entry_signal"] and sc > 60:
            return SectorStatus.BREAKOUT.value
        if r["exit_signal"] and sc < 40:
            return SectorStatus.BREAKDOWN.value
        if sc >= 80 and m >= 70:
            return SectorStatus.HOT.value
        if sc >= 65 and rs >= 60:
            return SectorStatus.WARMING.value
        if 35 <= sc < 65:
            return SectorStatus.NEUTRAL.value
        if 20 <= sc < 35 and m < 40:
            return SectorStatus.COOLING.value
        return SectorStatus.COLD.value

    df["rotation_status"] = df.apply(_status, axis=1)
    df["status_score"] = df["rotation_status"].map(STATUS_SCORES)
    return df


def _generate_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    def _signals(r: pd.Series) -> List[str]:
        sig: List[str] = []
        if r["rotation_status"] == "Breakout":
            sig.append(f"BREAKOUT: Enter {r['sector']}")
        if r["rotation_status"] == "Breakdown":
            sig.append(f"BREAKDOWN: Exit {r['sector']}")
        if r["momentum_acceleration"] > 80 and r["momentum_score"] < 70:
            sig.append("MOMENTUM ACCEL")
        if r["momentum_3m"] < 20 and r.get("avg_quality_score", 50) > 70:
            sig.append("QUALITY VALUE REVERSAL")
        if r["relative_strength"] > 10 and r["momentum_score"] > 60:
            sig.append("RS LEADER")
        if "volume_surge" in r["anomalies"]:
            sig.append("VOLUME SURGE")
        return sig

    df["edge_signals"] = df.apply(_signals, axis=1)
    df["signal_count"] = df["edge_signals"].str.len()
    df["edge_priority"] = df["signal_count"] * 10 + df["status_score"] + df["anomaly_count"] * 5
    return df


def _determine_rotation_phase(r: pd.Series) -> str:
    m1, m3, rs = r["momentum_1m"], r["momentum_3m"], r["relative_strength"]
    if m3 < 30 and m1 > m3:
        return "Accumulation"
    if 30 < m3 < 70 and rs > 0:
        return "Advancing"
    if m3 > 70 and m1 < m3:
        return "Distribution"
    if m3 > 50 and rs < 0:
        return "Declining"
    return "Neutral"


def _format_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "sector",
        "sector_score",
        "sector_rank",
        "rotation_status",
        "rotation_phase",
        "momentum_score",
        "relative_strength",
        "risk_adjusted_score",
        "sector_count",
        "edge_signals",
        "anomaly_count",
    ]
    df = df.sort_values(["edge_priority", "sector_score"], ascending=[False, False])
    df.reset_index(drop=True, inplace=True)
    num_cols = df.select_dtypes(float).columns
    df[num_cols] = df[num_cols].round(2)
    df["edge_summary"] = df["edge_signals"].apply(lambda x: " | ".join(x[:2]))
    return df[[c for c in display_cols if c in df.columns] + ["edge_summary"]]

# ---------------------------------------------------------------------------
# Dashboard helpers (unchanged API)
# ---------------------------------------------------------------------------

def get_sector_heatmap_data(sector_df: pd.DataFrame) -> pd.DataFrame:  # noqa: D400
    metrics = [
        "momentum_score",
        "relative_strength_score",
        "risk_adjusted_score",
        "volatility_score",
        "rotation_momentum",
    ]
    return sector_df[["sector", *[m for m in metrics if m in sector_df.columns]]].set_index("sector")


def get_rotation_summary(sector_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "hot_sectors": sector_df.loc[sector_df["rotation_status"] == "Hot", "sector"].tolist(),
        "cold_sectors": sector_df.loc[sector_df["rotation_status"] == "Cold", "sector"].tolist(),
        "breakout_sectors": sector_df.loc[sector_df["rotation_status"] == "Breakout", "sector"].tolist(),
        "sectors_with_signals": sector_df.loc[sector_df["signal_count"] > 0, "sector"].tolist(),
        "market_breadth": float((sector_df["relative_strength"] > 0).mean()),
        "rotation_intensity": float(sector_df["momentum_acceleration"].std()),
    }


def get_sector_recommendations(sector_df: pd.DataFrame, top_n: int = 3) -> List[Dict[str, str]]:
    rec: List[Dict[str, str]] = []
    top_mom = sector_df.nlargest(top_n, "momentum_score")
    for _, s in top_mom.iterrows():
        rec.append(
            {
                "sector": s["sector"],
                "action": "BUY",
                "rationale": f"Strong momentum ({s['momentum_score']:.0f}), {s['rotation_phase']}",
                "confidence": "High" if s["sector_score"] > 75 else "Medium",
            }
        )
    val = sector_df[(sector_df["momentum_3m"] < 30) & (sector_df.get("avg_quality_score", 50) > 60)]
    if not val.empty:
        best = val.iloc[val["avg_quality_score"].idxmax()]
        rec.append(
            {
                "sector": best["sector"],
                "action": "ACCUMULATE",
                "rationale": "Oversold quality sector, potential reversal",
                "confidence": "Medium",
            }
        )
    return rec

# ---------------------------------------------------------------------------
# Self‑test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo_df = pd.DataFrame(
            {
                "sector": ["IT", "Banking", "Energy", "Auto"],
                "avg_return_1m": [4, -2, 1, 3],
                "avg_return_3m": [6, -5, 2, 5],
                "volatility_1m": [3, 4, 2, 3],
            }
        )
        out = run_sector_mapper(demo_df)
        print(out.head())
