"""health_panel.py – Data‑Health Dashboard v2.1
================================================
Streamlit‑ready module that decouples heavy calculations from UI rendering and
adds stricter type‑checks, faster vectorised null scans, and a pluggable
`HealthConfig` for env‑tuning. Public API is unchanged: `calculate_data_health`,
`render_health_panel`, `assess_data_health`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class HealthConfig:
    nan_warn_pct: float = 50.0
    nan_issue_pct: float = 80.0
    zero_vol_threshold: float = 0.10  # 10 % rows
    score_bounds: tuple[int, int] = (0, 100)
    freshness_hours_warn: int = 72
    critical_columns: List[str] = ("ticker", "price", "final_score", "sector")

CFG = HealthConfig()

# ─────────────────────────────────────────────────────────────
# Core calc
# ─────────────────────────────────────────────────────────────
def _null_stats(df: pd.DataFrame) -> tuple[int, float]:
    nulls = df.isna().values.sum()
    pct = (nulls / df.size * 100) if df.size else 0.0
    return int(nulls), round(pct, 2)

def calculate_data_health(df: pd.DataFrame | None, meta: Dict[str, Any]) -> Dict[str, Any]:
    h = {"score": 100, "status": "healthy", "issues": [], "warnings": [], "tips": [], "metrics": {}}
    if df is None or df.empty:
        h.update({"score": 0, "status": "critical", "issues": ["No data loaded"], "metrics": {"total_rows": 0}})
        return h

    rows, cols = df.shape
    h["metrics"].update({"total_rows": rows, "total_cols": cols, **meta})

    # Nulls
    null_cnt, null_pct = _null_stats(df)
    h["metrics"].update({"nan_count": null_cnt, "nan_percentage": null_pct, "total_cells": df.size})
    if null_pct >= CFG.nan_issue_pct:
        h["issues"].append(f"{null_pct:.1f}% missing values")
        h["score"] -= 15
    elif null_pct >= CFG.nan_warn_pct:
        h["warnings"].append(f"High null percentage: {null_pct:.1f}%")
        h["score"] -= 5

    # Critical cols
    missing = [c for c in CFG.critical_columns if c not in df.columns]
    if missing:
        h["issues"].append("Missing critical cols: " + ", ".join(missing))
        h["score"] -= 30

    # Duplicate tickers
    if "ticker" in df.columns:
        dups = df.duplicated("ticker").sum()
        if dups:
            h["warnings"].append(f"{dups} duplicate tickers")
            h["score"] -= 5
            h["tips"].append("Remove duplicates for accuracy")
        h["metrics"]["duplicates"] = int(dups)

    # Price sanity
    if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
        zero_p = (df.price == 0).sum(); neg_p = (df.price < 0).sum()
        if zero_p: h["warnings"].append(f"{zero_p} zero prices"); h["score"] -= 5
        if neg_p: h["issues"].append(f"{neg_p} negative prices"); h["score"] -= 10
        h["metrics"]["price_range"] = f"₹{df.price.min():.2f}–₹{df.price.max():.2f}"

    # Score bounds
    scorecols = df.select_dtypes(np.number).filter(regex="score", axis=1)
    for col in scorecols:
        oob = ((scorecols[col] < CFG.score_bounds[0]) | (scorecols[col] > CFG.score_bounds[1])).sum()
        if oob:
            h["warnings"].append(f"{col}: {oob} out‑of‑range")
            h["score"] -= 3

    # Volume quality
    volcols = [c for c in df.columns if "volume" in c.lower() and "ratio" not in c.lower()]
    for vc in volcols:
        zeros = (df[vc] == 0).sum()
        if zeros / rows > CFG.zero_vol_threshold:
            h["warnings"].append(f"{vc}: {zeros} zero volume")

    # Freshness
    if "last_updated" in meta:
        try:
            age = (datetime.now() - pd.to_datetime(meta["last_updated"]).to_pydatetime()).total_seconds()/3600
            h["metrics"]["data_age_hours"] = round(age,1)
            if age > CFG.freshness_hours_warn:
                h["warnings"].append(f"Data {age:.0f} h old")
                h["score"] -= 10
                h["tips"].append("Consider a fresh reload")
        except Exception:
            pass

    # Final status mapping
    h["score"] = max(0,min(100,h["score"]))
    if h["score"]>=90: h["status"]="excellent"
    elif h["score"]>=75: h["status"]="good"
    elif h["score"]>=60: h["status"]="fair"
    elif h["score"]>=40: h["status"]="poor"
    else: h["status"]="critical"
    if h["score"]<75 and not h["tips"]:
        h["tips"].append("Review issues and warnings to improve data quality")
    return h

# ─────────────────────────────────────────────────────────────
# Streamlit UI (thin wrapper)
# ─────────────────────────────────────────────────────────────
COLOR_MAP={"excellent":"🟢","good":"🟡","fair":"🟠","poor":"🔴","critical":"🔴"}


def render_health_panel(df: pd.DataFrame, meta: Dict[str, Any]):
    st.sidebar.header("🏥 Data Health Monitor")
    h=calculate_data_health(df,meta)
    col1,col2=st.sidebar.columns([2,1])
    with col1:
        st.markdown(f"### {COLOR_MAP[h['status']]} Score")
        st.markdown(f"<h1 style='margin:0;'>{h['score']}</h1>",unsafe_allow_html=True)
    with col2:
        st.markdown("### Status")
        st.markdown(f"**{h['status'].title()}**")
    st.sidebar.divider()
    st.sidebar.metric("Total Stocks",h['metrics'].get('total_stocks',0))
    st.sidebar.metric("NaN %",f"{h['metrics']['nan_percentage']:.1f}%")
    st.sidebar.metric("Duplicates",h['metrics'].get('duplicates',0))
    if h['issues']:
        st.sidebar.error("\n".join(h['issues']))
    if h['warnings']:
        st.sidebar.warning("\n".join(h['warnings'][:4]))
    if h['tips']:
        st.sidebar.info("\n".join(h['tips'][:3]))

# ─────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    df=pd.DataFrame({"ticker":["AAA","BBB"],"price":[0,100],"sector":["X","Y"],"final_score":[95,50]})
    m={"total_stocks":len(df),"total_sectors":df.sector.nunique(),"load_time":datetime.now(),"source":"demo"}
    print(calculate_data_health(df,m))
