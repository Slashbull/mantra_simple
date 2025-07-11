# filters.py (FINAL, BULLETPROOF, FOR M.A.N.T.R.A.)

import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Return sorted, non-empty unique values from df[column]."""
    if column not in df:
        return []
    vals = df[column].dropna().astype(str).str.strip()
    return sorted(v for v in vals.unique() if v)

def _is_all(selected: List[str], all_values: List[str]) -> bool:
    """Treat as 'all' if nothing selected or all values selected."""
    return not selected or set(selected) == set(all_values)

def apply_basic_filters(
    df: pd.DataFrame,
    selected_tags: List[str],
    min_score: float,
    selected_sectors: List[str],
    selected_categories: List[str],
    debug: bool = False
) -> pd.DataFrame:
    """Basic filter by tag, score, sector, category."""
    trace = []
    out = df.copy()
    start = len(out)

    all_tags = get_unique_values(out, "tag")
    if "tag" in out and not _is_all(selected_tags, all_tags):
        out = out[out["tag"].isin(selected_tags)]
    trace.append(f"tags {start}->{len(out)}")

    if "final_score" in out:
        before = len(out)
        out = out[out["final_score"] >= min_score]
        trace.append(f"score {before}->{len(out)}")

    all_sectors = get_unique_values(out, "sector")
    if "sector" in out and not _is_all(selected_sectors, all_sectors):
        before = len(out)
        out = out[out["sector"].isin(selected_sectors)]
        trace.append(f"sectors {before}->{len(out)}")

    all_cats = get_unique_values(out, "category")
    if "category" in out and not _is_all(selected_categories, all_cats):
        before = len(out)
        out = out[out["category"].isin(selected_categories)]
        trace.append(f"cats {before}->{len(out)}")

    if debug:
        logger.info("basic_filters → " + " | ".join(trace))
    return out.reset_index(drop=True)

def apply_dma_filter(
    df: pd.DataFrame, dma_option: str, debug: bool = False
) -> pd.DataFrame:
    """Filter by DMA option."""
    out = df.copy()
    start = len(out)
    trace = []
    if dma_option == "Above 50D" and {"price", "sma_50d"}.issubset(out):
        out = out[out["price"] > out["sma_50d"]]
        trace.append(f"DMA50 {start}->{len(out)}")
    elif dma_option == "Above 200D" and {"price", "sma_200d"}.issubset(out):
        out = out[out["price"] > out["sma_200d"]]
        trace.append(f"DMA200 {start}->{len(out)}")
    if debug and trace:
        logger.info("dma_filter → " + " | ".join(trace))
    return out.reset_index(drop=True)

def apply_eps_growth_filter(
    df: pd.DataFrame, strong_eps_only: bool, debug: bool = False
) -> pd.DataFrame:
    """EPS filter (score ≥ 60) if strong_eps_only."""
    out = df.copy()
    start = len(out)
    if strong_eps_only and "eps_score" in out:
        out = out[out["eps_score"] >= 60]
        if debug:
            logger.info(f"eps_growth_filter → {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_exclude_high_filter(
    df: pd.DataFrame, exclude_high: bool, debug: bool = False
) -> pd.DataFrame:
    """Exclude stocks within 5% of 52W high."""
    out = df.copy()
    start = len(out)
    if exclude_high and "from_high_pct" in out:
        out = out[out["from_high_pct"] > 5]
        if debug:
            logger.info(f"exclude_high_filter → {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_anomaly_only_filter(
    df: pd.DataFrame, anomaly_only: bool, debug: bool = False
) -> pd.DataFrame:
    """Only return anomalies if anomaly_only is True."""
    out = df.copy()
    start = len(out)
    if anomaly_only and "anomaly" in out:
        out = out[out["anomaly"]]
        if debug:
            logger.info(f"anomaly_only_filter → {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_strategy_preset(
    df: pd.DataFrame, preset: str, debug: bool = False
) -> pd.DataFrame:
    """Apply a common preset filter."""
    out = df.copy()
    start = len(out)
    if preset == "High Momentum" and "momentum_score" in out:
        out = out[out["momentum_score"] >= 80]
        if debug:
            logger.info(f"strategy: High Momentum {start}->{len(out)}")
    elif preset == "Low PE + EPS Jumpers" and {"pe", "eps_score"}.issubset(out):
        out = out[(out["pe"] < 25) & (out["eps_score"] >= 70)]
        if debug:
            logger.info(f"strategy: Low PE + EPS Jumpers {start}->{len(out)}")
    elif preset == "Base Buy Zones" and "from_low_pct" in out:
        out = out[out["from_low_pct"] <= 15]
        if debug:
            logger.info(f"strategy: Base Buy Zones {start}->{len(out)}")
    elif preset == "Volume Spike" and "vol_ratio_1d_90d" in out:
        out = out[out["vol_ratio_1d_90d"] >= 3]
        if debug:
            logger.info(f"strategy: Volume Spike {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_search(
    df: pd.DataFrame, search_ticker: str, debug: bool = False
) -> pd.DataFrame:
    """Filter for ticker contains string (case-insensitive)."""
    out = df.copy()
    start = len(out)
    if search_ticker and "ticker" in out:
        out = out[out["ticker"].str.contains(search_ticker, case=False, na=False)]
        if debug:
            logger.info(f"search_ticker '{search_ticker}' {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_sort(
    df: pd.DataFrame, sort_by: str, ascending: bool = False, debug: bool = False
) -> pd.DataFrame:
    """Case-insensitive, robust sort."""
    out = df.copy()
    start = len(out)
    sort_key = sort_by.strip().lower()
    valid_cols = [c.lower() for c in out.columns]
    if sort_key in valid_cols:
        sort_col = out.columns[valid_cols.index(sort_key)]
        out = out.sort_values(by=sort_col, ascending=ascending)
        if debug:
            logger.info(f"sort_by {sort_by} {start}->{len(out)}")
    return out.reset_index(drop=True)

def apply_smart_filters(
    df: pd.DataFrame,
    selected_tags: List[str],
    min_score: float,
    selected_sectors: List[str],
    selected_categories: List[str],
    dma_option: str = "No filter",
    eps_only: bool = False,
    exclude_high: bool = False,
    anomaly_only: bool = False,
    preset: str = "None",
    search_ticker: str = "",
    sort_by: str = "final_score",
    ascending: bool = False,
    debug: bool = False,
    return_metrics: bool = False,
) -> Any:
    """All-in-one filtering pipeline. If return_metrics=True, returns (df, step_counts)."""
    metrics: Dict[str, int] = {}
    out = df

    metrics["initial"] = len(out)
    out = apply_basic_filters(out, selected_tags, min_score, selected_sectors, selected_categories, debug)
    metrics["after_basic"] = len(out)
    out = apply_dma_filter(out, dma_option, debug)
    metrics["after_dma"] = len(out)
    out = apply_eps_growth_filter(out, eps_only, debug)
    metrics["after_eps"] = len(out)
    out = apply_exclude_high_filter(out, exclude_high, debug)
    metrics["after_exclude_high"] = len(out)
    out = apply_anomaly_only_filter(out, anomaly_only, debug)
    metrics["after_anomaly"] = len(out)
    out = apply_strategy_preset(out, preset, debug)
    metrics["after_preset"] = len(out)
    out = apply_search(out, search_ticker, debug)
    metrics["after_search"] = len(out)
    out = apply_sort(out, sort_by, ascending, debug)
    metrics["final"] = len(out)

    if debug:
        logger.info("FILTER PIPELINE COUNTS: " + " | ".join(f"{k}:{v}" for k, v in metrics.items()))
    return (out.reset_index(drop=True), metrics) if return_metrics else out.reset_index(drop=True)

# --- Unique Extractors for Dashboard UI ---

def get_unique_tags(df: pd.DataFrame) -> List[str]:
    return get_unique_values(df, "tag")

def get_unique_sectors(df: pd.DataFrame) -> List[str]:
    return get_unique_values(df, "sector")

def get_unique_categories(df: pd.DataFrame) -> List[str]:
    return get_unique_values(df, "category")
