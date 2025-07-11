# decision_engine.py (FINAL, SIMPLE, DATA-DRIVEN, FOR M.A.N.T.R.A.)

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import date
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_THRESHOLDS = {
    "buy": 75,        # Final score >= 75 for Buy
    "watch": 60,      # Final score >= 60 for Watch
}

def get_thresholds(custom: Optional[Dict] = None) -> Dict[str, float]:
    thresholds = DEFAULT_THRESHOLDS.copy()
    if custom:
        thresholds.update(custom)
    return thresholds

# ============================================================================
# MAIN SIMPLE DECISION PIPELINE
# ============================================================================

def run_decision_engine(
    df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
    tag_date: Optional[str] = None,
    sort_by: str = "final_score",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Tag each stock as 'Buy', 'Watch', or 'Avoid' based ONLY on final_score.
    No advanced logic, no risk banding, no confidence band, no diagnostics.
    """
    if df.empty or "final_score" not in df.columns:
        logger.warning("No data or missing 'final_score' column. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.copy()
    thresholds = get_thresholds(thresholds)

    # Tag logic
    df["tag"] = "Avoid"
    df.loc[df["final_score"] >= thresholds["watch"], "tag"] = "Watch"
    df.loc[df["final_score"] >= thresholds["buy"], "tag"] = "Buy"

    # Tag colors for UI
    tag_color_map = {"Buy": "green", "Watch": "orange", "Avoid": "red"}
    df["tag_color"] = df["tag"].map(tag_color_map).fillna("gray")

    # Basic confidence (just scaled final_score for simple mode)
    df["confidence"] = df["final_score"].clip(0, 100)
    df["confidence_band"] = pd.cut(
        df["confidence"],
        bins=[-np.inf, 60, 80, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    # Basic signal strength for UI (text)
    df["signal_strength"] = df["final_score"].apply(
        lambda x: "🔥 Explosive" if x >= 90 else
                  "⚡ Strong" if x >= 80 else
                  "↑ Solid" if x >= 70 else
                  "→ Moderate" if x >= 60 else
                  "↓ Weak"
    )

    # No target, no risk, no reason in simple mode.
    df["tag_reason"] = ""
    df["avoid_reason"] = ""

    # Tag date (optional)
    if not tag_date:
        tag_date = str(date.today())
    df["tag_date"] = tag_date

    # Order by tag & score for UI
    tag_order = {"Buy": 2, "Watch": 1, "Avoid": 0}
    df["_tag_order"] = df["tag"].map(tag_order).fillna(0)
    df = df.sort_values(["_tag_order", sort_by], ascending=[False, ascending]).drop(columns=["_tag_order"])

    # Primary UI columns
    primary_cols = [
        "ticker", "company_name", "sector", "tag", "tag_color",
        "final_score", "signal_strength", "confidence", "confidence_band",
        "price", "tag_date"
    ]
    # Fill missing UI columns with blanks (if any)
    for col in primary_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[primary_cols + [c for c in df.columns if c not in primary_cols]]
    return df.reset_index(drop=True)

# ============================================================================
# OPTIONAL SIMPLE DIAGNOSTIC (NOT USED IN MAIN APP)
# ============================================================================

def simple_engine_diagnostics(df: pd.DataFrame) -> List[str]:
    """Very basic summary — for UI, not for quant analysis."""
    counts = df["tag"].value_counts()
    buy = counts.get("Buy", 0)
    watch = counts.get("Watch", 0)
    avoid = counts.get("Avoid", 0)
    total = len(df)
    msg = f"Buy: {buy} | Watch: {watch} | Avoid: {avoid} | Total: {total}"
    if buy == 0:
        return [msg, "⚠️ No Buy signals detected!"]
    elif buy > 0 and buy / max(total, 1) > 0.5:
        return [msg, "⚠️ More than half stocks are Buy. Maybe thresholds are too low."]
    else:
        return [msg]

# ============================================================================
# END OF FILE
# ============================================================================
