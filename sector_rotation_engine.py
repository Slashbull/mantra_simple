# sector_rotation_engine.py (PURE DATA-DRIVEN, FINAL, SIMPLE)

import pandas as pd
from typing import Dict, Any

def compute_sector_rotation(
    sector_df: pd.DataFrame,
    metric: str = "sector_avg_3m"
) -> pd.DataFrame:
    """
    Computes sector rotation tags based on the specified return metric.
    Returns DataFrame with ['sector', 'sector_score', 'sector_rank', 'rotation_status', 'sector_count'].
    """
    df = sector_df.copy()
    # Normalize
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^\w\s]", "", regex=True)
                  .str.replace(r"\s+", "_", regex=True)
    )
    if "sector" in df:
        df["sector"] = df["sector"].astype(str).str.strip().str.title()
    if "sector_count" in df:
        df["sector_count"] = pd.to_numeric(df["sector_count"], errors="coerce").fillna(0).astype(int)
    if metric in df:
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    else:
        raise ValueError(f"Column '{metric}' not found in sector_df.")

    # Calculate scores and rank
    df["sector_score"] = df[metric].rank(pct=True, na_option="bottom") * 100
    df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)

    # Quartile logic for rotation tag
    total = len(df)
    quart = max(1, int(round(total * 0.25)))
    half = max(1, int(round(total * 0.5)))
    df["rotation_status"] = "Weak"
    if total >= 3:
        df.loc[df["sector_rank"] <= quart, "rotation_status"] = "Hot"
        df.loc[(df["sector_rank"] > quart) & (df["sector_rank"] <= half), "rotation_status"] = "Moderate"

    # Output essential columns
    cols = [c for c in [
        "sector", "sector_score", "sector_rank", "rotation_status", "sector_count"
    ] if c in df]
    out = df[cols].sort_values("sector_score", ascending=False).reset_index(drop=True)
    return out

def sector_rotation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a simple summary: counts of each rotation tag and top sectors.
    """
    counts = df["rotation_status"].value_counts().to_dict()
    top_sectors = list(df[df["rotation_status"] == "Hot"]["sector"].head(3))
    return {
        "total_sectors": len(df),
        "hot_count": counts.get("Hot", 0),
        "moderate_count": counts.get("Moderate", 0),
        "weak_count": counts.get("Weak", 0),
        "hot_sectors": top_sectors,
    }
