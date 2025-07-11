# sector_mapper.py (PURE DATA-DRIVEN, BULLETPROOF, FINAL)

import pandas as pd

def run_sector_mapper(sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps sector rotation and strength for dashboard use.
    Input: sector_df with columns 'sector', 'sector_avg_3m', 'sector_count'.
    Output: DataFrame with ['sector', 'sector_score', 'sector_rank', 'rotation_status', 'sector_count'].
    """
    df = sector_df.copy()

    # 1. Normalize columns and sector names
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
    if "sector_avg_3m" in df:
        df["sector_avg_3m"] = pd.to_numeric(df["sector_avg_3m"], errors="coerce").fillna(0)

    # 2. Calculate percentile-based sector score (0-100)
    df["sector_score"] = df["sector_avg_3m"].rank(pct=True, na_option="bottom") * 100

    # 3. Rank (1 = best sector)
    df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)

    # 4. Rotation tag: Hot = top 25%, Moderate = next 25%, else Weak
    total = len(df)
    quartile = max(1, int(round(total * 0.25)))
    half = max(1, int(round(total * 0.5)))
    df["rotation_status"] = "Weak"
    if total >= 3:
        df.loc[df["sector_rank"] <= quartile, "rotation_status"] = "Hot"
        df.loc[(df["sector_rank"] > quartile) & (df["sector_rank"] <= half), "rotation_status"] = "Moderate"

    # 5. Essential output columns, sorted by score descending
    cols = [c for c in [
        "sector", "sector_score", "sector_rank", "rotation_status", "sector_count"
    ] if c in df]
    out = df[cols].sort_values("sector_score", ascending=False).reset_index(drop=True)
    return out
