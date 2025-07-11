# edge_finder.py (Minimal, All-Logic, Data-Driven)

import pandas as pd

def compute_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds key 'edge' flags and summary edge labels to each row.
    """
    df = df.copy()

    # 1. Momentum breakout (recent, near high)
    df["edge_momentum_breakout"] = (
        (df.get("ret_3d", 0) > 2) &
        (df.get("ret_7d", 0) > 5) &
        (df.get("price", 0) >= 0.98 * df.get("high_52w", 0))
    )

    # 2. Value outlier (cheap PE, strong EPS)
    df["edge_value_outlier"] = (
        (df.get("pe", 99) < 18) &
        (df.get("eps_score", 0) > 80) &
        (df.get("final_score", 0) > 70)
    )

    # 3. Sector leader
    df["edge_sector_leader"] = (
        (df.get("sector_score", 0) > 85) &
        (df.get("final_score", 0) > 80)
    )

    # 4. Volatility squeeze (low recent moves, high volume)
    df["edge_volatility_squeeze"] = (
        (df.get("ret_30d", 0).abs() < 2.5) &
        (df.get("ret_7d", 0).abs() < 1) &
        (df.get("vol_ratio_1d_90d", 0) > 2)
    )

    # 5. Fresh new high/low
    df["edge_new_high"] = df.get("price", 0) >= df.get("high_52w", 0)
    df["edge_new_low"]  = df.get("price", 0) <= df.get("low_52w", 0)

    # Label edges for UI or export
    edge_cols = [
        "edge_momentum_breakout", "edge_value_outlier", "edge_sector_leader",
        "edge_volatility_squeeze", "edge_new_high", "edge_new_low"
    ]
    pretty_labels = {
        "edge_momentum_breakout": "Momentum Breakout",
        "edge_value_outlier": "Value Outlier",
        "edge_sector_leader": "Sector Leader",
        "edge_volatility_squeeze": "Vol Squeeze",
        "edge_new_high": "New 52W High",
        "edge_new_low":  "New 52W Low"
    }
    df["edge_types"] = df.apply(
        lambda r: ", ".join([pretty_labels[col] for col in edge_cols if r.get(col, False)]),
        axis=1
    )
    df["has_edge"] = df[edge_cols].any(axis=1)
    df["edge_count"] = df[edge_cols].sum(axis=1)

    return df

def find_edges(df: pd.DataFrame, min_edges: int = 1) -> pd.DataFrame:
    """
    Return all stocks with >= min_edges.
    """
    df = compute_edge_signals(df)
    return df[df["edge_count"] >= min_edges].copy()

# (Optional) Small summary function
def edge_overview(df: pd.DataFrame):
    total = int(df["edge_count"].sum()) if "edge_count" in df else 0
    return {
        "total_edges": total,
        "top_edges": df[df["edge_count"] >= 2][["ticker", "edge_types"]].head(5).to_dict("records") if "edge_count" in df else []
    }
