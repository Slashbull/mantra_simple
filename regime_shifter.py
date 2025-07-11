# regime_shifter.py (SIMPLE, PURE DATA-DRIVEN)

from typing import Dict, Optional
import pandas as pd

# Simple regime weights (just dicts, no UI text)
REGIME_WEIGHTS = {
    "balanced":  {"momentum": 0.22, "value": 0.21, "eps": 0.19, "volume": 0.19, "sector": 0.19},
    "momentum":  {"momentum": 0.40, "value": 0.10, "eps": 0.15, "volume": 0.20, "sector": 0.15},
    "value":     {"momentum": 0.10, "value": 0.45, "eps": 0.15, "volume": 0.15, "sector": 0.15},
    "growth":    {"momentum": 0.17, "value": 0.08, "eps": 0.40, "volume": 0.15, "sector": 0.20},
    "volume":    {"momentum": 0.15, "value": 0.10, "eps": 0.10, "volume": 0.50, "sector": 0.15},
}

FACTOR_KEYS = ["momentum", "value", "eps", "volume", "sector"]

def get_regime_weights(regime: str = "balanced") -> Dict[str, float]:
    """Returns the weight dict for the selected regime."""
    return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["balanced"]).copy()

def auto_detect_regime(df: pd.DataFrame) -> str:
    """
    Auto-detects most suitable regime based on *current* data only.
    Returns: regime key ("momentum", "value", "growth", "volume", "balanced")
    """
    try:
        # 1. Momentum regime: If >60% of stocks up >6% in 3M
        if "ret_3m" in df and (df["ret_3m"] > 6).mean() > 0.6:
            return "momentum"
        # 2. Value regime: If median PE < 16 and median 1Y return < 3%
        if "pe" in df and "ret_1y" in df:
            if df["pe"].median() < 16 and df["ret_1y"].median() < 3:
                return "value"
        # 3. Growth regime: If >50% of stocks have EPS % change > 15 in recent qtr
        if "eps_change_pct" in df and (df["eps_change_pct"] > 15).mean() > 0.5:
            return "growth"
        # 4. Volume regime: If >35% of stocks have vol_ratio_1d_90d > 2.2
        if "vol_ratio_1d_90d" in df and (df["vol_ratio_1d_90d"] > 2.2).mean() > 0.35:
            return "volume"
        # Default: balanced
        return "balanced"
    except Exception:
        return "balanced"
