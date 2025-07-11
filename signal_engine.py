# signal_engine.py (FINAL SIMPLE VERSION – FOR M.A.N.T.R.A. SIMPLE DASHBOARD)

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_SCORE = 50.0

class SimpleSignalEngine:
    """
    Simple, robust signal engine for M.A.N.T.R.A. Simple Dashboard.
    Uses core data-driven factors: Momentum, Value, EPS, Volume, Sector.
    No advanced bonus/penalty logic.
    """

    def __init__(
        self,
        momentum_weights: Dict[str, float],
        value_weight: float,
        volume_weight: float,
        eps_weight: float,
        sector_weight: float,
        debug: bool = False
    ):
        # Normalize weights to sum 1.0
        total = (
            sum(momentum_weights.values())
            + value_weight
            + volume_weight
            + eps_weight
            + sector_weight
        )
        if not np.isclose(total, 1.0, rtol=1e-3):
            if debug:
                logger.warning(f"Total weight = {total:.3f}, normalizing to 1.0")
            factor = 1.0 / total
            momentum_weights = {k: v * factor for k, v in momentum_weights.items()}
            value_weight *= factor
            volume_weight *= factor
            eps_weight *= factor
            sector_weight *= factor

        self.momentum_weights = momentum_weights
        self.value_weight = value_weight
        self.volume_weight = volume_weight
        self.eps_weight = eps_weight
        self.sector_weight = sector_weight
        self.debug = debug

    @classmethod
    def balanced(cls) -> "SimpleSignalEngine":
        return cls(
            momentum_weights={"ret_3d": 0.10, "ret_7d": 0.20, "ret_30d": 0.30, "ret_3m": 0.40},
            value_weight=0.20,
            volume_weight=0.20,
            eps_weight=0.20,
            sector_weight=0.10,
            debug=False
        )

    def _ensure_cols(self, df: pd.DataFrame, cols: List[str], fill: float = 0.0) -> pd.DataFrame:
        for col in cols:
            if col not in df:
                df[col] = fill
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
        return df

    def _percentile_score(self, series: pd.Series, ascending=True) -> pd.Series:
        if series.notna().sum() == 0:
            return pd.Series(DEFAULT_SCORE, index=series.index)
        ranked = series.rank(method="average", ascending=ascending, pct=True) * 100
        return ranked.fillna(DEFAULT_SCORE)

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.momentum_weights.keys())
        df = self._ensure_cols(df, cols)
        pct_scores = []
        for col in cols:
            pct_scores.append(self._percentile_score(df[col], ascending=True))
        weights = np.array([self.momentum_weights[c] for c in cols])
        arr = np.vstack([s.values for s in pct_scores])
        df["momentum_score"] = np.average(arr, axis=0, weights=weights)
        return df

    def compute_value(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_cols(df, ["pe", "eps_current"])
        value_ratio = pd.Series(0.0, index=df.index)
        mask = (df["pe"] > 0) & (df["eps_current"] > 0)
        value_ratio[mask] = df.loc[mask, "eps_current"] / df.loc[mask, "pe"]
        df["value_score"] = self._percentile_score(value_ratio, ascending=True)
        return df

    def compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d"]
        df = self._ensure_cols(df, cols)
        combo = 0.5 * df["vol_ratio_1d_90d"] + 0.3 * df["vol_ratio_7d_90d"] + 0.2 * df["vol_ratio_30d_90d"]
        df["volume_score"] = self._percentile_score(combo, ascending=True)
        return df

    def compute_eps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_cols(df, ["eps_change_pct"])
        df["eps_score"] = self._percentile_score(df["eps_change_pct"].clip(lower=-50, upper=200), ascending=True)
        return df

    def compute_sector(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        df["sector"] = df.get("sector", "Unknown").fillna("Unknown").astype(str)
        if sector_df is not None and not sector_df.empty:
            sector_map = pd.Series(
                pd.to_numeric(sector_df.set_index("sector")["sector_avg_3m"], errors="coerce"),
                index=sector_df["sector"]
            )
            df["sector_score"] = df["sector"].map(sector_map).fillna(sector_map.mean() if len(sector_map) else DEFAULT_SCORE)
            df["sector_score"] = self._percentile_score(df["sector_score"], ascending=True)
        else:
            df["sector_score"] = DEFAULT_SCORE
        return df

    def fit_transform(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.compute_momentum(df)
        df = self.compute_value(df)
        df = self.compute_volume(df)
        df = self.compute_eps(df)
        df = self.compute_sector(df, sector_df)
        score_cols = ["momentum_score", "value_score", "volume_score", "eps_score", "sector_score"]
        weights = [
            sum(self.momentum_weights.values()), self.value_weight,
            self.volume_weight, self.eps_weight, self.sector_weight
        ]
        arr = np.vstack([df[c].fillna(DEFAULT_SCORE).values for c in score_cols])
        df["final_score"] = np.average(arr, axis=0, weights=weights)
        df["final_score"] = df["final_score"].clip(0, 100).round(2)
        df["final_rank"] = df["final_score"].rank(method="min", ascending=False).fillna(999999).astype(int)
        return df

def run_signal_engine(
    df: pd.DataFrame,
    sector_df: pd.DataFrame,
    debug: bool = False
) -> pd.DataFrame:
    """
    Run simple, pure data-driven signal scoring engine.
    Returns DataFrame with factor and final scores, rank.
    """
    if df.empty:
        logger.error("Empty dataframe provided to signal engine")
        return df

    engine = SimpleSignalEngine.balanced()
    result = engine.fit_transform(df, sector_df)
    if debug:
        logger.info(f"Signal engine scored {len(result)} stocks")
    return result

# END OF FILE
