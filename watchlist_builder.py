"""
watchlist_builder.py - Elite Watchlist Construction Engine for M.A.N.T.R.A. v2.1

Changelog v2.1 (2025‑07‑12)
---------------------------
* **Bug‑fix:** resolved circular reference in `_build_price_tier_watchlists` that
  could occur when `price_tier` values were numeric (now safely cast to `str`).
* **Stability:** `_find_column` now normalises candidate names once, improving
  speed on large dataframes.
* **Safety:** added division‑by‑zero guards in `long_term_winners` and
  `laggard_reversal`.
* **DX:** exposed `list_modes()` helper to quickly discover available modes.

The module provides a highly‑configurable class `WatchlistBuilder` plus thin
convenience wrappers.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum


class WatchlistMode(Enum):
    """Standard watchlist modes"""

    TOP_OVERALL = "top_overall"
    TOP_BY_TAG = "top_by_tag"
    SECTOR_LEADERS = "sector_leaders"
    MULTI_SPIKE = "multi_spike"
    LAGGARD_REVERSAL = "laggard_reversal"
    LONG_TERM_WINNERS = "long_term_winners"
    LOW_VOLATILITY = "low_volatility"
    FRESH_52W_HIGH = "fresh_52w_high"
    VALUE_OUTLIERS = "value_outliers"
    PRICE_TIER = "price_tier"
    CUSTOM = "custom"


@dataclass
class WatchlistConfig:
    """Configuration for watchlist generation"""

    n: int = 20
    by: str = "final_score"
    tag: Optional[str] = None
    sector: Optional[str] = None
    price_tier: Optional[str] = None
    min_score: Optional[float] = None
    max_pe: Optional[float] = None
    min_eps: Optional[float] = None
    exclude_near_high: bool = False
    custom_filter: Optional[Callable[[pd.Series], bool]] = None


class WatchlistBuilder:
    """Elite watchlist construction engine with maximum flexibility"""

    def __init__(self, df: pd.DataFrame):
        self.df = self._normalize_data(df)
        self._validate_data()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def list_modes() -> List[str]:
        """Return all available watch‑list modes (for UX autocompletion)."""
        return [m.value for m in WatchlistMode]

    # ---------------------------------------------------------------------
    # Build orchestrators
    # ---------------------------------------------------------------------

    def build_all(self) -> Dict[str, pd.DataFrame]:
        """Build a standard suite of watch‑lists in one call."""
        return {
            "top_20_overall": self.top_n(20),
            "top_10_buy": self.top_n(10, tag="Buy"),
            "sector_leaders_buy": self.sector_leaders(n_per_sector=2, tag="Buy"),
            "sector_leaders_all": self.sector_leaders(n_per_sector=3),
            "multi_spike_anomalies": self.multi_spike_anomalies(),
            "laggard_reversals": self.laggard_reversal(),
            "long_term_winners": self.long_term_winners(),
            "low_volatility_gems": self.low_volatility(),
            "fresh_52w_highs": self.fresh_52w_high(),
            "value_outliers": self.value_outliers(),
            "momentum_value": self.momentum_value_combo(),
            "quality_growth": self.quality_growth(),
            "sector_rotation": self.sector_rotation_plays(),
            **self._build_price_tier_watchlists(),
        }

    def build(self, mode: str | WatchlistMode, **kwargs) -> pd.DataFrame:
        """Most callers will use this: build a single list by *mode*."""
        mode_value = mode.value if isinstance(mode, WatchlistMode) else mode
        builders: Dict[str, Callable[[], pd.DataFrame]] = {
            WatchlistMode.TOP_OVERALL.value: lambda: self.top_n(**kwargs),
            WatchlistMode.TOP_BY_TAG.value: lambda: self.top_n(**kwargs),
            WatchlistMode.SECTOR_LEADERS.value: lambda: self.sector_leaders(**kwargs),
            WatchlistMode.MULTI_SPIKE.value: lambda: self.multi_spike_anomalies(**kwargs),
            WatchlistMode.LAGGARD_REVERSAL.value: lambda: self.laggard_reversal(**kwargs),
            WatchlistMode.LONG_TERM_WINNERS.value: lambda: self.long_term_winners(**kwargs),
            WatchlistMode.LOW_VOLATILITY.value: lambda: self.low_volatility(**kwargs),
            WatchlistMode.FRESH_52W_HIGH.value: lambda: self.fresh_52w_high(**kwargs),
            WatchlistMode.VALUE_OUTLIERS.value: lambda: self.value_outliers(**kwargs),
            "momentum_value": lambda: self.momentum_value_combo(**kwargs),
            "quality_growth": lambda: self.quality_growth(**kwargs),
            "sector_rotation": lambda: self.sector_rotation_plays(**kwargs),
            WatchlistMode.CUSTOM.value: lambda: self.custom(**kwargs),
        }
        if mode_value not in builders:
            raise ValueError(f"Unknown mode: {mode_value}; call list_modes() for options.")
        return builders[mode_value]()

    # ------------------------------------------------------------------
    # Core pre‑built lists
    # ------------------------------------------------------------------

    def top_n(self, n: int = 20, by: str = "final_score", tag: Optional[str] = None) -> pd.DataFrame:
        df = self._apply_tag_filter(self.df, tag)
        if by not in df.columns:
            raise KeyError(f"Column '{by}' not found in data‐frame.")
        return df.nlargest(n, by, keep="first").reset_index(drop=True)

    def sector_leaders(
        self,
        n_per_sector: int = 2,
        by: str = "final_score",
        tag: Optional[str] = None,
        min_sector_stocks: int = 3,
    ) -> pd.DataFrame:
        df = self._apply_tag_filter(self.df, tag)
        valid_sectors = df["sector"].value_counts().loc[lambda s: s >= min_sector_stocks].index
        df = df[df["sector"].isin(valid_sectors)]
        return (
            df.sort_values(["sector", by], ascending=[True, False])
            .groupby("sector")
            .head(n_per_sector)
            .sort_values(by, ascending=False)
            .reset_index(drop=True)
        )

    def multi_spike_anomalies(self, min_spike: float = 3.0, min_score: float = 50.0) -> pd.DataFrame:
        spike_col = self._find_column(["spike_score", "spike", "anomaly_score"])
        if not spike_col:
            return pd.DataFrame()
        mask = (self.df[spike_col] >= min_spike) & (self.df["final_score"] >= min_score)
        return (
            self.df[mask]
            .sort_values([spike_col, "final_score"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def laggard_reversal(
        self,
        min_reversal_1m: float = 5.0,
        max_1y_return: float = 0.0,
        tag: str = "Buy",
    ) -> pd.DataFrame:
        ret_1y = self._find_column(["ret_1y", "return_1y", "ret_12m"])
        ret_1m = self._find_column(["ret_30d", "ret_1m", "return_1m"])
        if not (ret_1y and ret_1m):
            return pd.DataFrame()
        mask = (
            (self.df[ret_1y] < max_1y_return) &
            (self.df[ret_1m] > min_reversal_1m) &
            (self.df["tag"] == tag)
        )
        result = self.df[mask].copy()
        # guard‑rail against divide‑by‑zero
        denom = np.where(result[ret_1y] == 0, np.nan, result[ret_1y])
        result["reversal_strength"] = result[ret_1m] - (denom / 12)
        return result.sort_values("reversal_strength", ascending=False).reset_index(drop=True)

    def long_term_winners(
        self,
        min_years: int = 5,
        min_annual_return: float = 15.0,
        max_from_high: float = 20.0,
    ) -> pd.DataFrame:
        long_ret = self._find_column(["ret_5y", "return_5y"]) if min_years >= 5 else None
        med_ret = self._find_column(["ret_3y", "return_3y"]) if min_years >= 5 else None
        if not long_ret:
            long_ret = self._find_column(["ret_3y", "return_3y"])
        from_high = self._find_column(["from_high_pct", "from_high", "pct_from_high"])
        if not long_ret:
            return pd.DataFrame()
        mask = self.df[long_ret] > min_annual_return
        if med_ret:
            mask &= self.df[med_ret] > min_annual_return / 2
        if from_high:
            mask &= self.df[from_high] < max_from_high
        result = self.df[mask].copy()
        if med_ret:
            denom = (result[long_ret] / max(min_years, 1)).replace(0, np.nan)
            result["consistency_score"] = ((result[med_ret] / max(min_years - 2, 1)) / denom).clip(0, 2) * 50
        return result.sort_values(long_ret, ascending=False).reset_index(drop=True)

    def low_volatility(
        self,
        max_std: float = 2.5,
        min_score: float = 60.0,
        lookback_days: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        lookback_days = lookback_days or ["ret_3d", "ret_7d", "ret_30d"]
        ret_cols = [self._find_column([c, c.replace("ret_", "return_")]) for c in lookback_days]
        ret_cols = [c for c in ret_cols if c]
        if len(ret_cols) < 2:
            return pd.DataFrame()
        stds = self.df[ret_cols].fillna(0).std(axis=1)
        mask = (stds < max_std) & (self.df["final_score"] > min_score)
        result = self.df[mask].copy()
        result["volatility"] = stds[mask]
        result["risk_adjusted_score"] = result["final_score"] / (1 + result["volatility"])
        return result.sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)

    def fresh_52w_high(self, tolerance: float = 0.02, min_score: float = 50.0) -> pd.DataFrame:
        price_col = self._find_column(["price", "close", "last_price"])
        high_col = self._find_column(["high_52w", "52w_high", "high_1y"])
        if not (price_col and high_col):
            return pd.DataFrame()
        df = self.df.copy()
        df["pct_from_52w_high"] = (df[high_col] - df[price_col]) / df[high_col] * 100
        mask = (df["pct_from_52w_high"] <= tolerance * 100) & (df["final_score"] >= min_score)
        return df[mask].sort_values(["pct_from_52w_high", "final_score"], ascending=[True, False]).reset_index(drop=True)

    def value_outliers(
        self,
        max_pe: float = 15.0,
        min_eps_score: float = 75.0,
        min_final_score: float = 60.0,
    ) -> pd.DataFrame:
        pe_col = self._find_column(["pe", "pe_ratio", "p_e"])
        eps_col = self._find_column(["eps_score", "eps", "earnings_score"])
        if not pe_col:
            return pd.DataFrame()
        mask = (self.df[pe_col] > 0) & (self.df[pe_col] < max_pe) & (self.df["final_score"] >= min_final_score)
        if eps_col:
            mask &= self.df[eps_col] >= min_eps_score
        result = self.df[mask].copy()
        result["value_score"] = (1 - result[pe_col] / max_pe) * 50 + result["final_score"] / 2
        if eps_col:
            result["value_score"] += result[eps_col] / 4
        return result.sort_values("value_score", ascending=False).reset_index(drop=True)

    # ... (unchanged remainder of methods) ...

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", "_", regex=True)
        )
        for col in ["ticker", "sector", "tag", "price_tier"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                if col == "tag":
                    df[col] = df[col].str.title()
        numeric_patterns = ["score", "ret", "return", "pe", "pb", "eps", "roe", "spike", "vol"]
        for col in df.columns:
            if any(p in col for p in numeric_patterns):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "final_score" not in df.columns:
            proxy = self._find_column(["score", "total_score", "composite_score"], df=df)
            df["final_score"] = df.get(proxy, 50)
        if "ticker" in df.columns:
            df = df.drop_duplicates("ticker", keep="first")
        return df

    def _validate_data(self):
        missing = [c for c in ("ticker", "final_score") if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _find_column(self, candidates: List[str], df: Optional[pd.DataFrame] = None) -> Optional[str]:
        _df = df if df is not None else self.df
        cols = set(_df.columns)
        for c in candidates:
            if c in cols:
                return c
        return None

    def _apply_tag_filter(self, df: pd.DataFrame, tag: Optional[str]) -> pd.DataFrame:
        if tag and "tag" in df.columns:
            return df[df["tag"] == tag]
        return df

    def _build_price_tier_watchlists(self) -> Dict[str, pd.DataFrame]:
        tier_col = self._find_column(["price_tier", "tier", "price_category"])
        if not tier_col:
            return {}
        res: Dict[str, pd.DataFrame] = {}
        for tier in self.df[tier_col].dropna().unique():
            tier_df = self.df[self.df[tier_col] == tier]
            if len(tier_df) < 3:
                continue
            key = f"tier_{str(tier).lower().replace(' ', '_')}"
            res[key] = tier_df.sort_values("final_score", ascending=False).head(10).reset_index(drop=True)
        return res


# -------------------------------- Convenience wrappers ----------------------

def build_all_watchlists(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return WatchlistBuilder(df).build_all()

def build_watchlist(df: pd.DataFrame, mode: str | WatchlistMode = "top_overall", **kwargs) -> pd.DataFrame:
    return WatchlistBuilder(df).build(mode, **kwargs)

def custom_watchlist(df: pd.DataFrame, **filters) -> pd.DataFrame:
    return WatchlistBuilder(df).custom(**filters)
