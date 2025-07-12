"""
filters.py – Ultimate Data Filtering Engine for M.A.N.T.R.A. Stock Intelligence System

Production-grade filtering pipeline with bullet-proof error handling, infinite extensibility,
and crystal-clear architecture. Built to last decades without modification.

Version: 1.0.0-FINAL
Status:  PRODUCTION-READY – DO NOT REWRITE
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────
class FilterMode(Enum):
    """Logical operators for combining filters."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


class ComparisonOperator(Enum):
    """Supported comparison operators for ColumnFilter."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NE = "!="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    REGEX = "regex"

# ─────────────────────────────────────────────────────────────
# Dataclass helpers
# ─────────────────────────────────────────────────────────────
@dataclass
class FilterResult:
    """
    Container for pipeline output plus rich metrics.
    """
    data: pd.DataFrame
    metrics: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    applied_filters: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    # Convenience properties
    @property
    def rows_filtered(self) -> int:
        if {"initial", "final"} <= self.metrics.keys():
            return self.metrics["initial"] - self.metrics["final"]
        return 0

    @property
    def filter_rate(self) -> float:
        if self.metrics.get("initial", 0):
            return self.rows_filtered / self.metrics["initial"] * 100
        return 0.0

# ─────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────
def safe_filter(func: Callable) -> Callable:
    """
    Ensure individual filters never crash the pipeline.
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("%s: received empty DataFrame", func.__name__)
            return df if df is not None else pd.DataFrame()

        try:
            result = func(df, *args, **kwargs)
            if result is None:
                logger.error("%s: filter returned None; using original DataFrame", func.__name__)
                return df
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: %s — returning original DataFrame", func.__name__, exc)
            return df

    return wrapper

# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────
class FilterBase(ABC):
    """
    Abstract base for all filters.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._execution_count = 0
        self._total_rows_filtered = 0

    @abstractmethod
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

    # Optional helpers
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "execution_count": self._execution_count,
            "total_rows_filtered": self._total_rows_filtered,
        }

    def validate_columns(
        self, df: pd.DataFrame, required: Sequence[str]
    ) -> Tuple[bool, List[str]]:
        missing = [c for c in required if c not in df.columns]
        return not missing, missing

# ─────────────────────────────────────────────────────────────
# Concrete filters
# ─────────────────────────────────────────────────────────────
class ColumnFilter(FilterBase):
    """
    Single-column filter with rich operator support.
    """

    def __init__(
        self,
        name: str,
        column: str,
        operator: Union[str, ComparisonOperator],
        value: Any,
        *,
        null_handling: str = "exclude",  # exclude | include | treat_as_zero
        case_sensitive: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(name, enabled)
        self.column = column
        self.operator = (
            operator if isinstance(operator, ComparisonOperator) else ComparisonOperator(operator)
        )
        self.value = value
        self.null_handling = null_handling
        self.case_sensitive = case_sensitive

    @safe_filter
    def apply(self, df: pd.DataFrame, **_) -> pd.DataFrame:  # noqa: D401
        if not self.enabled or self.column not in df.columns:
            return df

        self._execution_count += 1
        initial_len = len(df)

        col = df[self.column].copy()
        null_mask = col.isna()

        # ── handle nulls
        if self.null_handling == "exclude":
            valid = ~null_mask
        elif self.null_handling == "include":
            valid = pd.Series(True, index=df.index)
        else:  # treat_as_zero
            col = col.fillna(0)
            valid = pd.Series(True, index=df.index)

        # ── operator logic
        op = self.operator
        if op == ComparisonOperator.GT:
            mask = valid & (col > self.value)
        elif op == ComparisonOperator.GTE:
            mask = valid & (col >= self.value)
        elif op == ComparisonOperator.LT:
            mask = valid & (col < self.value)
        elif op == ComparisonOperator.LTE:
            mask = valid & (col <= self.value)
        elif op == ComparisonOperator.EQ:
            mask = valid & (col == self.value)
        elif op == ComparisonOperator.NE:
            mask = valid & (col != self.value)
        elif op == ComparisonOperator.BETWEEN:
            if isinstance(self.value, Sequence) and len(self.value) == 2:
                lo, hi = self.value
                mask = valid & (col >= lo) & (col <= hi)
            else:
                logger.error("%s: BETWEEN requires [min, max] list/tuple", self.name)
                return df
        elif op == ComparisonOperator.IN:
            vals = self.value if isinstance(self.value, (list, set, tuple)) else [self.value]
            mask = valid & col.isin(vals)
        elif op == ComparisonOperator.NOT_IN:
            vals = self.value if isinstance(self.value, (list, set, tuple)) else [self.value]
            mask = valid & ~col.isin(vals)
        elif op == ComparisonOperator.CONTAINS:
            pattern = str(self.value)
            if not self.case_sensitive:
                col = col.astype(str).str.lower()
                pattern = pattern.lower()
            mask = valid & col.astype(str).str.contains(pattern, na=False)
        elif op == ComparisonOperator.REGEX:
            mask = valid & col.astype(str).str.match(
                self.value, case=self.case_sensitive, na=False
            )
        else:  # pragma: no cover
            logger.error("%s: unknown operator %s", self.name, op)
            return df

        result = df[mask]
        self._total_rows_filtered += initial_len - len(result)
        return result


class MultiColumnFilter(FilterBase):
    """
    Combine several FilterBase objects with a boolean mode.
    """

    def __init__(
        self,
        name: str,
        filters: List[FilterBase],
        mode: FilterMode = FilterMode.AND,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(name, enabled)
        self.filters = filters
        self.mode = mode

    @safe_filter
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self.enabled or not self.filters:
            return df

        self._execution_count += 1
        initial_len = len(df)

        masks: List[pd.Series] = []
        for f in self.filters:
            if not f.enabled:
                continue
            filtered = f.apply(df, **kwargs)
            masks.append(df.index.isin(filtered.index))

        if not masks:
            return df

        # ── combine
        if self.mode == FilterMode.AND:
            combined = np.logical_and.reduce(masks)
        elif self.mode == FilterMode.OR:
            combined = np.logical_or.reduce(masks)
        elif self.mode == FilterMode.NOT:
            combined = ~masks[0]
        elif self.mode == FilterMode.XOR:
            combined = np.logical_xor.reduce(masks)
        else:  # pragma: no cover
            logger.error("%s: unknown mode %s", self.name, self.mode)
            return df

        result = df[combined]
        self._total_rows_filtered += initial_len - len(result)
        return result


class CustomFilter(FilterBase):
    """
    Accepts an arbitrary callable(df) → filtered_df.
    """

    def __init__(
        self,
        name: str,
        filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(name, enabled)
        self.filter_func = filter_func

    @safe_filter
    def apply(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        if not self.enabled:
            return df

        self._execution_count += 1
        initial_len = len(df)

        try:
            result = self.filter_func(df)
            if not isinstance(result, pd.DataFrame):
                logger.error("%s: custom filter must return DataFrame", self.name)
                return df
            self._total_rows_filtered += initial_len - len(result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: custom filter error – %s", self.name, exc)
            return df


class PresetFilter(FilterBase):
    """
    Convenience wrapper around common multi-step strategies.
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "high_momentum": {
            "filters": [
                ("momentum_score", ComparisonOperator.GTE, 80),
                ("final_score", ComparisonOperator.GTE, 70),
            ],
            "mode": FilterMode.AND,
        },
        "value_plays": {
            "filters": [
                ("pe", ComparisonOperator.GT, 0),
                ("pe", ComparisonOperator.LT, 25),
                ("eps_score", ComparisonOperator.GTE, 70),
            ],
            "mode": FilterMode.AND,
        },
        "breakout_candidates": {
            "filters": [
                ("from_high_pct", ComparisonOperator.BETWEEN, [0, 10]),
                ("vol_ratio_1d_90d", ComparisonOperator.GTE, 1.5),
                ("momentum_score", ComparisonOperator.GTE, 60),
            ],
            "mode": FilterMode.AND,
        },
        "oversold_quality": {
            "filters": [
                ("rsi", ComparisonOperator.LT, 30),
                ("final_score", ComparisonOperator.GTE, 70),
                ("from_low_pct", ComparisonOperator.LTE, 20),
            ],
            "mode": FilterMode.AND,
        },
        "volume_surge": {
            "filters": [
                ("vol_ratio_1d_90d", ComparisonOperator.GTE, 3),
                ("price", ComparisonOperator.GT, 0),
            ],
            "mode": FilterMode.AND,
        },
    }

    def __init__(self, preset_name: str, *, enabled: bool = True) -> None:
        if preset_name not in self.PRESETS:
            raise ValueError(
                f"Unknown preset {preset_name}. Choices: {list(self.PRESETS.keys())}"
            )
        super().__init__(f"preset_{preset_name}", enabled)
        self.preset_name = preset_name
        self._build_filters()

    def _build_filters(self) -> None:
        cfg = self.PRESETS[self.preset_name]
        flts = [
            ColumnFilter(
                f"{self.preset_name}_{i}", col, op, val
            )
            for i, (col, op, val) in enumerate(cfg["filters"])
        ]
        self.multi = MultiColumnFilter(self.preset_name, flts, cfg["mode"])

    @safe_filter
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self.enabled:
            return df
        return self.multi.apply(df, **kwargs)

# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
class FilterPipeline:
    """
    Executes a sequence of filters with full telemetry.
    """

    def __init__(
        self,
        *,
        track_metrics: bool = True,
        warn_on_empty: bool = True,
        debug: bool = False,
    ) -> None:
        self.track_metrics = track_metrics
        self.warn_on_empty = warn_on_empty
        self.debug = debug
        self.filters: List[FilterBase] = []
        self._history: List[FilterResult] = []

    # Builder helpers
    def add_filter(self, f: FilterBase) -> "FilterPipeline":
        self.filters.append(f)
        return self

    def remove_filter(self, name: str) -> bool:
        idx = next((i for i, f in enumerate(self.filters) if f.name == name), None)
        if idx is None:
            return False
        self.filters.pop(idx)
        return True

    def clear_filters(self) -> "FilterPipeline":
        self.filters.clear()
        return self

    # Core
    def apply(
        self,
        df: pd.DataFrame,
        *,
        return_metrics: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, FilterResult]:
        start = time.time()

        if df is None or df.empty:
            result = FilterResult(
                data=df if df is not None else pd.DataFrame(),
                warnings=["Input DataFrame is empty"],
            )
            return result if return_metrics else result.data

        result = FilterResult(data=df.copy())
        result.metrics["initial"] = len(df)
        current = df.copy()

        for f in self.filters:
            if not f.enabled:
                continue

            before = len(current)
            if self.debug:
                logger.debug("Applying filter %s", f.name)

            current = f.apply(current, **kwargs)

            if self.track_metrics:
                result.metrics[f"after_{f.name}"] = len(current)
                result.applied_filters.append(f.name)

            if current.empty and self.warn_on_empty:
                warn = f"Filter '{f.name}' eliminated all rows"
                result.warnings.append(warn)
                if self.debug:
                    logger.warning(warn)

            if self.debug and before != len(current):
                logger.debug("  %s: %d → %d rows", f.name, before, len(current))

        result.data = current
        result.metrics["final"] = len(current)
        result.execution_time_ms = (time.time() - start) * 1000
        self._history.append(result)
        self._history = self._history[-100:]  # keep last 100

        return result if return_metrics else result.data

    # Telemetry
    def get_pipeline_stats(self) -> Dict[str, Any]:
        stats = {
            "total_filters": len(self.filters),
            "enabled_filters": sum(f.enabled for f in self.filters),
            "filter_details": [f.get_stats() for f in self.filters],
            "total_executions": len(self._history),
        }
        if self._history:
            stats.update(
                {
                    "avg_rows_filtered": round(
                        np.mean([h.rows_filtered for h in self._history]), 2
                    ),
                    "avg_execution_time_ms": round(
                        np.mean([h.execution_time_ms for h in self._history]), 2
                    ),
                    "total_warnings": sum(len(h.warnings) for h in self._history),
                }
            )
        return stats

# ─────────────────────────────────────────────────────────────
# Builder helpers (for Streamlit UI, etc.)
# ─────────────────────────────────────────────────────────────
def _add_filters(pipeline: FilterPipeline, filters: Sequence[FilterBase]) -> None:
    for f in filters:
        pipeline.add_filter(f)


def build_basic_filters(
    selected_tags: Optional[List[str]] = None,
    min_score: float = 0,
    selected_sectors: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
) -> List[FilterBase]:
    flts: List[FilterBase] = []
    if selected_tags:
        flts.append(ColumnFilter("tag_filter", "tag", ComparisonOperator.IN, selected_tags))
    if min_score > 0:
        flts.append(ColumnFilter("score_filter", "final_score", ComparisonOperator.GTE, min_score))
    if selected_sectors:
        flts.append(ColumnFilter("sector_filter", "sector", ComparisonOperator.IN, selected_sectors))
    if selected_categories:
        flts.append(
            ColumnFilter("category_filter", "category", ComparisonOperator.IN, selected_categories)
        )
    return flts


def build_technical_filters(
    dma_option: str = "No filter",
    exclude_near_high: bool = False,
    rsi_range: Optional[Tuple[float, float]] = None,
) -> List[FilterBase]:
    flts: List[FilterBase] = []

    if dma_option == "Above 50D":
        flts.append(
            CustomFilter(
                "dma_50_filter",
                lambda df: df[df["price"] > df["sma_50d"]]
                if {"price", "sma_50d"} <= set(df.columns)
                else df,
            )
        )
    elif dma_option == "Above 200D":
        flts.append(
            CustomFilter(
                "dma_200_filter",
                lambda df: df[df["price"] > df["sma_200d"]]
                if {"price", "sma_200d"} <= set(df.columns)
                else df,
            )
        )

    if exclude_near_high:
        flts.append(
            ColumnFilter("exclude_high", "from_high_pct", ComparisonOperator.GT, 5),
        )

    if rsi_range and len(rsi_range) == 2:
        flts.append(ColumnFilter("rsi_filter", "rsi", ComparisonOperator.BETWEEN, rsi_range))

    return flts


def build_fundamental_filters(
    eps_growth_only: bool = False,
    pe_range: Optional[Tuple[float, float]] = None,
    min_roe: Optional[float] = None,
) -> List[FilterBase]:
    flts: List[FilterBase] = []

    if eps_growth_only:
        flts.append(ColumnFilter("eps_filter", "eps_score", ComparisonOperator.GTE, 60))

    if pe_range and len(pe_range) == 2:
        flts.append(
            MultiColumnFilter(
                "pe_range_filter",
                [
                    ColumnFilter("pe_min", "pe", ComparisonOperator.GTE, pe_range[0]),
                    ColumnFilter("pe_max", "pe", ComparisonOperator.LTE, pe_range[1]),
                    ColumnFilter("pe_valid", "pe", ComparisonOperator.GT, 0),
                ],
                FilterMode.AND,
            )
        )

    if min_roe is not None:
        flts.append(ColumnFilter("roe_filter", "roe", ComparisonOperator.GTE, min_roe))

    return flts

# ─────────────────────────────────────────────────────────────
# High-level helper
# ─────────────────────────────────────────────────────────────
def apply_smart_filters(
    df: pd.DataFrame,
    *,
    selected_tags: Optional[List[str]] = None,
    min_score: float = 0,
    selected_sectors: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    dma_option: str = "No filter",
    eps_only: bool = False,
    exclude_high: bool = False,
    anomaly_only: bool = False,
    preset: str = "None",
    search_ticker: str = "",
    sort_by: str = "final_score",
    ascending: bool = False,
    custom_filters: Optional[List[FilterBase]] = None,
    debug: bool = False,
    return_metrics: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, int]]]:
    pipeline = FilterPipeline(track_metrics=True, warn_on_empty=True, debug=debug)

    _add_filters(pipeline, build_basic_filters(selected_tags, min_score, selected_sectors, selected_categories))
    _add_filters(pipeline, build_technical_filters(dma_option, exclude_high))
    _add_filters(pipeline, build_fundamental_filters(eps_only))

    if anomaly_only:
        pipeline.add_filter(ColumnFilter("anomaly_filter", "anomaly", ComparisonOperator.EQ, True))

    if preset and preset.lower() != "none":
        try:
            pipeline.add_filter(PresetFilter(preset.lower().replace(" ", "_")))
        except ValueError:
            logger.warning("Unknown preset: %s", preset)

    if search_ticker:
        pipeline.add_filter(
            ColumnFilter(
                "search_filter",
                "ticker",
                ComparisonOperator.CONTAINS,
                search_ticker,
                case_sensitive=False,
            )
        )

    if custom_filters:
        _add_filters(pipeline, custom_filters)

    result = pipeline.apply(df, return_metrics=True)
    if sort_by in result.data.columns:
        result.data = result.data.sort_values(sort_by, ascending=ascending)

    if return_metrics:
        return result.data, result.metrics
    return result.data

# ─────────────────────────────────────────────────────────────
# Quick utilities
# ─────────────────────────────────────────────────────────────
def _unique(
    df: pd.DataFrame, column: str, *, dropna: bool = True, sort: bool = True
) -> List[str]:
    if column not in df.columns:
        return []
    vals = df[column]
    if dropna:
        vals = vals.dropna()
    vals = vals.astype(str).str.strip()
    uniq = [v for v in vals.unique() if v and v != "nan"]
    return sorted(uniq) if sort else uniq


get_unique_tags = lambda df: _unique(df, "tag")  # noqa: E731
get_unique_sectors = lambda df: _unique(df, "sector")  # noqa: E731
get_unique_categories = lambda df: _unique(df, "category")  # noqa: E731


def validate_dataframe(
    df: Optional[pd.DataFrame],
    required_columns: Optional[Sequence[str]] = None,
    *,
    min_rows: int = 0,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if df is None:
        return False, ["DataFrame is None"]
    if df.empty:
        issues.append("DataFrame is empty")
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows; minimum {min_rows} required")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
    return not issues, issues

# ─────────────────────────────────────────────────────────────
# Convenience global instance (optional)
# ─────────────────────────────────────────────────────────────
default_pipeline = FilterPipeline()

# EOF – filters.py v1.0.0-FINAL
