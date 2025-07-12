"""
alert_engine.py – M.A.N.T.R.A. Elite Alert Detection Engine
==========================================================
Production‑grade alert system with zero false positives,
maximum signal clarity, and infinite extensibility.
Built for a power user who demands perfection.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
import pandas as pd

__all__ = [
    "AlertPriority",
    "AlertCategory",
    "Alert",
    "AlertRule",
    "HighConvictionBuyRule",
    "MultiSpikeAnomalyRule",
    "SectorRotationRule",
    "FundamentalBreakoutRule",
    "MomentumSurgeRule",
    "CustomVolumeBreakoutRule",
    "AlertEngine",
    "detect_alerts",
    "format_alerts",
]

# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------


class AlertPriority(Enum):
    """Alert priority levels for actionability and urgency"""

    CRITICAL = 5  # Immediate action required (major opportunities/risks)
    HIGH = 4  # Review within hours (strong signals)
    MEDIUM = 3  # Review today (notable patterns)
    LOW = 2  # Review this week (emerging trends)
    INFO = 1  # Informational only (context)


class AlertCategory(Enum):
    """Alert categories for filtering and routing"""

    BUY_SIGNAL = "buy_signal"
    ANOMALY = "anomaly"
    SECTOR_ROTATION = "sector_rotation"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VALUE = "value"
    RISK = "risk"


# ---------------------------------------------------------------------------
# CORE ALERT OBJECT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """Immutable alert containing all signal information"""

    alert_id: str
    category: AlertCategory
    priority: AlertPriority
    title: str
    message: str
    tickers: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)

    def __post_init__(self):  # type: ignore[override]
        # dataclass with frozen=True calls __setattr__ via object.__setattr__
        if not self.alert_id:
            content = f"{self.category.value}_{self.title}_{','.join(sorted(self.tickers))}"
            object.__setattr__(
                self, "alert_id", hashlib.md5(content.encode()).hexdigest()[:12]
            )

    # ---------------------------------------------------------------------
    # SERIALISATION / DISPLAY HELPERS
    # ---------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "tickers": self.tickers,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "conditions": self.conditions,
        }

    def summary(self, max_tickers: int = 5) -> str:
        ticker_display = ", ".join(self.tickers[:max_tickers])
        if len(self.tickers) > max_tickers:
            ticker_display += f" (+{len(self.tickers) - max_tickers} more)"

        emoji = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚡",
            AlertPriority.MEDIUM: "📊",
            AlertPriority.LOW: "📌",
            AlertPriority.INFO: "ℹ️",
        }.get(self.priority, "")

        return f"{emoji} {self.title}: {ticker_display}"


# ---------------------------------------------------------------------------
# ALERT RULE BASE CLASS
# ---------------------------------------------------------------------------


class AlertRule(ABC):
    """Abstract base class for all alert rules"""

    def __init__(self, name: str, category: AlertCategory, enabled: bool = True):
        self.name = name
        self.category = category
        self.enabled = enabled

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        """Return an Alert if triggered, else None"""

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame, required: List[str]) -> bool:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[AlertRule:{self.name}] missing columns → {missing}")
            return False
        return True


# ---------------------------------------------------------------------------
# STANDARD RULES
# ---------------------------------------------------------------------------


class HighConvictionBuyRule(AlertRule):
    """Buy tag + high score (+ optional momentum)"""

    def __init__(self, min_score: float = 80.0, min_tickers: int = 1, require_momentum: bool = True):
        super().__init__("high_conviction_buy", AlertCategory.BUY_SIGNAL)
        self.min_score = min_score
        self.min_tickers = min_tickers
        self.require_momentum = require_momentum

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        if not self._validate_columns(df, ["ticker", "tag", "final_score"]):
            return None

        mask = (df["tag"] == "Buy") & (df["final_score"] >= self.min_score)
        if self.require_momentum and "momentum_score" in df.columns:
            mask &= df["momentum_score"] > 60

        candidates = df[mask]
        if len(candidates) < self.min_tickers:
            return None

        candidates = candidates.sort_values("final_score", ascending=False)
        avg_score = candidates["final_score"].mean()

        priority = (
            AlertPriority.CRITICAL
            if len(candidates) >= 5 and avg_score >= 85
            else AlertPriority.HIGH
            if len(candidates) >= 3 or avg_score >= 83
            else AlertPriority.MEDIUM
        )

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="High‑Conviction Buy Signals",
            message=f"{len(candidates)} stocks ≥ {self.min_score} (avg score {avg_score:.1f})",
            tickers=candidates["ticker"].tolist(),
            timestamp=datetime.now(),
            metadata={"count": len(candidates), "avg_score": round(avg_score, 2)},
            conditions=[f"final_score ≥ {self.min_score}", "tag == Buy"],
        )


class MultiSpikeAnomalyRule(AlertRule):
    """Detects clusters of spike_score anomalies (optionally volume‑confirmed)"""

    def __init__(self, min_spike_score: float = 3.0, min_anomalies: int = 2, check_volume: bool = True):
        super().__init__("multi_spike_anomaly", AlertCategory.ANOMALY)
        self.min_spike_score = min_spike_score
        self.min_anomalies = min_anomalies
        self.check_volume = check_volume

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        required = ["ticker", "spike_score"]
        if not self._validate_columns(df, required):
            return None

        mask = df["spike_score"] >= self.min_spike_score
        if "anomaly" in df.columns:
            mask &= df["anomaly"]
        if self.check_volume and "volume_spike" in df.columns:
            mask &= df["volume_spike"] > 2.0

        anomalies = df[mask]
        if len(anomalies) < self.min_anomalies:
            return None

        avg_spike = anomalies["spike_score"].mean()
        max_spike = anomalies["spike_score"].max()

        sector_conc = 0.0
        if "sector" in anomalies.columns and not anomalies["sector"].empty:
            sector_conc = anomalies["sector"].value_counts(normalize=True).iloc[0] * 100

        priority = (
            AlertPriority.CRITICAL
            if len(anomalies) >= 5 and avg_spike >= 4.0
            else AlertPriority.HIGH
            if len(anomalies) >= 3 and avg_spike >= 3.5
            else AlertPriority.MEDIUM
        )

        message = (
            f"{len(anomalies)} anomalies (avg {avg_spike:.1f}, max {max_spike:.1f})"
            + (f" – {sector_conc:.0f}% in one sector" if sector_conc > 40 else "")
        )

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="Multi‑Spike Anomaly Cluster",
            message=message,
            tickers=anomalies["ticker"].tolist(),
            timestamp=datetime.now(),
            metadata={"avg_spike": round(avg_spike, 2), "max_spike": round(max_spike, 2)},
            conditions=[f"spike_score ≥ {self.min_spike_score}", f"count ≥ {self.min_anomalies}"],
        )


class SectorRotationRule(AlertRule):
    """Flags hot/cold sectors based on sector_score mean"""

    def __init__(self, hot_threshold: float = 85.0, cold_threshold: float = 30.0, min_sector_stocks: int = 5):
        super().__init__("sector_rotation", AlertCategory.SECTOR_ROTATION)
        self.hot = hot_threshold
        self.cold = cold_threshold
        self.min_sector_stocks = min_sector_stocks

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        if not self._validate_columns(df, ["ticker", "sector", "sector_score"]):
            return None

        stats = (
            df.groupby("sector")
            .agg(avg_score=("sector_score", "mean"), count=("ticker", "count"))
            .reset_index()
        )
        stats = stats[stats["count"] >= self.min_sector_stocks]
        hot = stats[stats["avg_score"] >= self.hot]
        cold = stats[stats["avg_score"] <= self.cold]
        if hot.empty and cold.empty:
            return None

        parts, tickers = [], []
        if not hot.empty:
            top_hot = ", ".join(hot.sort_values("avg_score", ascending=False)["sector"].tolist())
            parts.append(f"HOT: {top_hot}")
            for sec in hot["sector"].tolist()[:2]:
                tickers.extend(df[df["sector"] == sec].nlargest(3, "final_score")["ticker"].tolist())
        if not cold.empty:
            top_cold = ", ".join(cold.sort_values("avg_score")["sector"].tolist())
            parts.append(f"COLD: {top_cold}")

        priority = AlertPriority.HIGH if not hot.empty and hot["avg_score"].max() >= 90 else AlertPriority.MEDIUM

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="Sector Rotation Signal",
            message=" | ".join(parts),
            tickers=tickers[:10],
            timestamp=datetime.now(),
            metadata={"hot": hot.to_dict("records"), "cold": cold.to_dict("records")},
            conditions=[f"hot ≥ {self.hot}", f"cold ≤ {self.cold}"],
        )


class FundamentalBreakoutRule(AlertRule):
    """Watch‑tag stocks with high eps_score and (optionally) attractive PE"""

    def __init__(self, min_eps_score: float = 80.0, max_pe: float = 25.0):
        super().__init__("fundamental_breakout", AlertCategory.FUNDAMENTAL)
        self.min_eps = min_eps_score
        self.max_pe = max_pe

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        if not self._validate_columns(df, ["ticker", "tag", "eps_score"]):
            return None

        mask = (df["tag"] == "Watch") & (df["eps_score"] >= self.min_eps)
        if "pe" in df.columns:
            value_mask = (df["pe"] > 0) & (df["pe"] <= self.max_pe)
            candidates = df[mask & value_mask]
            if candidates.empty:
                candidates = df[mask]
        else:
            candidates = df[mask]
        if candidates.empty:
            return None

        avg_eps = candidates["eps_score"].mean()
        priority = AlertPriority.HIGH if avg_eps >= self.min_eps + 10 else AlertPriority.MEDIUM

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="Fundamental Breakout Candidates",
            message=f"{len(candidates)} Watch stocks (avg EPS {avg_eps:.1f})",
            tickers=candidates["ticker"].tolist(),
            timestamp=datetime.now(),
            metadata={"avg_eps": round(avg_eps, 2)},
            conditions=[f"eps_score ≥ {self.min_eps}", "tag == Watch"],
        )


class MomentumSurgeRule(AlertRule):
    """Short + medium‑term return surge (volume optional)"""

    def __init__(self, short_term_threshold: float = 5.0, medium_term_threshold: float = 10.0, min_stocks: int = 3, check_volume: bool = True):
        super().__init__("momentum_surge", AlertCategory.MOMENTUM)
        self.st = short_term_threshold
        self.mt = medium_term_threshold
        self.min_stocks = min_stocks
        self.check_volume = check_volume

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        short_col = next((c for c in ["ret_7d", "return_1w"] if c in df.columns), None)
        medium_col = next((c for c in ["ret_30d", "return_1m"] if c in df.columns), None)
        if not short_col or not medium_col:
            return None

        mask = (df[short_col] >= self.st) & (df[medium_col] >= self.mt)
        if self.check_volume:
            vol_col = next((c for c in ["vol_ratio_1d_90d", "volume_ratio"] if c in df.columns), None)
            if vol_col:
                mask &= df[vol_col] > 1.5

        surges = df[mask]
        if len(surges) < self.min_stocks:
            return None

        surges = surges.assign(combined=surges[short_col] + surges[medium_col]).sort_values("combined", ascending=False)
        avg_short = surges[short_col].mean()
        avg_medium = surges[medium_col].mean()

        priority = AlertPriority.CRITICAL if len(surges) >= 5 and avg_short >= 10 else AlertPriority.HIGH if len(surges) >= 3 and avg_short >= 7 else AlertPriority.MEDIUM

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="Momentum Surge Detected",
            message=f"{len(surges)} stocks surged ({avg_short:.1f}%/wk, {avg_medium:.1f}%/mo)",
            tickers=surges["ticker"].tolist(),
            timestamp=datetime.now(),
            metadata={"avg_short": round(avg_short, 2), "avg_medium": round(avg_medium, 2)},
            conditions=[f"{short_col} ≥ {self.st}%", f"{medium_col} ≥ {self.mt}%"],
        )


# ---------------------------------------------------------------------------
# EXAMPLE CUSTOM RULE
# ---------------------------------------------------------------------------


class CustomVolumeBreakoutRule(AlertRule):
    """Detects huge 1‑day volume spike + price pop"""

    def __init__(self, volume_multiplier: float = 3.0, price_change_pct: float = 2.0):
        super().__init__("volume_breakout", AlertCategory.VOLUME)
        self.mult = volume_multiplier
        self.pct = price_change_pct

    def evaluate(self, df: pd.DataFrame, context: Dict[str, Any]) -> Optional[Alert]:
        if not self._validate_columns(df, ["ticker", "vol_ratio_1d_90d", "ret_1d"]):
            return None

        mask = (df["vol_ratio_1d_90d"] >= self.mult) & (df["ret_1d"] >= self.pct)
        hits = df[mask]
        if hits.empty:
            return None

        priority = AlertPriority.HIGH if len(hits) >= 3 else AlertPriority.MEDIUM

        return Alert(
            alert_id="",
            category=self.category,
            priority=priority,
            title="Volume Breakout",
            message=f"{len(hits)} stocks volume ≥ {self.mult}× & price ≥ {self.pct}%",
            tickers=hits["ticker"].tolist(),
            timestamp=datetime.now(),
            metadata={"volume_multiplier": self.mult, "price_change_pct": self.pct},
            conditions=[f"vol_ratio_1d_90d ≥ {self.mult}", f"ret_1d ≥ {self.pct}%"],
        )


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------


class AlertEngine:
    """Master orchestrator running registered AlertRules"""

    def __init__(self, rules: Optional[List[AlertRule]] = None, enable_deduplication: bool = True, history_limit: int = 1000):
        self.rules: Dict[str, AlertRule] = {}
        self.enable_dedup = enable_deduplication
        self.history_limit = history_limit
        self.alert_history: List[Alert] = []
        self.last_hashes: Set[str] = set()
        for rule in (rules if rules is not None else self._default_rules()):
            self.register_rule(rule)

    # ------------------------------------------------------------------
    # RULE MANAGEMENT
    # ------------------------------------------------------------------

    def _default_rules(self) -> List[AlertRule]:
        return [
            HighConvictionBuyRule(),
            MultiSpikeAnomalyRule(),
            SectorRotationRule(),
            FundamentalBreakoutRule(),
            MomentumSurgeRule(),
        ]

    def register_rule(self, rule: AlertRule) -> None:
        self.rules[rule.name] = rule

    def unregister_rule(self, name: str) -> bool:
        return self.rules.pop(name, None) is not None

    def get_available_rules(self) -> List[str]:
        return list(self.rules.keys())

    # ------------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------------

    def detect_alerts(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> List[Alert]:
        if df.empty:
            return []
        context = context or {}
        alerts: List[Alert] = []
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            try:
                alert = rule.evaluate(df, context)
                if alert is not None:
                    alerts.append(alert)
            except Exception as exc:
                print(f"[AlertEngine] rule '{rule.name}' errored → {exc}")
        if self.enable_dedup:
            alerts = self._deduplicate(alerts)
        self._update_history(alerts)
        return alerts

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _deduplicate(self, alerts: List[Alert]) -> List[Alert]:
        fresh, hashes = [], set()
        for alert in alerts:
            h = hashlib.md5(f"{alert.category.value}_{alert.title}_{','.join(sorted(alert.tickers))}".encode()).hexdigest()
            if h not in self.last_hashes:
                fresh.append(alert)
                hashes.add(h)
        self.last_hashes = hashes
        return fresh

    def _update_history(self, alerts: List[Alert]) -> None:
        self.alert_history.extend(alerts)
        if len(self.alert_history) > self.history_limit:
            self.alert_history = self.alert_history[-self.history_limit :]

    # ------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------

    def format_alerts(self, alerts: List[Alert], mode: str = "summary") -> str:
        if not alerts:
            return "✅ No new alerts detected."
        if mode == "json":
            return json.dumps([a.to_dict() for a in alerts], indent=2)
        if mode not in {"summary", "detailed"}:
            raise ValueError(f"Unknown format mode '{mode}'")

        if mode == "summary":
            lines = ["🎯 ALERTS", "--------------------------"]
            for pr in sorted({a.priority for a in alerts}, reverse=True):
                lines.append(f"\n{pr.name} priority:")
                for a in filter(lambda x: x.priority == pr, alerts):
                    lines.append(f"  {a.summary()}")
            return "\n".join(lines)

        # detailed
        lines = ["📊 DETAILED ALERT REPORT", "=" * 50]
        for idx, a in enumerate(alerts, 1):
            lines.extend([
                f"\n[{idx}] {a.title} ({a.priority.name})",
                f"Msg   : {a.message}",
                f"Tickers: {', '.join(a.tickers[:10])}",
                f"Time  : {a.timestamp:%Y-%m-%d %H:%M:%S}",
                f"Conditions: {' | '.join(a.conditions)}",
            ])
            if a.metadata:
                lines.append(f"Meta : {a.metadata}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"total_alerts": len(self.alert_history)}
        if not self.alert_history:
            return out
        by_cat, by_pri, tickers: Dict[str, int], Dict[str, int], Set[str]
        by_cat, by_pri, tickers = {}, {}, set()
        for a in self.alert_history:
            by_cat[a.category.value] = by_cat.get(a.category.value, 0) + 1
            by_pri[a.priority.name] = by_pri.get(a.priority.name, 0) + 1
            tickers.update(a.tickers)
        out.update({
            "by_category": by_cat,
            "by_priority": by_pri,
            "unique_tickers": len(tickers),
            "last_alert_time": self.alert_history[-1].timestamp.isoformat(),
        })
        return out


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPERS
# ---------------------------------------------------------------------------

def detect_alerts(df: pd.DataFrame, last_hashes: Optional[Set[str]] = None, rules: Optional[List[AlertRule]] = None) -> List[Alert]:
    engine = AlertEngine(rules=rules, enable_deduplication=bool(last_hashes))
    if last_hashes:
        engine.last_hashes = last_hashes
    return engine.detect_alerts(df)


def format_alerts(alerts: Union[List[Alert], Set[str]], mode: str = "summary") -> str:
    if isinstance(alerts, set):  # legacy hash set → no new alerts
        return "✅ No new alerts detected." if not alerts else "Deprecated format"
    engine = AlertEngine()
    return engine.format_alerts(alerts, mode)


# ---------------------------------------------------------------------------
# CLI SELF‑TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔧 Self‑test: building dummy DataFrame …")
    dummy = pd.DataFrame({
        "ticker": ["ABC", "DEF", "GHI", "JKL"],
        "tag": ["Buy", "Buy", "Watch", "Watch"],
        "final_score": [85, 87, 75, 60],
        "momentum_score": [65, 70, 50, 40],
        "ret_7d": [6, 8, 1, -2],
        "ret_30d": [12, 15, 3, -1],
        "vol_ratio_1d_90d": [3.5, 1.2, 0.9, 4.0],
        "ret_1d": [3, 0.5, -0.2, 5],
        "spike_score": [3.2, 0.5, 0.2, 4.1],
        "anomaly": [True, False, False, True],
        "volume_spike": [2.5, 0.8, 1.0, 3.0],
        "sector": ["Tech", "Tech", "Auto", "Auto"],
        "sector_score": [92, 92, 25, 25],
        "eps_score": [82, 85, 90, 60],
        "pe": [18, 21, 11, 35],
    })

    eng = AlertEngine(rules=[
        HighConvictionBuyRule(),
        MultiSpikeAnomalyRule(),
        SectorRotationRule(),
        FundamentalBreakoutRule(),
        MomentumSurgeRule(),
        CustomVolumeBreakoutRule(),
    ])

    alerts_list = eng.detect_alerts(dummy)
    print(eng.format_alerts(alerts_list, mode="summary"))
    print("\nStats:", json.dumps(eng.stats(), indent=2))
