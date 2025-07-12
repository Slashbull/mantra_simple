"""
regime_shifter.py – Market Regime Detector & Weight Optimizer
=============================================================
Production‑ready module that classifies the prevailing market regime and
returns an appropriate factor‑weight allocation.  
This rewrite focuses on:
• Safer math (NaN‑aware helpers, tolerance‑based validation)  
• Clearer typing & dataclasses  
• Extensible regime dictionary (easy to plug in new regimes)  
• API‑stability with the original draft

Author: OpenAI o3 – AI Quant Architect
License: Proprietary – M.A.N.T.R.A. System
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "RegimeType",
    "RegimeMetrics",
    "RegimeDetectionResult",
    "REGIME_WEIGHTS",
    "get_regime_weights",
    "auto_detect_regime",
    "get_regime_info",
    "validate_weights",
]

# ---------------------------------------------------------------------------
# ENUMS & DATA CLASSES
# ---------------------------------------------------------------------------


class RegimeType(str, Enum):
    BALANCED = "balanced"
    MOMENTUM = "momentum"
    VALUE = "value"
    GROWTH = "growth"
    VOLATILITY = "volatility"
    DEFENSIVE = "defensive"
    RECOVERY = "recovery"


@dataclass
class RegimeMetrics:
    momentum_strength: float = 0.0
    value_opportunity: float = 0.0
    growth_prevalence: float = 0.0
    volatility_level: float = 0.0
    market_breadth: float = 0.0
    volume_surge: float = 0.0
    defensive_need: float = 0.0
    recovery_potential: float = 0.0


@dataclass
class RegimeDetectionResult:
    regime: str
    confidence: float
    weights: Dict[str, float]
    metrics: RegimeMetrics
    explanation: str
    recommendations: List[str]
    timestamp: datetime


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

REQUIRED_FACTORS: set[str] = {"momentum", "value", "eps", "volume", "sector"}

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {"momentum": 0.20, "value": 0.20, "eps": 0.20, "volume": 0.20, "sector": 0.20},
    "momentum": {"momentum": 0.35, "value": 0.10, "eps": 0.15, "volume": 0.25, "sector": 0.15},
    "value": {"momentum": 0.10, "value": 0.40, "eps": 0.20, "volume": 0.15, "sector": 0.15},
    "growth": {"momentum": 0.15, "value": 0.10, "eps": 0.35, "volume": 0.20, "sector": 0.20},
    "volatility": {"momentum": 0.15, "value": 0.15, "eps": 0.15, "volume": 0.35, "sector": 0.20},
    "defensive": {"momentum": 0.10, "value": 0.30, "eps": 0.25, "volume": 0.10, "sector": 0.25},
    "recovery": {"momentum": 0.25, "value": 0.25, "eps": 0.20, "volume": 0.20, "sector": 0.10},
}


# ---------------------------------------------------------------------------
# PUBLIC APIS
# ---------------------------------------------------------------------------

def get_regime_weights(regime: str | RegimeType = RegimeType.BALANCED) -> Dict[str, float]:
    """Return a **copy** of the weight map for *regime*. Defaults to *balanced*."""
    key = regime.value if isinstance(regime, RegimeType) else str(regime)
    return REGIME_WEIGHTS.get(key, REGIME_WEIGHTS["balanced"]).copy()


def auto_detect_regime(df: pd.DataFrame, *, min_confidence: float = 0.60) -> RegimeDetectionResult:
    """Statistically determine the prevailing market regime."""
    metrics = _calc_metrics(df)
    scores: Dict[str, float] = {
        "momentum": _score_momentum(metrics),
        "value": _score_value(metrics),
        "growth": _score_growth(metrics),
        "volatility": _score_volatility(metrics),
        "defensive": _score_defensive(metrics),
        "recovery": _score_recovery(metrics),
        "balanced": 0.50,  # baseline
    }
    selected, confidence = max(scores.items(), key=lambda kv: kv[1])
    if confidence < min_confidence:
        selected, confidence = "balanced", 0.50
    weights = get_regime_weights(selected)
    explanation = _make_explanation(selected, metrics, confidence)
    recs = _make_recommendations(selected, metrics)
    return RegimeDetectionResult(selected, confidence, weights, metrics, explanation, recs, datetime.utcnow())


def get_regime_info(regime: str | RegimeType) -> Dict[str, Any]:
    """Return static metadata & weights for *regime*."""
    key = regime.value if isinstance(regime, RegimeType) else str(regime)
    base = {
        "balanced": ("Balanced", "Evenly diversified across factors", "Medium"),
        "momentum": ("Momentum", "Ride prevailing trends", "High"),
        "value": ("Value", "Seek undervaluation", "Medium"),
        "growth": ("Growth", "Back earnings expansion", "Medium‑High"),
        "volatility": ("Volatility", "Exploit price swings", "Very High"),
        "defensive": ("Defensive", "Capital preservation", "Low"),
        "recovery": ("Recovery", "Play post‑selloff rebounds", "Medium‑High"),
    }.get(key, ("Balanced", "Even distribution", "Medium"))
    return {
        "name": base[0],
        "description": base[1],
        "risk_level": base[2],
        "weights": get_regime_weights(key),
    }


def validate_weights(weights: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    """Ensure *weights* cover the **required** factors and sum≈1.0."""
    missing = REQUIRED_FACTORS - weights.keys()
    if missing:
        return False, f"Missing factors: {sorted(missing)}"
    total = sum(weights.get(f, 0.0) for f in REQUIRED_FACTORS)
    if not np.isclose(total, 1.0, atol=1e-3):
        return False, f"Weights must sum to 1.0 ±1e‑3 (got {total:.4f})"
    negatives = [f for f in REQUIRED_FACTORS if weights[f] < 0]
    if negatives:
        return False, f"Negative weights: {negatives}"
    return True, None


# ---------------------------------------------------------------------------
# INTERNAL HELPERS – METRICS
# ---------------------------------------------------------------------------

def _calc_metrics(df: pd.DataFrame) -> RegimeMetrics:
    m = RegimeMetrics()
    if {"ret_1m", "ret_3m"}.issubset(df.columns):
        m.momentum_strength = np.mean([
            _pct(df["ret_1m"] > 5),
            _pct(df["ret_3m"] > 10),
            max(0.0, _corr(df["ret_1m"], df["ret_3m"]))
        ])
    if {"pe", "pb"}.issubset(df.columns):
        valid_pe = df["pe"].between(0, 100)
        valid_pb = df["pb"].between(0, 10)
        m.value_opportunity = np.mean([
            _pct(df["pe"][valid_pe] < df["pe"][valid_pe].quantile(0.3)),
            _pct(df["pb"][valid_pb] < df["pb"][valid_pb].quantile(0.3)),
        ])
    if "eps_change_pct" in df.columns:
        rev = _pct(df.get("revenue_growth", pd.Series(dtype=float)) > 10)
        m.growth_prevalence = np.mean([
            _pct(df["eps_change_pct"] > 15),
            rev
        ])
    if "ret_1d" in df.columns and len(df) > 1:
        daily_vol = df["ret_1d"].std()
        extreme = _pct(df["ret_1d"].abs() > 3)
        m.volatility_level = min(1.0, (daily_vol / 2 + extreme) / 2)
    if "ret_1w" in df.columns:
        m.market_breadth = np.mean([
            _pct(df["ret_1w"] > 0),
            _pct(df["ret_1w"] > 3),
        ])
    if "vol_ratio_1d_90d" in df.columns:
        m.volume_surge = np.mean([
            _pct(df["vol_ratio_1d_90d"] > 2),
            _pct(df["vol_ratio_1d_90d"] > 3),
        ])
    if {"ret_1m", "rsi"}.issubset(df.columns):
        m.defensive_need = np.mean([
            _pct(df["ret_1m"] < -5),
            _pct(df["rsi"] < 30),
        ])
    if {"ret_3m", "ret_1w"}.issubset(df.columns):
        m.recovery_potential = np.sqrt(
            _pct(df["ret_3m"] < -10) * _pct(df["ret_1w"] > 2)
        )
    return m


# ---------------------------------------------------------------------------
# INTERNAL HELPERS – SCORING
# ---------------------------------------------------------------------------

def _score_momentum(m: RegimeMetrics) -> float:
    return m.momentum_strength * 0.4 + m.market_breadth * 0.3 + m.volume_surge * 0.2 + (1 - m.defensive_need) * 0.1

def _score_value(m: RegimeMetrics) -> float:
    return m.value_opportunity * 0.4 + m.defensive_need * 0.2 + (1 - m.momentum_strength) * 0.2 + (1 - m.volatility_level) * 0.2

def _score_growth(m: RegimeMetrics) -> float:
    return m.growth_prevalence * 0.5 + m.momentum_strength * 0.2 + m.market_breadth * 0.2 + (1 - m.defensive_need) * 0.1

def _score_volatility(m: RegimeMetrics) -> float:
    return m.volatility_level * 0.4 + m.volume_surge * 0.4 + (1 - m.market_breadth + 1) / 2 * 0.2

def _score_defensive(m: RegimeMetrics) -> float:
    return m.defensive_need * 0.4 + m.value_opportunity * 0.3 + (1 - m.volatility_level) * 0.2 + (1 - m.momentum_strength) * 0.1

def _score_recovery(m: RegimeMetrics) -> float:
    return m.recovery_potential * 0.4 + m.value_opportunity * 0.3 + m.momentum_strength * 0.2 + (1 - m.defensive_need) * 0.1


# ---------------------------------------------------------------------------
# INTERNAL HELPERS – UTILS
# ---------------------------------------------------------------------------

def _pct(mask: pd.Series | np.ndarray) -> float:
    """Return share of *True* in *mask* safely."""
    arr = np.asarray(mask, dtype=bool)
    return float(arr.mean()) if arr.size else 0.0


def _corr(a: pd.Series, b: pd.Series) -> float:
    if a is None or b is None:
        return 0.0
    joined = pd.concat([a, b], axis=1).dropna()
    return float(joined.corr().iloc[0, 1]) if len(joined) > 1 else 0.0


def _make_explanation(regime: str, m: RegimeMetrics, conf: float) -> str:
    base = {
        "momentum": f"Momentum strength {m.momentum_strength:.0%} with breadth {m.market_breadth:.0%}",
        "value": f"Value pocket {m.value_opportunity:.0%}; defensive need {m.defensive_need:.0%}",
        "growth": f"Growth prevalence {m.growth_prevalence:.0%}",
        "volatility": f"Volatility level {m.volatility_level:.0%} & volume surge {m.volume_surge:.0%}",
        "defensive": f"Market stress {m.defensive_need:.0%} → defensive stance",
        "recovery": f"Recovery potential {m.recovery_potential:.0%}",
        "balanced": "No dominant theme detected",
    }.get(regime, "Condition‑based selection")
    return f"{base}. Confidence {conf:.0%}."


def _make_recommendations(regime: str, m: RegimeMetrics) -> List[str]:
    rec: Dict[str, List[str]] = {
        "momentum": ["Favour trend‑following setups", "Tighten trailing stops"],
        "value": ["Screen low‑PE stocks", "Expect slower plays"],
        "growth": ["Prioritise EPS/revenue expansion", "Monitor valuation creep"],
        "volatility": ["Shorter holding periods", "Manage position size strictly"],
        "defensive": ["Tilt to quality & cashflows", "Limit leverage"],
        "recovery": ["Search oversold leaders", "Scale in gradually"],
        "balanced": ["Stay diversified", "No strong bias"]
    }
    return rec.get(regime, [])


# ---------------------------------------------------------------------------
# SELF‑TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke‑test with random data
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({
        "ret_1d": rng.normal(0, 2, 500),
        "ret_1w": rng.normal(0.5, 3, 500),
        "ret_1m": rng.normal(1, 5, 500),
        "ret_3m": rng.normal(3, 8, 500),
        "pe": rng.uniform(5, 40, 500),
        "pb": rng.uniform(0.5, 8, 500),
        "eps_change_pct": rng.normal(10, 20, 500),
        "revenue_growth": rng.normal(12, 15, 500),
        "vol_ratio_1d_90d": rng.uniform(0.5, 4, 500),
        "rsi": rng.uniform(20, 80, 500),
    })
    res = auto_detect_regime(demo)
    ok, err = validate_weights(res.weights)
    print("Selected:", res.regime, f"@ {res.confidence:.2f}")
    print("Valid weights:", ok, "|", err or "✓")
    print("Explanation:", res.explanation)
