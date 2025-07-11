# anomaly_detector.py (SIMPLE VERSION — M.A.N.T.R.A.)

import pandas as pd
import numpy as np

def run_anomaly_detector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight anomaly detector for core trading signals.
    Adds columns: anomaly, anomaly_type, anomaly_explanation.
    """
    df = df.copy()

    # --- Thresholds ---
    PRICE_SPIKE_1D = 5.0          # 1-day return > 5%
    VOLUME_SPIKE = 3.0            # 1d/90d volume ratio > 3x
    EPS_JUMP = 40.0               # EPS change > 40%
    BREAKOUT_PCT = 0.02           # Within 2% of 52w high

    # --- Ensure columns ---
    for col in ["ret_1d", "vol_ratio_1d_90d", "eps_change_pct", "price", "high_52w", "ticker"]:
        if col not in df.columns:
            df[col] = 0

    # --- Detect anomalies ---
    df["anomaly"] = False
    df["anomaly_type"] = ""
    df["anomaly_explanation"] = ""

    conditions = [
        (
            (df["ret_1d"] > PRICE_SPIKE_1D),
            "Price Spike",
            df["ret_1d"].map(lambda x: f"Price up {x:.1f}% today")
        ),
        (
            (df["vol_ratio_1d_90d"] > VOLUME_SPIKE),
            "Volume Spike",
            df["vol_ratio_1d_90d"].map(lambda x: f"Volume {x:.1f}x normal")
        ),
        (
            (df["eps_change_pct"] > EPS_JUMP),
            "EPS Jump",
            df["eps_change_pct"].map(lambda x: f"EPS jumped {x:.0f}%")
        ),
        (
            (df["high_52w"] > 0) &
            ((df["high_52w"] - df["price"]).abs() / df["high_52w"] < BREAKOUT_PCT),
            "52W Breakout",
            "Near 52W high"
        ),
    ]

    for cond, typ, expl in conditions:
        hit = cond & (~df["anomaly"])
        df.loc[hit, "anomaly"] = True
        df.loc[hit, "anomaly_type"] = typ
        df.loc[hit, "anomaly_explanation"] = expl if isinstance(expl, str) else expl[hit]

    # Optional: Multi-anomaly marker
    multi = df.groupby("ticker")["anomaly"].sum()
    tickers_with_multi = multi[multi > 1].index
    df.loc[df["ticker"].isin(tickers_with_multi), "anomaly_type"] = "Multi"
    df.loc[df["ticker"].isin(tickers_with_multi), "anomaly_explanation"] += " | Multiple alerts"

    return df
