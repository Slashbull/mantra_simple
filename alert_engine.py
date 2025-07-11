# alert_engine.py (SIMPLE VERSION)

import pandas as pd
from typing import Dict, Any

def buy_alert(df: pd.DataFrame) -> str:
    found = df[(df.get("tag") == "Buy") & (df.get("final_score", 0) >= 80)]
    if not found.empty:
        return f"{len(found)} high-score Buy(s): {', '.join(found['ticker'].head(5))}"
    return ""

def anomaly_alert(df: pd.DataFrame) -> str:
    found = df[df.get("anomaly", False) & (df.get("spike_score", 0) >= 3)]
    if not found.empty:
        return f"{len(found)} major anomaly spike(s): {', '.join(found['ticker'].head(5))}"
    return ""

def sector_hot_alert(df: pd.DataFrame) -> str:
    hots = df[df.get("sector_score", 0) > 90]["sector"].unique()
    if len(hots) > 0:
        return "Hot sector(s): " + ", ".join([str(x) for x in hots[:3]])
    return ""

def eps_watch_alert(df: pd.DataFrame) -> str:
    watch = df[(df.get("tag") == "Watch") & (df.get("eps_score", 0) >= 80)]
    if not watch.empty:
        return f"{len(watch)} Watch with strong EPS: {', '.join(watch['ticker'].head(5))}"
    return ""

# --- List of alert rules to check ---
ALERT_RULES = [
    ("High-Conviction Buy", buy_alert),
    ("Multi-Spike Anomaly", anomaly_alert),
    ("Sector Hot", sector_hot_alert),
    ("EPS Jumpers on Watch", eps_watch_alert),
]

def detect_alerts_simple(df: pd.DataFrame, last_alerts: Dict[str, str] = None) -> Dict[str, str]:
    """
    Run all alert rules, return {alert_key: message} for non-empty messages.
    Optionally dedupes with last_alerts (show only new/changed).
    """
    alerts = {}
    for name, func in ALERT_RULES:
        msg = func(df)
        if msg:
            alerts[name] = msg
    # Deduplicate
    if last_alerts:
        alerts = {k: v for k, v in alerts.items() if last_alerts.get(k) != v}
    return alerts

def format_alerts_simple(alerts: Dict[str, str]) -> str:
    if not alerts:
        return "✅ No new alerts."
    return "\n".join([f"{k}: {v}" for k, v in alerts.items()])

# Example usage:
# alerts = detect_alerts_simple(df, last_alerts)
# print(format_alerts_simple(alerts))
