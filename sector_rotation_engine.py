"""
sector_rotation_engine.py - Elite Sector Rotation Analysis Engine for M.A.N.T.R.A.

Production-grade sector rotation intelligence with statistical robustness and edge detection.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum


class RotationStatus(Enum):
   """Sector rotation status classifications"""
   EXPLOSIVE = "Explosive"  # Top 10% - exceptional outperformance
   HOT = "Hot"             # Top 25% - strong momentum
   WARMING = "Warming"     # 25-50% - gaining strength
   NEUTRAL = "Neutral"     # 50-75% - market performers
   COOLING = "Cooling"     # 75-90% - losing momentum
   COLD = "Cold"           # Bottom 10% - severe underperformance


@dataclass
class SectorAnalytics:
   """Advanced sector analytics container"""
   momentum_velocity: float  # Rate of change in performance
   consistency_score: float  # Performance stability
   relative_strength: float  # vs market average
   concentration_risk: float  # Stock concentration in sector
   edge_score: float        # Composite edge indicator


def compute_sector_rotation(
   sector_df: pd.DataFrame,
   metric: str = "sector_avg_3m",
   enable_advanced: bool = True
) -> pd.DataFrame:
   """
   Compute sector rotation analysis with intelligent scoring and classification.
   
   Args:
       sector_df: Input DataFrame with sector data
       metric: Primary metric for ranking (default: 3-month average return)
       enable_advanced: Enable advanced analytics and edge detection
       
   Returns:
       DataFrame with comprehensive sector rotation analysis
       
   Raises:
       ValueError: If required columns are missing or data is invalid
   """
   # Validate and normalize input
   df = _normalize_sector_data(sector_df)
   
   # Validate metric exists
   if metric not in df.columns:
       available_metrics = _find_available_metrics(df)
       if available_metrics:
           metric = available_metrics[0]
           print(f"Warning: '{metric}' not found. Using '{metric}' instead.")
       else:
           raise ValueError(f"No valid metric columns found in data")
   
   # Core rotation analysis
   df = _calculate_rotation_scores(df, metric)
   df = _assign_rotation_status(df)
   
   # Advanced analytics if enabled
   if enable_advanced and len(df) > 0:
       df = _calculate_advanced_metrics(df, metric)
       df = _detect_rotation_edges(df)
   
   # Format output
   output_df = _format_output(df, enable_advanced)
   
   return output_df


def sector_rotation_summary(
   df: pd.DataFrame,
   include_analytics: bool = True
) -> Dict[str, Any]:
   """
   Generate comprehensive summary of sector rotation state.
   
   Args:
       df: Sector rotation DataFrame from compute_sector_rotation()
       include_analytics: Include advanced analytics in summary
       
   Returns:
       Dictionary with rotation insights and statistics
   """
   summary = {
       "timestamp": pd.Timestamp.now().isoformat(),
       "total_sectors": len(df),
       "rotation_distribution": {},
       "top_performers": [],
       "bottom_performers": [],
       "market_state": "balanced"
   }
   
   # Rotation distribution
   if "rotation_status" in df.columns:
       status_counts = df["rotation_status"].value_counts()
       summary["rotation_distribution"] = status_counts.to_dict()
       
       # Market state assessment
       hot_pct = status_counts.get("Hot", 0) / len(df) if len(df) > 0 else 0
       cold_pct = status_counts.get("Cold", 0) / len(df) if len(df) > 0 else 0
       
       if hot_pct > 0.4:
           summary["market_state"] = "risk-on"
       elif cold_pct > 0.4:
           summary["market_state"] = "risk-off"
       elif "Explosive" in status_counts and status_counts["Explosive"] > 0:
           summary["market_state"] = "momentum"
   
   # Top/Bottom performers
   if "sector" in df.columns and "sector_score" in df.columns:
       top_n = min(3, len(df))
       summary["top_performers"] = df.nlargest(top_n, "sector_score")[
           ["sector", "sector_score", "rotation_status"]
       ].to_dict("records")
       
       summary["bottom_performers"] = df.nsmallest(top_n, "sector_score")[
           ["sector", "sector_score", "rotation_status"]
       ].to_dict("records")
   
   # Advanced analytics
   if include_analytics and "edge_score" in df.columns:
       summary["analytics"] = {
           "avg_edge_score": float(df["edge_score"].mean()),
           "sectors_with_edge": int((df["edge_score"] > 70).sum()),
           "momentum_leaders": df[df["momentum_velocity"] > 0]["sector"].tolist(),
           "value_opportunities": df[
               (df["sector_score"] < 30) & (df.get("consistency_score", 50) > 60)
           ]["sector"].tolist()
       }
   
   # Actionable insights
   summary["insights"] = _generate_insights(df, summary)
   
   return summary


def get_rotation_recommendations(
   df: pd.DataFrame,
   risk_tolerance: str = "moderate"
) -> List[Dict[str, Any]]:
   """
   Generate actionable sector rotation recommendations.
   
   Args:
       df: Sector rotation DataFrame
       risk_tolerance: "conservative", "moderate", or "aggressive"
       
   Returns:
       List of recommendation dictionaries
   """
   recommendations = []
   
   # Risk tolerance mappings
   risk_filters = {
       "conservative": lambda x: x["rotation_status"].isin(["Hot", "Warming"]) & (x.get("consistency_score", 50) > 60),
       "moderate": lambda x: x["rotation_status"].isin(["Explosive", "Hot", "Warming"]),
       "aggressive": lambda x: (x["rotation_status"] == "Explosive") | (x.get("edge_score", 0) > 80)
   }
   
   # Apply risk filter
   filter_func = risk_filters.get(risk_tolerance, risk_filters["moderate"])
   candidates = df[filter_func(df)].copy()
   
   # Generate recommendations
   for _, sector in candidates.iterrows():
       rec = {
           "sector": sector["sector"],
           "action": _determine_action(sector),
           "confidence": _calculate_confidence(sector),
           "rationale": _generate_rationale(sector),
           "risk_level": _assess_risk(sector)
       }
       recommendations.append(rec)
   
   # Sort by confidence
   recommendations.sort(key=lambda x: x["confidence"], reverse=True)
   
   return recommendations[:5]  # Top 5 recommendations


# Internal helper functions

def _normalize_sector_data(df: pd.DataFrame) -> pd.DataFrame:
   """Normalize and clean sector data with bulletproof handling."""
   df = df.copy()
   
   # Drop completely empty rows/columns
   df = df.dropna(how="all").dropna(axis=1, how="all")
   
   if df.empty:
       raise ValueError("Input DataFrame is empty after removing null rows/columns")
   
   # Standardize column names
   df.columns = (
       df.columns
       .astype(str)
       .str.strip()
       .str.lower()
       .str.replace(r'[^\w\s]', '', regex=True)
       .str.replace(r'\s+', '_', regex=True)
   )
   
   # Ensure sector column exists
   sector_col = None
   for col in ["sector", "sector_name", "industry", "segment"]:
       if col in df.columns:
           sector_col = col
           break
   
   if sector_col is None:
       raise ValueError("No sector/industry column found in data")
   
   # Standardize sector column
   if sector_col != "sector":
       df = df.rename(columns={sector_col: "sector"})
   
   df["sector"] = (
       df["sector"]
       .astype(str)
       .str.strip()
       .str.title()
       .replace(["Nan", "None", "", "Unknown"], pd.NA)
   )
   
   # Remove invalid sectors
   df = df.dropna(subset=["sector"])
   
   # Convert numeric columns
   numeric_patterns = ["avg", "return", "ret", "score", "vol", "pe", "pb", "count"]
   for col in df.columns:
       if any(pattern in col for pattern in numeric_patterns):
           df[col] = pd.to_numeric(df[col], errors="coerce")
   
   # Handle sector_count specially
   if "sector_count" in df.columns:
       df["sector_count"] = df["sector_count"].fillna(0).astype(int)
   
   return df


def _find_available_metrics(df: pd.DataFrame) -> List[str]:
   """Find available metric columns for ranking."""
   metric_patterns = ["avg", "return", "ret", "performance", "gain", "score"]
   exclude_patterns = ["count", "rank", "percentile"]
   
   candidates = []
   for col in df.columns:
       if any(pattern in col for pattern in metric_patterns):
           if not any(exclude in col for exclude in exclude_patterns):
               if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                   candidates.append(col)
   
   # Prioritize by time period (longer first)
   priority_order = ["1y", "6m", "3m", "1m", "1w", "1d"]
   
   sorted_candidates = []
   for period in priority_order:
       for candidate in candidates:
           if period in candidate and candidate not in sorted_candidates:
               sorted_candidates.append(candidate)
   
   # Add remaining candidates
   for candidate in candidates:
       if candidate not in sorted_candidates:
           sorted_candidates.append(candidate)
   
   return sorted_candidates


def _calculate_rotation_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
   """Calculate core rotation scores using percentile ranking."""
   # Handle missing values
   df[f"{metric}_clean"] = df[metric].fillna(df[metric].median())
   
   # Percentile-based scoring (0-100)
   df["sector_score"] = df[f"{metric}_clean"].rank(pct=True, method="average") * 100
   
   # Absolute ranking (1 = best)
   df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)
   
   # Z-score for statistical context
   if df[metric].std() > 0:
       df["z_score"] = (df[f"{metric}_clean"] - df[metric].mean()) / df[metric].std()
   else:
       df["z_score"] = 0
   
   # Clean up
   df = df.drop(columns=[f"{metric}_clean"])
   
   return df


def _assign_rotation_status(df: pd.DataFrame) -> pd.DataFrame:
   """Assign rotation status with statistical thresholds."""
   # Define percentile thresholds
   thresholds = {
       RotationStatus.EXPLOSIVE: 90,
       RotationStatus.HOT: 75,
       RotationStatus.WARMING: 50,
       RotationStatus.NEUTRAL: 25,
       RotationStatus.COOLING: 10,
       RotationStatus.COLD: 0
   }
   
   def get_status(score):
       for status, threshold in thresholds.items():
           if score >= threshold:
               return status.value
       return RotationStatus.COLD.value
   
   df["rotation_status"] = df["sector_score"].apply(get_status)
   
   # Add status rank for sorting
   status_ranks = {status.value: i for i, status in enumerate(RotationStatus)}
   df["status_rank"] = df["rotation_status"].map(status_ranks)
   
   return df


def _calculate_advanced_metrics(df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
   """Calculate advanced rotation metrics."""
   # Find additional time period metrics
   metrics_1m = [col for col in df.columns if "1m" in col and "avg" in col]
   metrics_3m = [col for col in df.columns if "3m" in col and "avg" in col]
   
   # Momentum velocity (acceleration)
   if metrics_1m and metrics_3m:
       df["momentum_velocity"] = df[metrics_1m[0]] - (df[metrics_3m[0]] / 3)
   else:
       df["momentum_velocity"] = 0
   
   # Consistency score (inverse of volatility if available)
   vol_cols = [col for col in df.columns if "vol" in col or "std" in col]
   if vol_cols:
       # Lower volatility = higher consistency
       df["consistency_score"] = 100 - (df[vol_cols[0]].rank(pct=True) * 100)
   else:
       df["consistency_score"] = 50  # Neutral if no volatility data
   
   # Relative strength vs market
   market_avg = df[primary_metric].mean()
   df["relative_strength"] = ((df[primary_metric] - market_avg) / abs(market_avg) * 100).fillna(0)
   
   # Concentration risk (based on sector count)
   if "sector_count" in df.columns:
       total_stocks = df["sector_count"].sum()
       df["concentration_risk"] = (df["sector_count"] / total_stocks * 100) if total_stocks > 0 else 0
   else:
       df["concentration_risk"] = 100 / len(df)  # Equal weight assumption
   
   # Composite edge score
   df["edge_score"] = (
       df["sector_score"] * 0.4 +
       df["consistency_score"] * 0.2 +
       (df["relative_strength"] + 100) / 2 * 0.2 +  # Normalize to 0-100
       (100 - df["concentration_risk"]) * 0.2  # Lower concentration = better
   ).clip(0, 100)
   
   return df


def _detect_rotation_edges(df: pd.DataFrame) -> pd.DataFrame:
   """Detect actionable rotation edges and anomalies."""
   edges = []
   
   for idx, row in df.iterrows():
       sector_edges = []
       
       # Momentum explosion
       if row.get("momentum_velocity", 0) > df["momentum_velocity"].quantile(0.9):
           sector_edges.append("momentum_surge")
       
       # Value opportunity
       if row["sector_score"] < 30 and row.get("consistency_score", 50) > 70:
           sector_edges.append("oversold_quality")
       
       # Rotation leader
       if row["rotation_status"] == "Explosive" and row.get("relative_strength", 0) > 10:
           sector_edges.append("rotation_leader")
       
       # Trend reversal
       if row.get("z_score", 0) < -1.5 and row.get("momentum_velocity", 0) > 0:
           sector_edges.append("potential_reversal")
       
       edges.append(edges)
   
   df["rotation_edges"] = edges
   df["edge_count"] = df["rotation_edges"].apply(len)
   
   return df


def _format_output(df: pd.DataFrame, include_advanced: bool) -> pd.DataFrame:
   """Format output for dashboard consumption."""
   # Core columns always included
   core_columns = [
       "sector", "sector_score", "sector_rank", "rotation_status", "sector_count"
   ]
   
   # Advanced columns if enabled
   advanced_columns = [
       "momentum_velocity", "consistency_score", "relative_strength",
       "edge_score", "edge_count"
   ]
   
   # Select columns
   output_columns = core_columns
   if include_advanced:
       output_columns.extend([col for col in advanced_columns if col in df.columns])
   
   # Filter to existing columns
   output_columns = [col for col in output_columns if col in df.columns]
   
   # Sort by score (best first)
   output_df = df[output_columns].sort_values(
       "sector_score", ascending=False
   ).reset_index(drop=True)
   
   # Round numeric columns
   numeric_cols = output_df.select_dtypes(include=[np.number]).columns
   output_df[numeric_cols] = output_df[numeric_cols].round(2)
   
   return output_df


def _generate_insights(df: pd.DataFrame, summary: Dict[str, Any]) -> List[str]:
   """Generate actionable insights from rotation analysis."""
   insights = []
   
   # Market state insight
   state = summary.get("market_state", "balanced")
   if state == "risk-on":
       insights.append("Market showing risk-on behavior - consider growth sectors")
   elif state == "risk-off":
       insights.append("Market in risk-off mode - focus on defensive sectors")
   
   # Rotation opportunities
   if "rotation_edges" in df.columns:
       reversal_sectors = df[df["rotation_edges"].apply(
           lambda x: "potential_reversal" in x if isinstance(x, list) else False
       )]["sector"].tolist()
       if reversal_sectors:
           insights.append(f"Potential reversals in: {', '.join(reversal_sectors)}")
   
   # Concentration warning
   if "concentration_risk" in df.columns:
       high_concentration = df[df["concentration_risk"] > 30]["sector"].tolist()
       if high_concentration:
           insights.append(f"High concentration in: {', '.join(high_concentration[:2])}")
   
   return insights


def _determine_action(sector: pd.Series) -> str:
   """Determine recommended action for sector."""
   if sector["rotation_status"] == "Explosive":
       return "STRONG BUY"
   elif sector["rotation_status"] == "Hot":
       return "BUY"
   elif sector["rotation_status"] == "Warming":
       return "ACCUMULATE"
   elif sector["rotation_status"] in ["Cooling", "Cold"]:
       return "REDUCE"
   else:
       return "HOLD"


def _calculate_confidence(sector: pd.Series) -> float:
   """Calculate confidence score for recommendation."""
   base_confidence = sector["sector_score"] / 100 * 0.5
   
   if "consistency_score" in sector:
       base_confidence += sector["consistency_score"] / 100 * 0.3
   
   if "edge_score" in sector:
       base_confidence += sector["edge_score"] / 100 * 0.2
   
   return min(base_confidence * 100, 95)


def _generate_rationale(sector: pd.Series) -> str:
   """Generate explanation for recommendation."""
   rationales = []
   
   if sector["rotation_status"] in ["Explosive", "Hot"]:
       rationales.append(f"{sector['rotation_status']} momentum")
   
   if sector.get("relative_strength", 0) > 10:
       rationales.append(f"Outperforming market by {sector['relative_strength']:.1f}%")
   
   if sector.get("momentum_velocity", 0) > 0:
       rationales.append("Accelerating trend")
   
   if not rationales:
       rationales.append(f"Ranked #{sector['sector_rank']} with score {sector['sector_score']:.1f}")
   
   return "; ".join(rationales)


def _assess_risk(sector: pd.Series) -> str:
   """Assess risk level of sector."""
   if sector.get("consistency_score", 50) < 30:
       return "High"
   elif sector.get("concentration_risk", 0) > 40:
       return "Medium-High"
   elif sector["rotation_status"] in ["Cooling", "Cold"]:
       return "Medium"
   else:
       return "Low-Medium"
