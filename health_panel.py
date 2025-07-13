"""
health_panel.py - M.A.N.T.R.A. System Health Monitor
===================================================
Modified version that works without psutil for faster deployment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import streamlit as st

# Try to import psutil, but don't fail if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.info("psutil not available - system resource monitoring disabled")

# Import from constants
from constants import DATA_QUALITY, REQUIRED_COLUMNS, PROCESSING_LIMITS

logger = logging.getLogger(__name__)

# ============================================================================
# HEALTH STATUS TYPES
# ============================================================================

class HealthStatus(Enum):
    """Overall health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"

class ComponentStatus(Enum):
    """Individual component status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILURE = "failure"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Any
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Overall system health report"""
    overall_status: HealthStatus
    overall_score: float
    data_health: Dict[str, HealthMetric]
    performance_health: Dict[str, HealthMetric]
    component_health: Dict[str, ComponentStatus]
    issues: List[Dict]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# HEALTH MONITOR
# ============================================================================

class HealthMonitor:
    """
    Comprehensive system health monitoring
    """
    
    def __init__(self):
        self.metrics_history = []
        self.performance_baseline = {}
        self.start_time = time.time()
        
    def check_system_health(
        self,
        stocks_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
        cache_stats: Optional[Dict] = None,
        processing_times: Optional[Dict] = None
    ) -> SystemHealth:
        """
        Perform comprehensive health check
        
        Args:
            stocks_df: Stock data
            sector_df: Sector data
            cache_stats: Cache statistics
            processing_times: Processing time metrics
            
        Returns:
            SystemHealth report
        """
        logger.info("Running system health check...")
        
        # Initialize health metrics
        data_health = {}
        performance_health = {}
        component_health = {}
        issues = []
        recommendations = []
        
        # Check data health
        if stocks_df is not None:
            data_health.update(self._check_data_health(stocks_df, "stocks"))
        if sector_df is not None:
            data_health.update(self._check_data_health(sector_df, "sectors"))
        
        # Check performance health
        performance_health.update(self._check_performance_health(processing_times))
        
        # Check component health
        component_health.update(self._check_component_health(cache_stats))
        
        # Check system resources (only if psutil available)
        if PSUTIL_AVAILABLE:
            resource_health = self._check_system_resources()
            performance_health.update(resource_health)
        
        # Aggregate issues
        for metric in data_health.values():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                issues.append({
                    'category': 'Data Quality',
                    'metric': metric.name,
                    'status': metric.status.value,
                    'message': metric.message
                })
        
        for metric in performance_health.values():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                issues.append({
                    'category': 'Performance',
                    'metric': metric.name,
                    'status': metric.status.value,
                    'message': metric.message
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            data_health, performance_health, component_health, issues
        )
        
        # Calculate overall status
        overall_status, overall_score = self._calculate_overall_health(
            data_health, performance_health, component_health
        )
        
        # Create health report
        health_report = SystemHealth(
            overall_status=overall_status,
            overall_score=overall_score,
            data_health=data_health,
            performance_health=performance_health,
            component_health=component_health,
            issues=issues,
            recommendations=recommendations
        )
        
        # Store in history
        self._update_history(health_report)
        
        logger.info(f"Health check complete: {overall_status.value} ({overall_score:.1f}%)")
        
        return health_report
    
    # ========================================================================
    # DATA HEALTH CHECKS
    # ========================================================================
    
    def _check_data_health(self, df: pd.DataFrame, data_type: str) -> Dict[str, HealthMetric]:
        """Check health of data"""
        health_metrics = {}
        
        # 1. Data freshness
        freshness_metric = self._check_data_freshness(df, data_type)
        health_metrics[f"{data_type}_freshness"] = freshness_metric
        
        # 2. Data completeness
        completeness_metric = self._check_data_completeness(df, data_type)
        health_metrics[f"{data_type}_completeness"] = completeness_metric
        
        # 3. Data quality
        quality_metric = self._check_data_quality(df, data_type)
        health_metrics[f"{data_type}_quality"] = quality_metric
        
        # 4. Schema compliance
        schema_metric = self._check_schema_compliance(df, data_type)
        health_metrics[f"{data_type}_schema"] = schema_metric
        
        # 5. Outliers and anomalies
        anomaly_metric = self._check_data_anomalies(df, data_type)
        health_metrics[f"{data_type}_anomalies"] = anomaly_metric
        
        return health_metrics
    
    def _check_data_freshness(self, df: pd.DataFrame, data_type: str) -> HealthMetric:
        """Check if data is recent"""
        # For this example, we'll check if we have recent data
        # In production, you'd check actual timestamps
        
        if len(df) == 0:
            return HealthMetric(
                name=f"{data_type}_freshness",
                value=0,
                status=HealthStatus.ERROR,
                message=f"No {data_type} data available"
            )
        
        # Check if data seems current (simplified check)
        if 'ret_1d' in df.columns:
            non_zero_returns = (df['ret_1d'] != 0).sum()
            freshness_pct = non_zero_returns / len(df) * 100
            
            if freshness_pct > 80:
                status = HealthStatus.EXCELLENT
                message = f"{data_type} data is fresh ({freshness_pct:.1f}% active)"
            elif freshness_pct > 60:
                status = HealthStatus.GOOD
                message = f"{data_type} data is reasonably fresh"
            else:
                status = HealthStatus.WARNING
                message = f"Stale {data_type} data detected"
        else:
            freshness_pct = 100
            status = HealthStatus.GOOD
            message = f"{data_type} data loaded"
        
        return HealthMetric(
            name=f"{data_type}_freshness",
            value=freshness_pct,
            status=status,
            message=message
        )
    
    def _check_data_completeness(self, df: pd.DataFrame, data_type: str) -> HealthMetric:
        """Check data completeness"""
        if len(df) == 0:
            return HealthMetric(
                name=f"{data_type}_completeness",
                value=0,
                status=HealthStatus.ERROR,
                message=f"No {data_type} data"
            )
        
        # Calculate null percentage
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        completeness_pct = (1 - null_cells / total_cells) * 100
        
        # Determine status
        if completeness_pct >= 95:
            status = HealthStatus.EXCELLENT
            message = f"Excellent {data_type} completeness"
        elif completeness_pct >= 80:
            status = HealthStatus.GOOD
            message = f"Good {data_type} completeness"
        elif completeness_pct >= 70:
            status = HealthStatus.WARNING
            message = f"Some missing {data_type} data"
        else:
            status = HealthStatus.CRITICAL
            message = f"Significant missing {data_type} data"
        
        # Find columns with most nulls
        null_by_column = df.isna().sum()
        worst_columns = null_by_column.nlargest(5).to_dict()
        
        return HealthMetric(
            name=f"{data_type}_completeness",
            value=completeness_pct,
            status=status,
            message=message,
            details={'worst_columns': worst_columns}
        )
    
    def _check_data_quality(self, df: pd.DataFrame, data_type: str) -> HealthMetric:
        """Check overall data quality"""
        quality_issues = []
        
        # Check for duplicates
        if 'ticker' in df.columns:
            duplicates = df['ticker'].duplicated().sum()
            if duplicates > 0:
                quality_issues.append(f"{duplicates} duplicate tickers")
        
        # Check for invalid values
        if 'price' in df.columns:
            invalid_prices = (df['price'] <= 0).sum()
            if invalid_prices > 0:
                quality_issues.append(f"{invalid_prices} invalid prices")
        
        if 'pe' in df.columns:
            extreme_pe = ((df['pe'] < -100) | (df['pe'] > 1000)).sum()
            if extreme_pe > 0:
                quality_issues.append(f"{extreme_pe} extreme PE values")
        
        # Check data types
        expected_numeric = ['price', 'volume_1d', 'pe', 'eps_current']
        for col in expected_numeric:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                quality_issues.append(f"{col} is not numeric")
        
        # Calculate quality score
        quality_score = 100 - len(quality_issues) * 10
        quality_score = max(0, quality_score)
        
        # Determine status
        if quality_score >= 90:
            status = HealthStatus.EXCELLENT
            message = f"Excellent {data_type} quality"
        elif quality_score >= 70:
            status = HealthStatus.GOOD
            message = f"Good {data_type} quality"
        elif quality_score >= 50:
            status = HealthStatus.WARNING
            message = f"{data_type} quality issues detected"
        else:
            status = HealthStatus.CRITICAL
            message = f"Serious {data_type} quality problems"
        
        return HealthMetric(
            name=f"{data_type}_quality",
            value=quality_score,
            status=status,
            message=message,
            details={'issues': quality_issues}
        )
    
    def _check_schema_compliance(self, df: pd.DataFrame, data_type: str) -> HealthMetric:
        """Check if schema matches requirements"""
        if data_type == "stocks" and REQUIRED_COLUMNS.get('stocks'):
            required = set(REQUIRED_COLUMNS['stocks'])
        elif data_type == "sectors" and REQUIRED_COLUMNS.get('sector'):
            required = set(REQUIRED_COLUMNS['sector'])
        else:
            required = set()
        
        if not required:
            return HealthMetric(
                name=f"{data_type}_schema",
                value=100,
                status=HealthStatus.GOOD,
                message="Schema check skipped"
            )
        
        actual = set(df.columns)
        missing = required - actual
        compliance_pct = (1 - len(missing) / len(required)) * 100
        
        if compliance_pct == 100:
            status = HealthStatus.EXCELLENT
            message = "Perfect schema compliance"
        elif compliance_pct >= 80:
            status = HealthStatus.GOOD
            message = "Good schema compliance"
        elif compliance_pct >= 60:
            status = HealthStatus.WARNING
            message = "Schema compliance issues"
        else:
            status = HealthStatus.CRITICAL
            message = "Major schema problems"
        
        return HealthMetric(
            name=f"{data_type}_schema",
            value=compliance_pct,
            status=status,
            message=message,
            details={'missing_columns': list(missing)}
        )
    
    def _check_data_anomalies(self, df: pd.DataFrame, data_type: str) -> HealthMetric:
        """Check for data anomalies"""
        anomalies = []
        
        # Check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:  # Check first 10 numeric columns
            if col in df.columns and len(df[col].dropna()) > 10:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        anomalies.append(f"{col}: {outliers} extreme outliers")
        
        # Check for data consistency
        if 'high_52w' in df.columns and 'low_52w' in df.columns:
            invalid_range = (df['high_52w'] < df['low_52w']).sum()
            if invalid_range > 0:
                anomalies.append(f"{invalid_range} invalid 52w ranges")
        
        # Score based on anomalies
        anomaly_score = 100 - len(anomalies) * 15
        anomaly_score = max(0, anomaly_score)
        
        if anomaly_score >= 85:
            status = HealthStatus.GOOD
            message = "Minimal anomalies detected"
        elif anomaly_score >= 60:
            status = HealthStatus.WARNING
            message = "Some anomalies detected"
        else:
            status = HealthStatus.CRITICAL
            message = "Significant anomalies found"
        
        return HealthMetric(
            name=f"{data_type}_anomalies",
            value=anomaly_score,
            status=status,
            message=message,
            details={'anomalies': anomalies}
        )
    
    # ========================================================================
    # PERFORMANCE HEALTH CHECKS
    # ========================================================================
    
    def _check_performance_health(self, processing_times: Optional[Dict]) -> Dict[str, HealthMetric]:
        """Check system performance health"""
        health_metrics = {}
        
        if not processing_times:
            processing_times = {}
        
        # Check data loading time
        load_time = processing_times.get('data_load', 0)
        if load_time > 0:
            if load_time < 3:
                status = HealthStatus.EXCELLENT
                message = "Fast data loading"
            elif load_time < 5:
                status = HealthStatus.GOOD
                message = "Acceptable load time"
            elif load_time < 10:
                status = HealthStatus.WARNING
                message = "Slow data loading"
            else:
                status = HealthStatus.CRITICAL
                message = "Very slow data loading"
            
            health_metrics['load_time'] = HealthMetric(
                name='load_time',
                value=load_time,
                status=status,
                message=f"{message} ({load_time:.1f}s)"
            )
        
        # Check processing time
        process_time = processing_times.get('total_processing', 0)
        if process_time > 0:
            if process_time < 5:
                status = HealthStatus.EXCELLENT
                message = "Fast processing"
            elif process_time < 10:
                status = HealthStatus.GOOD
                message = "Normal processing"
            elif process_time < 20:
                status = HealthStatus.WARNING
                message = "Slow processing"
            else:
                status = HealthStatus.CRITICAL
                message = "Very slow processing"
            
            health_metrics['process_time'] = HealthMetric(
                name='process_time',
                value=process_time,
                status=status,
                message=f"{message} ({process_time:.1f}s)"
            )
        
        return health_metrics
    
    def _check_system_resources(self) -> Dict[str, HealthMetric]:
        """Check system resource usage (only if psutil available)"""
        health_metrics = {}
        
        if not PSUTIL_AVAILABLE:
            # Return empty metrics if psutil not available
            return health_metrics
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 50:
                status = HealthStatus.EXCELLENT
                message = "Low CPU usage"
            elif cpu_percent < 70:
                status = HealthStatus.GOOD
                message = "Moderate CPU usage"
            elif cpu_percent < 90:
                status = HealthStatus.WARNING
                message = "High CPU usage"
            else:
                status = HealthStatus.CRITICAL
                message = "Critical CPU usage"
            
            health_metrics['cpu_usage'] = HealthMetric(
                name='cpu_usage',
                value=cpu_percent,
                status=status,
                message=f"{message} ({cpu_percent:.1f}%)"
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 60:
                status = HealthStatus.EXCELLENT
                message = "Good memory availability"
            elif memory_percent < 75:
                status = HealthStatus.GOOD
                message = "Adequate memory"
            elif memory_percent < 85:
                status = HealthStatus.WARNING
                message = "High memory usage"
            else:
                status = HealthStatus.CRITICAL
                message = "Critical memory usage"
            
            health_metrics['memory_usage'] = HealthMetric(
                name='memory_usage',
                value=memory_percent,
                status=status,
                message=f"{message} ({memory_percent:.1f}%)",
                details={
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
        
        return health_metrics
    
    # ========================================================================
    # COMPONENT HEALTH
    # ========================================================================
    
    def _check_component_health(self, cache_stats: Optional[Dict]) -> Dict[str, ComponentStatus]:
        """Check health of system components"""
        components = {}
        
        # Data loader
        components['data_loader'] = ComponentStatus.OPERATIONAL
        
        # Cache system
        if cache_stats:
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate > 50:
                components['cache'] = ComponentStatus.OPERATIONAL
            elif hit_rate > 20:
                components['cache'] = ComponentStatus.DEGRADED
            else:
                components['cache'] = ComponentStatus.FAILURE
        else:
            components['cache'] = ComponentStatus.UNKNOWN
        
        # Signal engine
        components['signal_engine'] = ComponentStatus.OPERATIONAL
        
        # Alert system
        components['alert_system'] = ComponentStatus.OPERATIONAL
        
        return components
    
    # ========================================================================
    # OVERALL HEALTH CALCULATION
    # ========================================================================
    
    def _calculate_overall_health(
        self,
        data_health: Dict[str, HealthMetric],
        performance_health: Dict[str, HealthMetric],
        component_health: Dict[str, ComponentStatus]
    ) -> Tuple[HealthStatus, float]:
        """Calculate overall system health"""
        
        # Weight different aspects
        weights = {
            'data': 0.5,
            'performance': 0.3,
            'components': 0.2
        }
        
        # Score data health
        data_scores = []
        for metric in data_health.values():
            if metric.status == HealthStatus.EXCELLENT:
                data_scores.append(100)
            elif metric.status == HealthStatus.GOOD:
                data_scores.append(80)
            elif metric.status == HealthStatus.WARNING:
                data_scores.append(60)
            elif metric.status == HealthStatus.CRITICAL:
                data_scores.append(30)
            else:
                data_scores.append(0)
        
        data_score = np.mean(data_scores) if data_scores else 50
        
        # Score performance health
        perf_scores = []
        for metric in performance_health.values():
            if metric.status == HealthStatus.EXCELLENT:
                perf_scores.append(100)
            elif metric.status == HealthStatus.GOOD:
                perf_scores.append(80)
            elif metric.status == HealthStatus.WARNING:
                perf_scores.append(60)
            elif metric.status == HealthStatus.CRITICAL:
                perf_scores.append(30)
            else:
                perf_scores.append(0)
        
        perf_score = np.mean(perf_scores) if perf_scores else 50
        
        # Score component health
        comp_scores = []
        for status in component_health.values():
            if status == ComponentStatus.OPERATIONAL:
                comp_scores.append(100)
            elif status == ComponentStatus.DEGRADED:
                comp_scores.append(60)
            elif status == ComponentStatus.FAILURE:
                comp_scores.append(0)
            else:
                comp_scores.append(50)
        
        comp_score = np.mean(comp_scores) if comp_scores else 50
        
        # Calculate overall score
        overall_score = (
            data_score * weights['data'] +
            perf_score * weights['performance'] +
            comp_score * weights['components']
        )
        
        # Determine status
        if overall_score >= 90:
            status = HealthStatus.EXCELLENT
        elif overall_score >= 75:
            status = HealthStatus.GOOD
        elif overall_score >= 60:
            status = HealthStatus.WARNING
        elif overall_score >= 40:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.ERROR
        
        return status, overall_score
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    def _generate_recommendations(
        self,
        data_health: Dict[str, HealthMetric],
        performance_health: Dict[str, HealthMetric],
        component_health: Dict[str, ComponentStatus],
        issues: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        for metric in data_health.values():
            if metric.status == HealthStatus.CRITICAL:
                if 'completeness' in metric.name:
                    recommendations.append("Check data source connectivity")
                elif 'quality' in metric.name:
                    recommendations.append("Review data cleaning procedures")
                elif 'schema' in metric.name:
                    recommendations.append("Update data schema mappings")
        
        # Performance recommendations
        for metric in performance_health.values():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if 'load_time' in metric.name:
                    recommendations.append("Consider caching more aggressively")
                elif 'cpu' in metric.name:
                    recommendations.append("Optimize calculation algorithms")
                elif 'memory' in metric.name:
                    recommendations.append("Reduce data retention period")
        
        # Component recommendations
        for comp, status in component_health.items():
            if status == ComponentStatus.DEGRADED:
                recommendations.append(f"Monitor {comp} performance")
            elif status == ComponentStatus.FAILURE:
                recommendations.append(f"Restart {comp} component")
        
        # General recommendations based on issues
        if len(issues) > 5:
            recommendations.append("Run full system diagnostic")
        
        # If psutil not available
        if not PSUTIL_AVAILABLE:
            recommendations.append("Install psutil for system monitoring")
        
        return list(set(recommendations))[:5]  # Top 5 unique recommendations
    
    def _update_history(self, health_report: SystemHealth):
        """Update health metrics history"""
        self.metrics_history.append({
            'timestamp': health_report.timestamp,
            'overall_score': health_report.overall_score,
            'status': health_report.overall_status.value
        })
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def render_health_panel(st_container, health_report: SystemHealth):
    """Render health panel in Streamlit sidebar"""
    with st_container:
        st.subheader("🏥 System Health")
        
        # Overall status
        status_color = {
            HealthStatus.EXCELLENT: "🟢",
            HealthStatus.GOOD: "🟢",
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.ERROR: "🔴"
        }
        
        status_icon = status_color.get(health_report.overall_status, "⚪")
        st.metric(
            "Overall Health",
            f"{status_icon} {health_report.overall_status.value.title()}",
            f"{health_report.overall_score:.1f}%"
        )
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        # Data health summary
        data_scores = [m.value for m in health_report.data_health.values()]
        avg_data_health = np.mean(data_scores) if data_scores else 0
        
        with col1:
            st.metric("Data Quality", f"{avg_data_health:.0f}%")
        
        # Performance summary
        perf_scores = [m.value for m in health_report.performance_health.values() 
                      if isinstance(m.value, (int, float))]
        avg_perf = np.mean(perf_scores) if perf_scores else 0
        
        with col2:
            st.metric("Performance", f"{avg_perf:.0f}%")
        
        # Issues
        if health_report.issues:
            st.warning(f"⚠️ {len(health_report.issues)} issues detected")
            
            with st.expander("View Issues"):
                for issue in health_report.issues[:5]:
                    st.text(f"• {issue['message']}")
        
        # Recommendations
        if health_report.recommendations:
            with st.expander("💡 Recommendations"):
                for rec in health_report.recommendations:
                    st.text(f"• {rec}")
        
        # Detailed metrics
        with st.expander("📊 Detailed Metrics"):
            # Data health
            st.text("Data Health:")
            for name, metric in health_report.data_health.items():
                st.text(f"  {name}: {metric.value:.1f}% - {metric.message}")
            
            # Performance
            st.text("\nPerformance:")
            for name, metric in health_report.performance_health.items():
                if isinstance(metric.value, (int, float)):
                    st.text(f"  {name}: {metric.value:.1f} - {metric.message}")
            
            # Components
            st.text("\nComponents:")
            for comp, status in health_report.component_health.items():
                st.text(f"  {comp}: {status.value}")
        
        # System monitoring note
        if not PSUTIL_AVAILABLE:
            st.caption("💡 System monitoring limited - psutil not installed")
        
        # Last updated
        st.caption(f"Last updated: {health_report.timestamp.strftime('%H:%M:%S')}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_system_health(
    stocks_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    cache_stats: Optional[Dict] = None,
    processing_times: Optional[Dict] = None
) -> SystemHealth:
    """
    Run system health check
    
    Returns:
        SystemHealth report
    """
    monitor = HealthMonitor()
    return monitor.check_system_health(
        stocks_df, sector_df, cache_stats, processing_times
    )

def get_health_summary(health_report: SystemHealth) -> Dict:
    """Get summary of health report"""
    return {
        'status': health_report.overall_status.value,
        'score': health_report.overall_score,
        'issues_count': len(health_report.issues),
        'critical_issues': len([i for i in health_report.issues if i['status'] == 'critical']),
        'recommendations_count': len(health_report.recommendations)
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("M.A.N.T.R.A. Health Panel")
    print("="*60)
    print("\nSystem health monitoring and diagnostics")
    print("\nHealth Checks:")
    print("  - Data freshness and completeness")
    print("  - Data quality and anomalies")
    print("  - System performance")
    print("  - Component status")
    if PSUTIL_AVAILABLE:
        print("  - Resource usage (CPU/Memory)")
    else:
        print("  - Resource usage (NOT AVAILABLE - psutil not installed)")
    print("\nUse check_system_health() to run diagnostics")
    print("="*60)
