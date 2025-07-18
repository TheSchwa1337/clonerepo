#!/usr/bin/env python3
"""
Schwabot Comprehensive Monitoring and Logging System
===================================================

Advanced monitoring and logging system for Schwabot trading engine:
- Real-time performance monitoring
- Mathematical state tracking
- System health monitoring
- Alert system
- Performance analytics
- Deep logging for all 46-day mathematical features
- Web dashboard for real-time monitoring
- Data visualization and reporting

This ensures complete visibility into the mathematical implementations and system performance.
"""

import asyncio
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import sqlite3
import threading
import queue
import websockets
import aiohttp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import hashlib
import pickle

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert levels for monitoring system."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Component types for monitoring."""
    TRADING_ENGINE = "trading_engine"
    MATHEMATICAL_CORE = "mathematical_core"
    API_INTEGRATION = "api_integration"
    STATE_PERSISTENCE = "state_persistence"
    PERFORMANCE_MONITOR = "performance_monitor"
    VAULT_LOGIC = "vault_logic"
    LANTERN_LOGIC = "lantern_logic"
    GHOST_ECHO = "ghost_echo"
    QUANTUM_INTEGRATION = "quantum_integration"

@dataclass
class SystemMetric:
    """System metric for monitoring."""
    timestamp: float
    component: str
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """Alert for monitoring system."""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any] = None
    resolved: bool = False

@dataclass
class MathematicalState:
    """Mathematical state for deep monitoring."""
    timestamp: float
    component: str
    asset: str
    state_hash: str
    zpe_value: float
    entropy_value: float
    vault_state: str
    lantern_trigger: bool
    ghost_echo_active: bool
    quantum_state: Optional[np.ndarray]
    vault_entries: int
    lantern_corps: int
    ferris_tiers: Dict[str, float]
    strategy_hashes: List[str]
    performance_metrics: Dict[str, float]

class MonitoringDatabase:
    """Database for storing monitoring data."""
    
    def __init__(self, db_path: str = "schwabot_monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Mathematical states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mathematical_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    component TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    zpe_value REAL NOT NULL,
                    entropy_value REAL NOT NULL,
                    vault_state TEXT NOT NULL,
                    lantern_trigger BOOLEAN NOT NULL,
                    ghost_echo_active BOOLEAN NOT NULL,
                    quantum_state BLOB,
                    vault_entries INTEGER NOT NULL,
                    lantern_corps INTEGER NOT NULL,
                    ferris_tiers TEXT NOT NULL,
                    strategy_hashes TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    successful_trades INTEGER NOT NULL,
                    total_profit REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_roi REAL NOT NULL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_consecutive_losses INTEGER,
                    max_consecutive_wins INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Component health table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS component_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time REAL,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Monitoring database initialized: {self.db_path}")
    
    def save_system_metric(self, metric: SystemMetric):
        """Save system metric to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics (timestamp, component, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp, metric.component, metric.metric_name,
                metric.value, metric.unit, json.dumps(metric.metadata) if metric.metadata else None
            ))
            conn.commit()
    
    def save_alert(self, alert: Alert):
        """Save alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, level, component, message, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert.timestamp, alert.level.value, alert.component,
                alert.message, json.dumps(alert.details) if alert.details else None
            ))
            conn.commit()
    
    def save_mathematical_state(self, state: MathematicalState):
        """Save mathematical state to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mathematical_states (
                    timestamp, component, asset, state_hash, zpe_value, entropy_value,
                    vault_state, lantern_trigger, ghost_echo_active, quantum_state,
                    vault_entries, lantern_corps, ferris_tiers, strategy_hashes,
                    performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.timestamp, state.component, state.asset, state.state_hash,
                state.zpe_value, state.entropy_value, state.vault_state,
                state.lantern_trigger, state.ghost_echo_active,
                pickle.dumps(state.quantum_state) if state.quantum_state is not None else None,
                state.vault_entries, state.lantern_corps,
                json.dumps(state.ferris_tiers), json.dumps(state.strategy_hashes),
                json.dumps(state.performance_metrics)
            ))
            conn.commit()
    
    def save_performance_history(self, metrics: Dict[str, Any]):
        """Save performance history to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_history (
                    timestamp, total_trades, successful_trades, total_profit, max_drawdown,
                    win_rate, avg_roi, sharpe_ratio, sortino_ratio, max_consecutive_losses,
                    max_consecutive_wins
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), metrics['total_trades'], metrics['successful_trades'],
                metrics['total_profit'], metrics['max_drawdown'], metrics['win_rate'],
                metrics['avg_roi'], metrics.get('sharpe_ratio', 0.0),
                metrics.get('sortino_ratio', 0.0), metrics['max_consecutive_losses'],
                metrics['max_consecutive_wins']
            ))
            conn.commit()
    
    def save_component_health(self, component: str, status: str, response_time: Optional[float] = None, 
                            error_count: int = 0, last_error: Optional[str] = None):
        """Save component health status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO component_health (timestamp, component, status, response_time, error_count, last_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (time.time(), component, status, response_time, error_count, last_error))
            conn.commit()
    
    def get_recent_metrics(self, component: str, metric_name: str, limit: int = 100) -> List[SystemMetric]:
        """Get recent metrics for a component."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM system_metrics 
                WHERE component = ? AND metric_name = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (component, metric_name, limit))
            
            rows = cursor.fetchall()
            metrics = []
            for row in rows:
                metric = SystemMetric(
                    timestamp=row[1],
                    component=row[2],
                    metric_name=row[3],
                    value=row[4],
                    unit=row[5],
                    metadata=json.loads(row[6]) if row[6] else None
                )
                metrics.append(metric)
            
            return metrics[::-1]  # Return in chronological order
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if level:
                cursor.execute("""
                    SELECT * FROM alerts 
                    WHERE resolved = FALSE AND level = ?
                    ORDER BY timestamp DESC
                """, (level.value,))
            else:
                cursor.execute("""
                    SELECT * FROM alerts 
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                """)
            
            rows = cursor.fetchall()
            alerts = []
            for row in rows:
                alert = Alert(
                    timestamp=row[1],
                    level=AlertLevel(row[2]),
                    component=row[3],
                    message=row[4],
                    details=json.loads(row[5]) if row[5] else None,
                    resolved=bool(row[6])
                )
                alerts.append(alert)
            
            return alerts
    
    def get_component_health(self, component: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get component health history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM component_health 
                WHERE component = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (component, limit))
            
            rows = cursor.fetchall()
            health_data = []
            for row in rows:
                health_data.append({
                    'timestamp': row[1],
                    'component': row[2],
                    'status': row[3],
                    'response_time': row[4],
                    'error_count': row[5],
                    'last_error': row[6]
                })
            
            return health_data

class PerformanceAnalyzer:
    """Advanced performance analysis system."""
    
    def __init__(self, monitoring_db: MonitoringDatabase):
        self.monitoring_db = monitoring_db
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def calculate_rolling_metrics(self, window_hours: int = 24) -> Dict[str, Any]:
        """Calculate rolling performance metrics."""
        cache_key = f"rolling_metrics_{window_hours}"
        if cache_key in self.analysis_cache:
            cache_time, cache_data = self.analysis_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cache_data
        
        # Get performance history for the window
        cutoff_time = time.time() - (window_hours * 3600)
        
        with sqlite3.connect(self.monitoring_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM performance_history 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff_time,))
            
            rows = cursor.fetchall()
        
        if not rows:
            return {}
        
        # Calculate rolling metrics
        metrics = []
        for row in rows:
            metrics.append({
                'timestamp': row[1],
                'total_trades': row[2],
                'successful_trades': row[3],
                'total_profit': row[4],
                'max_drawdown': row[5],
                'win_rate': row[6],
                'avg_roi': row[7],
                'sharpe_ratio': row[8],
                'sortino_ratio': row[9],
                'max_consecutive_losses': row[10],
                'max_consecutive_wins': row[11]
            })
        
        # Calculate rolling averages and trends
        df = pd.DataFrame(metrics)
        
        rolling_metrics = {
            'window_hours': window_hours,
            'data_points': len(metrics),
            'current_win_rate': metrics[-1]['win_rate'] if metrics else 0.0,
            'current_avg_roi': metrics[-1]['avg_roi'] if metrics else 0.0,
            'current_total_profit': metrics[-1]['total_profit'] if metrics else 0.0,
            'current_max_drawdown': metrics[-1]['max_drawdown'] if metrics else 0.0,
            'rolling_win_rate': df['win_rate'].mean() if not df.empty else 0.0,
            'rolling_avg_roi': df['avg_roi'].mean() if not df.empty else 0.0,
            'profit_trend': self._calculate_trend(df['total_profit']) if not df.empty else 0.0,
            'drawdown_trend': self._calculate_trend(df['max_drawdown']) if not df.empty else 0.0,
            'volatility': df['avg_roi'].std() if not df.empty else 0.0,
            'best_period': self._find_best_period(metrics),
            'worst_period': self._find_worst_period(metrics)
        }
        
        # Cache the results
        self.analysis_cache[cache_key] = (time.time(), rolling_metrics)
        
        return rolling_metrics
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend of a series (positive = improving, negative = declining)."""
        if len(series) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    def _find_best_period(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best performing period."""
        if not metrics:
            return {}
        
        best_metric = max(metrics, key=lambda x: x['avg_roi'])
        return {
            'timestamp': best_metric['timestamp'],
            'avg_roi': best_metric['avg_roi'],
            'win_rate': best_metric['win_rate'],
            'total_profit': best_metric['total_profit']
        }
    
    def _find_worst_period(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the worst performing period."""
        if not metrics:
            return {}
        
        worst_metric = min(metrics, key=lambda x: x['avg_roi'])
        return {
            'timestamp': worst_metric['timestamp'],
            'avg_roi': worst_metric['avg_roi'],
            'win_rate': worst_metric['win_rate'],
            'total_profit': worst_metric['total_profit']
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Get different time windows
        windows = [1, 6, 24, 168]  # 1 hour, 6 hours, 24 hours, 1 week
        
        report = {
            'timestamp': time.time(),
            'windows': {}
        }
        
        for window in windows:
            report['windows'][f'{window}h'] = self.calculate_rolling_metrics(window)
        
        # Add overall statistics
        report['overall'] = {
            'total_alerts': len(self.monitoring_db.get_active_alerts()),
            'critical_alerts': len(self.monitoring_db.get_active_alerts(AlertLevel.CRITICAL)),
            'system_health': self._assess_system_health()
        }
        
        return report
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        critical_alerts = len(self.monitoring_db.get_active_alerts(AlertLevel.CRITICAL))
        error_alerts = len(self.monitoring_db.get_active_alerts(AlertLevel.ERROR))
        
        if critical_alerts > 0:
            return "critical"
        elif error_alerts > 5:
            return "warning"
        else:
            return "healthy"

class AlertManager:
    """Advanced alert management system."""
    
    def __init__(self, monitoring_db: MonitoringDatabase):
        self.monitoring_db = monitoring_db
        self.alert_rules = self._initialize_alert_rules()
        self.alert_handlers = self._initialize_alert_handlers()
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules."""
        return {
            'performance_degradation': {
                'condition': lambda metrics: metrics.get('win_rate', 1.0) < 0.4,
                'level': AlertLevel.WARNING,
                'message': 'Performance degradation detected: Win rate below 40%'
            },
            'high_drawdown': {
                'condition': lambda metrics: metrics.get('max_drawdown', 0.0) > 0.2,
                'level': AlertLevel.ERROR,
                'message': 'High drawdown detected: Maximum drawdown exceeds 20%'
            },
            'low_roi': {
                'condition': lambda metrics: metrics.get('avg_roi', 0.0) < -0.05,
                'level': AlertLevel.WARNING,
                'message': 'Low ROI detected: Average ROI below -5%'
            },
            'component_failure': {
                'condition': lambda health: health.get('status') == 'failed',
                'level': AlertLevel.CRITICAL,
                'message': 'Component failure detected'
            },
            'high_error_rate': {
                'condition': lambda health: health.get('error_count', 0) > 10,
                'level': AlertLevel.ERROR,
                'message': 'High error rate detected'
            }
        }
    
    def _initialize_alert_handlers(self) -> Dict[AlertLevel, List[callable]]:
        """Initialize alert handlers."""
        return {
            AlertLevel.INFO: [self._log_alert],
            AlertLevel.WARNING: [self._log_alert, self._send_warning_notification],
            AlertLevel.ERROR: [self._log_alert, self._send_error_notification, self._trigger_recovery],
            AlertLevel.CRITICAL: [self._log_alert, self._send_critical_notification, self._trigger_emergency_stop]
        }
    
    def check_alerts(self, component: str, data: Dict[str, Any]):
        """Check for alerts based on component data."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](data):
                    alert = Alert(
                        timestamp=time.time(),
                        level=rule['level'],
                        component=component,
                        message=rule['message'],
                        details=data
                    )
                    
                    self.monitoring_db.save_alert(alert)
                    self._handle_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _handle_alert(self, alert: Alert):
        """Handle alert based on level."""
        handlers = self.alert_handlers.get(alert.level, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to system log."""
        logger.warning(f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message}")
    
    def _send_warning_notification(self, alert: Alert):
        """Send warning notification."""
        # Implement notification system (email, Slack, etc.)
        logger.info(f"Warning notification sent for {alert.component}")
    
    def _send_error_notification(self, alert: Alert):
        """Send error notification."""
        # Implement notification system
        logger.info(f"Error notification sent for {alert.component}")
    
    def _send_critical_notification(self, alert: Alert):
        """Send critical notification."""
        # Implement notification system
        logger.info(f"Critical notification sent for {alert.component}")
    
    def _trigger_recovery(self, alert: Alert):
        """Trigger recovery procedures."""
        logger.info(f"Recovery procedures triggered for {alert.component}")
    
    def _trigger_emergency_stop(self, alert: Alert):
        """Trigger emergency stop."""
        logger.critical(f"EMERGENCY STOP triggered for {alert.component}")

class SchwabotMonitoringSystem:
    """Comprehensive monitoring system for Schwabot."""
    
    def __init__(self):
        self.monitoring_db = MonitoringDatabase()
        self.performance_analyzer = PerformanceAnalyzer(self.monitoring_db)
        self.alert_manager = AlertManager(self.monitoring_db)
        self.metric_queue = queue.Queue()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Component health tracking
        self.component_health = defaultdict(lambda: {
            'status': 'unknown',
            'last_check': 0,
            'error_count': 0,
            'response_time': None
        })
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring system already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Process metrics from queue
                while not self.metric_queue.empty():
                    metric = self.metric_queue.get_nowait()
                    self.monitoring_db.save_system_metric(metric)
                
                # Check component health
                self._check_component_health()
                
                # Generate and save performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def record_metric(self, component: str, metric_name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
        """Record a system metric."""
        metric = SystemMetric(
            timestamp=time.time(),
            component=component,
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata
        )
        
        self.metric_queue.put(metric)
    
    def record_mathematical_state(self, component: str, asset: str, state_data: Dict[str, Any]):
        """Record mathematical state for deep monitoring."""
        state = MathematicalState(
            timestamp=time.time(),
            component=component,
            asset=asset,
            state_hash=state_data.get('state_hash', ''),
            zpe_value=state_data.get('zpe_value', 0.0),
            entropy_value=state_data.get('entropy_value', 0.0),
            vault_state=state_data.get('vault_state', 'idle'),
            lantern_trigger=state_data.get('lantern_trigger', False),
            ghost_echo_active=state_data.get('ghost_echo_active', False),
            quantum_state=state_data.get('quantum_state'),
            vault_entries=state_data.get('vault_entries', 0),
            lantern_corps=state_data.get('lantern_corps', 0),
            ferris_tiers=state_data.get('ferris_tiers', {}),
            strategy_hashes=state_data.get('strategy_hashes', []),
            performance_metrics=state_data.get('performance_metrics', {})
        )
        
        self.monitoring_db.save_mathematical_state(state)
        
        # Check for alerts
        self.alert_manager.check_alerts(component, state_data)
    
    def update_component_health(self, component: str, status: str, response_time: Optional[float] = None, error: Optional[str] = None):
        """Update component health status."""
        health = self.component_health[component]
        health['status'] = status
        health['last_check'] = time.time()
        
        if response_time is not None:
            health['response_time'] = response_time
        
        if error:
            health['error_count'] += 1
            health['last_error'] = error
        else:
            health['error_count'] = 0
            health['last_error'] = None
        
        # Save to database
        self.monitoring_db.save_component_health(
            component, status, response_time, health['error_count'], health['last_error']
        )
        
        # Check for alerts
        self.alert_manager.check_alerts(component, health)
    
    def _check_component_health(self):
        """Check health of all monitored components."""
        # This would typically check actual component health
        # For now, we'll just record the current state
        for component, health in self.component_health.items():
            if time.time() - health['last_check'] > 300:  # 5 minutes
                self.update_component_health(component, 'unknown')
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        # This would typically get metrics from the trading engine
        # For now, we'll use placeholder data
        placeholder_metrics = {
            'total_trades': 100,
            'successful_trades': 65,
            'total_profit': 1500.0,
            'max_drawdown': 0.15,
            'win_rate': 0.65,
            'avg_roi': 0.015,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'max_consecutive_losses': 3,
            'max_consecutive_wins': 8
        }
        
        self.monitoring_db.save_performance_history(placeholder_metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': time.time(),
            'monitoring_active': self.monitoring_active,
            'component_health': dict(self.component_health),
            'active_alerts': len(self.monitoring_db.get_active_alerts()),
            'performance_report': self.performance_analyzer.generate_performance_report(),
            'recent_metrics': self._get_recent_metrics_summary()
        }
    
    def _get_recent_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        # Get metrics for last hour
        cutoff_time = time.time() - 3600
        
        with sqlite3.connect(self.monitoring_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT component, metric_name, AVG(value), COUNT(*)
                FROM system_metrics 
                WHERE timestamp >= ?
                GROUP BY component, metric_name
            """, (cutoff_time,))
            
            rows = cursor.fetchall()
        
        summary = {}
        for row in rows:
            component, metric_name, avg_value, count = row
            if component not in summary:
                summary[component] = {}
            summary[component][metric_name] = {
                'average': avg_value,
                'count': count
            }
        
        return summary

async def main():
    """Test the monitoring system."""
    print("🔍 Testing Schwabot Monitoring System")
    print("=" * 50)
    
    # Initialize monitoring system
    monitoring = SchwabotMonitoringSystem()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Record some test metrics
    print("\n📊 Recording test metrics...")
    monitoring.record_metric('trading_engine', 'signal_generation_rate', 2.5, 'signals/min')
    monitoring.record_metric('trading_engine', 'execution_latency', 150.0, 'ms')
    monitoring.record_metric('mathematical_core', 'zpe_calculation_time', 25.0, 'ms')
    monitoring.record_metric('api_integration', 'api_response_time', 200.0, 'ms')
    
    # Record mathematical state
    print("\n🧮 Recording mathematical state...")
    test_state = {
        'state_hash': 'test_hash_123',
        'zpe_value': 0.6,
        'entropy_value': 0.4,
        'vault_state': 'accumulating',
        'lantern_trigger': True,
        'ghost_echo_active': False,
        'quantum_state': np.array([0.1, 0.2, 0.3, 0.4]),
        'vault_entries': 5,
        'lantern_corps': 3,
        'ferris_tiers': {'tier1': 1.0, 'tier2': 1.25},
        'strategy_hashes': ['hash1', 'hash2', 'hash3'],
        'performance_metrics': {'win_rate': 0.65, 'avg_roi': 0.015}
    }
    monitoring.record_mathematical_state('vault_logic', 'BTCUSDT', test_state)
    
    # Update component health
    print("\n🏥 Updating component health...")
    monitoring.update_component_health('trading_engine', 'healthy', 50.0)
    monitoring.update_component_health('api_integration', 'warning', 500.0, 'High latency')
    monitoring.update_component_health('mathematical_core', 'healthy', 25.0)
    
    # Wait for monitoring to process
    await asyncio.sleep(2)
    
    # Get system status
    print("\n📈 System Status:")
    status = monitoring.get_system_status()
    print(f"  Monitoring Active: {status['monitoring_active']}")
    print(f"  Active Alerts: {status['active_alerts']}")
    print(f"  Component Health: {len(status['component_health'])} components")
    
    # Get performance report
    print("\n📊 Performance Report:")
    report = status['performance_report']
    for window, data in report['windows'].items():
        print(f"  {window} window:")
        print(f"    Win Rate: {data.get('current_win_rate', 0):.2%}")
        print(f"    Avg ROI: {data.get('current_avg_roi', 0):.2%}")
        print(f"    Total Profit: ${data.get('current_total_profit', 0):.2f}")
    
    # Get active alerts
    print("\n🚨 Active Alerts:")
    alerts = monitoring.monitoring_db.get_active_alerts()
    for alert in alerts:
        print(f"  [{alert.level.value.upper()}] {alert.component}: {alert.message}")
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    print(f"\n✅ Monitoring system test completed!")
    print("🎯 Comprehensive monitoring and logging system is operational!")

if __name__ == "__main__":
    asyncio.run(main()) 