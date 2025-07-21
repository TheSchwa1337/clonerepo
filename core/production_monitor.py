#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 PRODUCTION MONITOR - ENTERPRISE MONITORING SYSTEM
===================================================

Comprehensive production monitoring system for Schwabot trading platform.
Provides real-time metrics, alerting, and performance tracking.

Features:
- Real-time system performance monitoring
- Trading performance metrics
- API endpoint monitoring
- Database performance tracking
- Memory and CPU usage monitoring
- Network latency monitoring
- Custom metric collection
- Alert threshold management
- Performance trend analysis
- Health check automation
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, deque

import requests
import yaml

# Import existing Schwabot components
try:
    from .system_health_monitor import system_health_monitor
    from .notification_system import notification_system
    from .encryption_manager import encryption_manager
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError:
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProductionMonitor:
    """Enterprise-grade production monitoring system."""
    
    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        self.config = self._load_config(config_path)
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history = []
        self.health_checks = {}
        self.performance_baselines = {}
        self.monitoring_active = False
        self.monitor_task = None
        
        # Initialize monitoring components
        self._initialize_monitoring()
        
        logger.info("Production monitor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Monitoring config not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading monitoring config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'monitoring': {
                'enabled': True,
                'interval_seconds': 30,
                'retention_days': 30,
                'max_history_size': 1000
            },
            'metrics': {
                'system': {
                    'cpu_usage': {'threshold': 80, 'alert': True},
                    'memory_usage': {'threshold': 85, 'alert': True},
                    'disk_usage': {'threshold': 90, 'alert': True},
                    'network_latency': {'threshold': 100, 'alert': True}
                },
                'trading': {
                    'api_response_time': {'threshold': 1000, 'alert': True},
                    'trade_execution_time': {'threshold': 5000, 'alert': True},
                    'error_rate': {'threshold': 5, 'alert': True},
                    'profit_loss': {'threshold': -1000, 'alert': True}
                },
                'application': {
                    'request_rate': {'threshold': 1000, 'alert': False},
                    'error_count': {'threshold': 10, 'alert': True},
                    'active_connections': {'threshold': 100, 'alert': False},
                    'queue_size': {'threshold': 50, 'alert': True}
                }
            },
            'alerts': {
                'email_enabled': True,
                'slack_enabled': False,
                'telegram_enabled': False,
                'cooldown_minutes': 15
            },
            'health_checks': {
                'api_endpoints': [
                    'http://localhost:5000/api/status',
                    'http://localhost:5000/api/health'
                ],
                'database': 'sqlite:///data/schwabot.db',
                'interval_seconds': 60
            }
        }
    
    def _initialize_monitoring(self):
        """Initialize monitoring components."""
        try:
            # Initialize health checks
            self._setup_health_checks()
            
            # Initialize performance baselines
            self._initialize_baselines()
            
            # Setup alert thresholds
            self._setup_alert_thresholds()
            
            logger.info("Monitoring components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring: {e}")
    
    def _setup_health_checks(self):
        """Setup health check endpoints."""
        try:
            health_config = self.config.get('health_checks', {})
            
            # API endpoint health checks
            api_endpoints = health_config.get('api_endpoints', [])
            for endpoint in api_endpoints:
                self.health_checks[f"api_{endpoint}"] = {
                    'url': endpoint,
                    'type': 'http',
                    'timeout': 10,
                    'expected_status': 200
                }
            
            # Database health check
            db_url = health_config.get('database')
            if db_url:
                self.health_checks['database'] = {
                    'url': db_url,
                    'type': 'database',
                    'timeout': 5
                }
            
            logger.info(f"Health checks configured: {len(self.health_checks)} endpoints")
            
        except Exception as e:
            logger.error(f"Error setting up health checks: {e}")
    
    def _initialize_baselines(self):
        """Initialize performance baselines."""
        try:
            # Get initial system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            self.performance_baselines = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_latency': 50,  # Default baseline
                'api_response_time': 200,  # Default baseline
                'trade_execution_time': 1000,  # Default baseline
                'error_rate': 0.1  # Default baseline
            }
            
            logger.info("Performance baselines initialized")
            
        except Exception as e:
            logger.error(f"Error initializing baselines: {e}")
    
    def _setup_alert_thresholds(self):
        """Setup alert thresholds from configuration."""
        try:
            metrics_config = self.config.get('metrics', {})
            
            for category, metrics in metrics_config.items():
                for metric_name, config in metrics.items():
                    threshold = config.get('threshold', 80)
                    alert_enabled = config.get('alert', True)
                    
                    # Store threshold configuration
                    self.alert_thresholds = getattr(self, 'alert_thresholds', {})
                    self.alert_thresholds[f"{category}_{metric_name}"] = {
                        'threshold': threshold,
                        'alert_enabled': alert_enabled,
                        'last_alert': None
                    }
            
            logger.info(f"Alert thresholds configured: {len(self.alert_thresholds)} metrics")
            
        except Exception as e:
            logger.error(f"Error setting up alert thresholds: {e}")
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return
            
            self.monitoring_active = True
            interval = self.config.get('monitoring', {}).get('interval_seconds', 30)
            
            logger.info(f"Starting production monitoring (interval: {interval}s)")
            
            # Start monitoring loop
            self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        try:
            if not self.monitoring_active:
                return
            
            self.monitoring_active = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Production monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Collect trading metrics
                trading_metrics = await self._collect_trading_metrics()
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                # Store metrics
                timestamp = datetime.now().isoformat()
                self._store_metrics('system', system_metrics, timestamp)
                self._store_metrics('trading', trading_metrics, timestamp)
                self._store_metrics('application', app_metrics, timestamp)
                self._store_metrics('health', health_status, timestamp)
                
                # Check for alerts
                await self._check_alerts(system_metrics, trading_metrics, app_metrics)
                
                # Update baselines
                self._update_baselines(system_metrics, trading_metrics, app_metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Network latency (ping localhost)
            network_latency = await self._measure_network_latency()
            
            return {
                'cpu_usage': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq': cpu_freq.current if cpu_freq else 0,
                'memory_usage': memory_percent,
                'memory_available_gb': memory_available,
                'disk_usage': disk_percent,
                'disk_free_gb': disk_free,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'network_latency_ms': network_latency
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def _collect_trading_metrics(self) -> Dict[str, Any]:
        """Collect trading performance metrics."""
        try:
            # This would integrate with your existing trading system
            # For now, we'll collect basic metrics
            
            # API response time (simulated)
            api_response_time = await self._measure_api_response_time()
            
            # Trade execution time (simulated)
            trade_execution_time = await self._measure_trade_execution_time()
            
            # Error rate (simulated)
            error_rate = await self._calculate_error_rate()
            
            # Profit/Loss (simulated)
            profit_loss = await self._get_profit_loss()
            
            return {
                'api_response_time_ms': api_response_time,
                'trade_execution_time_ms': trade_execution_time,
                'error_rate_percent': error_rate,
                'profit_loss_usd': profit_loss,
                'active_trades': 0,  # Would get from trading system
                'total_trades_today': 0,  # Would get from trading system
                'success_rate_percent': 95.0  # Would calculate from trading system
            }
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application performance metrics."""
        try:
            # Request rate (simulated)
            request_rate = await self._calculate_request_rate()
            
            # Error count
            error_count = await self._get_error_count()
            
            # Active connections
            active_connections = await self._get_active_connections()
            
            # Queue size
            queue_size = await self._get_queue_size()
            
            return {
                'request_rate_per_minute': request_rate,
                'error_count': error_count,
                'active_connections': active_connections,
                'queue_size': queue_size,
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2)
            }
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform health checks on configured endpoints."""
        try:
            health_results = {}
            
            for check_name, check_config in self.health_checks.items():
                try:
                    if check_config['type'] == 'http':
                        result = await self._check_http_endpoint(check_config)
                    elif check_config['type'] == 'database':
                        result = await self._check_database_connection(check_config)
                    else:
                        result = {'status': 'unknown', 'error': f"Unknown check type: {check_config['type']}"}
                    
                    health_results[check_name] = result
                    
                except Exception as e:
                    health_results[check_name] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            
            return health_results
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return {}
    
    async def _check_http_endpoint(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check HTTP endpoint health."""
        try:
            start_time = time.time()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(
                    config['url'],
                    timeout=config.get('timeout', 10)
                )
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            expected_status = config.get('expected_status', 200)
            status = 'healthy' if response.status_code == expected_status else 'unhealthy'
            
            return {
                'status': status,
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time_ms': None,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_database_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            # This would integrate with your database system
            # For now, we'll simulate a database check
            
            start_time = time.time()
            
            # Simulate database connection test
            await asyncio.sleep(0.1)
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _measure_network_latency(self) -> float:
        """Measure network latency."""
        try:
            # Simulate network latency measurement
            # In production, you'd ping actual endpoints
            return 5.0 + (time.time() % 10)  # Simulated latency 5-15ms
        except Exception as e:
            logger.error(f"Error measuring network latency: {e}")
            return 0.0
    
    async def _measure_api_response_time(self) -> float:
        """Measure API response time."""
        try:
            # Simulate API response time measurement
            return 50.0 + (time.time() % 100)  # Simulated 50-150ms
        except Exception as e:
            logger.error(f"Error measuring API response time: {e}")
            return 0.0
    
    async def _measure_trade_execution_time(self) -> float:
        """Measure trade execution time."""
        try:
            # Simulate trade execution time measurement
            return 1000.0 + (time.time() % 2000)  # Simulated 1-3 seconds
        except Exception as e:
            logger.error(f"Error measuring trade execution time: {e}")
            return 0.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate."""
        try:
            # Simulate error rate calculation
            return 0.5 + (time.time() % 2)  # Simulated 0.5-2.5%
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")
            return 0.0
    
    async def _get_profit_loss(self) -> float:
        """Get current profit/loss."""
        try:
            # Simulate profit/loss calculation
            return 100.0 + (time.time() % 500)  # Simulated $100-$600 profit
        except Exception as e:
            logger.error(f"Error getting profit/loss: {e}")
            return 0.0
    
    async def _calculate_request_rate(self) -> float:
        """Calculate request rate."""
        try:
            # Simulate request rate calculation
            return 50.0 + (time.time() % 100)  # Simulated 50-150 requests/min
        except Exception as e:
            logger.error(f"Error calculating request rate: {e}")
            return 0.0
    
    async def _get_error_count(self) -> int:
        """Get current error count."""
        try:
            # Simulate error count
            return int(time.time() % 5)  # Simulated 0-4 errors
        except Exception as e:
            logger.error(f"Error getting error count: {e}")
            return 0
    
    async def _get_active_connections(self) -> int:
        """Get active connections count."""
        try:
            # Simulate active connections
            return int(10 + (time.time() % 20))  # Simulated 10-30 connections
        except Exception as e:
            logger.error(f"Error getting active connections: {e}")
            return 0
    
    async def _get_queue_size(self) -> int:
        """Get current queue size."""
        try:
            # Simulate queue size
            return int(time.time() % 10)  # Simulated 0-9 items in queue
        except Exception as e:
            logger.error(f"Error getting queue size: {e}")
            return 0
    
    def _store_metrics(self, category: str, metrics: Dict[str, Any], timestamp: str):
        """Store metrics in history."""
        try:
            for metric_name, value in metrics.items():
                key = f"{category}_{metric_name}"
                self.metrics_history[key].append({
                    'value': value,
                    'timestamp': timestamp
                })
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    async def _check_alerts(self, system_metrics: Dict[str, Any], 
                           trading_metrics: Dict[str, Any], 
                           app_metrics: Dict[str, Any]):
        """Check for alert conditions."""
        try:
            all_metrics = {**system_metrics, **trading_metrics, **app_metrics}
            
            for metric_name, value in all_metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                
                threshold_key = f"system_{metric_name}"
                if threshold_key in self.alert_thresholds:
                    threshold_config = self.alert_thresholds[threshold_key]
                    
                    if threshold_config['alert_enabled'] and value > threshold_config['threshold']:
                        await self._trigger_alert(metric_name, value, threshold_config['threshold'])
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger an alert."""
        try:
            # Check cooldown
            cooldown_minutes = self.config.get('alerts', {}).get('cooldown_minutes', 15)
            last_alert = self.alert_thresholds.get(f"system_{metric_name}", {}).get('last_alert')
            
            if last_alert:
                last_alert_time = datetime.fromisoformat(last_alert)
                if datetime.now() - last_alert_time < timedelta(minutes=cooldown_minutes):
                    return  # Still in cooldown
            
            # Create alert
            alert = {
                'metric': metric_name,
                'value': value,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat(),
                'severity': 'warning' if value < threshold * 1.2 else 'critical'
            }
            
            # Store alert
            self.alert_history.append(alert)
            
            # Update last alert time
            if f"system_{metric_name}" in self.alert_thresholds:
                self.alert_thresholds[f"system_{metric_name}"]['last_alert'] = alert['timestamp']
            
            # Send notification if available
            if SCHWABOT_COMPONENTS_AVAILABLE:
                try:
                    await notification_system.send_alert('system_error', {
                        'error': f"Metric {metric_name} exceeded threshold",
                        'component': 'Production Monitor',
                        'severity': alert['severity'],
                        'value': value,
                        'threshold': threshold
                    })
                except Exception as e:
                    logger.error(f"Error sending alert notification: {e}")
            
            logger.warning(f"Alert triggered: {metric_name} = {value} (threshold: {threshold})")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _update_baselines(self, system_metrics: Dict[str, Any], 
                         trading_metrics: Dict[str, Any], 
                         app_metrics: Dict[str, Any]):
        """Update performance baselines."""
        try:
            all_metrics = {**system_metrics, **trading_metrics, **app_metrics}
            
            for metric_name, value in all_metrics.items():
                if isinstance(value, (int, float)) and metric_name in self.performance_baselines:
                    # Simple moving average update
                    current_baseline = self.performance_baselines[metric_name]
                    self.performance_baselines[metric_name] = (current_baseline + value) / 2
            
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
    
    def get_metrics(self, category: Optional[str] = None, 
                   metric_name: Optional[str] = None, 
                   limit: int = 100) -> Dict[str, Any]:
        """Get metrics from history."""
        try:
            if category and metric_name:
                key = f"{category}_{metric_name}"
                return {
                    'metric': key,
                    'data': list(self.metrics_history[key])[-limit:]
                }
            elif category:
                result = {}
                for key, data in self.metrics_history.items():
                    if key.startswith(f"{category}_"):
                        result[key] = list(data)[-limit:]
                return result
            else:
                result = {}
                for key, data in self.metrics_history.items():
                    result[key] = list(data)[-limit:]
                return result
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # Get latest metrics
            latest_metrics = {}
            for key, data in self.metrics_history.items():
                if data:
                    latest_metrics[key] = data[-1]
            
            return {
                'monitoring_active': self.monitoring_active,
                'total_metrics': len(self.metrics_history),
                'total_alerts': len(self.alert_history),
                'latest_metrics': latest_metrics,
                'performance_baselines': self.performance_baselines,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {}
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def export_metrics(self, filename: str, format: str = 'json') -> bool:
        """Export metrics to file."""
        try:
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(dict(self.metrics_history), f, indent=2)
            elif format == 'csv':
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric', 'value', 'timestamp'])
                    for key, data in self.metrics_history.items():
                        for entry in data:
                            writer.writerow([key, entry['value'], entry['timestamp']])
            
            logger.info(f"Metrics exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False

# Global production monitor instance
production_monitor = ProductionMonitor() 