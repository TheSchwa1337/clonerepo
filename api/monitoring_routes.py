#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITORING ROUTES - FLASK API ENDPOINTS
=========================================

Flask routes for production monitoring system.
Integrates with existing Schwabot Flask infrastructure.

Endpoints:
- GET /api/monitoring/status - Get monitoring system status
- GET /api/monitoring/metrics - Get current metrics
- GET /api/monitoring/alerts - Get alert history
- POST /api/monitoring/start - Start monitoring
- POST /api/monitoring/stop - Stop monitoring
- GET /api/monitoring/health - Get health check results
- GET /api/monitoring/baselines - Get performance baselines
- POST /api/monitoring/export - Export metrics data
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, send_file
from flask_socketio import emit

# Import monitoring system
try:
    from core.production_monitor import production_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create Flask blueprint
monitoring = Blueprint('monitoring', __name__)

@monitoring.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """Get monitoring system status."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available',
                'available': False
            }), 503
        
        status = production_monitor.get_current_status()
        
        return jsonify({
            'status': 'success',
            'available': True,
            'data': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'available': MONITORING_AVAILABLE
        }), 500

@monitoring.route('/api/monitoring/metrics', methods=['GET'])
def get_metrics():
    """Get metrics data."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get query parameters
        category = request.args.get('category', None)
        metric_name = request.args.get('metric', None)
        limit = request.args.get('limit', 100, type=int)
        
        # Get metrics
        metrics = production_monitor.get_metrics(category, metric_name, limit)
        
        return jsonify({
            'status': 'success',
            'data': {
                'metrics': metrics,
                'category': category,
                'metric_name': metric_name,
                'limit': limit,
                'total_metrics': len(metrics)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/alerts', methods=['GET'])
def get_alerts():
    """Get alert history."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        severity = request.args.get('severity', None)
        
        # Get alerts
        alerts = production_monitor.get_alert_history(limit)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.get('severity') == severity]
        
        return jsonify({
            'status': 'success',
            'data': {
                'alerts': alerts,
                'total_count': len(alerts),
                'limit': limit,
                'severity': severity
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start the monitoring system."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Start monitoring asynchronously
        asyncio.create_task(production_monitor.start_monitoring())
        
        return jsonify({
            'status': 'success',
            'message': 'Monitoring started successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop the monitoring system."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Stop monitoring asynchronously
        asyncio.create_task(production_monitor.stop_monitoring())
        
        return jsonify({
            'status': 'success',
            'message': 'Monitoring stopped successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/health', methods=['GET'])
def get_health_checks():
    """Get health check results."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get health metrics
        health_metrics = production_monitor.get_metrics('health', limit=1)
        
        # Calculate overall health status
        overall_status = 'healthy'
        failed_checks = 0
        
        for check_name, check_data in health_metrics.items():
            if check_data and check_data[0].get('value', {}).get('status') != 'healthy':
                failed_checks += 1
        
        if failed_checks > 0:
            overall_status = 'unhealthy' if failed_checks > len(health_metrics) / 2 else 'degraded'
        
        return jsonify({
            'status': 'success',
            'data': {
                'overall_status': overall_status,
                'health_checks': health_metrics,
                'failed_checks': failed_checks,
                'total_checks': len(health_metrics)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting health checks: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/baselines', methods=['GET'])
def get_baselines():
    """Get performance baselines."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        baselines = production_monitor.performance_baselines
        
        return jsonify({
            'status': 'success',
            'data': {
                'baselines': baselines,
                'total_baselines': len(baselines)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting baselines: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/export', methods=['POST'])
def export_metrics():
    """Export metrics data."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get request data
        data = request.get_json() or {}
        format_type = data.get('format', 'json')
        filename = data.get('filename', f'metrics_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Add file extension
        if format_type == 'json':
            filename += '.json'
        elif format_type == 'csv':
            filename += '.csv'
        
        # Export metrics
        success = production_monitor.export_metrics(filename, format_type)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Metrics exported to {filename}',
                'data': {
                    'filename': filename,
                    'format': format_type
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to export metrics'
            }), 500
        
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/thresholds', methods=['GET'])
def get_thresholds():
    """Get alert thresholds configuration."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        thresholds = getattr(production_monitor, 'alert_thresholds', {})
        
        return jsonify({
            'status': 'success',
            'data': {
                'thresholds': thresholds,
                'total_thresholds': len(thresholds)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting thresholds: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/thresholds', methods=['PUT'])
def update_thresholds():
    """Update alert thresholds."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No threshold data provided'
            }), 400
        
        # Update thresholds
        thresholds = getattr(production_monitor, 'alert_thresholds', {})
        
        for metric_name, threshold_config in data.items():
            if metric_name in thresholds:
                thresholds[metric_name].update(threshold_config)
        
        return jsonify({
            'status': 'success',
            'message': 'Thresholds updated successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@monitoring.route('/api/monitoring/summary', methods=['GET'])
def get_monitoring_summary():
    """Get monitoring system summary."""
    try:
        if not MONITORING_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Monitoring system not available'
            }), 503
        
        # Get current status
        status = production_monitor.get_current_status()
        
        # Get recent alerts
        recent_alerts = production_monitor.get_alert_history(10)
        
        # Get latest metrics
        latest_metrics = {}
        for key, data in production_monitor.metrics_history.items():
            if data:
                latest_metrics[key] = data[-1]
        
        # Calculate summary statistics
        total_metrics = len(production_monitor.metrics_history)
        total_alerts = len(production_monitor.alert_history)
        monitoring_active = production_monitor.monitoring_active
        
        # Calculate system health
        system_health = 'healthy'
        if recent_alerts:
            critical_alerts = [alert for alert in recent_alerts if alert.get('severity') == 'critical']
            if critical_alerts:
                system_health = 'critical'
            elif len(recent_alerts) > 5:
                system_health = 'warning'
        
        return jsonify({
            'status': 'success',
            'data': {
                'summary': {
                    'system_health': system_health,
                    'monitoring_active': monitoring_active,
                    'total_metrics': total_metrics,
                    'total_alerts': total_alerts,
                    'recent_alerts_count': len(recent_alerts)
                },
                'latest_metrics': latest_metrics,
                'recent_alerts': recent_alerts,
                'performance_baselines': production_monitor.performance_baselines
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# WebSocket event for real-time monitoring updates
def emit_monitoring_update(update_data: Dict[str, Any]):
    """Emit monitoring update to connected WebSocket clients."""
    try:
        if MONITORING_AVAILABLE:
            emit('monitoring_update', {
                'type': 'metrics_update',
                'data': update_data,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Error emitting monitoring update: {e}")

# Helper function to get system overview
def get_system_overview() -> Dict[str, Any]:
    """Get system overview for dashboard."""
    try:
        if not MONITORING_AVAILABLE:
            return {
                'monitoring_available': False,
                'status': 'unavailable'
            }
        
        status = production_monitor.get_current_status()
        
        # Extract key metrics
        latest_metrics = status.get('latest_metrics', {})
        
        # Calculate key performance indicators
        cpu_usage = latest_metrics.get('system_cpu_usage', {}).get('value', 0)
        memory_usage = latest_metrics.get('system_memory_usage', {}).get('value', 0)
        disk_usage = latest_metrics.get('system_disk_usage', {}).get('value', 0)
        
        # Determine overall system status
        if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
            system_status = 'critical'
        elif cpu_usage > 80 or memory_usage > 80 or disk_usage > 90:
            system_status = 'warning'
        else:
            system_status = 'healthy'
        
        return {
            'monitoring_available': True,
            'status': system_status,
            'monitoring_active': production_monitor.monitoring_active,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'total_alerts': len(production_monitor.alert_history),
            'total_metrics': len(production_monitor.metrics_history)
        }
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        return {
            'monitoring_available': False,
            'status': 'error',
            'error': str(e)
        } 