#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 NOTIFICATION ROUTES - FLASK API ENDPOINTS
============================================

Flask routes for notification system management.
Integrates with existing Schwabot Flask infrastructure.

Endpoints:
- GET /api/notifications/status - Get notification system status
- POST /api/notifications/test - Test notification channels
- GET /api/notifications/history - Get alert history
- POST /api/notifications/send - Send custom alert
- GET /api/notifications/config - Get notification configuration
- PUT /api/notifications/config - Update notification configuration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request
from flask_socketio import emit

# Import notification system
try:
    from core.notification_system import notification_system
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create Flask blueprint
notifications = Blueprint('notifications', __name__)

@notifications.route('/api/notifications/status', methods=['GET'])
def get_notification_status():
    """Get notification system status and configuration."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available',
                'available': False
            }), 503
        
        status = notification_system.get_delivery_status()
        
        return jsonify({
            'status': 'success',
            'available': True,
            'data': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting notification status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'available': NOTIFICATION_AVAILABLE
        }), 500

@notifications.route('/api/notifications/test', methods=['POST'])
def test_notifications():
    """Test notification channels."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get test parameters from request
        data = request.get_json() or {}
        test_type = data.get('type', 'trade_executed')
        channels = data.get('channels', None)
        
        # Test data
        test_data = {
            'symbol': 'BTC/USDT',
            'trade_type': 'BUY',
            'amount': '0.001',
            'price': '52000',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'profit': '+$25.50',
            'target': '52500',
            'current': '52000',
            'roi': '2.5',
            'loss': '-$15.00',
            'error': 'Test error message',
            'component': 'Test Component',
            'severity': 'medium',
            'total_value': '$10,250.00',
            'change': '+$250.00',
            'top_performer': 'ETH/USDT',
            'event': 'Price Breakout',
            'impact': 'Bullish',
            'recommendation': 'Consider buying'
        }
        
        # Run test
        if channels:
            results = asyncio.run(notification_system.send_alert(
                test_type, test_data, channels=channels
            ))
        else:
            results = asyncio.run(notification_system.send_alert(test_type, test_data))
        
        return jsonify({
            'status': 'success',
            'test_type': test_type,
            'channels': channels,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing notifications: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@notifications.route('/api/notifications/history', methods=['GET'])
def get_alert_history():
    """Get notification alert history."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        alert_type = request.args.get('type', None)
        
        # Get history
        history = notification_system.get_alert_history(limit)
        
        # Filter by type if specified
        if alert_type:
            history = [alert for alert in history if alert.get('type') == alert_type]
        
        return jsonify({
            'status': 'success',
            'data': {
                'history': history,
                'total_count': len(history),
                'limit': limit,
                'alert_type': alert_type
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@notifications.route('/api/notifications/send', methods=['POST'])
def send_custom_alert():
    """Send a custom alert."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        alert_type = data.get('type')
        alert_data = data.get('data', {})
        priority = data.get('priority')
        channels = data.get('channels')
        
        if not alert_type:
            return jsonify({
                'status': 'error',
                'message': 'Alert type is required'
            }), 400
        
        # Send alert
        results = asyncio.run(notification_system.send_alert(
            alert_type, alert_data, priority, channels
        ))
        
        return jsonify({
            'status': 'success',
            'alert_type': alert_type,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error sending custom alert: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@notifications.route('/api/notifications/config', methods=['GET'])
def get_notification_config():
    """Get notification configuration (without sensitive data)."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get configuration without sensitive data
        config = notification_system.config.copy()
        
        # Remove sensitive information
        if 'email' in config:
            config['email']['password'] = '***HIDDEN***'
        if 'sms' in config:
            config['sms']['twilio_auth_token'] = '***HIDDEN***'
        if 'telegram' in config:
            config['telegram']['bot_token'] = '***HIDDEN***'
        
        return jsonify({
            'status': 'success',
            'data': config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting notification config: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@notifications.route('/api/notifications/config', methods=['PUT'])
def update_notification_config():
    """Update notification configuration."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No configuration data provided'
            }), 400
        
        # Update configuration
        # Note: In a production system, you'd want to validate and save to file
        # For now, we'll just update the in-memory config
        for key, value in data.items():
            if key in notification_system.config:
                notification_system.config[key].update(value)
        
        # Reinitialize notification channels
        notification_system.email_enabled = notification_system.config.get('email', {}).get('enabled', False)
        notification_system.sms_enabled = notification_system.config.get('sms', {}).get('enabled', False)
        notification_system.telegram_enabled = notification_system.config.get('telegram', {}).get('enabled', False)
        notification_system.discord_enabled = notification_system.config.get('discord', {}).get('enabled', False)
        notification_system.slack_enabled = notification_system.config.get('slack', {}).get('enabled', False)
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating notification config: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@notifications.route('/api/notifications/rate-limits', methods=['GET'])
def get_rate_limits():
    """Get current rate limit status."""
    try:
        if not NOTIFICATION_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Notification system not available'
            }), 503
        
        # Get rate limit information
        rate_config = notification_system.config.get('rate_limiting', {})
        current_time = time.time()
        
        # Calculate current usage
        hourly_count = sum(1 for v in notification_system.rate_limit_cache.values() 
                          if current_time - v['timestamp'] < 3600)
        daily_count = sum(1 for v in notification_system.rate_limit_cache.values() 
                         if current_time - v['timestamp'] < 86400)
        
        return jsonify({
            'status': 'success',
            'data': {
                'hourly_limit': rate_config.get('max_alerts_per_hour', 10),
                'hourly_used': hourly_count,
                'hourly_remaining': max(0, rate_config.get('max_alerts_per_hour', 10) - hourly_count),
                'daily_limit': rate_config.get('max_alerts_per_day', 50),
                'daily_used': daily_count,
                'daily_remaining': max(0, rate_config.get('max_alerts_per_day', 50) - daily_count),
                'cooldown_minutes': rate_config.get('cooldown_minutes', 5)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# WebSocket event for real-time notification updates
def emit_notification_update(alert_data: Dict[str, Any]):
    """Emit notification update to connected WebSocket clients."""
    try:
        if NOTIFICATION_AVAILABLE:
            emit('notification_update', {
                'type': 'new_alert',
                'data': alert_data,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Error emitting notification update: {e}")

# Helper function to send trading alerts
def send_trading_alert(alert_type: str, trading_data: Dict[str, Any]):
    """Helper function to send trading-related alerts."""
    try:
        if NOTIFICATION_AVAILABLE:
            # Add timestamp if not present
            if 'timestamp' not in trading_data:
                trading_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Send alert asynchronously
            asyncio.create_task(notification_system.send_alert(alert_type, trading_data))
            
            # Emit WebSocket update
            emit_notification_update({
                'type': alert_type,
                'data': trading_data,
                'status': 'sent'
            })
            
            logger.info(f"Trading alert sent: {alert_type}")
            
    except Exception as e:
        logger.error(f"Error sending trading alert: {e}")

# Helper function to send system alerts
def send_system_alert(error_message: str, component: str, severity: str = 'medium'):
    """Helper function to send system error alerts."""
    try:
        if NOTIFICATION_AVAILABLE:
            alert_data = {
                'error': error_message,
                'component': component,
                'severity': severity,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Send alert asynchronously
            asyncio.create_task(notification_system.send_alert('system_error', alert_data))
            
            # Emit WebSocket update
            emit_notification_update({
                'type': 'system_error',
                'data': alert_data,
                'status': 'sent'
            })
            
            logger.info(f"System alert sent: {component} - {severity}")
            
    except Exception as e:
        logger.error(f"Error sending system alert: {e}")

# Import time module for rate limit calculations
import time 