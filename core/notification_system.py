#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 NOTIFICATION SYSTEM - PRODUCTION ALERT MANAGEMENT
====================================================

Comprehensive notification system for Schwabot trading alerts.
Integrates with existing Flask infrastructure for email, SMS, and Telegram notifications.

Features:
- Email notifications (SMTP)
- SMS notifications (Twilio)
- Telegram bot notifications
- Discord webhook notifications
- Slack webhook notifications
- Alert priority management
- Rate limiting and throttling
- Template-based messaging
- Delivery confirmation tracking
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import requests
import yaml

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Production notification system for Schwabot trading alerts."""
    
    def __init__(self, config_path: str = "config/notification_config.yaml"):
        self.config = self._load_config(config_path)
        self.alert_history = []
        self.rate_limit_cache = {}
        self.delivery_status = {}
        self.templates = self._load_templates()
        
        # Initialize notification channels
        self.email_enabled = self.config.get('email', {}).get('enabled', False)
        self.sms_enabled = self.config.get('sms', {}).get('enabled', False)
        self.telegram_enabled = self.config.get('telegram', {}).get('enabled', False)
        self.discord_enabled = self.config.get('discord', {}).get('enabled', False)
        self.slack_enabled = self.config.get('slack', {}).get('enabled', False)
        
        logger.info(f"Notification system initialized - Email: {self.email_enabled}, "
                   f"SMS: {self.sms_enabled}, Telegram: {self.telegram_enabled}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load notification configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Notification config not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading notification config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default notification configuration."""
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': [],
                'from_address': 'schwabot@yourdomain.com'
            },
            'sms': {
                'enabled': False,
                'twilio_account_sid': '',
                'twilio_auth_token': '',
                'twilio_phone_number': '',
                'recipients': []
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': '',
                'parse_mode': 'HTML'
            },
            'discord': {
                'enabled': False,
                'webhook_url': ''
            },
            'slack': {
                'enabled': False,
                'webhook_url': ''
            },
            'rate_limiting': {
                'max_alerts_per_hour': 10,
                'max_alerts_per_day': 50,
                'cooldown_minutes': 5
            },
            'alert_types': {
                'trade_executed': {'priority': 'high', 'channels': ['email', 'telegram']},
                'profit_target': {'priority': 'high', 'channels': ['email', 'sms', 'telegram']},
                'stop_loss': {'priority': 'critical', 'channels': ['email', 'sms', 'telegram', 'discord']},
                'system_error': {'priority': 'critical', 'channels': ['email', 'sms', 'telegram', 'slack']},
                'portfolio_update': {'priority': 'medium', 'channels': ['email', 'telegram']},
                'market_alert': {'priority': 'medium', 'channels': ['telegram', 'discord']}
            }
        }
    
    def _load_templates(self) -> Dict[str, str]:
        """Load notification message templates."""
        return {
            'trade_executed': """
🚀 **Trade Executed**
Symbol: {symbol}
Type: {trade_type}
Amount: {amount}
Price: {price}
Time: {timestamp}
Profit: {profit}
            """,
            'profit_target': """
💰 **Profit Target Reached**
Symbol: {symbol}
Target: {target}
Current: {current}
Profit: {profit}
ROI: {roi}%
            """,
            'stop_loss': """
⚠️ **Stop Loss Triggered**
Symbol: {symbol}
Loss: {loss}
Price: {price}
Time: {timestamp}
            """,
            'system_error': """
🚨 **System Error**
Error: {error}
Component: {component}
Time: {timestamp}
Severity: {severity}
            """,
            'portfolio_update': """
📊 **Portfolio Update**
Total Value: {total_value}
Change: {change}
ROI: {roi}%
Top Performer: {top_performer}
            """,
            'market_alert': """
📈 **Market Alert**
Symbol: {symbol}
Event: {event}
Impact: {impact}
Recommendation: {recommendation}
            """
        }
    
    async def send_alert(self, alert_type: str, data: Dict[str, Any], 
                        priority: Optional[str] = None, 
                        channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send alert through configured channels.
        
        Args:
            alert_type: Type of alert (trade_executed, profit_target, etc.)
            data: Alert data for template formatting
            priority: Override priority level
            channels: Override notification channels
            
        Returns:
            Dictionary with delivery status for each channel
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit(alert_type):
                return {'status': 'rate_limited', 'message': 'Rate limit exceeded'}
            
            # Get alert configuration
            alert_config = self.config['alert_types'].get(alert_type, {})
            alert_priority = priority or alert_config.get('priority', 'medium')
            alert_channels = channels or alert_config.get('channels', ['email'])
            
            # Format message
            template = self.templates.get(alert_type, "{alert_type}: {data}")
            message = template.format(**data)
            
            # Create alert record
            alert_id = f"{alert_type}_{int(time.time())}"
            alert_record = {
                'id': alert_id,
                'type': alert_type,
                'priority': alert_priority,
                'message': message,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'channels': alert_channels
            }
            
            # Send to each channel
            delivery_results = {}
            for channel in alert_channels:
                if channel == 'email' and self.email_enabled:
                    delivery_results['email'] = await self._send_email(message, alert_priority, data)
                elif channel == 'sms' and self.sms_enabled:
                    delivery_results['sms'] = await self._send_sms(message, alert_priority, data)
                elif channel == 'telegram' and self.telegram_enabled:
                    delivery_results['telegram'] = await self._send_telegram(message, alert_priority, data)
                elif channel == 'discord' and self.discord_enabled:
                    delivery_results['discord'] = await self._send_discord(message, alert_priority, data)
                elif channel == 'slack' and self.slack_enabled:
                    delivery_results['slack'] = await self._send_slack(message, alert_priority, data)
                else:
                    delivery_results[channel] = {'status': 'disabled', 'message': f'{channel} notifications disabled'}
            
            # Store alert history
            alert_record['delivery_results'] = delivery_results
            self.alert_history.append(alert_record)
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            logger.info(f"Alert sent: {alert_type} - {delivery_results}")
            return delivery_results
            
        except Exception as e:
            logger.error(f"Error sending alert {alert_type}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_email(self, message: str, priority: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification."""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{priority.upper()}] Schwabot Alert - {data.get('symbol', 'System')}"
            
            # Add priority headers
            if priority == 'critical':
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Create HTML message
            html_message = f"""
            <html>
            <body>
                <h2>🚀 Schwabot Trading Alert</h2>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <pre>{message}</pre>
                <hr>
                <p><em>This is an automated message from your Schwabot trading system.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            return {'status': 'sent', 'message': 'Email sent successfully'}
            
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_sms(self, message: str, priority: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS notification via Twilio."""
        try:
            sms_config = self.config['sms']
            
            # Truncate message for SMS
            if len(message) > 160:
                message = message[:157] + "..."
            
            for recipient in sms_config['recipients']:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{sms_config['twilio_account_sid']}/Messages.json"
                
                payload = {
                    'To': recipient,
                    'From': sms_config['twilio_phone_number'],
                    'Body': f"[{priority.upper()}] {message}"
                }
                
                response = requests.post(
                    url,
                    data=payload,
                    auth=(sms_config['twilio_account_sid'], sms_config['twilio_auth_token'])
                )
                
                if response.status_code != 201:
                    logger.error(f"SMS send error: {response.text}")
                    return {'status': 'error', 'message': response.text}
            
            return {'status': 'sent', 'message': 'SMS sent successfully'}
            
        except Exception as e:
            logger.error(f"SMS send error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_telegram(self, message: str, priority: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send Telegram notification."""
        try:
            telegram_config = self.config['telegram']
            
            # Format message for Telegram
            formatted_message = f"🚀 *Schwabot Alert*\n\n"
            formatted_message += f"*Priority:* {priority.upper()}\n"
            formatted_message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            formatted_message += message
            
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            
            payload = {
                'chat_id': telegram_config['chat_id'],
                'text': formatted_message,
                'parse_mode': telegram_config['parse_mode']
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Telegram send error: {response.text}")
                return {'status': 'error', 'message': response.text}
            
            return {'status': 'sent', 'message': 'Telegram message sent successfully'}
            
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_discord(self, message: str, priority: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send Discord webhook notification."""
        try:
            discord_config = self.config['discord']
            
            # Create Discord embed
            embed = {
                'title': f'🚀 Schwabot Alert - {priority.upper()}',
                'description': message,
                'color': 0x00ff00 if priority == 'high' else 0xff0000 if priority == 'critical' else 0xffff00,
                'timestamp': datetime.now().isoformat(),
                'footer': {
                    'text': 'Schwabot Trading System'
                }
            }
            
            payload = {
                'embeds': [embed]
            }
            
            response = requests.post(discord_config['webhook_url'], json=payload)
            
            if response.status_code != 204:
                logger.error(f"Discord send error: {response.text}")
                return {'status': 'error', 'message': response.text}
            
            return {'status': 'sent', 'message': 'Discord message sent successfully'}
            
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_slack(self, message: str, priority: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack webhook notification."""
        try:
            slack_config = self.config['slack']
            
            # Format message for Slack
            slack_message = f"🚀 *Schwabot Alert*\n"
            slack_message += f"*Priority:* {priority.upper()}\n"
            slack_message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            slack_message += message
            
            payload = {
                'text': slack_message
            }
            
            response = requests.post(slack_config['webhook_url'], json=payload)
            
            if response.status_code != 200:
                logger.error(f"Slack send error: {response.text}")
                return {'status': 'error', 'message': response.text}
            
            return {'status': 'sent', 'message': 'Slack message sent successfully'}
            
        except Exception as e:
            logger.error(f"Slack send error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if alert is within rate limits."""
        try:
            rate_config = self.config['rate_limiting']
            current_time = time.time()
            
            # Clean old entries
            self.rate_limit_cache = {
                k: v for k, v in self.rate_limit_cache.items() 
                if current_time - v['timestamp'] < 3600  # 1 hour
            }
            
            # Check hourly limit
            hourly_count = sum(1 for v in self.rate_limit_cache.values() 
                             if current_time - v['timestamp'] < 3600)
            
            if hourly_count >= rate_config['max_alerts_per_hour']:
                return False
            
            # Check daily limit
            daily_count = sum(1 for v in self.rate_limit_cache.values() 
                            if current_time - v['timestamp'] < 86400)  # 24 hours
            
            if daily_count >= rate_config['max_alerts_per_day']:
                return False
            
            # Add to cache
            self.rate_limit_cache[f"{alert_type}_{current_time}"] = {
                'type': alert_type,
                'timestamp': current_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow if rate limiting fails
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_delivery_status(self) -> Dict[str, Any]:
        """Get notification delivery status."""
        return {
            'email_enabled': self.email_enabled,
            'sms_enabled': self.sms_enabled,
            'telegram_enabled': self.telegram_enabled,
            'discord_enabled': self.discord_enabled,
            'slack_enabled': self.slack_enabled,
            'total_alerts_sent': len(self.alert_history),
            'recent_alerts': self.get_alert_history(10)
        }
    
    def test_notifications(self) -> Dict[str, Any]:
        """Test all notification channels."""
        test_data = {
            'symbol': 'BTC/USDT',
            'trade_type': 'BUY',
            'amount': '0.001',
            'price': '52000',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'profit': '+$25.50'
        }
        
        results = {}
        for alert_type in ['trade_executed', 'system_error']:
            results[alert_type] = asyncio.run(self.send_alert(alert_type, test_data))
        
        return results

# Global notification system instance
notification_system = NotificationSystem() 