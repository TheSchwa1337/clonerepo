#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 HIGH-VOLUME TRADING DASHBOARD
================================

Real-time monitoring dashboard for high-volume trading operations.
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from core.high_volume_trading_manager import high_volume_trading_manager
    from core.system_health_monitor import system_health_monitor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all core modules are available.")
    sys.exit(1)

class HighVolumeDashboard:
    """Real-time high-volume trading dashboard."""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.update_interval = 5.0  # seconds
        self.running = False
        self.update_thread = None
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.running = True
        self.update_thread = threading.Thread(target=self._monitoring_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        print("📊 High-volume trading dashboard started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        print("📊 High-volume trading dashboard stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.update_metrics()
                self.check_alerts()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"❌ Dashboard monitoring error: {e}")
                time.sleep(self.update_interval)
        
    def update_metrics(self):
        """Update real-time metrics."""
        try:
            # Get system status
            status = high_volume_trading_manager.get_system_status()
            
            # Get performance metrics
            perf_metrics = status.get('performance_metrics', {})
            
            # Get system health
            system_health = system_health_monitor.get_full_report()
            
            self.metrics = {
                "system_status": "ACTIVE" if status['trading_enabled'] else "INACTIVE",
                "trading_mode": "HIGH_VOLUME",
                "active_exchanges": status['active_exchanges'],
                "open_positions": len(high_volume_trading_manager.performance_monitor.trades),
                "daily_volume": perf_metrics.get('daily_pnl', 0),
                "current_pnl": perf_metrics.get('total_pnl', 0),
                "win_rate": perf_metrics.get('win_rate', 0.0),
                "profit_factor": perf_metrics.get('profit_factor', 0.0),
                "sharpe_ratio": perf_metrics.get('sharpe_ratio', 0.0),
                "max_drawdown": perf_metrics.get('max_drawdown', 0.0),
                "risk_level": self._calculate_risk_level(),
                "rate_limit_usage": self._get_rate_limit_usage(),
                "system_health": status.get('system_health', 'UNKNOWN'),
                "cpu_usage": system_health.get('cpu_info', {}).get('cpu_percent', 0),
                "memory_usage": system_health.get('memory_info', {}).get('percent', 0),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Metrics update error: {e}")
            
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level."""
        try:
            risk_manager = high_volume_trading_manager.risk_manager
            daily_loss = risk_manager.daily_loss
            consecutive_losses = risk_manager.consecutive_losses
            
            if daily_loss > 8.0 or consecutive_losses >= 4:
                return "HIGH"
            elif daily_loss > 5.0 or consecutive_losses >= 2:
                return "MEDIUM"
            else:
                return "LOW"
        except:
            return "UNKNOWN"
            
    def _get_rate_limit_usage(self) -> Dict[str, str]:
        """Get rate limit usage for exchanges."""
        usage = {}
        try:
            for name, exchange in high_volume_trading_manager.exchanges.items():
                # Calculate usage percentage (simplified)
                current_usage = len(exchange.rate_limit_tracker.get(int(time.time() / 60), []))
                max_requests = exchange.config.get('rate_limit_per_minute', 100)
                percentage = min(100, (current_usage / max_requests) * 100)
                usage[name] = f"{percentage:.1f}%"
        except:
            usage = {"binance": "0%", "coinbase": "0%", "kraken": "0%"}
        return usage
        
    def check_alerts(self):
        """Check for alerts and warnings."""
        alerts = []
        
        # Performance alerts
        if self.metrics.get('win_rate', 0) < 0.5:
            alerts.append("⚠️ Low win rate detected")
            
        if self.metrics.get('max_drawdown', 0) > 0.1:
            alerts.append("🚨 High drawdown detected")
            
        if self.metrics.get('risk_level') == "HIGH":
            alerts.append("🚨 High risk level detected")
            
        # System health alerts
        if self.metrics.get('cpu_usage', 0) > 80:
            alerts.append("⚠️ High CPU usage")
            
        if self.metrics.get('memory_usage', 0) > 80:
            alerts.append("⚠️ High memory usage")
            
        # Trading status alerts
        if not self.metrics.get('trading_enabled', False):
            alerts.append("⚠️ Trading system inactive")
            
        self.alerts = alerts
        
    def display_dashboard(self):
        """Display the dashboard."""
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        print("📊 HIGH-VOLUME TRADING DASHBOARD")
        print("=" * 60)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Status
        print("🎯 SYSTEM STATUS")
        print("-" * 30)
        status_emoji = "🟢" if self.metrics.get('system_status') == "ACTIVE" else "🔴"
        print(f"Status: {status_emoji} {self.metrics.get('system_status', 'UNKNOWN')}")
        print(f"Mode: {self.metrics.get('trading_mode', 'UNKNOWN')}")
        print(f"Active Exchanges: {self.metrics.get('active_exchanges', 0)}")
        print(f"Open Positions: {self.metrics.get('open_positions', 0)}")
        print()
        
        # Performance Metrics
        print("📈 PERFORMANCE METRICS")
        print("-" * 30)
        print(f"Daily P&L: ${self.metrics.get('daily_volume', 0):,.2f}")
        print(f"Total P&L: ${self.metrics.get('current_pnl', 0):,.2f}")
        print(f"Win Rate: {self.metrics.get('win_rate', 0.0):.1%}")
        print(f"Profit Factor: {self.metrics.get('profit_factor', 0.0):.2f}")
        print(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Max Drawdown: {self.metrics.get('max_drawdown', 0.0):.1%}")
        print()
        
        # Risk Management
        print("🛡️ RISK MANAGEMENT")
        print("-" * 30)
        risk_level = self.metrics.get('risk_level', 'UNKNOWN')
        risk_emoji = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
        print(f"Risk Level: {risk_emoji} {risk_level}")
        print(f"System Health: {self.metrics.get('system_health', 'UNKNOWN')}")
        print()
        
        # System Resources
        print("💻 SYSTEM RESOURCES")
        print("-" * 30)
        cpu_usage = self.metrics.get('cpu_usage', 0)
        memory_usage = self.metrics.get('memory_usage', 0)
        cpu_emoji = "🔴" if cpu_usage > 80 else "🟡" if cpu_usage > 60 else "🟢"
        memory_emoji = "🔴" if memory_usage > 80 else "🟡" if memory_usage > 60 else "🟢"
        print(f"CPU Usage: {cpu_emoji} {cpu_usage:.1f}%")
        print(f"Memory Usage: {memory_emoji} {memory_usage:.1f}%")
        print()
        
        # Rate Limit Usage
        print("⚡ RATE LIMIT USAGE")
        print("-" * 30)
        rate_usage = self.metrics.get('rate_limit_usage', {})
        for exchange, usage in rate_usage.items():
            print(f"{exchange.upper()}: {usage}")
        print()
        
        # Alerts
        if self.alerts:
            print("🚨 ACTIVE ALERTS")
            print("-" * 30)
            for alert in self.alerts:
                print(f"  {alert}")
            print()
        
        # Exchange Status
        print("🏦 EXCHANGE STATUS")
        print("-" * 30)
        exchanges = high_volume_trading_manager.exchanges
        for name, exchange in exchanges.items():
            status = "🟢 CONNECTED" if exchange.exchange else "🔴 DISCONNECTED"
            print(f"{name.upper()}: {status}")
        print()
        
        print("=" * 60)
        print("Press Ctrl+C to stop monitoring")
        
    def run_continuous_dashboard(self):
        """Run continuous dashboard updates."""
        try:
            self.start_monitoring()
            
            while True:
                self.display_dashboard()
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            self.stop_monitoring()
            print("✅ Dashboard stopped")

def main():
    """Main dashboard function."""
    try:
        dashboard = HighVolumeDashboard()
        dashboard.run_continuous_dashboard()
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 