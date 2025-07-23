#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 ADVANCED SECURITY MANAGER - ULTRA-REALISTIC DUMMY PACKET SYSTEM
==================================================================

Developed by Maxamillion M.A.A. DeLeon screen/pen name TheSchwa1337 ("The Schwa") & Nexus AI
– Recursive Systems Architects | Authors of Ω-B-Γ Logic & Alpha Encryption Protocol

This module provides CLI commands and GUI interface for the ultra-realistic dummy packet
security system with default auto-on protection for logical security.
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union

from .secure_trade_handler import SecureTradeHandler, secure_trade_payload
from .secure_trade_integration import SecureTradeIntegration, integrate_secure_trade_handler

logger = logging.getLogger(__name__)

class AdvancedSecurityManager:
    """
    🔐 Advanced Security Manager
    
    Provides CLI commands and GUI interface for ultra-realistic dummy packet security
    with default auto-on protection for logical security.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Advanced Security Manager."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize secure trade components
        self.secure_handler = SecureTradeHandler(self.config.get('secure_handler_config'))
        self.secure_integration = SecureTradeIntegration(self.config.get('integration_config'))
        
        # Security status
        self.security_enabled = self.config.get('default_enabled', True)
        self.auto_protection = self.config.get('auto_protection', True)
        self.logical_protection = self.config.get('logical_protection', True)
        
        # Statistics
        self.total_trades_protected = 0
        self.security_events = []
        
        # GUI components (will be initialized when needed)
        self.gui_components = {}
        
        self.logger.info("🔐 Advanced Security Manager initialized")
        self.logger.info(f"   Default Enabled: {self.security_enabled}")
        self.logger.info(f"   Auto Protection: {self.auto_protection}")
        self.logger.info(f"   Logical Protection: {self.logical_protection}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration with auto-on protection."""
        return {
            'default_enabled': True,  # Auto-on by default
            'auto_protection': True,  # Automatic protection
            'logical_protection': True,  # Logical security protection
            'secure_handler_config': {
                'dummy_packet_count': 2,
                'enable_dummy_injection': True,
                'enable_hash_id_routing': True,
                'security_logging': True,
                'ephemeral_weight': 0.25,
                'chacha20_weight': 0.25,
                'nonce_weight': 0.20,
                'dummy_weight': 0.15,
                'hash_id_weight': 0.15
            },
            'integration_config': {
                'enable_all_integrations': True,
                'force_secure_trades': True,
                'log_integration_events': True,
                'security_threshold': 80.0
            },
            'gui_config': {
                'theme': 'dark',
                'auto_refresh': True,
                'show_statistics': True,
                'show_security_events': True
            }
        }
    
    def enable_security(self) -> bool:
        """Enable advanced security protection."""
        try:
            self.security_enabled = True
            self.logger.info("🔐 Advanced Security Protection ENABLED")
            self._log_security_event('security_enabled', {})
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to enable security: {e}")
            return False
    
    def disable_security(self) -> bool:
        """Disable advanced security protection."""
        try:
            self.security_enabled = False
            self.logger.info("⚠️ Advanced Security Protection DISABLED")
            self._log_security_event('security_disabled', {})
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to disable security: {e}")
            return False
    
    def toggle_auto_protection(self) -> bool:
        """Toggle auto protection mode."""
        try:
            self.auto_protection = not self.auto_protection
            status = "ENABLED" if self.auto_protection else "DISABLED"
            self.logger.info(f"🔄 Auto Protection {status}")
            self._log_security_event('auto_protection_toggled', {'enabled': self.auto_protection})
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to toggle auto protection: {e}")
            return False
    
    def protect_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Protect a trade with ultra-realistic dummy packets."""
        try:
            if not self.security_enabled:
                self.logger.warning("⚠️ Security disabled, returning unprotected trade")
                return {'success': True, 'protected': False, 'trade_data': trade_data}
            
            # Secure the trade
            secure_result = self.secure_handler.secure_trade_payload(trade_data)
            
            if secure_result.success:
                self.total_trades_protected += 1
                self._log_security_event('trade_protected', {
                    'symbol': trade_data.get('symbol', 'unknown'),
                    'security_score': secure_result.security_score,
                    'dummy_count': len(secure_result.dummy_packets)
                })
                
                return {
                    'success': True,
                    'protected': True,
                    'secure_result': secure_result,
                    'statistics': self.get_statistics()
                }
            else:
                self.logger.error("❌ Failed to protect trade")
                return {'success': False, 'protected': False, 'error': 'Protection failed'}
                
        except Exception as e:
            self.logger.error(f"❌ Error protecting trade: {e}")
            return {'success': False, 'protected': False, 'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        try:
            return {
                'security_enabled': self.security_enabled,
                'auto_protection': self.auto_protection,
                'logical_protection': self.logical_protection,
                'total_trades_protected': self.total_trades_protected,
                'security_events_count': len(self.security_events),
                'secure_handler_status': self.secure_handler.get_security_status(),
                'integration_status': self.secure_integration.get_integration_status()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def get_security_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security events."""
        try:
            return self.security_events[-limit:] if self.security_events else []
        except Exception as e:
            self.logger.error(f"❌ Failed to get security events: {e}")
            return []
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        try:
            event = {
                'timestamp': time.time(),
                'event_type': event_type,
                'details': details
            }
            
            self.security_events.append(event)
            
            # Keep only last 1000 events
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log security event: {e}")
    
    def export_config(self, filename: str) -> bool:
        """Export security configuration to file."""
        try:
            config_data = {
                'timestamp': time.time(),
                'config': self.config,
                'statistics': self.get_statistics()
            }
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"✅ Configuration exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export configuration: {e}")
            return False
    
    def import_config(self, filename: str) -> bool:
        """Import security configuration from file."""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            if 'config' in config_data:
                self.config.update(config_data['config'])
                self.logger.info(f"✅ Configuration imported from {filename}")
                return True
            else:
                self.logger.error("❌ Invalid configuration file format")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to import configuration: {e}")
            return False

# Global instance for easy access
advanced_security_manager = AdvancedSecurityManager()

def cli_main():
    """Main CLI interface for Advanced Security Manager."""
    parser = argparse.ArgumentParser(
        description="🔐 Advanced Security Manager - Ultra-Realistic Dummy Packet System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.advanced_security_manager status
  python -m core.advanced_security_manager enable
  python -m core.advanced_security_manager disable
  python -m core.advanced_security_manager protect --symbol BTC/USDC --side buy --amount 0.1
  python -m core.advanced_security_manager statistics
  python -m core.advanced_security_manager events
  python -m core.advanced_security_manager gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show security status')
    
    # Enable command
    enable_parser = subparsers.add_parser('enable', help='Enable security protection')
    
    # Disable command
    disable_parser = subparsers.add_parser('disable', help='Disable security protection')
    
    # Toggle command
    toggle_parser = subparsers.add_parser('toggle', help='Toggle auto protection')
    
    # Protect command
    protect_parser = subparsers.add_parser('protect', help='Protect a trade')
    protect_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTC/USDC)')
    protect_parser.add_argument('--side', required=True, choices=['buy', 'sell'], help='Trade side')
    protect_parser.add_argument('--amount', required=True, type=float, help='Trade amount')
    protect_parser.add_argument('--price', type=float, help='Trade price')
    protect_parser.add_argument('--exchange', default='coinbase', help='Exchange name')
    protect_parser.add_argument('--strategy', default='ferris_ride_001', help='Strategy ID')
    
    # Statistics command
    stats_parser = subparsers.add_parser('statistics', help='Show security statistics')
    
    # Events command
    events_parser = subparsers.add_parser('events', help='Show security events')
    events_parser.add_argument('--limit', type=int, default=10, help='Number of events to show')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI interface')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('--file', default='security_config.json', help='Output file')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('--file', required=True, help='Input file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'status':
            show_status()
        elif args.command == 'enable':
            enable_security()
        elif args.command == 'disable':
            disable_security()
        elif args.command == 'toggle':
            toggle_auto_protection()
        elif args.command == 'protect':
            protect_trade_cli(args)
        elif args.command == 'statistics':
            show_statistics()
        elif args.command == 'events':
            show_events(args.limit)
        elif args.command == 'gui':
            launch_gui()
        elif args.command == 'export':
            export_config(args.file)
        elif args.command == 'import':
            import_config(args.file)
            
    except Exception as e:
        logger.error(f"❌ CLI command failed: {e}")
        sys.exit(1)

def show_status():
    """Show security status."""
    stats = advanced_security_manager.get_statistics()
    
    print("🔐 ADVANCED SECURITY MANAGER STATUS")
    print("=" * 50)
    print(f"Security Enabled: {'✅ YES' if stats['security_enabled'] else '❌ NO'}")
    print(f"Auto Protection: {'✅ YES' if stats['auto_protection'] else '❌ NO'}")
    print(f"Logical Protection: {'✅ YES' if stats['logical_protection'] else '❌ NO'}")
    print(f"Total Trades Protected: {stats['total_trades_protected']}")
    print(f"Security Events: {stats['security_events_count']}")
    
    # Show secure handler status
    handler_status = stats['secure_handler_status']
    print(f"\n🔐 Secure Handler Status:")
    print(f"   Key Pool Size: {handler_status['key_pool_size']}")
    print(f"   Cryptography Available: {'✅ YES' if handler_status['cryptography_available'] else '❌ NO'}")
    
    # Show integration status
    integration_status = stats['integration_status']
    print(f"\n🔗 Integration Status:")
    for component, active in integration_status['integrations_active'].items():
        status = "✅ ACTIVE" if active else "❌ INACTIVE"
        print(f"   {component}: {status}")

def enable_security():
    """Enable security protection."""
    if advanced_security_manager.enable_security():
        print("✅ Advanced Security Protection ENABLED")
    else:
        print("❌ Failed to enable security protection")

def disable_security():
    """Disable security protection."""
    if advanced_security_manager.disable_security():
        print("⚠️ Advanced Security Protection DISABLED")
    else:
        print("❌ Failed to disable security protection")

def toggle_auto_protection():
    """Toggle auto protection."""
    if advanced_security_manager.toggle_auto_protection():
        status = "ENABLED" if advanced_security_manager.auto_protection else "DISABLED"
        print(f"🔄 Auto Protection {status}")
    else:
        print("❌ Failed to toggle auto protection")

def protect_trade_cli(args):
    """Protect a trade via CLI."""
    trade_data = {
        'symbol': args.symbol,
        'side': args.side,
        'amount': args.amount,
        'price': args.price or 50000.0,
        'exchange': args.exchange,
        'strategy_id': args.strategy,
        'user_id': 'schwa_1337',
        'timestamp': time.time()
    }
    
    print(f"🔐 Protecting Trade: {trade_data['symbol']} {trade_data['side']} {trade_data['amount']}")
    
    result = advanced_security_manager.protect_trade(trade_data)
    
    if result['success'] and result['protected']:
        secure_result = result['secure_result']
        print(f"✅ Trade Protected Successfully!")
        print(f"   Security Score: {secure_result.security_score:.2f}/100")
        print(f"   Processing Time: {secure_result.processing_time:.4f}s")
        print(f"   Dummy Packets: {len(secure_result.dummy_packets)}")
        print(f"   Key ID: {secure_result.key_id}")
        print(f"   Hash ID: {secure_result.metadata.get('hash_id', 'N/A')}")
        
        # Show dummy packet details
        for i, dummy in enumerate(secure_result.dummy_packets):
            print(f"   Dummy {i+1}: {dummy['key_id']} | {dummy['pseudo_meta_tag']}")
    else:
        print(f"❌ Trade Protection Failed: {result.get('error', 'Unknown error')}")

def show_statistics():
    """Show security statistics."""
    stats = advanced_security_manager.get_statistics()
    
    print("📊 ADVANCED SECURITY STATISTICS")
    print("=" * 50)
    
    # Basic stats
    print(f"Total Trades Protected: {stats['total_trades_protected']}")
    print(f"Security Events: {stats['security_events_count']}")
    
    # Integration stats
    integration_stats = stats['integration_status']['statistics']
    print(f"\n🔗 Integration Statistics:")
    print(f"   Total Trades Secured: {integration_stats['total_trades_secured']}")
    print(f"   Success Rate: {integration_stats['success_rate']:.2%}")
    print(f"   Average Security Score: {integration_stats['average_security_score']:.2f}")
    print(f"   Average Processing Time: {integration_stats['average_processing_time']:.4f}s")
    
    # Layer weights
    handler_status = stats['secure_handler_status']
    layer_weights = handler_status['layer_weights']
    print(f"\n⚖️ Security Layer Weights:")
    for layer, weight in layer_weights.items():
        print(f"   {layer}: {weight:.2f}")

def show_events(limit: int):
    """Show security events."""
    events = advanced_security_manager.get_security_events(limit)
    
    print(f"📋 RECENT SECURITY EVENTS (Last {len(events)})")
    print("=" * 50)
    
    if not events:
        print("No security events found.")
        return
    
    for event in reversed(events):  # Show newest first
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['timestamp']))
        event_type = event['event_type']
        details = event['details']
        
        print(f"\n⏰ {timestamp}")
        print(f"   Event: {event_type}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")

def launch_gui():
    """Launch GUI interface."""
    try:
        from .advanced_security_gui import AdvancedSecurityGUI
        gui = AdvancedSecurityGUI(advanced_security_manager)
        gui.run()
    except ImportError as e:
        print(f"❌ GUI not available: {e}")
        print("Install required GUI dependencies to use the interface.")

def export_config(filename: str):
    """Export configuration."""
    if advanced_security_manager.export_config(filename):
        print(f"✅ Configuration exported to {filename}")
    else:
        print(f"❌ Failed to export configuration to {filename}")

def import_config(filename: str):
    """Import configuration."""
    if advanced_security_manager.import_config(filename):
        print(f"✅ Configuration imported from {filename}")
    else:
        print(f"❌ Failed to import configuration from {filename}")

if __name__ == "__main__":
    cli_main() 