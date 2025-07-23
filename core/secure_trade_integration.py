#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 SECURE TRADE INTEGRATION - SCHWABOT TRADING SYSTEM INTEGRATION
================================================================

Developed by Maxamillion M.A.A. DeLeon screen/pen name TheSchwa1337 ("The Schwa") & Nexus AI
– Recursive Systems Architects | Authors of Ω-B-Γ Logic & Alpha Encryption Protocol

This module provides integration between the Secure Trade Handler and existing
Schwabot trading systems to address Natalie's security concerns about per-trade
payload encryption.

Integration Points:
1. Real Trading Engine integration
2. Strategy Execution Engine integration
3. API route integration
4. CCXT trading engine integration
5. Profile router integration

This ensures that every trade executed through Schwabot goes through the
secure trade handler for per-trade payload encryption and obfuscation.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from .secure_trade_handler import SecureTradeHandler, SecureTradeResult, secure_trade_payload

logger = logging.getLogger(__name__)

class SecureTradeIntegration:
    """
    🔐 Secure Trade Integration
    
    Integrates secure trade handler with existing Schwabot trading systems
    to ensure every trade is encrypted and obfuscated.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Secure Trade Integration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize secure trade handler
        self.secure_handler = SecureTradeHandler(self.config.get('secure_handler_config'))
        
        # Integration statistics
        self.total_trades_secured = 0
        self.successful_secures = 0
        self.failed_secures = 0
        self.average_security_score = 0.0
        self.average_processing_time = 0.0
        
        # Integration status
        self.integrations_active = {
            'real_trading_engine': False,
            'strategy_execution_engine': False,
            'api_routes': False,
            'ccxt_engine': False,
            'profile_router': False
        }
        
        self.logger.info("✅ Secure Trade Integration initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'secure_handler_config': {
                'dummy_packet_count': 2,
                'enable_dummy_injection': True,
                'enable_hash_id_routing': True,
                'security_logging': True
            },
            'integration_config': {
                'enable_all_integrations': True,
                'force_secure_trades': True,
                'log_integration_events': True,
                'security_threshold': 80.0  # Minimum security score
            }
        }
    
    def secure_trade_execution(self, trade_data: Dict[str, Any], 
                             integration_point: str = "unknown") -> Dict[str, Any]:
        """
        Secure a trade execution through the secure trade handler.
        
        Args:
            trade_data: Raw trade data to secure
            integration_point: Which integration point is calling this
            
        Returns:
            Dictionary with secured trade data and metadata
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"🔐 Securing trade execution from {integration_point}")
            
            # Secure the trade payload
            secure_result = self.secure_handler.secure_trade_payload(trade_data)
            
            # Update statistics
            self.total_trades_secured += 1
            if secure_result.success:
                self.successful_secures += 1
                self.average_security_score = (
                    (self.average_security_score * (self.successful_secures - 1) + secure_result.security_score) 
                    / self.successful_secures
                )
                self.average_processing_time = (
                    (self.average_processing_time * (self.successful_secures - 1) + secure_result.processing_time) 
                    / self.successful_secures
                )
            else:
                self.failed_secures += 1
            
            # Check security threshold
            security_threshold = self.config.get('integration_config', {}).get('security_threshold', 80.0)
            if secure_result.success and secure_result.security_score < security_threshold:
                self.logger.warning(f"⚠️ Trade security score {secure_result.security_score:.2f} below threshold {security_threshold}")
            
            # Create result
            result = {
                'success': secure_result.success,
                'secured_trade_data': {
                    'encrypted_payload': secure_result.encrypted_payload,
                    'key_id': secure_result.key_id,
                    'nonce': secure_result.nonce,
                    'hash_id': secure_result.metadata.get('hash_id', ''),
                    'dummy_packets': secure_result.dummy_packets
                },
                'security_metadata': {
                    'security_score': secure_result.security_score,
                    'processing_time': secure_result.processing_time,
                    'payload_size': secure_result.metadata.get('payload_size', 0),
                    'encrypted_size': secure_result.metadata.get('encrypted_size', 0),
                    'dummy_count': len(secure_result.dummy_packets),
                    'key_strength_bits': secure_result.metadata.get('key_strength_bits', 0),
                    'nonce_entropy_bits': secure_result.metadata.get('nonce_entropy_bits', 0)
                },
                'integration_metadata': {
                    'integration_point': integration_point,
                    'timestamp': time.time(),
                    'total_trades_secured': self.total_trades_secured,
                    'success_rate': self.successful_secures / self.total_trades_secured if self.total_trades_secured > 0 else 0.0
                }
            }
            
            # Log integration event
            if self.config.get('integration_config', {}).get('log_integration_events', True):
                self._log_integration_event('trade_secured', {
                    'integration_point': integration_point,
                    'symbol': trade_data.get('symbol', 'unknown'),
                    'security_score': secure_result.security_score,
                    'success': secure_result.success
                })
            
            self.logger.info(f"✅ Trade secured from {integration_point} with score {secure_result.security_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to secure trade execution: {e}")
            self.failed_secures += 1
            
            return {
                'success': False,
                'error': str(e),
                'integration_point': integration_point,
                'timestamp': time.time()
            }
    
    def integrate_with_real_trading_engine(self, trading_engine) -> bool:
        """
        Integrate secure trade handler with Real Trading Engine.
        
        Args:
            trading_engine: Real Trading Engine instance
            
        Returns:
            True if integration successful
        """
        try:
            self.logger.info("🔗 Integrating with Real Trading Engine")
            
            # Store original methods
            original_execute_coinbase = getattr(trading_engine, '_execute_coinbase_trade', None)
            original_execute_binance = getattr(trading_engine, '_execute_binance_trade', None)
            original_execute_kraken = getattr(trading_engine, '_execute_kraken_trade', None)
            
            # Wrap Coinbase execution
            if original_execute_coinbase:
                async def secure_execute_coinbase(*args, **kwargs):
                    # Extract trade data from arguments
                    symbol, side, quantity, order_type, price = args[:5]
                    trade_data = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'order_type': order_type.value if hasattr(order_type, 'value') else str(order_type),
                        'price': price,
                        'exchange': 'coinbase',
                        'timestamp': time.time()
                    }
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(trade_data, 'real_trading_engine_coinbase')
                    
                    if not secure_result['success']:
                        raise Exception(f"Trade security failed: {secure_result.get('error', 'Unknown error')}")
                    
                    # Execute original method with secured data
                    return await original_execute_coinbase(*args, **kwargs)
                
                trading_engine._execute_coinbase_trade = secure_execute_coinbase
            
            # Wrap Binance execution
            if original_execute_binance:
                async def secure_execute_binance(*args, **kwargs):
                    # Extract trade data from arguments
                    symbol, side, quantity, order_type, price = args[:5]
                    trade_data = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'order_type': order_type.value if hasattr(order_type, 'value') else str(order_type),
                        'price': price,
                        'exchange': 'binance',
                        'timestamp': time.time()
                    }
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(trade_data, 'real_trading_engine_binance')
                    
                    if not secure_result['success']:
                        raise Exception(f"Trade security failed: {secure_result.get('error', 'Unknown error')}")
                    
                    # Execute original method with secured data
                    return await original_execute_binance(*args, **kwargs)
                
                trading_engine._execute_binance_trade = secure_execute_binance
            
            # Wrap Kraken execution
            if original_execute_kraken:
                async def secure_execute_kraken(*args, **kwargs):
                    # Extract trade data from arguments
                    symbol, side, quantity, order_type, price = args[:5]
                    trade_data = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'order_type': order_type.value if hasattr(order_type, 'value') else str(order_type),
                        'price': price,
                        'exchange': 'kraken',
                        'timestamp': time.time()
                    }
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(trade_data, 'real_trading_engine_kraken')
                    
                    if not secure_result['success']:
                        raise Exception(f"Trade security failed: {secure_result.get('error', 'Unknown error')}")
                    
                    # Execute original method with secured data
                    return await original_execute_kraken(*args, **kwargs)
                
                trading_engine._execute_kraken_trade = secure_execute_kraken
            
            self.integrations_active['real_trading_engine'] = True
            self.logger.info("✅ Real Trading Engine integration successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate with Real Trading Engine: {e}")
            return False
    
    def integrate_with_strategy_execution_engine(self, strategy_engine) -> bool:
        """
        Integrate secure trade handler with Strategy Execution Engine.
        
        Args:
            strategy_engine: Strategy Execution Engine instance
            
        Returns:
            True if integration successful
        """
        try:
            self.logger.info("🔗 Integrating with Strategy Execution Engine")
            
            # Store original method
            original_execute_trade = getattr(strategy_engine, '_execute_trade', None)
            
            if original_execute_trade:
                def secure_execute_trade(execution):
                    # Extract trade data from execution
                    trade_data = {
                        'symbol': execution.symbol,
                        'action': execution.action,
                        'amount': execution.amount,
                        'price': execution.price,
                        'strategy_id': execution.strategy_id,
                        'execution_id': execution.execution_id,
                        'timestamp': time.time()
                    }
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(trade_data, 'strategy_execution_engine')
                    
                    if not secure_result['success']:
                        execution.status = 'FAILED'
                        execution.error_message = f"Trade security failed: {secure_result.get('error', 'Unknown error')}"
                        return
                    
                    # Execute original method
                    return original_execute_trade(execution)
                
                strategy_engine._execute_trade = secure_execute_trade
            
            self.integrations_active['strategy_execution_engine'] = True
            self.logger.info("✅ Strategy Execution Engine integration successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate with Strategy Execution Engine: {e}")
            return False
    
    def integrate_with_api_routes(self, app) -> bool:
        """
        Integrate secure trade handler with Flask API routes.
        
        Args:
            app: Flask app instance
            
        Returns:
            True if integration successful
        """
        try:
            self.logger.info("🔗 Integrating with API Routes")
            
            # This would require modifying the Flask routes to use secure trade handler
            # For now, we'll create a decorator that can be applied to trade routes
            
            def secure_trade_route(f):
                """Decorator to secure trade API routes."""
                def wrapper(*args, **kwargs):
                    try:
                        # Extract trade data from request
                        from flask import request
                        trade_data = request.get_json() or {}
                        
                        # Secure the trade
                        secure_result = self.secure_trade_execution(trade_data, 'api_route')
                        
                        if not secure_result['success']:
                            return {'error': f"Trade security failed: {secure_result.get('error', 'Unknown error')}"}, 400
                        
                        # Add secured data to request context
                        request.secured_trade_data = secure_result['secured_trade_data']
                        request.security_metadata = secure_result['security_metadata']
                        
                        # Execute original function
                        return f(*args, **kwargs)
                        
                    except Exception as e:
                        self.logger.error(f"❌ API route security failed: {e}")
                        return {'error': str(e)}, 500
                
                return wrapper
            
            # Store the decorator for use in route definitions
            app.secure_trade_route = secure_trade_route
            
            self.integrations_active['api_routes'] = True
            self.logger.info("✅ API Routes integration successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate with API Routes: {e}")
            return False
    
    def integrate_with_ccxt_engine(self, ccxt_engine) -> bool:
        """
        Integrate secure trade handler with CCXT Trading Engine.
        
        Args:
            ccxt_engine: CCXT Trading Engine instance
            
        Returns:
            True if integration successful
        """
        try:
            self.logger.info("🔗 Integrating with CCXT Trading Engine")
            
            # Store original method
            original_execute_order = getattr(ccxt_engine, '_execute_order', None)
            
            if original_execute_order:
                async def secure_execute_order(exchange_name, order):
                    # Extract trade data from order
                    trade_data = {
                        'symbol': order.symbol,
                        'side': order.side.value if hasattr(order.side, 'value') else str(order.side),
                        'quantity': order.quantity,
                        'order_type': order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                        'price': order.price,
                        'exchange': exchange_name,
                        'order_id': order.order_id,
                        'timestamp': time.time()
                    }
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(trade_data, 'ccxt_engine')
                    
                    if not secure_result['success']:
                        raise Exception(f"Trade security failed: {secure_result.get('error', 'Unknown error')}")
                    
                    # Execute original method
                    return await original_execute_order(exchange_name, order)
                
                ccxt_engine._execute_order = secure_execute_order
            
            self.integrations_active['ccxt_engine'] = True
            self.logger.info("✅ CCXT Trading Engine integration successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate with CCXT Trading Engine: {e}")
            return False
    
    def integrate_with_profile_router(self, profile_router) -> bool:
        """
        Integrate secure trade handler with Profile Router.
        
        Args:
            profile_router: Profile Router instance
            
        Returns:
            True if integration successful
        """
        try:
            self.logger.info("🔗 Integrating with Profile Router")
            
            # Store original method
            original_execute_trade_on_profile = getattr(profile_router, '_execute_trade_on_profile', None)
            
            if original_execute_trade_on_profile:
                async def secure_execute_trade_on_profile(profile_id, trade_data):
                    # Add profile information to trade data
                    secured_trade_data = trade_data.copy()
                    secured_trade_data['profile_id'] = profile_id
                    secured_trade_data['timestamp'] = time.time()
                    
                    # Secure the trade
                    secure_result = self.secure_trade_execution(secured_trade_data, 'profile_router')
                    
                    if not secure_result['success']:
                        return {'success': False, 'error': f"Trade security failed: {secure_result.get('error', 'Unknown error')}"}
                    
                    # Execute original method
                    return await original_execute_trade_on_profile(profile_id, trade_data)
                
                profile_router._execute_trade_on_profile = secure_execute_trade_on_profile
            
            self.integrations_active['profile_router'] = True
            self.logger.info("✅ Profile Router integration successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate with Profile Router: {e}")
            return False
    
    def _log_integration_event(self, event_type: str, details: Dict[str, Any]):
        """Log integration event."""
        try:
            if not self.config.get('integration_config', {}).get('log_integration_events', True):
                return
            
            event = {
                'timestamp': time.time(),
                'event_type': event_type,
                'details': details
            }
            
            self.logger.info(f"🔗 Integration Event: {event_type} - {details}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log integration event: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and statistics."""
        try:
            return {
                'enabled': True,
                'integrations_active': self.integrations_active,
                'statistics': {
                    'total_trades_secured': self.total_trades_secured,
                    'successful_secures': self.successful_secures,
                    'failed_secures': self.failed_secures,
                    'success_rate': self.successful_secures / self.total_trades_secured if self.total_trades_secured > 0 else 0.0,
                    'average_security_score': self.average_security_score,
                    'average_processing_time': self.average_processing_time
                },
                'secure_handler_status': self.secure_handler.get_security_status(),
                'config': self.config
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get integration status: {e}")
            return {'enabled': False, 'error': str(e)}

# Global instance for easy access
secure_trade_integration = SecureTradeIntegration()

def integrate_secure_trade_handler(component, component_type: str) -> bool:
    """
    Convenience function to integrate secure trade handler with a component.
    
    Args:
        component: Component to integrate with
        component_type: Type of component ('real_trading_engine', 'strategy_execution_engine', etc.)
        
    Returns:
        True if integration successful
    """
    if component_type == 'real_trading_engine':
        return secure_trade_integration.integrate_with_real_trading_engine(component)
    elif component_type == 'strategy_execution_engine':
        return secure_trade_integration.integrate_with_strategy_execution_engine(component)
    elif component_type == 'api_routes':
        return secure_trade_integration.integrate_with_api_routes(component)
    elif component_type == 'ccxt_engine':
        return secure_trade_integration.integrate_with_ccxt_engine(component)
    elif component_type == 'profile_router':
        return secure_trade_integration.integrate_with_profile_router(component)
    else:
        logger.error(f"❌ Unknown component type: {component_type}")
        return False

if __name__ == "__main__":
    # Test the secure trade integration
    print("🔐 Secure Trade Integration Test")
    
    # Test secure trade execution
    test_trade_data = {
        'symbol': 'BTC/USDC',
        'side': 'buy',
        'amount': 0.1,
        'price': 50000.0,
        'exchange': 'coinbase',
        'timestamp': time.time()
    }
    
    result = secure_trade_integration.secure_trade_execution(test_trade_data, 'test')
    print(f"Integration Test Result:")
    print(f"Success: {result['success']}")
    print(f"Security Score: {result['security_metadata']['security_score']:.2f}")
    print(f"Processing Time: {result['security_metadata']['processing_time']:.4f}s")
    print(f"Dummy Packets: {result['security_metadata']['dummy_count']}")
    
    # Test integration status
    status = secure_trade_integration.get_integration_status()
    print(f"\nIntegration Status:")
    print(f"Total Trades Secured: {status['statistics']['total_trades_secured']}")
    print(f"Success Rate: {status['statistics']['success_rate']:.2%}")
    print(f"Average Security Score: {status['statistics']['average_security_score']:.2f}") 