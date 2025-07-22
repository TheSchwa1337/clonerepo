#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 HIGH-VOLUME TRADING API ROUTES
=================================

REST API endpoints for high-volume trading control and monitoring.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import asyncio
import logging

# Import the high-volume trading manager
try:
    from core.high_volume_trading_manager import high_volume_trading_manager
except ImportError:
    high_volume_trading_manager = None

high_volume_bp = Blueprint('high_volume', __name__, url_prefix='/api/high-volume')

@high_volume_bp.route('/status', methods=['GET'])
def get_high_volume_status():
    """Get high-volume trading system status."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    status = high_volume_trading_manager.get_system_status()
    
    return jsonify({
        "status": "ACTIVE" if status['trading_enabled'] else "INACTIVE",
        "mode": "HIGH_VOLUME",
        "exchanges": ["binance", "coinbase", "kraken"],
        "active_trades": status.get('active_trades', 0),
        "daily_volume": status.get('daily_volume', 0),
        "performance": status.get('performance_metrics', {}),
        "system_health": status.get('system_health', 'UNKNOWN'),
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/activate', methods=['POST'])
def activate_high_volume_trading():
    """Activate high-volume trading mode."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    try:
        # Run activation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(high_volume_trading_manager.activate_high_volume_mode())
        loop.close()
        
        return jsonify({
            "status": "ACTIVATED", 
            "message": "High-volume trading enabled",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Activation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@high_volume_bp.route('/emergency-stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all trading."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    try:
        # Run emergency stop asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(high_volume_trading_manager.emergency_stop())
        loop.close()
        
        return jsonify({
            "status": "STOPPED", 
            "message": "Emergency stop activated",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Emergency stop failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@high_volume_bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get real-time performance metrics."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    status = high_volume_trading_manager.get_system_status()
    metrics = status.get('performance_metrics', {})
    
    return jsonify({
        "total_trades": metrics.get('total_trades', 0),
        "win_rate": metrics.get('win_rate', 0.0),
        "profit_factor": metrics.get('profit_factor', 0.0),
        "sharpe_ratio": metrics.get('sharpe_ratio', 0.0),
        "max_drawdown": metrics.get('max_drawdown', 0.0),
        "daily_pnl": metrics.get('daily_pnl', 0.0),
        "total_pnl": metrics.get('total_pnl', 0.0),
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/exchanges', methods=['GET'])
def get_exchange_status():
    """Get exchange connection status."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    exchanges = {}
    for name, exchange in high_volume_trading_manager.exchanges.items():
        exchanges[name] = {
            "status": "CONNECTED" if exchange.exchange else "DISCONNECTED",
            "rate_limit_usage": "0%",  # Placeholder
            "last_update": datetime.now().isoformat()
        }
    
    return jsonify({
        "exchanges": exchanges,
        "total_connected": len([e for e in exchanges.values() if e['status'] == 'CONNECTED']),
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a high-volume trade."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'amount', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Execute trade asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            high_volume_trading_manager.execute_high_volume_trade(data)
        )
        loop.close()
        
        if result:
            return jsonify({
                "status": "SUCCESS",
                "trade_result": result,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "FAILED",
                "message": "Trade execution failed",
                "timestamp": datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Trade execution error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@high_volume_bp.route('/arbitrage/scan', methods=['GET'])
def scan_arbitrage():
    """Scan for arbitrage opportunities."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    try:
        # Scan arbitrage asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(high_volume_trading_manager.find_arbitrage_opportunities())
        loop.close()
        
        return jsonify({
            "status": "SUCCESS",
            "message": "Arbitrage scan completed",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Arbitrage scan failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@high_volume_bp.route('/config', methods=['GET'])
def get_configuration():
    """Get current configuration."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    config = high_volume_trading_manager.config
    
    # Return safe configuration (without sensitive data)
    safe_config = {
        "system_mode": config.get('system_mode', 'unknown'),
        "high_volume_trading": {
            "enabled": config.get('high_volume_trading', {}).get('enabled', False),
            "mode": config.get('high_volume_trading', {}).get('mode', 'unknown'),
            "max_concurrent_trades": config.get('high_volume_trading', {}).get('max_concurrent_trades', 0)
        },
        "risk_management": config.get('risk_management', {}),
        "performance": config.get('performance', {}),
        "exchanges": {
            name: {
                "enabled": exchange_config.get('enabled', False),
                "priority": exchange_config.get('priority', 0)
            }
            for name, exchange_config in config.get('exchanges', {}).items()
        }
    }
    
    return jsonify({
        "configuration": safe_config,
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/health', methods=['GET'])
def get_system_health():
    """Get system health status."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    status = high_volume_trading_manager.get_system_status()
    
    return jsonify({
        "overall_health": status.get('system_health', 'UNKNOWN'),
        "trading_enabled": status.get('trading_enabled', False),
        "active_exchanges": status.get('active_exchanges', 0),
        "performance_metrics": status.get('performance_metrics', {}),
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/trades', methods=['GET'])
def get_trade_history():
    """Get trade history."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    trades = high_volume_trading_manager.performance_monitor.trades
    
    # Limit to last 100 trades
    recent_trades = trades[-100:] if len(trades) > 100 else trades
    
    return jsonify({
        "trades": recent_trades,
        "total_trades": len(trades),
        "timestamp": datetime.now().isoformat()
    })

@high_volume_bp.route('/risk-status', methods=['GET'])
def get_risk_status():
    """Get current risk management status."""
    if not high_volume_trading_manager:
        return jsonify({"error": "High-volume trading manager not available"}), 500
        
    risk_manager = high_volume_trading_manager.risk_manager
    
    return jsonify({
        "daily_loss": risk_manager.daily_loss,
        "consecutive_losses": risk_manager.consecutive_losses,
        "max_drawdown": risk_manager.max_drawdown,
        "positions": risk_manager.positions,
        "timestamp": datetime.now().isoformat()
    }) 