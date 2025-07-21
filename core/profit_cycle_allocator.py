#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 PROFIT CYCLE ALLOCATOR - STRATEGY TRIGGER SYSTEM
==================================================

Profit cycle allocator that uses Ferris tick logic to allocate profit zones
and trigger appropriate strategies based on Kaprekar volatility analysis.

Features:
- Strategy allocation based on volatility signals
- Profit zone mapping and optimization
- Integration with existing trading strategies
- Dynamic strategy triggering system
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from .ferris_tick_logic import process_tick, process_tick_with_metadata

logger = logging.getLogger(__name__)

def allocate_profit_zone(price_tick: float) -> Dict[str, Any]:
    """
    Allocate profit zone based on price tick analysis.
    
    Args:
        price_tick: Float price value
        
    Returns:
        Dictionary with profit zone allocation and strategy triggers
    """
    try:
        route = process_tick(price_tick)
        
        allocation_result = {
            'timestamp': time.time(),
            'price_tick': price_tick,
            'routing_signal': route,
            'strategy_triggers': [],
            'profit_zone': None,
            'allocation_confidence': 0.0
        }
        
        if route == "vol_stable_basket":
            allocation_result['strategy_triggers'] = ["BTC_MICROHOLD_REBUY"]
            allocation_result['profit_zone'] = "stable_accumulation"
            allocation_result['allocation_confidence'] = 0.8
            
        elif route == "midrange_vol_logic":
            allocation_result['strategy_triggers'] = ["USDC_RSI_REBALANCE"]
            allocation_result['profit_zone'] = "balanced_trading"
            allocation_result['allocation_confidence'] = 0.7
            
        elif route == "escape_vol_guard":
            allocation_result['strategy_triggers'] = ["XRP_LIQUIDITY_VACUUM"]
            allocation_result['profit_zone'] = "aggressive_capture"
            allocation_result['allocation_confidence'] = 0.6
            
        elif route == "ghost_shell_evasion":
            allocation_result['strategy_triggers'] = ["ZBE_RECOVERY_PATH"]
            allocation_result['profit_zone'] = "defensive_recovery"
            allocation_result['allocation_confidence'] = 0.9
            
        else:
            allocation_result['strategy_triggers'] = ["FALLBACK_STRATEGY"]
            allocation_result['profit_zone'] = "unknown"
            allocation_result['allocation_confidence'] = 0.3
        
        return allocation_result
        
    except Exception as e:
        logger.error(f"Error in allocate_profit_zone for {price_tick}: {e}")
        return {
            'timestamp': time.time(),
            'price_tick': price_tick,
            'routing_signal': 'error',
            'strategy_triggers': ["ERROR_RECOVERY"],
            'profit_zone': 'error',
            'allocation_confidence': 0.0,
            'error': str(e)
        }


def trigger_strategy(strategy_name: str, price_tick: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trigger a specific strategy with price tick data.
    
    Args:
        strategy_name: Name of strategy to trigger
        price_tick: Float price value
        metadata: Optional metadata for strategy
        
    Returns:
        Dictionary with strategy trigger result
    """
    try:
        trigger_result = {
            'timestamp': time.time(),
            'strategy_name': strategy_name,
            'price_tick': price_tick,
            'trigger_status': 'pending',
            'execution_confidence': 0.0,
            'metadata': metadata or {}
        }
        
        # Strategy-specific logic
        if strategy_name == "BTC_MICROHOLD_REBUY":
            trigger_result['trigger_status'] = 'executing'
            trigger_result['execution_confidence'] = 0.85
            trigger_result['action'] = 'buy_btc_micro'
            trigger_result['position_size'] = 'micro'
            
        elif strategy_name == "USDC_RSI_REBALANCE":
            trigger_result['trigger_status'] = 'executing'
            trigger_result['execution_confidence'] = 0.75
            trigger_result['action'] = 'rebalance_usdc'
            trigger_result['position_size'] = 'medium'
            
        elif strategy_name == "XRP_LIQUIDITY_VACUUM":
            trigger_result['trigger_status'] = 'executing'
            trigger_result['execution_confidence'] = 0.65
            trigger_result['action'] = 'capture_liquidity'
            trigger_result['position_size'] = 'large'
            
        elif strategy_name == "ZBE_RECOVERY_PATH":
            trigger_result['trigger_status'] = 'executing'
            trigger_result['execution_confidence'] = 0.95
            trigger_result['action'] = 'emergency_recovery'
            trigger_result['position_size'] = 'defensive'
            
        else:
            trigger_result['trigger_status'] = 'unknown_strategy'
            trigger_result['execution_confidence'] = 0.0
            trigger_result['action'] = 'no_action'
        
        return trigger_result
        
    except Exception as e:
        logger.error(f"Error in trigger_strategy for {strategy_name}: {e}")
        return {
            'timestamp': time.time(),
            'strategy_name': strategy_name,
            'price_tick': price_tick,
            'trigger_status': 'error',
            'execution_confidence': 0.0,
            'error': str(e),
            'metadata': metadata or {}
        }


def trigger_failsafe(failsafe_name: str, price_tick: float) -> Dict[str, Any]:
    """
    Trigger a failsafe strategy for emergency situations.
    
    Args:
        failsafe_name: Name of failsafe to trigger
        price_tick: Float price value
        
    Returns:
        Dictionary with failsafe trigger result
    """
    try:
        failsafe_result = {
            'timestamp': time.time(),
            'failsafe_name': failsafe_name,
            'price_tick': price_tick,
            'failsafe_status': 'activated',
            'priority': 'high',
            'action_taken': 'defensive_positioning'
        }
        
        if failsafe_name == "ZBE_RECOVERY_PATH":
            failsafe_result['action_taken'] = 'emergency_recovery'
            failsafe_result['priority'] = 'critical'
            
        return failsafe_result
        
    except Exception as e:
        logger.error(f"Error in trigger_failsafe for {failsafe_name}: {e}")
        return {
            'timestamp': time.time(),
            'failsafe_name': failsafe_name,
            'price_tick': price_tick,
            'failsafe_status': 'error',
            'error': str(e)
        }


def batch_allocate_profit_zones(price_ticks: List[float]) -> List[Dict[str, Any]]:
    """
    Allocate profit zones for multiple price ticks.
    
    Args:
        price_ticks: List of float price values
        
    Returns:
        List of profit zone allocation results
    """
    try:
        results = []
        for price_tick in price_ticks:
            allocation = allocate_profit_zone(price_tick)
            results.append(allocation)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch_allocate_profit_zones: {e}")
        return []


def analyze_profit_allocation_patterns(allocations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in profit zone allocations.
    
    Args:
        allocations: List of allocation results
        
    Returns:
        Dictionary with pattern analysis
    """
    try:
        if not allocations:
            return {'error': 'No allocations to analyze'}
        
        # Count strategy triggers
        strategy_counts = {}
        profit_zone_counts = {}
        confidence_sum = 0.0
        
        for allocation in allocations:
            # Count strategies
            for strategy in allocation.get('strategy_triggers', []):
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Count profit zones
            profit_zone = allocation.get('profit_zone', 'unknown')
            profit_zone_counts[profit_zone] = profit_zone_counts.get(profit_zone, 0) + 1
            
            # Sum confidence
            confidence_sum += allocation.get('allocation_confidence', 0.0)
        
        avg_confidence = confidence_sum / len(allocations) if allocations else 0.0
        
        # Find dominant patterns
        dominant_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'none'
        dominant_zone = max(profit_zone_counts.items(), key=lambda x: x[1])[0] if profit_zone_counts else 'none'
        
        return {
            'total_allocations': len(allocations),
            'strategy_distribution': strategy_counts,
            'profit_zone_distribution': profit_zone_counts,
            'dominant_strategy': dominant_strategy,
            'dominant_profit_zone': dominant_zone,
            'average_confidence': avg_confidence,
            'allocation_results': allocations
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_profit_allocation_patterns: {e}")
        return {'error': str(e)}


def optimize_profit_allocation(price_tick: float, historical_allocations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optimize profit allocation based on historical performance.
    
    Args:
        price_tick: Current price tick
        historical_allocations: List of historical allocation results
        
    Returns:
        Dictionary with optimized allocation
    """
    try:
        # Get base allocation
        base_allocation = allocate_profit_zone(price_tick)
        
        # Analyze historical performance
        if historical_allocations:
            pattern_analysis = analyze_profit_allocation_patterns(historical_allocations)
            
            # Adjust confidence based on historical patterns
            if pattern_analysis.get('dominant_strategy') in base_allocation.get('strategy_triggers', []):
                base_allocation['allocation_confidence'] *= 1.1  # Boost confidence
            else:
                base_allocation['allocation_confidence'] *= 0.9  # Reduce confidence
            
            # Add optimization metadata
            base_allocation['optimization_data'] = {
                'historical_patterns': pattern_analysis,
                'confidence_adjustment': 'applied'
            }
        
        return base_allocation
        
    except Exception as e:
        logger.error(f"Error in optimize_profit_allocation: {e}")
        return allocate_profit_zone(price_tick)  # Fallback to base allocation


def validate_allocation_input(price_tick: float) -> bool:
    """
    Validate input for profit allocation.
    
    Args:
        price_tick: Price tick to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it's a positive number
        if not isinstance(price_tick, (int, float)) or price_tick <= 0:
            return False
        
        # Check if it's reasonable
        if price_tick > 1000000:  # 1 million limit
            return False
        
        return True
        
    except Exception:
        return False


# Test function for validation
def test_profit_cycle_allocator():
    """Test the profit cycle allocator with sample prices."""
    test_prices = [
        2045.29,    # Should trigger BTC_MICROHOLD_REBUY
        123.456,    # Should trigger USDC_RSI_REBALANCE
        9999.99,    # Should trigger XRP_LIQUIDITY_VACUUM
        1111.11,    # Should trigger ZBE_RECOVERY_PATH
    ]
    
    print("💰 Testing Profit Cycle Allocator...")
    for price in test_prices:
        allocation = allocate_profit_zone(price)
        print(f"Price: {price} → Zone: {allocation['profit_zone']} → Strategies: {allocation['strategy_triggers']}")
    
    print("✅ Profit Cycle Allocator test completed")


if __name__ == "__main__":
    test_profit_cycle_allocator() 