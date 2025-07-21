#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎡 FERRIS TICK LOGIC - VOLATILITY-BASED ROUTING
===============================================

Ferris tick logic that processes price ticks through Kaprekar analysis
and returns volatility-based routing signals for strategy allocation.

Features:
- Kaprekar-based volatility analysis
- Strategy routing based on convergence patterns
- Integration with existing Ferris wheel cycles
- Volatility signal generation for profit allocation
"""

import logging
import time
from typing import Dict, Any, Optional, List
from .tick_kaprekar_bridge import price_to_kaprekar_index, get_volatility_signal, analyze_price_volatility

logger = logging.getLogger(__name__)

def process_tick(price_tick: float) -> str:
    """
    Process a price tick through Kaprekar analysis and return routing signal.
    
    Args:
        price_tick: Float price value
        
    Returns:
        Routing signal string for strategy allocation
    """
    try:
        k_index = price_to_kaprekar_index(price_tick)

        if k_index == -1:
            return "ghost_shell_evasion"
        elif k_index < 3:
            return "vol_stable_basket"
        elif k_index <= 6:
            return "midrange_vol_logic"
        else:
            return "escape_vol_guard"
            
    except Exception as e:
        logger.error(f"Error in process_tick for {price_tick}: {e}")
        return "error_signal"


def process_tick_with_metadata(price_tick: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a price tick with additional metadata and return comprehensive analysis.
    
    Args:
        price_tick: Float price value
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary with comprehensive tick analysis
    """
    try:
        # Get basic volatility analysis
        volatility_analysis = analyze_price_volatility(price_tick)
        
        # Get routing signal
        routing_signal = process_tick(price_tick)
        
        # Create comprehensive result
        result = {
            'timestamp': time.time(),
            'price_tick': price_tick,
            'routing_signal': routing_signal,
            'volatility_analysis': volatility_analysis,
            'metadata': metadata or {}
        }
        
        # Add strategy recommendations based on signal
        result['strategy_recommendations'] = _get_strategy_recommendations(routing_signal)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_tick_with_metadata for {price_tick}: {e}")
        return {
            'timestamp': time.time(),
            'price_tick': price_tick,
            'routing_signal': 'error_signal',
            'error': str(e),
            'metadata': metadata or {}
        }


def _get_strategy_recommendations(routing_signal: str) -> List[str]:
    """
    Get strategy recommendations based on routing signal.
    
    Args:
        routing_signal: Routing signal from process_tick
        
    Returns:
        List of recommended strategies
    """
    strategy_map = {
        'ghost_shell_evasion': [
            'ZBE_RECOVERY_PATH',
            'GHOST_SHELL_DEFENSE',
            'EMERGENCY_STABILIZATION'
        ],
        'vol_stable_basket': [
            'BTC_MICROHOLD_REBUY',
            'STABLE_BASKET_ACCUMULATION',
            'CONSERVATIVE_POSITIONING'
        ],
        'midrange_vol_logic': [
            'USDC_RSI_REBALANCE',
            'MIDRANGE_VOL_STRATEGY',
            'BALANCED_POSITIONING'
        ],
        'escape_vol_guard': [
            'XRP_LIQUIDITY_VACUUM',
            'HIGH_VOL_ESCAPE',
            'AGGRESSIVE_POSITIONING'
        ],
        'error_signal': [
            'FALLBACK_STRATEGY',
            'ERROR_RECOVERY'
        ]
    }
    
    return strategy_map.get(routing_signal, ['DEFAULT_STRATEGY'])


def batch_process_ticks(price_ticks: List[float]) -> List[Dict[str, Any]]:
    """
    Process multiple price ticks in batch.
    
    Args:
        price_ticks: List of float price values
        
    Returns:
        List of tick processing results
    """
    try:
        results = []
        for price_tick in price_ticks:
            result = process_tick_with_metadata(price_tick)
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch_process_ticks: {e}")
        return []


def analyze_tick_patterns(price_ticks: List[float], window_size: int = 10) -> Dict[str, Any]:
    """
    Analyze patterns in a sequence of price ticks.
    
    Args:
        price_ticks: List of float price values
        window_size: Size of analysis window
        
    Returns:
        Dictionary with pattern analysis results
    """
    try:
        if len(price_ticks) < window_size:
            return {'error': 'Insufficient data for pattern analysis'}
        
        # Process recent ticks
        recent_ticks = price_ticks[-window_size:]
        tick_results = batch_process_ticks(recent_ticks)
        
        # Analyze signal distribution
        signal_counts = {}
        volatility_distribution = {}
        
        for result in tick_results:
            signal = result['routing_signal']
            vol_class = result['volatility_analysis']['volatility_classification']
            
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            volatility_distribution[vol_class] = volatility_distribution.get(vol_class, 0) + 1
        
        # Calculate pattern metrics
        dominant_signal = max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else 'unknown'
        signal_consistency = max(signal_counts.values()) / len(tick_results) if tick_results else 0
        
        return {
            'total_ticks': len(tick_results),
            'signal_distribution': signal_counts,
            'volatility_distribution': volatility_distribution,
            'dominant_signal': dominant_signal,
            'signal_consistency': signal_consistency,
            'tick_results': tick_results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_tick_patterns: {e}")
        return {'error': str(e)}


def get_ferris_cycle_position(price_tick: float, cycle_length: int = 16) -> Dict[str, Any]:
    """
    Get Ferris cycle position based on Kaprekar analysis.
    
    Args:
        price_tick: Float price value
        cycle_length: Length of Ferris cycle (default 16)
        
    Returns:
        Dictionary with cycle position information
    """
    try:
        k_index = price_to_kaprekar_index(price_tick)
        
        # Map Kaprekar index to cycle position
        if k_index == -1:
            cycle_position = 0  # Ghost shell position
        else:
            # Map iterations to cycle positions (1-16)
            cycle_position = min(k_index + 1, cycle_length)
        
        return {
            'price_tick': price_tick,
            'kaprekar_index': k_index,
            'cycle_position': cycle_position,
            'cycle_length': cycle_length,
            'cycle_progress': cycle_position / cycle_length
        }
        
    except Exception as e:
        logger.error(f"Error in get_ferris_cycle_position for {price_tick}: {e}")
        return {
            'price_tick': price_tick,
            'error': str(e)
        }


def validate_tick_input(price_tick: float) -> bool:
    """
    Validate tick input for processing.
    
    Args:
        price_tick: Price tick to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it's a positive number
        if not isinstance(price_tick, (int, float)) or price_tick <= 0:
            return False
        
        # Check if it's reasonable (not too large)
        if price_tick > 1000000:  # 1 million limit
            return False
        
        return True
        
    except Exception:
        return False


# Test function for validation
def test_ferris_tick_logic():
    """Test the Ferris tick logic with sample prices."""
    test_prices = [
        2045.29,    # Should route to vol_stable_basket
        123.456,    # Should route to midrange_vol_logic
        9999.99,    # Should route to escape_vol_guard
        1111.11,    # Should route to ghost_shell_evasion (non-convergent)
    ]
    
    print("🎡 Testing Ferris Tick Logic...")
    for price in test_prices:
        signal = process_tick(price)
        cycle_info = get_ferris_cycle_position(price)
        print(f"Price: {price} → Signal: {signal} → Cycle Position: {cycle_info.get('cycle_position', 'error')}")
    
    print("✅ Ferris Tick Logic test completed")


if __name__ == "__main__":
    test_ferris_tick_logic() 