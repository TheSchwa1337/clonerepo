#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌉 TICK-KAPREKAR BRIDGE - PRICE TO VOLATILITY INDEX
===================================================

Bridge module that connects price ticks to Kaprekar analysis by normalizing
float prices into 4-digit integers for volatility convergence indexing.

Features:
- Price normalization for Kaprekar processing
- Multi-decimal precision handling
- Integration with existing tick processing
- Volatility index generation
"""

import logging
import re
from typing import Dict, Any, Optional
from .kaprekar_engine import kaprekar_iterations, get_volatility_classification

logger = logging.getLogger(__name__)

def price_to_kaprekar_index(price: float) -> int:
    """
    Normalize float price into a 4-digit integer, feed into kaprekar.
    
    Args:
        price: Float price value (e.g., 2045.29)
        
    Returns:
        Kaprekar iteration count for normalized price
    """
    try:
        # Convert price to string and remove decimal point
        price_str = str(price)
        
        # Extract digits only (remove decimal point)
        digits_only = re.sub(r'[^\d]', '', price_str)
        
        # Take first 4 digits
        if len(digits_only) >= 4:
            normalized = int(digits_only[:4])
        else:
            # Pad with zeros if less than 4 digits
            normalized = int(digits_only.ljust(4, '0'))
        
        # Calculate Kaprekar iterations
        return kaprekar_iterations(normalized)
        
    except Exception as e:
        logger.error(f"Error in price_to_kaprekar_index for {price}: {e}")
        return -1


def analyze_price_volatility(price: float) -> Dict[str, Any]:
    """
    Analyze price volatility using Kaprekar convergence.
    
    Args:
        price: Float price value
        
    Returns:
        Dictionary with volatility analysis
    """
    try:
        k_index = price_to_kaprekar_index(price)
        volatility_class = get_volatility_classification(k_index)
        
        return {
            'price': price,
            'kaprekar_index': k_index,
            'volatility_classification': volatility_class,
            'convergent': k_index != -1,
            'normalized_price': _extract_normalized_price(price)
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_price_volatility for {price}: {e}")
        return {
            'price': price,
            'kaprekar_index': -1,
            'volatility_classification': 'error',
            'convergent': False,
            'error': str(e)
        }


def _extract_normalized_price(price: float) -> int:
    """
    Extract normalized 4-digit price from float.
    
    Args:
        price: Float price value
        
    Returns:
        Normalized 4-digit integer
    """
    try:
        price_str = str(price)
        digits_only = re.sub(r'[^\d]', '', price_str)
        
        if len(digits_only) >= 4:
            return int(digits_only[:4])
        else:
            return int(digits_only.ljust(4, '0'))
            
    except Exception:
        return 0


def batch_analyze_prices(prices: list) -> Dict[str, Any]:
    """
    Analyze multiple prices for volatility patterns.
    
    Args:
        prices: List of float prices
        
    Returns:
        Dictionary with batch analysis results
    """
    try:
        results = []
        convergent_count = 0
        total_iterations = 0
        volatility_distribution = {
            'low_volatility': 0,
            'medium_volatility': 0,
            'high_volatility': 0,
            'extreme_volatility': 0,
            'non_convergent': 0
        }
        
        for price in prices:
            analysis = analyze_price_volatility(price)
            results.append(analysis)
            
            if analysis['convergent']:
                convergent_count += 1
                total_iterations += analysis['kaprekar_index']
            
            # Update volatility distribution
            vol_class = analysis['volatility_classification']
            if vol_class in volatility_distribution:
                volatility_distribution[vol_class] += 1
        
        convergence_rate = convergent_count / len(prices) if prices else 0
        avg_iterations = total_iterations / convergent_count if convergent_count > 0 else 0
        
        return {
            'total_prices': len(prices),
            'convergent_count': convergent_count,
            'convergence_rate': convergence_rate,
            'average_iterations': avg_iterations,
            'volatility_distribution': volatility_distribution,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch_analyze_prices: {e}")
        return {}


def get_volatility_signal(price: float) -> str:
    """
    Get volatility signal based on Kaprekar analysis.
    
    Args:
        price: Float price value
        
    Returns:
        Volatility signal string
    """
    try:
        k_index = price_to_kaprekar_index(price)
        
        if k_index == -1:
            return "ghost_shell_evasion"
        elif k_index < 3:
            return "vol_stable_basket"
        elif k_index <= 6:
            return "midrange_vol_logic"
        else:
            return "escape_vol_guard"
            
    except Exception as e:
        logger.error(f"Error in get_volatility_signal for {price}: {e}")
        return "error_signal"


def validate_price_input(price: float) -> bool:
    """
    Validate price input for Kaprekar processing.
    
    Args:
        price: Price to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it's a positive number
        if not isinstance(price, (int, float)) or price <= 0:
            return False
            
        # Check if it can be converted to string
        price_str = str(price)
        if not price_str or price_str == '0':
            return False
            
        return True
        
    except Exception:
        return False


# Test function for validation
def test_tick_kaprekar_bridge():
    """Test the tick-kaprekar bridge with sample prices."""
    test_prices = [
        2045.29,    # Should normalize to 2045
        123.456,    # Should normalize to 1234
        9999.99,    # Should normalize to 9999
        1.234,      # Should normalize to 1234
        50000.0,    # Should normalize to 5000
    ]
    
    print("🌉 Testing Tick-Kaprekar Bridge...")
    for price in test_prices:
        k_index = price_to_kaprekar_index(price)
        signal = get_volatility_signal(price)
        normalized = _extract_normalized_price(price)
        print(f"Price: {price} → Normalized: {normalized} → K-Index: {k_index} → Signal: {signal}")
    
    print("✅ Tick-Kaprekar Bridge test completed")


if __name__ == "__main__":
    test_tick_kaprekar_bridge() 