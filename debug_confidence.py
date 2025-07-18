#!/usr/bin/env python3
"""
Debug Confidence Calculation
============================

This script tests the confidence calculation to see why it's returning 0.000.
"""

import numpy as np
from schwabot_trading_engine import MarketData, AssetClass, SchwabotTradingEngine

def test_confidence_calculation():
    """Test the confidence calculation with realistic values."""
    
    # Create test market data
    market_data = MarketData(
        timestamp=time.time(),
        asset="BTC",
        price=45000.0,
        volume=1000.0,
        bid=44995.0,
        ask=45005.0,
        spread=10.0,
        volatility=0.05,
        sentiment=0.8,  # High sentiment
        asset_class=AssetClass.CRYPTO
    )
    
    # Create test signal vector
    signal_vector = np.array([0.8, 0.7, 0.6])  # Strong signal
    
    # Calculate signal strength
    signal_strength = np.mean(signal_vector)
    print(f"Signal strength: {signal_strength}")
    
    # Calculate base confidence
    base_confidence = signal_strength * market_data.sentiment
    print(f"Base confidence: {base_confidence}")
    
    # Add volume factor
    volume_factor = min(1.0, market_data.volume / 1000.0)
    print(f"Volume factor: {volume_factor}")
    
    # Add volatility factor
    volatility_factor = 1.0
    if market_data.volatility < 0.01:
        volatility_factor = 0.7
    elif market_data.volatility > 0.15:
        volatility_factor = 0.8
    else:
        volatility_factor = 1.0
    print(f"Volatility factor: {volatility_factor}")
    
    # Calculate final confidence
    confidence = base_confidence * volume_factor * volatility_factor
    confidence = min(1.0, max(0.0, confidence))
    print(f"Final confidence: {confidence}")
    
    return confidence

if __name__ == "__main__":
    import time
    confidence = test_confidence_calculation()
    print(f"\nResult: {confidence}") 