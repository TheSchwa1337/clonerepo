#!/usr/bin/env python3
import sys
import os
import logging

# Setup debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add paths
sys.path.append('AOI_Base_Files_Schwabot/core')

from AOI_Base_Files_Schwabot.core.mode_integration_system import ModeIntegrationSystem, TradingMode

def debug_valid_market_data():
    """Debug the valid market data test."""
    system = ModeIntegrationSystem()
    
    # Test valid market data
    valid_market_data = {
        "price": 50000.0,
        "volume": 1000.0,
        "rsi": 30.0,  # Oversold
        "macd": 0.1,  # Positive
        "sentiment": 0.7,  # Positive
        "symbol": "BTC/USDC"
    }
    
    print("Testing valid market data:")
    print(f"  RSI: {valid_market_data['rsi']}")
    print(f"  MACD: {valid_market_data['macd']}")
    print(f"  Sentiment: {valid_market_data['sentiment']}")
    
    # Test each mode
    for mode in [TradingMode.DEFAULT, TradingMode.GHOST, TradingMode.HYBRID, TradingMode.PHANTOM]:
        print(f"\n--- Testing {mode.value.upper()} mode ---")
        system.current_mode = mode
        config = system.get_current_config()
        
        print(f"  Confidence threshold: {config.confidence_threshold}")
        print(f"  Can open position: {system._can_open_position(config)}")
        
        result = system.generate_trading_decision(valid_market_data)
        
        if result is not None:
            print(f"  ✅ Decision generated:")
            print(f"    Action: {result.action.value}")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Entry Price: ${result.entry_price:.2f}")
            print(f"    Stop Loss: ${result.stop_loss:.2f}")
            print(f"    Take Profit: ${result.take_profit:.2f}")
        else:
            print(f"  ❌ No decision generated")

if __name__ == "__main__":
    debug_valid_market_data() 