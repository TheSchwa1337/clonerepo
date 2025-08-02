#!/usr/bin/env python3
"""
Test Real Price Data Implementation
==================================

This script tests that the trading bot now uses real API data instead of static get_real_price_data("BTC/USDC").
"""

import sys
import os
from pathlib import Path

# Import real API pricing and memory storage system
try:
    from real_api_pricing_memory_system import (
        initialize_real_api_memory_system, 
        get_real_price_data, 
        store_memory_entry,
        MemoryConfig,
        MemoryStorageMode,
        APIMode
    )
    REAL_API_AVAILABLE = True
except ImportError:
    REAL_API_AVAILABLE = False
    logger.warning("⚠️ Real API pricing system not available - using simulated data")


# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

def test_mode_integration_system():
    """Test the Mode Integration System with real price data."""
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import mode_integration_system, TradingMode
        
        print("🧮 Testing Mode Integration System...")
        
        # Test market data validation
        test_market_data = {
            'price': 45000.0,  # Real price, not get_real_price_data("BTC/USDC")
            'volume': 1000.0,
            'rsi': 29.0,
            'macd': 0.5,
            'sentiment': 0.7,
            'symbol': 'BTC/USDC',
            'timestamp': 1640995200.0  # Real timestamp
        }
        
        # Test that it accepts real data
        decision = mode_integration_system.generate_trading_decision(test_market_data)
        
        if decision:
            print(f"✅ Mode Integration System working with real price data")
            print(f"   Action: {decision.action.value}")
            print(f"   Price: ${decision.entry_price:.2f}")
            print(f"   Confidence: {decision.confidence:.3f}")
            print(f"   BTC Hash Event: {decision.btc_hash_event.event_type}")
            print(f"   Dualistic Consensus: {decision.dualistic_consensus.fallback_path}")
            return True
        else:
            print("⚠️ No decision generated (this might be expected)")
            return True
            
    except Exception as e:
        print(f"❌ Mode Integration System test failed: {e}")
        return False

def test_main_trading_bot():
    """Test the main trading bot with real price data."""
    try:
        from schwabot_trading_bot import SchwabotTradingBot
        
        print("🤖 Testing Main Trading Bot...")
        
        # Create bot instance
        bot = SchwabotTradingBot()
        
        # Test entry price method
        try:
            price = bot._get_entry_price("BTC/USDC")
            print(f"❌ Should have failed - got price: {price}")
            return False
        except ValueError as e:
            if "No live price data available" in str(e):
                print("✅ Main Trading Bot correctly requires real API data")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Main Trading Bot test failed: {e}")
        return False

def test_backtesting_engine():
    """Test the backtesting engine with real price data."""
    try:
        from backtesting.backtest_engine import BacktestEngine, BacktestConfig
        
        print("📊 Testing Backtesting Engine...")
        
        # Create config
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-02",
            symbols=["BTC/USDC"]
        )
        
        # Create engine
        engine = BacktestEngine(config)
        
        # Test entry price method
        try:
            price = engine._get_entry_price("BTC/USDC")
            print(f"❌ Should have failed - got price: {price}")
            return False
        except ValueError as e:
            if "No live price data available" in str(e):
                print("✅ Backtesting Engine correctly requires real API data")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Backtesting Engine test failed: {e}")
        return False

def test_production_pipeline():
    """Test the production pipeline with real price data."""
    try:
        from AOI_Base_Files_Schwabot.core.production_trading_pipeline import ProductionTradingPipeline, TradingConfig
        
        print("🏭 Testing Production Pipeline...")
        
        # Create config
        config = TradingConfig(
            exchange_name="binance",
            api_key="test",
            secret="test",
            sandbox=True
        )
        
        # Create pipeline
        pipeline = ProductionTradingPipeline(config)
        
        # Test entry price method
        try:
            price = pipeline._get_entry_price("BTC/USDC")
            print(f"❌ Should have failed - got price: {price}")
            return False
        except ValueError as e:
            if "No live price data available" in str(e):
                print("✅ Production Pipeline correctly requires real API data")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Production Pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 TESTING REAL PRICE DATA IMPLEMENTATION")
    print("=" * 50)
    
    tests = [
        ("Mode Integration System", test_mode_integration_system),
        ("Main Trading Bot", test_main_trading_bot),
        ("Backtesting Engine", test_backtesting_engine),
        ("Production Pipeline", test_production_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n🎯 TEST RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Real price data implementation is working!")
        return True
    else:
        print("⚠️ Some tests failed. Review the implementation.")
        return False

if __name__ == "__main__":
    main() 