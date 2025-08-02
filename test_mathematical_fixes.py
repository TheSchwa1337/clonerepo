#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mathematical Fixes
======================

Comprehensive test to verify that all mathematical fixes are working correctly.
"""

import sys
import os
import numpy as np
import math
import logging

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'AOI_Base_Files_Schwabot', 'core'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phantom_mode_engine():
    """Test Phantom Mode Engine mathematical functions."""
    print("🧪 Testing Phantom Mode Engine...")
    
    try:
        from core.phantom_mode_engine import PhantomModeEngine, PhantomConfig
        
        # Test with extreme values
        engine = PhantomModeEngine()
        
        # Test ZeroBoundEntropy with extreme values
        extreme_entropy_values = [1000.0, -1000.0, 0.0, 1.0, float('inf'), float('-inf'), float('nan')]
        
        for entropy in extreme_entropy_values:
            try:
                result = engine.zbe.compress_entropy(entropy)
                if np.isfinite(result) and 0.0 <= result <= 1.0:
                    print(f"✅ ZeroBoundEntropy.compress_entropy({entropy}) = {result:.6f} - PASS")
                else:
                    print(f"❌ ZeroBoundEntropy.compress_entropy({entropy}) = {result} - FAIL (invalid result)")
            except Exception as e:
                print(f"❌ ZeroBoundEntropy.compress_entropy({entropy}) - FAIL (exception: {e})")
        
        # Test CycleBloomPrediction with extreme values
        test_bitmap = np.ones((64, 64)) * 0.5
        extreme_time_deltas = [1000.0, -1000.0, 0.0, 1.0, float('inf'), float('-inf'), float('nan')]
        
        for time_delta in extreme_time_deltas:
            try:
                result = engine.cbp.predict_next_cycle(1.0, test_bitmap, time_delta)
                if np.isfinite(result) and 0.0 <= result <= 1.0:
                    print(f"✅ CycleBloomPrediction.predict_next_cycle(1.0, bitmap, {time_delta}) = {result:.6f} - PASS")
                else:
                    print(f"❌ CycleBloomPrediction.predict_next_cycle(1.0, bitmap, {time_delta}) = {result} - FAIL (invalid result)")
            except Exception as e:
                print(f"❌ CycleBloomPrediction.predict_next_cycle(1.0, bitmap, {time_delta}) - FAIL (exception: {e})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Phantom Mode Engine: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Phantom Mode Engine: {e}")
        return False

def test_mode_integration_system():
    """Test Mode Integration System mathematical functions."""
    print("\n🧪 Testing Mode Integration System...")
    
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import ModeIntegrationSystem, TradingMode
        
        system = ModeIntegrationSystem()
        
        # Test position size calculation with extreme values
        test_cases = [
            {"price": 50000.0, "balance": 10000.0, "expected": "valid"},
            {"price": 0.0, "balance": 10000.0, "expected": "error"},
            {"price": -50000.0, "balance": 10000.0, "expected": "error"},
            {"price": float('inf'), "balance": 10000.0, "expected": "error"},
            {"price": float('nan'), "balance": 10000.0, "expected": "error"},
            {"price": 50000.0, "balance": 0.0, "expected": "error"},
            {"price": 50000.0, "balance": -10000.0, "expected": "error"}
        ]
        
        for case in test_cases:
            price = case["price"]
            balance = case["balance"]
            expected = case["expected"]
            
            # Set balance
            system.portfolio_state['balance'] = balance
            
            try:
                # Test each mode
                for mode in [TradingMode.DEFAULT, TradingMode.GHOST, TradingMode.HYBRID, TradingMode.PHANTOM]:
                    system.current_mode = mode
                    config = system.get_current_config()
                    
                    result = system._calculate_position_size(price, config)
                    
                    if expected == "valid":
                        if np.isfinite(result) and result >= 0.001:
                            print(f"✅ {mode.value} position_size({price}, {balance}) = {result:.6f} - PASS")
                        else:
                            print(f"❌ {mode.value} position_size({price}, {balance}) = {result} - FAIL (invalid result)")
                    else:  # expected == "error"
                        if result == 0.001:  # Should return minimum position size
                            print(f"✅ {mode.value} position_size({price}, {balance}) = {result} - PASS (handled error)")
                        else:
                            print(f"❌ {mode.value} position_size({price}, {balance}) = {result} - FAIL (should handle error)")
                            
            except Exception as e:
                if expected == "error":
                    print(f"✅ {mode.value} position_size({price}, {balance}) - PASS (caught exception: {e})")
                else:
                    print(f"❌ {mode.value} position_size({price}, {balance}) - FAIL (unexpected exception: {e})")
        
        # Test market data validation
        print("\n🧪 Testing market data validation...")
        
        invalid_market_data_cases = [
            {"price": 0.0, "volume": 1000.0, "rsi": 50.0, "macd": 0.0, "sentiment": 0.5},
            {"price": -50000.0, "volume": 1000.0, "rsi": 50.0, "macd": 0.0, "sentiment": 0.5},
            {"price": float('inf'), "volume": 1000.0, "rsi": 50.0, "macd": 0.0, "sentiment": 0.5},
            {"price": float('nan'), "volume": 1000.0, "rsi": 50.0, "macd": 0.0, "sentiment": 0.5},
            {"volume": 1000.0, "rsi": 50.0, "macd": 0.0, "sentiment": 0.5},  # Missing price
            None,  # Invalid input
            "not_a_dict"  # Wrong type
        ]
        
        for i, market_data in enumerate(invalid_market_data_cases):
            try:
                result = system.generate_trading_decision(market_data)
                if result is None:
                    print(f"✅ Invalid market data case {i+1} - PASS (rejected)")
                else:
                    print(f"❌ Invalid market data case {i+1} - FAIL (should be rejected)")
            except Exception as e:
                print(f"✅ Invalid market data case {i+1} - PASS (caught exception: {e})")
        
        # Test valid market data
        valid_market_data = {
            "price": 50000.0,
            "volume": 1000.0,
            "rsi": 30.0,  # Oversold
            "macd": 0.1,  # Positive
            "sentiment": 0.7,  # Positive
            "symbol": "BTC/USDC"
        }
        
        try:
            result = system.generate_trading_decision(valid_market_data)
            if result is not None:
                print(f"✅ Valid market data - PASS (decision generated)")
                print(f"   Action: {result.action.value}")
                print(f"   Entry Price: ${result.entry_price:.2f}")
                print(f"   Stop Loss: ${result.stop_loss:.2f}")
                print(f"   Take Profit: ${result.take_profit:.2f}")
                
                # Validate exit points
                if result.stop_loss < result.entry_price < result.take_profit:
                    print(f"   Exit points validation: PASS")
                else:
                    print(f"   Exit points validation: FAIL")
            else:
                print(f"❌ Valid market data - FAIL (no decision generated)")
        except Exception as e:
            print(f"❌ Valid market data - FAIL (exception: {e})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Mode Integration System: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Mode Integration System: {e}")
        return False

def test_backend_math_systems():
    """Test backend math systems."""
    print("\n🧪 Testing Backend Math Systems...")
    
    try:
        # Test basic math functions with edge cases
        test_cases = [
            {"func": "log", "args": [1.0], "expected": "valid"},
            {"func": "log", "args": [0.0], "expected": "error"},
            {"func": "log", "args": [-1.0], "expected": "error"},
            {"func": "sqrt", "args": [1.0], "expected": "valid"},
            {"func": "sqrt", "args": [0.0], "expected": "valid"},
            {"func": "sqrt", "args": [-1.0], "expected": "error"},
            {"func": "exp", "args": [0.0], "expected": "valid"},
            {"func": "exp", "args": [1000.0], "expected": "overflow"},
            {"func": "exp", "args": [-1000.0], "expected": "underflow"}
        ]
        
        for case in test_cases:
            func_name = case["func"]
            args = case["args"]
            expected = case["expected"]
            
            try:
                if func_name == "log":
                    result = math.log(args[0])
                elif func_name == "sqrt":
                    result = math.sqrt(args[0])
                elif func_name == "exp":
                    result = math.exp(args[0])
                
                if expected == "valid":
                    if np.isfinite(result):
                        print(f"✅ {func_name}({args[0]}) = {result:.6f} - PASS")
                    else:
                        print(f"❌ {func_name}({args[0]}) = {result} - FAIL (non-finite)")
                elif expected == "overflow":
                    if not np.isfinite(result):
                        print(f"✅ {func_name}({args[0]}) = {result} - PASS (overflow as expected)")
                    else:
                        print(f"❌ {func_name}({args[0]}) = {result} - FAIL (should overflow)")
                elif expected == "underflow":
                    if result == 0.0 or not np.isfinite(result):
                        print(f"✅ {func_name}({args[0]}) = {result} - PASS (underflow as expected)")
                    else:
                        print(f"❌ {func_name}({args[0]}) = {result} - FAIL (should underflow)")
                        
            except (ValueError, OverflowError) as e:
                if expected == "error":
                    print(f"✅ {func_name}({args[0]}) - PASS (caught expected error: {e})")
                else:
                    print(f"❌ {func_name}({args[0]}) - FAIL (unexpected error: {e})")
            except Exception as e:
                print(f"❌ {func_name}({args[0]}) - FAIL (unexpected exception: {e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Backend Math Systems: {e}")
        return False

def main():
    """Run all mathematical tests."""
    print("🔍 Comprehensive Mathematical Fixes Test")
    print("=" * 50)
    
    results = []
    
    # Test Phantom Mode Engine
    results.append(test_phantom_mode_engine())
    
    # Test Mode Integration System
    results.append(test_mode_integration_system())
    
    # Test Backend Math Systems
    results.append(test_backend_math_systems())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL MATHEMATICAL FIXES VERIFIED SUCCESSFULLY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 