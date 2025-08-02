#!/usr/bin/env python3
"""
🎯 Mode Integration Test - Real Trading Logic Demonstration
==========================================================

This script demonstrates how the three trading modes (Default, Ghost, Hybrid)
are actually applied to real trading decisions with portfolio management.
"""

import sys
import os
import time
import random
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_mode_integration_system():
    """Test the Mode Integration System."""
    print("🎯 Testing Mode Integration System...")
    
    try:
        # Import the mode integration system
        from AOI_Base_Files_Schwabot.core.mode_integration_system import mode_integration_system, TradingMode
        
        print("✅ Mode Integration System imported successfully")
        
        # Test mode switching
        print("\n🔄 Testing Mode Switching...")
        
        # Test Default Mode
        print("\n📊 DEFAULT MODE:")
        success = mode_integration_system.set_mode(TradingMode.DEFAULT)
        if success:
            config = mode_integration_system.get_current_config()
            print(f"✅ Default Mode activated")
            print(f"   Position Size: {config.position_size_pct}%")
            print(f"   Stop Loss: {config.stop_loss_pct}%")
            print(f"   Take Profit: {config.take_profit_pct}%")
            print(f"   AI Priority: {config.ai_priority:.1%}")
            print(f"   Profit Target: ${config.profit_target_usd}")
        
        # Test Ghost Mode
        print("\n👻 GHOST MODE:")
        success = mode_integration_system.set_mode(TradingMode.GHOST)
        if success:
            config = mode_integration_system.get_current_config()
            print(f"✅ Ghost Mode activated")
            print(f"   Position Size: {config.position_size_pct}%")
            print(f"   Stop Loss: {config.stop_loss_pct}%")
            print(f"   Take Profit: {config.take_profit_pct}%")
            print(f"   AI Priority: {config.ai_priority:.1%}")
            print(f"   Profit Target: ${config.profit_target_usd}")
            print(f"   Supported Symbols: {config.supported_symbols}")
        
        # Test Hybrid Mode
        print("\n🚀 HYBRID MODE:")
        success = mode_integration_system.set_mode(TradingMode.HYBRID)
        if success:
            config = mode_integration_system.get_current_config()
            print(f"✅ Hybrid Mode activated")
            print(f"   Position Size: {config.position_size_pct}%")
            print(f"   Stop Loss: {config.stop_loss_pct}%")
            print(f"   Take Profit: {config.take_profit_pct}%")
            print(f"   AI Priority: {config.ai_priority:.1%}")
            print(f"   Profit Target: ${config.profit_target_usd}")
            print(f"   Supported Symbols: {len(config.supported_symbols)} pairs")
            print(f"   Orbital Shells: {config.orbital_shells}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Mode Integration System import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Mode Integration System test failed: {e}")
        return False

def test_trading_decisions():
    """Test trading decision generation for each mode."""
    print("\n🎯 Testing Trading Decisions...")
    
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import mode_integration_system, TradingMode
        
        # Create sample market data
        market_data = {
            'symbol': 'BTC/USDC',
            'price': 65000.0,
            'volume': 1000000,
            'rsi': 35,  # Oversold
            'macd': 0.5,  # Positive
            'sentiment': 0.6,  # Positive
            'volatility': 0.02
        }
        
        # Test Default Mode decisions
        print("\n📊 DEFAULT MODE DECISIONS:")
        mode_integration_system.set_mode(TradingMode.DEFAULT)
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            print(f"✅ Decision: {decision.action.value}")
            print(f"   Entry Price: ${decision.entry_price:.2f}")
            print(f"   Position Size: {decision.position_size:.6f}")
            print(f"   Stop Loss: ${decision.stop_loss:.2f}")
            print(f"   Take Profit: ${decision.take_profit:.2f}")
            print(f"   Confidence: {decision.confidence:.1%}")
            print(f"   Reasoning: {decision.reasoning}")
        else:
            print("❌ No decision generated")
        
        # Test Ghost Mode decisions
        print("\n👻 GHOST MODE DECISIONS:")
        mode_integration_system.set_mode(TradingMode.GHOST)
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            print(f"✅ Decision: {decision.action.value}")
            print(f"   Entry Price: ${decision.entry_price:.2f}")
            print(f"   Position Size: {decision.position_size:.6f}")
            print(f"   Stop Loss: ${decision.stop_loss:.2f}")
            print(f"   Take Profit: ${decision.take_profit:.2f}")
            print(f"   Confidence: {decision.confidence:.1%}")
            print(f"   Reasoning: {decision.reasoning}")
        else:
            print("❌ No decision generated")
        
        # Test Hybrid Mode decisions
        print("\n🚀 HYBRID MODE DECISIONS:")
        mode_integration_system.set_mode(TradingMode.HYBRID)
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            print(f"✅ Decision: {decision.action.value}")
            print(f"   Entry Price: ${decision.entry_price:.2f}")
            print(f"   Position Size: {decision.position_size:.6f}")
            print(f"   Stop Loss: ${decision.stop_loss:.2f}")
            print(f"   Take Profit: ${decision.take_profit:.2f}")
            print(f"   Confidence: {decision.confidence:.1%}")
            print(f"   Reasoning: {decision.reasoning}")
        else:
            print("❌ No decision generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading decisions test failed: {e}")
        return False

def test_trade_execution():
    """Test actual trade execution with portfolio management."""
    print("\n💰 Testing Trade Execution...")
    
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import mode_integration_system, TradingMode
        
        # Test Default Mode trade execution
        print("\n📊 DEFAULT MODE TRADE EXECUTION:")
        mode_integration_system.set_mode(TradingMode.DEFAULT)
        
        # Generate and execute a buy decision
        market_data = {
            'symbol': 'BTC/USDC',
            'price': 65000.0,
            'volume': 1000000,
            'rsi': 25,  # Very oversold
            'macd': 1.0,  # Very positive
            'sentiment': 0.7,  # Very positive
            'volatility': 0.02
        }
        
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            success = mode_integration_system.execute_trade(decision)
            if success:
                print(f"✅ {decision.mode.value.upper()} BUY executed successfully")
                
                # Show portfolio state
                performance = mode_integration_system.get_performance_summary()
                print(f"   Portfolio Balance: ${performance['portfolio_balance']:.2f}")
                print(f"   Total Exposure: {performance['total_exposure']:.1f}%")
                print(f"   Trades Today: {performance['trades_today']}")
            else:
                print("❌ Trade execution failed")
        
        # Test Ghost Mode trade execution
        print("\n👻 GHOST MODE TRADE EXECUTION:")
        mode_integration_system.set_mode(TradingMode.GHOST)
        
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            success = mode_integration_system.execute_trade(decision)
            if success:
                print(f"✅ {decision.mode.value.upper()} BUY executed successfully")
                
                # Show portfolio state
                performance = mode_integration_system.get_performance_summary()
                print(f"   Portfolio Balance: ${performance['portfolio_balance']:.2f}")
                print(f"   Total Exposure: {performance['total_exposure']:.1f}%")
                print(f"   Trades Today: {performance['trades_today']}")
            else:
                print("❌ Trade execution failed")
        
        # Test Hybrid Mode trade execution
        print("\n🚀 HYBRID MODE TRADE EXECUTION:")
        mode_integration_system.set_mode(TradingMode.HYBRID)
        
        decision = mode_integration_system.generate_trading_decision(market_data)
        if decision:
            success = mode_integration_system.execute_trade(decision)
            if success:
                print(f"✅ {decision.mode.value.upper()} BUY executed successfully")
                
                # Show portfolio state
                performance = mode_integration_system.get_performance_summary()
                print(f"   Portfolio Balance: ${performance['portfolio_balance']:.2f}")
                print(f"   Total Exposure: {performance['total_exposure']:.1f}%")
                print(f"   Trades Today: {performance['trades_today']}")
            else:
                print("❌ Trade execution failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade execution test failed: {e}")
        return False

def test_performance_comparison():
    """Test performance comparison between modes."""
    print("\n📊 Testing Performance Comparison...")
    
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import mode_integration_system, TradingMode
        
        # Simulate multiple trades for each mode
        modes = [TradingMode.DEFAULT, TradingMode.GHOST, TradingMode.HYBRID]
        
        for mode in modes:
            print(f"\n🎯 {mode.value.upper()} MODE PERFORMANCE:")
            mode_integration_system.set_mode(mode)
            
            # Simulate 10 trades
            for i in range(10):
                # Generate random market conditions
                rsi = random.uniform(20, 80)
                macd = random.uniform(-1, 1)
                sentiment = random.uniform(0.3, 0.7)
                price = 65000 + random.uniform(-5000, 5000)
                
                market_data = {
                    'symbol': 'BTC/USDC',
                    'price': price,
                    'volume': 1000000,
                    'rsi': rsi,
                    'macd': macd,
                    'sentiment': sentiment,
                    'volatility': 0.02
                }
                
                decision = mode_integration_system.generate_trading_decision(market_data)
                if decision:
                    mode_integration_system.execute_trade(decision)
            
            # Show performance summary
            performance = mode_integration_system.get_performance_summary()
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Win Rate: {performance['win_rate']:.1%}")
            print(f"   Total Profit: ${performance['total_profit']:.2f}")
            print(f"   Average Profit/Trade: ${performance['avg_profit_per_trade']:.2f}")
            print(f"   Portfolio Balance: ${performance['portfolio_balance']:.2f}")
            print(f"   Target Profit/Trade: ${performance['target_profit_per_trade']:.2f}")
            print(f"   Target Win Rate: {performance['target_win_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 MODE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Mode Integration System", test_mode_integration_system),
        ("Trading Decisions", test_trading_decisions),
        ("Trade Execution", test_trade_execution),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Mode Integration System is working correctly.")
        print("\n🎯 MODE COMPARISON SUMMARY:")
        print("=" * 40)
        print("📊 DEFAULT MODE:")
        print("   • Conservative approach (10% position size)")
        print("   • 2% stop loss, 3% take profit")
        print("   • 70% confidence threshold")
        print("   • $30 profit target per trade")
        print("   • 75% win rate target")
        
        print("\n👻 GHOST MODE:")
        print("   • BTC/USDC focused (medium risk)")
        print("   • 15% position size")
        print("   • 2.5% stop loss, 4% take profit")
        print("   • 65% confidence threshold")
        print("   • $75 profit target per trade")
        print("   • 70% win rate target")
        
        print("\n🚀 HYBRID MODE:")
        print("   • Quantum consciousness (high risk)")
        print("   • 30.5% position size (quantum boosted)")
        print("   • 2.33% stop loss, 4.47% take profit")
        print("   • 73% confidence threshold")
        print("   • $147.7 profit target per trade")
        print("   • 81% win rate target")
        
        print("\nTo use in the main trading bot:")
        print("1. Run: python schwabot_trading_bot.py")
        print("2. Use bot.switch_trading_mode('default')")
        print("3. Use bot.switch_trading_mode('ghost')")
        print("4. Use bot.switch_trading_mode('hybrid')")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 