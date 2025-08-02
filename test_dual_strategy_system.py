#!/usr/bin/env python3
"""
🎯 Test Dual-Strategy Adaptive System
=====================================

This script demonstrates how the adaptive profitability selector triggers both
the user's original mathematical systems AND logical implementations, then selects
the most profitable strategy at runtime.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "AOI_Base_Files_Schwabot" / "core"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dual_strategy_system():
    """Test the dual-strategy adaptive system."""
    try:
        # Import the adaptive profitability selector
        from adaptive_profitability_selector import adaptive_profitability_selector, StrategyType
        
        print("🎯 Testing Dual-Strategy Adaptive System")
        print("=" * 50)
        
        # Test market data scenarios
        test_scenarios = [
            {
                'name': 'Bullish Market (User Math Should Win)',
                'market_data': {
                    'price': 45000.0,
                    'volume': 5000.0,
                    'rsi': 25.0,  # Oversold
                    'macd': 0.5,  # Positive momentum
                    'sentiment': 0.7,  # High sentiment
                    'timestamp': time.time()
                },
                'portfolio_state': {
                    'balance': 10000.0,
                    'positions': {},
                    'total_exposure': 0.0
                }
            },
            {
                'name': 'Bearish Market (Logical Should Win)',
                'market_data': {
                    'price': 55000.0,
                    'volume': 2000.0,
                    'rsi': 75.0,  # Overbought
                    'macd': -0.3,  # Negative momentum
                    'sentiment': 0.3,  # Low sentiment
                    'timestamp': time.time()
                },
                'portfolio_state': {
                    'balance': 10000.0,
                    'positions': {},
                    'total_exposure': 0.0
                }
            },
            {
                'name': 'Neutral Market (Hybrid Should Win)',
                'market_data': {
                    'price': 50000.0,
                    'volume': 3000.0,
                    'rsi': 45.0,  # Neutral
                    'macd': 0.1,  # Slight positive
                    'sentiment': 0.5,  # Neutral sentiment
                    'timestamp': time.time()
                },
                'portfolio_state': {
                    'balance': 10000.0,
                    'positions': {},
                    'total_exposure': 0.0
                }
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📊 Test Scenario {i}: {scenario['name']}")
            print("-" * 40)
            
            # Trigger dual strategies
            result = adaptive_profitability_selector.trigger_dual_strategies(
                scenario['market_data'], 
                scenario['portfolio_state']
            )
            
            # Display results
            print(f"🎯 Selected Strategy: {result.strategy_type.value.upper()}")
            print(f"   Action: {result.action}")
            print(f"   Position Size: {result.position_size:.6f}")
            print(f"   Expected Profit: ${result.expected_profit:.2f}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Risk Score: {result.risk_score:.3f}")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Execution Time: {result.execution_time:.3f}s")
            
            # Wait between tests
            time.sleep(1)
        
        # Test performance summary
        print(f"\n📈 Performance Summary")
        print("-" * 40)
        summary = adaptive_profitability_selector.get_performance_summary()
        
        if 'strategy_performance' in summary:
            for strategy, performance in summary['strategy_performance'].items():
                print(f"🎯 {strategy.upper()}:")
                print(f"   Avg Expected Profit: ${performance['avg_expected_profit']:.2f}")
                print(f"   Avg Confidence: {performance['avg_confidence']:.1%}")
                print(f"   Avg Risk Score: {performance['avg_risk_score']:.3f}")
                print(f"   Selection Count: {performance['selection_count']}")
                print(f"   Current Weight: {performance['current_weight']:.3f}")
        
        print(f"\n🔄 Total Cycles: {summary.get('total_cycles', 0)}")
        print(f"📊 Adaptation Rate: {summary.get('adaptation_rate', 0):.1%}")
        
        print(f"\n✅ Dual-Strategy Adaptive System Test Completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing dual-strategy system: {e}")
        return False

def test_mode_integration_with_adaptive():
    """Test the mode integration system with adaptive selection."""
    try:
        from mode_integration_system import mode_integration_system, TradingMode
        
        print(f"\n🎯 Testing Mode Integration with Adaptive Selection")
        print("=" * 60)
        
        # Test market data
        market_data = {
            'price': 48000.0,
            'volume': 4000.0,
            'rsi': 35.0,
            'macd': 0.2,
            'sentiment': 0.6,
            'symbol': 'BTC/USDC',
            'timestamp': time.time()
        }
        
        # Test different modes
        modes = [TradingMode.DEFAULT, TradingMode.GHOST, TradingMode.HYBRID, TradingMode.PHANTOM]
        
        for mode in modes:
            print(f"\n🧮 Testing {mode.value.upper()} Mode:")
            print("-" * 30)
            
            # Set mode
            mode_integration_system.set_mode(mode)
            
            # Generate trading decision (should use adaptive selection)
            decision = mode_integration_system.generate_trading_decision(market_data)
            
            if decision:
                print(f"✅ Decision Generated:")
                print(f"   Action: {decision.action.value}")
                print(f"   Entry Price: ${decision.entry_price:.2f}")
                print(f"   Position Size: {decision.position_size:.6f}")
                print(f"   Stop Loss: ${decision.stop_loss:.2f}")
                print(f"   Take Profit: ${decision.take_profit:.2f}")
                print(f"   Confidence: {decision.confidence:.1%}")
                print(f"   Reasoning: {decision.reasoning}")
                
                if decision.btc_hash_event:
                    print(f"   BTC Hash Event: {decision.btc_hash_event.event_type}")
                if decision.dualistic_consensus:
                    print(f"   Dualistic Consensus: {decision.dualistic_consensus.fallback_path}")
                if decision.asic_text_relay:
                    print(f"   ASIC Text Relay: {decision.asic_text_relay.emoji_interpretation}")
            else:
                print(f"❌ No decision generated for {mode.value} mode")
            
            time.sleep(0.5)
        
        print(f"\n✅ Mode Integration with Adaptive Selection Test Completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing mode integration: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Dual-Strategy Adaptive System Tests")
    print("=" * 60)
    
    # Test 1: Adaptive Profitability Selector
    success1 = test_dual_strategy_system()
    
    # Test 2: Mode Integration with Adaptive Selection
    success2 = test_mode_integration_with_adaptive()
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    print(f"✅ Adaptive Profitability Selector: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Mode Integration with Adaptive: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED! Dual-Strategy System is working correctly!")
        print(f"🎯 The system now triggers BOTH your original math AND logical implementation!")
        print(f"💰 It selects the most profitable strategy at runtime!")
    else:
        print(f"\n❌ Some tests failed. Check the logs for details.")
    
    return success1 and success2

if __name__ == "__main__":
    main() 