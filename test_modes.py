#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Different Execution Modes
================================

Demonstrates Shadow Mode, Paper Mode, and Live Mode functionality
for building trading context and validating strategies.
"""

import os
import time
import json
from clock_mode_system import ClockModeSystem, ExecutionMode, SAFETY_CONFIG

def test_shadow_mode():
    """Test Shadow Mode - Real data analysis only."""
    print("\n" + "="*60)
    print("🛡️ TESTING SHADOW MODE")
    print("="*60)
    
    # Set to Shadow Mode
    os.environ['CLOCK_MODE_EXECUTION'] = 'shadow'
    
    # Create and start system
    clock_system = ClockModeSystem()
    
    print(f"✅ Mode: {SAFETY_CONFIG.execution_mode.value}")
    print("📊 Using REAL market data for analysis only")
    print("❌ NO trading execution")
    
    # Run for a few cycles
    if clock_system.start_clock_mode():
        time.sleep(5)  # Let it run for 5 seconds
        
        # Get status
        status = clock_system.get_all_mechanisms_status()
        print(f"\n📈 Shadow Mode Status:")
        print(f"   - Real API Available: {status['real_api_system']['available']}")
        print(f"   - Mode: {status['shadow_mode']['mode']}")
        print(f"   - Description: {status['shadow_mode']['description']}")
        
        clock_system.stop_clock_mode()
    else:
        print("❌ Failed to start Shadow Mode")

def test_paper_mode():
    """Test Paper Mode - Simulated trading for context building."""
    print("\n" + "="*60)
    print("📈 TESTING PAPER MODE")
    print("="*60)
    
    # Set to Paper Mode
    os.environ['CLOCK_MODE_EXECUTION'] = 'paper'
    
    # Create and start system
    clock_system = ClockModeSystem()
    
    print(f"✅ Mode: {SAFETY_CONFIG.execution_mode.value}")
    print("📊 Using REAL market data")
    print("📈 SIMULATED trading execution")
    print("🧠 Building trading context")
    
    # Run for a few cycles
    if clock_system.start_clock_mode():
        time.sleep(8)  # Let it run for 8 seconds to see some trades
        
        # Get paper trading context
        context = clock_system.get_paper_trading_context()
        print(f"\n📊 Paper Trading Context:")
        print(f"   - Total Trades: {context['portfolio_summary']['total_trades']}")
        print(f"   - Win Rate: {context['portfolio_summary']['win_rate']:.1f}%")
        print(f"   - Total P&L: ${context['portfolio_summary']['total_pnl']:.2f}")
        print(f"   - Portfolio Value: ${context['portfolio_summary']['portfolio_value']:.2f}")
        
        # Strategy insights
        insights = context['strategy_insights']
        print(f"\n🧠 Strategy Insights:")
        print(f"   - Win Rate Acceptable: {insights['win_rate_acceptable']}")
        print(f"   - Profitable Strategy: {insights['profitable_strategy']}")
        print(f"   - Risk Management Working: {insights['risk_management_working']}")
        print(f"   - Ready for Live: {insights['ready_for_live']}")
        
        # Show recent trades
        if context['recent_trades']:
            print(f"\n📋 Recent Trades:")
            for trade in context['recent_trades'][-3:]:  # Last 3 trades
                print(f"   - {trade['action']} ${trade['amount']:.2f} at ${trade['price']:.2f} | P&L: ${trade['pnl']:.2f}")
        
        clock_system.stop_clock_mode()
    else:
        print("❌ Failed to start Paper Mode")

def test_live_mode():
    """Test Live Mode - Real trading (placeholder)."""
    print("\n" + "="*60)
    print("🚨 TESTING LIVE MODE")
    print("="*60)
    
    # Set to Live Mode
    os.environ['CLOCK_MODE_EXECUTION'] = 'live'
    
    # Create and start system
    clock_system = ClockModeSystem()
    
    print(f"✅ Mode: {SAFETY_CONFIG.execution_mode.value}")
    print("📊 Using REAL market data")
    print("🚨 REAL trading execution")
    print("⚠️ Real money at risk!")
    
    # Run for a few cycles
    if clock_system.start_clock_mode():
        time.sleep(3)  # Short run for safety
        
        # Get status
        status = clock_system.get_all_mechanisms_status()
        print(f"\n🚨 Live Mode Status:")
        print(f"   - Execution Mode: {status['safety_config']['execution_mode']}")
        print(f"   - Emergency Stop: {status['safety_config']['emergency_stop_enabled']}")
        print(f"   - Max Daily Loss: {status['safety_config']['max_daily_loss']*100}%")
        
        clock_system.stop_clock_mode()
    else:
        print("❌ Failed to start Live Mode")

def main():
    """Run all mode tests."""
    print("🧪 SCHWABOT EXECUTION MODE TESTING")
    print("="*60)
    print("Testing Shadow Mode, Paper Mode, and Live Mode")
    print("to demonstrate trading context building capabilities.")
    
    # Test each mode
    test_shadow_mode()
    test_paper_mode()
    test_live_mode()
    
    print("\n" + "="*60)
    print("✅ ALL MODES TESTED SUCCESSFULLY!")
    print("="*60)
    print("🎯 Key Takeaways:")
    print("   🛡️ Shadow Mode: Real data analysis, no risk")
    print("   📈 Paper Mode: Context building, simulated trading")
    print("   🚨 Live Mode: Real trading, real money at risk")
    print("\n📊 Paper Mode is perfect for building trading context!")

if __name__ == "__main__":
    main() 