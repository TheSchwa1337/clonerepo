#!/usr/bin/env python3
"""
🎯 DEMO: Unified Live Backtesting System
========================================

This demo shows the OFFICIAL backtesting system that uses LIVE API DATA
without placing real trades. This is what you call "backtesting" - testing
strategies on real market data without real money.
"""

import asyncio
import logging
from datetime import datetime
from unified_live_backtesting_system import BacktestConfig, BacktestMode, start_live_backtest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_live_backtesting():
    """Demo the unified live backtesting system."""
    print("🎯 UNIFIED LIVE BACKTESTING DEMO")
    print("=" * 50)
    print("This is the OFFICIAL backtesting system that uses LIVE API DATA")
    print("without placing real trades. This is what you call 'backtesting'.")
    print()
    
    # Configuration for live backtesting
    config = BacktestConfig(
        mode=BacktestMode.LIVE_API_BACKTEST,
        symbols=["BTCUSDT", "ETHUSDT"],
        exchanges=["binance"],
        initial_balance=10000.0,
        backtest_duration_hours=0.1,  # 6 minutes for demo
        enable_ai_analysis=True,
        enable_risk_management=True,
        enable_performance_optimization=True,
        data_update_interval=5.0,  # 5 seconds for demo
        min_confidence=0.6
    )
    
    print("📊 Backtest Configuration:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Symbols: {config.symbols}")
    print(f"   Exchanges: {config.exchanges}")
    print(f"   Initial Balance: ${config.initial_balance:,.2f}")
    print(f"   Duration: {config.backtest_duration_hours} hours")
    print(f"   AI Analysis: {'Enabled' if config.enable_ai_analysis else 'Disabled'}")
    print(f"   Risk Management: {'Enabled' if config.enable_risk_management else 'Disabled'}")
    print(f"   Performance Optimization: {'Enabled' if config.enable_performance_optimization else 'Disabled'}")
    print()
    
    print("🚀 Starting live backtesting...")
    print("   This will connect to real exchange APIs and stream live market data.")
    print("   No real trades will be placed - only simulated execution.")
    print("   The system gets smarter as it processes real market data.")
    print()
    
    try:
        # Start the backtest
        result = await start_live_backtest(config)
        
        print("✅ BACKTEST COMPLETED!")
        print("=" * 50)
        print(f"🎯 Backtest ID: {result.backtest_id}")
        print(f"📅 Start Time: {result.start_time}")
        print(f"📅 End Time: {result.end_time}")
        print(f"⏱️ Duration: {result.end_time - result.start_time}")
        print()
        
        print("💰 PERFORMANCE RESULTS:")
        print(f"   Total Return: {result.total_return:.2f}%")
        print(f"   Final Balance: ${result.final_balance:,.2f}")
        print(f"   Profit/Loss: ${result.final_balance - config.initial_balance:,.2f}")
        print()
        
        print("📈 TRADING STATISTICS:")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Winning Trades: {result.winning_trades}")
        print(f"   Losing Trades: {result.losing_trades}")
        print(f"   Win Rate: {result.win_rate:.1f}%")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print()
        
        print("📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"   Total Fees: ${result.total_fees:.2f}")
        print(f"   Total Slippage: ${result.total_slippage:.2f}")
        print()
        
        print("🤖 AI & STRATEGY ANALYSIS:")
        print(f"   AI Analysis Accuracy: {result.ai_analysis_accuracy:.1%}")
        print(f"   Risk Management Score: {result.risk_management_score:.1%}")
        print(f"   Mathematical Consensus: {result.mathematical_consensus.get('confidence', 0):.1%}")
        print()
        
        print("📋 STRATEGY PERFORMANCE:")
        strategy = result.strategy_performance
        print(f"   Average Trade Profit: ${strategy.get('avg_trade_profit', 0):.2f}")
        print(f"   Best Trade: ${strategy.get('best_trade', 0):.2f}")
        print(f"   Worst Trade: ${strategy.get('worst_trade', 0):.2f}")
        print()
        
        # Performance summary
        if result.total_return > 0:
            print("🎉 EXCELLENT PERFORMANCE!")
            print("   The strategy showed positive returns on live market data.")
        elif result.total_return == 0:
            print("📊 NEUTRAL PERFORMANCE")
            print("   The strategy maintained the initial balance.")
        else:
            print("📉 ROOM FOR IMPROVEMENT")
            print("   The strategy needs optimization for live market conditions.")
        
        print()
        print("🎯 This backtesting system:")
        print("   ✅ Uses REAL LIVE API DATA from actual exchanges")
        print("   ✅ Tests the FULL SCHWABOT TRADING PIPELINE")
        print("   ✅ Includes AI ANALYSIS and RISK MANAGEMENT")
        print("   ✅ Provides COMPREHENSIVE PERFORMANCE METRICS")
        print("   ✅ Gets SMARTER as it processes real market data")
        print("   ✅ Places NO REAL TRADES (simulated execution only)")
        print()
        print("🚀 Your system is ready for live trading validation!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")

async def main():
    """Main demo function."""
    await demo_live_backtesting()

if __name__ == "__main__":
    print("🎯 Starting Unified Live Backtesting Demo...")
    print("   This demonstrates your OFFICIAL backtesting system.")
    print("   It uses LIVE API DATA without placing real trades.")
    print()
    
    asyncio.run(main()) 