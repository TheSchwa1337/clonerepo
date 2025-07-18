#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Backtest Runner
========================

Simple script to run backtests with real historical data.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtest_engine import BacktestConfig, BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def run_sample_backtest():
    """Run a sample backtest with BTC/USDT data."""
    
    # Configuration for backtest
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-06-30",
        symbols=["BTCUSDT"],
        initial_balance=10000.0,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,   # 0.05%
        data_source="binance",
        timeframe="1h",
        enable_ai_analysis=True,
        enable_risk_management=True,
        max_positions=3,
        risk_per_trade=0.02
    )
    
    print("🚀 Starting Schwabot Backtest")
    print("=" * 50)
    print(f"📅 Period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial Balance: ${config.initial_balance:,.2f}")
    print(f"📊 Symbols: {', '.join(config.symbols)}")
    print(f"⏱️  Timeframe: {config.timeframe}")
    print(f"🤖 AI Analysis: {'Enabled' if config.enable_ai_analysis else 'Disabled'}")
    print(f"🛡️  Risk Management: {'Enabled' if config.enable_risk_management else 'Disabled'}")
    print("=" * 50)
    
    # Create and run backtest
    engine = BacktestEngine(config)
    result = await engine.run_backtest()
    
    # Display results
    print("\n📈 Backtest Results")
    print("=" * 50)
    print(f"💰 Final Balance: ${result.final_balance:,.2f}")
    print(f"📊 Total Return: {result.total_return:.2%}")
    print(f"🔄 Total Trades: {result.total_trades}")
    print(f"✅ Winning Trades: {result.winning_trades}")
    print(f"❌ Losing Trades: {result.losing_trades}")
    print(f"🎯 Win Rate: {result.win_rate:.2%}")
    print(f"📉 Max Drawdown: {result.max_drawdown:.2%}")
    print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"💵 Total Commission: ${result.total_commission:.2f}")
    print(f"📈 Total Slippage: ${result.total_slippage:.2f}")
    
    if result.performance_metrics:
        print(f"💰 Total P&L: ${result.performance_metrics.get('total_pnl', 0):.2f}")
        print(f"📊 Avg Trade P&L: ${result.performance_metrics.get('avg_trade_pnl', 0):.2f}")
        print(f"📈 Profit Factor: {result.performance_metrics.get('profit_factor', 0):.3f}")
    
    print("=" * 50)
    
    # Save results to file
    results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        import json
        json.dump({
            "config": {
                "start_date": result.config.start_date,
                "end_date": result.config.end_date,
                "symbols": result.config.symbols,
                "initial_balance": result.config.initial_balance,
                "timeframe": result.config.timeframe
            },
            "results": {
                "final_balance": result.final_balance,
                "total_return": result.total_return,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "total_commission": result.total_commission,
                "total_slippage": result.total_slippage
            },
            "performance_metrics": result.performance_metrics
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    return result

async def run_custom_backtest():
    """Run a custom backtest with user input."""
    
    print("🎯 Custom Backtest Configuration")
    print("=" * 40)
    
    # Get user input
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    symbols_input = input("Symbols (comma-separated, e.g., BTCUSDT,ETHUSDT): ").strip()
    symbols = [s.strip() for s in symbols_input.split(",")]
    initial_balance = float(input("Initial balance ($): ").strip())
    timeframe = input("Timeframe (1m,5m,15m,1h,4h,1d): ").strip() or "1h"
    
    # Configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_balance=initial_balance,
        timeframe=timeframe,
        enable_ai_analysis=True,
        enable_risk_management=True
    )
    
    print(f"\n🚀 Running custom backtest...")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Timeframe: {timeframe}")
    
    # Create and run backtest
    engine = BacktestEngine(config)
    result = await engine.run_backtest()
    
    # Display results
    print("\n📈 Backtest Results")
    print("=" * 40)
    print(f"💰 Final Balance: ${result.final_balance:,.2f}")
    print(f"📊 Total Return: {result.total_return:.2%}")
    print(f"🔄 Total Trades: {result.total_trades}")
    print(f"✅ Win Rate: {result.win_rate:.2%}")
    print(f"📉 Max Drawdown: {result.max_drawdown:.2%}")
    print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
    
    return result

async def main():
    """Main function."""
    print("🎯 Schwabot Backtest Runner")
    print("=" * 40)
    print("1. Run sample backtest (BTC/USDT)")
    print("2. Run custom backtest")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    try:
        if choice == "1":
            await run_sample_backtest()
        elif choice == "2":
            await run_custom_backtest()
        elif choice == "3":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice. Running sample backtest...")
            await run_sample_backtest()
    except KeyboardInterrupt:
        print("\n⏹️  Backtest interrupted by user")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        logger.exception("Backtest failed")

if __name__ == "__main__":
    asyncio.run(main()) 