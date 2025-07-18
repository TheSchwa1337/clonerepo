#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Schwabot backtesting system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, HistoricalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_historical_data_loader():
    """Test the historical data loader."""
    print("🧪 Testing Historical Data Loader")
    print("=" * 40)
    
    loader = HistoricalDataLoader()
    
    # Test loading Binance data
    print("📊 Testing Binance data loading...")
    df = await loader.load_binance_data(
        symbol="BTCUSDT",
        start_date="2024-06-01",
        end_date="2024-06-07",
        interval="1h"
    )
    
    if not df.empty:
        print(f"✅ Successfully loaded {len(df)} data points")
        print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        # Test conversion to market data points
        print("\n🔄 Testing conversion to MarketDataPoint objects...")
        market_data_points = loader.convert_to_market_data_points(df, "BTCUSDT")
        print(f"✅ Converted {len(market_data_points)} data points")
        
        if market_data_points:
            sample_point = market_data_points[0]
            print(f"📊 Sample data point:")
            print(f"   Symbol: {sample_point.symbol}")
            print(f"   Price: ${sample_point.price:.2f}")
            print(f"   Volume: {sample_point.volume:.2f}")
            print(f"   Price Change: {sample_point.price_change:.4f}")
            print(f"   Volatility: {sample_point.volatility:.4f}")
            print(f"   Sentiment: {sample_point.sentiment:.3f}")
        
        return True
    else:
        print("❌ Failed to load Binance data")
        return False

async def test_backtest_engine():
    """Test the backtest engine."""
    print("\n🧪 Testing Backtest Engine")
    print("=" * 40)
    
    # Create a short backtest configuration
    config = BacktestConfig(
        start_date="2024-06-01",
        end_date="2024-06-07",
        symbols=["BTCUSDT"],
        initial_balance=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        data_source="binance",
        timeframe="1h",
        enable_ai_analysis=True,
        enable_risk_management=True,
        max_positions=3,
        risk_per_trade=0.02
    )
    
    print(f"📅 Test period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial balance: ${config.initial_balance:,.2f}")
    print(f"📊 Symbols: {', '.join(config.symbols)}")
    print(f"⏱️  Timeframe: {config.timeframe}")
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    print("\n🚀 Running backtest...")
    result = await engine.run_backtest()
    
    # Display results
    print("\n📈 Backtest Results")
    print("=" * 40)
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
    
    if result.trade_history:
        print(f"\n📋 Trade History (first 5 trades):")
        for i, trade in enumerate(result.trade_history[:5]):
            print(f"   {i+1}. {trade['action']} {trade['symbol']} @ ${trade['price']:.2f}")
    
    return result.total_trades > 0  # Consider successful if trades were executed

async def test_quick_backtest():
    """Run a quick backtest to verify functionality."""
    print("\n⚡ Quick Backtest Test")
    print("=" * 40)
    
    # Very short period for quick test
    config = BacktestConfig(
        start_date="2024-06-01",
        end_date="2024-06-02",  # Just 1 day
        symbols=["BTCUSDT"],
        initial_balance=10000.0,
        data_source="binance",
        timeframe="1h",
        enable_ai_analysis=True,
        enable_risk_management=True
    )
    
    engine = BacktestEngine(config)
    result = await engine.run_backtest()
    
    print(f"✅ Quick backtest completed")
    print(f"📊 Data points processed: {len(result.equity_curve)}")
    print(f"🔄 Trades executed: {result.total_trades}")
    print(f"💰 Final balance: ${result.final_balance:,.2f}")
    
    return result.total_trades >= 0  # Success if no errors

async def main():
    """Run all backtesting tests."""
    print("🧪 Schwabot Backtesting System Test")
    print("=" * 50)
    
    try:
        # Test 1: Historical data loader
        print("\n1️⃣ Testing Historical Data Loader...")
        data_loader_success = await test_historical_data_loader()
        
        # Test 2: Quick backtest
        print("\n2️⃣ Testing Quick Backtest...")
        quick_backtest_success = await test_quick_backtest()
        
        # Test 3: Full backtest (if quick test passed)
        if quick_backtest_success:
            print("\n3️⃣ Testing Full Backtest...")
            full_backtest_success = await test_backtest_engine()
        else:
            print("⏭️  Skipping full backtest due to quick test failure")
            full_backtest_success = False
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 30)
        print(f"📊 Data Loader: {'✅ PASS' if data_loader_success else '❌ FAIL'}")
        print(f"⚡ Quick Backtest: {'✅ PASS' if quick_backtest_success else '❌ FAIL'}")
        print(f"🚀 Full Backtest: {'✅ PASS' if full_backtest_success else '❌ FAIL'}")
        
        overall_success = data_loader_success and quick_backtest_success and full_backtest_success
        
        if overall_success:
            print("\n🎉 All backtesting tests passed!")
            print("✅ The backtesting system is ready to use with real historical data")
        else:
            print("\n⚠️  Some tests failed. Check the logs for details.")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Backtesting test failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 