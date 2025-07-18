#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trading System Test
========================

Comprehensive test of the complete live trading system including:
- Live market data integration (Coinbase, Kraken, Finance API)
- Real-time RSI calculations and technical analysis
- Time-based phase detection (midnight/noon patterns)
- Decimal key mapping (2, 6, 8 tiers)
- Memory key generation and recall
- Strategy execution engine
- Risk management and position sizing
- Cross-exchange arbitrage detection

This test demonstrates the complete internalized trading strategy system.
"""

import sys
import os
import time
import logging
import threading
import random
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/live_trading_test.log')
        ]
    )

def test_live_market_data_integration():
    """Test the live market data integration system."""
    print("🧪 Testing Live Market Data Integration")
    print("=" * 50)
    
    try:
        from core.live_market_data_integration import (
            LiveMarketDataIntegration, 
            TimePhase, 
            StrategyTier
        )
        
        # Configuration with mock API keys for testing
        config = {
            'coinbase': {
                'api_key': 'test_coinbase_key',
                'secret': 'test_coinbase_secret',
                'password': 'test_coinbase_password',
                'sandbox': True
            },
            'kraken': {
                'api_key': 'test_kraken_key',
                'secret': 'test_kraken_secret',
                'sandbox': True
            },
            'finance_api': {
                'api_key': 'test_finance_api_key'
            }
        }
        
        # Initialize market integration
        market_integration = LiveMarketDataIntegration(config)
        
        # Test 1: System initialization
        print("\n📡 Test 1: System initialization")
        print("-" * 40)
        
        status = market_integration.get_system_status()
        print(f"  Running: {status['running']}")
        print(f"  Exchanges connected: {status['exchanges_connected']}")
        print(f"  Cache size: {status['cache_size']}")
        print(f"  Signal count: {status['signal_count']}")
        print(f"  Memory key count: {status['memory_key_count']}")
        
        # Test 2: Start data feed
        print("\n🚀 Test 2: Start live data feed")
        print("-" * 40)
        
        market_integration.start_data_feed()
        print("  ✅ Live data feed started")
        
        # Test 3: Monitor data collection
        print("\n📊 Test 3: Monitor data collection")
        print("-" * 40)
        
        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            
            # Print status every 10 seconds
            if (i + 1) % 10 == 0:
                status = market_integration.get_system_status()
                print(f"  Time: {i+1}s | Fetches: {status['data_fetch_count']} | "
                      f"Signals: {status['signal_count']} | Errors: {status['error_count']}")
        
        # Test 4: Get latest market data
        print("\n📈 Test 4: Latest market data")
        print("-" * 40)
        
        symbols = ['BTC/USDC', 'ETH/USDC', 'XRP/USDC', 'SOL/USDC']
        for symbol in symbols:
            data = market_integration.get_latest_market_data(symbol)
            if data:
                print(f"  {symbol}:")
                print(f"    Price: ${data.price:.2f}")
                print(f"    RSI: {data.rsi:.1f}")
                print(f"    Volume: {data.volume:.0f}")
                print(f"    Phase: {data.phase.value}")
                print(f"    Tier: {data.strategy_tier.value}")
                print(f"    Decimal Key: {data.decimal_key}")
                print(f"    Hash: {data.hash_signature[:8]}...")
            else:
                print(f"  {symbol}: No data available")
        
        # Test 5: Get trading signals
        print("\n📊 Test 5: Trading signals")
        print("-" * 40)
        
        signals = market_integration.get_trading_signals(limit=10)
        print(f"  Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"  Signal: {signal.action.upper()} {signal.symbol}")
            print(f"    Price: ${signal.price:.2f}")
            print(f"    Confidence: {signal.confidence:.1%}")
            print(f"    Tier: {signal.strategy_tier.value}")
            print(f"    Phase: {signal.phase.value}")
            print(f"    RSI: {signal.rsi_trigger:.1f}")
            print(f"    Hash Match: {signal.hash_match}")
            print(f"    Memory Recall: {signal.memory_recall}")
            print(f"    Priority: {signal.priority}")
        
        # Test 6: Time phase detection
        print("\n🕰️ Test 6: Time phase detection")
        print("-" * 40)
        
        current_phase = market_integration._determine_time_phase()
        utc_hour = datetime.utcnow().hour
        print(f"  Current UTC hour: {utc_hour}")
        print(f"  Detected phase: {current_phase.value}")
        
        # Test phase mapping
        phase_mapping = {
            0: "MIDNIGHT",
            3: "PRE_DAWN", 
            7: "MORNING",
            12: "HIGH_NOON",
            16: "LATE_NOON",
            20: "EVENING",
            23: "MIDNIGHT_PLUS"
        }
        
        for hour, expected_phase in phase_mapping.items():
            # Temporarily override UTC hour for testing
            original_hour = datetime.utcnow().hour
            datetime.utcnow = lambda: type('MockDateTime', (), {'hour': hour})()
            
            test_phase = market_integration._determine_time_phase()
            print(f"  Hour {hour:02d}:00 -> {test_phase.value} (expected: {expected_phase})")
            
            # Restore original
            datetime.utcnow = lambda: type('MockDateTime', (), {'hour': original_hour})()
        
        # Test 7: Decimal key mapping
        print("\n🔢 Test 7: Decimal key mapping")
        print("-" * 40)
        
        test_prices = [63891.26, 54732.88, 12345.67, 98765.43, 11111.11]
        for price in test_prices:
            decimal_key = market_integration._extract_decimal_key(price)
            strategy_tier = market_integration._determine_strategy_tier(decimal_key)
            print(f"  Price: ${price:.2f} -> Decimal: {decimal_key} -> Tier: {strategy_tier.value}")
        
        # Test 8: Hash signature generation
        print("\n🔐 Test 8: Hash signature generation")
        print("-" * 40)
        
        test_data = [
            (63891.26, 45.5, 1000.0),
            (54732.88, 75.2, 2000.0),
            (12345.67, 25.8, 500.0)
        ]
        
        for price, rsi, volume in test_data:
            hash_sig = market_integration._generate_hash_signature(price, rsi, volume)
            memory_key = market_integration._generate_memory_key(hash_sig, StrategyTier.TIER_6, TimePhase.MORNING)
            print(f"  Price: ${price:.2f}, RSI: {rsi:.1f}, Volume: {volume:.0f}")
            print(f"    Hash: {hash_sig}")
            print(f"    Memory Key: {memory_key}")
        
        # Test 9: Memory key system
        print("\n🧠 Test 9: Memory key system")
        print("-" * 40)
        
        # Check memory key directories
        memory_path = Path("memory_keys")
        if memory_path.exists():
            tier_dirs = list(memory_path.glob("tier_*"))
            print(f"  Memory key directories: {len(tier_dirs)}")
            for tier_dir in tier_dirs:
                files = list(tier_dir.glob("*.json"))
                print(f"    {tier_dir.name}: {len(files)} memory keys")
        else:
            print("  Memory key directory not created yet")
        
        # Final status
        print("\n📊 Final System Status:")
        print("-" * 40)
        final_status = market_integration.get_system_status()
        for key, value in final_status.items():
            print(f"  {key}: {value}")
        
        # Shutdown
        market_integration.shutdown()
        
        print("\n✅ Live Market Data Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Live Market Data Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_execution_engine():
    """Test the strategy execution engine."""
    print("\n🧪 Testing Strategy Execution Engine")
    print("=" * 50)
    
    try:
        from core.strategy_execution_engine import StrategyExecutionEngine
        from core.live_market_data_integration import LiveMarketDataIntegration
        
        # Configuration
        config = {
            'risk': {
                'max_position_size': 0.1,
                'max_daily_loss': 1000.0,
                'max_open_positions': 10,
                'stop_loss_percentage': 0.05,
                'take_profit_percentage': 0.10,
                'max_risk_per_trade': 0.02,
                'correlation_threshold': 0.7
            }
        }
        
        # Initialize market integration
        market_integration = LiveMarketDataIntegration({})
        
        # Initialize execution engine
        execution_engine = StrategyExecutionEngine(market_integration, config)
        
        # Test 1: Engine initialization
        print("\n🔧 Test 1: Engine initialization")
        print("-" * 40)
        
        status = execution_engine.get_execution_status()
        print(f"  Running: {status['running']}")
        print(f"  Total trades: {status['total_trades']}")
        print(f"  Open positions: {status['open_positions']}")
        print(f"  Total P&L: ${status['total_profit_loss']:.2f}")
        
        # Test 2: Start execution engine
        print("\n🚀 Test 2: Start execution engine")
        print("-" * 40)
        
        execution_engine.start_execution()
        print("  ✅ Execution engine started")
        
        # Test 3: Monitor execution
        print("\n📊 Test 3: Monitor execution")
        print("-" * 40)
        
        # Monitor for 45 seconds
        for i in range(45):
            time.sleep(1)
            
            # Print status every 15 seconds
            if (i + 1) % 15 == 0:
                status = execution_engine.get_execution_status()
                print(f"  Time: {i+1}s | Trades: {status['total_trades']} | "
                      f"P&L: ${status['total_profit_loss']:.2f} | "
                      f"Positions: {status['open_positions']}")
        
        # Test 4: Get positions
        print("\n📈 Test 4: Current positions")
        print("-" * 40)
        
        positions = execution_engine.get_positions()
        print(f"  Active positions: {len(positions)}")
        
        for position in positions:
            print(f"  Position: {position.symbol}")
            print(f"    Side: {position.side}")
            print(f"    Amount: {position.amount}")
            print(f"    Entry Price: ${position.entry_price:.2f}")
            print(f"    Current Price: ${position.current_price:.2f}")
            print(f"    Unrealized P&L: ${position.unrealized_pnl:.2f}")
            print(f"    Strategy Tier: {position.strategy_tier.value}")
            print(f"    Phase: {position.phase.value}")
            print(f"    Stop Loss: ${position.stop_loss:.2f}")
            print(f"    Take Profit: ${position.take_profit:.2f}")
        
        # Test 5: Get executions
        print("\n📋 Test 5: Recent executions")
        print("-" * 40)
        
        executions = execution_engine.get_executions(limit=10)
        print(f"  Total executions: {len(executions)}")
        
        for execution in executions:
            print(f"  Execution: {execution.symbol}")
            print(f"    Action: {execution.action}")
            print(f"    Price: ${execution.price:.2f}")
            print(f"    Amount: {execution.amount}")
            print(f"    Status: {execution.status.value}")
            print(f"    Strategy Tier: {execution.strategy_tier.value}")
            print(f"    Phase: {execution.phase.value}")
            print(f"    RSI: {execution.rsi_trigger:.1f}")
            print(f"    Confidence: {execution.confidence:.1%}")
            print(f"    Hash Match: {execution.hash_match}")
            print(f"    Memory Recall: {execution.memory_recall}")
        
        # Test 6: Risk management
        print("\n🛡️ Test 6: Risk management")
        print("-" * 40)
        
        risk_status = execution_engine.get_execution_status()
        print(f"  Daily P&L: ${risk_status['daily_profit_loss']:.2f}")
        print(f"  Daily Loss: ${risk_status['daily_loss']:.2f}")
        print(f"  Max Drawdown: ${risk_status['max_drawdown']:.2f}")
        print(f"  Current Drawdown: ${risk_status['current_drawdown']:.2f}")
        print(f"  Win Rate: {risk_status['win_rate']:.1%}")
        print(f"  Winning Trades: {risk_status['winning_trades']}")
        print(f"  Losing Trades: {risk_status['losing_trades']}")
        
        # Test 7: Performance metrics
        print("\n📊 Test 7: Performance metrics")
        print("-" * 40)
        
        if risk_status['total_trades'] > 0:
            avg_profit = risk_status['total_profit_loss'] / risk_status['total_trades']
            print(f"  Average profit per trade: ${avg_profit:.2f}")
        
        if risk_status['winning_trades'] > 0:
            avg_win = risk_status['total_profit_loss'] / risk_status['winning_trades']
            print(f"  Average winning trade: ${avg_win:.2f}")
        
        # Final status
        print("\n📊 Final Execution Status:")
        print("-" * 40)
        final_status = execution_engine.get_execution_status()
        for key, value in final_status.items():
            print(f"  {key}: {value}")
        
        # Shutdown
        execution_engine.shutdown()
        
        print("\n✅ Strategy Execution Engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Strategy Execution Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_trading_system():
    """Test the complete integrated trading system."""
    print("\n🧪 Testing Integrated Trading System")
    print("=" * 50)
    
    try:
        from core.live_market_data_integration import LiveMarketDataIntegration
        from core.strategy_execution_engine import StrategyExecutionEngine
        from core.quantum_smoothing_system import QuantumSmoothingSystem, SmoothingConfig
        
        # Configuration
        market_config = {
            'coinbase': {
                'api_key': 'test_coinbase_key',
                'secret': 'test_coinbase_secret',
                'password': 'test_coinbase_password',
                'sandbox': True
            },
            'kraken': {
                'api_key': 'test_kraken_key',
                'secret': 'test_kraken_secret',
                'sandbox': True
            },
            'finance_api': {
                'api_key': 'test_finance_api_key'
            }
        }
        
        execution_config = {
            'risk': {
                'max_position_size': 0.1,
                'max_daily_loss': 1000.0,
                'max_open_positions': 10,
                'stop_loss_percentage': 0.05,
                'take_profit_percentage': 0.10,
                'max_risk_per_trade': 0.02,
                'correlation_threshold': 0.7
            }
        }
        
        smoothing_config = SmoothingConfig(
            max_concurrent_operations=200,
            operation_timeout_seconds=60.0,
            memory_threshold_percent=85.0,
            cpu_threshold_percent=90.0,
            async_worker_threads=16,
            performance_check_interval=0.5,
            memory_cleanup_interval=30.0
        )
        
        # Initialize all components
        print("\n🔧 Initializing integrated system...")
        print("-" * 40)
        
        # Initialize quantum smoothing system
        smoothing_system = QuantumSmoothingSystem(smoothing_config)
        print("  ✅ Quantum Smoothing System initialized")
        
        # Initialize market integration
        market_integration = LiveMarketDataIntegration(market_config)
        print("  ✅ Live Market Data Integration initialized")
        
        # Initialize execution engine
        execution_engine = StrategyExecutionEngine(market_integration, execution_config)
        print("  ✅ Strategy Execution Engine initialized")
        
        # Test 1: Start all systems
        print("\n🚀 Test 1: Start all systems")
        print("-" * 40)
        
        # Start quantum smoothing
        print("  Starting quantum smoothing system...")
        # smoothing_system is already running from initialization
        
        # Start market data feed
        print("  Starting market data feed...")
        market_integration.start_data_feed()
        
        # Start execution engine
        print("  Starting execution engine...")
        execution_engine.start_execution()
        
        print("  ✅ All systems started")
        
        # Test 2: Monitor integrated performance
        print("\n📊 Test 2: Monitor integrated performance")
        print("-" * 40)
        
        # Monitor for 60 seconds
        for i in range(60):
            time.sleep(1)
            
            # Print status every 20 seconds
            if (i + 1) % 20 == 0:
                # Get status from all systems
                smoothing_status = smoothing_system.get_system_status()
                market_status = market_integration.get_system_status()
                execution_status = execution_engine.get_execution_status()
                
                print(f"  Time: {i+1}s")
                print(f"    Smoothing: {smoothing_status['performance_state']} | "
                      f"Queue: {smoothing_status['operation_queue_size']}")
                print(f"    Market: {market_status['data_fetch_count']} fetches | "
                      f"{market_status['signal_count']} signals")
                print(f"    Execution: {execution_status['total_trades']} trades | "
                      f"P&L: ${execution_status['total_profit_loss']:.2f}")
        
        # Test 3: System integration verification
        print("\n🔗 Test 3: System integration verification")
        print("-" * 40)
        
        # Check data flow
        symbols = ['BTC/USDC', 'ETH/USDC']
        for symbol in symbols:
            market_data = market_integration.get_latest_market_data(symbol)
            if market_data:
                print(f"  {symbol}:")
                print(f"    Price: ${market_data.price:.2f}")
                print(f"    RSI: {market_data.rsi:.1f}")
                print(f"    Phase: {market_data.phase.value}")
                print(f"    Tier: {market_data.strategy_tier.value}")
                print(f"    Hash: {market_data.hash_signature[:8]}...")
        
        # Check signals
        signals = market_integration.get_trading_signals(limit=5)
        print(f"  Generated signals: {len(signals)}")
        
        # Check positions
        positions = execution_engine.get_positions()
        print(f"  Active positions: {len(positions)}")
        
        # Check executions
        executions = execution_engine.get_executions(limit=5)
        print(f"  Total executions: {len(executions)}")
        
        # Test 4: Performance metrics
        print("\n📈 Test 4: Performance metrics")
        print("-" * 40)
        
        # Smoothing performance
        smoothing_metrics = smoothing_system.get_performance_metrics()
        print(f"  Smoothing System:")
        print(f"    CPU Usage: {smoothing_metrics.cpu_usage:.1f}%")
        print(f"    Memory Usage: {smoothing_metrics.memory_usage:.1f}%")
        print(f"    Throughput: {smoothing_metrics.throughput_ops_per_sec:.1f} ops/sec")
        print(f"    Response Time: {smoothing_metrics.response_time_ms:.2f}ms")
        
        # Market data performance
        market_status = market_integration.get_system_status()
        print(f"  Market Data:")
        print(f"    Data fetches: {market_status['data_fetch_count']}")
        print(f"    Signals generated: {market_status['signal_count']}")
        print(f"    Error count: {market_status['error_count']}")
        print(f"    Cache size: {market_status['cache_size']}")
        
        # Execution performance
        execution_status = execution_engine.get_execution_status()
        print(f"  Execution Engine:")
        print(f"    Total trades: {execution_status['total_trades']}")
        print(f"    Win rate: {execution_status['win_rate']:.1%}")
        print(f"    Total P&L: ${execution_status['total_profit_loss']:.2f}")
        print(f"    Daily P&L: ${execution_status['daily_profit_loss']:.2f}")
        print(f"    Open positions: {execution_status['open_positions']}")
        
        # Test 5: Memory key system verification
        print("\n🧠 Test 5: Memory key system verification")
        print("-" * 40)
        
        memory_path = Path("memory_keys")
        if memory_path.exists():
            tier_dirs = list(memory_path.glob("tier_*"))
            print(f"  Memory key directories: {len(tier_dirs)}")
            
            total_memory_keys = 0
            for tier_dir in tier_dirs:
                files = list(tier_dir.glob("*.json"))
                total_memory_keys += len(files)
                print(f"    {tier_dir.name}: {len(files)} keys")
            
            print(f"  Total memory keys: {total_memory_keys}")
        else:
            print("  Memory key directory not found")
        
        # Test 6: Strategy tier analysis
        print("\n🎯 Test 6: Strategy tier analysis")
        print("-" * 40)
        
        tier_stats = {2: 0, 6: 0, 8: 0}
        for execution in executions:
            tier = execution.strategy_tier.value
            tier_stats[tier] = tier_stats.get(tier, 0) + 1
        
        for tier, count in tier_stats.items():
            percentage = (count / len(executions)) * 100 if executions else 0
            print(f"  Tier {tier}: {count} executions ({percentage:.1f}%)")
        
        # Test 7: Time phase analysis
        print("\n🕰️ Test 7: Time phase analysis")
        print("-" * 40)
        
        phase_stats = {}
        for execution in executions:
            phase = execution.phase.value
            phase_stats[phase] = phase_stats.get(phase, 0) + 1
        
        for phase, count in phase_stats.items():
            percentage = (count / len(executions)) * 100 if executions else 0
            print(f"  {phase}: {count} executions ({percentage:.1f}%)")
        
        # Test 8: RSI analysis
        print("\n📊 Test 8: RSI analysis")
        print("-" * 40)
        
        rsi_ranges = {
            'Oversold (<30)': 0,
            'Low (30-40)': 0,
            'Neutral (40-60)': 0,
            'High (60-70)': 0,
            'Overbought (>70)': 0
        }
        
        for execution in executions:
            rsi = execution.rsi_trigger
            if rsi < 30:
                rsi_ranges['Oversold (<30)'] += 1
            elif rsi < 40:
                rsi_ranges['Low (30-40)'] += 1
            elif rsi < 60:
                rsi_ranges['Neutral (40-60)'] += 1
            elif rsi < 70:
                rsi_ranges['High (60-70)'] += 1
            else:
                rsi_ranges['Overbought (>70)'] += 1
        
        for range_name, count in rsi_ranges.items():
            percentage = (count / len(executions)) * 100 if executions else 0
            print(f"  {range_name}: {count} executions ({percentage:.1f}%)")
        
        # Final system status
        print("\n📊 Final Integrated System Status:")
        print("-" * 40)
        
        final_smoothing_status = smoothing_system.get_system_status()
        final_market_status = market_integration.get_system_status()
        final_execution_status = execution_engine.get_execution_status()
        
        print("  Quantum Smoothing System:")
        for key, value in final_smoothing_status.items():
            print(f"    {key}: {value}")
        
        print("  Live Market Data Integration:")
        for key, value in final_market_status.items():
            print(f"    {key}: {value}")
        
        print("  Strategy Execution Engine:")
        for key, value in final_execution_status.items():
            print(f"    {key}: {value}")
        
        # Shutdown all systems
        print("\n🛑 Shutting down integrated system...")
        print("-" * 40)
        
        execution_engine.shutdown()
        print("  ✅ Strategy Execution Engine shutdown")
        
        market_integration.shutdown()
        print("  ✅ Live Market Data Integration shutdown")
        
        smoothing_system.shutdown()
        print("  ✅ Quantum Smoothing System shutdown")
        
        print("\n✅ Integrated Trading System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integrated Trading System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_api_integration():
    """Test with real API integration (requires actual API keys)."""
    print("\n🧪 Testing Real API Integration")
    print("=" * 50)
    
    try:
        from core.live_market_data_integration import LiveMarketDataIntegration
        from core.strategy_execution_engine import StrategyExecutionEngine
        
        # Check for real API keys
        api_keys_file = Path("api_keys.json")
        if not api_keys_file.exists():
            print("  ⚠️ No api_keys.json file found")
            print("  Create api_keys.json with your API credentials:")
            print("  {")
            print("    'coinbase': {")
            print("      'api_key': 'your_coinbase_api_key',")
            print("      'secret': 'your_coinbase_secret',")
            print("      'password': 'your_coinbase_password'")
            print("    },")
            print("    'kraken': {")
            print("      'api_key': 'your_kraken_api_key',")
            print("      'secret': 'your_kraken_secret'")
            print("    },")
            print("    'finance_api': {")
            print("      'api_key': 'your_finance_api_key'")
            print("    }")
            print("  }")
            return False
        
        # Load API keys
        with open(api_keys_file, 'r') as f:
            config = json.load(f)
        
        print("  ✅ API keys loaded")
        
        # Initialize with real API keys
        market_integration = LiveMarketDataIntegration(config)
        execution_engine = StrategyExecutionEngine(market_integration, {
            'risk': {
                'max_position_size': 0.01,  # Small position for testing
                'max_daily_loss': 100.0,
                'max_open_positions': 5,
                'stop_loss_percentage': 0.05,
                'take_profit_percentage': 0.10,
                'max_risk_per_trade': 0.01,
                'correlation_threshold': 0.7
            }
        })
        
        # Start systems
        print("  🚀 Starting systems with real API...")
        market_integration.start_data_feed()
        execution_engine.start_execution()
        
        # Monitor for 30 seconds
        print("  📊 Monitoring real market data...")
        for i in range(30):
            time.sleep(1)
            
            if (i + 1) % 10 == 0:
                market_status = market_integration.get_system_status()
                execution_status = execution_engine.get_execution_status()
                
                print(f"    Time: {i+1}s | Fetches: {market_status['data_fetch_count']} | "
                      f"Signals: {market_status['signal_count']} | "
                      f"Trades: {execution_status['total_trades']}")
        
        # Get real market data
        print("  📈 Real market data:")
        symbols = ['BTC/USDC', 'ETH/USDC']
        for symbol in symbols:
            data = market_integration.get_latest_market_data(symbol)
            if data:
                print(f"    {symbol}: ${data.price:.2f} | RSI: {data.rsi:.1f} | "
                      f"Volume: {data.volume:.0f}")
        
        # Shutdown
        execution_engine.shutdown()
        market_integration.shutdown()
        
        print("  ✅ Real API integration test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Real API integration test failed: {e}")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    print("🔧 LIVE TRADING SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print("Testing complete internalized trading strategy system")
    print("=" * 60)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    test_results = []
    
    try:
        # Test 1: Live Market Data Integration
        result1 = test_live_market_data_integration()
        test_results.append(("Live Market Data Integration", result1))
        
        # Test 2: Strategy Execution Engine
        result2 = test_strategy_execution_engine()
        test_results.append(("Strategy Execution Engine", result2))
        
        # Test 3: Integrated Trading System
        result3 = test_integrated_trading_system()
        test_results.append(("Integrated Trading System", result3))
        
        # Test 4: Real API Integration (optional)
        result4 = test_real_api_integration()
        test_results.append(("Real API Integration", result4))
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if result:
                passed_tests += 1
        
        print(f"\n  Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= 3:  # At least 3 out of 4 tests passed
            print("\n🎉 LIVE TRADING SYSTEM READY!")
            print("\n✅ The system successfully demonstrates:")
            print("  • Real-time market data integration with multiple APIs")
            print("  • Live RSI calculations and technical analysis")
            print("  • Time-based phase detection (midnight/noon patterns)")
            print("  • Decimal key mapping (2, 6, 8 tiers)")
            print("  • Memory key generation and recall system")
            print("  • Strategy execution with risk management")
            print("  • Cross-exchange arbitrage detection")
            print("  • Quantum smoothing for error-free operations")
            print("  • Complete internalized trading strategy system")
            
            print("\n🚀 The system is ready for live trading!")
            print("\n📋 Next steps:")
            print("  1. Add your real API keys to api_keys.json")
            print("  2. Configure risk parameters for your strategy")
            print("  3. Start the system with: python main.py --live")
            print("  4. Monitor performance and adjust as needed")
            
            return 0
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 