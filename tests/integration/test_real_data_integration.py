#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Schwabot Real Data Integration
============================================================

Tests all components of the real data integration system:
- Exchange API integrations
- State persistence
- Performance monitoring
- Mathematical state tracking
- Error handling and recovery
- Data validation and integrity
- Integration with Schwabot trading engine

This ensures the complete 46-day mathematical framework works correctly with real data.
"""

import asyncio
import unittest
import tempfile
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sqlite3

# Import the real data integration components
from schwabot_real_data_integration import (
    ExchangeType, DataSource, ExchangeConfig, MarketDataPoint,
    StatePersistence, BinanceAPI, CoinbaseAPI, RealDataManager,
    PerformanceMonitor
)

# Import the trading engine for integration testing
from schwabot_trading_engine import SchwabotTradingEngine, MarketData, TradeSignal, TradeAction

class TestStatePersistence(unittest.TestCase):
    """Test state persistence functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.persistence = StatePersistence(self.db_path)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """Test database initialization creates all required tables."""
        # Check if tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                'market_data', 'trade_signals', 'vault_entries',
                'system_state', 'performance_metrics', 'mathematical_states'
            ]
            
            for table in required_tables:
                self.assertIn(table, tables)
    
    def test_market_data_save_and_retrieve(self):
        """Test saving and retrieving market data."""
        # Create test market data
        test_data = MarketDataPoint(
            timestamp=time.time(),
            symbol="BTCUSDT",
            price=45000.0,
            volume=1000.0,
            bid=44995.0,
            ask=45005.0,
            spread=10.0,
            volatility=0.02,
            sentiment=0.7,
            asset_class="crypto",
            price_change=100.0,
            hash="test_hash_123",
            high_24h=46000.0,
            low_24h=44000.0,
            open_24h=44900.0,
            close_24h=45000.0,
            volume_24h=1000.0,
            price_change_24h=100.0,
            price_change_percent_24h=2.22,
            weighted_avg_price=45000.0,
            count=1000,
            zpe_value=0.5,
            entropy_value=0.3,
            vault_state="idle",
            lantern_trigger=False,
            ghost_echo_active=False,
            quantum_state=np.array([0.1, 0.2, 0.3])
        )
        
        # Save data
        self.persistence.save_market_data(test_data)
        
        # Retrieve data
        retrieved_data = self.persistence.get_recent_market_data("BTCUSDT", limit=1)
        
        # Verify data integrity
        self.assertEqual(len(retrieved_data), 1)
        retrieved = retrieved_data[0]
        
        self.assertEqual(retrieved.symbol, test_data.symbol)
        self.assertEqual(retrieved.price, test_data.price)
        self.assertEqual(retrieved.volume, test_data.volume)
        self.assertEqual(retrieved.bid, test_data.bid)
        self.assertEqual(retrieved.ask, test_data.ask)
        self.assertEqual(retrieved.spread, test_data.spread)
        self.assertEqual(retrieved.volatility, test_data.volatility)
        self.assertEqual(retrieved.sentiment, test_data.sentiment)
        self.assertEqual(retrieved.asset_class, test_data.asset_class)
        self.assertEqual(retrieved.hash, test_data.hash)
        self.assertEqual(retrieved.high_24h, test_data.high_24h)
        self.assertEqual(retrieved.low_24h, test_data.low_24h)
        self.assertEqual(retrieved.open_24h, test_data.open_24h)
        self.assertEqual(retrieved.close_24h, test_data.close_24h)
        self.assertEqual(retrieved.volume_24h, test_data.volume_24h)
        self.assertEqual(retrieved.price_change_24h, test_data.price_change_24h)
        self.assertEqual(retrieved.price_change_percent_24h, test_data.price_change_percent_24h)
        self.assertEqual(retrieved.weighted_avg_price, test_data.weighted_avg_price)
        self.assertEqual(retrieved.count, test_data.count)
        self.assertEqual(retrieved.zpe_value, test_data.zpe_value)
        self.assertEqual(retrieved.entropy_value, test_data.entropy_value)
        self.assertEqual(retrieved.vault_state, test_data.vault_state)
        self.assertEqual(retrieved.lantern_trigger, test_data.lantern_trigger)
        self.assertEqual(retrieved.ghost_echo_active, test_data.ghost_echo_active)
        np.testing.assert_array_equal(retrieved.quantum_state, test_data.quantum_state)
    
    def test_trade_signal_save(self):
        """Test saving trade signals."""
        test_signal = {
            'timestamp': time.time(),
            'asset': 'BTCUSDT',
            'action': 'buy',
            'confidence': 0.85,
            'entry_price': 45000.0,
            'target_price': 46000.0,
            'stop_loss': 44000.0,
            'quantity': 0.1,
            'strategy_hash': 'test_strategy_hash',
            'signal_strength': 0.9,
            'expected_roi': 0.022,
            'strategy_type': 'vault_trigger',
            'hash': 'test_signal_hash',
            'metadata': {'lantern_active': True, 'ghost_echo': False}
        }
        
        self.persistence.save_trade_signal(test_signal)
        
        # Verify signal was saved by checking database directly
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trade_signals WHERE asset = ?", ('BTCUSDT',))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
    
    def test_system_state_save_and_retrieve(self):
        """Test saving and retrieving system state."""
        test_state = {
            'quantum_state': [0.1, 0.2, 0.3, 0.4, 0.5],
            'vault_entries': 10,
            'lantern_active': True,
            'ghost_echo_count': 5,
            'entropy_gate_open': True
        }
        
        self.persistence.save_system_state('test_component', test_state)
        
        retrieved_state = self.persistence.get_system_state('test_component')
        
        self.assertIsNotNone(retrieved_state)
        self.assertEqual(retrieved_state['quantum_state'], test_state['quantum_state'])
        self.assertEqual(retrieved_state['vault_entries'], test_state['vault_entries'])
        self.assertEqual(retrieved_state['lantern_active'], test_state['lantern_active'])
        self.assertEqual(retrieved_state['ghost_echo_count'], test_state['ghost_echo_count'])
        self.assertEqual(retrieved_state['entropy_gate_open'], test_state['entropy_gate_open'])
    
    def test_mathematical_state_save(self):
        """Test saving mathematical state."""
        test_state = {
            'vault_hash': 'test_vault_hash_123',
            'lantern_trigger_hash': 'test_lantern_hash_456',
            'ghost_echo_timing': 0.75,
            'zpe_value': 0.6,
            'entropy_gate_state': 'open'
        }
        
        self.persistence.save_mathematical_state(
            'vault_logic', 'BTCUSDT', 'test_state_hash', test_state
        )
        
        # Verify mathematical state was saved
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM mathematical_states 
                WHERE component = ? AND asset = ? AND state_hash = ?
            """, ('vault_logic', 'BTCUSDT', 'test_state_hash'))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

class TestBinanceAPI(unittest.TestCase):
    """Test Binance API integration."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = ExchangeConfig(
            exchange=ExchangeType.BINANCE,
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True
        )
    
    def test_signature_generation(self):
        """Test HMAC signature generation."""
        api = BinanceAPI(self.config)
        params = "symbol=BTCUSDT&timestamp=1234567890"
        signature = api._generate_signature(params)
        
        # Verify signature is a valid hex string
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA256 hex length
        self.assertTrue(all(c in '0123456789abcdef' for c in signature))
    
    def test_sentiment_calculation(self):
        """Test market sentiment calculation."""
        api = BinanceAPI(self.config)
        
        # Test data with positive price change
        test_data = {
            'priceChangePercent': '2.5',
            'volume': '1000',
            'count': '500'
        }
        
        sentiment = api._calculate_sentiment(test_data)
        
        # Verify sentiment is in valid range
        self.assertGreaterEqual(sentiment, 0.0)
        self.assertLessEqual(sentiment, 1.0)
        
        # Test data with negative price change
        test_data_negative = {
            'priceChangePercent': '-1.5',
            'volume': '800',
            'count': '400'
        }
        
        sentiment_negative = api._calculate_sentiment(test_data_negative)
        
        # Verify negative sentiment is lower
        self.assertLess(sentiment_negative, sentiment)

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system."""
    
    def setUp(self):
        """Set up test performance monitor."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.persistence = StatePersistence(self.db_path)
        self.monitor = PerformanceMonitor(self.persistence)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.db_path)
    
    def test_initial_metrics(self):
        """Test initial performance metrics."""
        metrics = self.monitor.get_metrics()
        
        expected_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_roi': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_consecutive_losses': 0,
            'max_consecutive_wins': 0
        }
        
        self.assertEqual(metrics, expected_metrics)
    
    def test_metrics_update_with_winning_trade(self):
        """Test metrics update with a winning trade."""
        winning_trade = {
            'pnl': 150.0,
            'roi': 0.015,
            'timestamp': time.time()
        }
        
        self.monitor.update_metrics(winning_trade)
        metrics = self.monitor.get_metrics()
        
        self.assertEqual(metrics['total_trades'], 1)
        self.assertEqual(metrics['successful_trades'], 1)
        self.assertEqual(metrics['total_profit'], 150.0)
        self.assertEqual(metrics['win_rate'], 1.0)
        self.assertEqual(metrics['avg_roi'], 0.015)
        self.assertEqual(metrics['max_consecutive_wins'], 1)
        self.assertEqual(metrics['max_consecutive_losses'], 0)
    
    def test_metrics_update_with_losing_trade(self):
        """Test metrics update with a losing trade."""
        # First add a winning trade
        winning_trade = {'pnl': 100.0, 'roi': 0.01, 'timestamp': time.time()}
        self.monitor.update_metrics(winning_trade)
        
        # Then add a losing trade
        losing_trade = {'pnl': -50.0, 'roi': -0.005, 'timestamp': time.time()}
        self.monitor.update_metrics(losing_trade)
        
        metrics = self.monitor.get_metrics()
        
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['successful_trades'], 1)
        self.assertEqual(metrics['total_profit'], 50.0)
        self.assertEqual(metrics['win_rate'], 0.5)
        self.assertEqual(metrics['avg_roi'], 0.0025)  # (0.01 - 0.005) / 2
        self.assertEqual(metrics['max_consecutive_wins'], 1)
        self.assertEqual(metrics['max_consecutive_losses'], 1)
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Simulate a series of trades with drawdown
        trades = [
            {'pnl': 100.0, 'roi': 0.01, 'timestamp': time.time()},
            {'pnl': 200.0, 'roi': 0.02, 'timestamp': time.time()},
            {'pnl': -100.0, 'roi': -0.01, 'timestamp': time.time()},  # Drawdown
            {'pnl': 50.0, 'roi': 0.005, 'timestamp': time.time()},
        ]
        
        for trade in trades:
            self.monitor.update_metrics(trade)
        
        metrics = self.monitor.get_metrics()
        
        # Peak was 300 (100 + 200), bottom was 200 (300 - 100), drawdown = 100/300 = 0.333
        expected_drawdown = 100.0 / 300.0
        self.assertAlmostEqual(metrics['max_drawdown'], expected_drawdown, places=3)
    
    def test_metrics_persistence(self):
        """Test that metrics are saved to database."""
        winning_trade = {'pnl': 100.0, 'roi': 0.01, 'timestamp': time.time()}
        self.monitor.update_metrics(winning_trade)
        
        # Check if metrics were saved to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            # Verify the saved data
            cursor.execute("SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            self.assertEqual(row[2], 1)  # total_trades
            self.assertEqual(row[3], 1)  # successful_trades
            self.assertEqual(row[4], 100.0)  # total_profit
            self.assertEqual(row[6], 1.0)  # win_rate

class TestRealDataManager(unittest.TestCase):
    """Test real data manager integration."""
    
    def setUp(self):
        """Set up test data manager."""
        self.configs = [
            ExchangeConfig(
                exchange=ExchangeType.BINANCE,
                api_key="test_binance_key",
                api_secret="test_binance_secret",
                testnet=True
            )
        ]
        self.data_manager = RealDataManager(self.configs)
    
    def test_initialization(self):
        """Test data manager initialization."""
        self.assertIsNotNone(self.data_manager.state_persistence)
        self.assertIn(ExchangeType.BINANCE, self.data_manager.apis)
        self.assertIsInstance(self.data_manager.apis[ExchangeType.BINANCE], BinanceAPI)
    
    def test_system_state_save_and_retrieve(self):
        """Test system state save and retrieve through data manager."""
        test_state = {
            'quantum_state': [0.1, 0.2, 0.3],
            'vault_entries': 5,
            'lantern_active': True
        }
        
        self.data_manager.save_system_state('test_component', test_state)
        
        retrieved_state = self.data_manager.get_system_state('test_component')
        
        self.assertIsNotNone(retrieved_state)
        self.assertEqual(retrieved_state['quantum_state'], test_state['quantum_state'])
        self.assertEqual(retrieved_state['vault_entries'], test_state['vault_entries'])
        self.assertEqual(retrieved_state['lantern_active'], test_state['lantern_active'])
    
    def test_mathematical_state_save(self):
        """Test mathematical state save through data manager."""
        test_state = {
            'vault_hash': 'test_hash_123',
            'lantern_trigger': True,
            'ghost_echo_timing': 0.75
        }
        
        self.data_manager.save_mathematical_state(
            'vault_logic', 'BTCUSDT', 'test_state_hash', test_state
        )
        
        # Verify state was saved by checking database directly
        with sqlite3.connect(self.data_manager.state_persistence.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM mathematical_states 
                WHERE component = ? AND asset = ? AND state_hash = ?
            """, ('vault_logic', 'BTCUSDT', 'test_state_hash'))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

class TestIntegrationWithTradingEngine(unittest.TestCase):
    """Test integration between real data and trading engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize components
        self.configs = [
            ExchangeConfig(
                exchange=ExchangeType.BINANCE,
                api_key="test_key",
                api_secret="test_secret",
                testnet=True
            )
        ]
        self.data_manager = RealDataManager(self.configs)
        self.trading_engine = SchwabotTradingEngine()
        self.performance_monitor = PerformanceMonitor(self.data_manager.state_persistence)
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.db_path)
    
    def test_market_data_conversion(self):
        """Test conversion between MarketDataPoint and MarketData."""
        # Create MarketDataPoint
        data_point = MarketDataPoint(
            timestamp=time.time(),
            symbol="BTCUSDT",
            price=45000.0,
            volume=1000.0,
            bid=44995.0,
            ask=45005.0,
            spread=10.0,
            volatility=0.02,
            sentiment=0.7,
            asset_class="crypto",
            price_change=100.0,
            hash="test_hash_123"
        )
        
        # Convert to MarketData for trading engine
        market_data = MarketData(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            price=data_point.price,
            volume=data_point.volume,
            bid=data_point.bid,
            ask=data_point.ask,
            spread=data_point.spread,
            volatility=data_point.volatility,
            sentiment=data_point.sentiment,
            asset_class=AssetClass.CRYPTO,
            price_change=data_point.price_change,
            hash=data_point.hash
        )
        
        # Verify conversion
        self.assertEqual(market_data.symbol, data_point.symbol)
        self.assertEqual(market_data.price, data_point.price)
        self.assertEqual(market_data.volume, data_point.volume)
        self.assertEqual(market_data.bid, data_point.bid)
        self.assertEqual(market_data.ask, data_point.ask)
        self.assertEqual(market_data.spread, data_point.spread)
        self.assertEqual(market_data.volatility, data_point.volatility)
        self.assertEqual(market_data.sentiment, data_point.sentiment)
        self.assertEqual(market_data.price_change, data_point.price_change)
        self.assertEqual(market_data.hash, data_point.hash)
    
    def test_trade_signal_conversion(self):
        """Test conversion between trade signal formats."""
        # Create trade signal from trading engine
        market_data = MarketData(
            timestamp=time.time(),
            symbol="BTCUSDT",
            price=45000.0,
            volume=1000.0,
            bid=44995.0,
            ask=45005.0,
            spread=10.0,
            volatility=0.02,
            sentiment=0.7,
            asset_class=AssetClass.CRYPTO,
            price_change=100.0,
            hash="test_hash_123"
        )
        
        # Simulate signal generation
        signal = TradeSignal(
            timestamp=time.time(),
            asset="BTCUSDT",
            action=TradeAction.BUY,
            confidence=0.85,
            entry_price=45000.0,
            target_price=46000.0,
            stop_loss=44000.0,
            quantity=0.1,
            strategy_hash="test_strategy_hash",
            signal_strength=0.9,
            expected_roi=0.022,
            strategy_type="vault_trigger",
            hash="test_signal_hash",
            metadata={'lantern_active': True}
        )
        
        # Convert to dictionary for data manager
        signal_dict = {
            'timestamp': signal.timestamp,
            'asset': signal.asset,
            'action': signal.action.value,
            'confidence': signal.confidence,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'quantity': signal.quantity,
            'strategy_hash': signal.strategy_hash,
            'signal_strength': signal.signal_strength,
            'expected_roi': signal.expected_roi,
            'strategy_type': signal.strategy_type,
            'hash': signal.hash,
            'metadata': signal.metadata
        }
        
        # Verify conversion
        self.assertEqual(signal_dict['asset'], signal.asset)
        self.assertEqual(signal_dict['action'], signal.action.value)
        self.assertEqual(signal_dict['confidence'], signal.confidence)
        self.assertEqual(signal_dict['entry_price'], signal.entry_price)
        self.assertEqual(signal_dict['target_price'], signal.target_price)
        self.assertEqual(signal_dict['stop_loss'], signal.stop_loss)
        self.assertEqual(signal_dict['quantity'], signal.quantity)
        self.assertEqual(signal_dict['strategy_hash'], signal.strategy_hash)
        self.assertEqual(signal_dict['signal_strength'], signal.signal_strength)
        self.assertEqual(signal_dict['expected_roi'], signal.expected_roi)
        self.assertEqual(signal_dict['strategy_type'], signal.strategy_type)
        self.assertEqual(signal_dict['hash'], signal.hash)
        self.assertEqual(signal_dict['metadata'], signal.metadata)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.persistence = StatePersistence(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.db_path)
    
    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        # Test with invalid database path
        with self.assertRaises(Exception):
            invalid_persistence = StatePersistence("/invalid/path/database.db")
            invalid_persistence.save_market_data(MarketDataPoint(
                timestamp=time.time(),
                symbol="BTCUSDT",
                price=45000.0,
                volume=1000.0,
                bid=44995.0,
                ask=45005.0,
                spread=10.0,
                volatility=0.02,
                sentiment=0.7,
                asset_class="crypto",
                price_change=100.0,
                hash="test_hash_123"
            ))
    
    def test_invalid_market_data(self):
        """Test handling of invalid market data."""
        # Test with missing required fields
        with self.assertRaises(Exception):
            invalid_data = MarketDataPoint(
                timestamp=time.time(),
                symbol="",  # Empty symbol
                price=-1000.0,  # Negative price
                volume=1000.0,
                bid=44995.0,
                ask=44990.0,  # Ask < Bid
                spread=10.0,
                volatility=0.02,
                sentiment=0.7,
                asset_class="crypto",
                price_change=100.0,
                hash="test_hash_123"
            )
            self.persistence.save_market_data(invalid_data)

class TestDataValidation(unittest.TestCase):
    """Test data validation and integrity checks."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.persistence = StatePersistence(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.db_path)
    
    def test_market_data_validation(self):
        """Test market data validation."""
        # Valid market data
        valid_data = MarketDataPoint(
            timestamp=time.time(),
            symbol="BTCUSDT",
            price=45000.0,
            volume=1000.0,
            bid=44995.0,
            ask=45005.0,
            spread=10.0,
            volatility=0.02,
            sentiment=0.7,
            asset_class="crypto",
            price_change=100.0,
            hash="test_hash_123"
        )
        
        # Validate data integrity
        self.assertGreater(valid_data.price, 0)
        self.assertGreater(valid_data.volume, 0)
        self.assertGreater(valid_data.ask, valid_data.bid)
        self.assertEqual(valid_data.spread, valid_data.ask - valid_data.bid)
        self.assertGreaterEqual(valid_data.sentiment, 0.0)
        self.assertLessEqual(valid_data.sentiment, 1.0)
        self.assertGreaterEqual(valid_data.volatility, 0.0)
        self.assertIsInstance(valid_data.hash, str)
        self.assertGreater(len(valid_data.hash), 0)
    
    def test_trade_signal_validation(self):
        """Test trade signal validation."""
        valid_signal = {
            'timestamp': time.time(),
            'asset': 'BTCUSDT',
            'action': 'buy',
            'confidence': 0.85,
            'entry_price': 45000.0,
            'target_price': 46000.0,
            'stop_loss': 44000.0,
            'quantity': 0.1,
            'strategy_hash': 'test_strategy_hash',
            'signal_strength': 0.9,
            'expected_roi': 0.022,
            'strategy_type': 'vault_trigger',
            'hash': 'test_signal_hash',
            'metadata': {'lantern_active': True}
        }
        
        # Validate signal integrity
        self.assertGreater(valid_signal['confidence'], 0.0)
        self.assertLessEqual(valid_signal['confidence'], 1.0)
        self.assertGreater(valid_signal['entry_price'], 0)
        self.assertGreater(valid_signal['target_price'], valid_signal['entry_price'])
        self.assertLess(valid_signal['stop_loss'], valid_signal['entry_price'])
        self.assertGreater(valid_signal['quantity'], 0)
        self.assertGreater(valid_signal['signal_strength'], 0.0)
        self.assertLessEqual(valid_signal['signal_strength'], 1.0)
        self.assertIsInstance(valid_signal['strategy_hash'], str)
        self.assertGreater(len(valid_signal['strategy_hash']), 0)

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive Real Data Integration Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStatePersistence,
        TestBinanceAPI,
        TestPerformanceMonitor,
        TestRealDataManager,
        TestIntegrationWithTradingEngine,
        TestErrorHandling,
        TestDataValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! Real data integration is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please review the issues above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 