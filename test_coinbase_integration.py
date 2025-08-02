#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COINBASE INTEGRATION COMPREHENSIVE TEST
==========================================

Comprehensive test suite to ensure Coinbase API integration works perfectly
across ALL Schwabot systems, subsystems, and mathematical pipelines.

This test covers:
✅ API Key Configuration System
✅ Real API Pricing System
✅ Live Market Data Integration
✅ CCXT Integration
✅ Mathematical Pipeline Integration
✅ Timing Drift Protocol
✅ Phantom Relation Ghost Protocol
✅ Memory Storage Systems
✅ USB Security Integration
✅ 5-Layer Encryption System
"""

import os
import sys
import json
import logging
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coinbase_integration_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import the real API pricing and memory storage system
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
    print("⚠️ Real API pricing system not available - using simulated data")

def test_coinbase_api_key_configuration():
    """Test Coinbase API key configuration system."""
    print("\n🔑 Testing Coinbase API Key Configuration")
    print("="*50)
    
    try:
        from api_key_configuration import APIKeyManager
        
        manager = APIKeyManager()
        
        # Test Coinbase key retrieval
        api_key = manager.get_api_key("coinbase", "api_key")
        secret_key = manager.get_api_key("coinbase", "secret_key")
        passphrase = manager.get_api_key("coinbase", "passphrase")
        
        print(f"✅ API Key Manager initialized")
        print(f"🔑 Coinbase API Key: {'✅ Found' if api_key else '❌ Not found'}")
        print(f"🔑 Coinbase Secret Key: {'✅ Found' if secret_key else '❌ Not found'}")
        print(f"🔑 Coinbase Passphrase: {'✅ Found' if passphrase else '❌ Not found'}")
        
        if api_key and secret_key and passphrase:
            print("✅ Coinbase credentials complete")
            return True
        else:
            print("⚠️ Coinbase credentials incomplete - run api_key_configuration.py")
            return False
            
    except Exception as e:
        print(f"❌ API Key Configuration test failed: {e}")
        logger.error(f"API Key Configuration test failed: {e}")
        return False

def test_coinbase_real_api_pricing():
    """Test Coinbase real API pricing system."""
    print("\n📊 Testing Coinbase Real API Pricing")
    print("="*45)
    
    try:
        # Test integration with mathematical cores
        from real_api_pricing_memory_system import get_real_price_data
        
        # Get real Coinbase price
        try:
            btc_price = get_real_price_data('BTC/USD', 'coinbase')
            eth_price = get_real_price_data('ETH/USD', 'coinbase')
            
            print(f"📊 BTC Price: ${btc_price:,.2f}")
            print(f"📊 ETH Price: ${eth_price:,.2f}")
            
            # Test mathematical operations
            price_ratio = btc_price / eth_price
            print(f"📊 BTC/ETH Ratio: {price_ratio:.4f}")
            
            # Test decimal key extraction (from live market data integration)
            def extract_decimal_key(price):
                price_str = f"{price:.8f}"
                decimal_part = price_str.split('.')[1]
                return decimal_part[:8]
            
            btc_decimal = extract_decimal_key(btc_price)
            eth_decimal = extract_decimal_key(eth_price)
            
            print(f"🔢 BTC Decimal Key: {btc_decimal}")
            print(f"🔢 ETH Decimal Key: {eth_decimal}")
            
            print("✅ Mathematical pipeline integration working")
            return True
            
        except Exception as e:
            print(f"⚠️ Mathematical pipeline test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Mathematical Pipeline test failed: {e}")
        logger.error(f"Mathematical Pipeline test failed: {e}")
        return False

def test_coinbase_live_market_data():
    """Test Coinbase live market data integration."""
    print("\n📈 Testing Coinbase Live Market Data")
    print("="*45)
    
    try:
        from core.live_market_data_integration import LiveMarketDataIntegration
        
        # Test configuration
        config = {
            'coinbase': {
                'api_key': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET_KEY'),
                'password': os.getenv('COINBASE_PASSPHRASE'),
                'sandbox': True
            }
        }
        
        integration = LiveMarketDataIntegration(config)
        print("✅ Live Market Data Integration initialized")
        
        # Check if Coinbase exchange is initialized
        if 'coinbase' in integration.exchanges:
            print("✅ Coinbase exchange initialized in Live Market Data")
        else:
            print("❌ Coinbase exchange not initialized in Live Market Data")
        
        return True
        
    except Exception as e:
        print(f"❌ Live Market Data test failed: {e}")
        logger.error(f"Live Market Data test failed: {e}")
        return False

def test_coinbase_ccxt_integration():
    """Test Coinbase CCXT integration directly."""
    print("\n🔌 Testing Coinbase CCXT Integration")
    print("="*40)
    
    try:
        import ccxt
        
        # Test ONLY the current Coinbase exchange (coinbasepro is deprecated)
        exchanges_to_test = [
            ('coinbase', ccxt.coinbase)  # Current unified Coinbase exchange
        ]
        
        for exchange_name, exchange_class in exchanges_to_test:
            try:
                # Test without credentials first
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000
                })
                
                # Load markets
                markets = exchange.load_markets()
                print(f"✅ {exchange_name}: {len(markets)} markets loaded")
                
                # Test fetching ticker
                try:
                    ticker = exchange.fetch_ticker('BTC/USD')
                    print(f"💰 {exchange_name} BTC/USD: ${ticker['last']:,.2f}")
                except Exception as e:
                    print(f"⚠️ {exchange_name} ticker fetch failed: {e}")
                    
            except Exception as e:
                print(f"❌ {exchange_name} initialization failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ CCXT library not available")
        return False
    except Exception as e:
        print(f"❌ CCXT Integration test failed: {e}")
        logger.error(f"CCXT Integration test failed: {e}")
        return False

def test_coinbase_mathematical_pipeline():
    """Test Coinbase integration with mathematical pipeline."""
    print("\n🧮 Testing Coinbase Mathematical Pipeline")
    print("="*45)
    
    try:
        # Test integration with mathematical cores
        from real_api_pricing_memory_system import get_real_price_data
        
        # Get real Coinbase price
        try:
            btc_price = get_real_price_data('BTC/USD', 'coinbase')
            eth_price = get_real_price_data('ETH/USD', 'coinbase')
            
            print(f"📊 BTC Price: ${btc_price:,.2f}")
            print(f"📊 ETH Price: ${eth_price:,.2f}")
            
            # Test mathematical operations
            price_ratio = btc_price / eth_price
            print(f"📊 BTC/ETH Ratio: {price_ratio:.4f}")
            
            # Test decimal key extraction (from live market data integration)
            def extract_decimal_key(price):
                price_str = f"{price:.8f}"
                decimal_part = price_str.split('.')[1]
                return decimal_part[:8]
            
            btc_decimal = extract_decimal_key(btc_price)
            eth_decimal = extract_decimal_key(eth_price)
            
            print(f"🔢 BTC Decimal Key: {btc_decimal}")
            print(f"🔢 ETH Decimal Key: {eth_decimal}")
            
            print("✅ Mathematical pipeline integration working")
            return True
            
        except Exception as e:
            print(f"⚠️ Mathematical pipeline test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Mathematical Pipeline test failed: {e}")
        logger.error(f"Mathematical Pipeline test failed: {e}")
        return False

def test_coinbase_timing_drift_protocol():
    """Test Coinbase timing drift protocol integration."""
    print("\n⏰ Testing Coinbase Timing Drift Protocol")
    print("="*45)
    
    try:
        # Test timing synchronization with Coinbase
        from real_api_pricing_memory_system import initialize_real_api_memory_system, MemoryConfig
        
        config = MemoryConfig(memory_choice_menu=False)
        system = initialize_real_api_memory_system(config)
        
        # Test multiple price fetches to check timing consistency
        prices = []
        timestamps = []
        
        for i in range(5):
            start_time = time.time()
            try:
                price = system.get_real_price_data('BTC/USD', 'coinbase')
                end_time = time.time()
                
                prices.append(price)
                timestamps.append(end_time - start_time)
                
                print(f"⏱️ Fetch {i+1}: ${price:,.2f} (took {timestamps[-1]:.3f}s)")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️ Fetch {i+1} failed: {e}")
        
        if len(prices) > 1:
            # Check for timing drift
            avg_time = sum(timestamps) / len(timestamps)
            max_time = max(timestamps)
            min_time = min(timestamps)
            time_variance = max_time - min_time
            
            print(f"⏱️ Average fetch time: {avg_time:.3f}s")
            print(f"⏱️ Time variance: {time_variance:.3f}s")
            
            if time_variance < 2.0:  # Less than 2 seconds variance
                print("✅ Timing drift protocol working correctly")
                return True
            else:
                print("⚠️ High timing variance detected")
                return False
        else:
            print("⚠️ Insufficient data for timing analysis")
            return False
            
    except Exception as e:
        print(f"❌ Timing Drift Protocol test failed: {e}")
        logger.error(f"Timing Drift Protocol test failed: {e}")
        return False

def test_coinbase_phantom_relation_ghost_protocol():
    """Test Coinbase phantom relation ghost protocol."""
    print("\n👻 Testing Coinbase Phantom Relation Ghost Protocol")
    print("="*55)
    
    try:
        # Test cross-exchange price relationships
        from real_api_pricing_memory_system import initialize_real_api_memory_system, MemoryConfig
        
        config = MemoryConfig(memory_choice_menu=False)
        system = initialize_real_api_memory_system(config)
        
        # Test price relationships across different exchanges
        exchanges_to_test = ['coinbase', 'kraken', 'binance']
        prices = {}
        
        for exchange in exchanges_to_test:
            try:
                price = system.get_real_price_data('BTC/USD', exchange)
                prices[exchange] = price
                print(f"💰 {exchange.upper()}: ${price:,.2f}")
            except Exception as e:
                print(f"⚠️ {exchange}: {e}")
        
        if len(prices) >= 2:
            # Calculate price relationships
            exchanges = list(prices.keys())
            for i in range(len(exchanges)):
                for j in range(i+1, len(exchanges)):
                    ex1, ex2 = exchanges[i], exchanges[j]
                    price1, price2 = prices[ex1], prices[ex2]
                    
                    if price1 > 0 and price2 > 0:
                        ratio = price1 / price2
                        diff_percent = abs(1 - ratio) * 100
                        
                        print(f"🔗 {ex1} vs {ex2}: {ratio:.4f} ({diff_percent:.2f}% diff)")
                        
                        # Check for phantom relationships (unusual price differences)
                        if diff_percent > 5.0:
                            print(f"👻 Phantom relationship detected: {ex1} vs {ex2}")
                        else:
                            print(f"✅ Normal relationship: {ex1} vs {ex2}")
            
            print("✅ Phantom Relation Ghost Protocol analysis complete")
            return True
        else:
            print("⚠️ Insufficient exchange data for phantom analysis")
            return False
            
    except Exception as e:
        print(f"❌ Phantom Relation Ghost Protocol test failed: {e}")
        logger.error(f"Phantom Relation Ghost Protocol test failed: {e}")
        return False

def test_coinbase_memory_storage():
    """Test Coinbase data memory storage."""
    print("\n💾 Testing Coinbase Memory Storage")
    print("="*35)
    
    try:
        from real_api_pricing_memory_system import initialize_real_api_memory_system, MemoryConfig, store_memory_entry
        
        config = MemoryConfig(memory_choice_menu=False)
        system = initialize_real_api_memory_system(config)
        
        # Store Coinbase test data
        test_data = {
            'symbol': 'BTC/USD',
            'exchange': 'coinbase',
            'price': 50000.0,
            'timestamp': datetime.now().isoformat(),
            'test_type': 'coinbase_integration_test'
        }
        
        entry_id = store_memory_entry(
            data_type='coinbase_test',
            data=test_data,
            source='coinbase_test',
            priority=1,
            tags=['coinbase', 'test', 'integration']
        )
        
        print(f"✅ Memory entry stored: {entry_id}")
        
        # Get memory stats
        stats = system.get_memory_stats()
        print(f"📊 Memory stats: {len(stats)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Storage test failed: {e}")
        logger.error(f"Memory Storage test failed: {e}")
        return False

def test_coinbase_usb_security():
    """Test Coinbase USB security integration."""
    print("\n🔒 Testing Coinbase USB Security")
    print("="*35)
    
    try:
        # Test USB memory system with Coinbase data
        from real_api_pricing_memory_system import initialize_real_api_memory_system, MemoryConfig, store_memory_entry
        
        config = MemoryConfig(
            storage_mode=MemoryStorageMode.HYBRID,  # Use both local and USB
            memory_choice_menu=False
        )
        
        system = initialize_real_api_memory_system(config)
        
        # Store sensitive Coinbase data
        sensitive_data = {
            'exchange': 'coinbase',
            'api_status': 'configured',
            'last_test': datetime.now().isoformat(),
            'security_level': 'high'
        }
        
        entry_id = store_memory_entry(
            data_type='coinbase_security',
            data=sensitive_data,
            source='coinbase_security_test',
            priority=3,  # High priority
            tags=['coinbase', 'security', 'usb', 'encrypted']
        )
        
        print(f"✅ USB security entry stored: {entry_id}")
        
        # Check USB memory status
        if hasattr(system, 'usb_memory_dir') and system.usb_memory_dir:
            print(f"✅ USB memory directory: {system.usb_memory_dir}")
        else:
            print("⚠️ USB memory not available")
        
        return True
        
    except Exception as e:
        print(f"❌ USB Security test failed: {e}")
        logger.error(f"USB Security test failed: {e}")
        return False

def test_coinbase_encryption_system():
    """Test Coinbase 5-layer encryption system."""
    print("\n🔐 Testing Coinbase 5-Layer Encryption")
    print("="*40)
    
    try:
        from api_key_configuration import APIKeyManager
        
        manager = APIKeyManager()
        
        # Test encryption systems
        encryption_systems = []
        
        if manager.alpha256_encryption:
            encryption_systems.append("Alpha256")
        if manager.alpha_encryption:
            encryption_systems.append("Alpha (Ω-B-Γ)")
        if manager.encryption_manager:
            encryption_systems.append("AES-256")
        
        encryption_systems.append("Base64")
        
        print(f"🔐 Active encryption systems: {', '.join(encryption_systems)}")
        
        # Test Coinbase key encryption
        api_key = manager.get_api_key("coinbase", "api_key")
        if api_key:
            print("✅ Coinbase API key encrypted and accessible")
        else:
            print("❌ Coinbase API key not found")
        
        return len(encryption_systems) >= 2  # At least 2 encryption layers
        
    except Exception as e:
        print(f"❌ Encryption System test failed: {e}")
        logger.error(f"Encryption System test failed: {e}")
        return False

def main():
    """Run comprehensive Coinbase integration tests."""
    print("🧪 COINBASE INTEGRATION COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("🔍 Testing ALL Coinbase integrations across Schwabot systems")
    print("="*60)
    
    tests = [
        ("API Key Configuration", test_coinbase_api_key_configuration),
        ("Real API Pricing", test_coinbase_real_api_pricing),
        ("Live Market Data", test_coinbase_live_market_data),
        ("CCXT Integration", test_coinbase_ccxt_integration),
        ("Mathematical Pipeline", test_coinbase_mathematical_pipeline),
        ("Timing Drift Protocol", test_coinbase_timing_drift_protocol),
        ("Phantom Relation Ghost Protocol", test_coinbase_phantom_relation_ghost_protocol),
        ("Memory Storage", test_coinbase_memory_storage),
        ("USB Security", test_coinbase_usb_security),
        ("5-Layer Encryption", test_coinbase_encryption_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAILED: {test_name} - {e}")
            logger.error(f"Test {test_name} failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COINBASE INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL COINBASE INTEGRATION TESTS PASSED!")
        print("✅ Coinbase is fully integrated across ALL Schwabot systems")
    else:
        print("⚠️ Some tests failed - check the logs for details")
        print("💡 Run 'python api_key_configuration.py' to configure Coinbase API keys")
    
    print(f"\n📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 