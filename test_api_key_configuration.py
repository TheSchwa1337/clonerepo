#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 API Key Configuration Test Script
====================================

Test script to verify the API key configuration system is working properly.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_api_key_config.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_api_key_configuration():
    """Test the API key configuration system."""
    print("🧪 Testing API Key Configuration System")
    print("="*50)
    
    try:
        # Import the API key manager
        from api_key_configuration import APIKeyManager
        
        # Initialize the manager
        manager = APIKeyManager()
        
        print("✅ API Key Manager initialized successfully")
        
        # Test USB detection
        print(f"\n🔍 USB Detection: {'✅ Detected' if manager.usb_detected else '❌ Not detected'}")
        
        # Test configuration file
        print(f"📁 Config file: {manager.api_keys_file}")
        print(f"📁 Config exists: {'✅ Yes' if manager.api_keys_file.exists() else '❌ No'}")
        
        # Test loading configuration
        config = manager._load_config()
        print(f"📋 Config loaded: {'✅ Yes' if config else '❌ No'}")
        
        if config:
            print(f"🔑 Services configured: {len(config)}")
            for service in config.keys():
                print(f"   • {service}")
        
        # Test API key retrieval
        print("\n🔑 Testing API Key Retrieval:")
        test_services = ['binance', 'coinbase', 'kraken', 'openai']
        
        for service in test_services:
            api_key = manager.get_api_key(service, "api_key")
            has_key = manager.has_api_key(service, "api_key")
            
            status = "✅ Has key" if has_key else "❌ No key"
            print(f"   {service}: {status}")
        
        print("\n✅ API Key Configuration Test Completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure api_key_configuration.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        logger.error(f"Test failed: {e}")
        return False

def test_real_api_integration():
    """Test real API integration with configured keys."""
    print("\n🧪 Testing Real API Integration")
    print("="*40)
    
    try:
        # Initialize real API system
        from real_api_pricing_memory_system import initialize_real_api_memory_system, MemoryConfig, MemoryStorageMode, APIMode
        
        config = MemoryConfig(
            storage_mode=MemoryStorageMode.AUTO,
            api_mode=APIMode.REAL_API_ONLY,
            memory_choice_menu=False
        )
        
        system = initialize_real_api_memory_system(config)
        print("✅ Real API System initialized successfully")
        
        # Test getting real price data
        try:
            price = system.get_real_price_data('BTC/USDC', 'kraken')
            print(f"💰 BTC Price: ${price:,.2f}")
        except Exception as e:
            print(f"⚠️ Price test failed (expected without real keys): {e}")
        
        # Show configured exchanges
        api_keys = system._load_api_keys()
        if api_keys:
            print(f"\n🔑 API Keys loaded: {len(api_keys)}")
            for exchange in api_keys.keys():
                print(f"   • {exchange}: ✅ Configured")
        else:
            print("❌ No API keys found")
        
        print("✅ Real API Integration Test Completed")
        
    except Exception as e:
        print(f"❌ Real API integration test failed: {e}")
        logger.error(f"❌ Real API integration test failed: {e}")

def show_configuration_help():
    """Show help for configuring API keys."""
    print("\n💡 API Key Configuration Help")
    print("="*40)
    print("To configure API keys, run:")
    print("   python api_key_configuration.py")
    print("\nOr manually create a configuration file at:")
    print("   config/keys/api_keys.json")
    print("\nExample configuration format:")
    print("""
{
  "binance": {
    "api_key": "your_binance_api_key",
    "secret_key": "your_binance_secret_key",
    "_metadata": {
      "last_updated": "2025-01-17T20:16:07.843264",
      "encrypted": true
    }
  },
  "coinbase": {
    "api_key": "your_coinbase_api_key", 
    "secret_key": "your_coinbase_secret_key",
    "_metadata": {
      "last_updated": "2025-01-17T20:16:07.845262",
      "encrypted": true
    }
  }
}
""")

def main():
    """Main test function."""
    print("🧪 Schwabot API Key Configuration Test")
    print("="*50)
    
    # Test API key configuration
    config_test_passed = test_api_key_configuration()
    
    # Test real API integration
    integration_test_passed = test_real_api_integration()
    
    # Show results
    print("\n📊 Test Results")
    print("="*20)
    print(f"API Key Configuration: {'✅ PASSED' if config_test_passed else '❌ FAILED'}")
    print(f"Real API Integration: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if not config_test_passed or not integration_test_passed:
        show_configuration_help()
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    main() 