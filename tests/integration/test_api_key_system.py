#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Key Management System Test
=============================

Tests the new comprehensive API key management system to ensure it works correctly.
"""

import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_api_key_manager_import():
    """Test that the API key manager can be imported."""
    print("🧪 Testing API key manager import...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager, show_api_configuration
        print("✅ API key manager imported successfully")
        return True
    except Exception as e:
        print(f"❌ API key manager import failed: {e}")
        return False

def test_api_key_manager_initialization():
    """Test that the API key manager can be initialized."""
    print("\n🧪 Testing API key manager initialization...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        print("✅ API key manager initialized successfully")
        
        # Test configuration loading
        if manager.api_config:
            print(f"✅ API configuration loaded: {len(manager.api_config)} categories")
            
            # Check categories
            categories = list(manager.api_config.keys())
            print(f"✅ Categories: {categories}")
            
            # Check services
            total_services = sum(len(category) for category in manager.api_config.values())
            print(f"✅ Total services: {total_services}")
            
        return True
    except Exception as e:
        print(f"❌ API key manager initialization failed: {e}")
        return False

def test_api_key_encryption():
    """Test API key encryption and decryption."""
    print("\n🧪 Testing API key encryption...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        
        # Test encryption/decryption
        test_key = "test_api_key_12345"
        encrypted = manager._encrypt_key(test_key)
        decrypted = manager._decrypt_key(encrypted)
        
        if decrypted == test_key:
            print("✅ API key encryption/decryption working correctly")
        else:
            print("❌ API key encryption/decryption failed")
            return False
            
        return True
    except Exception as e:
        print(f"❌ API key encryption test failed: {e}")
        return False

def test_api_key_storage():
    """Test API key storage and retrieval."""
    print("\n🧪 Testing API key storage...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        
        # Test saving and loading keys
        test_keys = {
            "binance.api_key": "test_binance_key",
            "openai.api_key": "test_openai_key",
            "telegram.bot_token": "test_telegram_token"
        }
        
        # Save test keys
        for key_path, value in test_keys.items():
            manager.encrypted_keys[key_path] = manager._encrypt_key(value)
        
        manager._save_encrypted_keys()
        print("✅ API keys saved successfully")
        
        # Reload and verify
        manager2 = APIKeyManager()
        for key_path, expected_value in test_keys.items():
            actual_value = manager2.get_api_key(*key_path.split('.'))
            if actual_value == expected_value:
                print(f"✅ Key {key_path} retrieved correctly")
            else:
                print(f"❌ Key {key_path} retrieval failed")
                return False
        
        # Clean up test keys
        for key_path in test_keys.keys():
            if key_path in manager2.encrypted_keys:
                del manager2.encrypted_keys[key_path]
        manager2._save_encrypted_keys()
        print("✅ Test keys cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ API key storage test failed: {e}")
        return False

def test_api_key_validation():
    """Test API key validation methods."""
    print("\n🧪 Testing API key validation...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        
        # Test has_api_key method
        if not manager.has_api_key("binance", "api_key"):
            print("✅ has_api_key correctly returns False for missing key")
        else:
            print("❌ has_api_key incorrectly returns True for missing key")
            return False
        
        # Test with a real key
        manager.encrypted_keys["binance.api_key"] = manager._encrypt_key("test_key")
        if manager.has_api_key("binance", "api_key"):
            print("✅ has_api_key correctly returns True for existing key")
        else:
            print("❌ has_api_key incorrectly returns False for existing key")
            return False
        
        # Clean up
        del manager.encrypted_keys["binance.api_key"]
        
        return True
    except Exception as e:
        print(f"❌ API key validation test failed: {e}")
        return False

def test_launcher_integration():
    """Test that the launcher integrates with the API key manager."""
    print("\n🧪 Testing launcher integration...")
    
    try:
        from AOI_Base_Files_Schwabot.schwabot_launcher import SchwabotLauncher
        launcher = SchwabotLauncher()
        print("✅ Launcher imported and initialized successfully")
        
        # Test API status update
        launcher._update_api_status()
        print("✅ API status update method works")
        
        return True
    except Exception as e:
        print(f"❌ Launcher integration test failed: {e}")
        return False

def test_api_configuration_structure():
    """Test the API configuration structure."""
    print("\n🧪 Testing API configuration structure...")
    
    try:
        from AOI_Base_Files_Schwabot.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        
        # Check required categories
        required_categories = ["trading_exchanges", "ai_services", "data_services", "monitoring"]
        for category in required_categories:
            if category in manager.api_config:
                print(f"✅ Category '{category}' found")
            else:
                print(f"❌ Category '{category}' missing")
                return False
        
        # Check trading exchanges
        trading_exchanges = manager.api_config["trading_exchanges"]
        expected_exchanges = ["binance", "coinbase", "kraken"]
        for exchange in expected_exchanges:
            if exchange in trading_exchanges:
                config = trading_exchanges[exchange]
                if "name" in config and "description" in config and "instructions" in config:
                    print(f"✅ Exchange '{exchange}' properly configured")
                else:
                    print(f"❌ Exchange '{exchange}' missing required fields")
                    return False
            else:
                print(f"❌ Exchange '{exchange}' missing")
                return False
        
        # Check AI services
        ai_services = manager.api_config["ai_services"]
        expected_ai_services = ["openai", "anthropic", "google_gemini"]
        for service in expected_ai_services:
            if service in ai_services:
                config = ai_services[service]
                if "name" in config and "description" in config and "instructions" in config:
                    print(f"✅ AI service '{service}' properly configured")
                else:
                    print(f"❌ AI service '{service}' missing required fields")
                    return False
            else:
                print(f"❌ AI service '{service}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ API configuration structure test failed: {e}")
        return False

def main():
    """Run all API key management tests."""
    print("🔑 API KEY MANAGEMENT SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        test_api_key_manager_import,
        test_api_key_manager_initialization,
        test_api_key_encryption,
        test_api_key_storage,
        test_api_key_validation,
        test_launcher_integration,
        test_api_configuration_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL API KEY MANAGEMENT TESTS PASSED!")
        print("\n✅ API Key Management System Features:")
        print("   • Comprehensive API key configuration interface")
        print("   • Clear labeling and instructions for each service")
        print("   • Secure encryption and storage")
        print("   • Integration with launcher system")
        print("   • Support for trading exchanges, AI services, data providers, and notifications")
        print("   • User-friendly GUI with proper explanations")
        return 0
    else:
        print("⚠️ Some API key management tests failed - check the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 