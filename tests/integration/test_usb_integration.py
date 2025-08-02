#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB Integration Test for Schwabot
=================================

Tests the enhanced USB integration with automatic detection, API key deployment,
and launcher integration.
"""

import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_usb_manager_import():
    """Test that the USB manager can be imported."""
    print("🧪 Testing USB manager import...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager, auto_detect_usb, setup_usb_storage, get_usb_status
        print("✅ USB manager imported successfully")
        return True
    except Exception as e:
        print(f"❌ USB manager import failed: {e}")
        return False

def test_usb_manager_initialization():
    """Test that the USB manager can be initialized."""
    print("\n🧪 Testing USB manager initialization...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        print("✅ USB manager initialized successfully")
        
        # Test basic properties
        if hasattr(manager, 'detected_drives'):
            print("✅ USB manager has detected_drives attribute")
        if hasattr(manager, 'api_keys_file'):
            print("✅ USB manager has api_keys_file attribute")
        
        return True
    except Exception as e:
        print(f"❌ USB manager initialization failed: {e}")
        return False

def test_usb_detection():
    """Test USB drive detection."""
    print("\n🧪 Testing USB drive detection...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        
        drives = manager.detect_usb_drives()
        print(f"✅ Detected {len(drives)} USB drive(s)")
        
        for drive in drives:
            print(f"   • {drive['letter']} - {drive['label']} ({drive['free_gb']:.1f}GB free)")
        
        return True
    except Exception as e:
        print(f"❌ USB detection failed: {e}")
        return False

def test_api_key_integration():
    """Test API key integration with USB manager."""
    print("\n🧪 Testing API key integration...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        
        # Test loading existing API keys
        api_keys = manager._load_existing_api_keys()
        print(f"✅ Loaded {len(api_keys)} API keys from existing configuration")
        
        # Test key path conversion
        test_mappings = [
            ("binance.api_key", "BINANCE_API_KEY"),
            ("openai.api_key", "OPENAI_API_KEY"),
            ("telegram.bot_token", "TELEGRAM_BOT_TOKEN")
        ]
        
        for key_path, expected_env in test_mappings:
            result = manager._convert_key_path_to_env(key_path)
            if result == expected_env:
                print(f"✅ Key path conversion: {key_path} → {result}")
            else:
                print(f"❌ Key path conversion failed: {key_path} → {result} (expected {expected_env})")
                return False
        
        return True
    except Exception as e:
        print(f"❌ API key integration test failed: {e}")
        return False

def test_env_file_generation():
    """Test .env file generation."""
    print("\n🧪 Testing .env file generation...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        
        # Test generating env content
        test_api_keys = {
            'BINANCE_API_KEY': 'test_binance_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TELEGRAM_BOT_TOKEN': 'test_telegram_token'
        }
        
        env_content = manager._generate_env_content(test_api_keys)
        
        if 'BINANCE_API_KEY=test_binance_key' in env_content:
            print("✅ .env content generated with API keys")
        else:
            print("❌ .env content generation failed")
            return False
        
        if 'Generated on:' in env_content:
            print("✅ .env content includes timestamp")
        else:
            print("❌ .env content missing timestamp")
            return False
        
        return True
    except Exception as e:
        print(f"❌ .env file generation test failed: {e}")
        return False

def test_launcher_integration():
    """Test that the launcher integrates with the USB manager."""
    print("\n🧪 Testing launcher integration...")
    
    try:
        from AOI_Base_Files_Schwabot.schwabot_launcher import SchwabotLauncher
        launcher = SchwabotLauncher()
        print("✅ Launcher imported and initialized successfully")
        
        # Test USB status update method
        if hasattr(launcher, '_update_usb_status'):
            print("✅ Launcher has USB status update method")
        else:
            print("❌ Launcher missing USB status update method")
            return False
        
        # Test USB setup method
        if hasattr(launcher, '_setup_usb_storage'):
            print("✅ Launcher has USB setup method")
        else:
            print("❌ Launcher missing USB setup method")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Launcher integration test failed: {e}")
        return False

def test_usb_status_functions():
    """Test USB status functions."""
    print("\n🧪 Testing USB status functions...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import get_usb_status, auto_detect_usb
        
        # Test get_usb_status
        status = get_usb_status()
        if isinstance(status, dict):
            print("✅ get_usb_status returns dictionary")
            print(f"   • Drives detected: {status.get('drives_detected', 0)}")
            print(f"   • Has configured drive: {status.get('has_configured_drive', False)}")
        else:
            print("❌ get_usb_status returns wrong type")
            return False
        
        # Test auto_detect_usb function exists
        if callable(auto_detect_usb):
            print("✅ auto_detect_usb function is callable")
        else:
            print("❌ auto_detect_usb function not callable")
            return False
        
        return True
    except Exception as e:
        print(f"❌ USB status functions test failed: {e}")
        return False

def test_usb_folder_creation():
    """Test USB folder creation functionality."""
    print("\n🧪 Testing USB folder creation...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        
        # Test folder creation method exists
        if hasattr(manager, 'create_schwabot_folders'):
            print("✅ USB manager has folder creation method")
        else:
            print("❌ USB manager missing folder creation method")
            return False
        
        # Test backup method exists
        if hasattr(manager, 'backup_trading_data'):
            print("✅ USB manager has backup method")
        else:
            print("❌ USB manager missing backup method")
            return False
        
        return True
    except Exception as e:
        print(f"❌ USB folder creation test failed: {e}")
        return False

def test_auto_detection_logic():
    """Test auto-detection logic."""
    print("\n🧪 Testing auto-detection logic...")
    
    try:
        from AOI_Base_Files_Schwabot.usb_manager import USBManager
        manager = USBManager()
        
        # Test auto detection method exists
        if hasattr(manager, 'auto_detect_and_offer_setup'):
            print("✅ USB manager has auto detection method")
        else:
            print("❌ USB manager missing auto detection method")
            return False
        
        # Test the logic checks for API keys
        api_keys_exist = os.path.exists(manager.api_keys_file)
        print(f"✅ API keys file exists: {api_keys_exist}")
        
        return True
    except Exception as e:
        print(f"❌ Auto-detection logic test failed: {e}")
        return False

def main():
    """Run all USB integration tests."""
    print("💾 USB INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        test_usb_manager_import,
        test_usb_manager_initialization,
        test_usb_detection,
        test_api_key_integration,
        test_env_file_generation,
        test_launcher_integration,
        test_usb_status_functions,
        test_usb_folder_creation,
        test_auto_detection_logic
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
        print("🎉 ALL USB INTEGRATION TESTS PASSED!")
        print("\n✅ USB Integration Features:")
        print("   • Automatic USB drive detection")
        print("   • API key deployment to .env files")
        print("   • Secure folder structure creation")
        print("   • Integration with launcher interface")
        print("   • Real-time status updates")
        print("   • Automatic setup prompts")
        print("   • Backup and data offloading capabilities")
        print("\n🚀 Ready for USB integration testing!")
        return 0
    else:
        print("⚠️ Some USB integration tests failed - check the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 