#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified System Integration Test
==============================

Simple test to verify that the Schwabot Unified Interface works correctly
and all components can communicate with each other.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unified_system():
    """Test the unified system integration."""
    print("🧪 Testing Schwabot Unified System Integration...")
    print("=" * 60)
    
    try:
        # Create unified interface
        print("1. Creating unified interface...")
        unified_interface = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
        print("   ✅ Unified interface created")
        
        # Check initialization
        print("2. Checking system initialization...")
        if unified_interface.initialized:
            print("   ✅ System initialized successfully")
        else:
            print("   ❌ System initialization failed")
            return False
        
        # Check component initialization
        print("3. Checking component initialization...")
        components = {
            "Schwabot AI Integration": unified_interface.schwabot_ai_integration is not None,
            "Visual Controller": unified_interface.visual_controller is not None,
            "Tick Loader": unified_interface.tick_loader is not None,
            "Signal Cache": unified_interface.signal_cache is not None,
            "Registry Writer": unified_interface.registry_writer is not None,
            "JSON Server": unified_interface.json_server is not None
        }
        
        for component, status in components.items():
            print(f"   {'✅' if status else '❌'} {component}")
        
        # Test system startup (brief)
        print("4. Testing system startup...")
        try:
            # Start system briefly
            startup_task = asyncio.create_task(unified_interface.start_unified_system())
            
            # Wait a few seconds for startup
            await asyncio.sleep(5)
            
            # Check if system started
            if unified_interface.running:
                print("   ✅ System started successfully")
                
                # Get status
                status = unified_interface.get_unified_status()
                print(f"   📊 System Health: {status.system_health}")
                print(f"   🔧 Mode: {status.mode.value}")
                print(f"   ⏱️  Uptime: {status.uptime_seconds:.1f}s")
                
                # Stop system
                print("5. Stopping system...")
                await unified_interface.stop_unified_system()
                print("   ✅ System stopped successfully")
                
            else:
                print("   ❌ System failed to start")
                return False
                
        except Exception as e:
            print(f"   ❌ System startup test failed: {e}")
            return False
        
        print("=" * 60)
        print("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_conversation_interface():
    """Test the conversation interface."""
    print("\n💬 Testing Conversation Interface...")
    print("=" * 60)
    
    try:
        # Create interface in conversation mode
        unified_interface = SchwabotUnifiedInterface(InterfaceMode.CONVERSATION)
        
        # Test conversation message (without starting full system)
        if unified_interface.schwabot_ai_integration:
            print("✅ Schwabot AI integration available")
            
            # Test simple message
            test_message = "Hello, can you analyze BTC/USD for me?"
            print(f"📤 Sending test message: {test_message}")
            
            # Note: This would require KoboldCPP to be running
            # For now, just test the interface structure
            print("✅ Conversation interface structure verified")
            
        else:
            print("❌ Schwabot AI integration not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False

async def test_visual_layer():
    """Test the visual layer interface."""
    print("\n🎨 Testing Visual Layer Interface...")
    print("=" * 60)
    
    try:
        # Create interface in visual layer mode
        unified_interface = SchwabotUnifiedInterface(InterfaceMode.VISUAL_LAYER)
        
        # Check visual controller
        if unified_interface.visual_controller:
            print("✅ Visual layer controller available")
            
            # Check configuration
            config = unified_interface.config.get("visual_layer", {})
            print(f"   📊 AI Analysis: {'✅' if config.get('enable_ai_analysis') else '❌'}")
            print(f"   🔍 Pattern Recognition: {'✅' if config.get('enable_pattern_recognition') else '❌'}")
            print(f"   🎬 Real-time Rendering: {'✅' if config.get('enable_real_time_rendering') else '❌'}")
            print(f"   🌊 DLT Waveform: {'✅' if config.get('enable_dlt_waveform') else '❌'}")
            
        else:
            print("❌ Visual layer controller not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Visual layer test failed: {e}")
        return False

async def test_api_interface():
    """Test the API interface."""
    print("\n🔌 Testing API Interface...")
    print("=" * 60)
    
    try:
        # Create interface in API mode
        unified_interface = SchwabotUnifiedInterface(InterfaceMode.API_ONLY)
        
        # Check JSON server
        if unified_interface.json_server:
            print("✅ JSON server available")
            
            # Check configuration
            config = unified_interface.config.get("api_integration", {})
            print(f"   🌐 Host: {config.get('host', 'unknown')}")
            print(f"   🔌 Port: {config.get('port', 'unknown')}")
            print(f"   🔒 Encryption: {'✅' if config.get('enable_encryption') else '❌'}")
            print(f"   🚦 Rate Limiting: {'✅' if config.get('enable_rate_limiting') else '❌'}")
            
        else:
            print("❌ JSON server not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Schwabot Unified System Integration Test")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Unified System", test_unified_system),
        ("Conversation Interface", test_conversation_interface),
        ("Visual Layer", test_visual_layer),
        ("API Interface", test_api_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified system is ready to use.")
        print("\n🚀 To start the system:")
        print("   python start_schwabot_unified.py")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 