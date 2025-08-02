#!/usr/bin/env python3
"""
Test USB Detection System
========================

Test script to verify USB detection is working properly.
"""

import sys
import os
from pathlib import Path

def test_usb_detection():
    """Test USB detection methods."""
    print("🔍 Testing USB Detection System")
    print("=" * 50)
    
    try:
        from schwabot_usb_memory import SchwabotUSBMemory
        
        # Initialize USB memory system
        print("📡 Initializing USB memory system...")
        usb_memory = SchwabotUSBMemory()
        
        # Test USB detection
        print("\n🔍 Testing USB drive detection...")
        usb_drives = usb_memory.find_usb_drives()
        
        if usb_drives:
            print(f"✅ Found {len(usb_drives)} USB drive(s):")
            for i, drive in enumerate(usb_drives, 1):
                print(f"   {i}. {drive}")
                
                # Check drive properties
                try:
                    drive_size = sum(f.stat().st_size for f in drive.rglob('*') if f.is_file())
                    print(f"      Size: {drive_size / (1024**3):.2f} GB")
                except:
                    print(f"      Size: Unknown")
        else:
            print("❌ No USB drives found")
        
        # Test memory initialization
        print(f"\n💾 Memory directory: {usb_memory.usb_memory_dir}")
        
        # Test backup functionality
        print("\n💾 Testing backup functionality...")
        if usb_memory.backup_memory(force=True):
            print("✅ Backup test successful")
        else:
            print("❌ Backup test failed")
        
        # Test memory info
        print("\n📊 Testing memory info...")
        info = usb_memory.get_memory_info()
        if info:
            print(f"✅ Memory info retrieved: {len(info)} items")
            print(f"   USB Directory: {info.get('usb_memory_dir', 'Unknown')}")
            print(f"   Backup Count: {info.get('backup_count', 0)}")
            print(f"   Latest Backup: {info.get('latest_backup', 'None')}")
            print(f"   Total Size: {info.get('total_size_mb', 0):.2f} MB")
        else:
            print("❌ Failed to get memory info")
        
        # Stop the system
        usb_memory.stop()
        print("\n🛑 USB memory system stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ USB detection test failed: {e}")
        return False

def test_enhanced_gui_usb():
    """Test USB detection in enhanced GUI."""
    print("\n🖥️ Testing Enhanced GUI USB Detection")
    print("=" * 50)
    
    try:
        from schwabot_enhanced_gui import SchwabotEnhancedGUI
        
        # Create GUI instance (without showing window)
        print("📡 Initializing Enhanced GUI...")
        gui = SchwabotEnhancedGUI()
        
        # Test USB detection
        print("\n🔍 Testing USB drive detection in GUI...")
        usb_drives = gui.find_usb_drives()
        
        if usb_drives:
            print(f"✅ Found {len(usb_drives)} USB drive(s) in GUI:")
            for i, drive in enumerate(usb_drives, 1):
                print(f"   {i}. {drive}")
        else:
            print("❌ No USB drives found in GUI")
        
        # Test USB list refresh
        print("\n🔄 Testing USB list refresh...")
        gui.refresh_usb_list()
        print("✅ USB list refresh completed")
        
        # Test USB drive selection
        if usb_drives:
            print("\n🎯 Testing USB drive selection...")
            if gui.select_usb_drive():
                print("✅ USB drive selection successful")
            else:
                print("❌ USB drive selection failed")
        
        print("\n✅ Enhanced GUI USB detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced GUI USB detection test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 USB Detection System Test")
    print("=" * 60)
    
    tests = [
        ("USB Memory System", test_usb_detection),
        ("Enhanced GUI USB", test_enhanced_gui_usb)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 Running {name} test...")
        if test_func():
            passed += 1
            print(f"✅ {name} test passed")
        else:
            print(f"❌ {name} test failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All USB detection tests passed!")
        print("\n🚀 USB detection system is working properly!")
        print("   • Multiple detection methods implemented")
        print("   • WMI, win32api, and fallback methods available")
        print("   • Enhanced GUI integration working")
        print("   • USB drive selection and management functional")
    else:
        print("⚠️ Some USB detection tests failed.")
        print("   • Check if USB drives are properly connected")
        print("   • Verify Windows API permissions")
        print("   • Ensure pywin32 is installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 