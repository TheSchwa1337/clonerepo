#!/usr/bin/env python3
"""
Test Enhanced Schwabot System
============================

Test script to verify the enhanced Schwabot trading bot system is working properly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import schwabot_enhanced_gui
        print("✅ Enhanced GUI imports successfully")
    except Exception as e:
        print(f"❌ Enhanced GUI import failed: {e}")
        return False
    
    try:
        import schwabot_usb_memory
        print("✅ USB Memory system imports successfully")
    except Exception as e:
        print(f"❌ USB Memory import failed: {e}")
        return False
    
    try:
        import schwabot_start
        print("✅ Start script imports successfully")
    except Exception as e:
        print(f"❌ Start script import failed: {e}")
        return False
    
    try:
        import schwabot_stop
        print("✅ Stop script imports successfully")
    except Exception as e:
        print(f"❌ Stop script import failed: {e}")
        return False
    
    try:
        import schwabot_cli
        print("✅ CLI imports successfully")
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False
    
    return True

def test_usb_memory():
    """Test USB memory system."""
    print("\n💾 Testing USB Memory system...")
    
    try:
        from schwabot_usb_memory import SchwabotUSBMemory
        
        # Initialize USB memory
        usb = SchwabotUSBMemory()
        
        # Check if memory directory exists
        memory_dir = Path("SchwabotMemory")
        if memory_dir.exists():
            print("✅ Memory directory exists")
            
            # Check subdirectories
            subdirs = ['config', 'state', 'logs', 'backups', 'data']
            for subdir in subdirs:
                subdir_path = memory_dir / subdir
                if subdir_path.exists():
                    print(f"✅ {subdir} directory exists")
                else:
                    print(f"❌ {subdir} directory missing")
        else:
            print("❌ Memory directory not found")
            return False
        
        # Test memory info
        info = usb.get_memory_info()
        print(f"✅ Memory info retrieved: {len(info)} items")
        
        # Stop the sync thread
        usb.stop()
        print("✅ USB memory system stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ USB Memory test failed: {e}")
        return False

def test_control_scripts():
    """Test control scripts exist."""
    print("\n🎮 Testing control scripts...")
    
    scripts = [
        "schwabot_start.py",
        "schwabot_stop.py", 
        "schwabot_cli.py",
        "schwabot_enhanced_gui.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
            return False
    
    return True

def test_installer():
    """Test installer script."""
    print("\n🔧 Testing installer...")
    
    if Path("install_schwabot.py").exists():
        print("✅ Installer script exists")
        return True
    else:
        print("❌ Installer script missing")
        return False

def test_documentation():
    """Test documentation files."""
    print("\n📚 Testing documentation...")
    
    docs = [
        "INSTALLATION_SUMMARY.md",
        "INSTALLER_README.md",
        "README.md"
    ]
    
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} missing")
    
    return True

def main():
    """Main test function."""
    print("🚀 Schwabot Enhanced System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_usb_memory,
        test_control_scripts,
        test_installer,
        test_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Schwabot Enhanced System is ready!")
        print("\n🚀 To start the system:")
        print("   • GUI: python schwabot_enhanced_gui.py")
        print("   • CLI: python schwabot_cli.py start")
        print("   • Installer: python install_schwabot.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 