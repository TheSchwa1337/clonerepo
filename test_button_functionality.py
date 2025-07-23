#!/usr/bin/env python3
"""
🧪 Button Functionality Test - Verify All Mode Buttons Work
==========================================================

Quick test to verify that all mode buttons are working correctly:
- Ghost Mode Button
- Hybrid Mode Button  
- Ferris Ride Mode Button
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_mode_managers():
    """Test that all mode managers are available."""
    print("🧪 Testing Mode Manager Availability...")
    
    # Test Ghost Mode Manager
    try:
        from AOI_Base_Files_Schwabot.core.ghost_mode_manager import ghost_mode_manager
        print("✅ Ghost Mode Manager: AVAILABLE")
        ghost_status = ghost_mode_manager.get_ghost_mode_status()
        print(f"   Status: {ghost_status.get('active', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Ghost Mode Manager: NOT AVAILABLE - {e}")
    
    # Test Hybrid Mode Manager
    try:
        from AOI_Base_Files_Schwabot.core.hybrid_mode_manager import hybrid_mode_manager
        print("✅ Hybrid Mode Manager: AVAILABLE")
        hybrid_status = hybrid_mode_manager.get_hybrid_mode_status()
        print(f"   Status: {hybrid_status.get('active', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Hybrid Mode Manager: NOT AVAILABLE - {e}")
    
    # Test Ferris Ride Manager
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_manager import ferris_ride_manager
        print("✅ Ferris Ride Manager: AVAILABLE")
        ferris_status = ferris_ride_manager.get_ferris_ride_status()
        print(f"   Status: {ferris_status.get('active', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Ferris Ride Manager: NOT AVAILABLE - {e}")

def test_gui_imports():
    """Test that GUI can import all mode managers."""
    print("\n🎨 Testing GUI Import Capability...")
    
    try:
        from AOI_Base_Files_Schwabot.visual_controls_gui import (
            GHOST_MODE_AVAILABLE, 
            HYBRID_MODE_AVAILABLE, 
            FERRIS_RIDE_MODE_AVAILABLE
        )
        
        print(f"✅ Ghost Mode Available in GUI: {GHOST_MODE_AVAILABLE}")
        print(f"✅ Hybrid Mode Available in GUI: {HYBRID_MODE_AVAILABLE}")
        print(f"✅ Ferris Ride Mode Available in GUI: {FERRIS_RIDE_MODE_AVAILABLE}")
        
        if all([GHOST_MODE_AVAILABLE, HYBRID_MODE_AVAILABLE, FERRIS_RIDE_MODE_AVAILABLE]):
            print("🎉 ALL MODE BUTTONS WILL BE AVAILABLE IN GUI!")
        else:
            print("⚠️ Some mode buttons may not be available in GUI")
            
    except ImportError as e:
        print(f"❌ GUI Import Failed: {e}")

def test_button_activation():
    """Test button activation functionality."""
    print("\n🎯 Testing Button Activation Functionality...")
    
    # Test Ferris Ride activation
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_manager import ferris_ride_manager
        
        print("🎡 Testing Ferris Ride Mode Activation...")
        
        # Check current status
        initial_status = ferris_ride_manager.get_ferris_ride_status()
        print(f"   Initial Status: {'Active' if initial_status['active'] else 'Inactive'}")
        
        # Test activation
        if not initial_status['active']:
            success = ferris_ride_manager.activate_ferris_ride_mode()
            print(f"   Activation Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
            
            if success:
                # Test deactivation
                deactivate_success = ferris_ride_manager.deactivate_ferris_ride_mode()
                print(f"   Deactivation Result: {'✅ SUCCESS' if deactivate_success else '❌ FAILED'}")
        else:
            print("   Mode already active, testing deactivation...")
            deactivate_success = ferris_ride_manager.deactivate_ferris_ride_mode()
            print(f"   Deactivation Result: {'✅ SUCCESS' if deactivate_success else '❌ FAILED'}")
            
    except Exception as e:
        print(f"❌ Ferris Ride Activation Test Failed: {e}")

def main():
    """Main test function."""
    print("🧪 BUTTON FUNCTIONALITY TEST SUITE")
    print("=" * 50)
    
    test_mode_managers()
    test_gui_imports()
    test_button_activation()
    
    print("\n" + "=" * 50)
    print("🎯 BUTTON FUNCTIONALITY SUMMARY")
    print("=" * 50)
    print("✅ All mode managers should be available")
    print("✅ All buttons should appear in GUI Settings tab")
    print("✅ Activate/Deactivate functionality should work")
    print("✅ Status checking should work")
    print("✅ Requirements validation should work")
    
    print("\n🚀 To test the buttons:")
    print("1. Run: python demo_visual_controls.py")
    print("2. Go to Settings tab")
    print("3. Look for these buttons:")
    print("   • 🎯 Activate Ghost Mode")
    print("   • 🚀 Activate Hybrid Mode")
    print("   • 🎡 Activate Ferris Ride Mode")
    print("4. Click each button to test functionality")
    
    print("\n🎡 Ferris Ride Mode is the revolutionary auto-trading system!")
    print("   • Auto-detection of capital and USDC pairs")
    print("   • Pattern studying for 3+ days before entry")
    print("   • Hash pattern matching for precise timing")
    print("   • Confidence building through profits")
    print("   • Ferris RDE mathematical framework")
    print("   • USB backup system")
    print("   • Everything to USDC / USDC to Everything strategy")

if __name__ == "__main__":
    main() 