#!/usr/bin/env python3
import sys
import os
import numpy as np

# Add paths
sys.path.append('core')
sys.path.append('AOI_Base_Files_Schwabot/core')

def test_phantom_mode():
    """Quick test of Phantom Mode Engine."""
    try:
        from core.phantom_mode_engine import PhantomModeEngine
        engine = PhantomModeEngine()
        
        # Test extreme values
        result1 = engine.zbe.compress_entropy(1000.0)
        result2 = engine.zbe.compress_entropy(-1000.0)
        result3 = engine.zbe.compress_entropy(0.0)
        
        print(f"Phantom Mode Engine:")
        print(f"  compress_entropy(1000.0) = {result1:.6f}")
        print(f"  compress_entropy(-1000.0) = {result2:.6f}")
        print(f"  compress_entropy(0.0) = {result3:.6f}")
        
        # Check if results are valid
        valid = all(np.isfinite(r) and 0 <= r <= 1 for r in [result1, result2, result3])
        print(f"  All results valid: {valid}")
        return valid
        
    except Exception as e:
        print(f"Phantom Mode Engine test failed: {e}")
        return False

def test_mode_integration():
    """Quick test of Mode Integration System."""
    try:
        from AOI_Base_Files_Schwabot.core.mode_integration_system import ModeIntegrationSystem, TradingMode
        system = ModeIntegrationSystem()
        
        # Test position size calculation
        system.portfolio_state['balance'] = 10000.0
        
        # Test valid case
        system.current_mode = TradingMode.DEFAULT
        config = system.get_current_config()
        result1 = system._calculate_position_size(50000.0, config)
        
        # Test invalid cases
        result2 = system._calculate_position_size(0.0, config)
        result3 = system._calculate_position_size(-50000.0, config)
        
        print(f"Mode Integration System:")
        print(f"  position_size(50000.0) = {result1:.6f}")
        print(f"  position_size(0.0) = {result2:.6f}")
        print(f"  position_size(-50000.0) = {result3:.6f}")
        
        # Check if error cases are handled
        valid = (result1 > 0 and result2 == 0.001 and result3 == 0.001)
        print(f"  Error handling correct: {valid}")
        return valid
        
    except Exception as e:
        print(f"Mode Integration System test failed: {e}")
        return False

def main():
    print("Quick Mathematical Fixes Test")
    print("=" * 40)
    
    phantom_ok = test_phantom_mode()
    mode_ok = test_mode_integration()
    
    print("\nSummary:")
    print(f"  Phantom Mode Engine: {'✅ PASS' if phantom_ok else '❌ FAIL'}")
    print(f"  Mode Integration System: {'✅ PASS' if mode_ok else '❌ FAIL'}")
    
    if phantom_ok and mode_ok:
        print("\n🎉 ALL MATHEMATICAL FIXES WORKING CORRECTLY!")
        return True
    else:
        print("\n⚠️ SOME ISSUES REMAIN - REVIEW REQUIRED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 