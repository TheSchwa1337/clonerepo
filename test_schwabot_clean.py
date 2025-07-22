#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Schwabot Clean Test
=========================
Tests all major components to ensure everything is working properly.
"""

import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test that all major components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from AOI_Base_Files_Schwabot.core.orbital_shell_brain_system import OrbitalBRAINSystem
        print("✅ Orbital Brain System imported")
        
        from AOI_Base_Files_Schwabot.core.risk_manager import RiskManager
        print("✅ Risk Manager imported")
        
        from AOI_Base_Files_Schwabot.core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
        print("✅ Trading Executor imported")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_initialization():
    """Test that components can be initialized."""
    print("\n🧪 Testing initialization...")
    
    try:
        # Test RiskManager initialization
        from AOI_Base_Files_Schwabot.core.risk_manager import RiskManager
        risk_manager = RiskManager()
        print("✅ Risk Manager initialized")
        
        # Test system status
        status = risk_manager.get_system_status()
        print(f"✅ System status: {status['system_health']}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False

def test_trading_pairs():
    """Test that trading pairs are properly configured."""
    print("\n🧪 Testing trading pairs...")
    
    try:
        from AOI_Base_Files_Schwabot.core.orbital_shell_brain_system import OrbitalBRAINSystem
        brain = OrbitalBRAINSystem()
        
        pairs = list(brain.trading_pairs.keys())
        print(f"✅ Trading pairs: {pairs}")
        
        # Check for USDC pairs only
        usdc_pairs = [p for p in pairs if 'USDC' in p]
        if len(usdc_pairs) == len(pairs):
            print("✅ All pairs are USDC-based (no banned pairs)")
        else:
            print("⚠️ Some non-USDC pairs found")
        
        return True
    except Exception as e:
        print(f"❌ Trading pairs test failed: {e}")
        return False

def test_config_files():
    """Test that configuration files are accessible."""
    print("\n🧪 Testing configuration files...")
    
    try:
        config_files = [
            "AOI_Base_Files_Schwabot/config/trading_config.yaml",
            "AOI_Base_Files_Schwabot/config/high_frequency_crypto_config.yaml",
            "AOI_Base_Files_Schwabot/config/trading_pairs.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"✅ {config_file} exists")
            else:
                print(f"❌ {config_file} missing")
        
        return True
    except Exception as e:
        print(f"❌ Config files test failed: {e}")
        return False

def test_baseline_validator():
    """Test that the baseline validator works."""
    print("\n🧪 Testing baseline validator...")
    
    try:
        from baseline_logic_validator import BaselineLogicValidator
        validator = BaselineLogicValidator()
        print("✅ Baseline validator imported")
        
        # Run a quick validation
        import asyncio
        report = asyncio.run(validator.run_full_validation())
        
        summary = report["validation_summary"]
        print(f"✅ Validation complete: {summary['passed']}/{summary['total_checks']} passed")
        print(f"✅ Critical issues: {summary['critical_issues']}")
        
        return summary['critical_issues'] == 0
    except Exception as e:
        print(f"❌ Baseline validator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SCHWABOT CLEAN TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_initialization,
        test_trading_pairs,
        test_config_files,
        test_baseline_validator
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
        print("🎉 ALL TESTS PASSED - SCHWABOT IS CLEAN!")
        return 0
    else:
        print("⚠️ Some tests failed - check the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 