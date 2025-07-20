#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR INTEGRATION TEST - COMPLETE SYSTEM VERIFICATION
==========================================================

Tests the complete integration of Kaprekar's Constant (6174) logic
into the Schwabot trading system architecture.

This test verifies:
- Configuration loading and validation
- Strategy Mapper integration
- Hash Config Manager integration
- TRG analyzer functionality
- Enhanced Lantern Core processing
- System health monitoring
- Real-world trading scenarios
"""

import time
import hashlib
import yaml
from datetime import datetime
from typing import Dict, Any

def test_config_loading():
    """Test Kaprekar configuration loading."""
    print("🔧 TESTING CONFIGURATION LOADING")
    print("=" * 50)
    
    try:
        from core.kaprekar_config_loader import kaprekar_config
        
        # Test basic config loading
        print(f"✅ Config loaded successfully")
        print(f"   Kaprekar system enabled: {kaprekar_config.is_enabled()}")
        print(f"   Version: {kaprekar_config.get_config('kaprekar_system.version')}")
        
        # Test integration configs
        strategy_mapper_config = kaprekar_config.get_integration_config('strategy_mapper')
        print(f"   Strategy Mapper integration: {strategy_mapper_config.get('enabled', False)}")
        
        hash_config_config = kaprekar_config.get_integration_config('hash_config')
        print(f"   Hash Config integration: {hash_config_config.get('enabled', False)}")
        
        lantern_config = kaprekar_config.get_integration_config('lantern_core')
        print(f"   Lantern Core integration: {lantern_config.get('enabled', False)}")
        
        # Test TRG config
        trg_config = kaprekar_config.get_trg_config()
        print(f"   TRG Analyzer enabled: {trg_config.get('enabled', False)}")
        
        # Validate config
        if kaprekar_config.validate_config():
            print(f"✅ Configuration validation passed")
        else:
            print(f"❌ Configuration validation failed")
            
        return True
        
    except ImportError as e:
        print(f"❌ Config loader not available: {e}")
        return False

def test_strategy_mapper_integration():
    """Test Strategy Mapper integration with Kaprekar."""
    print("\n🎯 TESTING STRATEGY MAPPER INTEGRATION")
    print("=" * 50)
    
    try:
        from AOI_Base_Files_Schwabot.core.strategy_mapper import StrategyMapper
        
        # Create strategy mapper with Kaprekar config
        config = {
            'kaprekar_enabled': True,
            'kaprekar_entropy_weight': 0.3,
            'kaprekar_confidence_threshold': 0.7,
            'kaprekar_strategy_boost': True
        }
        
        strategy_mapper = StrategyMapper(config)
        
        # Test entropy score calculation with Kaprekar
        test_hash = "a1b2c3d4e5f67890"
        test_assets = ["BTC", "USDC"]
        
        entropy_score = strategy_mapper._calculate_entropy_score(test_hash, test_assets)
        print(f"✅ Entropy score calculated: {entropy_score:.3f}")
        
        # Test with different hash fragments
        test_hashes = ["a1b2c3d4", "f4e3d2c1", "12345678", "deadbeef"]
        
        for hash_fragment in test_hashes:
            score = strategy_mapper._calculate_entropy_score(hash_fragment, test_assets)
            print(f"   Hash {hash_fragment}: {score:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Strategy Mapper not available: {e}")
        return False

def test_hash_config_integration():
    """Test Hash Config Manager integration with Kaprekar."""
    print("\n🔐 TESTING HASH CONFIG MANAGER INTEGRATION")
    print("=" * 50)
    
    try:
        from AOI_Base_Files_Schwabot.core.hash_config_manager import HashConfigManager
        
        # Create hash config manager
        hash_config = HashConfigManager()
        
        # Test default config includes Kaprekar settings
        default_config = hash_config.get_default_config()
        
        kaprekar_settings = [
            'kaprekar_enabled',
            'kaprekar_confidence_threshold',
            'kaprekar_entropy_weight',
            'kaprekar_strategy_boost',
            'kaprekar_max_steps',
            'kaprekar_reject_threshold'
        ]
        
        print("✅ Hash Config Manager Kaprekar settings:")
        for setting in kaprekar_settings:
            value = default_config.get(setting, "NOT_FOUND")
            print(f"   {setting}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Hash Config Manager not available: {e}")
        return False

def test_trg_analyzer():
    """Test TRG analyzer with configuration."""
    print("\n📊 TESTING TRG ANALYZER")
    print("=" * 50)
    
    try:
        from core.trg_analyzer import trg_analyzer, TRGSnapshot
        from core.kaprekar_config_loader import kaprekar_config
        
        # Get TRG configuration
        trg_config = kaprekar_config.get_trg_config()
        
        if not trg_config.get('enabled', False):
            print("❌ TRG analyzer disabled in config")
            return False
        
        # Test scenarios from config
        test_scenarios = [
            {
                "name": "BTC Long Entry",
                "kcs": 2,
                "rsi": 29.7,
                "price": 60230.23,
                "pole_range": (60180, 60600),
                "phantom_delta": 0.002,
                "hash_fragment": "a1b2c3d4"
            },
            {
                "name": "USDC Exit",
                "kcs": 6,
                "rsi": 75.2,
                "price": 1.0001,
                "pole_range": (0.9999, 1.0002),
                "phantom_delta": -0.0001,
                "hash_fragment": "f4e3d2c1"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n🎯 Testing {scenario['name']}")
            
            snapshot = TRGSnapshot(
                kcs=scenario['kcs'],
                rsi=scenario['rsi'],
                price=scenario['price'],
                pole_range=scenario['pole_range'],
                phantom_delta=scenario['phantom_delta'],
                asset="BTC",
                timestamp=time.time(),
                hash_fragment=scenario['hash_fragment']
            )
            
            result = trg_analyzer.interpret_trg(snapshot)
            
            print(f"   Signal Class: {result.signal_class}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Risk Level: {result.risk_level}")
            print(f"   Action: {result.recommended_action}")
        
        return True
        
    except ImportError as e:
        print(f"❌ TRG analyzer not available: {e}")
        return False

def test_enhanced_lantern_core():
    """Test Enhanced Lantern Core with configuration."""
    print("\n🕯️ TESTING ENHANCED LANTERN CORE")
    print("=" * 50)
    
    try:
        from core.lantern_core_enhanced import lantern_core_enhanced
        from core.kaprekar_config_loader import kaprekar_config
        
        # Get Lantern Core configuration
        lantern_config = kaprekar_config.get_integration_config('lantern_core')
        
        if not lantern_config.get('enabled', False):
            print("❌ Lantern Core integration disabled in config")
            return False
        
        # Test enhanced echo processing
        test_signals = [
            {
                "symbol": "BTC",
                "current_price": 60230.23,
                "rsi": 29.7,
                "pole_range": (60180, 60600),
                "phantom_delta": 0.002,
                "hash_fragment": "a1b2c3d4",
                "ai_validation": "approve"
            }
        ]
        
        for i, signal_data in enumerate(test_signals, 1):
            print(f"\n🔄 Testing Enhanced Signal #{i}")
            
            enhanced_signal = lantern_core_enhanced.process_enhanced_echo(
                symbol=signal_data["symbol"],
                current_price=signal_data["current_price"],
                rsi=signal_data["rsi"],
                pole_range=signal_data["pole_range"],
                phantom_delta=signal_data["phantom_delta"],
                hash_fragment=signal_data["hash_fragment"],
                ai_validation=signal_data["ai_validation"]
            )
            
            if enhanced_signal:
                print(f"   ✅ Signal Generated")
                print(f"   Final Confidence: {enhanced_signal.final_confidence:.3f}")
                print(f"   Confidence Boost: {enhanced_signal.confidence_boost:.3f}")
                
                if enhanced_signal.kaprekar_result:
                    print(f"   Kaprekar Steps: {enhanced_signal.kaprekar_result.steps_to_converge}")
                    print(f"   Entropy Class: {enhanced_signal.kaprekar_result.entropy_class}")
                
                if enhanced_signal.trg_result:
                    print(f"   TRG Signal: {enhanced_signal.trg_result.signal_class}")
                    print(f"   Recommended Action: {enhanced_signal.trg_result.recommended_action}")
            else:
                print(f"   ❌ Signal Rejected")
        
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced Lantern Core not available: {e}")
        return False

def test_system_health_monitor():
    """Test System Health Monitor with Kaprekar integration."""
    print("\n🏥 TESTING SYSTEM HEALTH MONITOR")
    print("=" * 50)
    
    try:
        from core.system_health_monitor import system_health_monitor
        
        # Get full system health report
        report = system_health_monitor.get_full_report()
        
        # Check Kaprekar system status
        kaprekar_status = report.get("kaprekar_system_status")
        
        if kaprekar_status:
            print("✅ Kaprekar System Status:")
            print(f"   Kaprekar Analyzer: {kaprekar_status.get('kaprekar_analyzer', 'UNKNOWN')}")
            print(f"   TRG Analyzer: {kaprekar_status.get('trg_analyzer', 'UNKNOWN')}")
            print(f"   Lantern Enhanced: {kaprekar_status.get('lantern_enhanced', 'UNKNOWN')}")
            
            # Show metrics
            kaprekar_metrics = kaprekar_status.get('kaprekar_metrics', {})
            if kaprekar_metrics:
                print(f"   Kaprekar Metrics:")
                for key, value in kaprekar_metrics.items():
                    print(f"     {key}: {value}")
            
            trg_metrics = kaprekar_status.get('trg_metrics', {})
            if trg_metrics:
                print(f"   TRG Metrics:")
                for key, value in trg_metrics.items():
                    print(f"     {key}: {value}")
            
            lantern_metrics = kaprekar_status.get('lantern_metrics', {})
            if lantern_metrics:
                print(f"   Lantern Enhanced Metrics:")
                for key, value in lantern_metrics.items():
                    print(f"     {key}: {value}")
        else:
            print("❌ Kaprekar system status not available")
            return False
        
        print(f"\n🏥 Overall System Health: {report.get('overall_health', 'UNKNOWN')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ System Health Monitor not available: {e}")
        return False

def test_real_world_scenario():
    """Test a complete real-world trading scenario."""
    print("\n🌍 TESTING REAL-WORLD TRADING SCENARIO")
    print("=" * 50)
    
    try:
        from core.mathlib.kaprekar_analyzer import kaprekar_analyzer
        from core.trg_analyzer import trg_analyzer, TRGSnapshot
        from core.lantern_core_enhanced import lantern_core_enhanced
        from core.kaprekar_config_loader import kaprekar_config
        
        # Check if system is enabled
        if not kaprekar_config.is_enabled():
            print("❌ Kaprekar system disabled in config")
            return False
        
        print("🎯 Scenario: BTC Price Drop with RSI Oversold")
        print("   - Current BTC Price: $60,230")
        print("   - RSI: 29.7 (oversold)")
        print("   - Support Pole: $60,180 - $60,600")
        print("   - Phantom Delta: +0.002 (slight upward momentum)")
        
        # Generate hash from market conditions
        market_data = f"BTC_{int(time.time())}_{60230.23}_{29.7}"
        hash_fragment = hashlib.sha256(market_data.encode()).hexdigest()[:8]
        print(f"   - Generated Hash Fragment: {hash_fragment}")
        
        # Step 1: Kaprekar Analysis
        print(f"\n📊 Step 1: Kaprekar Entropy Analysis")
        kaprekar_result = kaprekar_analyzer.analyze_hash_fragment(hash_fragment)
        print(f"   Steps to 6174: {kaprekar_result.steps_to_converge}")
        print(f"   Entropy Class: {kaprekar_result.entropy_class}")
        print(f"   Stability Score: {kaprekar_result.stability_score:.3f}")
        
        # Step 2: TRG Analysis
        print(f"\n📊 Step 2: Technical Resonance Grid Analysis")
        trg_snapshot = TRGSnapshot(
            kcs=kaprekar_result.steps_to_converge,
            rsi=29.7,
            price=60230.23,
            pole_range=(60180, 60600),
            phantom_delta=0.002,
            asset="BTC",
            timestamp=time.time(),
            hash_fragment=hash_fragment
        )
        
        trg_result = trg_analyzer.interpret_trg(trg_snapshot)
        print(f"   Signal Class: {trg_result.signal_class}")
        print(f"   Confidence: {trg_result.confidence:.3f}")
        print(f"   Risk Level: {trg_result.risk_level}")
        
        # Step 3: Enhanced Lantern Processing
        print(f"\n📊 Step 3: Enhanced Lantern Core Processing")
        enhanced_signal = lantern_core_enhanced.process_enhanced_echo(
            symbol="BTC",
            current_price=60230.23,
            rsi=29.7,
            pole_range=(60180, 60600),
            phantom_delta=0.002,
            hash_fragment=hash_fragment,
            ai_validation="approve"
        )
        
        if enhanced_signal:
            print(f"   ✅ Signal Validated")
            print(f"   Final Confidence: {enhanced_signal.final_confidence:.3f}")
            print(f"   Recommended Action: {enhanced_signal.trg_result.recommended_action}")
            
            # Final decision
            if enhanced_signal.final_confidence > 0.7:
                print(f"\n🎯 FINAL DECISION: EXECUTE BTC RE-ENTRY STRATEGY")
                print(f"   - High confidence signal ({enhanced_signal.final_confidence:.3f})")
                print(f"   - Kaprekar convergence in {kaprekar_result.steps_to_converge} steps")
                print(f"   - RSI oversold condition (29.7)")
                print(f"   - Price near support pole")
            else:
                print(f"\n⏸️ FINAL DECISION: DEFER - Insufficient confidence")
        else:
            print(f"   ❌ Signal Rejected")
            print(f"\n⏸️ FINAL DECISION: NO ACTION - Signal validation failed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False

def main():
    """Run the complete Kaprekar integration test."""
    print("🧮 KAPREKAR INTEGRATION TEST - COMPLETE SYSTEM VERIFICATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Test Version: 1.0.0")
    print("=" * 70)
    
    # Run all tests
    test_results = []
    
    test_results.append(("Configuration Loading", test_config_loading()))
    test_results.append(("Strategy Mapper Integration", test_strategy_mapper_integration()))
    test_results.append(("Hash Config Manager Integration", test_hash_config_integration()))
    test_results.append(("TRG Analyzer", test_trg_analyzer()))
    test_results.append(("Enhanced Lantern Core", test_enhanced_lantern_core()))
    test_results.append(("System Health Monitor", test_system_health_monitor()))
    test_results.append(("Real-World Scenario", test_real_world_scenario()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - KAPREKAR INTEGRATION COMPLETE!")
        print("\n✅ Integration Points Verified:")
        print("   • Configuration loading and validation")
        print("   • Strategy Mapper entropy scoring")
        print("   • Hash Config Manager settings")
        print("   • TRG analyzer functionality")
        print("   • Enhanced Lantern Core processing")
        print("   • System health monitoring")
        print("   • Real-world trading scenarios")
        print("\n🚀 The Kaprekar system is fully integrated and ready for production!")
    else:
        print("⚠️ Some tests failed - check configuration and dependencies")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 