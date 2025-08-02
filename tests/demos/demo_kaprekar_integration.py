#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR INTEGRATION DEMO - SCHWABOT SYSTEM
=============================================

Demonstrates how Kaprekar's Constant (6174) logic integrates seamlessly
with the existing Schwabot trading system architecture.

This demo shows:
- Kaprekar entropy analysis
- TRG (Technical Resonance Grid) integration
- Enhanced Lantern Core processing
- System health monitoring integration
- Real-time signal classification
"""

import time
import hashlib
from datetime import datetime
from typing import Dict, Any

def demo_kaprekar_analyzer():
    """Demo the core Kaprekar analyzer functionality."""
    print("🧮 KAPREKAR ANALYZER DEMO")
    print("=" * 50)
    
    try:
        from core.mathlib.kaprekar_analyzer import kaprekar_analyzer
        
        # Test with different hash fragments
        test_hashes = [
            "a1b2c3d4",  # Should converge to 6174
            "f4e3d2c1",  # Another test hash
            "12345678",  # Test hash
            "deadbeef"   # Test hash
        ]
        
        for hash_fragment in test_hashes:
            print(f"\n📊 Analyzing hash fragment: {hash_fragment}")
            
            # Perform Kaprekar analysis
            result = kaprekar_analyzer.analyze_hash_fragment(hash_fragment)
            
            print(f"  Input Number: {result.input_number}")
            print(f"  Steps to 6174: {result.steps_to_converge}")
            print(f"  Entropy Class: {result.entropy_class}")
            print(f"  Stability Score: {result.stability_score:.2f}")
            print(f"  Is Convergent: {result.is_convergent}")
            print(f"  Strategy Recommendation: {kaprekar_analyzer.get_strategy_recommendation(result)}")
            
            if result.convergence_path:
                print(f"  Convergence Path: {' → '.join(map(str, result.convergence_path))}")
        
        # Show performance metrics
        metrics = kaprekar_analyzer.get_performance_metrics()
        print(f"\n📈 Kaprekar Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
    except ImportError as e:
        print(f"❌ Kaprekar analyzer not available: {e}")

def demo_trg_analyzer():
    """Demo the TRG (Technical Resonance Grid) analyzer."""
    print("\n📊 TRG ANALYZER DEMO")
    print("=" * 50)
    
    try:
        from core.trg_analyzer import trg_analyzer, TRGSnapshot
        
        # Create test scenarios
        test_scenarios = [
            {
                "name": "BTC Long Entry Signal",
                "kcs": 2,
                "rsi": 29.7,
                "price": 60230.23,
                "pole_range": (60180, 60600),
                "phantom_delta": 0.002,
                "hash_fragment": "a1b2c3d4"
            },
            {
                "name": "USDC Exit Signal",
                "kcs": 6,
                "rsi": 75.2,
                "price": 1.0001,
                "pole_range": (0.9999, 1.0002),
                "phantom_delta": -0.0001,
                "hash_fragment": "f4e3d2c1"
            },
            {
                "name": "Phantom Band Swing",
                "kcs": 4,
                "rsi": 52.1,
                "price": 45000.0,
                "pole_range": (44800, 45200),
                "phantom_delta": 0.005,
                "hash_fragment": "12345678"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n🎯 {scenario['name']}")
            
            # Create TRG snapshot
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
            
            # Perform TRG analysis
            result = trg_analyzer.interpret_trg(snapshot)
            
            print(f"  Signal Class: {result.signal_class}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Risk Level: {result.risk_level}")
            print(f"  Recommended Action: {result.recommended_action}")
            print(f"  Technical Context:")
            for key, value in result.technical_context.items():
                print(f"    {key}: {value}")
        
        # Show performance metrics
        metrics = trg_analyzer.get_performance_metrics()
        print(f"\n📈 TRG Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
    except ImportError as e:
        print(f"❌ TRG analyzer not available: {e}")

def demo_lantern_enhanced():
    """Demo the enhanced Lantern Core with Kaprekar integration."""
    print("\n🕯️ LANTERN CORE ENHANCED DEMO")
    print("=" * 50)
    
    try:
        from core.lantern_core_enhanced import lantern_core_enhanced
        
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
            },
            {
                "symbol": "USDC",
                "current_price": 1.0001,
                "rsi": 75.2,
                "pole_range": (0.9999, 1.0002),
                "phantom_delta": -0.0001,
                "hash_fragment": "f4e3d2c1",
                "ai_validation": "reject"
            }
        ]
        
        for i, signal_data in enumerate(test_signals, 1):
            print(f"\n🔄 Processing Enhanced Signal #{i}")
            
            # Process enhanced echo
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
                print(f"  ✅ Enhanced Signal Generated")
                print(f"  Symbol: {enhanced_signal.echo_signal.symbol}")
                print(f"  Confidence Boost: {enhanced_signal.confidence_boost:.2f}")
                print(f"  Final Confidence: {enhanced_signal.final_confidence:.2f}")
                print(f"  AI Validation: {enhanced_signal.ai_validation}")
                
                if enhanced_signal.kaprekar_result:
                    print(f"  Kaprekar Steps: {enhanced_signal.kaprekar_result.steps_to_converge}")
                    print(f"  Entropy Class: {enhanced_signal.kaprekar_result.entropy_class}")
                
                if enhanced_signal.trg_result:
                    print(f"  TRG Signal Class: {enhanced_signal.trg_result.signal_class}")
                    print(f"  Recommended Action: {enhanced_signal.trg_result.recommended_action}")
                
                # Get strategy recommendation
                strategy = lantern_core_enhanced.get_strategy_recommendation(enhanced_signal)
                print(f"  Strategy Recommendation: {strategy}")
            else:
                print(f"  ❌ Signal Rejected")
        
        # Show Kaprekar metrics
        metrics = lantern_core_enhanced.get_kaprekar_metrics()
        print(f"\n📈 Enhanced Lantern Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
    except ImportError as e:
        print(f"❌ Enhanced Lantern Core not available: {e}")

def demo_system_integration():
    """Demo the complete system integration."""
    print("\n🔗 SYSTEM INTEGRATION DEMO")
    print("=" * 50)
    
    try:
        from core.system_health_monitor import system_health_monitor
        
        # Get full system health report
        print("📊 Generating Full System Health Report...")
        report = system_health_monitor.get_full_report()
        
        # Display Kaprekar system status
        kaprekar_status = report.get("kaprekar_system_status")
        if kaprekar_status:
            print(f"\n🧮 Kaprekar System Status:")
            print(f"  Kaprekar Analyzer: {kaprekar_status.get('kaprekar_analyzer', 'UNKNOWN')}")
            print(f"  TRG Analyzer: {kaprekar_status.get('trg_analyzer', 'UNKNOWN')}")
            print(f"  Lantern Enhanced: {kaprekar_status.get('lantern_enhanced', 'UNKNOWN')}")
            
            # Show metrics
            kaprekar_metrics = kaprekar_status.get('kaprekar_metrics', {})
            if kaprekar_metrics:
                print(f"  Kaprekar Metrics:")
                for key, value in kaprekar_metrics.items():
                    print(f"    {key}: {value}")
            
            trg_metrics = kaprekar_status.get('trg_metrics', {})
            if trg_metrics:
                print(f"  TRG Metrics:")
                for key, value in trg_metrics.items():
                    print(f"    {key}: {value}")
            
            lantern_metrics = kaprekar_status.get('lantern_metrics', {})
            if lantern_metrics:
                print(f"  Lantern Enhanced Metrics:")
                for key, value in lantern_metrics.items():
                    print(f"    {key}: {value}")
        else:
            print("❌ Kaprekar system status not available")
        
        # Show overall system health
        print(f"\n🏥 Overall System Health: {report.get('overall_health', 'UNKNOWN')}")
        
    except ImportError as e:
        print(f"❌ System health monitor not available: {e}")

def demo_real_world_scenario():
    """Demo a real-world trading scenario with Kaprekar integration."""
    print("\n🌍 REAL-WORLD TRADING SCENARIO DEMO")
    print("=" * 50)
    
    try:
        from core.mathlib.kaprekar_analyzer import kaprekar_analyzer
        from core.trg_analyzer import trg_analyzer, TRGSnapshot
        from core.lantern_core_enhanced import lantern_core_enhanced
        
        # Simulate a real trading scenario
        print("🎯 Scenario: BTC Price Drop with RSI Oversold")
        print("   - Current BTC Price: $60,230")
        print("   - RSI: 29.7 (oversold)")
        print("   - Support Pole: $60,180 - $60,600")
        print("   - Phantom Delta: +0.002 (slight upward momentum)")
        
        # Generate a hash from current market conditions
        market_data = f"BTC_{int(time.time())}_{60230.23}_{29.7}"
        hash_fragment = hashlib.sha256(market_data.encode()).hexdigest()[:8]
        print(f"   - Generated Hash Fragment: {hash_fragment}")
        
        # Step 1: Kaprekar Analysis
        print(f"\n📊 Step 1: Kaprekar Entropy Analysis")
        kaprekar_result = kaprekar_analyzer.analyze_hash_fragment(hash_fragment)
        print(f"   Steps to 6174: {kaprekar_result.steps_to_converge}")
        print(f"   Entropy Class: {kaprekar_result.entropy_class}")
        print(f"   Stability Score: {kaprekar_result.stability_score:.2f}")
        
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
        print(f"   Confidence: {trg_result.confidence:.2f}")
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
            print(f"   Final Confidence: {enhanced_signal.final_confidence:.2f}")
            print(f"   Recommended Action: {enhanced_signal.trg_result.recommended_action}")
            
            # Final decision
            if enhanced_signal.final_confidence > 0.7:
                print(f"\n🎯 FINAL DECISION: EXECUTE BTC RE-ENTRY STRATEGY")
                print(f"   - High confidence signal ({enhanced_signal.final_confidence:.2f})")
                print(f"   - Kaprekar convergence in {kaprekar_result.steps_to_converge} steps")
                print(f"   - RSI oversold condition (29.7)")
                print(f"   - Price near support pole")
            else:
                print(f"\n⏸️ FINAL DECISION: DEFER - Insufficient confidence")
        else:
            print(f"   ❌ Signal Rejected")
            print(f"\n⏸️ FINAL DECISION: NO ACTION - Signal validation failed")
        
    except ImportError as e:
        print(f"❌ Required components not available: {e}")

def main():
    """Run the complete Kaprekar integration demo."""
    print("🧮 KAPREKAR INTEGRATION DEMO - SCHWABOT SYSTEM")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Demo Version: 1.0.0")
    print("=" * 60)
    
    # Run all demos
    demo_kaprekar_analyzer()
    demo_trg_analyzer()
    demo_lantern_enhanced()
    demo_system_integration()
    demo_real_world_scenario()
    
    print("\n" + "=" * 60)
    print("✅ KAPREKAR INTEGRATION DEMO COMPLETE")
    print("=" * 60)
    print("\n🎯 Key Integration Points:")
    print("   • Kaprekar entropy analysis enhances signal stability")
    print("   • TRG combines Kaprekar with technical indicators")
    print("   • Enhanced Lantern Core provides confidence boosting")
    print("   • System health monitor tracks Kaprekar performance")
    print("   • Seamless integration with existing Schwabot architecture")

if __name__ == "__main__":
    main() 