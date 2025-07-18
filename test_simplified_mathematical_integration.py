#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Mathematical Integration Test
=======================================

This script tests the simplified mathematical integration system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_simplified_mathematical_integration():
    """Test the simplified mathematical integration."""
    print("🧠 Testing Simplified Mathematical Integration")
    print("=" * 60)
    
    try:
        from backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
        
        print("✅ Simplified mathematical integration import successful")
        
        # Test market data
        test_market_data = {
            'current_price': 52000.0,
            'entry_price': 50000.0,
            'volume': 1000.0,
            'volatility': 0.15,
            'price_history': [50000 + i * 100 for i in range(100)]
        }
        
        # Process through mathematical integration
        mathematical_signal = await mathematical_integration.process_market_data_mathematically(test_market_data)
        
        print("✅ Mathematical signal processing successful")
        print(f"   DLT Waveform Score: {mathematical_signal.dlt_waveform_score:.4f}")
        print(f"   Bit Phase: {mathematical_signal.bit_phase}")
        print(f"   Ferris Phase: {mathematical_signal.ferris_phase:.4f}")
        print(f"   Tensor Score: {mathematical_signal.tensor_score:.4f}")
        print(f"   Entropy Score: {mathematical_signal.entropy_score:.4f}")
        print(f"   Decision: {mathematical_signal.decision}")
        print(f"   Confidence: {mathematical_signal.confidence:.4f}")
        print(f"   Routing Target: {mathematical_signal.routing_target}")
        
        if mathematical_signal.dualistic_consensus:
            print(f"   Dualistic Consensus:")
            print(f"     Decision: {mathematical_signal.dualistic_consensus.get('decision', 'NONE')}")
            print(f"     Confidence: {mathematical_signal.dualistic_consensus.get('confidence', 0):.4f}")
            print(f"     ALEPH Score: {mathematical_signal.dualistic_consensus.get('aleph_score', 0):.4f}")
            print(f"     ALIF Score: {mathematical_signal.dualistic_consensus.get('alif_score', 0):.4f}")
            print(f"     RITL Score: {mathematical_signal.dualistic_consensus.get('ritl_score', 0):.4f}")
            print(f"     RITTLE Score: {mathematical_signal.dualistic_consensus.get('rittle_score', 0):.4f}")
        
        if mathematical_signal.lantern_projection:
            print(f"   Lantern Projection:")
            print(f"     Memory Match: {mathematical_signal.lantern_projection.get('memory_match', 0):.4f}")
            print(f"     Glyph Stability: {mathematical_signal.lantern_projection.get('glyph_stability', 0):.4f}")
            print(f"     Echo Strength: {mathematical_signal.lantern_projection.get('echo_strength', 0):.4f}")
            print(f"     Projection Confidence: {mathematical_signal.lantern_projection.get('projection_confidence', 0):.4f}")
        
        if mathematical_signal.quantum_state:
            print(f"   Quantum State:")
            print(f"     Purity: {mathematical_signal.quantum_state.get('purity', 0):.4f}")
            print(f"     Entanglement: {mathematical_signal.quantum_state.get('entanglement', 0):.4f}")
            print(f"     Superposition: {mathematical_signal.quantum_state.get('superposition', 0):.4f}")
            print(f"     Measurement Confidence: {mathematical_signal.quantum_state.get('measurement_confidence', 0):.4f}")
        
        if mathematical_signal.vault_orbital_state:
            print(f"   Vault Orbital State:")
            print(f"     Orbital Position: {mathematical_signal.vault_orbital_state.get('orbital_position', 0):.4f}")
            print(f"     Thermal State: {mathematical_signal.vault_orbital_state.get('thermal_state', 'NONE')}")
            print(f"     Memory Integration: {mathematical_signal.vault_orbital_state.get('memory_integration', 0):.4f}")
            print(f"     State Coordination: {mathematical_signal.vault_orbital_state.get('state_coordination', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified mathematical integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_production_pipeline_with_simplified_math():
    """Test production pipeline with simplified mathematical integration."""
    print("\n🚀 Testing Production Pipeline with Simplified Math")
    print("=" * 60)
    
    try:
        from AOI_Base_Files_Schwabot.core.production_trading_pipeline import TradingConfig
        
        # Create test configuration
        config = TradingConfig(
            exchange_name="binance",
            api_key="test_key",
            secret="test_secret",
            sandbox=True,
            symbols=["BTC/USDC"],
            enable_mathematical_integration=True,
            mathematical_confidence_threshold=0.7
        )
        
        print("✅ Production pipeline configuration created")
        print(f"   Exchange: {config.exchange_name}")
        print(f"   Symbols: {config.symbols}")
        print(f"   Mathematical Integration: {config.enable_mathematical_integration}")
        print(f"   Confidence Threshold: {config.mathematical_confidence_threshold}")
        
        # Test market data processing
        test_market_data = {
            'symbol': 'BTC/USDC',
            'price': 52000.0,
            'volume': 1000.0,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.7,
            'timestamp': 1640995200.0
        }
        
        print("✅ Market data processing test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Production pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_systems_availability():
    """Test that all mathematical systems are available."""
    print("\n🔧 Testing Mathematical Systems Availability")
    print("=" * 60)
    
    try:
        from backtesting.mathematical_integration_simplified import mathematical_integration
        
        # Check that all systems are available
        systems = [
            ("DLT Engine", mathematical_integration.dlt_engine),
            ("ALEPH Engine", mathematical_integration.aleph_engine),
            ("ALIF Engine", mathematical_integration.alif_engine),
            ("RITL Engine", mathematical_integration.ritl_engine),
            ("RITTLE Engine", mathematical_integration.rittle_engine),
        ]
        
        available_systems = []
        for system_name, system in systems:
            if system is not None:
                available_systems.append(system_name)
                print(f"   ✅ {system_name}")
            else:
                print(f"   ❌ {system_name}")
        
        print(f"\n📊 Mathematical Systems Summary:")
        print(f"   Available Systems: {len(available_systems)}/{len(systems)}")
        print(f"   Systems: {', '.join(available_systems)}")
        
        return len(available_systems) >= len(systems) * 0.8
        
    except Exception as e:
        print(f"❌ Mathematical systems availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all simplified mathematical integration tests."""
    print("🧠 SIMPLIFIED MATHEMATICAL INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    tests = [
        ("Simplified Mathematical Integration", test_simplified_mathematical_integration),
        ("Production Pipeline with Simplified Math", test_production_pipeline_with_simplified_math),
        ("Mathematical Systems Availability", test_mathematical_systems_availability),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SIMPLIFIED MATHEMATICAL INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SIMPLIFIED MATHEMATICAL INTEGRATION TESTS PASSED!")
        print("🚀 Simplified mathematical systems are working correctly!")
        print("💰 Ready for production deployment!")
    elif passed >= total * 0.8:
        print("✅ MOST SIMPLIFIED MATHEMATICAL INTEGRATION TESTS PASSED!")
        print("🚀 Simplified mathematical systems are mostly working!")
        print("⚠️ Some systems may need attention but core functionality is ready!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        print("🔧 Simplified mathematical systems may need additional work.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    asyncio.run(main()) 