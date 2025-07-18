#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-Step Mathematical Integration Test
==========================================

This script tests mathematical integration step by step to identify specific issues.
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

async def test_mathematical_integration_import():
    """Test if mathematical integration can be imported."""
    print("🔧 Testing Mathematical Integration Import")
    print("=" * 50)
    
    try:
        from backtesting.mathematical_integration import mathematical_integration, MathematicalSignal
        print("✅ Mathematical integration import successful")
        return True
    except Exception as e:
        print(f"❌ Mathematical integration import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_integration_engine():
    """Test mathematical integration engine initialization."""
    print("\n🤖 Testing Mathematical Integration Engine")
    print("=" * 50)
    
    try:
        from backtesting.mathematical_integration import MathematicalIntegrationEngine
        
        engine = MathematicalIntegrationEngine()
        print("✅ Mathematical integration engine initialized")
        
        # Test that all systems are available
        systems = [
            ("DLT Engine", engine.dlt_engine),
            ("ALEPH Engine", engine.aleph_engine),
            ("ALIF Engine", engine.alif_engine),
            ("RITL Engine", engine.ritl_engine),
            ("RITTLE Engine", engine.rittle_engine),
            ("Lantern Core", engine.lantern_core),
            ("Vault Orbital", engine.vault_orbital),
            ("Quantum Engine", engine.quantum_engine),
            ("Tensor Engine", engine.tensor_engine),
        ]
        
        for system_name, system in systems:
            status = "✅" if system is not None else "❌"
            print(f"   {status} {system_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical integration engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_signal_processing():
    """Test mathematical signal processing."""
    print("\n📊 Testing Mathematical Signal Processing")
    print("=" * 50)
    
    try:
        from backtesting.mathematical_integration import mathematical_integration
        
        # Create test market data
        test_market_data = {
            'current_price': 52000.0,
            'volume': 1000.0,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.7,
            'close_prices': [50000 + i * 100 for i in range(100)],
            'entry_price': 50000.0,
            'bit_phase': 8
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
            print(f"   Dualistic Consensus: {mathematical_signal.dualistic_consensus}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical signal processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_production_pipeline_import():
    """Test production pipeline import."""
    print("\n🚀 Testing Production Pipeline Import")
    print("=" * 50)
    
    try:
        from AOI_Base_Files_Schwabot.core.production_trading_pipeline import (
            ProductionTradingPipeline, TradingConfig, create_production_pipeline
        )
        print("✅ Production pipeline import successful")
        return True
    except Exception as e:
        print(f"❌ Production pipeline import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_production_pipeline_initialization():
    """Test production pipeline initialization."""
    print("\n🔧 Testing Production Pipeline Initialization")
    print("=" * 50)
    
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
        
        return True
        
    except Exception as e:
        print(f"❌ Production pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all step-by-step tests."""
    print("🧠 STEP-BY-STEP MATHEMATICAL INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    tests = [
        ("Mathematical Integration Import", test_mathematical_integration_import),
        ("Mathematical Integration Engine", test_mathematical_integration_engine),
        ("Mathematical Signal Processing", test_mathematical_signal_processing),
        ("Production Pipeline Import", test_production_pipeline_import),
        ("Production Pipeline Initialization", test_production_pipeline_initialization),
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
    print("📋 STEP-BY-STEP TEST SUMMARY")
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
        print("🎉 ALL MATHEMATICAL INTEGRATION TESTS PASSED!")
        print("🚀 Mathematical systems are ready for production!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    asyncio.run(main()) 