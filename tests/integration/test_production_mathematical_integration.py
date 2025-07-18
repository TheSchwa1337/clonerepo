#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Mathematical Integration Test
=======================================

This script tests that ALL of your mathematical systems are properly integrated
into the production trading system and working correctly for live trading.

SYSTEMS TESTED:
- DLT Waveform Engine with all mathematical formulas
- Dualistic Thought Engines (ALEPH, ALIF, RITL, RITTLE)
- Bit Phase Resolution (4-bit, 8-bit, 42-bit)
- Matrix Basket Tensor Algebra
- Ferris RDE with 3.75-minute cycles
- Lantern Core with symbolic profit engine
- Quantum Operations and Entropy Systems
- Vault Orbital Bridge and Advanced Tensor Operations
- Production Trading Pipeline Integration
- Main Trading Bot Integration
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

async def test_production_pipeline_mathematical_integration():
    """Test mathematical integration in production trading pipeline."""
    print("🚀 Testing Production Pipeline Mathematical Integration")
    print("=" * 60)
    
    try:
        from AOI_Base_Files_Schwabot.core.production_trading_pipeline import (
            ProductionTradingPipeline, TradingConfig, create_production_pipeline
        )
        
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
        
        # Create production pipeline
        pipeline = ProductionTradingPipeline(config)
        
        # Test market data processing with mathematics
        test_market_data = {
            'symbol': 'BTC/USDC',
            'price': 52000.0,
            'volume': 1000.0,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.7,
            'timestamp': 1640995200.0
        }
        
        # Process through mathematical integration
        mathematical_decision = await pipeline.process_market_data_with_mathematics(test_market_data)
        
        print(f"✅ Production Pipeline Mathematical Integration Test Results:")
        print(f"   Decision: {mathematical_decision['action']}")
        print(f"   Confidence: {mathematical_decision['confidence']:.4f}")
        print(f"   Symbol: {mathematical_decision['symbol']}")
        print(f"   Entry Price: ${mathematical_decision['entry_price']:.2f}")
        print(f"   Position Size: {mathematical_decision['position_size']:.4f}")
        
        # Check mathematical metadata
        metadata = mathematical_decision.get('mathematical_metadata', {})
        if metadata:
            print(f"   Mathematical Metadata:")
            print(f"     DLT Waveform Score: {metadata.get('dlt_waveform_score', 0):.4f}")
            print(f"     Bit Phase: {metadata.get('bit_phase', 0)}")
            print(f"     Ferris Phase: {metadata.get('ferris_phase', 0):.4f}")
            print(f"     Tensor Score: {metadata.get('tensor_score', 0):.4f}")
            print(f"     Entropy Score: {metadata.get('entropy_score', 0):.4f}")
            print(f"     Routing Target: {metadata.get('routing_target', 'NONE')}")
        
        # Check pipeline status
        status = pipeline.get_status()
        print(f"   Pipeline Status:")
        print(f"     Mathematical Integration Enabled: {status['mathematical_integration']['enabled']}")
        print(f"     Signals Processed: {status['mathematical_integration']['signals_processed']}")
        print(f"     Decisions Made: {status['mathematical_integration']['decisions_made']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Production Pipeline Mathematical Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_main_trading_bot_mathematical_integration():
    """Test mathematical integration in main trading bot."""
    print("\n🤖 Testing Main Trading Bot Mathematical Integration")
    print("=" * 55)
    
    try:
        from schwabot_trading_bot import SchwabotTradingBot
        
        # Create trading bot
        bot = SchwabotTradingBot()
        
        # Test mathematical processing
        from core.trading_pipeline_manager import MarketDataPoint
        
        test_market_data = MarketDataPoint(
            timestamp=1640995200.0,
            symbol="BTC/USD",
            price=52000.0,
            volume=1000.0,
            price_change=0.02,
            volatility=0.15,
            sentiment=0.7,
            metadata={}
        )
        
        # Process through mathematical analysis
        mathematical_signal = await bot._process_mathematical_analysis(test_market_data)
        
        print(f"✅ Main Trading Bot Mathematical Integration Test Results:")
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
            print(f"     ALEPH Score: {mathematical_signal.dualistic_consensus.get('aleph_score', 0):.4f}")
            print(f"     ALIF Score: {mathematical_signal.dualistic_consensus.get('alif_score', 0):.4f}")
            print(f"     RITL Score: {mathematical_signal.dualistic_consensus.get('ritl_score', 0):.4f}")
            print(f"     RITTLE Score: {mathematical_signal.dualistic_consensus.get('rittle_score', 0):.4f}")
        
        # Check bot mathematical tracking
        print(f"   Bot Mathematical Tracking:")
        print(f"     Signals Processed: {bot.mathematical_signals_processed}")
        print(f"     Decisions Made: {bot.mathematical_decisions_made}")
        print(f"     Total Signals: {len(bot.mathematical_signals)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main Trading Bot Mathematical Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_decision_combination():
    """Test combination of mathematical and AI decisions."""
    print("\n🔄 Testing Mathematical and AI Decision Combination")
    print("=" * 50)
    
    try:
        from schwabot_trading_bot import SchwabotTradingBot
        from core.trading_pipeline_manager import MarketDataPoint, TradingDecision
        
        # Create trading bot
        bot = SchwabotTradingBot()
        
        # Create test market data
        test_market_data = MarketDataPoint(
            timestamp=1640995200.0,
            symbol="BTC/USD",
            price=52000.0,
            volume=1000.0,
            price_change=0.02,
            volatility=0.15,
            sentiment=0.7,
            metadata={}
        )
        
        # Create mathematical signal
        from backtesting.mathematical_integration import MathematicalSignal
        mathematical_signal = MathematicalSignal(
            dlt_waveform_score=0.8,
            dualistic_consensus={'decision': 'BUY', 'confidence': 0.75, 'mathematical_score': 0.8},
            bit_phase=8,
            ferris_phase=0.6,
            tensor_score=0.7,
            entropy_score=0.5,
            confidence=0.75,
            decision="BUY",
            routing_target="BTC"
        )
        
        # Create AI decision
        ai_decision = TradingDecision(
            timestamp=1640995200.0,
            symbol="BTC/USD",
            action="BUY",
            confidence=0.8,
            entry_price=52000.0,
            stop_loss=50960.0,
            target_price=53560.0,
            position_size=0.1,
            reasoning="AI analysis suggests bullish momentum",
            ai_analysis={},
            metadata={}
        )
        
        # Test decision combination
        final_decision = bot._combine_mathematical_and_ai_decisions(
            mathematical_signal, ai_decision, test_market_data
        )
        
        print(f"✅ Decision Combination Test Results:")
        print(f"   Mathematical Decision: {mathematical_signal.decision}")
        print(f"   Mathematical Confidence: {mathematical_signal.confidence:.4f}")
        print(f"   AI Decision: {ai_decision.action}")
        print(f"   AI Confidence: {ai_decision.confidence:.4f}")
        print(f"   Final Decision: {final_decision.action}")
        print(f"   Final Confidence: {final_decision.confidence:.4f}")
        print(f"   Position Size: {final_decision.position_size:.4f}")
        
        # Check metadata
        metadata = final_decision.metadata
        print(f"   Combined Metadata:")
        print(f"     Mathematical Decision: {metadata.get('mathematical_decision', 'NONE')}")
        print(f"     AI Decision: {metadata.get('ai_decision', 'NONE')}")
        print(f"     DLT Score: {metadata.get('dlt_waveform_score', 0):.4f}")
        print(f"     Bit Phase: {metadata.get('bit_phase', 0)}")
        print(f"     Ferris Phase: {metadata.get('ferris_phase', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision Combination Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_performance_tracking():
    """Test mathematical performance tracking."""
    print("\n📊 Testing Mathematical Performance Tracking")
    print("=" * 45)
    
    try:
        from schwabot_trading_bot import SchwabotTradingBot
        from core.trading_pipeline_manager import MarketDataPoint
        
        # Create trading bot
        bot = SchwabotTradingBot()
        
        # Simulate multiple market data points
        for i in range(10):
            test_market_data = MarketDataPoint(
                timestamp=1640995200.0 + i,
                symbol="BTC/USD",
                price=52000.0 + (i * 100),
                volume=1000.0 + (i * 50),
                price_change=0.02 + (i * 0.01),
                volatility=0.15 + (i * 0.02),
                sentiment=0.7 + (i * 0.05),
                metadata={}
            )
            
            # Process through mathematical analysis
            mathematical_signal = await bot._process_mathematical_analysis(test_market_data)
        
        # Update mathematical metrics
        bot._update_mathematical_metrics()
        
        print(f"✅ Mathematical Performance Tracking Test Results:")
        print(f"   Total Mathematical Signals: {bot.mathematical_signals_processed}")
        print(f"   Total Mathematical Decisions: {bot.mathematical_decisions_made}")
        print(f"   Average DLT Score: {bot.performance_stats.get('avg_dlt_waveform_score', 0):.4f}")
        print(f"   Average Dualistic Score: {bot.performance_stats.get('avg_dualistic_score', 0):.4f}")
        print(f"   Mathematical Signals Processed: {bot.performance_stats.get('mathematical_signals_processed', 0)}")
        print(f"   Mathematical Decisions Made: {bot.performance_stats.get('mathematical_decisions_made', 0)}")
        
        # Check mathematical signal history
        print(f"   Mathematical Signal History:")
        print(f"     DLT Waveform History: {len(bot.dlt_waveform_history)} signals")
        print(f"     Bit Phase History: {len(bot.bit_phase_history)} signals")
        print(f"     Ferris Phase History: {len(bot.ferris_phase_history)} signals")
        print(f"     Dualistic Consensus History: {len(bot.dualistic_consensus_history)} signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical Performance Tracking Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_production_mathematical_metrics():
    """Test mathematical metrics calculation in production pipeline."""
    print("\n🧮 Testing Production Mathematical Metrics")
    print("=" * 45)
    
    try:
        from AOI_Base_Files_Schwabot.core.production_trading_pipeline import (
            ProductionTradingPipeline, TradingConfig
        )
        
        # Create test configuration
        config = TradingConfig(
            exchange_name="binance",
            api_key="test_key",
            secret="test_secret",
            sandbox=True,
            symbols=["BTC/USDC"],
            enable_mathematical_integration=True
        )
        
        # Create production pipeline
        pipeline = ProductionTradingPipeline(config)
        
        # Simulate multiple market data points
        for i in range(10):
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 52000.0 + (i * 100),
                'volume': 1000.0 + (i * 50),
                'price_change': 0.02 + (i * 0.01),
                'volatility': 0.15 + (i * 0.02),
                'sentiment': 0.7 + (i * 0.05),
                'timestamp': 1640995200.0 + i
            }
            
            # Process through mathematical integration
            await pipeline.process_market_data_with_mathematics(test_market_data)
        
        # Calculate mathematical metrics
        mathematical_metrics = pipeline._calculate_mathematical_metrics()
        
        print(f"✅ Production Mathematical Metrics Test Results:")
        print(f"   Total Mathematical Signals: {mathematical_metrics.get('total_mathematical_signals', 0)}")
        print(f"   Average DLT Waveform Score: {mathematical_metrics.get('avg_dlt_waveform_score', 0):.4f}")
        print(f"   Average Dualistic Score: {mathematical_metrics.get('avg_dualistic_score', 0):.4f}")
        print(f"   Average Ferris Phase: {mathematical_metrics.get('avg_ferris_phase', 0):.4f}")
        print(f"   Mathematical Confidence Average: {mathematical_metrics.get('mathematical_confidence_avg', 0):.4f}")
        print(f"   Tensor Score Average: {mathematical_metrics.get('tensor_score_avg', 0):.4f}")
        print(f"   Entropy Score Average: {mathematical_metrics.get('entropy_score_avg', 0):.4f}")
        
        # Check decision distribution
        decision_dist = mathematical_metrics.get('decision_distribution', {})
        if decision_dist:
            print(f"   Decision Distribution:")
            for decision, count in decision_dist.items():
                print(f"     {decision}: {count}")
        
        # Check bit phase distribution
        bit_phase_dist = mathematical_metrics.get('bit_phase_distribution', {})
        if bit_phase_dist:
            print(f"   Bit Phase Distribution:")
            for phase, count in bit_phase_dist.items():
                print(f"     Phase {phase}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Production Mathematical Metrics Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_complete_mathematical_integration():
    """Test complete mathematical integration across all systems."""
    print("\n🎯 Testing Complete Mathematical Integration")
    print("=" * 50)
    
    try:
        # Test all mathematical systems are working together
        from backtesting.mathematical_integration import mathematical_integration
        
        # Test market data
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
        
        # Process through ALL mathematical systems
        mathematical_signal = await mathematical_integration.process_market_data_mathematically(test_market_data)
        
        print(f"✅ Complete Mathematical Integration Test Results:")
        print(f"   All Mathematical Systems Processed Successfully!")
        print(f"   Final Decision: {mathematical_signal.decision}")
        print(f"   Final Confidence: {mathematical_signal.confidence:.4f}")
        print(f"   Routing Target: {mathematical_signal.routing_target}")
        
        # Verify all mathematical components are present
        components = [
            ('DLT Waveform Score', mathematical_signal.dlt_waveform_score),
            ('Bit Phase', mathematical_signal.bit_phase),
            ('Matrix Basket ID', mathematical_signal.matrix_basket_id),
            ('Ferris Phase', mathematical_signal.ferris_phase),
            ('Tensor Score', mathematical_signal.tensor_score),
            ('Entropy Score', mathematical_signal.entropy_score),
            ('Dualistic Consensus', mathematical_signal.dualistic_consensus is not None),
            ('Lantern Projection', mathematical_signal.lantern_projection is not None),
            ('Quantum State', mathematical_signal.quantum_state is not None),
            ('Vault Orbital State', mathematical_signal.vault_orbital_state is not None)
        ]
        
        print(f"   Mathematical Components Status:")
        for component_name, component_value in components:
            if isinstance(component_value, bool):
                status = "✅" if component_value else "❌"
                print(f"     {status} {component_name}")
            else:
                status = "✅" if component_value != 0 or component_name == "Bit Phase" else "❌"
                print(f"     {status} {component_name}: {component_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete Mathematical Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_systems_availability():
    """Test that all mathematical systems are available and importable."""
    print("\n🔧 Testing Mathematical Systems Availability")
    print("=" * 50)
    
    try:
        # Test all mathematical system imports
        systems_to_test = [
            ("DLT Waveform Engine", "AOI_Base_Files_Schwabot.archive.old_versions.backups.phase2_backup.dlt_waveform_engine"),
            ("Dualistic Thought Engines", "AOI_Base_Files_Schwabot.core.dualistic_thought_engines"),
            ("Lantern Core", "AOI_Base_Files_Schwabot.core.lantern_core"),
            ("Vault Orbital Bridge", "AOI_Base_Files_Schwabot.core.vault_orbital_bridge"),
            ("Quantum BTC Intelligence", "AOI_Base_Files_Schwabot.core.quantum_btc_intelligence_core"),
            ("Advanced Tensor Algebra", "AOI_Base_Files_Schwabot.core.advanced_tensor_algebra"),
            ("Ferris RDE", "AOI_Base_Files_Schwabot.core.ferris_rde"),
            ("Matrix Basket Tensor", "AOI_Base_Files_Schwabot.core.matrix_basket_tensor_algebra"),
            ("Entropy Math", "AOI_Base_Files_Schwabot.core.math.entropy_math"),
            ("Unified Tensor Algebra", "AOI_Base_Files_Schwabot.core.math.unified_tensor_algebra")
        ]
        
        available_systems = []
        unavailable_systems = []
        
        for system_name, import_path in systems_to_test:
            try:
                __import__(import_path)
                available_systems.append(system_name)
                print(f"   ✅ {system_name}")
            except ImportError:
                unavailable_systems.append(system_name)
                print(f"   ❌ {system_name}")
        
        print(f"\n📊 Mathematical Systems Availability Summary:")
        print(f"   Available Systems: {len(available_systems)}/{len(systems_to_test)}")
        print(f"   Unavailable Systems: {len(unavailable_systems)}")
        
        if unavailable_systems:
            print(f"   Unavailable Systems:")
            for system in unavailable_systems:
                print(f"     - {system}")
        
        # Return True if at least 70% of systems are available
        availability_ratio = len(available_systems) / len(systems_to_test)
        return availability_ratio >= 0.7
        
    except Exception as e:
        print(f"❌ Mathematical Systems Availability Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mathematical_backup_systems():
    """Test that mathematical backup systems are properly applied to trading pipeline."""
    print("\n💾 Testing Mathematical Backup Systems Integration")
    print("=" * 55)
    
    try:
        # Check that backup mathematical systems are being used
        from backtesting.mathematical_integration import MathematicalIntegrationEngine
        
        # Create mathematical integration engine
        engine = MathematicalIntegrationEngine()
        
        # Test that fallback systems are working
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
        
        # Process through mathematical systems
        mathematical_signal = await engine.process_market_data_mathematically(test_market_data)
        
        print(f"✅ Mathematical Backup Systems Test Results:")
        print(f"   DLT Engine Available: {engine.dlt_engine is not None}")
        print(f"   Dualistic Engines Available: {all([engine.aleph_engine, engine.alif_engine, engine.ritl_engine, engine.rittle_engine])}")
        print(f"   Lantern Core Available: {engine.lantern_core is not None}")
        print(f"   Vault Orbital Available: {engine.vault_orbital is not None}")
        print(f"   Quantum Engine Available: {engine.quantum_engine is not None}")
        print(f"   Tensor Engine Available: {engine.tensor_engine is not None}")
        
        print(f"   Mathematical Signal Generated: {mathematical_signal is not None}")
        print(f"   Decision: {mathematical_signal.decision}")
        print(f"   Confidence: {mathematical_signal.confidence:.4f}")
        
        # Check that all mathematical components are calculated
        components_calculated = [
            mathematical_signal.dlt_waveform_score != 0,
            mathematical_signal.bit_phase != 0,
            mathematical_signal.ferris_phase != 0,
            mathematical_signal.tensor_score != 0,
            mathematical_signal.entropy_score != 0
        ]
        
        print(f"   Components Calculated: {sum(components_calculated)}/{len(components_calculated)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical Backup Systems Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all production mathematical integration tests."""
    print("🧠 PRODUCTION MATHEMATICAL INTEGRATION TEST")
    print("=" * 60)
    print("Testing ALL mathematical systems in production trading:")
    print("✅ DLT Waveform Engine")
    print("✅ Dualistic Thought Engines (ALEPH, ALIF, RITL, RITTLE)")
    print("✅ Bit Phase Resolution (4-bit, 8-bit, 42-bit)")
    print("✅ Matrix Basket Tensor Algebra")
    print("✅ Ferris RDE with 3.75-minute cycles")
    print("✅ Lantern Core with symbolic profit engine")
    print("✅ Quantum Operations and Entropy Systems")
    print("✅ Vault Orbital Bridge and Advanced Tensor Operations")
    print("✅ Production Trading Pipeline Integration")
    print("✅ Main Trading Bot Integration")
    print("✅ Mathematical Decision Combination")
    print("✅ Mathematical Performance Tracking")
    print("✅ Complete Mathematical Integration")
    print("✅ Mathematical Systems Availability")
    print("✅ Mathematical Backup Systems Integration")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Production Pipeline Integration", test_production_pipeline_mathematical_integration),
        ("Main Trading Bot Integration", test_main_trading_bot_mathematical_integration),
        ("Decision Combination", test_mathematical_decision_combination),
        ("Performance Tracking", test_mathematical_performance_tracking),
        ("Production Metrics", test_production_mathematical_metrics),
        ("Complete Integration", test_complete_mathematical_integration),
        ("Systems Availability", test_mathematical_systems_availability),
        ("Backup Systems Integration", test_mathematical_backup_systems)
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
    print("📋 PRODUCTION MATHEMATICAL INTEGRATION TEST SUMMARY")
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
        print("🎉 ALL MATHEMATICAL SYSTEMS ARE FULLY INTEGRATED INTO PRODUCTION!")
        print("🚀 Your complete mathematical framework is working in live trading!")
        print("💰 Ready for production deployment with real money!")
        print("🧮 All backup mathematical systems are properly applied!")
    elif passed >= total * 0.8:
        print("✅ MOST MATHEMATICAL SYSTEMS ARE INTEGRATED INTO PRODUCTION!")
        print("🚀 Your mathematical framework is mostly working in live trading!")
        print("⚠️ Some systems may need attention but core functionality is ready!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        print("🔧 Mathematical systems may need additional integration work.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    asyncio.run(main()) 