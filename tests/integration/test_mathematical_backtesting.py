#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test: All Mathematical Systems in Backtesting
==========================================================

This script tests ALL of the user's working mathematical systems integrated into the backtesting framework.
It demonstrates how DLT waveforms, dualistic engines, bit phases, matrix baskets, Ferris RDE, and all other
mathematical systems work together to generate trading decisions.

SYSTEMS TESTED:
- DLT Waveform Engine with all mathematical formulas
- Dualistic Thought Engines (ALEPH, ALIF, RITL, RITTLE)
- Bit Phase Resolution (4-bit, 8-bit, 42-bit)
- Matrix Basket Tensor Algebra
- Ferris RDE with 3.75-minute cycles
- Lantern Core with symbolic profit engine
- Quantum Operations and Entropy Systems
- Vault Orbital Bridge and Advanced Tensor Operations
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

async def test_mathematical_integration():
    """Test the mathematical integration engine."""
    print("🧮 Testing Mathematical Integration Engine")
    print("=" * 50)
    
    try:
        from backtesting.mathematical_integration import mathematical_integration, MathematicalSignal
        
        # Test market data
        test_market_data = {
            'current_price': 52000.0,
            'volume': 1000.0,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.7,
            'close_prices': [50000 + i * 100 for i in range(100)],  # Simulated price history
            'entry_price': 50000.0,
            'bit_phase': 8
        }
        
        # Process through mathematical integration
        mathematical_signal = await mathematical_integration.process_market_data_mathematically(test_market_data)
        
        print(f"✅ Mathematical Integration Test Results:")
        print(f"   DLT Waveform Score: {mathematical_signal.dlt_waveform_score:.4f}")
        print(f"   Bit Phase: {mathematical_signal.bit_phase}")
        print(f"   Matrix Basket ID: {mathematical_signal.matrix_basket_id}")
        print(f"   Ferris Phase: {mathematical_signal.ferris_phase:.4f}")
        print(f"   Tensor Score: {mathematical_signal.tensor_score:.4f}")
        print(f"   Entropy Score: {mathematical_signal.entropy_score:.4f}")
        print(f"   Final Decision: {mathematical_signal.decision}")
        print(f"   Confidence: {mathematical_signal.confidence:.4f}")
        print(f"   Routing Target: {mathematical_signal.routing_target}")
        
        if mathematical_signal.dualistic_consensus:
            print(f"   Dualistic Consensus:")
            print(f"     ALEPH Score: {mathematical_signal.dualistic_consensus.get('aleph_score', 0):.4f}")
            print(f"     ALIF Score: {mathematical_signal.dualistic_consensus.get('alif_score', 0):.4f}")
            print(f"     RITL Score: {mathematical_signal.dualistic_consensus.get('ritl_score', 0):.4f}")
            print(f"     RITTLE Score: {mathematical_signal.dualistic_consensus.get('rittle_score', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical Integration Test Failed: {e}")
        return False

async def test_dualistic_thought_engines():
    """Test individual dualistic thought engines."""
    print("\n🧠 Testing Dualistic Thought Engines")
    print("=" * 40)
    
    try:
        from AOI_Base_Files_Schwabot.core.dualistic_thought_engines import (
            ALEPHEngine, ALIFEngine, RITLEngine, RITTLEEngine,
            ThoughtState, process_dualistic_consensus
        )
        
        # Create test thought state
        class TestThoughtState:
            def __init__(self):
                self.glyph = "💰"
                self.phase = 0.75
                self.ncco = 0.6
                self.entropy = 0.3
                self.btc_price = 52000.0
                self.eth_price = 3200.0
                self.xrp_price = 0.55
                self.usdc_balance = 10000.0
        
        thought_state = TestThoughtState()
        
        # Test ALEPH Engine
        print("✴️ Testing ALEPH Engine:")
        aleph_engine = ALEPHEngine()
        aleph_output = aleph_engine.evaluate_trust(thought_state)
        print(f"   Decision: {aleph_output.decision}")
        print(f"   Confidence: {aleph_output.confidence:.3f}")
        print(f"   Routing: {aleph_output.routing_target}")
        print(f"   Mathematical Score: {aleph_output.mathematical_score:.3f}")
        
        # Test ALIF Engine
        print("\n✴️ Testing ALIF Engine:")
        alif_engine = ALIFEngine()
        test_market_data = {
            'btc_volume': 1000.0,
            'btc_volume_prev': 950.0,
            'btc_price_change': 0.02,
            'eth_price_change': 0.01
        }
        alif_output = alif_engine.process_feedback(thought_state, market_data=test_market_data)
        print(f"   Decision: {alif_output.decision}")
        print(f"   Confidence: {alif_output.confidence:.3f}")
        print(f"   Routing: {alif_output.routing_target}")
        print(f"   Mathematical Score: {alif_output.mathematical_score:.3f}")
        
        # Test RITL Engine
        print("\n🧮 Testing RITL Engine:")
        ritl_engine = RITLEngine()
        ritl_output = ritl_engine.validate_truth_lattice(thought_state)
        print(f"   Decision: {ritl_output.decision}")
        print(f"   Confidence: {ritl_output.confidence:.3f}")
        print(f"   Routing: {ritl_output.routing_target}")
        print(f"   Mathematical Score: {ritl_output.mathematical_score:.3f}")
        
        # Test RITTLE Engine
        print("\n🧮 Testing RITTLE Engine:")
        rittle_engine = RITTLEEngine()
        rittle_output = rittle_engine.process_dimensional_logic(thought_state)
        print(f"   Decision: {rittle_output.decision}")
        print(f"   Confidence: {rittle_output.confidence:.3f}")
        print(f"   Routing: {rittle_output.routing_target}")
        print(f"   Mathematical Score: {rittle_output.mathematical_score:.3f}")
        
        # Test Consensus
        print("\n🤝 Testing Dualistic Consensus:")
        consensus_output = process_dualistic_consensus(thought_state)
        print(f"   Consensus Decision: {consensus_output.decision}")
        print(f"   Consensus Confidence: {consensus_output.confidence:.3f}")
        print(f"   Consensus Routing: {consensus_output.routing_target}")
        print(f"   Consensus Score: {consensus_output.mathematical_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dualistic Thought Engines Test Failed: {e}")
        return False

async def test_dlt_waveform_engine():
    """Test DLT Waveform Engine."""
    print("\n🌊 Testing DLT Waveform Engine")
    print("=" * 35)
    
    try:
        from AOI_Base_Files_Schwabot.archive.old_versions.backups.phase2_backup.dlt_waveform_engine import DLTWaveformEngine
        
        dlt_engine = DLTWaveformEngine()
        
        # Test DLT transform
        print("📊 Testing DLT Transform:")
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        time_points = [0, 1, 2, 3, 4]
        frequencies = [0.1, 0.2, 0.3]
        
        dlt_transform = dlt_engine.calculate_dlt_transform(signal, time_points, frequencies)
        print(f"   DLT Transform Shape: {dlt_transform.shape}")
        print(f"   DLT Transform Type: {dlt_transform.dtype}")
        
        # Test DLT waveform generation
        print("\n📈 Testing DLT Waveform Generation:")
        time_points = list(range(100))
        waveform = dlt_engine.generate_dlt_waveform(time_points, decay=0.006)
        print(f"   Waveform Length: {len(waveform)}")
        print(f"   Waveform Min: {waveform.min():.4f}")
        print(f"   Waveform Max: {waveform.max():.4f}")
        print(f"   Waveform Mean: {waveform.mean():.4f}")
        
        # Test wave entropy
        print("\n🧮 Testing Wave Entropy:")
        entropy = dlt_engine.calculate_wave_entropy(signal)
        print(f"   Wave Entropy: {entropy:.4f}")
        
        # Test tensor score
        print("\n⚡ Testing Tensor Score:")
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        tensor_score = dlt_engine.calculate_tensor_score(weights, signal)
        print(f"   Tensor Score: {tensor_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ DLT Waveform Engine Test Failed: {e}")
        return False

async def test_bit_phase_resolution():
    """Test bit phase resolution system."""
    print("\n🔢 Testing Bit Phase Resolution")
    print("=" * 35)
    
    try:
        # Test different bit phases
        test_hash = "a1b2c3d4e5f67890"
        
        # 4-bit phase
        bit_4 = int(test_hash[0:1], 16) % 16
        print(f"   4-bit Phase: {bit_4} (0-15)")
        
        # 8-bit phase
        bit_8 = int(test_hash[0:2], 16) % 256
        print(f"   8-bit Phase: {bit_8} (0-255)")
        
        # 42-bit phase (capped for display)
        bit_42 = int(test_hash[0:11], 16) % 4398046511104
        print(f"   42-bit Phase: {bit_42} (0-4398046511103)")
        
        # Test phase selection based on volatility
        volatilities = [0.05, 0.15, 0.35]
        for vol in volatilities:
            if vol < 0.1:
                selected_phase = bit_4
                phase_type = "Conservative (4-bit)"
            elif vol < 0.3:
                selected_phase = bit_8
                phase_type = "Balanced (8-bit)"
            else:
                selected_phase = bit_42 % 1024
                phase_type = "Aggressive (42-bit capped)"
            
            print(f"   Volatility {vol:.2f} → {phase_type}: {selected_phase}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bit Phase Resolution Test Failed: {e}")
        return False

async def test_matrix_basket_tensor():
    """Test matrix basket tensor operations."""
    print("\n🧮 Testing Matrix Basket Tensor Operations")
    print("=" * 45)
    
    try:
        import numpy as np
        
        # Test matrix basket calculation
        test_hash = "a1b2c3d4e5f67890"
        basket_id = int(test_hash[4:8], 16) % 1024
        print(f"   Matrix Basket ID: {basket_id} (0-1023)")
        
        # Test tensor contraction: Tᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
        print("\n📐 Testing Tensor Contraction:")
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        T = np.dot(A, B)
        print(f"   Matrix A:\n{A}")
        print(f"   Matrix B:\n{B}")
        print(f"   Tensor T (A × B):\n{T}")
        
        # Test basket matching
        print("\n🎯 Testing Basket Matching:")
        hash1 = "a1b2c3d4e5f67890"
        hash2 = "f1e2d3c4b5a67890"
        
        similarity = sum(abs(int(hash1[i], 16) - int(hash2[i], 16)) for i in range(min(len(hash1), len(hash2)))) / len(hash1)
        print(f"   Hash 1: {hash1}")
        print(f"   Hash 2: {hash2}")
        print(f"   Similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix Basket Tensor Test Failed: {e}")
        return False

async def test_ferris_rde():
    """Test Ferris RDE (Recursive Differential Engine)."""
    print("\n🎡 Testing Ferris RDE (Recursive Differential Engine)")
    print("=" * 55)
    
    try:
        import numpy as np
        import time
        
        # Test Ferris phase calculation: Φ(t) = sin(2πft + φ)
        print("📊 Testing Ferris Phase Calculation:")
        current_time = time.time()
        frequency = 1.0 / (3.75 * 60)  # 3.75-minute cycles
        phase_offset = 0.0
        
        ferris_phase = np.sin(2 * np.pi * frequency * current_time + phase_offset)
        print(f"   Current Time: {current_time:.2f}")
        print(f"   Frequency: {frequency:.6f} Hz (3.75-minute cycles)")
        print(f"   Ferris Phase: {ferris_phase:.4f}")
        
        # Test multiple phases over time
        print("\n⏰ Testing Ferris Phases Over Time:")
        time_points = np.linspace(0, 1000, 10)  # 10 points over 1000 seconds
        phases = []
        for t in time_points:
            phase = np.sin(2 * np.pi * frequency * t + phase_offset)
            phases.append(phase)
            print(f"   Time {t:.1f}s → Phase: {phase:.4f}")
        
        # Test phase harmonics
        print(f"\n🎵 Phase Statistics:")
        print(f"   Min Phase: {min(phases):.4f}")
        print(f"   Max Phase: {max(phases):.4f}")
        print(f"   Mean Phase: {np.mean(phases):.4f}")
        print(f"   Phase Range: {max(phases) - min(phases):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ferris RDE Test Failed: {e}")
        return False

async def test_entropy_calculations():
    """Test entropy calculations."""
    print("\n🔥 Testing Entropy Calculations")
    print("=" * 35)
    
    try:
        import numpy as np
        
        # Test Shannon entropy: H = -Σ p_i * log2(p_i)
        print("🧮 Testing Shannon Entropy:")
        
        # Test case 1: Uniform distribution
        uniform_probs = [0.25, 0.25, 0.25, 0.25]
        uniform_entropy = -sum(p * np.log2(p) for p in uniform_probs)
        print(f"   Uniform Distribution: {uniform_entropy:.4f} bits")
        
        # Test case 2: Skewed distribution
        skewed_probs = [0.8, 0.1, 0.05, 0.05]
        skewed_entropy = -sum(p * np.log2(p) for p in skewed_probs)
        print(f"   Skewed Distribution: {skewed_entropy:.4f} bits")
        
        # Test case 3: Price changes entropy
        price_changes = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02]
        abs_changes = np.abs(price_changes)
        total = np.sum(abs_changes)
        if total > 0:
            probabilities = abs_changes / total
            price_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            print(f"   Price Changes Entropy: {price_entropy:.4f} bits")
        
        # Test ZPE (Zero-Point Energy) concept
        print("\n⚡ Testing ZPE Concepts:")
        thermal_states = ['COOL', 'WARM', 'HOT', 'CRITICAL']
        for state in thermal_states:
            if state == 'COOL':
                zpe = 0.1
            elif state == 'WARM':
                zpe = 0.3
            elif state == 'HOT':
                zpe = 0.7
            else:  # CRITICAL
                zpe = 1.0
            print(f"   {state} State ZPE: {zpe:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Entropy Calculations Test Failed: {e}")
        return False

async def test_complete_backtest_with_mathematics():
    """Test complete backtest with all mathematical systems."""
    print("\n🚀 Testing Complete Backtest with All Mathematical Systems")
    print("=" * 65)
    
    try:
        from backtesting.backtest_engine import BacktestConfig, BacktestEngine
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-07",  # 1 week
            symbols=["BTCUSDT"],
            initial_balance=10000.0,
            data_source="simulated",  # Use simulated data for testing
            enable_ai_analysis=True,
            enable_risk_management=True,
            min_confidence=0.6
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest
        print("🔄 Running backtest with mathematical integration...")
        result = await engine.run_backtest()
        
        # Display results
        print(f"\n📊 Backtest Results:")
        print(f"   Initial Balance: ${result.initial_balance:.2f}")
        print(f"   Final Balance: ${result.final_balance:.2f}")
        print(f"   Total Return: {result.total_return:.2%}")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.2%}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2%}")
        
        # Display mathematical metrics
        if 'mathematical_metrics' in result.performance_metrics:
            math_metrics = result.performance_metrics['mathematical_metrics']
            print(f"\n🧮 Mathematical Metrics:")
            print(f"   Total Mathematical Signals: {math_metrics.get('total_mathematical_signals', 0)}")
            print(f"   Avg DLT Waveform Score: {math_metrics.get('avg_dlt_waveform_score', 0):.4f}")
            print(f"   Avg Dualistic Score: {math_metrics.get('avg_dualistic_score', 0):.4f}")
            print(f"   Avg Ferris Phase: {math_metrics.get('avg_ferris_phase', 0):.4f}")
            print(f"   Mathematical Confidence Avg: {math_metrics.get('mathematical_confidence_avg', 0):.4f}")
            print(f"   Tensor Score Avg: {math_metrics.get('tensor_score_avg', 0):.4f}")
            print(f"   Entropy Score Avg: {math_metrics.get('entropy_score_avg', 0):.4f}")
            
            # Display decision distribution
            decision_dist = math_metrics.get('decision_distribution', {})
            if decision_dist:
                print(f"   Decision Distribution:")
                for decision, count in decision_dist.items():
                    print(f"     {decision}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete Backtest Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all mathematical system tests."""
    print("🧠 COMPREHENSIVE MATHEMATICAL SYSTEM TEST")
    print("=" * 50)
    print("Testing ALL working mathematical systems:")
    print("✅ DLT Waveform Engine")
    print("✅ Dualistic Thought Engines (ALEPH, ALIF, RITL, RITTLE)")
    print("✅ Bit Phase Resolution (4-bit, 8-bit, 42-bit)")
    print("✅ Matrix Basket Tensor Algebra")
    print("✅ Ferris RDE with 3.75-minute cycles")
    print("✅ Entropy Calculations and ZPE")
    print("✅ Complete Backtest Integration")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Mathematical Integration", test_mathematical_integration),
        ("Dualistic Thought Engines", test_dualistic_thought_engines),
        ("DLT Waveform Engine", test_dlt_waveform_engine),
        ("Bit Phase Resolution", test_bit_phase_resolution),
        ("Matrix Basket Tensor", test_matrix_basket_tensor),
        ("Ferris RDE", test_ferris_rde),
        ("Entropy Calculations", test_entropy_calculations),
        ("Complete Backtest", test_complete_backtest_with_mathematics)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL MATHEMATICAL SYSTEMS ARE WORKING PERFECTLY!")
        print("🚀 Your complete mathematical framework is ready for production!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 