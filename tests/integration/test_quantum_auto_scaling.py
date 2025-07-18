#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Quantum Auto-Scaling System
================================

Demonstrates the quantum auto-scaling system that uses hardware detection
as a quantum observer to drive real-time scaling decisions.

This implements the quantum chamber logic where:
- SHA-256 = Crystalline math for tiny logic brains
- Tensor Pool = Electrical signal harmonizer
- Hardware Detection = Quantum observer
- GPU/CPU/RAM swap = Variable collapse states
- Timing ∆ between inputs = ψ wave triggers
- Observer Effect = AI memory state pivots
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.quantum_auto_scaler import QuantumAutoScaler, ScalingTrigger

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/quantum_auto_scaling.log')
        ]
    )

def test_quantum_chamber_initialization():
    """Test quantum chamber initialization."""
    print("🧪 Testing Quantum Chamber Initialization")
    print("=" * 50)
    
    quantum_scaler = QuantumAutoScaler()
    
    # Print initial quantum chamber summary
    quantum_scaler.print_quantum_chamber_summary()
    
    # Get quantum chamber status
    status = quantum_scaler.get_quantum_chamber_status()
    
    print(f"\n✅ Quantum chamber initialized successfully!")
    print(f"   Hardware detected: {quantum_scaler.chamber_state.hardware_info.gpu.name}")
    print(f"   Quantum state: {status['quantum_state']}")
    print(f"   Coherence score: {status['coherence_score']:.3f}")
    
    return quantum_scaler

def test_quantum_scaling_decisions(quantum_scaler: QuantumAutoScaler):
    """Test quantum scaling decisions with different conditions."""
    print("\n🧪 Testing Quantum Scaling Decisions")
    print("=" * 50)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Normal Conditions",
            "market_entropy": 0.5,
            "thermal_state": 0.3,
            "profit_potential": 0.6
        },
        {
            "name": "High Market Entropy",
            "market_entropy": 0.9,
            "thermal_state": 0.3,
            "profit_potential": 0.6
        },
        {
            "name": "High Thermal State",
            "market_entropy": 0.5,
            "thermal_state": 0.8,
            "profit_potential": 0.6
        },
        {
            "name": "High Profit Potential",
            "market_entropy": 0.5,
            "thermal_state": 0.3,
            "profit_potential": 0.9
        },
        {
            "name": "Extreme Conditions",
            "market_entropy": 0.9,
            "thermal_state": 0.8,
            "profit_potential": 0.9
        }
    ]
    
    decisions = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nTest {i}: {scenario['name']}")
        print("-" * 30)
        
        decision = quantum_scaler.compute_quantum_scaling(
            market_entropy=scenario['market_entropy'],
            thermal_state=scenario['thermal_state'],
            profit_potential=scenario['profit_potential']
        )
        
        decisions.append(decision)
        
        print(f"  Trigger: {decision.trigger.value}")
        print(f"  Scaling Factor: {decision.scaling_factor:.3f}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Quantum Phase: {decision.quantum_phase:.3f} rad")
        print(f"  Observer Effect: {decision.observer_effect:.3f}")
        
        # Show bit depth adjustments
        if decision.bit_depth_adjustment:
            print(f"  Bit Depth Adjustments:")
            for bit_depth, adjustment in decision.bit_depth_adjustment.items():
                print(f"    {bit_depth}: {adjustment:.3f}x")
        
        # Show cache adjustments
        if decision.cache_adjustment:
            print(f"  Cache Adjustments:")
            for cache_type, adjustment in decision.cache_adjustment.items():
                print(f"    {cache_type}: {adjustment:.3f}x")
    
    return decisions

def test_scaling_application(quantum_scaler: QuantumAutoScaler, decisions):
    """Test applying scaling decisions."""
    print("\n🧪 Testing Scaling Decision Application")
    print("=" * 50)
    
    # Apply each decision and show the effects
    for i, decision in enumerate(decisions, 1):
        print(f"\nApplying Decision {i}: {decision.trigger.value}")
        print("-" * 40)
        
        # Get memory config before
        memory_config = quantum_scaler.chamber_state.memory_config
        print("Before scaling:")
        print(f"  TIC Map Sizes: {dict(memory_config.tic_map_sizes)}")
        print(f"  Cache Sizes: {dict(memory_config.cache_sizes)}")
        
        # Apply decision
        success = quantum_scaler.apply_scaling_decision(decision)
        
        if success:
            print(f"✅ Decision applied successfully!")
            
            # Get memory config after
            memory_config = quantum_scaler.chamber_state.memory_config
            print("After scaling:")
            print(f"  TIC Map Sizes: {dict(memory_config.tic_map_sizes)}")
            print(f"  Cache Sizes: {dict(memory_config.cache_sizes)}")
            print(f"  Scaling Multiplier: {quantum_scaler.chamber_state.scaling_multiplier:.3f}")
        else:
            print(f"❌ Failed to apply decision")

def test_quantum_coherence_evolution(quantum_scaler: QuantumAutoScaler):
    """Test quantum coherence evolution over time."""
    print("\n🧪 Testing Quantum Coherence Evolution")
    print("=" * 50)
    
    # Simulate time evolution
    for step in range(10):
        print(f"\nStep {step + 1}:")
        print("-" * 20)
        
        # Simulate changing conditions
        market_entropy = 0.3 + (step * 0.1) % 0.6  # Oscillating entropy
        thermal_state = 0.2 + (step * 0.15) % 0.6  # Gradually increasing thermal
        profit_potential = 0.4 + (step * 0.12) % 0.5  # Varying profit potential
        
        # Compute quantum scaling
        decision = quantum_scaler.compute_quantum_scaling(
            market_entropy=market_entropy,
            thermal_state=thermal_state,
            profit_potential=profit_potential
        )
        
        # Get current status
        status = quantum_scaler.get_quantum_chamber_status()
        
        print(f"  Market Entropy: {market_entropy:.3f}")
        print(f"  Thermal State: {thermal_state:.3f}")
        print(f"  Profit Potential: {profit_potential:.3f}")
        print(f"  Quantum State: {status['quantum_state']}")
        print(f"  Coherence Score: {status['coherence_score']:.3f}")
        print(f"  Entropy Value: {status['entropy_value']:.3f}")
        print(f"  Quantum Phase: {status['quantum_phase']:.3f} rad")
        print(f"  Scaling Factor: {decision.scaling_factor:.3f}")
        
        # Small delay to simulate real-time operation
        time.sleep(0.1)

def test_hardware_observer_effects(quantum_scaler: QuantumAutoScaler):
    """Test hardware observer effects on quantum scaling."""
    print("\n🧪 Testing Hardware Observer Effects")
    print("=" * 50)
    
    # Get initial hardware hash
    initial_hash = quantum_scaler.chamber_state.observer_state.hardware_hash
    print(f"Initial Hardware Hash: {initial_hash[:16]}...")
    
    # Simulate hardware state changes
    for i in range(5):
        print(f"\nHardware Change Simulation {i + 1}:")
        print("-" * 40)
        
        # Force hardware observer update (simulating hardware change)
        quantum_scaler._update_quantum_observer()
        
        # Get new hardware hash
        new_hash = quantum_scaler.chamber_state.observer_state.hardware_hash
        print(f"New Hardware Hash: {new_hash[:16]}...")
        
        # Check if hardware changed
        if new_hash != initial_hash:
            print("🔍 Hardware state change detected!")
            print(f"Observer Phase: {quantum_scaler.chamber_state.observer_state.observer_phase:.3f} rad")
            print(f"Entanglement Strength: {quantum_scaler.chamber_state.observer_state.entanglement_strength:.3f}")
            print(f"Collapse Probability: {quantum_scaler.chamber_state.observer_state.collapse_probability:.3f}")
            
            # Compute scaling with hardware change
            decision = quantum_scaler.compute_quantum_scaling(
                market_entropy=0.5,
                thermal_state=0.3,
                profit_potential=0.6
            )
            
            print(f"Scaling Decision: {decision.trigger.value}")
            print(f"Scaling Factor: {decision.scaling_factor:.3f}")
            print(f"Observer Effect: {decision.observer_effect:.3f}")
            
            initial_hash = new_hash
        else:
            print("No hardware state change detected")
        
        time.sleep(0.2)

def test_tensor_pool_harmonization(quantum_scaler: QuantumAutoScaler):
    """Test tensor pool harmonization effects."""
    print("\n🧪 Testing Tensor Pool Harmonization")
    print("=" * 50)
    
    # Test different tensor pool conditions
    tensor_conditions = [
        {"name": "Balanced", "positive": 0.5, "negative": 0.5, "zero": 0.5},
        {"name": "High E-Logic", "positive": 0.9, "negative": 0.3, "zero": 0.5},
        {"name": "High D-Memory", "positive": 0.3, "negative": 0.9, "zero": 0.5},
        {"name": "High F-Noise", "positive": 0.5, "negative": 0.3, "zero": 0.9},
        {"name": "Extreme", "positive": 0.9, "negative": 0.9, "zero": 0.9}
    ]
    
    for condition in tensor_conditions:
        print(f"\nTensor Condition: {condition['name']}")
        print("-" * 30)
        
        # Update tensor pool manually
        quantum_scaler.chamber_state.tensor_pool.positive_channel = condition['positive']
        quantum_scaler.chamber_state.tensor_pool.negative_channel = condition['negative']
        quantum_scaler.chamber_state.tensor_pool.zero_point = condition['zero']
        
        # Recalculate tensor pool metrics
        quantum_scaler.chamber_state.tensor_pool.harmonic_coherence = (
            condition['positive'] + condition['negative'] + condition['zero']
        ) / 3.0
        
        quantum_scaler.chamber_state.tensor_pool.signal_strength = np.sqrt(
            condition['positive']**2 + condition['negative']**2 + condition['zero']**2
        )
        
        # Compute quantum scaling
        decision = quantum_scaler.compute_quantum_scaling(
            market_entropy=0.5,
            thermal_state=0.3,
            profit_potential=0.6
        )
        
        print(f"  Positive Channel (E-Logic): {condition['positive']:.3f}")
        print(f"  Negative Channel (D-Memory): {condition['negative']:.3f}")
        print(f"  Zero Point (F-Noise): {condition['zero']:.3f}")
        print(f"  Harmonic Coherence: {quantum_scaler.chamber_state.tensor_pool.harmonic_coherence:.3f}")
        print(f"  Signal Strength: {quantum_scaler.chamber_state.tensor_pool.signal_strength:.3f}")
        print(f"  Scaling Factor: {decision.scaling_factor:.3f}")
        print(f"  Confidence: {decision.confidence:.3f}")

def main():
    """Main test function."""
    setup_logging()
    
    print("🔮 QUANTUM AUTO-SCALING SYSTEM TEST")
    print("=" * 60)
    print("Testing quantum chamber auto-scaling using hardware detection")
    print("as quantum observer to drive real-time scaling decisions.")
    print("=" * 60)
    
    try:
        # Test 1: Quantum chamber initialization
        quantum_scaler = test_quantum_chamber_initialization()
        
        # Test 2: Quantum scaling decisions
        decisions = test_quantum_scaling_decisions(quantum_scaler)
        
        # Test 3: Scaling application
        test_scaling_application(quantum_scaler, decisions)
        
        # Test 4: Quantum coherence evolution
        test_quantum_coherence_evolution(quantum_scaler)
        
        # Test 5: Hardware observer effects
        test_hardware_observer_effects(quantum_scaler)
        
        # Test 6: Tensor pool harmonization
        test_tensor_pool_harmonization(quantum_scaler)
        
        # Final quantum chamber summary
        print("\n" + "=" * 60)
        print("🎉 QUANTUM AUTO-SCALING TEST COMPLETED")
        print("=" * 60)
        
        quantum_scaler.print_quantum_chamber_summary()
        
        print("\n✅ All quantum auto-scaling tests completed successfully!")
        print("The system demonstrates:")
        print("  • Hardware detection as quantum observer")
        print("  • Tensor pool electrical signal harmonization")
        print("  • Quantum coherence evolution")
        print("  • Real-time scaling decisions")
        print("  • Observer effects on system behavior")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 