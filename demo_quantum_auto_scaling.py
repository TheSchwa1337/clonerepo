#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Auto-Scaling Demo
=========================

Simple demonstration of the quantum auto-scaling system that uses
hardware detection as a quantum observer to drive real-time scaling decisions.

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

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def demo_quantum_auto_scaling():
    """Demonstrate quantum auto-scaling system."""
    print("🔮 QUANTUM AUTO-SCALING SYSTEM DEMO")
    print("=" * 60)
    print("Demonstrating quantum chamber auto-scaling using hardware detection")
    print("as quantum observer to drive real-time scaling decisions.")
    print("=" * 60)
    
    try:
        # Import the quantum auto-scaler
        from core.quantum_auto_scaler import QuantumAutoScaler, ScalingTrigger
        
        print("\n🧪 Initializing Quantum Chamber...")
        
        # Initialize quantum auto-scaler
        quantum_scaler = QuantumAutoScaler()
        
        # Print initial quantum chamber summary
        quantum_scaler.print_quantum_chamber_summary()
        
        print("\n🧪 Testing Quantum Scaling Decisions...")
        print("-" * 50)
        
        # Test different scenarios
        scenarios = [
            {
                "name": "Normal Trading",
                "market_entropy": 0.5,
                "thermal_state": 0.3,
                "profit_potential": 0.6
            },
            {
                "name": "High Market Volatility",
                "market_entropy": 0.9,
                "thermal_state": 0.3,
                "profit_potential": 0.6
            },
            {
                "name": "High Thermal Load",
                "market_entropy": 0.5,
                "thermal_state": 0.8,
                "profit_potential": 0.6
            },
            {
                "name": "High Profit Opportunity",
                "market_entropy": 0.5,
                "thermal_state": 0.3,
                "profit_potential": 0.9
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\nTest {i}: {scenario['name']}")
            print("-" * 30)
            
            decision = quantum_scaler.compute_quantum_scaling(
                market_entropy=scenario['market_entropy'],
                thermal_state=scenario['thermal_state'],
                profit_potential=scenario['profit_potential']
            )
            
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
        
        print("\n🧪 Applying Scaling Decisions...")
        print("-" * 50)
        
        # Apply the last decision (high profit opportunity)
        last_decision = quantum_scaler.compute_quantum_scaling(
            market_entropy=0.5,
            thermal_state=0.3,
            profit_potential=0.9
        )
        
        success = quantum_scaler.apply_scaling_decision(last_decision)
        
        if success:
            print("✅ Quantum scaling decision applied successfully!")
            
            # Show updated memory configuration
            memory_config = quantum_scaler.chamber_state.memory_config
            print(f"\nUpdated Memory Configuration:")
            print(f"  TIC Map Sizes: {dict(memory_config.tic_map_sizes)}")
            print(f"  Cache Sizes: {dict(memory_config.cache_sizes)}")
            print(f"  Scaling Multiplier: {quantum_scaler.chamber_state.scaling_multiplier:.3f}")
        else:
            print("❌ Failed to apply quantum scaling decision")
        
        print("\n🧪 Quantum Chamber Status...")
        print("-" * 50)
        
        # Get final quantum chamber status
        status = quantum_scaler.get_quantum_chamber_status()
        
        print(f"Quantum State: {status['quantum_state']}")
        print(f"Coherence Score: {status['coherence_score']:.3f}")
        print(f"Entropy Value: {status['entropy_value']:.3f}")
        print(f"Quantum Phase: {status['quantum_phase']:.3f} rad")
        print(f"Observer Coherence: {status['observer_coherence']:.3f}")
        print(f"Tensor Harmonic Coherence: {status['tensor_harmonic_coherence']:.3f}")
        print(f"Tensor Signal Strength: {status['tensor_signal_strength']:.3f}")
        print(f"Scaling History: {status['scaling_history_count']} decisions")
        
        print("\n" + "=" * 60)
        print("🎉 QUANTUM AUTO-SCALING DEMO COMPLETED")
        print("=" * 60)
        
        print("\n✅ The quantum auto-scaling system demonstrates:")
        print("  • Hardware detection as quantum observer")
        print("  • Tensor pool electrical signal harmonization")
        print("  • Real-time scaling decisions based on market conditions")
        print("  • Observer effects on system behavior")
        print("  • Performance-preserving memory management")
        
        print("\n🔮 Quantum Chamber Logic Implemented:")
        print("  • SHA-256 = Crystalline math for tiny logic brains")
        print("  • Tensor Pool = Electrical signal harmonizer")
        print("  • Hardware Detection = Quantum observer")
        print("  • GPU/CPU/RAM swap = Variable collapse states")
        print("  • Timing ∆ between inputs = ψ wave triggers")
        print("  • Observer Effect = AI memory state pivots")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the quantum auto-scaler module is available.")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    setup_logging()
    
    success = demo_quantum_auto_scaling()
    
    if success:
        print("\n🚀 Ready to implement quantum auto-scaling in your trading system!")
        print("The system automatically scales based on:")
        print("  1. Hardware capabilities (quantum observer state)")
        print("  2. Market conditions (entropy pings)")
        print("  3. Thermal state (mirror feedback)")
        print("  4. Profit potential (harmonic tensor sync)")
        return 0
    else:
        print("\n❌ Demo failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 