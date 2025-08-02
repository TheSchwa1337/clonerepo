#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 ENTROPY SIGNAL INTEGRATION TEST
==================================

Test the entropy signal integration system to ensure it's working properly.
"""

import sys
import time
from typing import List, Tuple

def test_entropy_integration():
    """Test the entropy signal integration system."""
    print("🧪 Testing Entropy Signal Integration")
    print("="*50)
    
    try:
        # Import the entropy signal integrator
        from AOI_Base_Files_Schwabot.core.entropy_signal_integration import initialize_entropy_integration
        
        # Initialize the integrator
        integrator = initialize_entropy_integration()
        print("✅ Entropy Signal Integrator initialized")
        
        # Create sample order book data
        bids: List[Tuple[float, float]] = [
            (50000.0, 1.5),  # (price, volume)
            (49999.0, 2.0),
            (49998.0, 1.0),
            (49997.0, 0.5),
            (49996.0, 1.2)
        ]
        
        asks: List[Tuple[float, float]] = [
            (50001.0, 1.0),  # (price, volume)
            (50002.0, 1.5),
            (50003.0, 2.0),
            (50004.0, 0.8),
            (50005.0, 1.3)
        ]
        
        print(f"📊 Sample order book:")
        print(f"   Bids: {bids[:3]}...")
        print(f"   Asks: {asks[:3]}...")
        
        # Process entropy signal
        signal = integrator.process_entropy_signal(bids, asks)
        
        print(f"🌊 Entropy Signal Results:")
        print(f"   Entropy Value: {signal.entropy_value:.6f}")
        print(f"   Routing State: {signal.routing_state}")
        print(f"   Quantum State: {signal.quantum_state}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Processing Time: {signal.metadata.get('processing_time_ms', 0):.2f}ms")
        
        # Test with different market conditions
        print(f"\n🔄 Testing different market conditions:")
        
        # High volatility scenario
        high_vol_bids = [(50000.0, 0.1), (49995.0, 0.1), (49990.0, 0.1)]
        high_vol_asks = [(50010.0, 0.1), (50015.0, 0.1), (50020.0, 0.1)]
        
        high_vol_signal = integrator.process_entropy_signal(high_vol_bids, high_vol_asks)
        print(f"   High Volatility: {high_vol_signal.entropy_value:.6f} -> {high_vol_signal.routing_state}")
        
        # Low volatility scenario
        low_vol_bids = [(50000.0, 1.0), (49999.5, 1.0), (49999.0, 1.0)]
        low_vol_asks = [(50000.5, 1.0), (50001.0, 1.0), (50001.5, 1.0)]
        
        low_vol_signal = integrator.process_entropy_signal(low_vol_bids, low_vol_asks)
        print(f"   Low Volatility: {low_vol_signal.entropy_value:.6f} -> {low_vol_signal.routing_state}")
        
        # Get performance metrics
        metrics = integrator.get_performance_metrics()
        if metrics:
            latest_metric = metrics[-1]
            print(f"\n📈 Performance Metrics:")
            print(f"   Detection Rate: {latest_metric.entropy_detection_rate:.3f}")
            print(f"   Signal Latency: {latest_metric.signal_latency_ms:.2f}ms")
            print(f"   Routing Accuracy: {latest_metric.routing_accuracy:.3f}")
            print(f"   Quantum Activation Rate: {latest_metric.quantum_state_activation_rate:.3f}")
        
        # Get timing cycles
        cycles = integrator.get_timing_cycles()
        print(f"\n⏱️ Timing Cycles:")
        for cycle_name, cycle in cycles.items():
            print(f"   {cycle_name}: {cycle.current_interval_ms}ms (multiplier: {cycle.entropy_multiplier:.2f})")
        
        print(f"\n✅ Entropy Signal Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Entropy Signal Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_entropy_integration()
    sys.exit(0 if success else 1) 