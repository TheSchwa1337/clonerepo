#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phantom Mode Test Script
========================

Test script to demonstrate Schwabot Phantom Mode functionality.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime, timedelta

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.phantom_mode_engine import PhantomModeEngine, PhantomConfig
from core.phantom_mode_integration import PhantomModeIntegration

def simulate_market_data(duration_minutes: int = 60, volatility: float = 0.02) -> tuple:
    """
    Simulate realistic market data with BTC/USDC price movements.
    """
    print(f"🎯 Simulating {duration_minutes} minutes of BTC/USDC market data...")
    
    base_price = 50000.0  # Starting BTC price
    prices = [base_price]
    timestamps = [time.time()]
    volumes = [1000.0]  # Starting volume
    
    # Time intervals (1-minute data)
    interval_seconds = 60
    
    for i in range(1, duration_minutes):
        # Simulate price movement with some trend and noise
        trend = np.random.normal(0, volatility * 0.1)  # Small trend component
        noise = np.random.normal(0, volatility)  # Random noise
        volume_impact = np.random.normal(0, volatility * 0.05)  # Volume impact
        
        # Calculate new price
        price_change = trend + noise + volume_impact
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price stays reasonable
        new_price = max(new_price, 1000.0)  # Minimum $1000
        new_price = min(new_price, 100000.0)  # Maximum $100,000
        
        prices.append(new_price)
        timestamps.append(timestamps[-1] + interval_seconds)
        
        # Simulate volume changes
        volume_change = np.random.normal(0, 0.1)
        new_volume = volumes[-1] * (1 + volume_change)
        new_volume = max(new_volume, 100.0)  # Minimum volume
        volumes.append(new_volume)
        
    return prices, timestamps, volumes

def test_phantom_mode_engine():
    """Test the core Phantom Mode engine."""
    print("\n🧠 Testing Phantom Mode Engine...")
    
    # Create Phantom Mode engine with custom config
    config = PhantomConfig(
        wec_window_size=128,
        zbe_threshold=0.25,
        pt_phase_threshold=0.6,
        pt_entropy_threshold=0.4
    )
    
    engine = PhantomModeEngine(config)
    
    # Simulate market data
    prices, timestamps, volumes = simulate_market_data(30)  # 30 minutes
    
    print(f"📊 Processed {len(prices)} price points")
    
    # Process through Phantom Mode
    decisions = []
    for i in range(10, len(prices)):  # Start after initial window
        window_prices = prices[:i+1]
        window_timestamps = timestamps[:i+1]
        
        decision = engine.process_market_data(window_prices, window_timestamps, volumes[:i+1])
        decisions.append(decision)
        
        if decision.get('action') == 'execute_trade':
            print(f"🔥 Phantom Mode triggered at price ${prices[i]:.2f}")
            print(f"   Confidence: {decision.get('confidence', 0):.3f}")
            print(f"   Phase Alignment: {decision.get('phase_alignment', 0):.3f}")
            print(f"   Bloom Probability: {decision.get('bloom_probability', 0):.3f}")
            print("   ---")
    
    # Get final status
    status = engine.get_phantom_status()
    print(f"\n📈 Phantom Mode Status:")
    print(f"   Active: {status['phantom_mode_active']}")
    print(f"   Total Trades: {status['total_trades']}")
    print(f"   Recent Accuracy: {status['recent_accuracy']:.3f}")
    print(f"   Bitmap Strength: {status['bitmap_strength']:.2f}")
    
    return engine

def test_phantom_integration():
    """Test the full Phantom Mode integration with node management."""
    print("\n🔗 Testing Phantom Mode Integration...")
    
    # Create integration
    integration = PhantomModeIntegration()
    
    # Start monitoring
    integration.start_monitoring()
    
    # Simulate market data
    prices, timestamps, volumes = simulate_market_data(45)  # 45 minutes
    
    print(f"📊 Processing {len(prices)} price points through integration...")
    
    # Process through integration
    integrated_decisions = []
    for i in range(15, len(prices)):  # Start after initial window
        window_prices = prices[:i+1]
        window_timestamps = timestamps[:i+1]
        
        decision = integration.process_market_data(window_prices, window_timestamps, volumes[:i+1])
        integrated_decisions.append(decision)
        
        if decision.get('action') == 'execute_trade':
            print(f"🚀 Integrated Phantom Trade triggered!")
            print(f"   Price: ${prices[i]:.2f}")
            print(f"   Confidence: {decision.get('confidence', 0):.3f}")
            
            # Check execution details
            execution = decision.get('execution', {})
            if execution:
                nodes = execution.get('execution_nodes', [])
                print(f"   Execution Nodes: {len(nodes)}")
                for node in nodes:
                    print(f"     - {node['node']}: {node['load']:.1%} load ({node['role']})")
            
            print("   ---")
    
    # Get system status
    system_status = integration.get_system_status()
    print(f"\n🏗️ System Status:")
    print(f"   Integration Active: {system_status['integration_active']}")
    print(f"   Phantom Mode Active: {system_status['phantom_mode']['phantom_mode_active']}")
    
    # Node status
    print(f"\n🖥️ Node Status:")
    for node_name, node_info in system_status['node_status'].items():
        print(f"   {node_name}:")
        print(f"     Temperature: {node_info['temperature']:.1f}°C")
        print(f"     Memory Usage: {node_info['memory_usage']:.1%}")
        print(f"     GPU Utilization: {node_info['gpu_utilization']:.1%}")
        print(f"     Thermal Warning: {node_info['thermal_warning']}")
        print(f"     Entropy Drift: {node_info['entropy_drift']:.3f}")
    
    # Stop monitoring
    integration.stop_monitoring()
    
    return integration

def demonstrate_thermal_management():
    """Demonstrate thermal management and load balancing."""
    print("\n🌡️ Demonstrating Thermal Management...")
    
    integration = PhantomModeIntegration()
    integration.start_monitoring()
    
    # Simulate thermal stress scenario
    print("Simulating thermal stress on XFX 7970...")
    
    # Override XFX temperature to simulate thermal stress
    xfx_status = integration.load_balancer.node_status['xfx_7970']
    xfx_status.temperature = 82.0  # Near thermal limit
    xfx_status.thermal_warning = True
    
    # Process market data
    prices, timestamps, volumes = simulate_market_data(20)
    
    for i in range(10, len(prices)):
        window_prices = prices[:i+1]
        window_timestamps = timestamps[:i+1]
        
        decision = integration.process_market_data(window_prices, window_timestamps, volumes[:i+1])
        
        if decision.get('action') == 'execute_trade':
            execution = decision.get('execution', {})
            if execution:
                load_dist = execution.get('load_distribution', {})
                print(f"\n🔥 Thermal Management Response:")
                print(f"   XFX 7970: {load_dist.get('xfx_7970', {}).get('load_percentage', 0):.1%} load")
                print(f"   Pi 4: {load_dist.get('pi4', {}).get('load_percentage', 0):.1%} load")
                print(f"   GTX 1070: {load_dist.get('gtx_1070', {}).get('load_percentage', 0):.1%} load")
                break
    
    integration.stop_monitoring()

def export_phantom_data():
    """Export Phantom Mode data for analysis."""
    print("\n📊 Exporting Phantom Mode Data...")
    
    # Create engine and process some data
    engine = PhantomModeEngine()
    prices, timestamps, volumes = simulate_market_data(60)
    
    for i in range(20, len(prices)):
        window_prices = prices[:i+1]
        window_timestamps = timestamps[:i+1]
        engine.process_market_data(window_prices, window_timestamps, volumes[:i+1])
    
    # Export data
    phantom_data = engine.export_phantom_data()
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phantom_mode_data_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(phantom_data, f, indent=2)
    
    print(f"📁 Phantom Mode data exported to: {filename}")
    print(f"   Trade History: {len(phantom_data['trade_history'])} records")
    print(f"   Trigger History: {len(phantom_data['trigger_history'])} records")
    print(f"   Profit History: {len(phantom_data['profit_history'])} records")

def main():
    """Main test function."""
    print("🧬 SCHWABOT PHANTOM MODE TEST SUITE")
    print("=" * 50)
    
    try:
        # Test core engine
        engine = test_phantom_mode_engine()
        
        # Test full integration
        integration = test_phantom_integration()
        
        # Demonstrate thermal management
        demonstrate_thermal_management()
        
        # Export data
        export_phantom_data()
        
        print("\n✅ All Phantom Mode tests completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Wave Entropy Capture (WEC)")
        print("   • Zero-Bound Entropy Compression (ZBE)")
        print("   • Bitmap Drift Memory Encoding (BDME)")
        print("   • Ghost Phase Alignment Function (GPAF)")
        print("   • Phantom Trigger Function (PTF)")
        print("   • Node Load Balancing")
        print("   • Thermal Management")
        print("   • Recursive Learning")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 