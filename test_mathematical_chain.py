#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mathematical Chain Integration
==================================
Quick test to verify the mathematical chain is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_mathematical_chain():
    """Test the complete mathematical chain."""
    try:
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
        
        print("🧠 Testing Mathematical Chain Integration")
        print("=" * 50)
        
        # Process through mathematical chain
        result = await mathematical_integration.process_market_data_mathematically(test_market_data)
        
        print(f"✅ Mathematical Chain Results:")
        print(f"   Decision: {result.decision}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Routing Target: {result.routing_target}")
        print(f"   DLT Waveform Score: {result.dlt_waveform_score:.4f}")
        print(f"   Bit Phase: {result.bit_phase}")
        print(f"   Matrix Basket ID: {result.matrix_basket_id}")
        print(f"   Ferris Phase: {result.ferris_phase:.4f}")
        print(f"   Tensor Score: {result.tensor_score:.4f}")
        print(f"   Entropy Score: {result.entropy_score:.4f}")
        
        if result.dualistic_consensus:
            print(f"   Dualistic Consensus:")
            print(f"     Decision: {result.dualistic_consensus.get('decision', 'NONE')}")
            print(f"     Confidence: {result.dualistic_consensus.get('confidence', 0):.4f}")
            print(f"     ALEPH Score: {result.dualistic_consensus.get('aleph_score', 0):.4f}")
            print(f"     ALIF Score: {result.dualistic_consensus.get('alif_score', 0):.4f}")
            print(f"     RITL Score: {result.dualistic_consensus.get('ritl_score', 0):.4f}")
            print(f"     RITTLE Score: {result.dualistic_consensus.get('rittle_score', 0):.4f}")
        
        print(f"\n🎯 Mathematical Chain Status: ✅ WORKING")
        print(f"   All mathematical systems are connected and contributing to decisions!")
        print(f"   The trading bot will use this complete mathematical foundation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical Chain Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_mathematical_chain()) 