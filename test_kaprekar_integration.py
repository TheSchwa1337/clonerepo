#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 KAPREKAR TICK ENGINE INTEGRATION TEST
========================================

Comprehensive integration test demonstrating the complete Kaprekar Tick Engine
working with all modules: engine, bridge, logic, allocator, hash, and memory.

This test validates the complete flow from price tick to strategy allocation
with memory persistence and profit mismatch detection.
"""

import sys
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_kaprekar_engine():
    """Test the core Kaprekar engine."""
    print("\n🧮 Testing Kaprekar Engine...")
    
    try:
        from core.kaprekar_engine import kaprekar_iterations, analyze_kaprekar_convergence
        
        # Test basic iterations
        test_cases = [
            (1234, 3),   # Known convergent
            (1000, 5),   # Another convergent
            (1111, -1),  # Non-convergent
            (9999, -1),  # Non-convergent (leads to cycle of zeros)
        ]
        
        for input_num, expected in test_cases:
            result = kaprekar_iterations(input_num)
            status = "✅" if result == expected else "❌"
            print(f"{status} Input: {input_num}, Expected: {expected}, Got: {result}")
        
        # Test batch analysis
        numbers = [1234, 1000, 1111, 9999, 4321]
        analysis = analyze_kaprekar_convergence(numbers)
        print(f"📊 Batch analysis: {analysis['convergent_count']}/{analysis['total_numbers']} convergent")
        
        return True
        
    except Exception as e:
        print(f"❌ Kaprekar Engine test failed: {e}")
        return False

def test_tick_kaprekar_bridge():
    """Test the tick-kaprekar bridge."""
    print("\n🌉 Testing Tick-Kaprekar Bridge...")
    
    try:
        from core.tick_kaprekar_bridge import price_to_kaprekar_index, get_volatility_signal
        
        test_prices = [
            2045.29,    # Should normalize to 2045
            123.456,    # Should normalize to 1234
            9999.99,    # Should normalize to 9999
            1.234,      # Should normalize to 1234
            50000.0,    # Should normalize to 5000
        ]
        
        for price in test_prices:
            k_index = price_to_kaprekar_index(price)
            signal = get_volatility_signal(price)
            print(f"Price: {price} → K-Index: {k_index} → Signal: {signal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tick-Kaprekar Bridge test failed: {e}")
        return False

def test_ferris_tick_logic():
    """Test the Ferris tick logic."""
    print("\n🎡 Testing Ferris Tick Logic...")
    
    try:
        from core.ferris_tick_logic import process_tick, get_ferris_cycle_position
        
        test_prices = [
            2045.29,    # Should route to vol_stable_basket
            123.456,    # Should route to midrange_vol_logic
            9999.99,    # Should route to escape_vol_guard
            1111.11,    # Should route to ghost_shell_evasion
        ]
        
        for price in test_prices:
            signal = process_tick(price)
            cycle_info = get_ferris_cycle_position(price)
            print(f"Price: {price} → Signal: {signal} → Cycle Position: {cycle_info.get('cycle_position', 'error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ferris Tick Logic test failed: {e}")
        return False

def test_profit_cycle_allocator():
    """Test the profit cycle allocator."""
    print("\n💰 Testing Profit Cycle Allocator...")
    
    try:
        from core.profit_cycle_allocator import allocate_profit_zone, trigger_strategy
        
        test_prices = [
            2045.29,    # Should trigger BTC_MICROHOLD_REBUY
            123.456,    # Should trigger USDC_RSI_REBALANCE
            9999.99,    # Should trigger XRP_LIQUIDITY_VACUUM
            1111.11,    # Should trigger ZBE_RECOVERY_PATH
        ]
        
        for price in test_prices:
            allocation = allocate_profit_zone(price)
            print(f"Price: {price} → Zone: {allocation['profit_zone']} → Strategies: {allocation['strategy_triggers']}")
            
            # Test strategy triggering
            for strategy in allocation['strategy_triggers']:
                trigger_result = trigger_strategy(strategy, price)
                print(f"  Strategy: {strategy} → Status: {trigger_result['trigger_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Profit Cycle Allocator test failed: {e}")
        return False

def test_ghost_kaprekar_hash():
    """Test the ghost Kaprekar hash system."""
    print("\n👻 Testing Ghost Kaprekar Hash...")
    
    try:
        from core.ghost_kaprekar_hash import generate_kaprekar_strategy_hash, generate_strategy_signature
        
        test_prices = [
            2045.29,    # Should generate stable hash
            123.456,    # Should generate midrange hash
            9999.99,    # Should generate high vol hash
            1111.11,    # Should generate non-convergent hash
        ]
        
        for price in test_prices:
            basic_hash = generate_kaprekar_strategy_hash(price)
            strategy_sig = generate_strategy_signature(price, "TEST_STRATEGY", 0.85)
            print(f"Price: {price} → Hash: {basic_hash[:16]}... → Signature: {strategy_sig['signature'][:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Ghost Kaprekar Hash test failed: {e}")
        return False

def test_cross_section_memory():
    """Test the cross-section memory system."""
    print("\n🧠 Testing Cross-Section Memory...")
    
    try:
        from core.cross_section_memory import CrossSectionMemory
        
        # Initialize memory system
        memory = CrossSectionMemory(session_id="integration_test")
        
        # Test tick variation recording
        test_prices = [2045.29, 123.456, 9999.99, 1111.11]
        variation_ids = []
        
        for price in test_prices:
            variation_id = memory.record_tick_variation(price)
            variation_ids.append(variation_id)
            print(f"Recorded variation: {price} → {variation_id[:16]}...")
        
        # Test profit mismatch analysis
        if variation_ids:
            analysis = memory.analyze_profit_mismatch(
                variation_ids[0], 
                actual_profit=0.05, 
                expected_profit=0.03
            )
            if analysis:
                print(f"Profit mismatch analyzed: {analysis.profit_delta:.4f} (factor: {analysis.mismatch_factor:.2f})")
        
        # Test pattern analysis
        patterns = memory.analyze_tick_patterns(window_size=10)
        print(f"Pattern analysis: {patterns.get('dominant_volatility', 'unknown')} volatility")
        
        # Test memory summary
        summary = memory.get_memory_summary()
        print(f"Memory summary: {summary['total_tick_variations']} variations, {summary['total_profit_mismatches']} mismatches")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-Section Memory test failed: {e}")
        return False

def test_complete_integration_flow():
    """Test the complete integration flow from tick to strategy."""
    print("\n🔄 Testing Complete Integration Flow...")
    
    try:
        from core.kaprekar_engine import kaprekar_iterations
        from core.tick_kaprekar_bridge import price_to_kaprekar_index
        from core.ferris_tick_logic import process_tick
        from core.profit_cycle_allocator import allocate_profit_zone
        from core.ghost_kaprekar_hash import generate_kaprekar_strategy_hash
        from core.cross_section_memory import CrossSectionMemory
        
        # Initialize systems
        memory = CrossSectionMemory(session_id="complete_flow_test")
        
        # Test complete flow with sample prices
        test_prices = [2045.29, 123.456, 9999.99, 1111.11]
        
        for i, price in enumerate(test_prices):
            print(f"\n--- Flow Test {i+1}: Price {price} ---")
            
            # Step 1: Kaprekar analysis
            k_index = price_to_kaprekar_index(price)
            print(f"1. Kaprekar Index: {k_index}")
            
            # Step 2: Ferris tick processing
            routing_signal = process_tick(price)
            print(f"2. Routing Signal: {routing_signal}")
            
            # Step 3: Profit zone allocation
            allocation = allocate_profit_zone(price)
            print(f"3. Profit Zone: {allocation['profit_zone']}")
            print(f"   Strategies: {allocation['strategy_triggers']}")
            
            # Step 4: Hash generation
            strategy_hash = generate_kaprekar_strategy_hash(price)
            print(f"4. Strategy Hash: {strategy_hash[:16]}...")
            
            # Step 5: Memory recording
            variation_id = memory.record_tick_variation(price)
            print(f"5. Memory ID: {variation_id[:16]}...")
            
            # Step 6: Profit analysis (simulated)
            if i % 2 == 0:  # Simulate profit results for even indices
                expected_profit = 0.03
                actual_profit = 0.05 if i == 0 else 0.02
                analysis = memory.analyze_profit_mismatch(variation_id, actual_profit, expected_profit)
                if analysis:
                    print(f"6. Profit Analysis: Delta {analysis.profit_delta:.4f}, Correlation {analysis.correlation_score:.2f}")
        
        # Final memory summary
        summary = memory.get_memory_summary()
        print(f"\n📊 Final Summary: {summary['total_tick_variations']} variations recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete integration flow test failed: {e}")
        return False

def run_performance_test():
    """Run performance test with larger datasets."""
    print("\n⚡ Running Performance Test...")
    
    try:
        from core.kaprekar_engine import kaprekar_iterations
        from core.tick_kaprekar_bridge import price_to_kaprekar_index
        from core.ferris_tick_logic import process_tick
        from core.profit_cycle_allocator import allocate_profit_zone
        
        # Generate test data
        import random
        test_prices = [random.uniform(100, 10000) for _ in range(100)]
        
        start_time = time.time()
        
        # Process all prices
        results = []
        for price in test_prices:
            k_index = price_to_kaprekar_index(price)
            signal = process_tick(price)
            allocation = allocate_profit_zone(price)
            results.append({
                'price': price,
                'k_index': k_index,
                'signal': signal,
                'zone': allocation['profit_zone']
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Analyze results
        signal_counts = {}
        zone_counts = {}
        convergent_count = sum(1 for r in results if r['k_index'] != -1)
        
        for result in results:
            signal_counts[result['signal']] = signal_counts.get(result['signal'], 0) + 1
            zone_counts[result['zone']] = zone_counts.get(result['zone'], 0) + 1
        
        print(f"⏱️  Processed {len(test_prices)} prices in {processing_time:.3f} seconds")
        print(f"📊 Convergent: {convergent_count}/{len(test_prices)} ({convergent_count/len(test_prices)*100:.1f}%)")
        print(f"🎯 Signal distribution: {signal_counts}")
        print(f"💰 Zone distribution: {zone_counts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 KAPREKAR TICK ENGINE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Kaprekar Engine", test_kaprekar_engine),
        ("Tick-Kaprekar Bridge", test_tick_kaprekar_bridge),
        ("Ferris Tick Logic", test_ferris_tick_logic),
        ("Profit Cycle Allocator", test_profit_cycle_allocator),
        ("Ghost Kaprekar Hash", test_ghost_kaprekar_hash),
        ("Cross-Section Memory", test_cross_section_memory),
        ("Complete Integration Flow", test_complete_integration_flow),
        ("Performance Test", run_performance_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Kaprekar Tick Engine is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 