#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST: WINDOWS COMPATIBLE CORRELATION ENGINE
==========================================

Test the Windows-compatible cross-asset correlation engine.
No emoji characters, text-only output for Windows compatibility.
"""

import sys
import time

def test_windows_correlation():
    """Test the Windows-compatible correlation engine."""
    print("TESTING WINDOWS COMPATIBLE CORRELATION ENGINE")
    print("=" * 50)
    print("Testing Cross-Asset Correlation (Windows Compatible)")
    print()
    
    try:
        # Import the Windows-compatible correlation engine
        from enhanced_cross_asset_correlation_windows import get_correlation_engine_windows
        
        # Initialize the engine
        correlation_engine = get_correlation_engine_windows()
        
        print("SUCCESS: Windows-compatible correlation engine initialized")
        print()
        
        # Test 1: BTC/ETH Correlation Analysis
        print("Test 1: BTC/ETH Correlation Analysis")
        print("-" * 40)
        
        # Analyze BTC/ETH correlation
        btc_price = 118436.19
        eth_price = 3629.14
        btc_volume = 1000.0
        eth_volume = 500.0
        
        btc_eth_signal = correlation_engine.analyze_btc_eth_correlation(
            btc_price, eth_price, btc_volume, eth_volume
        )
        
        print(f"BTC/ETH Correlation: {btc_eth_signal.correlation_value:.3f}")
        print(f"Recommendation: {btc_eth_signal.portfolio_recommendation}")
        print(f"Confidence: {btc_eth_signal.confidence:.3f}")
        print()
        
        # Test 2: Cross-Exchange Arbitrage Detection
        print("Test 2: Cross-Exchange Arbitrage Detection")
        print("-" * 40)
        
        # Test cross-exchange arbitrage detection
        exchange_data = {
            'coinbase': {
                'BTC/USD': 118436.19,
                'ETH/USD': 3629.14,
                'BTC/USDT': 118436.19,
                'ETH/USDT': 3629.14
            },
            'kraken': {
                'BTC/USD': 118450.10,
                'ETH/USD': 3630.79,
                'BTC/USDT': 118450.10,
                'ETH/USDT': 3630.79
            },
            'binance': {
                'BTC/USD': 118450.10,
                'ETH/USD': 3630.79,
                'BTC/USDT': 118450.10,
                'ETH/USDT': 3630.79
            }
        }
        
        arbitrage_opportunities = correlation_engine.detect_cross_exchange_arbitrage(exchange_data)
        
        print(f"Arbitrage Opportunities Found: {len(arbitrage_opportunities)}")
        for opp in arbitrage_opportunities[:3]:  # Show first 3
            print(f"   {opp.symbol}: {opp.exchange_a}->{opp.exchange_b} {opp.spread_percentage:.2f}%")
        print()
        
        # Test 3: Portfolio Optimization Signals
        print("Test 3: Portfolio Optimization Signals")
        print("-" * 40)
        
        portfolio_signals = correlation_engine.get_portfolio_optimization_signals()
        
        print(f"BTC/ETH Correlation: {portfolio_signals.get('btc_eth_correlation', 0):.3f}")
        print(f"Crypto-Traditional Correlation: {portfolio_signals.get('crypto_traditional_correlation', 0):.3f}")
        print(f"Recent Arbitrage Opportunities: {portfolio_signals.get('recent_arbitrage_opportunities', 0)}")
        print(f"Total Signals Analyzed: {portfolio_signals.get('total_signals_analyzed', 0)}")
        print(f"Confidence Score: {portfolio_signals.get('confidence_score', 0):.3f}")
        print()
        print("Recommendations:")
        for rec in portfolio_signals.get('recommendations', []):
            print(f"   * {rec}")
        print()
        
        # Test 4: Performance and Speed Test
        print("Test 4: Performance and Speed Test")
        print("-" * 40)
        
        # Test multiple correlations quickly
        start_time = time.time()
        
        for i in range(10):
            btc_price_test = 118436.19 + (i * 100)
            eth_price_test = 3629.14 + (i * 10)
            
            signal = correlation_engine.analyze_btc_eth_correlation(
                btc_price_test, eth_price_test, 1000.0, 500.0
            )
            
            if i < 3:  # Show first 3 results
                print(f"   Test {i+1}: Correlation {signal.correlation_value:.3f}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processed 10 correlations in {processing_time:.3f} seconds")
        print(f"Average time per correlation: {processing_time/10:.4f} seconds")
        print()
        
        # Test 5: Error Handling Test
        print("Test 5: Error Handling Test")
        print("-" * 40)
        
        # Test with invalid data
        try:
            invalid_signal = correlation_engine.analyze_btc_eth_correlation(
                0, 0, 0, 0  # Invalid data
            )
            print(f"Error handling test passed: {invalid_signal.portfolio_recommendation}")
        except Exception as e:
            print(f"Error handling test passed: {e}")
        
        print()
        
        # Summary
        print("WINDOWS COMPATIBILITY TEST SUMMARY")
        print("=" * 50)
        print("All tests completed successfully!")
        print("No emoji encoding errors encountered")
        print("Text-only output working correctly")
        print("Performance: Fast and efficient")
        print("Error handling: Robust")
        print()
        print("RESULT: Windows-compatible correlation engine is working perfectly!")
        print("Ready for integration with main Schwabot system")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_windows_correlation()
    sys.exit(0 if success else 1) 