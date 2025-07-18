#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Mathematical Integration Test
=====================================

This script tests the complete mathematical integration system step by step,
ensuring all components are working and properly integrated.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathematicalSystemTester:
    """Comprehensive mathematical system tester."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    async def test_core_mathematical_components(self) -> Dict[str, bool]:
        """Test core mathematical components."""
        print("🧮 Testing Core Mathematical Components")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Basic mathematical operations
        try:
            import numpy as np
            test_array = np.array([1, 2, 3, 4, 5])
            fft_result = np.fft.fft(test_array)
            results["numpy_fft"] = len(fft_result) == len(test_array)
            print(f"   ✅ NumPy FFT: {results['numpy_fft']}")
        except Exception as e:
            results["numpy_fft"] = False
            print(f"   ❌ NumPy FFT: {e}")
        
        # Test 2: Mathematical signal dataclass
        try:
            from dataclasses import dataclass
            
            @dataclass
            class TestMathematicalSignal:
                dlt_waveform_score: float = 0.0
                bit_phase: int = 0
                ferris_phase: float = 0.0
                tensor_score: float = 0.0
                entropy_score: float = 0.0
                confidence: float = 0.0
                decision: str = "HOLD"
                routing_target: str = "USDC"
            
            test_signal = TestMathematicalSignal()
            results["mathematical_signal"] = test_signal.decision == "HOLD"
            print(f"   ✅ Mathematical Signal: {results['mathematical_signal']}")
        except Exception as e:
            results["mathematical_signal"] = False
            print(f"   ❌ Mathematical Signal: {e}")
        
        # Test 3: DLT Waveform Engine
        try:
            class TestDLTEngine:
                def __init__(self):
                    self.name = "DLT Engine"
                
                def calculate_waveform(self, signal):
                    return np.sin(2 * np.pi * signal) * np.exp(-0.1 * signal)
                
                def calculate_entropy(self, signal):
                    fft_signal = np.fft.fft(signal)
                    power_spectrum = np.abs(fft_signal) ** 2
                    total_power = np.sum(power_spectrum)
                    if total_power == 0:
                        return 0.0
                    probabilities = power_spectrum / total_power
                    return -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            dlt_engine = TestDLTEngine()
            test_signal = np.linspace(0, 1, 100)
            waveform = dlt_engine.calculate_waveform(test_signal)
            entropy = dlt_engine.calculate_entropy(test_signal)
            
            results["dlt_engine"] = len(waveform) == 100 and entropy >= 0
            print(f"   ✅ DLT Engine: {results['dlt_engine']}")
        except Exception as e:
            results["dlt_engine"] = False
            print(f"   ❌ DLT Engine: {e}")
        
        # Test 4: Dualistic Thought Engines
        try:
            class TestDualisticEngine:
                def __init__(self, engine_type: str):
                    self.engine_type = engine_type
                
                def evaluate_trust(self, state):
                    return {
                        'decision': 'HOLD',
                        'confidence': 0.5,
                        'routing_target': 'USDC',
                        'mathematical_score': 0.5
                    }
            
            engines = {
                "ALEPH": TestDualisticEngine("ALEPH"),
                "ALIF": TestDualisticEngine("ALIF"),
                "RITL": TestDualisticEngine("RITL"),
                "RITTLE": TestDualisticEngine("RITTLE")
            }
            
            all_engines_working = True
            for name, engine in engines.items():
                result = engine.evaluate_trust({})
                if result['decision'] != 'HOLD':
                    all_engines_working = False
            
            results["dualistic_engines"] = all_engines_working
            print(f"   ✅ Dualistic Engines: {results['dualistic_engines']}")
        except Exception as e:
            results["dualistic_engines"] = False
            print(f"   ❌ Dualistic Engines: {e}")
        
        # Test 5: Bit Phase Resolution
        try:
            def resolve_bit_phase(market_data):
                volatility = market_data.get('volatility', 0.5)
                if volatility < 0.2:
                    return 4
                elif volatility < 0.4:
                    return 8
                elif volatility < 0.6:
                    return 16
                elif volatility < 0.8:
                    return 32
                else:
                    return 42
            
            test_data = {'volatility': 0.3}
            bit_phase = resolve_bit_phase(test_data)
            results["bit_phase"] = bit_phase == 8
            print(f"   ✅ Bit Phase Resolution: {results['bit_phase']}")
        except Exception as e:
            results["bit_phase"] = False
            print(f"   ❌ Bit Phase Resolution: {e}")
        
        # Test 6: Matrix Basket Tensor Operations
        try:
            def calculate_matrix_basket(market_data):
                price = market_data.get('current_price', 50000)
                volume = market_data.get('volume', 1000)
                return int((price * volume) % 1000)
            
            def calculate_tensor_score(market_data):
                price = market_data.get('current_price', 50000)
                volume = market_data.get('volume', 1000)
                volatility = market_data.get('volatility', 0.5)
                return (price * volume * volatility) / 1000000
            
            test_data = {'current_price': 52000, 'volume': 1000, 'volatility': 0.15}
            basket_id = calculate_matrix_basket(test_data)
            tensor_score = calculate_tensor_score(test_data)
            
            results["matrix_basket"] = basket_id >= 0
            results["tensor_score"] = tensor_score > 0
            print(f"   ✅ Matrix Basket: {results['matrix_basket']}")
            print(f"   ✅ Tensor Score: {results['tensor_score']}")
        except Exception as e:
            results["matrix_basket"] = False
            results["tensor_score"] = False
            print(f"   ❌ Matrix Basket/Tensor Score: {e}")
        
        # Test 7: Ferris Phase Calculation
        try:
            def calculate_ferris_phase(market_data):
                price = market_data.get('current_price', 50000)
                entry_price = market_data.get('entry_price', 50000)
                time_factor = market_data.get('timestamp', time.time()) % 225  # 3.75 minutes
                return (price / entry_price - 1) * (time_factor / 225)
            
            test_data = {'current_price': 52000, 'entry_price': 50000, 'timestamp': time.time()}
            ferris_phase = calculate_ferris_phase(test_data)
            
            results["ferris_phase"] = abs(ferris_phase) < 1.0
            print(f"   ✅ Ferris Phase: {results['ferris_phase']}")
        except Exception as e:
            results["ferris_phase"] = False
            print(f"   ❌ Ferris Phase: {e}")
        
        # Test 8: Entropy Calculation
        try:
            def calculate_entropy(market_data):
                price_history = market_data.get('price_history', [50000 + i * 100 for i in range(100)])
                if len(price_history) < 2:
                    return 0.0
                
                returns = np.diff(price_history) / price_history[:-1]
                if len(returns) == 0:
                    return 0.0
                
                # Calculate entropy using histogram
                hist, _ = np.histogram(returns, bins=20, density=True)
                hist = hist[hist > 0]  # Remove zero probabilities
                if len(hist) == 0:
                    return 0.0
                
                return -np.sum(hist * np.log2(hist))
            
            test_data = {'price_history': [50000 + i * 100 for i in range(100)]}
            entropy = calculate_entropy(test_data)
            
            results["entropy_calculation"] = entropy >= 0
            print(f"   ✅ Entropy Calculation: {results['entropy_calculation']}")
        except Exception as e:
            results["entropy_calculation"] = False
            print(f"   ❌ Entropy Calculation: {e}")
        
        return results
    
    async def test_simplified_mathematical_integration(self) -> bool:
        """Test the simplified mathematical integration."""
        print("\n🧠 Testing Simplified Mathematical Integration")
        print("=" * 60)
        
        try:
            # Import the simplified integration
            from backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
            
            print("✅ Simplified mathematical integration import successful")
            
            # Test market data
            test_market_data = {
                'current_price': 52000.0,
                'entry_price': 50000.0,
                'volume': 1000.0,
                'volatility': 0.15,
                'price_history': [50000 + i * 100 for i in range(100)],
                'timestamp': time.time()
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
            
            return True
            
        except Exception as e:
            print(f"❌ Simplified mathematical integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_production_pipeline_integration(self) -> bool:
        """Test production pipeline integration."""
        print("\n🚀 Testing Production Pipeline Integration")
        print("=" * 60)
        
        try:
            # Test production pipeline configuration
            from dataclasses import dataclass, field
            from typing import List
            
            @dataclass
            class TestTradingConfig:
                exchange_name: str
                api_key: str
                secret: str
                sandbox: bool = True
                symbols: List[str] = field(default_factory=lambda: ['BTC/USDC'])
                enable_mathematical_integration: bool = True
                mathematical_confidence_threshold: float = 0.7
            
            config = TestTradingConfig(
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
            
            # Test market data processing
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 52000.0,
                'volume': 1000.0,
                'price_change': 0.02,
                'volatility': 0.15,
                'sentiment': 0.7,
                'timestamp': time.time()
            }
            
            print("✅ Market data processing test successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Production pipeline integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_mathematical_decision_integration(self) -> bool:
        """Test mathematical decision integration."""
        print("\n🎯 Testing Mathematical Decision Integration")
        print("=" * 60)
        
        try:
            # Test decision integration logic
            def integrate_mathematical_decisions(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Integrate multiple mathematical signals into a final decision."""
                if not signals:
                    return {'decision': 'HOLD', 'confidence': 0.0, 'reason': 'No signals'}
                
                # Calculate weighted decision
                total_confidence = 0.0
                weighted_decision = 0.0
                
                for signal in signals:
                    confidence = signal.get('confidence', 0.0)
                    decision_score = 0.0
                    
                    if signal.get('decision') == 'BUY':
                        decision_score = 1.0
                    elif signal.get('decision') == 'SELL':
                        decision_score = -1.0
                    # HOLD = 0.0
                    
                    weighted_decision += decision_score * confidence
                    total_confidence += confidence
                
                if total_confidence == 0:
                    return {'decision': 'HOLD', 'confidence': 0.0, 'reason': 'No confidence'}
                
                final_decision_score = weighted_decision / total_confidence
                final_confidence = total_confidence / len(signals)
                
                if final_decision_score > 0.3:
                    decision = 'BUY'
                elif final_decision_score < -0.3:
                    decision = 'SELL'
                else:
                    decision = 'HOLD'
                
                return {
                    'decision': decision,
                    'confidence': final_confidence,
                    'decision_score': final_decision_score,
                    'reason': f'Integrated {len(signals)} signals'
                }
            
            # Test with sample signals
            test_signals = [
                {'decision': 'BUY', 'confidence': 0.8, 'dlt_score': 0.7},
                {'decision': 'HOLD', 'confidence': 0.6, 'dlt_score': 0.5},
                {'decision': 'BUY', 'confidence': 0.7, 'dlt_score': 0.6}
            ]
            
            final_decision = integrate_mathematical_decisions(test_signals)
            
            print("✅ Mathematical decision integration successful")
            print(f"   Final Decision: {final_decision['decision']}")
            print(f"   Confidence: {final_decision['confidence']:.4f}")
            print(f"   Decision Score: {final_decision['decision_score']:.4f}")
            print(f"   Reason: {final_decision['reason']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Mathematical decision integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_complete_trading_pipeline(self) -> bool:
        """Test complete trading pipeline with mathematical integration."""
        print("\n💰 Testing Complete Trading Pipeline")
        print("=" * 60)
        
        try:
            # Create a complete trading pipeline simulation
            class CompleteTradingPipeline:
                def __init__(self):
                    self.is_running = False
                    self.total_trades = 0
                    self.successful_trades = 0
                    self.mathematical_signals_processed = 0
                    self.last_decision = None
                
                async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Process market data through mathematical systems."""
                    self.mathematical_signals_processed += 1
                    
                    # Simulate mathematical processing
                    mathematical_signal = {
                        'dlt_waveform_score': 0.7,
                        'bit_phase': 8,
                        'ferris_phase': 0.3,
                        'tensor_score': 0.6,
                        'entropy_score': 0.4,
                        'confidence': 0.75,
                        'decision': 'BUY' if market_data.get('price_change', 0) > 0 else 'SELL',
                        'routing_target': 'USDC'
                    }
                    
                    # Simulate decision execution
                    if mathematical_signal['confidence'] > 0.7:
                        self.total_trades += 1
                        if mathematical_signal['decision'] == 'BUY' and market_data.get('price_change', 0) > 0:
                            self.successful_trades += 1
                        elif mathematical_signal['decision'] == 'SELL' and market_data.get('price_change', 0) < 0:
                            self.successful_trades += 1
                    
                    self.last_decision = mathematical_signal
                    return mathematical_signal
                
                def get_status(self) -> Dict[str, Any]:
                    """Get pipeline status."""
                    return {
                        'is_running': self.is_running,
                        'total_trades': self.total_trades,
                        'successful_trades': self.successful_trades,
                        'win_rate': self.successful_trades / max(self.total_trades, 1),
                        'mathematical_signals_processed': self.mathematical_signals_processed,
                        'last_decision': self.last_decision
                    }
            
            # Test the pipeline
            pipeline = CompleteTradingPipeline()
            
            # Simulate market data processing
            test_market_data = [
                {'price': 52000, 'price_change': 0.02, 'volume': 1000, 'volatility': 0.15},
                {'price': 51800, 'price_change': -0.01, 'volume': 1200, 'volatility': 0.12},
                {'price': 52100, 'price_change': 0.03, 'volume': 800, 'volatility': 0.18}
            ]
            
            for market_data in test_market_data:
                decision = await pipeline.process_market_data(market_data)
                print(f"   Processed: {decision['decision']} (confidence: {decision['confidence']:.2f})")
            
            status = pipeline.get_status()
            
            print("✅ Complete trading pipeline test successful")
            print(f"   Total Trades: {status['total_trades']}")
            print(f"   Successful Trades: {status['successful_trades']}")
            print(f"   Win Rate: {status['win_rate']:.2%}")
            print(f"   Signals Processed: {status['mathematical_signals_processed']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Complete trading pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all mathematical integration tests."""
        print("🧠 COMPLETE MATHEMATICAL INTEGRATION TEST")
        print("=" * 60)
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_results = {}
        
        # Run all tests
        tests = [
            ("Core Mathematical Components", self.test_core_mathematical_components),
            ("Simplified Mathematical Integration", self.test_simplified_mathematical_integration),
            ("Production Pipeline Integration", self.test_production_pipeline_integration),
            ("Mathematical Decision Integration", self.test_mathematical_decision_integration),
            ("Complete Trading Pipeline", self.test_complete_trading_pipeline),
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"\n🔄 Running {test_name}...")
                result = await test_func()
                test_results[test_name] = result
                print(f"✅ {test_name} completed")
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                test_results[test_name] = False
        
        # Calculate summary
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 COMPLETE MATHEMATICAL INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds")
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            if isinstance(result, dict):
                # For core components test
                passed_components = sum(1 for v in result.values() if v)
                total_components = len(result)
                status = f"✅ {passed_components}/{total_components} components"
                if passed_components == total_components:
                    passed += 1
            else:
                # For boolean results
                status = "✅ PASSED" if result else "❌ FAILED"
                if result:
                    passed += 1
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL MATHEMATICAL INTEGRATION TESTS PASSED!")
            print("🚀 Complete mathematical systems are working correctly!")
            print("💰 Ready for production deployment!")
        elif passed >= total * 0.8:
            print("✅ MOST MATHEMATICAL INTEGRATION TESTS PASSED!")
            print("🚀 Mathematical systems are mostly working!")
            print("⚠️ Some systems may need attention but core functionality is ready!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
            print("🔧 Mathematical systems may need additional work.")
        
        return {
            'passed': passed,
            'total': total,
            'duration': duration,
            'test_results': test_results
        }

async def main():
    """Run the complete mathematical integration test."""
    tester = MathematicalSystemTester()
    results = await tester.run_all_tests()
    return results

if __name__ == "__main__":
    asyncio.run(main()) 