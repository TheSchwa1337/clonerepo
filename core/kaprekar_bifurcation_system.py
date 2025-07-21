#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 KAPREKAR BIFURCATION SYSTEM - CHAOS THEORY INTEGRATION
=========================================================

Advanced entropy routing with Kaprekar bifurcation detection, quantum-inspired
superposition trading, and chaos theory integration for market regime analysis.

Features:
- Advanced Entropy Routing with Kaprekar Bifurcation
- Quantum-Inspired Superposition Trading
- Chaos Theory Integration
- Market Regime Change Detection
- Adaptive Strategy Selection
- Emergency Halt Protocols
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import random

# Import existing components
try:
    from .mathlib.kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult
    from .tick_kaprekar_bridge import price_to_kaprekar_index
    KAPREKAR_COMPONENTS_AVAILABLE = True
except ImportError:
    KAPREKAR_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BifurcationPoint:
    """Bifurcation point detection result."""
    detected: bool
    chaos_level: float
    regime_change_probability: float
    recommended_strategy: str
    gradient_change: float
    timestamp: datetime

@dataclass
class QuantumState:
    """Quantum-inspired trading state."""
    buy_probability: float
    sell_probability: float
    hold_probability: float
    hedge_probability: float
    superposition_entropy: float
    collapse_trigger: str
    timestamp: datetime

class KaprekarBifurcationDetector:
    """Detect bifurcation points in Kaprekar convergence as market regime changes."""
    
    def __init__(self):
        self.bifurcation_threshold = 0.618  # Golden ratio threshold
        self.chaos_indicators: Dict[str, float] = {}
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        self.convergence_history: deque = deque(maxlen=100)
        
    def detect_convergence_bifurcation(self, price_sequence: List[float]) -> BifurcationPoint:
        """Detect when Kaprekar convergence patterns suddenly change (bifurcation)."""
        try:
            if len(price_sequence) < 10:
                return self._create_default_bifurcation_point()
            
            # Calculate convergence series
            convergence_series = []
            for price in price_sequence[-20:]:  # Last 20 prices
                if self.kaprekar_analyzer:
                    normalized = self._normalize_price(price)
                    result = self.kaprekar_analyzer.analyze_kaprekar(normalized)
                    convergence_series.append(result.steps_to_converge)
                else:
                    normalized = self._normalize_price(price)
                    convergence_series.append(self._simple_kaprekar_steps(normalized))
            
            # Store in history
            self.convergence_history.extend(convergence_series)
            
            # Calculate convergence gradient
            gradient = self._calculate_convergence_gradient(convergence_series)
            
            # Detect bifurcation points
            bifurcation_points = self._detect_bifurcation_points(gradient)
            
            # Calculate chaos metrics
            chaos_level = self._calculate_chaos_intensity(gradient)
            regime_change_probability = self._predict_regime_change(bifurcation_points)
            recommended_strategy = self._select_chaos_strategy(gradient)
            
            return BifurcationPoint(
                detected=len(bifurcation_points) > 0,
                chaos_level=chaos_level,
                regime_change_probability=regime_change_probability,
                recommended_strategy=recommended_strategy,
                gradient_change=np.mean(gradient) if gradient else 0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting convergence bifurcation: {e}")
            return self._create_default_bifurcation_point()
    
    def _normalize_price(self, price: float) -> int:
        """Normalize price to 4-digit number."""
        try:
            scaled = abs(price) * 100
            normalized = int(scaled) % 10000
            return max(1000, normalized)
        except Exception:
            return 1234
    
    def _simple_kaprekar_steps(self, n: int) -> int:
        """Simple Kaprekar steps calculation."""
        if n < 1000 or n > 9999:
            return 4
            
        steps = 0
        seen = set()
        current = n
        
        while current != 6174 and steps < 7:
            digits = sorted(f"{current:04d}")
            small = int("".join(digits))
            large = int("".join(digits[::-1]))
            current = large - small
            
            if current in seen or current == 0:
                return 4
                
            seen.add(current)
            steps += 1
            
        return steps
    
    def _calculate_convergence_gradient(self, convergence_series: List[int]) -> List[float]:
        """Calculate convergence gradient."""
        try:
            if len(convergence_series) < 2:
                return [0.0]
            
            gradient = []
            for i in range(1, len(convergence_series)):
                change = convergence_series[i] - convergence_series[i-1]
                gradient.append(change)
            
            return gradient
            
        except Exception:
            return [0.0]
    
    def _detect_bifurcation_points(self, gradient: List[float]) -> List[int]:
        """Detect bifurcation points in gradient."""
        try:
            bifurcation_points = []
            
            for i in range(1, len(gradient)):
                gradient_change = abs(gradient[i] - gradient[i-1])
                if gradient_change > self.bifurcation_threshold:
                    bifurcation_points.append(i)
            
            return bifurcation_points
            
        except Exception:
            return []
    
    def _calculate_chaos_intensity(self, gradient: List[float]) -> float:
        """Calculate chaos intensity from gradient."""
        try:
            if not gradient:
                return 0.0
            
            # Calculate multiple chaos indicators
            variance = np.var(gradient)
            max_change = max(abs(g) for g in gradient) if gradient else 0.0
            zero_crossings = sum(1 for i in range(1, len(gradient)) 
                               if gradient[i] * gradient[i-1] < 0)
            
            # Normalize indicators
            normalized_variance = min(1.0, variance / 10.0)
            normalized_max_change = min(1.0, max_change / 7.0)
            normalized_crossings = min(1.0, zero_crossings / len(gradient)) if gradient else 0.0
            
            # Combined chaos intensity
            chaos_intensity = (normalized_variance + normalized_max_change + normalized_crossings) / 3.0
            
            return chaos_intensity
            
        except Exception:
            return 0.0
    
    def _predict_regime_change(self, bifurcation_points: List[int]) -> float:
        """Predict probability of regime change."""
        try:
            if not bifurcation_points:
                return 0.1  # Low probability if no bifurcations
            
            # More bifurcation points = higher probability
            num_bifurcations = len(bifurcation_points)
            probability = min(0.9, num_bifurcations * 0.2)
            
            return probability
            
        except Exception:
            return 0.1
    
    def _select_chaos_strategy(self, gradient: List[float]) -> str:
        """Select trading strategy based on chaos analysis."""
        try:
            if not gradient:
                return "standard_kaprekar_routing"
            
            chaos_level = self._calculate_chaos_intensity(gradient)
            
            if chaos_level > 0.8:
                return "emergency_halt_protocol"
            elif chaos_level > 0.6:
                return "regime_transition_strategy"
            elif chaos_level > 0.4:
                return "volatility_breakout_strategy"
            else:
                return "standard_kaprekar_routing"
                
        except Exception:
            return "standard_kaprekar_routing"
    
    def _create_default_bifurcation_point(self) -> BifurcationPoint:
        """Create default bifurcation point for error cases."""
        return BifurcationPoint(
            detected=False,
            chaos_level=0.0,
            regime_change_probability=0.1,
            recommended_strategy="standard_kaprekar_routing",
            gradient_change=0.0,
            timestamp=datetime.now()
        )
    
    def adaptive_entropy_routing(self, bifurcation_data: BifurcationPoint) -> str:
        """Dynamically adjust entropy routing based on bifurcation detection."""
        try:
            if bifurcation_data.chaos_level > 0.8:
                return "emergency_halt_protocol"
            elif bifurcation_data.regime_change_probability > 0.7:
                return "regime_transition_strategy"
            elif bifurcation_data.chaos_level > 0.5:
                return "volatility_breakout_strategy"
            else:
                return "standard_kaprekar_routing"
                
        except Exception as e:
            logger.error(f"Error in adaptive entropy routing: {e}")
            return "standard_kaprekar_routing"

class QuantumTradingStates:
    """Quantum-inspired superposition of multiple trading states."""
    
    def __init__(self):
        self.state_probabilities: Dict[str, float] = {}
        self.entanglement_matrix: Dict[str, Dict[str, float]] = {}
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
    def create_superposition_state(self, market_signals: Dict[str, Any]) -> QuantumState:
        """Create quantum-inspired superposition of possible trading states."""
        try:
            states = ['buy', 'sell', 'hold', 'hedge']
            probabilities = {}
            
            for state in states:
                kaprekar_confidence = self._calculate_kaprekar_confidence(market_signals, state)
                ai_consensus = self._get_ai_state_probability(market_signals, state)
                ghost_memory = self._get_memory_state_probability(market_signals, state)
                
                # Quantum-inspired probability calculation
                # Using geometric mean for quantum-like behavior
                probabilities[state] = (kaprekar_confidence * ai_consensus * ghost_memory) ** 0.5
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                normalized_probabilities = {state: prob/total for state, prob in probabilities.items()}
            else:
                normalized_probabilities = {state: 0.25 for state in states}
            
            # Calculate superposition entropy
            entropy = self._calculate_superposition_entropy(normalized_probabilities)
            
            return QuantumState(
                buy_probability=normalized_probabilities.get('buy', 0.25),
                sell_probability=normalized_probabilities.get('sell', 0.25),
                hold_probability=normalized_probabilities.get('hold', 0.25),
                hedge_probability=normalized_probabilities.get('hedge', 0.25),
                superposition_entropy=entropy,
                collapse_trigger="pending",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating superposition state: {e}")
            return self._create_default_quantum_state()
    
    def _calculate_kaprekar_confidence(self, market_signals: Dict[str, Any], state: str) -> float:
        """Calculate Kaprekar confidence for a trading state."""
        try:
            if not self.kaprekar_analyzer:
                return 0.5
            
            # Extract relevant market data
            price = market_signals.get('price', 0.0)
            volume = market_signals.get('volume', 0.0)
            volatility = market_signals.get('volatility', 0.0)
            
            # Calculate Kaprekar analysis
            normalized_price = self._normalize_value(price)
            result = self.kaprekar_analyzer.analyze_kaprekar(normalized_price)
            
            # Map state to confidence based on convergence
            if state == 'buy':
                # Fast convergence (1-3 steps) favors buy
                if result.steps_to_converge <= 3:
                    return 0.8
                else:
                    return 0.3
            elif state == 'sell':
                # Slow convergence (5-7 steps) favors sell
                if result.steps_to_converge >= 5:
                    return 0.8
                else:
                    return 0.3
            elif state == 'hold':
                # Moderate convergence (3-5 steps) favors hold
                if 3 <= result.steps_to_converge <= 5:
                    return 0.8
                else:
                    return 0.3
            else:  # hedge
                # Chaotic convergence favors hedge
                if not result.is_convergent:
                    return 0.8
                else:
                    return 0.3
                    
        except Exception:
            return 0.5
    
    def _normalize_value(self, value: float) -> int:
        """Normalize value to 4-digit number."""
        try:
            scaled = abs(value) * 100
            normalized = int(scaled) % 10000
            return max(1000, normalized)
        except Exception:
            return 1234
    
    def _get_ai_state_probability(self, market_signals: Dict[str, Any], state: str) -> float:
        """Get AI consensus probability for a state."""
        try:
            # Simulate AI consensus based on market signals
            momentum = market_signals.get('momentum', 0.0)
            rsi = market_signals.get('rsi', 50.0)
            volatility = market_signals.get('volatility', 0.0)
            
            if state == 'buy':
                if momentum > 0.1 and rsi < 70:
                    return 0.7
                else:
                    return 0.3
            elif state == 'sell':
                if momentum < -0.1 and rsi > 30:
                    return 0.7
                else:
                    return 0.3
            elif state == 'hold':
                if abs(momentum) < 0.05 and 30 <= rsi <= 70:
                    return 0.7
                else:
                    return 0.3
            else:  # hedge
                if volatility > 0.5:
                    return 0.7
                else:
                    return 0.3
                    
        except Exception:
            return 0.5
    
    def _get_memory_state_probability(self, market_signals: Dict[str, Any], state: str) -> float:
        """Get memory-based probability for a state."""
        try:
            # Simulate memory-based probability
            # In a real implementation, this would query the ghost memory system
            
            # Random but consistent probability based on state
            random.seed(hash(state) % 1000)
            base_prob = random.uniform(0.3, 0.7)
            
            # Adjust based on market conditions
            volatility = market_signals.get('volatility', 0.0)
            if volatility > 0.5:
                base_prob *= 0.8  # Reduce confidence in high volatility
            
            return base_prob
            
        except Exception:
            return 0.5
    
    def _calculate_superposition_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate entropy of superposition state."""
        try:
            entropy = 0.0
            for prob in probabilities.values():
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            return entropy
            
        except Exception:
            return 1.0
    
    def _create_default_quantum_state(self) -> QuantumState:
        """Create default quantum state for error cases."""
        return QuantumState(
            buy_probability=0.25,
            sell_probability=0.25,
            hold_probability=0.25,
            hedge_probability=0.25,
            superposition_entropy=2.0,
            collapse_trigger="error",
            timestamp=datetime.now()
        )
    
    def collapse_to_action(self, superposition_state: QuantumState, market_trigger: str) -> str:
        """Collapse superposition to concrete action based on market trigger."""
        try:
            # Get trigger weights
            trigger_weights = self._get_trigger_weights(market_trigger)
            
            # Calculate weighted probabilities
            weighted_probabilities = {
                'buy': superposition_state.buy_probability * trigger_weights.get('buy', 1.0),
                'sell': superposition_state.sell_probability * trigger_weights.get('sell', 1.0),
                'hold': superposition_state.hold_probability * trigger_weights.get('hold', 1.0),
                'hedge': superposition_state.hedge_probability * trigger_weights.get('hedge', 1.0)
            }
            
            # Select highest probability state
            best_action = max(weighted_probabilities, key=weighted_probabilities.get)
            
            return best_action
            
        except Exception as e:
            logger.error(f"Error collapsing superposition: {e}")
            return "hold"
    
    def _get_trigger_weights(self, market_trigger: str) -> Dict[str, float]:
        """Get trigger weights for different states."""
        try:
            # Define trigger weights based on market conditions
            if market_trigger == "high_volatility":
                return {'buy': 0.8, 'sell': 0.8, 'hold': 0.5, 'hedge': 1.2}
            elif market_trigger == "trend_breakout":
                return {'buy': 1.2, 'sell': 1.2, 'hold': 0.3, 'hedge': 0.8}
            elif market_trigger == "mean_reversion":
                return {'buy': 0.8, 'sell': 0.8, 'hold': 1.2, 'hedge': 0.5}
            elif market_trigger == "chaos_detected":
                return {'buy': 0.3, 'sell': 0.3, 'hold': 0.5, 'hedge': 1.5}
            else:
                return {'buy': 1.0, 'sell': 1.0, 'hold': 1.0, 'hedge': 1.0}
                
        except Exception:
            return {'buy': 1.0, 'sell': 1.0, 'hold': 1.0, 'hedge': 1.0}

class CrossAssetKaprekarMatrix:
    """Cross-asset Kaprekar correlation matrix for multi-asset analysis."""
    
    def __init__(self):
        self.asset_pairs = ['BTC/USDC', 'ETH/USDC', 'XRP/USDC', 'ADA/USDC']
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
    def calculate_cross_asset_convergence(self, asset_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze Kaprekar convergence patterns across multiple assets."""
        try:
            convergence_matrix = {}
            
            for asset, price_data in asset_data.items():
                if price_data:
                    convergence_speeds = []
                    for price in price_data[-10:]:  # Last 10 prices
                        if self.kaprekar_analyzer:
                            normalized = self._normalize_price(price)
                            result = self.kaprekar_analyzer.analyze_kaprekar(normalized)
                            convergence_speeds.append(result.steps_to_converge)
                        else:
                            normalized = self._normalize_price(price)
                            convergence_speeds.append(self._simple_kaprekar_steps(normalized))
                    
                    if convergence_speeds:
                        convergence_matrix[asset] = {
                            'convergence_speed': convergence_speeds,
                            'convergence_stability': self._calculate_stability(convergence_speeds),
                            'convergence_trend': self._calculate_trend(convergence_speeds)
                        }
            
            # Calculate cross-correlations
            correlation_map = self._calculate_convergence_correlations(convergence_matrix)
            
            return {
                'dominant_asset': self._find_dominant_convergence_asset(convergence_matrix),
                'correlation_strength': self._calculate_overall_correlation(correlation_map),
                'arbitrage_opportunities': self._detect_arbitrage_patterns(correlation_map),
                'portfolio_rebalance_signal': self._generate_rebalance_signal(correlation_map)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cross-asset convergence: {e}")
            return self._create_default_cross_asset_result()
    
    def _normalize_price(self, price: float) -> int:
        """Normalize price to 4-digit number."""
        try:
            scaled = abs(price) * 100
            normalized = int(scaled) % 10000
            return max(1000, normalized)
        except Exception:
            return 1234
    
    def _simple_kaprekar_steps(self, n: int) -> int:
        """Simple Kaprekar steps calculation."""
        if n < 1000 or n > 9999:
            return 4
            
        steps = 0
        seen = set()
        current = n
        
        while current != 6174 and steps < 7:
            digits = sorted(f"{current:04d}")
            small = int("".join(digits))
            large = int("".join(digits[::-1]))
            current = large - small
            
            if current in seen or current == 0:
                return 4
                
            seen.add(current)
            steps += 1
            
        return steps
    
    def _calculate_stability(self, convergence_speeds: List[int]) -> float:
        """Calculate convergence stability."""
        try:
            if len(convergence_speeds) < 2:
                return 0.5
            
            variance = np.var(convergence_speeds)
            stability = max(0.1, 1.0 - (variance / 10.0))
            return stability
            
        except Exception:
            return 0.5
    
    def _calculate_trend(self, convergence_speeds: List[int]) -> float:
        """Calculate convergence trend."""
        try:
            if len(convergence_speeds) < 2:
                return 0.0
            
            x = np.arange(len(convergence_speeds))
            y = np.array(convergence_speeds)
            
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = np.tanh(slope)
            
            return normalized_slope
            
        except Exception:
            return 0.0
    
    def _calculate_convergence_correlations(self, convergence_matrix: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlations between asset convergence patterns."""
        try:
            correlation_map = {}
            assets = list(convergence_matrix.keys())
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    speeds1 = convergence_matrix[asset1]['convergence_speed']
                    speeds2 = convergence_matrix[asset2]['convergence_speed']
                    
                    if len(speeds1) == len(speeds2) and len(speeds1) > 1:
                        correlation = np.corrcoef(speeds1, speeds2)[0, 1]
                        if not np.isnan(correlation):
                            pair_name = f"{asset1}-{asset2}"
                            correlation_map[pair_name] = correlation
            
            return correlation_map
            
        except Exception:
            return {}
    
    def _find_dominant_convergence_asset(self, convergence_matrix: Dict[str, Dict[str, Any]]) -> str:
        """Find the asset with the most stable convergence."""
        try:
            if not convergence_matrix:
                return "BTC/USDC"
            
            best_asset = None
            best_stability = -1.0
            
            for asset, data in convergence_matrix.items():
                stability = data.get('convergence_stability', 0.0)
                if stability > best_stability:
                    best_stability = stability
                    best_asset = asset
            
            return best_asset or "BTC/USDC"
            
        except Exception:
            return "BTC/USDC"
    
    def _calculate_overall_correlation(self, correlation_map: Dict[str, float]) -> float:
        """Calculate overall correlation strength."""
        try:
            if not correlation_map:
                return 0.0
            
            correlations = list(correlation_map.values())
            return sum(correlations) / len(correlations)
            
        except Exception:
            return 0.0
    
    def _detect_arbitrage_patterns(self, correlation_map: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities based on convergence pattern divergence."""
        try:
            arbitrage_signals = []
            
            for asset_pair, correlation in correlation_map.items():
                if correlation < -0.5:  # Strong negative correlation
                    arbitrage_signals.append({
                        'pair': asset_pair,
                        'type': 'convergence_divergence_arbitrage',
                        'expected_profit': self._calculate_arbitrage_profit(asset_pair, correlation),
                        'risk_level': self._assess_arbitrage_risk(asset_pair, correlation)
                    })
            
            return arbitrage_signals
            
        except Exception:
            return []
    
    def _calculate_arbitrage_profit(self, asset_pair: str, correlation: float) -> float:
        """Calculate expected arbitrage profit."""
        try:
            # Simple profit calculation based on correlation strength
            profit = abs(correlation) * 0.1  # 10% base profit for perfect correlation
            return min(0.05, profit)  # Cap at 5%
            
        except Exception:
            return 0.01
    
    def _assess_arbitrage_risk(self, asset_pair: str, correlation: float) -> str:
        """Assess risk level for arbitrage opportunity."""
        try:
            if abs(correlation) > 0.8:
                return "LOW"
            elif abs(correlation) > 0.6:
                return "MEDIUM"
            else:
                return "HIGH"
                
        except Exception:
            return "HIGH"
    
    def _generate_rebalance_signal(self, correlation_map: Dict[str, float]) -> Dict[str, Any]:
        """Generate portfolio rebalancing signal."""
        try:
            overall_correlation = self._calculate_overall_correlation(correlation_map)
            
            if overall_correlation > 0.7:
                return {
                    'rebalance_needed': True,
                    'action': 'reduce_correlation',
                    'confidence': 0.8
                }
            elif overall_correlation < 0.2:
                return {
                    'rebalance_needed': True,
                    'action': 'increase_correlation',
                    'confidence': 0.6
                }
            else:
                return {
                    'rebalance_needed': False,
                    'action': 'maintain',
                    'confidence': 0.5
                }
                
        except Exception:
            return {
                'rebalance_needed': False,
                'action': 'maintain',
                'confidence': 0.3
            }
    
    def _create_default_cross_asset_result(self) -> Dict[str, Any]:
        """Create default cross-asset result for error cases."""
        return {
            'dominant_asset': 'BTC/USDC',
            'correlation_strength': 0.0,
            'arbitrage_opportunities': [],
            'portfolio_rebalance_signal': {
                'rebalance_needed': False,
                'action': 'maintain',
                'confidence': 0.3
            }
        }

# Global instances for easy access
bifurcation_detector = KaprekarBifurcationDetector()
quantum_states = QuantumTradingStates()
cross_asset_matrix = CrossAssetKaprekarMatrix() 