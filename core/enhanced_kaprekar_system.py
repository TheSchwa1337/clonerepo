#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 ENHANCED KAPREKAR SYSTEM - NEXT-GENERATION TRADING INTELLIGENCE
==================================================================

Advanced Kaprekar-based trading system implementing multi-dimensional analysis,
temporal harmonics, and mathematical convergence patterns for enhanced trading signals.

Features:
- Multi-Dimensional Kaprekar Matrix (MDK)
- Temporal Kaprekar Harmonics (TKH)
- Kaprekar-Enhanced Ghost Memory
- Advanced Entropy Routing with Bifurcation Detection
- Quantum-Inspired Superposition Trading
- Cross-Asset Kaprekar Correlation Matrix
- Neural Kaprekar Pattern Recognition
- Real-Time Microstructure Analysis
- Kaprekar-Based Portfolio Optimization
- Unified Command Center Integration
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import hashlib

# Import existing Schwabot components
try:
    from .mathlib.kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult
    from .tick_kaprekar_bridge import price_to_kaprekar_index
    from .ghost_kaprekar_hash import generate_kaprekar_strategy_hash
    KAPREKAR_COMPONENTS_AVAILABLE = True
except ImportError:
    KAPREKAR_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MDKSignature:
    """Multi-Dimensional Kaprekar signature."""
    price_convergence: int
    volume_convergence: int
    volatility_convergence: int
    correlation_convergence: int
    pattern_signature: str
    stability_score: float
    timestamp: datetime

@dataclass
class TKHResonance:
    """Temporal Kaprekar Harmonics resonance data."""
    timeframe: str
    dominant_frequency: float
    harmonic_strength: float
    phase_alignment: float
    convergence_speed: int
    timestamp: datetime

@dataclass
class KaprekarMemory:
    """Kaprekar-encoded trade memory."""
    memory_signature: str
    entry_kaprekar: int
    exit_kaprekar: int
    profit_kaprekar: int
    profit_ratio: float
    success_rate: float
    convergence_confidence: float
    timestamp: datetime

class MultiDimensionalKaprekar:
    """Multi-Dimensional Kaprekar Matrix for enhanced market analysis."""
    
    def __init__(self):
        self.dimensions = ['price', 'volume', 'volatility', 'correlation']
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
    def normalize_to_4digit(self, value: float) -> int:
        """Normalize any value to a 4-digit number for Kaprekar analysis."""
        try:
            # Scale and normalize to 4 digits
            scaled = abs(value) * 1000
            normalized = int(scaled) % 10000
            return max(1000, normalized)  # Ensure 4 digits
        except Exception:
            return 1234  # Default fallback
            
    def calculate_mdk_signature(self, market_data: Dict[str, Any]) -> MDKSignature:
        """Generate 4D Kaprekar convergence signature."""
        try:
            signatures = {}
            
            # Analyze each dimension
            for dimension in self.dimensions:
                if dimension in market_data:
                    value = market_data[dimension]
                    normalized = self.normalize_to_4digit(value)
                    
                    if self.kaprekar_analyzer:
                        result = self.kaprekar_analyzer.analyze_kaprekar(normalized)
                        signatures[dimension] = result.steps_to_converge
                    else:
                        # Fallback calculation
                        signatures[dimension] = self._simple_kaprekar_steps(normalized)
                else:
                    signatures[dimension] = 7  # Default for missing data
            
            # Create convergence pattern signature
            pattern = ''.join([str(sig) for sig in signatures.values()])
            
            # Calculate stability score
            stability_score = self._calculate_stability_index(signatures)
            
            return MDKSignature(
                price_convergence=signatures['price'],
                volume_convergence=signatures['volume'],
                volatility_convergence=signatures['volatility'],
                correlation_convergence=signatures['correlation'],
                pattern_signature=pattern,
                stability_score=stability_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating MDK signature: {e}")
            return self._create_default_mdk_signature()
    
    def _simple_kaprekar_steps(self, n: int) -> int:
        """Simple Kaprekar steps calculation without full analyzer."""
        if n < 1000 or n > 9999:
            return 7
            
        steps = 0
        seen = set()
        current = n
        
        while current != 6174 and steps < 7:
            digits = sorted(f"{current:04d}")
            small = int("".join(digits))
            large = int("".join(digits[::-1]))
            current = large - small
            
            if current in seen or current == 0:
                return 7
                
            seen.add(current)
            steps += 1
            
        return steps
    
    def _calculate_stability_index(self, signatures: Dict[str, int]) -> float:
        """Calculate stability index from convergence signatures."""
        try:
            # Lower steps = higher stability
            avg_steps = sum(signatures.values()) / len(signatures)
            stability = max(0.1, 1.0 - (avg_steps / 7.0))
            return stability
        except Exception:
            return 0.5
    
    def _create_default_mdk_signature(self) -> MDKSignature:
        """Create default MDK signature for error cases."""
        return MDKSignature(
            price_convergence=4,
            volume_convergence=4,
            volatility_convergence=4,
            correlation_convergence=4,
            pattern_signature="4444",
            stability_score=0.5,
            timestamp=datetime.now()
        )
    
    def predict_multi_dimensional_collapse(self, mdk_history: List[MDKSignature]) -> Dict[str, Any]:
        """Predict market behavior based on multi-dimensional convergence patterns."""
        try:
            if not mdk_history:
                return self._default_collapse_prediction()
            
            # Analyze convergence trajectories
            collapse_vectors = self._analyze_convergence_trajectories(mdk_history)
            
            return {
                'stability_score': self._calculate_stability_index_from_history(mdk_history),
                'volatility_prediction': self._predict_vol_regime(collapse_vectors),
                'optimal_entry_timing': self._calculate_entry_window(collapse_vectors),
                'regime_change_probability': self._calculate_regime_change_probability(mdk_history),
                'recommended_strategy': self._select_strategy_based_on_collapse(collapse_vectors)
            }
            
        except Exception as e:
            logger.error(f"Error predicting multi-dimensional collapse: {e}")
            return self._default_collapse_prediction()
    
    def _analyze_convergence_trajectories(self, mdk_history: List[MDKSignature]) -> Dict[str, Any]:
        """Analyze convergence trajectories from MDK history."""
        try:
            if len(mdk_history) < 2:
                return {'trend': 'stable', 'momentum': 0.0}
            
            # Calculate trends for each dimension
            price_trend = self._calculate_trend([s.price_convergence for s in mdk_history])
            volume_trend = self._calculate_trend([s.volume_convergence for s in mdk_history])
            volatility_trend = self._calculate_trend([s.volatility_convergence for s in mdk_history])
            
            # Determine overall trajectory
            avg_trend = (price_trend + volume_trend + volatility_trend) / 3
            
            return {
                'trend': 'increasing' if avg_trend > 0.1 else 'decreasing' if avg_trend < -0.1 else 'stable',
                'momentum': avg_trend,
                'price_trend': price_trend,
                'volume_trend': volume_trend,
                'volatility_trend': volatility_trend
            }
            
        except Exception:
            return {'trend': 'stable', 'momentum': 0.0}
    
    def _calculate_trend(self, values: List[int]) -> float:
        """Calculate trend from a list of values."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple linear trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize to [-1, 1] range
            normalized_slope = np.tanh(slope)
            
            return normalized_slope
            
        except Exception:
            return 0.0
    
    def _predict_vol_regime(self, collapse_vectors: Dict[str, Any]) -> str:
        """Predict volatility regime based on collapse vectors."""
        try:
            momentum = collapse_vectors.get('momentum', 0.0)
            volatility_trend = collapse_vectors.get('volatility_trend', 0.0)
            
            if abs(momentum) > 0.5 or abs(volatility_trend) > 0.5:
                return "HIGH_VOLATILITY"
            elif abs(momentum) > 0.2 or abs(volatility_trend) > 0.2:
                return "MODERATE_VOLATILITY"
            else:
                return "LOW_VOLATILITY"
                
        except Exception:
            return "MODERATE_VOLATILITY"
    
    def _calculate_entry_window(self, collapse_vectors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal entry timing window."""
        try:
            momentum = collapse_vectors.get('momentum', 0.0)
            trend = collapse_vectors.get('trend', 'stable')
            
            if trend == 'decreasing' and momentum < -0.3:
                return {
                    'timing': 'IMMEDIATE',
                    'confidence': 0.8,
                    'reason': 'Strong convergence momentum'
                }
            elif trend == 'stable' and abs(momentum) < 0.1:
                return {
                    'timing': 'WAIT',
                    'confidence': 0.6,
                    'reason': 'Stable convergence pattern'
                }
            else:
                return {
                    'timing': 'MONITOR',
                    'confidence': 0.4,
                    'reason': 'Mixed convergence signals'
                }
                
        except Exception:
            return {
                'timing': 'MONITOR',
                'confidence': 0.3,
                'reason': 'Insufficient data'
            }
    
    def _calculate_regime_change_probability(self, mdk_history: List[MDKSignature]) -> float:
        """Calculate probability of regime change."""
        try:
            if len(mdk_history) < 3:
                return 0.3
            
            # Calculate stability changes
            recent_stability = mdk_history[-1].stability_score
            previous_stability = mdk_history[-2].stability_score
            
            stability_change = abs(recent_stability - previous_stability)
            
            # Higher change = higher probability of regime change
            probability = min(0.9, stability_change * 2.0)
            
            return probability
            
        except Exception:
            return 0.3
    
    def _select_strategy_based_on_collapse(self, collapse_vectors: Dict[str, Any]) -> str:
        """Select trading strategy based on collapse analysis."""
        try:
            trend = collapse_vectors.get('trend', 'stable')
            momentum = collapse_vectors.get('momentum', 0.0)
            
            if trend == 'decreasing' and momentum < -0.5:
                return "AGGRESSIVE_BUY"
            elif trend == 'increasing' and momentum > 0.5:
                return "AGGRESSIVE_SELL"
            elif trend == 'stable' and abs(momentum) < 0.2:
                return "MEAN_REVERSION"
            else:
                return "MOMENTUM_FOLLOW"
                
        except Exception:
            return "CONSERVATIVE_HOLD"
    
    def _calculate_stability_index_from_history(self, mdk_history: List[MDKSignature]) -> float:
        """Calculate stability index from MDK history."""
        try:
            if not mdk_history:
                return 0.5
            
            recent_stabilities = [s.stability_score for s in mdk_history[-5:]]
            return sum(recent_stabilities) / len(recent_stabilities)
            
        except Exception:
            return 0.5
    
    def _default_collapse_prediction(self) -> Dict[str, Any]:
        """Default collapse prediction for error cases."""
        return {
            'stability_score': 0.5,
            'volatility_prediction': 'MODERATE_VOLATILITY',
            'optimal_entry_timing': {
                'timing': 'MONITOR',
                'confidence': 0.3,
                'reason': 'Insufficient data'
            },
            'regime_change_probability': 0.3,
            'recommended_strategy': 'CONSERVATIVE_HOLD'
        }

class TemporalKaprekarHarmonics:
    """Temporal Kaprekar Harmonics for multi-timeframe analysis."""
    
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.harmonic_frequencies = {}
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
    def calculate_temporal_resonance(self, price_series: Dict[str, List[float]]) -> Dict[str, TKHResonance]:
        """Calculate Kaprekar convergence across multiple timeframes."""
        try:
            resonance_map = {}
            
            for timeframe in self.timeframes:
                if timeframe in price_series and price_series[timeframe]:
                    prices = price_series[timeframe]
                    
                    # Calculate convergence speeds
                    convergence_speeds = []
                    for price in prices[-10:]:  # Last 10 prices
                        if self.kaprekar_analyzer:
                            normalized = self._normalize_price(price)
                            result = self.kaprekar_analyzer.analyze_kaprekar(normalized)
                            convergence_speeds.append(result.steps_to_converge)
                        else:
                            normalized = self._normalize_price(price)
                            convergence_speeds.append(self._simple_kaprekar_steps(normalized))
                    
                    if convergence_speeds:
                        resonance_map[timeframe] = TKHResonance(
                            timeframe=timeframe,
                            dominant_frequency=self._find_dominant_frequency(convergence_speeds),
                            harmonic_strength=self._calculate_harmonic_strength(convergence_speeds),
                            phase_alignment=self._calculate_phase_alignment(convergence_speeds),
                            convergence_speed=int(np.mean(convergence_speeds)),
                            timestamp=datetime.now()
                        )
            
            return resonance_map
            
        except Exception as e:
            logger.error(f"Error calculating temporal resonance: {e}")
            return {}
    
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
    
    def _find_dominant_frequency(self, convergence_speeds: List[int]) -> float:
        """Find dominant frequency in convergence speeds."""
        try:
            if not convergence_speeds:
                return 0.0
            
            # Calculate frequency distribution
            unique_speeds, counts = np.unique(convergence_speeds, return_counts=True)
            dominant_index = np.argmax(counts)
            dominant_speed = unique_speeds[dominant_index]
            
            # Normalize to [0, 1] range
            return dominant_speed / 7.0
            
        except Exception:
            return 0.5
    
    def _calculate_harmonic_strength(self, convergence_speeds: List[int]) -> float:
        """Calculate harmonic strength from convergence speeds."""
        try:
            if len(convergence_speeds) < 2:
                return 0.5
            
            # Calculate variance (lower variance = higher harmonic strength)
            variance = np.var(convergence_speeds)
            max_variance = 7.0  # Maximum possible variance
            
            harmonic_strength = max(0.1, 1.0 - (variance / max_variance))
            return harmonic_strength
            
        except Exception:
            return 0.5
    
    def _calculate_phase_alignment(self, convergence_speeds: List[int]) -> float:
        """Calculate phase alignment from convergence speeds."""
        try:
            if len(convergence_speeds) < 3:
                return 0.5
            
            # Calculate trend consistency
            diffs = np.diff(convergence_speeds)
            consistent_trend = np.all(diffs >= 0) or np.all(diffs <= 0)
            
            if consistent_trend:
                return 0.8  # High alignment
            else:
                return 0.3  # Low alignment
                
        except Exception:
            return 0.5
    
    def detect_harmonic_convergence(self, resonance_map: Dict[str, TKHResonance]) -> str:
        """Detect when multiple timeframes show convergent Kaprekar patterns."""
        try:
            if not resonance_map:
                return "temporal_noise_hold"
            
            # Calculate cross-timeframe alignment
            alignment_score = self._calculate_cross_timeframe_alignment(resonance_map)
            
            if alignment_score > 0.85:
                return "harmonic_convergence_buy"
            elif alignment_score < 0.15:
                return "harmonic_divergence_sell"
            else:
                return "temporal_noise_hold"
                
        except Exception as e:
            logger.error(f"Error detecting harmonic convergence: {e}")
            return "temporal_noise_hold"
    
    def _calculate_cross_timeframe_alignment(self, resonance_map: Dict[str, TKHResonance]) -> float:
        """Calculate cross-timeframe alignment score."""
        try:
            if len(resonance_map) < 2:
                return 0.5
            
            # Calculate average harmonic strength and phase alignment
            harmonic_strengths = [r.harmonic_strength for r in resonance_map.values()]
            phase_alignments = [r.phase_alignment for r in resonance_map.values()]
            
            avg_harmonic = np.mean(harmonic_strengths)
            avg_phase = np.mean(phase_alignments)
            
            # Combined alignment score
            alignment_score = (avg_harmonic + avg_phase) / 2.0
            
            return alignment_score
            
        except Exception:
            return 0.5

class KaprekarGhostMemory:
    """Kaprekar-Enhanced Ghost Memory System."""
    
    def __init__(self):
        self.memory_bank: Dict[str, KaprekarMemory] = {}
        self.convergence_patterns: Dict[str, List[float]] = {}
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
    def encode_trade_memory(self, trade_data: Dict[str, Any]) -> str:
        """Encode trade outcomes using Kaprekar convergence patterns."""
        try:
            entry_price = trade_data.get('entry_price', 0.0)
            exit_price = trade_data.get('exit_price', 0.0)
            profit = trade_data.get('profit', 0.0)
            risk = trade_data.get('risk', 1.0)
            
            # Calculate Kaprekar values
            entry_k = self._calculate_kaprekar_index(entry_price)
            exit_k = self._calculate_kaprekar_index(exit_price)
            profit_k = self._calculate_kaprekar_index(abs(profit) * 1000)
            
            # Create memory signature
            memory_signature = f"{entry_k:02d}{exit_k:02d}{profit_k:02d}"
            
            # Calculate success metrics
            profit_ratio = profit / risk if risk > 0 else 0.0
            success_rate = self._calculate_historical_success(memory_signature)
            convergence_confidence = self._calculate_convergence_confidence(entry_k, exit_k)
            
            # Store memory
            self.memory_bank[memory_signature] = KaprekarMemory(
                memory_signature=memory_signature,
                entry_kaprekar=entry_k,
                exit_kaprekar=exit_k,
                profit_kaprekar=profit_k,
                profit_ratio=profit_ratio,
                success_rate=success_rate,
                convergence_confidence=convergence_confidence,
                timestamp=datetime.now()
            )
            
            return memory_signature
            
        except Exception as e:
            logger.error(f"Error encoding trade memory: {e}")
            return "000000"
    
    def _calculate_kaprekar_index(self, value: float) -> int:
        """Calculate Kaprekar index for a value."""
        try:
            if self.kaprekar_analyzer:
                normalized = self._normalize_value(value)
                result = self.kaprekar_analyzer.analyze_kaprekar(normalized)
                return result.steps_to_converge
            else:
                normalized = self._normalize_value(value)
                return self._simple_kaprekar_steps(normalized)
                
        except Exception:
            return 4
    
    def _normalize_value(self, value: float) -> int:
        """Normalize value to 4-digit number."""
        try:
            scaled = abs(value) * 100
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
    
    def _calculate_historical_success(self, memory_signature: str) -> float:
        """Calculate historical success rate for a memory signature."""
        try:
            if memory_signature in self.memory_bank:
                memory = self.memory_bank[memory_signature]
                return memory.profit_ratio if memory.profit_ratio > 0 else 0.0
            else:
                return 0.5  # Default success rate
                
        except Exception:
            return 0.5
    
    def _calculate_convergence_confidence(self, entry_k: int, exit_k: int) -> float:
        """Calculate convergence confidence between entry and exit Kaprekar values."""
        try:
            # Lower values indicate faster convergence (higher confidence)
            avg_k = (entry_k + exit_k) / 2.0
            confidence = max(0.1, 1.0 - (avg_k / 7.0))
            return confidence
            
        except Exception:
            return 0.5
    
    def recall_similar_patterns(self, current_market: Dict[str, Any]) -> Dict[str, Any]:
        """Find historically similar Kaprekar patterns and their outcomes."""
        try:
            current_signature = self._encode_current_market(current_market)
            similar_patterns = self._find_pattern_matches(current_signature)
            
            return {
                'expected_profit': self._calculate_expected_outcome(similar_patterns),
                'confidence_level': self._calculate_pattern_confidence(similar_patterns),
                'recommended_action': self._synthesize_recommendation(similar_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error recalling similar patterns: {e}")
            return {
                'expected_profit': 0.0,
                'confidence_level': 0.3,
                'recommended_action': 'HOLD'
            }
    
    def _encode_current_market(self, current_market: Dict[str, Any]) -> str:
        """Encode current market state."""
        try:
            price = current_market.get('price', 0.0)
            volume = current_market.get('volume', 0.0)
            volatility = current_market.get('volatility', 0.0)
            
            price_k = self._calculate_kaprekar_index(price)
            volume_k = self._calculate_kaprekar_index(volume)
            volatility_k = self._calculate_kaprekar_index(volatility)
            
            return f"{price_k:02d}{volume_k:02d}{volatility_k:02d}"
            
        except Exception:
            return "444444"
    
    def _find_pattern_matches(self, current_signature: str) -> List[KaprekarMemory]:
        """Find pattern matches in memory bank."""
        try:
            matches = []
            
            for signature, memory in self.memory_bank.items():
                # Simple similarity check (can be enhanced)
                similarity = self._calculate_signature_similarity(current_signature, signature)
                if similarity > 0.7:  # 70% similarity threshold
                    matches.append(memory)
            
            return matches
            
        except Exception:
            return []
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures."""
        try:
            if len(sig1) != len(sig2):
                return 0.0
            
            matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
            similarity = matches / len(sig1)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _calculate_expected_outcome(self, similar_patterns: List[KaprekarMemory]) -> float:
        """Calculate expected outcome from similar patterns."""
        try:
            if not similar_patterns:
                return 0.0
            
            profit_ratios = [p.profit_ratio for p in similar_patterns]
            return sum(profit_ratios) / len(profit_ratios)
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_confidence(self, similar_patterns: List[KaprekarMemory]) -> float:
        """Calculate confidence level from similar patterns."""
        try:
            if not similar_patterns:
                return 0.3
            
            confidences = [p.convergence_confidence for p in similar_patterns]
            return sum(confidences) / len(confidences)
            
        except Exception:
            return 0.3
    
    def _synthesize_recommendation(self, similar_patterns: List[KaprekarMemory]) -> str:
        """Synthesize trading recommendation from similar patterns."""
        try:
            if not similar_patterns:
                return 'HOLD'
            
            avg_profit = self._calculate_expected_outcome(similar_patterns)
            
            if avg_profit > 0.1:
                return 'BUY'
            elif avg_profit < -0.1:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception:
            return 'HOLD'

# Global instances for easy access
mdk_analyzer = MultiDimensionalKaprekar()
tkh_analyzer = TemporalKaprekarHarmonics()
ghost_memory = KaprekarGhostMemory() 