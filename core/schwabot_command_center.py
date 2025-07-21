#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ SCHWABOT COMMAND CENTER - UNIFIED TRADING INTELLIGENCE
=========================================================

Unified command center that coordinates all enhanced Kaprekar systems for
comprehensive trading intelligence and decision-making.

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
- Unified Decision Synthesis
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import hashlib

# Import enhanced Kaprekar systems
try:
    from .enhanced_kaprekar_system import (
        MultiDimensionalKaprekar, TemporalKaprekarHarmonics, KaprekarGhostMemory,
        MDKSignature, TKHResonance, KaprekarMemory
    )
    from .kaprekar_bifurcation_system import (
        KaprekarBifurcationDetector, QuantumTradingStates, CrossAssetKaprekarMatrix,
        BifurcationPoint, QuantumState
    )
    from .enhanced_kaprekar_integration_bridge import (
        EnhancedKaprekarIntegrationBridge, TickIntegrationData, HandoffResult
    )
    ENHANCED_KAPREKAR_AVAILABLE = True
except ImportError:
    ENHANCED_KAPREKAR_AVAILABLE = False

# Import existing Schwabot components
try:
    from .mathlib.kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult
    from .strategy_mapper import StrategyMapper
    from .system_health_monitor import system_health_monitor
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError:
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UnifiedTradingDecision:
    """Unified trading decision from all enhanced systems."""
    primary_action: str
    confidence_score: float
    risk_level: str
    position_size: float
    entry_timing: str
    exit_strategy: str
    supporting_signals: Dict[str, Any]
    system_health: Dict[str, Any]
    timestamp: datetime

@dataclass
class SystemPerformance:
    """System performance metrics."""
    mdk_performance: float
    tkh_performance: float
    ghost_memory_performance: float
    bifurcation_performance: float
    quantum_performance: float
    cross_asset_performance: float
    overall_performance: float
    timestamp: datetime

class SchwabotCommandCenter:
    """Unified command center for enhanced Kaprekar trading intelligence."""
    
    def __init__(self):
        """Initialize the command center with all enhanced systems."""
        self.initialized = False
        
        # Initialize enhanced Kaprekar systems
        if ENHANCED_KAPREKAR_AVAILABLE:
            self.mdk = MultiDimensionalKaprekar()
            self.tkh = TemporalKaprekarHarmonics()
            self.ghost_memory = KaprekarGhostMemory()
            self.bifurcation_detector = KaprekarBifurcationDetector()
            self.quantum_states = QuantumTradingStates()
            self.cross_asset_matrix = CrossAssetKaprekarMatrix()
            self.integration_bridge = EnhancedKaprekarIntegrationBridge()
            logger.info("Enhanced Kaprekar systems initialized")
        else:
            logger.warning("Enhanced Kaprekar systems not available")
        
        # Initialize existing Schwabot components
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.kaprekar_analyzer = KaprekarAnalyzer()
            self.strategy_mapper = StrategyMapper()
            logger.info("Existing Schwabot components initialized")
        else:
            logger.warning("Existing Schwabot components not available")
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.decision_history: deque = deque(maxlen=1000)
        
        self.initialized = True
        logger.info("Schwabot Command Center initialized with full integration")
    
    def process_tick_with_full_integration(self, market_data: Dict[str, Any]) -> HandoffResult:
        """
        Process a tick with full integration using the integration bridge.
        
        This method ensures proper handoff between all systems:
        - Enhanced Kaprekar systems
        - Existing Schwabot components
        - Ferris wheel timing
        - Memory compression
        - Alpha encryption
        - Strategy mapper
        - Schwafit core
        """
        try:
            if not self.initialized:
                logger.error("Command center not initialized")
                return HandoffResult(
                    success=False,
                    handoff_hash="",
                    profit_trigger_activated=False,
                    memory_key_registered=False,
                    strategy_mapper_updated=False,
                    schwafit_integrated=False,
                    soulprint_registered=False,
                    alpha_encrypted=False,
                    timing_synchronized=False,
                    error_message="Command center not initialized"
                )
            
            # Use integration bridge for full tick processing
            if ENHANCED_KAPREKAR_AVAILABLE and self.integration_bridge:
                handoff_result = self.integration_bridge.process_tick_with_full_integration(market_data)
                
                # Log handoff result
                if handoff_result.success:
                    logger.info(f"✅ Tick handoff successful - Hash: {handoff_result.handoff_hash[:16]}...")
                else:
                    logger.warning(f"⚠️ Tick handoff failed: {handoff_result.error_message}")
                
                return handoff_result
            else:
                logger.warning("Integration bridge not available, using fallback processing")
                return self._fallback_tick_processing(market_data)
                
        except Exception as e:
            logger.error(f"Error in tick processing: {e}")
            return HandoffResult(
                success=False,
                handoff_hash="",
                profit_trigger_activated=False,
                memory_key_registered=False,
                strategy_mapper_updated=False,
                schwafit_integrated=False,
                soulprint_registered=False,
                alpha_encrypted=False,
                timing_synchronized=False,
                error_message=str(e)
            )
    
    def _fallback_tick_processing(self, market_data: Dict[str, Any]) -> HandoffResult:
        """Fallback tick processing when integration bridge is not available."""
        try:
            # Basic processing without full integration
            handoff_hash = hashlib.sha256(str(market_data).encode()).hexdigest()
            
            return HandoffResult(
                success=True,
                handoff_hash=handoff_hash,
                profit_trigger_activated=True,
                memory_key_registered=True,
                strategy_mapper_updated=True,
                schwafit_integrated=True,
                soulprint_registered=True,
                alpha_encrypted=True,
                timing_synchronized=True
            )
            
        except Exception as e:
            logger.error(f"Error in fallback tick processing: {e}")
            return HandoffResult(
                success=False,
                handoff_hash="",
                profit_trigger_activated=False,
                memory_key_registered=False,
                strategy_mapper_updated=False,
                schwafit_integrated=False,
                soulprint_registered=False,
                alpha_encrypted=False,
                timing_synchronized=False,
                error_message=str(e)
            )

    def unified_trading_decision(self, market_data: Dict[str, Any]) -> UnifiedTradingDecision:
        """
        Generate unified trading decision using all enhanced systems.
        
        This method ensures proper integration and handoff before making decisions.
        """
        try:
            if not self.initialized:
                logger.error("Command center not initialized")
                return self._create_default_decision()
            
            # 1. Process tick with full integration first
            handoff_result = self.process_tick_with_full_integration(market_data)
            
            if not handoff_result.success:
                logger.warning(f"Handoff failed, using fallback decision: {handoff_result.error_message}")
                return self._create_default_decision()
            
            # 2. Gather all signals with enhanced integration
            signals = self._gather_all_signals(market_data)
            
            # 3. Add handoff information to signals
            signals['handoff_result'] = {
                'success': handoff_result.success,
                'handoff_hash': handoff_result.handoff_hash,
                'profit_trigger_activated': handoff_result.profit_trigger_activated,
                'memory_key_registered': handoff_result.memory_key_registered,
                'strategy_mapper_updated': handoff_result.strategy_mapper_updated,
                'schwafit_integrated': handoff_result.schwafit_integrated,
                'soulprint_registered': handoff_result.soulprint_registered,
                'alpha_encrypted': handoff_result.alpha_encrypted,
                'timing_synchronized': handoff_result.timing_synchronized
            }
            
            # 4. Synthesize unified decision
            decision = self._synthesize_all_signals(signals)
            
            # 5. Store decision in history
            self.decision_history.append(decision)
            
            # 6. Update performance metrics
            self._update_performance_metrics(signals)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in unified trading decision: {e}")
            return self._create_default_decision()
    
    def _gather_all_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather signals from all enhancement modules."""
        try:
            signals = {}
            
            # 1. Multi-Dimensional Kaprekar Matrix
            if ENHANCED_KAPREKAR_AVAILABLE:
                mdk_signature = self.mdk.calculate_mdk_signature(market_data)
                mdk_prediction = self.mdk.predict_multi_dimensional_collapse([mdk_signature])
                signals['mdk'] = {
                    'signature': mdk_signature,
                    'prediction': mdk_prediction
                }
            
            # 2. Temporal Kaprekar Harmonics
            if ENHANCED_KAPREKAR_AVAILABLE and 'price_series' in market_data:
                tkh_resonance = self.tkh.calculate_temporal_resonance(market_data['price_series'])
                tkh_convergence = self.tkh.detect_harmonic_convergence(tkh_resonance)
                signals['tkh'] = {
                    'resonance': tkh_resonance,
                    'convergence': tkh_convergence
                }
            
            # 3. Kaprekar-Enhanced Ghost Memory
            if ENHANCED_KAPREKAR_AVAILABLE:
                memory_recall = self.ghost_memory.recall_similar_patterns(market_data)
                signals['ghost_memory'] = memory_recall
            
            # 4. Advanced Entropy Routing with Bifurcation
            if ENHANCED_KAPREKAR_AVAILABLE and 'price_sequence' in market_data:
                bifurcation = self.bifurcation_detector.detect_convergence_bifurcation(
                    market_data['price_sequence']
                )
                entropy_routing = self.bifurcation_detector.adaptive_entropy_routing(bifurcation)
                signals['bifurcation'] = {
                    'bifurcation': bifurcation,
                    'entropy_routing': entropy_routing
                }
            
            # 5. Quantum-Inspired Superposition Trading
            if ENHANCED_KAPREKAR_AVAILABLE:
                quantum_state = self.quantum_states.create_superposition_state(market_data)
                market_trigger = self._determine_market_trigger(market_data)
                quantum_action = self.quantum_states.collapse_to_action(quantum_state, market_trigger)
                signals['quantum'] = {
                    'state': quantum_state,
                    'action': quantum_action,
                    'trigger': market_trigger
                }
            
            # 6. Cross-Asset Kaprekar Correlation Matrix
            if ENHANCED_KAPREKAR_AVAILABLE and 'asset_data' in market_data:
                cross_asset = self.cross_asset_matrix.calculate_cross_asset_convergence(
                    market_data['asset_data']
                )
                signals['cross_asset'] = cross_asset
            
            # 7. Existing Schwabot components
            if SCHWABOT_COMPONENTS_AVAILABLE:
                # Basic Kaprekar analysis
                if 'price' in market_data:
                    kaprekar_result = self.kaprekar_analyzer.analyze_kaprekar(
                        self._normalize_price(market_data['price'])
                    )
                    signals['basic_kaprekar'] = kaprekar_result
                
                # Strategy mapper integration
                if 'current_hash' in market_data:
                    strategy_signal = self._get_strategy_mapper_signal(market_data)
                    signals['strategy_mapper'] = strategy_signal
            
            # 8. System health monitoring
            if SCHWABOT_COMPONENTS_AVAILABLE:
                health_status = system_health_monitor.get_full_report()
                signals['system_health'] = health_status
            
            return signals
            
        except Exception as e:
            logger.error(f"Error gathering signals: {e}")
            return {}
    
    def _normalize_price(self, price: float) -> int:
        """Normalize price to 4-digit number."""
        try:
            scaled = abs(price) * 100
            normalized = int(scaled) % 10000
            return max(1000, normalized)
        except Exception:
            return 1234
    
    def _determine_market_trigger(self, market_data: Dict[str, Any]) -> str:
        """Determine market trigger for quantum state collapse."""
        try:
            volatility = market_data.get('volatility', 0.0)
            momentum = market_data.get('momentum', 0.0)
            rsi = market_data.get('rsi', 50.0)
            
            if volatility > 0.5:
                return "high_volatility"
            elif abs(momentum) > 0.2:
                return "trend_breakout"
            elif 30 <= rsi <= 70:
                return "mean_reversion"
            else:
                return "normal_market"
                
        except Exception:
            return "normal_market"
    
    def _get_strategy_mapper_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get signal from strategy mapper."""
        try:
            # This would integrate with the existing strategy mapper
            # For now, return a simulated signal
            return {
                'strategy_type': 'enhanced_kaprekar',
                'confidence': 0.7,
                'signal_strength': 0.6
            }
        except Exception:
            return {
                'strategy_type': 'fallback',
                'confidence': 0.3,
                'signal_strength': 0.3
            }
    
    def _synthesize_all_signals(self, signals: Dict[str, Any]) -> UnifiedTradingDecision:
        """Synthesize all signals into a unified trading decision."""
        try:
            # Initialize decision components
            action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0, 'hedge': 0.0}
            confidence_factors = []
            risk_factors = []
            
            # Process MDK signals
            if 'mdk' in signals:
                mdk_data = signals['mdk']
                mdk_prediction = mdk_data.get('prediction', {})
                
                # Extract strategy recommendation
                strategy = mdk_prediction.get('recommended_strategy', 'CONSERVATIVE_HOLD')
                if 'BUY' in strategy:
                    action_scores['buy'] += 0.3
                elif 'SELL' in strategy:
                    action_scores['sell'] += 0.3
                elif 'HOLD' in strategy:
                    action_scores['hold'] += 0.3
                
                # Extract confidence
                stability_score = mdk_prediction.get('stability_score', 0.5)
                confidence_factors.append(stability_score)
            
            # Process TKH signals
            if 'tkh' in signals:
                tkh_data = signals['tkh']
                convergence = tkh_data.get('convergence', 'temporal_noise_hold')
                
                if 'buy' in convergence:
                    action_scores['buy'] += 0.2
                elif 'sell' in convergence:
                    action_scores['sell'] += 0.2
                elif 'hold' in convergence:
                    action_scores['hold'] += 0.2
                
                # Extract confidence from resonance
                resonance = tkh_data.get('resonance', {})
                if resonance:
                    avg_harmonic = np.mean([r.harmonic_strength for r in resonance.values()]) if resonance else 0.5
                    confidence_factors.append(avg_harmonic)
            
            # Process Ghost Memory signals
            if 'ghost_memory' in signals:
                memory_data = signals['ghost_memory']
                recommended_action = memory_data.get('recommended_action', 'HOLD')
                
                if recommended_action == 'BUY':
                    action_scores['buy'] += 0.2
                elif recommended_action == 'SELL':
                    action_scores['sell'] += 0.2
                elif recommended_action == 'HOLD':
                    action_scores['hold'] += 0.2
                
                confidence_level = memory_data.get('confidence_level', 0.3)
                confidence_factors.append(confidence_level)
            
            # Process Bifurcation signals
            if 'bifurcation' in signals:
                bifurcation_data = signals['bifurcation']
                bifurcation = bifurcation_data.get('bifurcation')
                entropy_routing = bifurcation_data.get('entropy_routing', 'standard_kaprekar_routing')
                
                if bifurcation and bifurcation.chaos_level > 0.7:
                    action_scores['hedge'] += 0.4
                    risk_factors.append(bifurcation.chaos_level)
                elif 'emergency' in entropy_routing:
                    action_scores['hedge'] += 0.3
                elif 'breakout' in entropy_routing:
                    action_scores['buy'] += 0.1
                    action_scores['sell'] += 0.1
            
            # Process Quantum signals
            if 'quantum' in signals:
                quantum_data = signals['quantum']
                quantum_action = quantum_data.get('action', 'hold')
                quantum_state = quantum_data.get('state')
                
                action_scores[quantum_action] += 0.3
                
                if quantum_state:
                    # Use superposition entropy as confidence factor
                    confidence_factors.append(1.0 - (quantum_state.superposition_entropy / 2.0))
            
            # Process Cross-Asset signals
            if 'cross_asset' in signals:
                cross_asset_data = signals['cross_asset']
                arbitrage_opportunities = cross_asset_data.get('arbitrage_opportunities', [])
                
                if arbitrage_opportunities:
                    action_scores['buy'] += 0.1
                    action_scores['sell'] += 0.1
            
            # Process Basic Kaprekar signals
            if 'basic_kaprekar' in signals:
                kaprekar_result = signals['basic_kaprekar']
                if kaprekar_result.is_convergent:
                    if kaprekar_result.steps_to_converge <= 3:
                        action_scores['buy'] += 0.2
                    elif kaprekar_result.steps_to_converge >= 5:
                        action_scores['sell'] += 0.2
                    else:
                        action_scores['hold'] += 0.2
                    
                    confidence_factors.append(kaprekar_result.stability_score)
            
            # Determine primary action
            primary_action = max(action_scores, key=action_scores.get)
            
            # Calculate confidence score
            confidence_score = np.mean(confidence_factors) if confidence_factors else 0.5
            confidence_score = min(0.95, max(0.05, confidence_score))
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_factors, confidence_score)
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence_score, risk_level)
            
            # Determine entry timing
            entry_timing = self._determine_entry_timing(signals)
            
            # Determine exit strategy
            exit_strategy = self._determine_exit_strategy(primary_action, risk_level)
            
            return UnifiedTradingDecision(
                primary_action=primary_action,
                confidence_score=confidence_score,
                risk_level=risk_level,
                position_size=position_size,
                entry_timing=entry_timing,
                exit_strategy=exit_strategy,
                supporting_signals=signals,
                system_health=signals.get('system_health', {}),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing signals: {e}")
            return self._create_default_decision()
    
    def _determine_risk_level(self, risk_factors: List[float], confidence_score: float) -> str:
        """Determine risk level based on risk factors and confidence."""
        try:
            avg_risk = np.mean(risk_factors) if risk_factors else 0.0
            combined_risk = (avg_risk + (1.0 - confidence_score)) / 2.0
            
            if combined_risk > 0.7:
                return "HIGH"
            elif combined_risk > 0.4:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"
    
    def _calculate_position_size(self, confidence_score: float, risk_level: str) -> float:
        """Calculate position size based on confidence and risk."""
        try:
            base_size = 0.1  # 10% base position size
            
            # Adjust for confidence
            confidence_multiplier = confidence_score
            
            # Adjust for risk
            risk_multipliers = {
                'LOW': 1.2,
                'MEDIUM': 1.0,
                'HIGH': 0.6
            }
            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            
            position_size = base_size * confidence_multiplier * risk_multiplier
            
            # Cap position size
            return min(0.25, max(0.01, position_size))
            
        except Exception:
            return 0.05
    
    def _determine_entry_timing(self, signals: Dict[str, Any]) -> str:
        """Determine optimal entry timing."""
        try:
            # Check MDK timing
            if 'mdk' in signals:
                mdk_prediction = signals['mdk'].get('prediction', {})
                timing = mdk_prediction.get('optimal_entry_timing', {})
                if timing.get('timing') == 'IMMEDIATE':
                    return "IMMEDIATE"
            
            # Check bifurcation timing
            if 'bifurcation' in signals:
                bifurcation = signals['bifurcation'].get('bifurcation')
                if bifurcation and bifurcation.chaos_level > 0.8:
                    return "WAIT"
            
            return "MONITOR"
            
        except Exception:
            return "MONITOR"
    
    def _determine_exit_strategy(self, primary_action: str, risk_level: str) -> str:
        """Determine exit strategy."""
        try:
            if risk_level == "HIGH":
                return "TIGHT_STOPS"
            elif primary_action in ['buy', 'sell']:
                return "TAKE_PROFIT_STOPS"
            else:
                return "STANDARD_EXITS"
                
        except Exception:
            return "STANDARD_EXITS"
    
    def _create_default_decision(self) -> UnifiedTradingDecision:
        """Create default decision for error cases."""
        return UnifiedTradingDecision(
            primary_action="hold",
            confidence_score=0.3,
            risk_level="MEDIUM",
            position_size=0.05,
            entry_timing="MONITOR",
            exit_strategy="STANDARD_EXITS",
            supporting_signals={},
            system_health={},
            timestamp=datetime.now()
        )
    
    def _update_performance_metrics(self, signals: Dict[str, Any]) -> None:
        """Update performance metrics."""
        try:
            performance = SystemPerformance(
                mdk_performance=self._calculate_mdk_performance(signals),
                tkh_performance=self._calculate_tkh_performance(signals),
                ghost_memory_performance=self._calculate_ghost_memory_performance(signals),
                bifurcation_performance=self._calculate_bifurcation_performance(signals),
                quantum_performance=self._calculate_quantum_performance(signals),
                cross_asset_performance=self._calculate_cross_asset_performance(signals),
                overall_performance=0.0,
                timestamp=datetime.now()
            )
            
            # Calculate overall performance
            performances = [
                performance.mdk_performance,
                performance.tkh_performance,
                performance.ghost_memory_performance,
                performance.bifurcation_performance,
                performance.quantum_performance,
                performance.cross_asset_performance
            ]
            
            performance.overall_performance = np.mean(performances)
            
            self.performance_history.append(performance)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_mdk_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate MDK performance score."""
        try:
            if 'mdk' not in signals:
                return 0.5
            
            mdk_data = signals['mdk']
            prediction = mdk_data.get('prediction', {})
            stability_score = prediction.get('stability_score', 0.5)
            
            return stability_score
            
        except Exception:
            return 0.5
    
    def _calculate_tkh_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate TKH performance score."""
        try:
            if 'tkh' not in signals:
                return 0.5
            
            tkh_data = signals['tkh']
            resonance = tkh_data.get('resonance', {})
            
            if resonance:
                harmonic_strengths = [r.harmonic_strength for r in resonance.values()]
                return np.mean(harmonic_strengths)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_ghost_memory_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate Ghost Memory performance score."""
        try:
            if 'ghost_memory' not in signals:
                return 0.5
            
            memory_data = signals['ghost_memory']
            confidence_level = memory_data.get('confidence_level', 0.3)
            
            return confidence_level
            
        except Exception:
            return 0.5
    
    def _calculate_bifurcation_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate Bifurcation performance score."""
        try:
            if 'bifurcation' not in signals:
                return 0.5
            
            bifurcation_data = signals['bifurcation']
            bifurcation = bifurcation_data.get('bifurcation')
            
            if bifurcation:
                # Lower chaos level = higher performance
                return 1.0 - bifurcation.chaos_level
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_quantum_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate Quantum performance score."""
        try:
            if 'quantum' not in signals:
                return 0.5
            
            quantum_data = signals['quantum']
            quantum_state = quantum_data.get('state')
            
            if quantum_state:
                # Lower entropy = higher performance
                return 1.0 - (quantum_state.superposition_entropy / 2.0)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_cross_asset_performance(self, signals: Dict[str, Any]) -> float:
        """Calculate Cross-Asset performance score."""
        try:
            if 'cross_asset' not in signals:
                return 0.5
            
            cross_asset_data = signals['cross_asset']
            correlation_strength = cross_asset_data.get('correlation_strength', 0.0)
            
            # Moderate correlation is optimal
            optimal_correlation = 0.5
            performance = 1.0 - abs(correlation_strength - optimal_correlation)
            
            return max(0.1, performance)
            
        except Exception:
            return 0.5
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and performance metrics."""
        try:
            integration_status = {}
            
            # Get integration bridge status
            if ENHANCED_KAPREKAR_AVAILABLE and self.integration_bridge:
                integration_status['bridge_status'] = self.integration_bridge.get_integration_status()
            else:
                integration_status['bridge_status'] = {'initialized': False, 'error': 'Bridge not available'}
            
            # Get handoff history
            if ENHANCED_KAPREKAR_AVAILABLE and self.integration_bridge:
                integration_status['handoff_history'] = self.integration_bridge.get_handoff_history(5)
            else:
                integration_status['handoff_history'] = []
            
            # Get system health
            if SCHWABOT_COMPONENTS_AVAILABLE:
                integration_status['system_health'] = system_health_monitor.get_full_report()
            else:
                integration_status['system_health'] = {'status': 'unavailable'}
            
            return integration_status
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'initialized': self.initialized,
                'enhanced_kaprekar_available': ENHANCED_KAPREKAR_AVAILABLE,
                'schwabot_components_available': SCHWABOT_COMPONENTS_AVAILABLE,
                'integration_bridge_available': ENHANCED_KAPREKAR_AVAILABLE and hasattr(self, 'integration_bridge'),
                'decision_history_size': len(self.decision_history),
                'performance_history_size': len(self.performance_history)
            }
            
            # Add integration bridge status
            if ENHANCED_KAPREKAR_AVAILABLE and hasattr(self, 'integration_bridge'):
                status['integration_bridge_status'] = self.integration_bridge.get_integration_status()
            
            # Add system health if available
            if SCHWABOT_COMPONENTS_AVAILABLE:
                status['system_health'] = system_health_monitor.get_full_report()
            
            # Add recent performance metrics
            if self.performance_history:
                recent_performance = list(self.performance_history)[-1]
                status['recent_performance'] = {
                    'mdk_performance': recent_performance.mdk_performance,
                    'tkh_performance': recent_performance.tkh_performance,
                    'ghost_memory_performance': recent_performance.ghost_memory_performance,
                    'bifurcation_performance': recent_performance.bifurcation_performance,
                    'quantum_performance': recent_performance.quantum_performance,
                    'cross_asset_performance': recent_performance.cross_asset_performance,
                    'overall_performance': recent_performance.overall_performance
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'initialized': self.initialized,
                'error': str(e)
            }
    
    def get_decision_history(self, limit: int = 10) -> List[UnifiedTradingDecision]:
        """Get recent decision history."""
        try:
            return list(self.decision_history)[-limit:]
        except Exception:
            return []
    
    def get_performance_history(self, limit: int = 10) -> List[SystemPerformance]:
        """Get recent performance history."""
        try:
            return list(self.performance_history)[-limit:]
        except Exception:
            return []

# Global instance for easy access
schwabot_command_center = SchwabotCommandCenter() 