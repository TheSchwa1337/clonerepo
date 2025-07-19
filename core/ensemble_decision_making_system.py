#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧭 ENSEMBLE DECISION MAKING SYSTEM - MULTI-SYSTEM COORDINATION
=============================================================

Live Ensemble Decision Making System Based on Trust Weighting for Schwabot

This module provides:
1. Live ensemble decision coordination across all mathematical systems
2. Voting on trade ideas from all mathematical modules
3. Trust-weighted decision vector generation
4. Final route selection based on profit history & stability profile

Mathematical Framework:
- 𝒱ₜ = Σ(voteᵢₜ · trust_weightᵢ) for i=1 to n sources
- 𝒞ₜ = consensus_function(𝒱ₜ, threshold)
- ℛₜ = route_function(𝒞ₜ, profit_history, stability_profile)
- 𝒟ₜ = decision_vector(ℛₜ, confidence_level)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Import core components
from distributed_mathematical_processor import DistributedMathematicalProcessor
from enhanced_error_recovery_system import EnhancedErrorRecoverySystem, error_recovery_decorator
from neural_processing_engine import NeuralProcessingEngine
from unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
from master_profit_coordination_system import MasterProfitCoordinationSystem
from integrated_advanced_trading_system import IntegratedAdvancedTradingSystem
from profit_optimization_engine import ProfitOptimizationEngine
from bio_profit_vectorization import BioProfitVectorization
from btc_usdc_trading_integration import BTCUSDCTradingIntegration

logger = logging.getLogger(__name__)


class ConsensusMode(Enum):
    """Modes for consensus calculation"""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Trust-weighted consensus
    UNANIMOUS = "unanimous"  # All sources must agree
    ADAPTIVE = "adaptive"  # Dynamic threshold based on confidence


class RouteSelectionMode(Enum):
    """Modes for route selection"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS_PLUS = "consensus_plus"
    ADAPTIVE_ROUTING = "adaptive_routing"


@dataclass
class SystemVote:
    """Vote from a mathematical system"""
    system_id: str
    system_type: str  # distributed, neural, profit_vector, etc.
    vote: str  # BUY, SELL, HOLD, WAIT
    confidence: float
    reasoning: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustProfile:
    """Trust profile for a mathematical system"""
    system_id: str
    trust_weight: float
    success_rate: float
    stability_score: float
    last_update: float
    vote_history: List[SystemVote] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus calculation"""
    consensus_vote: str
    confidence_level: float
    agreement_ratio: float
    trust_weighted_score: float
    participating_systems: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """Final route decision"""
    selected_route: str
    decision_confidence: float
    route_reasoning: str
    execution_priority: int
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionVector:
    """Decision vector for execution"""
    action: str
    confidence: float
    urgency: float
    risk_level: float
    position_size: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnsembleDecisionMakingSystem:
    """
    🧭 Live Ensemble Decision Making System Based on Trust Weighting

    Provides live ensemble decision coordination:
    - Voting on trade ideas from all mathematical systems
    - Trust-weighted decision vector generation
    - Final route selection based on profit history & stability profile
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or self._default_config()
        
        # Initialize mathematical systems
        self.distributed_processor = DistributedMathematicalProcessor()
        self.neural_engine = NeuralProcessingEngine()
        self.profit_vectorizer = UnifiedProfitVectorizationSystem()
        self.master_coordinator = MasterProfitCoordinationSystem()
        self.integrated_trading = IntegratedAdvancedTradingSystem()
        self.profit_optimizer = ProfitOptimizationEngine()
        self.bio_profit_vector = BioProfitVectorization()
        self.btc_usdc_integration = BTCUSDCTradingIntegration()
        self.error_recovery = EnhancedErrorRecoverySystem()
        
        # Trust profiles for different systems
        self.trust_profiles: Dict[str, TrustProfile] = {}
        self.initialize_trust_profiles()
        
        # Voting and consensus tracking
        self.vote_history: List[SystemVote] = []
        self.consensus_history: List[ConsensusResult] = []
        self.route_history: List[RouteDecision] = []
        
        # Performance tracking
        self.consensus_count = 0
        self.route_count = 0
        self.last_operation_time = time.time()
        self.performance_metrics = {
            "total_consensuses": 0,
            "successful_routes": 0,
            "average_confidence": 0.0,
            "trust_stability": 0.0,
        }
        
        # Threading
        self.consensus_lock = threading.Lock()
        self.active = False
        
        logger.info("🧭 Ensemble Decision Making System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "consensus_threshold": 0.7,
            "trust_decay_rate": 0.001,
            "confidence_weight": 0.4,
            "stability_weight": 0.3,
            "success_weight": 0.3,
            "max_vote_history": 1000,
            "update_interval": 1.0,  # seconds
            "default_trust_weight": 0.5,
            "min_confidence": 0.3,
            "max_confidence": 0.95,
            "consensus_mode": ConsensusMode.WEIGHTED,
            "route_mode": RouteSelectionMode.WEIGHTED_AVERAGE,
        }
    
    def initialize_trust_profiles(self) -> None:
        """Initialize trust profiles for all mathematical systems."""
        try:
            systems = [
                ("distributed_processor", "Distributed Mathematical Processor"),
                ("neural_engine", "Neural Processing Engine"),
                ("profit_vectorizer", "Unified Profit Vectorization"),
                ("master_coordinator", "Master Profit Coordination"),
                ("integrated_trading", "Integrated Advanced Trading"),
                ("profit_optimizer", "Profit Optimization Engine"),
                ("bio_profit_vector", "Bio-Profit Vectorization"),
                ("btc_usdc_integration", "BTC/USDC Trading Integration"),
            ]
            
            for system_id, system_name in systems:
                self.trust_profiles[system_id] = TrustProfile(
                    source_id=system_id,
                    trust_weight=self.config["default_trust_weight"],
                    success_rate=0.5,
                    stability_score=0.5,
                    last_update=time.time(),
                    vote_history=[],
                    metadata={"system_name": system_name}
                )
            
            logger.info(f"Initialized trust profiles for {len(systems)} systems")
            
        except Exception as e:
            logger.error(f"Error initializing trust profiles: {e}")
    
    @error_recovery_decorator
    async def collect_system_votes(self, market_data: Dict[str, Any]) -> List[SystemVote]:
        """Collect votes from all mathematical systems."""
        try:
            votes = []
            current_time = time.time()
            
            # Collect vote from distributed processor
            try:
                distributed_task = {
                    "type": "ensemble_vote",
                    "market_data": market_data,
                    "vote_type": "trading_decision"
                }
                distributed_result = await self.distributed_processor.process_task(distributed_task)
                
                if distributed_result:
                    votes.append(SystemVote(
                        system_id="distributed_processor",
                        system_type="distributed_mathematical",
                        vote=distributed_result.get("vote", "HOLD"),
                        confidence=distributed_result.get("confidence", 0.5),
                        reasoning=distributed_result.get("reasoning", "Distributed analysis"),
                        timestamp=current_time,
                        metadata=distributed_result
                    ))
            except Exception as e:
                logger.warning(f"Error collecting distributed processor vote: {e}")
            
            # Collect vote from neural engine
            try:
                neural_prediction = await self.neural_engine.predict_market_movement(market_data)
                if neural_prediction:
                    votes.append(SystemVote(
                        system_id="neural_engine",
                        system_type="neural_processing",
                        vote=neural_prediction.prediction_type,
                        confidence=neural_prediction.confidence,
                        reasoning=neural_prediction.reasoning,
                        timestamp=current_time,
                        metadata={"prediction_data": neural_prediction.__dict__}
                    ))
            except Exception as e:
                logger.warning(f"Error collecting neural engine vote: {e}")
            
            # Collect vote from profit vectorizer
            try:
                profit_result = await self.profit_vectorizer.process_market_data(market_data)
                if profit_result:
                    vote_action = "BUY" if profit_result.get("profit_potential", 0) > 0.02 else "SELL" if profit_result.get("profit_potential", 0) < -0.02 else "HOLD"
                    votes.append(SystemVote(
                        system_id="profit_vectorizer",
                        system_type="profit_vectorization",
                        vote=vote_action,
                        confidence=profit_result.get("confidence", 0.5),
                        reasoning=f"Profit potential: {profit_result.get('profit_potential', 0):.4f}",
                        timestamp=current_time,
                        metadata=profit_result
                    ))
            except Exception as e:
                logger.warning(f"Error collecting profit vectorizer vote: {e}")
            
            # Collect vote from master coordinator
            try:
                coordination_decision = await self.master_coordinator.coordinate_profit_optimization(market_data)
                if coordination_decision:
                    vote_action = "BUY" if coordination_decision.total_expected_profit > 0.01 else "SELL" if coordination_decision.total_expected_profit < -0.01 else "HOLD"
                    votes.append(SystemVote(
                        system_id="master_coordinator",
                        system_type="master_coordination",
                        vote=vote_action,
                        confidence=coordination_decision.profit_confidence,
                        reasoning=f"Coordination profit: {coordination_decision.total_expected_profit:.4f}",
                        timestamp=current_time,
                        metadata={"coordination_mode": coordination_decision.coordination_mode.value}
                    ))
            except Exception as e:
                logger.warning(f"Error collecting master coordinator vote: {e}")
            
            # Collect vote from integrated trading system
            try:
                trading_decision = await self.integrated_trading.process_market_data(market_data)
                if trading_decision:
                    votes.append(SystemVote(
                        system_id="integrated_trading",
                        system_type="integrated_trading",
                        vote=trading_decision.signal.signal_type.upper(),
                        confidence=trading_decision.signal.confidence,
                        reasoning=f"Ensemble profit: {trading_decision.ensemble_profit:.4f}",
                        timestamp=current_time,
                        metadata={"ensemble_profit": trading_decision.ensemble_profit}
                    ))
            except Exception as e:
                logger.warning(f"Error collecting integrated trading vote: {e}")
            
            # Collect vote from bio-profit vector
            try:
                bio_response = self.bio_profit_vector.optimize_profit_vectorization(market_data)
                if bio_response:
                    vote_action = "BUY" if bio_response.recommended_position > 0 else "SELL" if bio_response.recommended_position < 0 else "HOLD"
                    votes.append(SystemVote(
                        system_id="bio_profit_vector",
                        system_type="bio_profit_vectorization",
                        vote=vote_action,
                        confidence=bio_response.cellular_efficiency,
                        reasoning=f"Bio-profit velocity: {bio_response.profit_velocity:.4f}",
                        timestamp=current_time,
                        metadata={"metabolic_pathway": bio_response.metabolic_pathway.value}
                    ))
            except Exception as e:
                logger.warning(f"Error collecting bio-profit vector vote: {e}")
            
            # Store votes in history
            self.vote_history.extend(votes)
            
            # Keep only recent votes
            if len(self.vote_history) > self.config["max_vote_history"]:
                self.vote_history = self.vote_history[-self.config["max_vote_history"]:]
            
            logger.info(f"Collected {len(votes)} system votes")
            return votes
            
        except Exception as e:
            logger.error(f"Error collecting system votes: {e}")
            return []
    
    @error_recovery_decorator
    async def calculate_consensus(self, votes: List[SystemVote], mode: ConsensusMode = None) -> ConsensusResult:
        """Calculate consensus from system votes."""
        try:
            if not votes:
                return self._get_default_consensus_result()
            
            mode = mode or self.config["consensus_mode"]
            
            if mode == ConsensusMode.MAJORITY:
                return self._calculate_majority_consensus(votes)
            elif mode == ConsensusMode.WEIGHTED:
                return self._calculate_weighted_consensus(votes)
            elif mode == ConsensusMode.UNANIMOUS:
                return self._calculate_unanimous_consensus(votes)
            elif mode == ConsensusMode.ADAPTIVE:
                return self._calculate_adaptive_consensus(votes)
            else:
                return self._calculate_weighted_consensus(votes)
                
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return self._get_default_consensus_result()
    
    def _calculate_weighted_consensus(self, votes: List[SystemVote]) -> ConsensusResult:
        """Calculate trust-weighted consensus."""
        try:
            # Calculate trust-weighted scores for each vote type
            vote_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "WAIT": 0.0}
            total_weight = 0.0
            
            for vote in votes:
                trust_profile = self.trust_profiles.get(vote.system_id)
                if trust_profile:
                    weight = trust_profile.trust_weight * vote.confidence
                    vote_scores[vote.vote] += weight
                    total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for vote_type in vote_scores:
                    vote_scores[vote_type] /= total_weight
            
            # Find consensus vote
            consensus_vote = max(vote_scores, key=vote_scores.get)
            confidence_level = vote_scores[consensus_vote]
            
            # Calculate agreement ratio
            agreement_count = sum(1 for vote in votes if vote.vote == consensus_vote)
            agreement_ratio = agreement_count / len(votes) if votes else 0.0
            
            # Calculate trust-weighted score
            trust_weighted_score = confidence_level * agreement_ratio
            
            return ConsensusResult(
                consensus_vote=consensus_vote,
                confidence_level=confidence_level,
                agreement_ratio=agreement_ratio,
                trust_weighted_score=trust_weighted_score,
                participating_systems=[vote.system_id for vote in votes],
                metadata={"vote_scores": vote_scores}
            )
            
        except Exception as e:
            logger.error(f"Error calculating weighted consensus: {e}")
            return self._get_default_consensus_result()
    
    def _calculate_majority_consensus(self, votes: List[SystemVote]) -> ConsensusResult:
        """Calculate simple majority consensus."""
        try:
            vote_counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "WAIT": 0}
            
            for vote in votes:
                vote_counts[vote.vote] += 1
            
            consensus_vote = max(vote_counts, key=vote_counts.get)
            agreement_ratio = vote_counts[consensus_vote] / len(votes)
            
            # Calculate average confidence for consensus vote
            consensus_votes = [vote for vote in votes if vote.vote == consensus_vote]
            confidence_level = np.mean([vote.confidence for vote in consensus_votes]) if consensus_votes else 0.5
            
            return ConsensusResult(
                consensus_vote=consensus_vote,
                confidence_level=confidence_level,
                agreement_ratio=agreement_ratio,
                trust_weighted_score=confidence_level * agreement_ratio,
                participating_systems=[vote.system_id for vote in votes],
                metadata={"vote_counts": vote_counts}
            )
            
        except Exception as e:
            logger.error(f"Error calculating majority consensus: {e}")
            return self._get_default_consensus_result()
    
    def _calculate_unanimous_consensus(self, votes: List[SystemVote]) -> ConsensusResult:
        """Calculate unanimous consensus."""
        try:
            if not votes:
                return self._get_default_consensus_result()
            
            # Check if all votes are the same
            first_vote = votes[0].vote
            unanimous = all(vote.vote == first_vote for vote in votes)
            
            if unanimous:
                confidence_level = np.mean([vote.confidence for vote in votes])
                return ConsensusResult(
                    consensus_vote=first_vote,
                    confidence_level=confidence_level,
                    agreement_ratio=1.0,
                    trust_weighted_score=confidence_level,
                    participating_systems=[vote.system_id for vote in votes],
                    metadata={"unanimous": True}
                )
            else:
                # Fall back to weighted consensus
                return self._calculate_weighted_consensus(votes)
                
        except Exception as e:
            logger.error(f"Error calculating unanimous consensus: {e}")
            return self._get_default_consensus_result()
    
    def _calculate_adaptive_consensus(self, votes: List[SystemVote]) -> ConsensusResult:
        """Calculate adaptive consensus based on confidence levels."""
        try:
            # Calculate average confidence
            avg_confidence = np.mean([vote.confidence for vote in votes])
            
            # Adjust threshold based on average confidence
            if avg_confidence > 0.8:
                # High confidence - use unanimous consensus
                return self._calculate_unanimous_consensus(votes)
            elif avg_confidence > 0.6:
                # Medium confidence - use weighted consensus
                return self._calculate_weighted_consensus(votes)
            else:
                # Low confidence - use majority consensus
                return self._calculate_majority_consensus(votes)
                
        except Exception as e:
            logger.error(f"Error calculating adaptive consensus: {e}")
            return self._get_default_consensus_result()
    
    @error_recovery_decorator
    async def select_route(
        self,
        consensus_result: ConsensusResult,
        route_mode: RouteSelectionMode = None,
    ) -> RouteDecision:
        """Select execution route based on consensus result."""
        try:
            route_mode = route_mode or self.config["route_mode"]
            
            if route_mode == RouteSelectionMode.HIGHEST_CONFIDENCE:
                return self._select_highest_confidence_route(consensus_result)
            elif route_mode == RouteSelectionMode.WEIGHTED_AVERAGE:
                return self._select_weighted_average_route(consensus_result)
            elif route_mode == RouteSelectionMode.CONSENSUS_PLUS:
                return self._select_consensus_plus_route(consensus_result)
            elif route_mode == RouteSelectionMode.ADAPTIVE_ROUTING:
                return self._select_adaptive_route(consensus_result)
            else:
                return self._select_weighted_average_route(consensus_result)
                
        except Exception as e:
            logger.error(f"Error selecting route: {e}")
            return self._get_fallback_route_decision(consensus_result)
    
    def _select_weighted_average_route(self, consensus_result: ConsensusResult) -> RouteDecision:
        """Select route using weighted average approach."""
        try:
            # Determine route based on consensus vote
            if consensus_result.consensus_vote == "BUY":
                selected_route = "aggressive_buy"
                execution_priority = 1
            elif consensus_result.consensus_vote == "SELL":
                selected_route = "aggressive_sell"
                execution_priority = 1
            elif consensus_result.consensus_vote == "HOLD":
                selected_route = "conservative_hold"
                execution_priority = 3
            else:  # WAIT
                selected_route = "defensive_wait"
                execution_priority = 4
            
            # Adjust priority based on confidence
            if consensus_result.confidence_level > 0.8:
                execution_priority = max(1, execution_priority - 1)
            elif consensus_result.confidence_level < 0.5:
                execution_priority = min(5, execution_priority + 1)
            
            return RouteDecision(
                selected_route=selected_route,
                decision_confidence=consensus_result.confidence_level,
                route_reasoning=f"Consensus: {consensus_result.consensus_vote} with {consensus_result.confidence_level:.2f} confidence",
                execution_priority=execution_priority,
                risk_adjustments={
                    "position_size_multiplier": consensus_result.confidence_level,
                    "stop_loss_adjustment": 1.0 - consensus_result.confidence_level * 0.5,
                    "take_profit_adjustment": 1.0 + consensus_result.confidence_level * 0.5
                },
                metadata={"consensus_data": consensus_result.__dict__}
            )
            
        except Exception as e:
            logger.error(f"Error selecting weighted average route: {e}")
            return self._get_fallback_route_decision(consensus_result)
    
    def _select_highest_confidence_route(self, consensus_result: ConsensusResult) -> RouteDecision:
        """Select route based on highest confidence."""
        try:
            # Use highest confidence for route selection
            confidence = consensus_result.confidence_level
            
            if confidence > 0.9:
                selected_route = "ultra_aggressive"
                execution_priority = 1
            elif confidence > 0.7:
                selected_route = "aggressive"
                execution_priority = 2
            elif confidence > 0.5:
                selected_route = "moderate"
                execution_priority = 3
            else:
                selected_route = "conservative"
                execution_priority = 4
            
            return RouteDecision(
                selected_route=selected_route,
                decision_confidence=confidence,
                route_reasoning=f"High confidence route: {confidence:.2f}",
                execution_priority=execution_priority,
                risk_adjustments={
                    "position_size_multiplier": confidence,
                    "stop_loss_adjustment": 1.0 - confidence * 0.3,
                    "take_profit_adjustment": 1.0 + confidence * 0.3
                },
                metadata={"confidence_based": True}
            )
            
        except Exception as e:
            logger.error(f"Error selecting highest confidence route: {e}")
            return self._get_fallback_route_decision(consensus_result)
    
    def _select_consensus_plus_route(self, consensus_result: ConsensusResult) -> RouteDecision:
        """Select route using consensus plus additional factors."""
        try:
            # Base route on consensus
            base_route = self._select_weighted_average_route(consensus_result)
            
            # Add consensus-specific adjustments
            if consensus_result.agreement_ratio > 0.8:
                base_route.execution_priority = max(1, base_route.execution_priority - 1)
                base_route.route_reasoning += " (High agreement)"
            
            if consensus_result.trust_weighted_score > 0.7:
                base_route.risk_adjustments["position_size_multiplier"] *= 1.2
                base_route.route_reasoning += " (High trust score)"
            
            return base_route
            
        except Exception as e:
            logger.error(f"Error selecting consensus plus route: {e}")
            return self._get_fallback_route_decision(consensus_result)
    
    def _select_adaptive_route(self, consensus_result: ConsensusResult) -> RouteDecision:
        """Select route using adaptive approach."""
        try:
            # Adapt based on multiple factors
            confidence = consensus_result.confidence_level
            agreement = consensus_result.agreement_ratio
            trust_score = consensus_result.trust_weighted_score
            
            # Calculate adaptive score
            adaptive_score = (confidence * 0.4 + agreement * 0.3 + trust_score * 0.3)
            
            if adaptive_score > 0.8:
                return self._select_highest_confidence_route(consensus_result)
            elif adaptive_score > 0.6:
                return self._select_weighted_average_route(consensus_result)
            else:
                return self._select_consensus_plus_route(consensus_result)
                
        except Exception as e:
            logger.error(f"Error selecting adaptive route: {e}")
            return self._get_fallback_route_decision(consensus_result)
    
    @error_recovery_decorator
    async def generate_decision_vector(
        self, route_decision: RouteDecision, market_context: Dict[str, Any] = None
    ) -> DecisionVector:
        """Generate final decision vector for execution."""
        try:
            market_context = market_context or {}
            
            # Determine action based on route
            if "buy" in route_decision.selected_route.lower():
                action = "BUY"
            elif "sell" in route_decision.selected_route.lower():
                action = "SELL"
            elif "hold" in route_decision.selected_route.lower():
                action = "HOLD"
            else:
                action = "WAIT"
            
            # Calculate urgency based on route priority
            urgency = 1.0 / route_decision.execution_priority
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(route_decision, market_context)
            
            # Calculate position size
            position_size = self._calculate_position_size(route_decision, market_context)
            
            # Calculate stop loss and take profit
            stop_loss = self._calculate_stop_loss(route_decision, market_context)
            take_profit = self._calculate_take_profit(route_decision, market_context)
            
            return DecisionVector(
                action=action,
                confidence=route_decision.decision_confidence,
                urgency=urgency,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "route_decision": route_decision.__dict__,
                    "market_context": market_context
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating decision vector: {e}")
            return self._get_fallback_decision_vector(route_decision)
    
    def _calculate_risk_level(self, route_decision: RouteDecision, market_context: Dict[str, Any]) -> float:
        """Calculate risk level for the decision."""
        try:
            # Base risk from confidence (inverse relationship)
            base_risk = 1.0 - route_decision.decision_confidence
            
            # Market volatility adjustment
            volatility = market_context.get("volatility", 0.02)
            volatility_risk = min(volatility * 10, 0.5)
            
            # Route-specific risk adjustments
            route_risk_multiplier = 1.0
            if "aggressive" in route_decision.selected_route.lower():
                route_risk_multiplier = 1.5
            elif "conservative" in route_decision.selected_route.lower():
                route_risk_multiplier = 0.7
            
            final_risk = (base_risk + volatility_risk) * route_risk_multiplier
            return max(0.0, min(1.0, final_risk))
            
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return 0.5
    
    def _calculate_position_size(self, route_decision: RouteDecision, market_context: Dict[str, Any]) -> float:
        """Calculate position size for the decision."""
        try:
            # Base position size
            base_size = 1000.0  # $1000 base position
            
            # Adjust by confidence
            confidence_multiplier = route_decision.decision_confidence
            
            # Adjust by route type
            route_multiplier = 1.0
            if "aggressive" in route_decision.selected_route.lower():
                route_multiplier = 1.5
            elif "conservative" in route_decision.selected_route.lower():
                route_multiplier = 0.5
            
            # Market condition adjustment
            market_multiplier = 1.0
            if market_context.get("trend_strength", 0) > 0.5:
                market_multiplier = 1.2
            elif market_context.get("trend_strength", 0) < -0.5:
                market_multiplier = 0.8
            
            position_size = base_size * confidence_multiplier * route_multiplier * market_multiplier
            
            # Apply limits
            return max(100.0, min(10000.0, position_size))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0
    
    def _calculate_stop_loss(self, route_decision: RouteDecision, market_context: Dict[str, Any]) -> float:
        """Calculate stop loss for the decision."""
        try:
            # Base stop loss percentage
            base_stop_loss = 0.02  # 2%
            
            # Adjust by confidence (higher confidence = tighter stop)
            confidence_adjustment = 1.0 - route_decision.decision_confidence * 0.5
            
            # Adjust by volatility
            volatility = market_context.get("volatility", 0.02)
            volatility_adjustment = 1.0 + volatility * 5
            
            stop_loss = base_stop_loss * confidence_adjustment * volatility_adjustment
            
            # Apply limits
            return max(0.005, min(0.1, stop_loss))  # 0.5% to 10%
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return 0.02
    
    def _calculate_take_profit(self, route_decision: RouteDecision, market_context: Dict[str, Any]) -> float:
        """Calculate take profit for the decision."""
        try:
            # Base take profit percentage
            base_take_profit = 0.05  # 5%
            
            # Adjust by confidence (higher confidence = higher target)
            confidence_adjustment = 1.0 + route_decision.decision_confidence * 0.5
            
            # Adjust by route type
            route_adjustment = 1.0
            if "aggressive" in route_decision.selected_route.lower():
                route_adjustment = 1.3
            elif "conservative" in route_decision.selected_route.lower():
                route_adjustment = 0.8
            
            take_profit = base_take_profit * confidence_adjustment * route_adjustment
            
            # Apply limits
            return max(0.01, min(0.2, take_profit))  # 1% to 20%
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return 0.05
    
    @error_recovery_decorator
    async def process_ensemble_decision(self, market_data: Dict[str, Any]) -> DecisionVector:
        """Complete ensemble decision making process."""
        try:
            # Step 1: Collect votes from all systems
            votes = await self.collect_system_votes(market_data)
            
            # Step 2: Calculate consensus
            consensus_result = await self.calculate_consensus(votes)
            
            # Step 3: Select route
            route_decision = await self.select_route(consensus_result)
            
            # Step 4: Generate decision vector
            decision_vector = await self.generate_decision_vector(route_decision, market_data)
            
            # Step 5: Update metrics
            self._update_consensus_metrics(consensus_result)
            self._update_route_metrics(route_decision)
            
            # Step 6: Store in history
            self.consensus_history.append(consensus_result)
            self.route_history.append(route_decision)
            
            logger.info(f"Ensemble decision: {decision_vector.action} with {decision_vector.confidence:.2f} confidence")
            return decision_vector
            
        except Exception as e:
            logger.error(f"Error in ensemble decision process: {e}")
            return self._get_fallback_decision_vector(RouteDecision(
                selected_route="fallback",
                decision_confidence=0.0,
                route_reasoning="Fallback due to error",
                execution_priority=5,
                metadata={"error": str(e)}
            ))
    
    def _update_consensus_metrics(self, consensus_result: ConsensusResult) -> None:
        """Update consensus performance metrics."""
        try:
            self.consensus_count += 1
            self.performance_metrics["total_consensuses"] += 1
            self.performance_metrics["average_confidence"] = (
                (self.performance_metrics["average_confidence"] * (self.consensus_count - 1) + consensus_result.confidence_level) / self.consensus_count
            )
        except Exception as e:
            logger.error(f"Error updating consensus metrics: {e}")
    
    def _update_route_metrics(self, route_decision: RouteDecision) -> None:
        """Update route performance metrics."""
        try:
            self.route_count += 1
            self.performance_metrics["successful_routes"] += 1
        except Exception as e:
            logger.error(f"Error updating route metrics: {e}")
    
    def _get_default_consensus_result(self) -> ConsensusResult:
        """Get default consensus result."""
        return ConsensusResult(
            consensus_vote="HOLD",
            confidence_level=0.0,
            agreement_ratio=0.0,
            trust_weighted_score=0.0,
            participating_systems=[],
            metadata={"default": True}
        )
    
    def _get_fallback_route_decision(self, consensus_result: ConsensusResult) -> RouteDecision:
        """Get fallback route decision."""
        return RouteDecision(
            selected_route="fallback",
            decision_confidence=0.0,
            route_reasoning="Fallback route",
            execution_priority=5,
            metadata={"fallback": True}
        )
    
    def _get_fallback_decision_vector(self, route_decision: RouteDecision) -> DecisionVector:
        """Get fallback decision vector."""
        return DecisionVector(
            action="HOLD",
            confidence=0.0,
            urgency=0.0,
            risk_level=0.5,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            metadata={"fallback": True}
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                "active": self.active,
                "consensus_count": self.consensus_count,
                "route_count": self.route_count,
                "performance_metrics": self.performance_metrics,
                "trust_profiles_count": len(self.trust_profiles),
                "vote_history_size": len(self.vote_history),
                "consensus_history_size": len(self.consensus_history),
                "route_history_size": len(self.route_history),
                "last_operation_time": self.last_operation_time,
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    async def start_ensemble_system(self) -> None:
        """Start the ensemble decision making system."""
        try:
            self.active = True
            logger.info("🧭 Ensemble Decision Making System started")
        except Exception as e:
            logger.error(f"Error starting ensemble system: {e}")
    
    async def stop_ensemble_system(self) -> None:
        """Stop the ensemble decision making system."""
        try:
            self.active = False
            logger.info("🧭 Ensemble Decision Making System stopped")
        except Exception as e:
            logger.error(f"Error stopping ensemble system: {e}")


def create_ensemble_decision_making_system(config: Dict[str, Any] = None) -> EnsembleDecisionMakingSystem:
    """Factory function to create an Ensemble Decision Making System."""
    return EnsembleDecisionMakingSystem(config)


# Global instance for easy access
ensemble_decision_making_system = create_ensemble_decision_making_system() 