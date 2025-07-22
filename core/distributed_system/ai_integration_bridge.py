#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Integration Bridge - Schwabot Distributed AI System
======================================================

Connects all AI models (KoboldCPP, Schwabot AI) with the distributed context
system and provides unified decision-making capabilities.

Features:
- Multi-AI model integration
- Context-aware decision making
- Real-time AI communication
- Decision consensus and voting
- Performance optimization
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class AIModelType(Enum):
    """Types of AI models in the system."""
    KOBOLDCPP = "koboldcpp"
    SCHWABOT_AI = "schwabot_ai"
    EXTERNAL_LLM = "external_llm"
    ENSEMBLE = "ensemble"

class DecisionType(Enum):
    """Types of AI decisions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ADJUST_POSITION = "adjust_position"
    EMERGENCY_STOP = "emergency_stop"
    NO_ACTION = "no_action"

@dataclass
class AIDecision:
    """AI decision with context and confidence."""
    model_type: AIModelType
    decision_type: DecisionType
    confidence: float
    reasoning: str
    context_data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsensusDecision:
    """Consensus decision from multiple AI models."""
    final_decision: DecisionType
    confidence: float
    model_votes: Dict[AIModelType, AIDecision]
    consensus_reasoning: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AIIntegrationBridge:
    """Bridge for integrating multiple AI models with context system."""
    
    def __init__(self, distributed_manager=None, context_ingestion=None):
        self.distributed_manager = distributed_manager
        self.context_ingestion = context_ingestion
        self.ai_models = {}
        self.decision_history = []
        self.consensus_history = []
        self.is_running = False
        
        # AI model configurations
        self.model_configs = {
            AIModelType.KOBOLDCPP: {
                "enabled": True,
                "weight": 0.4,
                "min_confidence": 0.6,
                "context_window": 1000
            },
            AIModelType.SCHWABOT_AI: {
                "enabled": True,
                "weight": 0.4,
                "min_confidence": 0.7,
                "context_window": 800
            },
            AIModelType.EXTERNAL_LLM: {
                "enabled": False,
                "weight": 0.2,
                "min_confidence": 0.8,
                "context_window": 500
            }
        }
        
        # Decision processing
        self.decision_queue = asyncio.Queue()
        self.processing_tasks = []
        
        logger.info("Initialized AIIntegrationBridge")
    
    async def start(self):
        """Start the AI integration bridge."""
        logger.info("Starting AIIntegrationBridge...")
        
        self.is_running = True
        
        # Initialize AI models
        await self._initialize_ai_models()
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._decision_processing_loop()),
            asyncio.create_task(self._ai_health_monitoring()),
            asyncio.create_task(self._performance_optimization_loop())
        ]
        
        logger.info("AIIntegrationBridge started successfully")
    
    async def stop(self):
        """Stop the AI integration bridge."""
        logger.info("Stopping AIIntegrationBridge...")
        
        self.is_running = False
        
        # Stop AI models
        for model_type, model in self.ai_models.items():
            if hasattr(model, 'stop'):
                await model.stop()
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("AIIntegrationBridge stopped")
    
    async def _initialize_ai_models(self):
        """Initialize all available AI models."""
        logger.info("Initializing AI models...")
        
        # Initialize KoboldCPP
        if self.model_configs[AIModelType.KOBOLDCPP]["enabled"]:
            try:
                kobold_model = await self._initialize_koboldcpp()
                if kobold_model:
                    self.ai_models[AIModelType.KOBOLDCPP] = kobold_model
                    logger.info("KoboldCPP model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize KoboldCPP: {e}")
        
        # Initialize Schwabot AI
        if self.model_configs[AIModelType.SCHWABOT_AI]["enabled"]:
            try:
                schwabot_model = await self._initialize_schwabot_ai()
                if schwabot_model:
                    self.ai_models[AIModelType.SCHWABOT_AI] = schwabot_model
                    logger.info("Schwabot AI model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Schwabot AI: {e}")
        
        # Initialize External LLM (if configured)
        if self.model_configs[AIModelType.EXTERNAL_LLM]["enabled"]:
            try:
                external_model = await self._initialize_external_llm()
                if external_model:
                    self.ai_models[AIModelType.EXTERNAL_LLM] = external_model
                    logger.info("External LLM model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize External LLM: {e}")
        
        logger.info(f"Initialized {len(self.ai_models)} AI models")
    
    async def _initialize_koboldcpp(self):
        """Initialize KoboldCPP model."""
        try:
            # Import KoboldCPP integration
            from core.koboldcpp_integration import KoboldCPPIntegration
            
            model = KoboldCPPIntegration()
            await model.initialize()
            
            return model
        except ImportError:
            logger.warning("KoboldCPP integration not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing KoboldCPP: {e}")
            return None
    
    async def _initialize_schwabot_ai(self):
        """Initialize Schwabot AI model."""
        try:
            # Import Schwabot AI integration
            from AOI_Base_Files_Schwabot.core.schwabot_ai import SchwabotAI
            
            model = SchwabotAI()
            await model.initialize()
            
            return model
        except ImportError:
            logger.warning("Schwabot AI integration not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing Schwabot AI: {e}")
            return None
    
    async def _initialize_external_llm(self):
        """Initialize External LLM model."""
        try:
            # This would connect to external LLM APIs
            # For now, return a placeholder
            return None
        except Exception as e:
            logger.error(f"Error initializing External LLM: {e}")
            return None
    
    async def request_decision(self, context_data: Dict[str, Any], 
                             symbols: List[str] = None) -> ConsensusDecision:
        """Request a decision from all AI models with consensus."""
        logger.info(f"Requesting AI decision for symbols: {symbols}")
        
        # Prepare context for AI models
        ai_context = await self._prepare_ai_context(context_data, symbols)
        
        # Get decisions from all models
        model_decisions = {}
        for model_type, model in self.ai_models.items():
            try:
                decision = await self._get_model_decision(model, model_type, ai_context)
                if decision and decision.confidence >= self.model_configs[model_type]["min_confidence"]:
                    model_decisions[model_type] = decision
                    logger.debug(f"Got decision from {model_type.value}: {decision.decision_type.value}")
            except Exception as e:
                logger.error(f"Error getting decision from {model_type.value}: {e}")
        
        if not model_decisions:
            logger.warning("No valid decisions from AI models")
            return self._create_default_decision()
        
        # Create consensus decision
        consensus = await self._create_consensus_decision(model_decisions)
        
        # Store in history
        self.consensus_history.append(consensus)
        self.decision_history.extend(model_decisions.values())
        
        # Keep history manageable
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]
        if len(self.decision_history) > 5000:
            self.decision_history = self.decision_history[-5000:]
        
        # Ingest decision into context system
        if self.context_ingestion:
            await self.context_ingestion.ingest_ai_decision({
                "consensus_decision": consensus.final_decision.value,
                "confidence": consensus.confidence,
                "model_votes": len(model_decisions),
                "reasoning": consensus.consensus_reasoning,
                "symbols": symbols
            }, "ai_integration_bridge")
        
        logger.info(f"Consensus decision: {consensus.final_decision.value} (confidence: {consensus.confidence:.2f})")
        return consensus
    
    async def _prepare_ai_context(self, context_data: Dict[str, Any], 
                                symbols: List[str] = None) -> Dict[str, Any]:
        """Prepare context data for AI models."""
        ai_context = {
            "trading_data": [],
            "tensor_math": [],
            "system_health": [],
            "recent_decisions": [],
            "symbols": symbols or []
        }
        
        # Get context from ingestion system
        if self.context_ingestion:
            # Get recent context data
            recent_context = self.context_ingestion.get_context_for_ai(limit=200)
            
            for context_item in recent_context:
                if context_item["data_type"] == "trading_data":
                    ai_context["trading_data"].append(context_item)
                elif context_item["data_type"] == "tensor_math":
                    ai_context["tensor_math"].append(context_item)
                elif context_item["data_type"] == "system_health":
                    ai_context["system_health"].append(context_item)
                elif context_item["data_type"] == "ai_decision":
                    ai_context["recent_decisions"].append(context_item)
        
        # Add current context data
        ai_context["current_context"] = context_data
        
        # Filter by symbols if specified
        if symbols:
            ai_context["trading_data"] = [
                item for item in ai_context["trading_data"]
                if item.get("data", {}).get("symbol") in symbols
            ]
        
        return ai_context
    
    async def _get_model_decision(self, model, model_type: AIModelType, 
                                context: Dict[str, Any]) -> Optional[AIDecision]:
        """Get decision from a specific AI model."""
        try:
            # Prepare model-specific context
            model_context = self._prepare_model_context(model_type, context)
            
            # Get decision from model
            if hasattr(model, 'get_decision'):
                result = await model.get_decision(model_context)
            elif hasattr(model, 'analyze'):
                result = await model.analyze(model_context)
            else:
                logger.warning(f"Model {model_type.value} has no decision method")
                return None
            
            # Parse result into AIDecision
            decision = self._parse_model_result(result, model_type, context)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting decision from {model_type.value}: {e}")
            return None
    
    def _prepare_model_context(self, model_type: AIModelType, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context specific to a model type."""
        config = self.model_configs[model_type]
        context_window = config["context_window"]
        
        # Limit context based on model's window
        model_context = {
            "trading_data": context["trading_data"][-context_window:],
            "tensor_math": context["tensor_math"][-context_window:],
            "system_health": context["system_health"][-50:],  # Less system health data
            "recent_decisions": context["recent_decisions"][-100:],
            "current_context": context["current_context"],
            "symbols": context["symbols"]
        }
        
        return model_context
    
    def _parse_model_result(self, result: Any, model_type: AIModelType, 
                          context: Dict[str, Any]) -> AIDecision:
        """Parse model result into AIDecision."""
        try:
            # Handle different result formats
            if isinstance(result, dict):
                decision_type = DecisionType(result.get("decision", "no_action"))
                confidence = float(result.get("confidence", 0.5))
                reasoning = str(result.get("reasoning", "No reasoning provided"))
            elif isinstance(result, str):
                # Simple string result
                decision_type = self._parse_decision_string(result)
                confidence = 0.7  # Default confidence
                reasoning = result
            else:
                # Fallback
                decision_type = DecisionType.NO_ACTION
                confidence = 0.5
                reasoning = "Unable to parse model result"
            
            return AIDecision(
                model_type=model_type,
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                context_data=context,
                timestamp=time.time(),
                metadata={
                    "model_type": model_type.value,
                    "parsed_from": type(result).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing model result: {e}")
            return AIDecision(
                model_type=model_type,
                decision_type=DecisionType.NO_ACTION,
                confidence=0.0,
                reasoning=f"Error parsing result: {e}",
                context_data=context,
                timestamp=time.time()
            )
    
    def _parse_decision_string(self, decision_str: str) -> DecisionType:
        """Parse decision string into DecisionType."""
        decision_str = decision_str.lower().strip()
        
        if "buy" in decision_str:
            return DecisionType.BUY
        elif "sell" in decision_str:
            return DecisionType.SELL
        elif "hold" in decision_str or "wait" in decision_str:
            return DecisionType.HOLD
        elif "adjust" in decision_str or "modify" in decision_str:
            return DecisionType.ADJUST_POSITION
        elif "stop" in decision_str or "emergency" in decision_str:
            return DecisionType.EMERGENCY_STOP
        else:
            return DecisionType.NO_ACTION
    
    async def _create_consensus_decision(self, model_decisions: Dict[AIModelType, AIDecision]) -> ConsensusDecision:
        """Create consensus decision from multiple model decisions."""
        if len(model_decisions) == 1:
            # Single decision
            decision = list(model_decisions.values())[0]
            return ConsensusDecision(
                final_decision=decision.decision_type,
                confidence=decision.confidence,
                model_votes=model_decisions,
                consensus_reasoning=decision.reasoning,
                timestamp=time.time()
            )
        
        # Multiple decisions - use weighted voting
        decision_votes = {}
        total_weight = 0.0
        
        for model_type, decision in model_decisions.items():
            weight = self.model_configs[model_type]["weight"]
            total_weight += weight
            
            if decision.decision_type not in decision_votes:
                decision_votes[decision.decision_type] = {
                    "weight": 0.0,
                    "confidence_sum": 0.0,
                    "reasonings": []
                }
            
            decision_votes[decision.decision_type]["weight"] += weight
            decision_votes[decision.decision_type]["confidence_sum"] += decision.confidence * weight
            decision_votes[decision.decision_type]["reasonings"].append(decision.reasoning)
        
        # Find the decision with highest weight
        best_decision = max(decision_votes.items(), key=lambda x: x[1]["weight"])
        final_decision_type = best_decision[0]
        
        # Calculate weighted confidence
        weighted_confidence = best_decision[1]["confidence_sum"] / best_decision[1]["weight"]
        
        # Create consensus reasoning
        consensus_reasoning = self._create_consensus_reasoning(decision_votes, final_decision_type)
        
        return ConsensusDecision(
            final_decision=final_decision_type,
            confidence=weighted_confidence,
            model_votes=model_decisions,
            consensus_reasoning=consensus_reasoning,
            timestamp=time.time(),
            metadata={
                "total_models": len(model_decisions),
                "decision_votes": {k.value: v for k, v in decision_votes.items()}
            }
        )
    
    def _create_consensus_reasoning(self, decision_votes: Dict, 
                                  final_decision: DecisionType) -> str:
        """Create consensus reasoning from multiple model reasonings."""
        if final_decision not in decision_votes:
            return "No consensus reached"
        
        reasonings = decision_votes[final_decision]["reasonings"]
        
        if len(reasonings) == 1:
            return reasonings[0]
        
        # Combine reasonings
        combined = f"Consensus reached with {len(reasonings)} models agreeing on {final_decision.value}. "
        combined += "Key reasoning: " + reasonings[0]  # Use first reasoning as primary
        
        if len(reasonings) > 1:
            combined += f" Additional support: {reasonings[1]}"
        
        return combined
    
    def _create_default_decision(self) -> ConsensusDecision:
        """Create a default decision when no models respond."""
        return ConsensusDecision(
            final_decision=DecisionType.NO_ACTION,
            confidence=0.0,
            model_votes={},
            consensus_reasoning="No AI models available or all models failed",
            timestamp=time.time()
        )
    
    async def _decision_processing_loop(self):
        """Process decision requests in background."""
        while self.is_running:
            try:
                # Process any pending decision requests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in decision processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _ai_health_monitoring(self):
        """Monitor health of AI models."""
        while self.is_running:
            try:
                for model_type, model in self.ai_models.items():
                    if hasattr(model, 'get_health'):
                        health = await model.get_health()
                        if health.get("status") != "healthy":
                            logger.warning(f"AI model {model_type.value} health issue: {health}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in AI health monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _performance_optimization_loop(self):
        """Optimize AI model performance."""
        while self.is_running:
            try:
                # Adjust model weights based on performance
                await self._adjust_model_weights()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance optimization: {e}")
                await asyncio.sleep(300)
    
    async def _adjust_model_weights(self):
        """Adjust model weights based on recent performance."""
        if len(self.consensus_history) < 10:
            return
        
        # Analyze recent decisions
        recent_decisions = self.consensus_history[-50:]
        
        # Calculate performance metrics for each model
        model_performance = {}
        for model_type in self.ai_models.keys():
            model_performance[model_type] = {
                "accuracy": 0.0,
                "confidence": 0.0,
                "participation": 0.0
            }
        
        # This would implement actual performance analysis
        # For now, just log that optimization is happening
        logger.debug("Performing AI model weight optimization")
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get status of all AI models."""
        status = {
            "total_models": len(self.ai_models),
            "active_models": 0,
            "model_status": {},
            "recent_decisions": len(self.decision_history),
            "consensus_decisions": len(self.consensus_history)
        }
        
        for model_type, model in self.ai_models.items():
            model_status = {
                "enabled": True,
                "weight": self.model_configs[model_type]["weight"],
                "min_confidence": self.model_configs[model_type]["min_confidence"]
            }
            
            if hasattr(model, 'get_health'):
                try:
                    health = asyncio.run(model.get_health())
                    model_status["health"] = health
                    if health.get("status") == "healthy":
                        status["active_models"] += 1
                except:
                    model_status["health"] = {"status": "unknown"}
            
            status["model_status"][model_type.value] = model_status
        
        return status
    
    def get_decision_history(self, limit: int = 100) -> List[ConsensusDecision]:
        """Get recent decision history."""
        return self.consensus_history[-limit:]

# Global instance
_ai_bridge: Optional[AIIntegrationBridge] = None

def get_ai_bridge() -> AIIntegrationBridge:
    """Get the global AI bridge instance."""
    global _ai_bridge
    if _ai_bridge is None:
        from .distributed_node_manager import get_distributed_manager
        from .real_time_context_ingestion import get_context_ingestion
        
        distributed_manager = get_distributed_manager()
        context_ingestion = get_context_ingestion()
        _ai_bridge = AIIntegrationBridge(distributed_manager, context_ingestion)
    return _ai_bridge

async def start_ai_integration():
    """Start the AI integration system."""
    bridge = get_ai_bridge()
    await bridge.start()
    return bridge

if __name__ == "__main__":
    # Test the AI integration bridge
    async def test():
        bridge = get_ai_bridge()
        await bridge.start()
        
        # Test with sample context
        context_data = {
            "symbol": "BTC/USD",
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": time.time()
        }
        
        decision = await bridge.request_decision(context_data, ["BTC/USD"])
        print(f"AI Decision: {decision.final_decision.value} (confidence: {decision.confidence:.2f})")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await bridge.stop()
    
    asyncio.run(test()) 