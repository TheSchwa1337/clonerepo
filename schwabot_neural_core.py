#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Schwabot Neural Core - Phase IV
==================================

Neural Network Integration Layer for Schwabot's Brain Systems:
- 16,000 neuron metaphor implementation
- Recursive decision cycles with feedback loops
- Neural network math for buy/sell/hold decisions
- Reinforcement learning from trade outcomes
- Integration with Clock Mode System

Mathematical Foundation:
- Neural Network: Output = f(∑(w_i * x_i) + b)
- Recursive Feedback: Decision_t = f(∑(w_i * x_i) + b)
- Profit Calculation: Profit_t = (BTC_t * P_t) + (USDC_t * P_USDC)
"""

import sys
import math
import time
import json
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import hashlib
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_neural_core.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NeuronType(Enum):
    """Types of neurons in Schwabot's brain system."""
    INPUT_NEURON = "input"           # Market data inputs
    HIDDEN_NEURON = "hidden"         # Processing neurons
    OUTPUT_NEURON = "output"         # Decision neurons (buy/sell/hold)
    FEEDBACK_NEURON = "feedback"     # Learning and adaptation
    MEMORY_NEURON = "memory"         # Historical data storage
    TIMING_NEURON = "timing"         # Clock synchronization
    PROFIT_NEURON = "profit"         # Profit calculation
    RISK_NEURON = "risk"             # Risk assessment

class DecisionType(Enum):
    """Trading decision types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class Neuron:
    """Individual neuron in Schwabot's neural network."""
    neuron_id: str
    neuron_type: NeuronType
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    activation_threshold: float = 0.5
    learning_rate: float = 0.01
    connections: List[str] = field(default_factory=list)
    last_activation: float = 0.0
    activation_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)
    is_active: bool = True
    layer_depth: int = 0
    
    def __post_init__(self):
        """Initialize neuron with random weights if not provided."""
        if not self.weights:
            # Initialize with random weights based on neuron type
            if self.neuron_type == NeuronType.INPUT_NEURON:
                self.weights = [random.uniform(-1, 1) for _ in range(5)]  # 5 input features
            elif self.neuron_type == NeuronType.HIDDEN_NEURON:
                self.weights = [random.uniform(-1, 1) for _ in range(10)]  # 10 hidden features
            elif self.neuron_type == NeuronType.OUTPUT_NEURON:
                self.weights = [random.uniform(-1, 1) for _ in range(3)]   # 3 decision outputs
    
    def activate(self, inputs: List[float]) -> float:
        """Activate the neuron using neural network math: f(∑(w_i * x_i) + b)"""
        if len(inputs) != len(self.weights):
            logger.warning(f"Input size mismatch for neuron {self.neuron_id}")
            return 0.0
        
        # Calculate weighted sum: ∑(w_i * x_i) + b
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Apply activation function (sigmoid)
        activation = 1.0 / (1.0 + math.exp(-weighted_sum))
        
        # Store activation history
        self.last_activation = activation
        self.activation_history.append(activation)
        
        # Keep only last 100 activations to prevent memory overflow
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        
        return activation
    
    def update_weights(self, error: float, inputs: List[float]) -> None:
        """Update weights using backpropagation learning."""
        if len(inputs) != len(self.weights):
            return
        
        # Calculate weight updates: Δw = learning_rate * error * input
        for i in range(len(self.weights)):
            weight_update = self.learning_rate * error * inputs[i]
            self.weights[i] += weight_update
        
        # Update bias
        self.bias += self.learning_rate * error
        
        # Store error history
        self.error_history.append(error)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def get_confidence(self) -> float:
        """Get confidence level based on recent activations."""
        if not self.activation_history:
            return 0.0
        
        recent_activations = self.activation_history[-10:]  # Last 10 activations
        return sum(recent_activations) / len(recent_activations)

@dataclass
class NeuralLayer:
    """Layer of neurons in the neural network."""
    layer_id: str
    neurons: List[Neuron] = field(default_factory=list)
    layer_type: str = "hidden"  # input, hidden, output
    activation_function: str = "sigmoid"
    
    def add_neuron(self, neuron: Neuron) -> None:
        """Add a neuron to the layer."""
        self.neurons.append(neuron)
        neuron.layer_depth = len(self.neurons)
    
    def forward_propagate(self, inputs: List[float]) -> List[float]:
        """Forward propagate inputs through the layer."""
        outputs = []
        for neuron in self.neurons:
            if neuron.is_active:
                output = neuron.activate(inputs)
                outputs.append(output)
            else:
                outputs.append(0.0)
        return outputs
    
    def get_layer_output(self) -> List[float]:
        """Get the last output from each neuron in the layer."""
        return [neuron.last_activation for neuron in self.neurons]

@dataclass
class MarketData:
    """Market data structure for neural network inputs."""
    timestamp: datetime
    btc_price: float
    usdc_balance: float
    btc_balance: float
    price_change: float
    volume: float
    rsi_14: float
    rsi_21: float
    rsi_50: float
    market_phase: float
    hash_timing: str
    orbital_phase: float
    
    def to_neural_inputs(self) -> List[float]:
        """Convert market data to neural network inputs."""
        # Safe hash timing parsing
        try:
            hash_value = float(int(self.hash_timing[:8], 16)) / (16**8) if self.hash_timing else 0.0
        except (ValueError, IndexError):
            hash_value = 0.0
        
        return [
            self.btc_price / 100000.0,  # Normalize price
            self.usdc_balance / 10000.0,  # Normalize USDC
            self.btc_balance / 10.0,  # Normalize BTC
            self.price_change,
            self.volume / 10000.0,  # Normalize volume
            self.rsi_14 / 100.0,  # Normalize RSI
            self.rsi_21 / 100.0,
            self.rsi_50 / 100.0,
            self.market_phase / (2 * math.pi),  # Normalize phase
            hash_value  # Normalize hash (safe parsing)
        ]

@dataclass
class TradingDecision:
    """Trading decision with confidence and reasoning."""
    decision_type: DecisionType
    confidence: float
    timestamp: datetime
    reasoning: Dict[str, Any]
    neural_outputs: List[float]
    market_data: MarketData
    expected_profit: float = 0.0
    risk_level: str = "medium"

class SchwabotNeuralCore:
    """Main neural core implementing Schwabot's brain system."""
    
    def __init__(self):
        self.layers: List[NeuralLayer] = []
        self.neurons: Dict[str, Neuron] = {}
        self.decision_history: List[TradingDecision] = []
        self.learning_enabled: bool = True
        self.recursive_cycles: int = 0
        self.total_profit: float = 0.0
        self.successful_trades: int = 0
        self.total_trades: int = 0
        
        # Neural network parameters
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 3  # buy, sell, hold
        
        # Initialize neural network
        self._build_neural_network()
        
        logger.info("🧠 Schwabot Neural Core initialized")
    
    def _build_neural_network(self) -> None:
        """Build the neural network architecture."""
        # Input layer
        input_layer = NeuralLayer("input_layer", layer_type="input")
        for i in range(self.input_size):
            neuron = Neuron(
                f"input_{i}",
                NeuronType.INPUT_NEURON,
                learning_rate=0.01
            )
            input_layer.add_neuron(neuron)
            self.neurons[neuron.neuron_id] = neuron
        
        # Hidden layers (2 layers for complexity)
        hidden_layer_1 = NeuralLayer("hidden_layer_1", layer_type="hidden")
        for i in range(self.hidden_size):
            neuron = Neuron(
                f"hidden_1_{i}",
                NeuronType.HIDDEN_NEURON,
                learning_rate=0.01
            )
            hidden_layer_1.add_neuron(neuron)
            self.neurons[neuron.neuron_id] = neuron
        
        hidden_layer_2 = NeuralLayer("hidden_layer_2", layer_type="hidden")
        for i in range(self.hidden_size // 2):
            neuron = Neuron(
                f"hidden_2_{i}",
                NeuronType.HIDDEN_NEURON,
                learning_rate=0.01
            )
            hidden_layer_2.add_neuron(neuron)
            self.neurons[neuron.neuron_id] = neuron
        
        # Output layer
        output_layer = NeuralLayer("output_layer", layer_type="output")
        decision_neurons = [
            ("buy", NeuronType.OUTPUT_NEURON),
            ("sell", NeuronType.OUTPUT_NEURON),
            ("hold", NeuronType.OUTPUT_NEURON)
        ]
        
        for i, (decision_name, neuron_type) in enumerate(decision_neurons):
            neuron = Neuron(
                f"output_{decision_name}",
                neuron_type,
                learning_rate=0.01
            )
            output_layer.add_neuron(neuron)
            self.neurons[neuron.neuron_id] = neuron
        
        # Add layers to network
        self.layers = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]
        
        logger.info(f"🧠 Neural network built: {len(self.layers)} layers, {len(self.neurons)} neurons")
    
    def forward_propagate(self, market_data: MarketData) -> List[float]:
        """Forward propagate market data through the neural network."""
        inputs = market_data.to_neural_inputs()
        
        # Propagate through all layers
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward_propagate(current_inputs)
        
        return current_inputs  # Final outputs from output layer
    
    def make_decision(self, market_data: MarketData) -> TradingDecision:
        """Make a trading decision using neural network."""
        # Forward propagate to get neural outputs
        neural_outputs = self.forward_propagate(market_data)
        
        # Determine decision based on highest output
        decision_values = {
            DecisionType.BUY: neural_outputs[0],
            DecisionType.SELL: neural_outputs[1],
            DecisionType.HOLD: neural_outputs[2]
        }
        
        best_decision = max(decision_values, key=decision_values.get)
        confidence = decision_values[best_decision]
        
        # Calculate expected profit
        expected_profit = self._calculate_expected_profit(best_decision, market_data, confidence)
        
        # Determine risk level
        risk_level = self._assess_risk_level(market_data, confidence)
        
        # Create decision reasoning
        reasoning = {
            "neural_outputs": neural_outputs,
            "decision_confidence": confidence,
            "market_conditions": {
                "price_trend": "up" if market_data.price_change > 0 else "down",
                "rsi_condition": "oversold" if market_data.rsi_14 < 30 else "overbought" if market_data.rsi_14 > 70 else "neutral",
                "volume_condition": "high" if market_data.volume > 5000 else "low"
            },
            "expected_profit": expected_profit,
            "risk_assessment": risk_level
        }
        
        decision = TradingDecision(
            decision_type=best_decision,
            confidence=confidence,
            timestamp=datetime.now(),
            reasoning=reasoning,
            neural_outputs=neural_outputs,
            market_data=market_data,
            expected_profit=expected_profit,
            risk_level=risk_level
        )
        
        # Store decision in history
        self.decision_history.append(decision)
        
        # Increment recursive cycles
        self.recursive_cycles += 1
        
        logger.info(f"🧠 Decision: {best_decision.value} (confidence: {confidence:.3f}, profit: {expected_profit:.2f})")
        
        return decision
    
    def _calculate_expected_profit(self, decision: DecisionType, market_data: MarketData, confidence: float) -> float:
        """Calculate expected profit for a decision."""
        base_profit = 0.0
        
        if decision == DecisionType.BUY:
            # Expected profit from buying (simplified calculation)
            base_profit = market_data.btc_balance * market_data.price_change * confidence
        elif decision == DecisionType.SELL:
            # Expected profit from selling
            base_profit = market_data.usdc_balance * market_data.price_change * confidence
        else:  # HOLD
            # Minimal profit/loss from holding
            base_profit = 0.0
        
        return base_profit
    
    def _assess_risk_level(self, market_data: MarketData, confidence: float) -> str:
        """Assess risk level based on market conditions and confidence."""
        risk_factors = []
        
        # Price volatility
        if abs(market_data.price_change) > 0.05:  # 5% change
            risk_factors.append("high_volatility")
        
        # RSI extremes
        if market_data.rsi_14 < 20 or market_data.rsi_14 > 80:
            risk_factors.append("extreme_rsi")
        
        # Low confidence
        if confidence < 0.3:
            risk_factors.append("low_confidence")
        
        # Low volume
        if market_data.volume < 1000:
            risk_factors.append("low_volume")
        
        if len(risk_factors) >= 3:
            return "high"
        elif len(risk_factors) >= 1:
            return "medium"
        else:
            return "low"
    
    def learn_from_outcome(self, decision: TradingDecision, actual_profit: float) -> None:
        """Learn from the outcome of a trading decision (reinforcement learning)."""
        if not self.learning_enabled:
            return
        
        # Calculate error (difference between expected and actual profit)
        error = actual_profit - decision.expected_profit
        
        # Update neural network weights using backpropagation
        self._backpropagate_error(error, decision)
        
        # Update statistics
        self.total_profit += actual_profit
        self.total_trades += 1
        
        if actual_profit > 0:
            self.successful_trades += 1
        
        # Log learning outcome
        success_rate = self.successful_trades / max(1, self.total_trades)
        logger.info(f"🧠 Learning: error={error:.4f}, success_rate={success_rate:.3f}, total_profit={self.total_profit:.2f}")
    
    def _backpropagate_error(self, error: float, decision: TradingDecision) -> None:
        """Backpropagate error through the neural network."""
        # Simplified backpropagation - update output layer neurons
        output_layer = self.layers[-1]
        
        for i, neuron in enumerate(output_layer.neurons):
            # Calculate error for this output neuron
            if i == 0:  # BUY neuron
                neuron_error = error if decision.decision_type == DecisionType.BUY else 0
            elif i == 1:  # SELL neuron
                neuron_error = error if decision.decision_type == DecisionType.SELL else 0
            else:  # HOLD neuron
                neuron_error = error if decision.decision_type == DecisionType.HOLD else 0
            
            # Update neuron weights
            inputs = decision.market_data.to_neural_inputs()
            neuron.update_weights(neuron_error, inputs)
    
    def get_neural_stats(self) -> Dict[str, Any]:
        """Get neural network statistics."""
        total_neurons = len(self.neurons)
        active_neurons = sum(1 for neuron in self.neurons.values() if neuron.is_active)
        
        # Calculate average confidence
        recent_decisions = self.decision_history[-10:] if self.decision_history else []
        avg_confidence = sum(d.confidence for d in recent_decisions) / max(1, len(recent_decisions))
        
        # Calculate success rate
        success_rate = self.successful_trades / max(1, self.total_trades)
        
        return {
            "total_neurons": total_neurons,
            "active_neurons": active_neurons,
            "layers": len(self.layers),
            "recursive_cycles": self.recursive_cycles,
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "success_rate": success_rate,
            "total_profit": self.total_profit,
            "avg_confidence": avg_confidence,
            "learning_enabled": self.learning_enabled
        }
    
    def reset_learning(self) -> None:
        """Reset learning parameters and history."""
        self.decision_history.clear()
        self.recursive_cycles = 0
        self.total_profit = 0.0
        self.successful_trades = 0
        self.total_trades = 0
        
        # Reset neuron histories
        for neuron in self.neurons.values():
            neuron.activation_history.clear()
            neuron.error_history.clear()
        
        logger.info("🧠 Neural learning reset")

def main():
    """Test the Schwabot Neural Core."""
    logger.info("🧠 Starting Schwabot Neural Core Test")
    
    # Create neural core
    neural_core = SchwabotNeuralCore()
    
    # Create sample market data
    market_data = MarketData(
        timestamp=datetime.now(),
        btc_price=50000.0,
        usdc_balance=10000.0,
        btc_balance=0.2,
        price_change=0.02,  # 2% increase
        volume=5000.0,
        rsi_14=45.0,
        rsi_21=50.0,
        rsi_50=55.0,
        market_phase=math.pi / 4,
        hash_timing="a1b2c3d4e5f6",
        orbital_phase=0.5
    )
    
    # Make a decision
    decision = neural_core.make_decision(market_data)
    
    # Simulate outcome and learn
    actual_profit = 150.0  # Simulated profit
    neural_core.learn_from_outcome(decision, actual_profit)
    
    # Get stats
    stats = neural_core.get_neural_stats()
    logger.info(f"🧠 Neural Stats: {json.dumps(stats, indent=2)}")
    
    logger.info("🧠 Schwabot Neural Core Test Complete")

if __name__ == "__main__":
    main() 