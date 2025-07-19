#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 PROFIT OPTIMIZATION ENGINE - ADVANCED PORTFOLIO OPTIMIZATION
==============================================================

Advanced portfolio optimization engine using various mathematical methods
including gradient descent, genetic algorithms, and other optimization techniques.

This module provides:
- Portfolio weight optimization
- Risk-adjusted return maximization
- Multiple optimization algorithms
- Constraint handling
- Performance metrics calculation

All functions are pure and can be unit-tested in isolation.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods for profit calculation."""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"


class RiskMetric(Enum):
    """Risk metrics for portfolio optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    optimal_parameters: Dict[str, float]
    objective_value: float
    convergence: bool
    iterations: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioWeights:
    """Portfolio weight allocation."""
    weights: np.ndarray
    assets: List[str]
    total_weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitOptimizationEngine:
    """
    Main profit optimization engine.

    Provides methods for optimizing trading strategies, portfolio allocation,
    and risk management parameters.
    """
    
    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize the optimization engine."""
        self.risk_free_rate = risk_free_rate
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("🎯 Profit Optimization Engine initialized")
    
    def optimize_portfolio_weights(
        self,
        returns: np.ndarray,
        method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT,
        risk_metric: RiskMetric = RiskMetric.SHARPE_RATIO,
        constraints: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights for maximum risk-adjusted returns.

        Args:
            returns: Historical returns matrix (time x assets)
            method: Optimization method to use
            risk_metric: Risk metric to optimize
            constraints: Optimization constraints
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimization result with optimal weights
        """
        start_time = time.time()
        
        try:
            if method == OptimizationMethod.GRADIENT_DESCENT:
                result = self._gradient_descent_optimization(
                    returns, risk_metric, constraints, max_iterations, tolerance
                )
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(
                    returns, risk_metric, constraints, max_iterations
                )
            elif method == OptimizationMethod.SIMULATED_ANNEALING:
                result = self._simulated_annealing_optimization(
                    returns, risk_metric, constraints, max_iterations
                )
            elif method == OptimizationMethod.PARTICLE_SWARM:
                result = self._particle_swarm_optimization(
                    returns, risk_metric, constraints, max_iterations
                )
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Update execution time
            result.execution_time = time.time() - start_time
            
            # Store in history
            self.optimization_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return OptimizationResult(
                optimal_parameters={},
                objective_value=0.0,
                convergence=False,
                iterations=0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _gradient_descent_optimization(
        self,
        returns: np.ndarray,
        risk_metric: RiskMetric,
        constraints: Optional[Dict[str, Any]],
        max_iterations: int,
        tolerance: float,
    ) -> OptimizationResult:
        """Gradient descent optimization."""
        num_assets = returns.shape[1]
        
        # Initialize weights
        weights = np.ones(num_assets) / num_assets
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns.T, ddof=1)
        
        for iteration in range(max_iterations):
            # Calculate objective function and gradient
            if risk_metric == RiskMetric.SHARPE_RATIO:
                objective, gradient = self._sharpe_ratio_gradient(weights, returns, cov_matrix)
            else:
                objective, gradient = self._generic_risk_gradient(weights, returns, cov_matrix, risk_metric)
            
            # Update weights
            learning_rate = 0.1
            weights_new = weights + learning_rate * gradient
            
            # Apply constraints
            if constraints:
                weights_new = self._apply_constraints(weights_new, constraints)
            
            # Check convergence
            weight_change = np.linalg.norm(weights_new - weights)
            if weight_change < tolerance:
                break
            
            weights = weights_new
        
        return OptimizationResult(
            optimal_parameters={"weights": weights.tolist()},
            objective_value=objective,
            convergence=weight_change < tolerance,
            iterations=iteration + 1,
            execution_time=0.0,  # Will be calculated in real implementation
            metadata={
                "method": "gradient_descent",
                "risk_metric": risk_metric.value,
                "final_weight_change": float(weight_change),
            },
        )
    
    def _genetic_algorithm_optimization(
        self,
        returns: np.ndarray,
        risk_metric: RiskMetric,
        constraints: Optional[Dict[str, Any]],
        max_iterations: int,
    ) -> OptimizationResult:
        """Genetic algorithm optimization."""
        num_assets = returns.shape[1]
        population_size = 50
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            population.append(weights)
        
        best_weights = None
        best_objective = float('-inf')
        
        for generation in range(max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for weights in population:
                objective = self._calculate_objective(weights, returns, risk_metric)
                fitness_scores.append(objective)
                
                if objective > best_objective:
                    best_objective = objective
                    best_weights = weights.copy()
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            # Crossover and mutation
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    # Crossover
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    crossover_point = np.random.randint(1, num_assets)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                    
                    # Normalize
                    child1 = child1 / np.sum(child1)
                    child2 = child2 / np.sum(child2)
                    
                    # Mutation
                    if np.random.random() < mutation_rate:
                        child1 += np.random.normal(0, 0.01, num_assets)
                        child1 = np.maximum(child1, 0)
                        child1 = child1 / np.sum(child1)
                    
                    if np.random.random() < mutation_rate:
                        child2 += np.random.normal(0, 0.01, num_assets)
                        child2 = np.maximum(child2, 0)
                        child2 = child2 / np.sum(child2)
                    
                    new_population[i] = child1
                    new_population[i + 1] = child2
            
            population = new_population
        
        return OptimizationResult(
            optimal_parameters={"weights": best_weights.tolist() if best_weights is not None else []},
            objective_value=best_objective,
            convergence=True,
            iterations=max_iterations,
            execution_time=0.0,
            metadata={
                "method": "genetic_algorithm",
                "risk_metric": risk_metric.value,
                "population_size": population_size,
                "mutation_rate": mutation_rate,
            },
        )
    
    def _simulated_annealing_optimization(
        self,
        returns: np.ndarray,
        risk_metric: RiskMetric,
        constraints: Optional[Dict[str, Any]],
        max_iterations: int,
    ) -> OptimizationResult:
        """Simulated annealing optimization."""
        num_assets = returns.shape[1]
        
        # Initialize
        current_weights = np.ones(num_assets) / num_assets
        current_objective = self._calculate_objective(current_weights, returns, risk_metric)
        
        best_weights = current_weights.copy()
        best_objective = current_objective
        
        # Annealing parameters
        initial_temperature = 1.0
        final_temperature = 0.01
        cooling_rate = (final_temperature / initial_temperature) ** (1.0 / max_iterations)
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor_weights = current_weights + np.random.normal(0, 0.01, num_assets)
            neighbor_weights = np.maximum(neighbor_weights, 0)
            neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
            
            neighbor_objective = self._calculate_objective(neighbor_weights, returns, risk_metric)
            
            # Accept or reject
            delta_e = neighbor_objective - current_objective
            if delta_e > 0 or np.random.random() < np.exp(delta_e / temperature):
                current_weights = neighbor_weights
                current_objective = neighbor_objective
                
                if current_objective > best_objective:
                    best_weights = current_weights.copy()
                    best_objective = current_objective
            
            # Cool down
            temperature *= cooling_rate
        
        return OptimizationResult(
            optimal_parameters={"weights": best_weights.tolist()},
            objective_value=best_objective,
            convergence=True,
            iterations=max_iterations,
            execution_time=0.0,
            metadata={
                "method": "simulated_annealing",
                "risk_metric": risk_metric.value,
                "initial_temperature": initial_temperature,
                "final_temperature": final_temperature,
            },
        )
    
    def _particle_swarm_optimization(
        self,
        returns: np.ndarray,
        risk_metric: RiskMetric,
        constraints: Optional[Dict[str, Any]],
        max_iterations: int,
    ) -> OptimizationResult:
        """Particle swarm optimization."""
        num_assets = returns.shape[1]
        num_particles = 30
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_objective = []
        
        for _ in range(num_particles):
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            particles.append(weights)
            velocities.append(np.random.normal(0, 0.01, num_assets))
            
            objective = self._calculate_objective(weights, returns, risk_metric)
            personal_best.append(weights.copy())
            personal_best_objective.append(objective)
        
        # Global best
        global_best_idx = np.argmax(personal_best_objective)
        global_best = personal_best[global_best_idx].copy()
        global_best_objective = personal_best_objective[global_best_idx]
        
        # PSO parameters
        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        for iteration in range(max_iterations):
            for i in range(num_particles):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (personal_best[i] - particles[i]) +
                    c2 * r2 * (global_best - particles[i])
                )
                
                # Update position
                particles[i] += velocities[i]
                particles[i] = np.maximum(particles[i], 0)
                particles[i] = particles[i] / np.sum(particles[i])
                
                # Evaluate
                objective = self._calculate_objective(particles[i], returns, risk_metric)
                
                # Update personal best
                if objective > personal_best_objective[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_objective[i] = objective
                    
                    # Update global best
                    if objective > global_best_objective:
                        global_best = particles[i].copy()
                        global_best_objective = objective
        
        return OptimizationResult(
            optimal_parameters={"weights": global_best.tolist()},
            objective_value=global_best_objective,
            convergence=True,
            iterations=max_iterations,
            execution_time=0.0,
            metadata={
                "method": "particle_swarm",
                "risk_metric": risk_metric.value,
                "num_particles": num_particles,
                "inertia": w,
                "cognitive_param": c1,
                "social_param": c2,
            },
        )
    
    def _sharpe_ratio_gradient(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate Sharpe ratio and its gradient."""
        # Portfolio return and risk
        portfolio_return = np.mean(returns @ weights)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Gradient
        if portfolio_risk > 0:
            gradient = (
                (np.mean(returns, axis=0) - self.risk_free_rate) / portfolio_risk -
                (portfolio_return - self.risk_free_rate) * (cov_matrix @ weights) / (portfolio_risk ** 3)
            )
        else:
            gradient = np.zeros_like(weights)
        
        return sharpe_ratio, gradient
    
    def _generic_risk_gradient(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray, risk_metric: RiskMetric) -> Tuple[float, np.ndarray]:
        """Calculate generic risk metric and gradient."""
        if risk_metric == RiskMetric.SORTINO_RATIO:
            return self._sortino_ratio_gradient(weights, returns)
        elif risk_metric == RiskMetric.MAX_DRAWDOWN:
            return self._max_drawdown_gradient(weights, returns)
        else:
            # Default to negative variance (minimize risk)
            portfolio_risk = weights.T @ cov_matrix @ weights
            gradient = 2 * cov_matrix @ weights
            return -portfolio_risk, -gradient
    
    def _sortino_ratio_gradient(self, weights: np.ndarray, returns: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate Sortino ratio and its gradient."""
        portfolio_returns = returns @ weights
        portfolio_mean = np.mean(portfolio_returns)
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < portfolio_mean]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((downside_returns - portfolio_mean) ** 2))
        else:
            downside_deviation = 0.0
        
        # Sortino ratio
        if downside_deviation > 0:
            sortino_ratio = (portfolio_mean - self.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = float('inf')
        
        # Gradient (simplified)
        gradient = np.mean(returns, axis=0) - self.risk_free_rate
        
        return sortino_ratio, gradient
    
    def _max_drawdown_gradient(self, weights: np.ndarray, returns: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate max drawdown and its gradient."""
        portfolio_returns = returns @ weights
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Gradient (simplified)
        gradient = np.mean(returns, axis=0)
        
        return -max_drawdown, gradient
    
    def _calculate_objective(self, weights: np.ndarray, returns: np.ndarray, risk_metric: RiskMetric) -> float:
        """Calculate objective function value."""
        if risk_metric == RiskMetric.SHARPE_RATIO:
            portfolio_return = np.mean(returns @ weights)
            portfolio_risk = np.sqrt(weights.T @ np.cov(returns.T, ddof=1) @ weights)
            if portfolio_risk > 0:
                return (portfolio_return - self.risk_free_rate) / portfolio_risk
            else:
                return float('-inf')
        elif risk_metric == RiskMetric.SORTINO_RATIO:
            portfolio_returns = returns @ weights
            portfolio_mean = np.mean(portfolio_returns)
            downside_returns = portfolio_returns[portfolio_returns < portfolio_mean]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(np.mean((downside_returns - portfolio_mean) ** 2))
                if downside_deviation > 0:
                    return (portfolio_mean - self.risk_free_rate) / downside_deviation
            return float('-inf')
        else:
            # Default to negative variance
            return -weights.T @ np.cov(returns.T, ddof=1) @ weights
    
    def _apply_constraints(self, weights: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Apply optimization constraints."""
        # Ensure non-negative weights
        weights = np.maximum(weights, 0)
        
        # Apply weight bounds if specified
        if 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            weights = np.maximum(weights, min_weight)
        
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            weights = np.minimum(weights, max_weight)
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the optimization engine."""
        if not self.optimization_history:
            return {}
        
        successful_optimizations = [r for r in self.optimization_history if r.convergence]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "successful_optimizations": len(successful_optimizations),
            "success_rate": len(successful_optimizations) / len(self.optimization_history),
            "average_execution_time": np.mean([r.execution_time for r in self.optimization_history]),
            "average_iterations": np.mean([r.iterations for r in self.optimization_history]),
            "best_objective_value": max([r.objective_value for r in self.optimization_history]),
        }


def create_profit_optimization_engine(risk_free_rate: float = 0.02) -> ProfitOptimizationEngine:
    """Factory function to create a Profit Optimization Engine."""
    return ProfitOptimizationEngine(risk_free_rate)


# Global instance for easy access
profit_optimization_engine = create_profit_optimization_engine() 