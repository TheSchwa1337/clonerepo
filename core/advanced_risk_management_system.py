#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ ADVANCED RISK MANAGEMENT SYSTEM - SYSTEM RELIABILITY ENGINE
==============================================================

Advanced risk management system for the Schwabot trading engine.

Features:
- Dynamic risk assessment and monitoring
- Position sizing optimization
- Risk-adjusted returns calculation
- Portfolio stress testing
- Real-time risk monitoring
- VaR and CVaR calculations
- Risk factor analysis
- Dynamic risk limits

Mathematical Foundation:
- Value at Risk (VaR): P(L > VaR) = α
- Conditional VaR (CVaR): E[L|L > VaR]
- Risk-adjusted returns: Sharpe ratio, Sortino ratio
- Position sizing: Kelly criterion, optimal f
- Portfolio optimization: Modern Portfolio Theory
"""

import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

# Import existing Schwabot components
try:
    from advanced_tensor_algebra import AdvancedTensorAlgebra
    from entropy_math import EntropyMathSystem
    from distributed_mathematical_processor import DistributedMathematicalProcessor
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"🛡️ Advanced Risk Management using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 Advanced Risk Management using CPU fallback: {_backend}")

__all__ = [
    "AdvancedRiskManagementSystem",
    "RiskMetrics",
    "PositionSizer",
    "PortfolioStressTester",
    "RiskMonitor",
    "RiskFactor",
    "RiskLimit",
    "RiskAssessment",
]


class RiskLevel(Enum):
    """Risk levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(Enum):
    """Risk factors for analysis."""
    MARKET = "market"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    OPERATIONAL = "operational"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% VaR
    max_drawdown: float = 0.15  # 15% max drawdown
    max_leverage: float = 2.0  # 2x leverage
    max_concentration: float = 0.25  # 25% single asset
    min_liquidity: float = 1000000  # $1M minimum liquidity
    max_correlation: float = 0.7  # 70% correlation limit


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    max_drawdown: float  # Maximum drawdown
    volatility: float  # Portfolio volatility
    beta: float  # Market beta
    correlation: float  # Market correlation
    concentration: float  # Portfolio concentration
    leverage: float  # Current leverage
    liquidity_score: float  # Liquidity assessment
    risk_score: float  # Overall risk score (0-1)


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    timestamp: float
    risk_level: RiskLevel
    risk_metrics: RiskMetrics
    risk_factors: Dict[RiskFactor, float]
    recommendations: List[str]
    alerts: List[str]
    position_adjustments: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionSizer:
    """
    Advanced position sizing using multiple methodologies.
    
    Mathematical Foundation:
    - Kelly Criterion: f* = (bp - q) / b
    - Optimal f: f* = (W * T) / (L * T)
    - Risk Parity: Equal risk contribution
    - Volatility Targeting: Position size ∝ 1/volatility
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize position sizer."""
        self.risk_free_rate = risk_free_rate
        self.position_history = []
        self.sizing_performance = {}

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning amount
            avg_loss: Average losing amount
            
        Returns:
            Optimal position size as fraction of capital
        """
        try:
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f* = (bp - q) / b
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate  # Win probability
            q = 1 - p  # Loss probability
            
            kelly_fraction = (b * p - q) / b
            
            # Constrain to reasonable bounds
            return max(0.0, min(0.25, kelly_fraction))
            
        except Exception as e:
            logger.error(f"Kelly criterion calculation failed: {e}")
            return 0.0

    def optimal_f(self, returns: List[float], max_risk: float = 0.02) -> float:
        """
        Calculate optimal f position size.
        
        Args:
            returns: Historical returns
            max_risk: Maximum risk per trade
            
        Returns:
            Optimal position size
        """
        try:
            if not returns:
                return 0.0
            
            returns_array = xp.array(returns)
            
            # Calculate TWR (Terminal Wealth Relative)
            def calculate_twr(f: float) -> float:
                twr = 1.0
                for ret in returns_array:
                    twr *= (1 + f * ret)
                return twr
            
            # Find optimal f using optimization
            best_f = 0.0
            best_twr = 1.0
            
            for f in xp.linspace(0.01, 0.5, 50):
                twr = calculate_twr(f)
                if twr > best_twr:
                    best_twr = twr
                    best_f = f
            
            # Apply risk constraint
            return min(best_f, max_risk)
            
        except Exception as e:
            logger.error(f"Optimal f calculation failed: {e}")
            return 0.0

    def risk_parity_sizing(self, volatilities: Dict[str, float], target_risk: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk parity position sizes.
        
        Args:
            volatilities: Asset volatilities
            target_risk: Target risk per asset
            
        Returns:
            Position sizes for each asset
        """
        try:
            if not volatilities:
                return {}
            
            position_sizes = {}
            total_risk = 0.0
            
            for asset, vol in volatilities.items():
                if vol > 0:
                    # Equal risk contribution
                    position_size = target_risk / vol
                    position_sizes[asset] = position_size
                    total_risk += target_risk
            
            # Normalize to sum to 1
            if total_risk > 0:
                for asset in position_sizes:
                    position_sizes[asset] /= total_risk
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"Risk parity sizing failed: {e}")
            return {}

    def volatility_targeting(self, current_vol: float, target_vol: float, current_size: float) -> float:
        """
        Adjust position size based on volatility targeting.
        
        Args:
            current_vol: Current volatility
            target_vol: Target volatility
            current_size: Current position size
            
        Returns:
            Adjusted position size
        """
        try:
            if current_vol <= 0:
                return current_size
            
            # Volatility targeting: new_size = current_size * (target_vol / current_vol)
            adjusted_size = current_size * (target_vol / current_vol)
            
            # Apply reasonable bounds
            return max(0.0, min(1.0, adjusted_size))
            
        except Exception as e:
            logger.error(f"Volatility targeting failed: {e}")
            return current_size


class PortfolioStressTester:
    """
    Portfolio stress testing and scenario analysis.
    
    Mathematical Foundation:
    - Historical stress testing
    - Monte Carlo simulation
    - Scenario analysis
    - Extreme value theory
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize stress tester."""
        self.confidence_level = confidence_level
        self.scenarios = {}
        self.stress_results = []

    def historical_stress_test(self, portfolio_returns: List[float], stress_periods: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Perform historical stress testing.
        
        Args:
            portfolio_returns: Historical portfolio returns
            stress_periods: List of (start, end) indices for stress periods
            
        Returns:
            Stress test results
        """
        try:
            results = {}
            
            for i, (start, end) in enumerate(stress_periods):
                if start < len(portfolio_returns) and end <= len(portfolio_returns):
                    stress_returns = portfolio_returns[start:end]
                    
                    if stress_returns:
                        # Calculate stress metrics
                        cumulative_return = np.prod([1 + r for r in stress_returns]) - 1
                        max_drawdown = self._calculate_max_drawdown(stress_returns)
                        volatility = np.std(stress_returns) * np.sqrt(252)  # Annualized
                        
                        results[f"stress_period_{i}"] = {
                            "cumulative_return": cumulative_return,
                            "max_drawdown": max_drawdown,
                            "volatility": volatility,
                            "var_95": np.percentile(stress_returns, 5)
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Historical stress test failed: {e}")
            return {}

    def monte_carlo_stress_test(self, portfolio_returns: List[float], n_simulations: int = 10000, 
                               time_horizon: int = 252) -> Dict[str, float]:
        """
        Perform Monte Carlo stress testing.
        
        Args:
            portfolio_returns: Historical portfolio returns
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            
        Returns:
            Monte Carlo stress test results
        """
        try:
            if not portfolio_returns:
                return {}
            
            returns_array = xp.array(portfolio_returns)
            mean_return = xp.mean(returns_array)
            std_return = xp.std(returns_array)
            
            # Generate random scenarios
            simulated_returns = xp.random.normal(mean_return, std_return, (n_simulations, time_horizon))
            
            # Calculate portfolio values
            portfolio_values = xp.cumprod(1 + simulated_returns, axis=1)
            
            # Calculate stress metrics
            final_values = portfolio_values[:, -1]
            max_drawdowns = xp.array([self._calculate_max_drawdown(sim) for sim in simulated_returns])
            
            results = {
                "mean_final_value": float(xp.mean(final_values)),
                "var_95_final_value": float(xp.percentile(final_values, 5)),
                "cvar_95_final_value": float(xp.mean(final_values[final_values <= xp.percentile(final_values, 5)])),
                "mean_max_drawdown": float(xp.mean(max_drawdowns)),
                "var_95_max_drawdown": float(xp.percentile(max_drawdowns, 95)),
                "worst_case_return": float(xp.min(final_values) - 1)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Monte Carlo stress test failed: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: Union[List[float], xp.ndarray]) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            if isinstance(returns, list):
                returns = xp.array(returns)
            
            cumulative = xp.cumprod(1 + returns)
            running_max = xp.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            return float(xp.min(drawdown))
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0


class RiskMonitor:
    """
    Real-time risk monitoring and alerting system.
    
    Features:
    - Real-time risk metrics calculation
    - Risk limit monitoring
    - Alert generation
    - Risk factor tracking
    """

    def __init__(self, risk_limits: RiskLimit) -> None:
        """Initialize risk monitor."""
        self.risk_limits = risk_limits
        self.risk_history = []
        self.alerts = []
        self.risk_factors = {}

    def calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            portfolio_data: Portfolio data including returns, positions, etc.
            
        Returns:
            Risk metrics
        """
        try:
            returns = portfolio_data.get("returns", [])
            positions = portfolio_data.get("positions", {})
            market_data = portfolio_data.get("market_data", {})
            
            if not returns:
                return self._get_default_risk_metrics()
            
            returns_array = xp.array(returns)
            
            # Calculate basic metrics
            volatility = float(xp.std(returns_array) * xp.sqrt(252))
            mean_return = float(xp.mean(returns_array) * 252)
            
            # VaR and CVaR
            var_95 = float(xp.percentile(returns_array, 5))
            cvar_95 = float(xp.mean(returns_array[returns_array <= var_95]))
            
            # Risk-adjusted returns
            excess_return = mean_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = float(xp.std(downside_returns) * xp.sqrt(252)) if len(downside_returns) > 0 else 0.0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
            
            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(returns_array)
            
            # Beta and correlation
            market_returns = market_data.get("market_returns", returns_array)
            if len(market_returns) == len(returns_array):
                correlation = float(xp.corrcoef(returns_array, market_returns)[0, 1])
                beta = correlation * volatility / float(xp.std(market_returns) * xp.sqrt(252)) if len(market_returns) > 0 else 1.0
            else:
                correlation = 0.0
                beta = 1.0
            
            # Concentration and leverage
            concentration = self._calculate_concentration(positions)
            leverage = self._calculate_leverage(positions)
            
            # Liquidity score
            liquidity_score = self._calculate_liquidity_score(positions, market_data)
            
            # Overall risk score
            risk_score = self._calculate_overall_risk_score(
                var_95, max_drawdown, concentration, leverage, volatility
            )
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                concentration=concentration,
                leverage=leverage,
                liquidity_score=liquidity_score,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return self._get_default_risk_metrics()

    def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[str]:
        """
        Check risk metrics against limits.
        
        Args:
            risk_metrics: Current risk metrics
            
        Returns:
            List of risk limit violations
        """
        try:
            violations = []
            
            # Check VaR limit
            if abs(risk_metrics.var_95) > self.risk_limits.max_portfolio_risk:
                violations.append(f"VaR limit exceeded: {risk_metrics.var_95:.4f} > {self.risk_limits.max_portfolio_risk}")
            
            # Check max drawdown
            if abs(risk_metrics.max_drawdown) > self.risk_limits.max_drawdown:
                violations.append(f"Max drawdown limit exceeded: {risk_metrics.max_drawdown:.4f} > {self.risk_limits.max_drawdown}")
            
            # Check leverage
            if risk_metrics.leverage > self.risk_limits.max_leverage:
                violations.append(f"Leverage limit exceeded: {risk_metrics.leverage:.2f} > {self.risk_limits.max_leverage}")
            
            # Check concentration
            if risk_metrics.concentration > self.risk_limits.max_concentration:
                violations.append(f"Concentration limit exceeded: {risk_metrics.concentration:.4f} > {self.risk_limits.max_concentration}")
            
            # Check liquidity
            if risk_metrics.liquidity_score < self.risk_limits.min_liquidity:
                violations.append(f"Liquidity below minimum: {risk_metrics.liquidity_score:.0f} < {self.risk_limits.min_liquidity}")
            
            return violations
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return []

    def _calculate_max_drawdown(self, returns: xp.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = xp.cumprod(1 + returns)
            running_max = xp.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(xp.min(drawdown))
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_concentration(self, positions: Dict[str, float]) -> float:
        """Calculate portfolio concentration."""
        try:
            if not positions:
                return 0.0
            
            position_values = list(positions.values())
            total_value = sum(position_values)
            
            if total_value == 0:
                return 0.0
            
            # Herfindahl-Hirschman Index
            weights = [v / total_value for v in position_values]
            concentration = sum(w**2 for w in weights)
            
            return concentration
            
        except Exception as e:
            logger.error(f"Concentration calculation failed: {e}")
            return 0.0

    def _calculate_leverage(self, positions: Dict[str, float]) -> float:
        """Calculate portfolio leverage."""
        try:
            if not positions:
                return 1.0
            
            total_long = sum(v for v in positions.values() if v > 0)
            total_short = abs(sum(v for v in positions.values() if v < 0))
            
            total_exposure = total_long + total_short
            net_exposure = total_long - total_short
            
            if net_exposure == 0:
                return 1.0
            
            return total_exposure / abs(net_exposure)
            
        except Exception as e:
            logger.error(f"Leverage calculation failed: {e}")
            return 1.0

    def _calculate_liquidity_score(self, positions: Dict[str, float], market_data: Dict[str, Any]) -> float:
        """Calculate liquidity score."""
        try:
            if not positions:
                return 0.0
            
            total_liquidity = 0.0
            for symbol, position in positions.items():
                volume = market_data.get(f"{symbol}_volume", 1000000)  # Default $1M
                total_liquidity += min(abs(position), volume)
            
            return total_liquidity
            
        except Exception as e:
            logger.error(f"Liquidity score calculation failed: {e}")
            return 0.0

    def _calculate_overall_risk_score(self, var_95: float, max_drawdown: float, 
                                    concentration: float, leverage: float, volatility: float) -> float:
        """Calculate overall risk score (0-1)."""
        try:
            # Normalize each component to 0-1
            var_score = min(1.0, abs(var_95) / 0.1)  # 10% VaR = max risk
            drawdown_score = min(1.0, abs(max_drawdown) / 0.5)  # 50% drawdown = max risk
            concentration_score = concentration  # Already 0-1
            leverage_score = min(1.0, leverage / 5.0)  # 5x leverage = max risk
            volatility_score = min(1.0, volatility / 0.5)  # 50% volatility = max risk
            
            # Weighted average
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # VaR, drawdown, concentration, leverage, volatility
            risk_score = (
                var_score * weights[0] +
                drawdown_score * weights[1] +
                concentration_score * weights[2] +
                leverage_score * weights[3] +
                volatility_score * weights[4]
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Overall risk score calculation failed: {e}")
            return 0.5

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics."""
        return RiskMetrics(
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            beta=1.0,
            correlation=0.0,
            concentration=0.0,
            leverage=1.0,
            liquidity_score=0.0,
            risk_score=0.0
        )


class AdvancedRiskManagementSystem:
    """
    Advanced Risk Management System - Complete risk management solution.
    
    Features:
    - Dynamic risk assessment
    - Position sizing optimization
    - Risk-adjusted returns
    - Portfolio stress testing
    - Real-time risk monitoring
    - Risk factor analysis
    - Dynamic risk limits
    """

    def __init__(self, risk_limits: RiskLimit = None) -> None:
        """Initialize Advanced Risk Management System."""
        self.risk_limits = risk_limits or RiskLimit()
        
        # Initialize components
        self.position_sizer = PositionSizer()
        self.stress_tester = PortfolioStressTester()
        self.risk_monitor = RiskMonitor(self.risk_limits)
        
        # Performance tracking
        self.risk_assessments = []
        self.position_adjustments = []
        self.risk_alerts = []
        
        # Initialize Schwabot components if available
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.tensor_algebra = AdvancedTensorAlgebra()
            self.entropy_system = EntropyMathSystem()
            self.distributed_processor = DistributedMathematicalProcessor()
        
        logger.info("🛡️ Advanced Risk Management System initialized")

    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> RiskAssessment:
        """
        Perform comprehensive portfolio risk assessment.
        
        Args:
            portfolio_data: Portfolio data including positions, returns, market data
            
        Returns:
            Risk assessment result
        """
        try:
            # Calculate risk metrics
            risk_metrics = self.risk_monitor.calculate_risk_metrics(portfolio_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(portfolio_data, risk_metrics)
            
            # Check risk limits
            violations = self.risk_monitor.check_risk_limits(risk_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_metrics, risk_factors, violations)
            
            # Generate alerts
            alerts = self._generate_alerts(risk_metrics, violations)
            
            # Calculate position adjustments
            position_adjustments = self._calculate_position_adjustments(portfolio_data, risk_metrics)
            
            # Create risk assessment
            assessment = RiskAssessment(
                timestamp=time.time(),
                risk_level=risk_level,
                risk_metrics=risk_metrics,
                risk_factors=risk_factors,
                recommendations=recommendations,
                alerts=alerts,
                position_adjustments=position_adjustments
            )
            
            # Store assessment
            self.risk_assessments.append(assessment)
            
            # Keep history manageable
            if len(self.risk_assessments) > 1000:
                self.risk_assessments = self.risk_assessments[-500:]
            
            return assessment
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return self._get_default_assessment()

    def optimize_position_sizes(self, portfolio_data: Dict[str, Any], 
                              target_risk: float = 0.02) -> Dict[str, float]:
        """
        Optimize position sizes using multiple methodologies.
        
        Args:
            portfolio_data: Portfolio data
            target_risk: Target risk per position
            
        Returns:
            Optimized position sizes
        """
        try:
            positions = portfolio_data.get("positions", {})
            returns = portfolio_data.get("returns", [])
            volatilities = portfolio_data.get("volatilities", {})
            
            if not positions:
                return {}
            
            optimized_sizes = {}
            
            for symbol, current_size in positions.items():
                # Get historical performance for this symbol
                symbol_returns = portfolio_data.get(f"{symbol}_returns", returns)
                
                if symbol_returns:
                    # Calculate win rate and average win/loss
                    positive_returns = [r for r in symbol_returns if r > 0]
                    negative_returns = [r for r in symbol_returns if r < 0]
                    
                    win_rate = len(positive_returns) / len(symbol_returns) if symbol_returns else 0.5
                    avg_win = np.mean(positive_returns) if positive_returns else 0.01
                    avg_loss = abs(np.mean(negative_returns)) if negative_returns else 0.01
                    
                    # Kelly criterion
                    kelly_size = self.position_sizer.kelly_criterion(win_rate, avg_win, avg_loss)
                    
                    # Optimal f
                    optimal_f_size = self.position_sizer.optimal_f(symbol_returns, target_risk)
                    
                    # Volatility targeting
                    current_vol = volatilities.get(symbol, 0.2)
                    vol_target_size = self.position_sizer.volatility_targeting(current_vol, 0.15, current_size)
                    
                    # Combine methodologies (weighted average)
                    optimized_size = (
                        kelly_size * 0.4 +
                        optimal_f_size * 0.3 +
                        vol_target_size * 0.3
                    )
                    
                    # Apply risk limits
                    optimized_size = min(optimized_size, self.risk_limits.max_position_size)
                    
                    optimized_sizes[symbol] = optimized_size
            
            return optimized_sizes
            
        except Exception as e:
            logger.error(f"Position size optimization failed: {e}")
            return {}

    def perform_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive stress testing.
        
        Args:
            portfolio_data: Portfolio data
            
        Returns:
            Stress test results
        """
        try:
            returns = portfolio_data.get("returns", [])
            
            if not returns:
                return {"error": "No return data available"}
            
            # Historical stress testing
            stress_periods = [
                (0, min(30, len(returns))),  # Last 30 days
                (0, min(90, len(returns))),  # Last 90 days
                (0, min(252, len(returns)))  # Last year
            ]
            
            historical_results = self.stress_tester.historical_stress_test(returns, stress_periods)
            
            # Monte Carlo stress testing
            monte_carlo_results = self.stress_tester.monte_carlo_stress_test(returns)
            
            # Combine results
            stress_results = {
                "historical_stress_test": historical_results,
                "monte_carlo_stress_test": monte_carlo_results,
                "timestamp": time.time()
            }
            
            # Store results
            self.stress_tester.stress_results.append(stress_results)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {"error": str(e)}

    def _determine_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """Determine risk level based on metrics."""
        try:
            risk_score = risk_metrics.risk_score
            
            if risk_score < 0.25:
                return RiskLevel.LOW
            elif risk_score < 0.5:
                return RiskLevel.MEDIUM
            elif risk_score < 0.75:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
                
        except Exception as e:
            logger.error(f"Risk level determination failed: {e}")
            return RiskLevel.MEDIUM

    def _calculate_risk_factors(self, portfolio_data: Dict[str, Any], 
                              risk_metrics: RiskMetrics) -> Dict[RiskFactor, float]:
        """Calculate risk factors."""
        try:
            risk_factors = {}
            
            # Market risk
            risk_factors[RiskFactor.MARKET] = risk_metrics.beta
            
            # Volatility risk
            risk_factors[RiskFactor.VOLATILITY] = risk_metrics.volatility
            
            # Liquidity risk
            risk_factors[RiskFactor.LIQUIDITY] = 1.0 - risk_metrics.liquidity_score / 10000000  # Normalize
            
            # Concentration risk
            risk_factors[RiskFactor.CONCENTRATION] = risk_metrics.concentration
            
            # Correlation risk
            risk_factors[RiskFactor.CORRELATION] = abs(risk_metrics.correlation)
            
            # Leverage risk
            risk_factors[RiskFactor.LEVERAGE] = risk_metrics.leverage / 5.0  # Normalize to 5x max
            
            # Operational risk (placeholder)
            risk_factors[RiskFactor.OPERATIONAL] = 0.1  # Base operational risk
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk factor calculation failed: {e}")
            return {factor: 0.0 for factor in RiskFactor}

    def _generate_recommendations(self, risk_metrics: RiskMetrics, 
                                risk_factors: Dict[RiskFactor, float],
                                violations: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        try:
            recommendations = []
            
            # High VaR recommendations
            if abs(risk_metrics.var_95) > self.risk_limits.max_portfolio_risk * 0.8:
                recommendations.append("Consider reducing position sizes to lower VaR")
            
            # High drawdown recommendations
            if abs(risk_metrics.max_drawdown) > self.risk_limits.max_drawdown * 0.8:
                recommendations.append("Implement tighter stop-losses to limit drawdown")
            
            # High concentration recommendations
            if risk_metrics.concentration > self.risk_limits.max_concentration * 0.8:
                recommendations.append("Diversify portfolio to reduce concentration risk")
            
            # High leverage recommendations
            if risk_metrics.leverage > self.risk_limits.max_leverage * 0.8:
                recommendations.append("Reduce leverage to improve risk profile")
            
            # Low Sharpe ratio recommendations
            if risk_metrics.sharpe_ratio < 0.5:
                recommendations.append("Optimize portfolio for better risk-adjusted returns")
            
            # High volatility recommendations
            if risk_metrics.volatility > 0.3:
                recommendations.append("Consider volatility targeting strategies")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Monitor risk metrics closely"]

    def _generate_alerts(self, risk_metrics: RiskMetrics, violations: List[str]) -> List[str]:
        """Generate risk alerts."""
        try:
            alerts = []
            
            # Add limit violations as alerts
            alerts.extend(violations)
            
            # Add threshold-based alerts
            if risk_metrics.risk_score > 0.8:
                alerts.append("CRITICAL: Portfolio risk level is very high")
            elif risk_metrics.risk_score > 0.6:
                alerts.append("WARNING: Portfolio risk level is elevated")
            
            if risk_metrics.sharpe_ratio < 0:
                alerts.append("WARNING: Portfolio has negative risk-adjusted returns")
            
            if risk_metrics.liquidity_score < self.risk_limits.min_liquidity * 0.5:
                alerts.append("WARNING: Portfolio liquidity is low")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return ["Risk monitoring system error"]

    def _calculate_position_adjustments(self, portfolio_data: Dict[str, Any], 
                                      risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Calculate recommended position adjustments."""
        try:
            positions = portfolio_data.get("positions", {})
            adjustments = {}
            
            if not positions:
                return adjustments
            
            # Calculate adjustment factor based on risk level
            if risk_metrics.risk_score > 0.8:
                adjustment_factor = 0.5  # Reduce positions by 50%
            elif risk_metrics.risk_score > 0.6:
                adjustment_factor = 0.8  # Reduce positions by 20%
            elif risk_metrics.risk_score < 0.2:
                adjustment_factor = 1.2  # Increase positions by 20%
            else:
                adjustment_factor = 1.0  # No adjustment
            
            for symbol, current_size in positions.items():
                new_size = current_size * adjustment_factor
                adjustments[symbol] = new_size - current_size
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Position adjustment calculation failed: {e}")
            return {}

    def _get_default_assessment(self) -> RiskAssessment:
        """Get default risk assessment."""
        return RiskAssessment(
            timestamp=time.time(),
            risk_level=RiskLevel.MEDIUM,
            risk_metrics=self.risk_monitor._get_default_risk_metrics(),
            risk_factors={factor: 0.0 for factor in RiskFactor},
            recommendations=["Monitor risk metrics"],
            alerts=["Risk assessment failed"],
            position_adjustments={}
        )

    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics."""
        try:
            if not self.risk_assessments:
                return {"error": "No risk assessments available"}
            
            recent_assessments = self.risk_assessments[-100:]  # Last 100 assessments
            
            risk_levels = [assessment.risk_level.value for assessment in recent_assessments]
            risk_scores = [assessment.risk_metrics.risk_score for assessment in recent_assessments]
            
            return {
                "total_assessments": len(self.risk_assessments),
                "recent_assessments": len(recent_assessments),
                "average_risk_score": float(np.mean(risk_scores)),
                "risk_level_distribution": {
                    level: risk_levels.count(level) for level in set(risk_levels)
                },
                "max_risk_score": float(np.max(risk_scores)),
                "min_risk_score": float(np.min(risk_scores)),
                "risk_score_volatility": float(np.std(risk_scores)),
                "recent_alerts": len([a for assessment in recent_assessments for a in assessment.alerts]),
                "position_adjustments": len(self.position_adjustments)
            }
            
        except Exception as e:
            logger.error(f"Risk statistics calculation failed: {e}")
            return {"error": str(e)}

    def cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            self.risk_assessments.clear()
            self.position_adjustments.clear()
            self.risk_alerts.clear()
            
            if SCHWABOT_COMPONENTS_AVAILABLE:
                self.tensor_algebra.clear_cache()
                self.entropy_system.clear_cache()
                self.distributed_processor.cleanup_resources()
            
            logger.info("🛡️ Advanced Risk Management System resources cleaned up")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

    def __del__(self) -> None:
        """Cleanup on destruction."""
        try:
            self.cleanup_resources()
        except Exception:
            pass


# Global instance for easy access
advanced_risk_management_system = AdvancedRiskManagementSystem()

# Convenience functions
def assess_portfolio_risk(portfolio_data: Dict[str, Any]) -> RiskAssessment:
    """Convenience function for portfolio risk assessment."""
    return advanced_risk_management_system.assess_portfolio_risk(portfolio_data)

def optimize_position_sizes(portfolio_data: Dict[str, Any], target_risk: float = 0.02) -> Dict[str, float]:
    """Convenience function for position size optimization."""
    return advanced_risk_management_system.optimize_position_sizes(portfolio_data, target_risk)

def perform_stress_test(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for stress testing."""
    return advanced_risk_management_system.perform_stress_test(portfolio_data) 