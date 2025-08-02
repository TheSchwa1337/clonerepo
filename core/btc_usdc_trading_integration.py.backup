#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
₿ BTC/USDC TRADING INTEGRATION - SPECIALIZED TRADING SYSTEM
===========================================================

Specialized integration for BTC/USDC trading with:
- Optimized order execution
- Enhanced risk management
- Market microstructure analysis
- Integration with mathematical systems
- Portfolio balancing coordination
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Import core components
from algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer
from distributed_mathematical_processor import DistributedMathematicalProcessor
from enhanced_error_recovery_system import EnhancedErrorRecoverySystem, error_recovery_decorator
from neural_processing_engine import NeuralProcessingEngine
from unified_profit_vectorization_system import UnifiedProfitVectorizationSystem

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Trading decision with comprehensive analysis."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    confidence: float
    risk_score: float
    profit_potential: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class BTCUSDCTradingConfig:
    """BTC/USDC trading configuration."""
    symbol: str = "BTC/USDC"
    base_order_size: float = 0.01  # 0.01 BTC
    max_order_size: float = 0.1  # 0.1 BTC max
    min_order_size: float = 0.001  # 0.001 BTC min
    
    # Risk management
    max_position_size_pct: float = 0.2  # 20% max position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_daily_trades: int = 50
    
    # Market analysis
    order_book_depth: int = 100
    spread_threshold: float = 0.001  # 0.1% max spread
    volume_threshold: float = 1000000  # $1M min volume
    
    # Mathematical integration
    entropy_threshold: float = 0.6
    potential_threshold: float = 0.7
    
    # Portfolio integration
    enable_portfolio_balancing: bool = True
    rebalance_threshold: float = 0.05  # 5% rebalance threshold


class BTCUSDCTradingIntegration:
    """Specialized BTC/USDC trading integration."""
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = BTCUSDCTradingConfig(**(config or {}).get("btc_usdc_config", {}))
        
        # Initialize core components
        self.distributed_processor = DistributedMathematicalProcessor()
        self.neural_engine = NeuralProcessingEngine()
        self.profit_vectorizer = UnifiedProfitVectorizationSystem()
        self.error_recovery = EnhancedErrorRecoverySystem()
        
        # Trading components (will be injected)
        self.portfolio_balancer: Optional[AlgorithmicPortfolioBalancer] = None
        
        # Trading state
        self.current_position = 0.0
        self.daily_trades = 0
        self.last_trade_time = 0
        self.position_entry_price = 0.0
        
        # Market data cache
        self.market_data_cache = {}
        self.order_book_cache = {}
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info("₿ BTC/USDC Trading Integration initialized")
    
    async def inject_trading_components(self, portfolio_balancer: AlgorithmicPortfolioBalancer) -> None:
        """Inject trading components into the integration."""
        try:
            self.portfolio_balancer = portfolio_balancer
            logger.info("🔌 Trading components injected into BTC/USDC Integration")
        except Exception as e:
            logger.error(f"Error injecting trading components: {e}")
    
    @error_recovery_decorator
    async def initialize(self) -> bool:
        """Initialize the trading integration."""
        try:
            # Initialize distributed processor
            await self.distributed_processor.initialize()
            
            # Initialize neural engine
            await self.neural_engine.initialize()
            
            # Initialize profit vectorizer
            await self.profit_vectorizer.initialize()
            
            # Load initial market data
            await self._update_market_data()
            
            logger.info("₿ BTC/USDC Trading Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing BTC/USDC trading integration: {e}")
            return False
    
    @error_recovery_decorator
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """Process market data and generate trading decisions."""
        try:
            # Update market data cache
            self.market_data_cache.update(market_data)
            
            # Check if we should trade
            if not self._should_trade():
                return None
            
            # Analyze market conditions
            market_analysis = await self._analyze_market_conditions()
            
            # Check mathematical signals
            mathematical_signal = await self._check_mathematical_signals()
            
            # Check portfolio balancing needs
            portfolio_signal = await self._check_portfolio_balancing()
            
            # Generate trading decision
            decision = await self._generate_trading_decision(
                market_analysis, mathematical_signal, portfolio_signal
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None
    
    @error_recovery_decorator
    async def execute_trade(self, decision: TradingDecision) -> bool:
        """Execute a trading decision."""
        try:
            # Validate decision
            if not self._validate_trading_decision(decision):
                return False
            
            # Simulate trade execution (replace with actual exchange integration)
            success = await self._simulate_trade_execution(decision)
            
            if success:
                # Update position
                await self._update_position(decision)
                
                # Update portfolio state
                if self.portfolio_balancer:
                    await self.portfolio_balancer.update_portfolio_state(self.market_data_cache)
                
                # Log trade
                self._log_trade(decision)
                
                # Check if rebalancing is needed
                if self.config.enable_portfolio_balancing and self.portfolio_balancer:
                    await self._check_and_execute_rebalancing()
                
                logger.info(f"Trade executed: {decision.symbol} {decision.action} {decision.quantity}")
                return True
            else:
                logger.warning(f"Trade execution failed: {decision.symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            # Extract market data
            price = self.market_data_cache.get("price", 50000.0)
            volume = self.market_data_cache.get("volume", 1000000.0)
            volatility = self.market_data_cache.get("volatility", 0.02)
            
            # Calculate spread
            spread = self._calculate_spread(self.order_book_cache)
            
            # Calculate volatility
            calculated_volatility = self._calculate_volatility()
            
            # Market condition assessment
            market_condition = "normal"
            if volatility > 0.05:
                market_condition = "high_volatility"
            elif volatility < 0.01:
                market_condition = "low_volatility"
            
            # Volume assessment
            volume_condition = "normal"
            if volume > self.config.volume_threshold * 2:
                volume_condition = "high_volume"
            elif volume < self.config.volume_threshold * 0.5:
                volume_condition = "low_volume"
            
            return {
                "price": price,
                "volume": volume,
                "volatility": volatility,
                "calculated_volatility": calculated_volatility,
                "spread": spread,
                "market_condition": market_condition,
                "volume_condition": volume_condition,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                "price": 50000.0,
                "volume": 1000000.0,
                "volatility": 0.02,
                "calculated_volatility": 0.02,
                "spread": 0.001,
                "market_condition": "normal",
                "volume_condition": "normal",
                "timestamp": time.time()
            }
    
    async def _check_mathematical_signals(self) -> Optional[Dict[str, Any]]:
        """Check mathematical signals for trading."""
        try:
            # Use distributed processor for mathematical analysis
            math_task = {
                "type": "btc_usdc_analysis",
                "market_data": self.market_data_cache,
                "current_position": self.current_position,
                "config": {
                    "entropy_threshold": self.config.entropy_threshold,
                    "potential_threshold": self.config.potential_threshold
                }
            }
            
            result = await self.distributed_processor.process_task(math_task)
            
            if result and result.get("signal_strength", 0) > 0.5:
                return {
                    "signal_strength": result.get("signal_strength", 0),
                    "entropy_score": result.get("entropy_score", 0),
                    "potential_score": result.get("potential_score", 0),
                    "recommended_action": result.get("recommended_action", "hold"),
                    "confidence": result.get("confidence", 0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking mathematical signals: {e}")
            return None
    
    async def _check_portfolio_balancing(self) -> Optional[Dict[str, Any]]:
        """Check portfolio balancing needs."""
        try:
            if not self.portfolio_balancer:
                return None
            
            portfolio_state = await self.portfolio_balancer.get_portfolio_state()
            
            if portfolio_state.get("rebalancing_needed", False):
                return {
                    "rebalancing_needed": True,
                    "current_allocation": portfolio_state.get("allocation_weights", {}),
                    "target_allocation": portfolio_state.get("target_weights", {}),
                    "rebalancing_amount": portfolio_state.get("rebalancing_amount", 0.0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking portfolio balancing: {e}")
            return None
    
    async def _generate_trading_decision(
        self,
        market_analysis: Dict[str, Any],
        mathematical_signal: Optional[Dict[str, Any]],
        portfolio_signal: Optional[Dict[str, Any]],
    ) -> Optional[TradingDecision]:
        """Generate trading decision based on all analyses."""
        try:
            # Base decision parameters
            symbol = self.config.symbol
            price = market_analysis["price"]
            timestamp = time.time()
            
            # Calculate signal strength
            signal_strength = 0.0
            action = "hold"
            confidence = 0.0
            
            # Mathematical signal contribution
            if mathematical_signal:
                math_strength = mathematical_signal["signal_strength"]
                signal_strength += math_strength * 0.6
                action = mathematical_signal["recommended_action"]
                confidence = mathematical_signal["confidence"]
            
            # Portfolio signal contribution
            if portfolio_signal:
                portfolio_strength = 0.8 if portfolio_signal["rebalancing_needed"] else 0.0
                signal_strength += portfolio_strength * 0.4
                if portfolio_strength > 0.5:
                    action = "buy" if portfolio_signal["rebalancing_amount"] > 0 else "sell"
            
            # Market condition adjustment
            market_condition = market_analysis["market_condition"]
            if market_condition == "high_volatility":
                signal_strength *= 0.7  # Reduce signal strength in high volatility
            elif market_condition == "low_volatility":
                signal_strength *= 1.2  # Increase signal strength in low volatility
            
            # Calculate position size
            quantity = self._calculate_position_size(signal_strength)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(market_analysis, signal_strength)
            
            # Calculate profit potential
            profit_potential = self._calculate_profit_potential(signal_strength, market_analysis)
            
            # Final decision
            if signal_strength > 0.6 and confidence > 0.5:
                return TradingDecision(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    confidence=confidence,
                    risk_score=risk_score,
                    profit_potential=profit_potential,
                    timestamp=timestamp,
                    metadata={
                        "signal_strength": signal_strength,
                        "market_condition": market_condition,
                        "mathematical_signal": mathematical_signal,
                        "portfolio_signal": portfolio_signal
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return None
    
    def _should_trade(self) -> bool:
        """Check if we should trade based on various conditions."""
        try:
            # Check daily trade limit
            if self.daily_trades >= self.config.max_daily_trades:
                return False
            
            # Check time since last trade
            time_since_last_trade = time.time() - self.last_trade_time
            if time_since_last_trade < 60:  # Minimum 1 minute between trades
                return False
            
            # Check market data availability
            if not self.market_data_cache:
                return False
            
            # Check spread threshold
            spread = self._calculate_spread(self.order_book_cache)
            if spread > self.config.spread_threshold:
                return False
            
            # Check volume threshold
            volume = self.market_data_cache.get("volume", 0)
            if volume < self.config.volume_threshold:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if should trade: {e}")
            return False
    
    def _validate_trading_decision(self, decision: TradingDecision) -> bool:
        """Validate trading decision."""
        try:
            # Check basic parameters
            if not decision.symbol or decision.quantity <= 0:
                return False
            
            # Check position size limits
            max_position = self._calculate_max_position()
            if decision.action == "buy" and self.current_position + decision.quantity > max_position:
                return False
            elif decision.action == "sell" and self.current_position - decision.quantity < -max_position:
                return False
            
            # Check order size limits
            if decision.quantity < self.config.min_order_size:
                return False
            if decision.quantity > self.config.max_order_size:
                return False
            
            # Check confidence threshold
            if decision.confidence < 0.3:
                return False
            
            # Check risk score
            if decision.risk_score > 0.8:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trading decision: {e}")
            return False
    
    def _calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength."""
        try:
            # Base position size
            base_size = self.config.base_order_size
            
            # Scale by signal strength
            scaled_size = base_size * signal_strength
            
            # Apply limits
            scaled_size = max(scaled_size, self.config.min_order_size)
            scaled_size = min(scaled_size, self.config.max_order_size)
            
            return scaled_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config.min_order_size
    
    def _calculate_max_position(self) -> float:
        """Calculate maximum position size."""
        try:
            # This would typically be based on account balance
            # For now, use a fixed value
            account_balance = 100000.0  # $100k
            max_position_value = account_balance * self.config.max_position_size_pct
            current_price = self.market_data_cache.get("price", 50000.0)
            
            return max_position_value / current_price
            
        except Exception as e:
            logger.error(f"Error calculating max position: {e}")
            return 0.1  # Default to 0.1 BTC
    
    def _calculate_spread(self, order_book: Dict[str, Any]) -> float:
        """Calculate current spread from order book."""
        try:
            if not order_book:
                return 0.001  # Default spread
            
            best_bid = order_book.get("bids", [[50000.0, 1.0]])[0][0]
            best_ask = order_book.get("asks", [[50001.0, 1.0]])[0][0]
            
            spread = (best_ask - best_bid) / best_bid
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return 0.001
    
    def _calculate_volatility(self) -> float:
        """Calculate current volatility."""
        try:
            # This would typically use historical price data
            # For now, use a simple calculation based on recent price changes
            prices = self.market_data_cache.get("price_history", [50000.0])
            
            if len(prices) < 2:
                return 0.02  # Default volatility
            
            returns = []
            for i in range(1, len(prices)):
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if returns:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                return volatility
            else:
                return 0.02
                
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _calculate_risk_score(self, market_analysis: Dict[str, Any], signal_strength: float) -> float:
        """Calculate risk score for the trading decision."""
        try:
            # Base risk from volatility
            volatility = market_analysis.get("volatility", 0.02)
            base_risk = min(volatility * 10, 1.0)
            
            # Risk reduction from signal strength
            signal_risk_reduction = signal_strength * 0.3
            
            # Risk increase from spread
            spread = market_analysis.get("spread", 0.001)
            spread_risk = min(spread * 100, 0.5)
            
            # Final risk score
            risk_score = base_risk + spread_risk - signal_risk_reduction
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _calculate_profit_potential(self, signal_strength: float, market_analysis: Dict[str, Any]) -> float:
        """Calculate profit potential for the trading decision."""
        try:
            # Base profit from signal strength
            base_profit = signal_strength * 0.05  # 5% max profit potential
            
            # Market condition adjustment
            market_condition = market_analysis.get("market_condition", "normal")
            if market_condition == "high_volatility":
                base_profit *= 1.5  # Higher potential in volatile markets
            elif market_condition == "low_volatility":
                base_profit *= 0.8  # Lower potential in stable markets
            
            # Volume adjustment
            volume_condition = market_analysis.get("volume_condition", "normal")
            if volume_condition == "high_volume":
                base_profit *= 1.2  # Higher potential with high volume
            elif volume_condition == "low_volume":
                base_profit *= 0.7  # Lower potential with low volume
            
            return max(0.0, base_profit)
            
        except Exception as e:
            logger.error(f"Error calculating profit potential: {e}")
            return 0.0
    
    async def _simulate_trade_execution(self, decision: TradingDecision) -> bool:
        """Simulate trade execution (replace with actual exchange integration)."""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Simulate execution success (90% success rate)
            success = np.random.random() > 0.1
            
            if success:
                # Update trade count
                self.daily_trades += 1
                self.last_trade_time = time.time()
            
            return success
            
        except Exception as e:
            logger.error(f"Error simulating trade execution: {e}")
            return False
    
    async def _update_position(self, decision: TradingDecision) -> None:
        """Update current position after trade execution."""
        try:
            if decision.action == "buy":
                self.current_position += decision.quantity
                if self.current_position > 0:
                    self.position_entry_price = decision.price
            elif decision.action == "sell":
                self.current_position -= decision.quantity
                if self.current_position < 0:
                    self.position_entry_price = decision.price
                    
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def _check_and_execute_rebalancing(self) -> None:
        """Check and execute portfolio rebalancing."""
        try:
            if not self.portfolio_balancer:
                return
            
            # Check if rebalancing is needed
            portfolio_state = await self.portfolio_balancer.get_portfolio_state()
            
            if portfolio_state.get("rebalancing_needed", False):
                # Execute rebalancing
                rebalancing_result = await self.portfolio_balancer.execute_rebalancing()
                
                if rebalancing_result.get("success", False):
                    logger.info("Portfolio rebalancing executed successfully")
                else:
                    logger.warning("Portfolio rebalancing failed")
                    
        except Exception as e:
            logger.error(f"Error checking and executing rebalancing: {e}")
    
    def _log_trade(self, decision: TradingDecision) -> None:
        """Log trade information."""
        try:
            trade_log = {
                "timestamp": decision.timestamp,
                "symbol": decision.symbol,
                "action": decision.action,
                "quantity": decision.quantity,
                "price": decision.price,
                "confidence": decision.confidence,
                "risk_score": decision.risk_score,
                "profit_potential": decision.profit_potential,
                "current_position": self.current_position,
                "daily_trades": self.daily_trades
            }
            
            self.trade_history.append(trade_log)
            
            # Keep only last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    async def _update_market_data(self) -> None:
        """Update market data cache."""
        try:
            # This would typically fetch from exchange API
            # For now, use simulated data
            self.market_data_cache.update({
                "price": 50000.0 + np.random.normal(0, 100),
                "volume": 1000000.0 + np.random.normal(0, 100000),
                "volatility": 0.02 + np.random.normal(0, 0.01),
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                "current_position": self.current_position,
                "position_entry_price": self.position_entry_price,
                "daily_trades": self.daily_trades,
                "last_trade_time": self.last_trade_time,
                "market_data": self.market_data_cache,
                "trade_history_size": len(self.trade_history),
                "performance_metrics": self.performance_metrics,
                "config": {
                    "symbol": self.config.symbol,
                    "max_position_size_pct": self.config.max_position_size_pct,
                    "stop_loss_pct": self.config.stop_loss_pct,
                    "take_profit_pct": self.config.take_profit_pct
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    async def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.trade_history:
                return {}
            
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get("profit_potential", 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit metrics
            total_profit = sum(trade.get("profit_potential", 0) for trade in self.trade_history)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0.0
            
            # Calculate risk metrics
            avg_risk = np.mean([trade.get("risk_score", 0) for trade in self.trade_history])
            max_risk = max([trade.get("risk_score", 0) for trade in self.trade_history])
            
            # Calculate confidence metrics
            avg_confidence = np.mean([trade.get("confidence", 0) for trade in self.trade_history])
            
            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "avg_profit": avg_profit,
                "avg_risk": avg_risk,
                "max_risk": max_risk,
                "avg_confidence": avg_confidence,
                "current_position": self.current_position,
                "daily_trades": self.daily_trades
            }
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}


def create_btc_usdc_integration(config: Dict[str, Any] = None) -> BTCUSDCTradingIntegration:
    """Factory function to create a BTC/USDC Trading Integration."""
    return BTCUSDCTradingIntegration(config)


# Global instance for easy access
btc_usdc_integration = create_btc_usdc_integration() 