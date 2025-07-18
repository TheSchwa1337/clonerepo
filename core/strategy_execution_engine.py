#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Execution Engine for Schwabot Trading System
====================================================

Real-time strategy execution engine that integrates with live market data
to execute trading strategies based on:
- RSI triggers and analysis
- Time-based phase detection (midnight/noon patterns)
- Decimal key mapping (2, 6, 8 tiers)
- Memory key recall and learning
- Cross-exchange arbitrage
- Volume-based triggers
- Risk management and position sizing

This engine executes the internalized trading strategies with real API data.
"""

import asyncio
import time
import logging
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

from .live_market_data_integration import (
    LiveMarketDataIntegration, 
    MarketData, 
    TradingSignal, 
    TimePhase, 
    StrategyTier,
    MemoryKey
)

logger = logging.getLogger(__name__)

# =============================================================================
# STRATEGY EXECUTION ENUMS AND DATA STRUCTURES
# =============================================================================

class ExecutionStatus(Enum):
    """Strategy execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RiskLevel(Enum):
    """Risk levels for position sizing."""
    CONSERVATIVE = "conservative"    # Tier 2
    MODERATE = "moderate"           # Tier 6
    AGGRESSIVE = "aggressive"       # Tier 8

@dataclass
class StrategyExecution:
    """Strategy execution record."""
    execution_id: str
    signal_id: str
    timestamp: float
    symbol: str
    action: str
    price: float
    amount: float
    strategy_tier: StrategyTier
    phase: TimePhase
    rsi_trigger: float
    volume_trigger: float
    hash_match: bool
    memory_recall: bool
    confidence: float
    priority: str
    status: ExecutionStatus
    exchange: str
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    execution_time: Optional[float] = None
    profit_loss: Optional[float] = None
    outcome: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class Position:
    """Trading position."""
    position_id: str
    symbol: str
    side: str  # long, short
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    strategy_tier: StrategyTier
    phase: TimePhase
    entry_time: float
    last_update: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.1  # Maximum position size in BTC
    max_daily_loss: float = 1000.0  # Maximum daily loss in USD
    max_open_positions: int = 10    # Maximum open positions
    stop_loss_percentage: float = 0.05  # 5% stop loss
    take_profit_percentage: float = 0.10  # 10% take profit
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    correlation_threshold: float = 0.7  # Maximum correlation between positions

# =============================================================================
# STRATEGY EXECUTION ENGINE
# =============================================================================

class StrategyExecutionEngine:
    """Real-time strategy execution engine."""
    
    def __init__(self, market_integration: LiveMarketDataIntegration, config: Dict[str, Any]):
        self.market_integration = market_integration
        self.config = config
        self.risk_config = RiskConfig(**config.get('risk', {}))
        
        # Execution state
        self.executions: Dict[str, StrategyExecution] = {}
        self.positions: Dict[str, Position] = {}
        self.pending_signals: List[TradingSignal] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_loss = 0.0
        self.daily_profit_loss = 0.0
        self.last_daily_reset = time.time()
        
        # Risk management
        self.daily_loss = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Threading
        self.running = False
        self.execution_thread = None
        self.risk_monitor_thread = None
        
        # Callbacks
        self.execution_callbacks: List[Callable] = []
        self.risk_callbacks: List[Callable] = []
        
        # Initialize
        self._initialize_engine()
        
        logger.info("🔧 Strategy Execution Engine initialized")
    
    def _initialize_engine(self):
        """Initialize the execution engine."""
        try:
            # Create execution directory
            self.execution_path = Path("executions")
            self.execution_path.mkdir(exist_ok=True)
            
            # Load historical executions
            self._load_historical_executions()
            
            # Load existing positions
            self._load_existing_positions()
            
            logger.info("✅ Strategy Execution Engine initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Error initializing execution engine: {e}")
            raise
    
    def start_execution(self):
        """Start the strategy execution engine."""
        try:
            self.running = True
            
            # Start execution thread
            self.execution_thread = threading.Thread(
                target=self._execution_loop,
                daemon=True,
                name="StrategyExecutor"
            )
            self.execution_thread.start()
            
            # Start risk monitor thread
            self.risk_monitor_thread = threading.Thread(
                target=self._risk_monitor_loop,
                daemon=True,
                name="RiskMonitor"
            )
            self.risk_monitor_thread.start()
            
            logger.info("🚀 Strategy Execution Engine started")
            
        except Exception as e:
            logger.error(f"❌ Error starting execution engine: {e}")
            raise
    
    def _execution_loop(self):
        """Main execution loop."""
        while self.running:
            try:
                # Get latest signals from market integration
                signals = self.market_integration.get_trading_signals(limit=50)
                
                # Process new signals
                for signal in signals:
                    if self._should_execute_signal(signal):
                        self._execute_signal(signal)
                
                # Update existing positions
                self._update_positions()
                
                # Check for exit conditions
                self._check_exit_conditions()
                
                # Sleep between iterations
                time.sleep(1.0)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"❌ Execution loop error: {e}")
                time.sleep(5.0)
    
    def _risk_monitor_loop(self):
        """Risk monitoring loop."""
        while self.running:
            try:
                # Check daily loss limit
                if self.daily_loss < -self.risk_config.max_daily_loss:
                    logger.critical(f"🚨 Daily loss limit exceeded: ${self.daily_loss:.2f}")
                    self._emergency_stop()
                
                # Check drawdown
                if self.current_drawdown < self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
                
                # Check position limits
                if len(self.positions) > self.risk_config.max_open_positions:
                    logger.warning(f"⚠️ Too many open positions: {len(self.positions)}")
                    self._reduce_positions()
                
                # Reset daily metrics if needed
                self._check_daily_reset()
                
                # Sleep between checks
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Risk monitor error: {e}")
                time.sleep(30.0)
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """Determine if a signal should be executed."""
        try:
            # Check if signal is already processed
            if any(execution.signal_id == signal.signal_id for execution in self.executions.values()):
                return False
            
            # Check risk limits
            if not self._check_risk_limits(signal):
                return False
            
            # Check confidence threshold
            if signal.confidence < 0.6:  # Minimum 60% confidence
                return False
            
            # Check priority
            if signal.priority == "critical":
                return True
            
            # Check strategy tier conditions
            if signal.strategy_tier == StrategyTier.TIER_8:
                return True  # High frequency always executes
            
            # Check phase conditions
            if signal.phase in [TimePhase.MIDNIGHT, TimePhase.HIGH_NOON]:
                return True
            
            # Check hash match and memory recall
            if signal.hash_match and signal.memory_recall:
                return True
            
            # Check RSI conditions
            if signal.rsi_trigger < 25 or signal.rsi_trigger > 75:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking signal execution: {e}")
            return False
    
    def _check_risk_limits(self, signal: TradingSignal) -> bool:
        """Check if signal meets risk limits."""
        try:
            # Check daily loss limit
            if self.daily_loss < -self.risk_config.max_daily_loss * 0.8:  # 80% of limit
                logger.warning("⚠️ Approaching daily loss limit, rejecting signal")
                return False
            
            # Check position size limit
            if signal.amount > self.risk_config.max_position_size:
                logger.warning(f"⚠️ Position size too large: {signal.amount}")
                return False
            
            # Check open positions limit
            if len(self.positions) >= self.risk_config.max_open_positions:
                logger.warning("⚠️ Maximum open positions reached")
                return False
            
            # Check correlation with existing positions
            if self._check_position_correlation(signal):
                logger.warning("⚠️ High correlation with existing position")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return False
    
    def _check_position_correlation(self, signal: TradingSignal) -> bool:
        """Check correlation with existing positions."""
        try:
            # Simplified correlation check
            # In a real implementation, this would calculate actual correlation
            for position in self.positions.values():
                if position.symbol == signal.symbol:
                    return True  # Same symbol = high correlation
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking position correlation: {e}")
            return False
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal."""
        try:
            # Create execution record
            execution_id = f"exec_{int(time.time() * 1000000)}"
            
            execution = StrategyExecution(
                execution_id=execution_id,
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                action=signal.action,
                price=signal.price,
                amount=signal.amount,
                strategy_tier=signal.strategy_tier,
                phase=signal.phase,
                rsi_trigger=signal.rsi_trigger,
                volume_trigger=signal.volume_trigger,
                hash_match=signal.hash_match,
                memory_recall=signal.memory_recall,
                confidence=signal.confidence,
                priority=signal.priority,
                status=ExecutionStatus.PENDING,
                exchange=signal.exchange
            )
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Execute the trade
            self._execute_trade(execution)
            
            logger.info(f"📊 Executing signal: {signal.action} {signal.symbol} "
                       f"@ ${signal.price:.2f} (Tier {signal.strategy_tier.value})")
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
    
    def _execute_trade(self, execution: StrategyExecution):
        """Execute the actual trade."""
        try:
            # Update status
            execution.status = ExecutionStatus.EXECUTING
            execution.execution_time = time.time()
            
            # Simulate order placement (replace with real exchange API calls)
            order_id = self._place_order(execution)
            execution.order_id = order_id
            
            # Simulate order execution
            execution_price = self._simulate_order_execution(execution)
            execution.execution_price = execution_price
            
            # Update status
            execution.status = ExecutionStatus.COMPLETED
            
            # Create position if buy action
            if execution.action == "buy":
                self._create_position(execution)
            
            # Update metrics
            self._update_execution_metrics(execution)
            
            # Save execution
            self._save_execution(execution)
            
            # Trigger callbacks
            self._trigger_execution_callbacks(execution)
            
            logger.info(f"✅ Trade executed: {execution.action} {execution.symbol} "
                       f"@ ${execution.execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
    
    def _place_order(self, execution: StrategyExecution) -> str:
        """Place order on exchange (simulated)."""
        try:
            # In a real implementation, this would use the exchange API
            # For now, simulate order placement
            order_id = f"order_{int(time.time() * 1000000)}"
            
            # Add small delay to simulate network latency
            time.sleep(0.1)
            
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            raise
    
    def _simulate_order_execution(self, execution: StrategyExecution) -> float:
        """Simulate order execution (replace with real execution)."""
        try:
            # Get current market price
            market_data = self.market_integration.get_latest_market_data(execution.symbol)
            
            if market_data:
                # Simulate slippage
                if execution.action == "buy":
                    execution_price = market_data.price * (1 + 0.001)  # 0.1% slippage
                else:
                    execution_price = market_data.price * (1 - 0.001)  # 0.1% slippage
            else:
                execution_price = execution.price
            
            return execution_price
            
        except Exception as e:
            logger.error(f"❌ Error simulating order execution: {e}")
            return execution.price
    
    def _create_position(self, execution: StrategyExecution):
        """Create a new trading position."""
        try:
            position_id = f"pos_{int(time.time() * 1000000)}"
            
            # Calculate stop loss and take profit
            stop_loss = execution.execution_price * (1 - self.risk_config.stop_loss_percentage)
            take_profit = execution.execution_price * (1 + self.risk_config.take_profit_percentage)
            
            position = Position(
                position_id=position_id,
                symbol=execution.symbol,
                side="long" if execution.action == "buy" else "short",
                amount=execution.amount,
                entry_price=execution.execution_price,
                current_price=execution.execution_price,
                unrealized_pnl=0.0,
                strategy_tier=execution.strategy_tier,
                phase=execution.phase,
                entry_time=execution.execution_time,
                last_update=time.time(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Store position
            self.positions[position_id] = position
            
            logger.info(f"📈 Position created: {position.side} {position.symbol} "
                       f"@ ${position.entry_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error creating position: {e}")
    
    def _update_positions(self):
        """Update existing positions with current market data."""
        try:
            current_time = time.time()
            
            for position_id, position in self.positions.items():
                # Get current market data
                market_data = self.market_integration.get_latest_market_data(position.symbol)
                
                if market_data:
                    # Update current price
                    position.current_price = market_data.price
                    position.last_update = current_time
                    
                    # Calculate unrealized P&L
                    if position.side == "long":
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.amount
                    
                    # Check stop loss and take profit
                    if position.stop_loss and position.take_profit:
                        if position.side == "long":
                            if position.current_price <= position.stop_loss:
                                self._close_position(position_id, "stop_loss")
                            elif position.current_price >= position.take_profit:
                                self._close_position(position_id, "take_profit")
                        else:  # short
                            if position.current_price >= position.stop_loss:
                                self._close_position(position_id, "stop_loss")
                            elif position.current_price <= position.take_profit:
                                self._close_position(position_id, "take_profit")
                
        except Exception as e:
            logger.error(f"❌ Error updating positions: {e}")
    
    def _close_position(self, position_id: str, reason: str):
        """Close a trading position."""
        try:
            position = self.positions.get(position_id)
            if not position:
                return
            
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            
            # Update daily metrics
            self.daily_profit_loss += realized_pnl
            self.daily_loss += realized_pnl
            
            # Update total metrics
            self.total_profit_loss += realized_pnl
            self.total_trades += 1
            
            if realized_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Remove position
            del self.positions[position_id]
            
            # Update memory key with outcome
            self._update_memory_key_outcome(position, realized_pnl)
            
            logger.info(f"📉 Position closed: {position.symbol} {reason} "
                       f"P&L: ${realized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
    
    def _check_exit_conditions(self):
        """Check for exit conditions on positions."""
        try:
            current_time = time.time()
            
            for position_id, position in list(self.positions.items()):
                # Check time-based exit
                position_age = current_time - position.entry_time
                
                # Exit based on strategy tier
                if position.strategy_tier == StrategyTier.TIER_2 and position_age > 3600:  # 1 hour
                    self._close_position(position_id, "time_exit_tier2")
                elif position.strategy_tier == StrategyTier.TIER_6 and position_age > 1800:  # 30 minutes
                    self._close_position(position_id, "time_exit_tier6")
                elif position.strategy_tier == StrategyTier.TIER_8 and position_age > 900:  # 15 minutes
                    self._close_position(position_id, "time_exit_tier8")
                
                # Check phase-based exit
                current_phase = self._get_current_phase()
                if position.phase != current_phase:
                    # Exit if phase has changed
                    self._close_position(position_id, "phase_change")
                
        except Exception as e:
            logger.error(f"❌ Error checking exit conditions: {e}")
    
    def _get_current_phase(self) -> TimePhase:
        """Get current time phase."""
        try:
            utc_hour = datetime.utcnow().hour
            
            if utc_hour == 0:
                return TimePhase.MIDNIGHT
            elif 1 <= utc_hour <= 2:
                return TimePhase.PRE_DAWN
            elif 3 <= utc_hour <= 11:
                return TimePhase.MORNING
            elif utc_hour == 12:
                return TimePhase.HIGH_NOON
            elif 13 <= utc_hour <= 19:
                return TimePhase.LATE_NOON
            elif 20 <= utc_hour <= 22:
                return TimePhase.EVENING
            else:  # 23
                return TimePhase.MIDNIGHT_PLUS
                
        except Exception as e:
            logger.error(f"❌ Error getting current phase: {e}")
            return TimePhase.MORNING
    
    def _update_memory_key_outcome(self, position: Position, pnl: float):
        """Update memory key with trade outcome."""
        try:
            # Find corresponding execution
            for execution in self.executions.values():
                if (execution.symbol == position.symbol and 
                    execution.execution_time == position.entry_time):
                    
                    # Update memory key
                    memory_key = execution.signal_id  # Simplified
                    
                    # Determine outcome
                    if pnl > 0:
                        outcome = "win"
                    elif pnl < 0:
                        outcome = "loss"
                    else:
                        outcome = "neutral"
                    
                    # Update memory key in market integration
                    # This would update the memory key system
                    logger.info(f"🧠 Memory key updated: {outcome} (P&L: ${pnl:.2f})")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Error updating memory key outcome: {e}")
    
    def _update_execution_metrics(self, execution: StrategyExecution):
        """Update execution metrics."""
        try:
            # Calculate P&L if position was closed
            if execution.status == ExecutionStatus.COMPLETED:
                # This would be calculated when position is closed
                pass
                
        except Exception as e:
            logger.error(f"❌ Error updating execution metrics: {e}")
    
    def _save_execution(self, execution: StrategyExecution):
        """Save execution to file."""
        try:
            # Create execution file
            file_path = self.execution_path / f"{execution.execution_id}.json"
            
            # Convert to dict
            execution_dict = {
                'execution_id': execution.execution_id,
                'signal_id': execution.signal_id,
                'timestamp': execution.timestamp,
                'symbol': execution.symbol,
                'action': execution.action,
                'price': execution.price,
                'amount': execution.amount,
                'strategy_tier': execution.strategy_tier.value,
                'phase': execution.phase.value,
                'rsi_trigger': execution.rsi_trigger,
                'volume_trigger': execution.volume_trigger,
                'hash_match': execution.hash_match,
                'memory_recall': execution.memory_recall,
                'confidence': execution.confidence,
                'priority': execution.priority,
                'status': execution.status.value,
                'exchange': execution.exchange,
                'order_id': execution.order_id,
                'execution_price': execution.execution_price,
                'execution_time': execution.execution_time,
                'profit_loss': execution.profit_loss,
                'outcome': execution.outcome,
                'error_message': execution.error_message
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(execution_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving execution: {e}")
    
    def _load_historical_executions(self):
        """Load historical executions from files."""
        try:
            for file_path in self.execution_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Create execution object
                    execution = StrategyExecution(
                        execution_id=data['execution_id'],
                        signal_id=data['signal_id'],
                        timestamp=data['timestamp'],
                        symbol=data['symbol'],
                        action=data['action'],
                        price=data['price'],
                        amount=data['amount'],
                        strategy_tier=StrategyTier(data['strategy_tier']),
                        phase=TimePhase(data['phase']),
                        rsi_trigger=data['rsi_trigger'],
                        volume_trigger=data['volume_trigger'],
                        hash_match=data['hash_match'],
                        memory_recall=data['memory_recall'],
                        confidence=data['confidence'],
                        priority=data['priority'],
                        status=ExecutionStatus(data['status']),
                        exchange=data['exchange'],
                        order_id=data.get('order_id'),
                        execution_price=data.get('execution_price'),
                        execution_time=data.get('execution_time'),
                        profit_loss=data.get('profit_loss'),
                        outcome=data.get('outcome'),
                        error_message=data.get('error_message')
                    )
                    
                    self.executions[execution.execution_id] = execution
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error loading execution file {file_path}: {e}")
                    continue
                    
            logger.info(f"📊 Loaded {len(self.executions)} historical executions")
            
        except Exception as e:
            logger.error(f"❌ Error loading historical executions: {e}")
    
    def _load_existing_positions(self):
        """Load existing positions (simplified)."""
        try:
            # In a real implementation, this would load from persistent storage
            # For now, start with empty positions
            pass
            
        except Exception as e:
            logger.error(f"❌ Error loading existing positions: {e}")
    
    def _check_daily_reset(self):
        """Check if daily metrics should be reset."""
        try:
            current_time = time.time()
            
            # Reset daily metrics every 24 hours
            if current_time - self.last_daily_reset > 86400:  # 24 hours
                self.daily_profit_loss = 0.0
                self.daily_loss = 0.0
                self.last_daily_reset = current_time
                
                logger.info("🔄 Daily metrics reset")
                
        except Exception as e:
            logger.error(f"❌ Error checking daily reset: {e}")
    
    def _emergency_stop(self):
        """Execute emergency stop."""
        try:
            logger.critical("🚨 EMERGENCY STOP EXECUTED")
            
            # Close all positions
            for position_id in list(self.positions.keys()):
                self._close_position(position_id, "emergency_stop")
            
            # Stop execution
            self.running = False
            
            # Trigger risk callbacks
            self._trigger_risk_callbacks("emergency_stop")
            
        except Exception as e:
            logger.error(f"❌ Error during emergency stop: {e}")
    
    def _reduce_positions(self):
        """Reduce number of positions."""
        try:
            # Close oldest positions
            positions_list = list(self.positions.items())
            positions_list.sort(key=lambda x: x[1].entry_time)
            
            # Close oldest 20% of positions
            close_count = max(1, len(positions_list) // 5)
            
            for i in range(close_count):
                position_id, position = positions_list[i]
                self._close_position(position_id, "position_reduction")
                
        except Exception as e:
            logger.error(f"❌ Error reducing positions: {e}")
    
    def _trigger_execution_callbacks(self, execution: StrategyExecution):
        """Trigger execution callbacks."""
        try:
            for callback in self.execution_callbacks:
                try:
                    callback(execution)
                except Exception as e:
                    logger.error(f"❌ Error in execution callback: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error triggering execution callbacks: {e}")
    
    def _trigger_risk_callbacks(self, event: str):
        """Trigger risk callbacks."""
        try:
            for callback in self.risk_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"❌ Error in risk callback: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error triggering risk callbacks: {e}")
    
    def add_execution_callback(self, callback: Callable):
        """Add execution callback."""
        self.execution_callbacks.append(callback)
    
    def add_risk_callback(self, callback: Callable):
        """Add risk callback."""
        self.risk_callbacks.append(callback)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get execution engine status."""
        try:
            return {
                'running': self.running,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades),
                'total_profit_loss': self.total_profit_loss,
                'daily_profit_loss': self.daily_profit_loss,
                'daily_loss': self.daily_loss,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'open_positions': len(self.positions),
                'pending_executions': len([e for e in self.executions.values() 
                                         if e.status == ExecutionStatus.PENDING]),
                'active_executions': len([e for e in self.executions.values() 
                                        if e.status == ExecutionStatus.EXECUTING])
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting execution status: {e}")
            return {}
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    def get_executions(self, limit: int = 100) -> List[StrategyExecution]:
        """Get recent executions."""
        try:
            executions = list(self.executions.values())
            executions.sort(key=lambda x: x.timestamp, reverse=True)
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting executions: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the execution engine."""
        try:
            logger.info("🛑 Shutting down Strategy Execution Engine...")
            
            self.running = False
            
            # Wait for threads to finish
            if self.execution_thread:
                self.execution_thread.join(timeout=10.0)
            
            if self.risk_monitor_thread:
                self.risk_monitor_thread.join(timeout=10.0)
            
            # Close all positions
            for position_id in list(self.positions.keys()):
                self._close_position(position_id, "shutdown")
            
            logger.info("✅ Strategy Execution Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for strategy execution engine demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'risk': {
            'max_position_size': 0.1,
            'max_daily_loss': 1000.0,
            'max_open_positions': 10,
            'stop_loss_percentage': 0.05,
            'take_profit_percentage': 0.10,
            'max_risk_per_trade': 0.02,
            'correlation_threshold': 0.7
        }
    }
    
    # Initialize market integration (with mock data for demo)
    market_integration = LiveMarketDataIntegration({})
    
    # Initialize execution engine
    execution_engine = StrategyExecutionEngine(market_integration, config)
    
    try:
        print("🔧 Strategy Execution Engine Demo")
        print("=" * 50)
        
        # Start execution engine
        execution_engine.start_execution()
        
        print("🚀 Execution engine started")
        print("⏳ Monitoring for trading signals...")
        
        # Monitor for 60 seconds
        for i in range(60):
            time.sleep(1)
            
            # Print status every 15 seconds
            if (i + 1) % 15 == 0:
                status = execution_engine.get_execution_status()
                print(f"📊 Status: {status['total_trades']} trades, "
                      f"P&L: ${status['total_profit_loss']:.2f}, "
                      f"Positions: {status['open_positions']}")
        
        # Get final status
        print("\n📈 Final Execution Status:")
        print("-" * 40)
        final_status = execution_engine.get_execution_status()
        for key, value in final_status.items():
            print(f"  {key}: {value}")
        
        # Get positions
        print("\n📊 Current Positions:")
        print("-" * 40)
        positions = execution_engine.get_positions()
        for position in positions:
            print(f"  {position.symbol}: {position.side} {position.amount} "
                  f"@ ${position.entry_price:.2f} (P&L: ${position.unrealized_pnl:.2f})")
        
        # Get recent executions
        print("\n📋 Recent Executions:")
        print("-" * 40)
        executions = execution_engine.get_executions(limit=10)
        for execution in executions:
            print(f"  {execution.symbol}: {execution.action} @ ${execution.price:.2f} "
                  f"({execution.status.value})")
        
    finally:
        # Shutdown
        execution_engine.shutdown()


if __name__ == "__main__":
    main() 