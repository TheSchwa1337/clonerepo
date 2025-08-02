#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Schwabot Integrated System - Phase IV Complete
=================================================

Complete integration of Clock Mode System + Neural Core:
- Mechanical timing precision + Neural decision making
- Recursive cycles with feedback loops
- Real-time market analysis and trading decisions
- Continuous learning and adaptation
- Safety-first execution with multiple layers

This is the complete Schwabot trading system that can run continuously.
"""

import sys
import math
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import random
import os

# Import our systems
from clock_mode_system import ClockModeSystem, SAFETY_CONFIG
from schwabot_neural_core import SchwabotNeuralCore, MarketData, TradingDecision, DecisionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_integrated_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SchwabotIntegratedSystem:
    """Complete Schwabot trading system integrating clock and neural components."""
    
    def __init__(self):
        # Initialize subsystems
        self.clock_system = ClockModeSystem()
        self.neural_core = SchwabotNeuralCore()
        
        # System state
        self.is_running = False
        self.cycle_count = 0
        self.last_decision_time = 0.0
        self.current_balances = {
            'btc': 0.2,
            'usdc': 10000.0
        }
        
        # Performance tracking
        self.total_profit = 0.0
        self.trade_history = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Integration thread
        self.integration_thread = None
        
        logger.info("🤖 Schwabot Integrated System initialized")
    
    def start_system(self) -> bool:
        """Start the complete Schwabot system."""
        if self.is_running:
            logger.warning("System already running")
            return False
        
        # Start clock mode system
        if not self.clock_system.start_clock_mode():
            logger.error("❌ Failed to start clock mode system")
            return False
        
        # Start integration thread
        self.is_running = True
        self.integration_thread = threading.Thread(
            target=self._integration_loop,
            daemon=True
        )
        self.integration_thread.start()
        
        logger.info("🤖 Schwabot Integrated System started")
        return True
    
    def stop_system(self) -> bool:
        """Stop the complete Schwabot system."""
        self.is_running = False
        
        # Stop clock mode system
        self.clock_system.stop_clock_mode()
        
        # Wait for integration thread
        if self.integration_thread:
            self.integration_thread.join(timeout=10.0)
        
        logger.info("🤖 Schwabot Integrated System stopped")
        return True
    
    def _integration_loop(self) -> None:
        """Main integration loop combining clock timing with neural decisions."""
        while self.is_running:
            try:
                # Get clock mechanism status
                clock_status = self.clock_system.get_all_mechanisms_status()
                
                # Extract market data from clock system
                market_data = self._extract_market_data(clock_status)
                
                # Make neural decision
                decision = self.neural_core.make_decision(market_data)
                
                # Execute decision if conditions are met
                if self._should_execute_decision(decision, clock_status):
                    self._execute_decision(decision)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log cycle information
                self._log_cycle_info(clock_status, decision)
                
                # Increment cycle count
                self.cycle_count += 1
                
                # Sleep based on clock timing
                sleep_time = self._calculate_sleep_time(clock_status)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in integration loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _extract_market_data(self, clock_status: Dict[str, Any]) -> MarketData:
        """Extract market data from clock system status."""
        # Get the first mechanism's data (simplified)
        mechanisms = clock_status.get("mechanisms", {})
        if not mechanisms:
            # Fallback to simulated data
            return self._create_simulated_market_data()
        
        # Extract data from the first available mechanism
        mechanism_id = list(mechanisms.keys())[0]
        mechanism_data = mechanisms[mechanism_id]
        
        # Get market data from cache or create simulated
        market_cache = self.clock_system.market_data_cache.get(mechanism_id, {})
        
        # Create MarketData object
        market_data = MarketData(
            timestamp=datetime.now(),
            btc_price=market_cache.get('price', 50000.0),
            usdc_balance=self.current_balances['usdc'],
            btc_balance=self.current_balances['btc'],
            price_change=market_cache.get('price_change', 0.0),
            volume=market_cache.get('volume', 5000.0),
            rsi_14=45.0,  # Would calculate from historical data
            rsi_21=50.0,
            rsi_50=55.0,
            market_phase=mechanism_data.get('market_phase', 0.0),
            hash_timing=self._generate_hash_timing(),
            orbital_phase=0.5
        )
        
        return market_data
    
    def _create_simulated_market_data(self) -> MarketData:
        """Create simulated market data for testing."""
        return MarketData(
            timestamp=datetime.now(),
            btc_price=50000.0 + random.uniform(-1000, 1000),
            usdc_balance=self.current_balances['usdc'],
            btc_balance=self.current_balances['btc'],
            price_change=random.uniform(-0.05, 0.05),
            volume=random.uniform(1000, 10000),
            rsi_14=random.uniform(30, 70),
            rsi_21=random.uniform(30, 70),
            rsi_50=random.uniform(30, 70),
            market_phase=random.uniform(0, 2 * math.pi),
            hash_timing=self._generate_hash_timing(),
            orbital_phase=random.uniform(0, 1)
        )
    
    def _generate_hash_timing(self) -> str:
        """Generate hash timing for market data."""
        hash_input = f"{time.time()}:{self.cycle_count}:{random.random()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _should_execute_decision(self, decision: TradingDecision, clock_status: Dict[str, Any]) -> bool:
        """Determine if a decision should be executed."""
        # Check safety conditions
        if SAFETY_CONFIG.execution_mode.value == "shadow":
            logger.info("🛡️ SHADOW MODE - Decision would be: " + decision.decision_type.value)
            return False
        
        # Check confidence threshold
        if decision.confidence < SAFETY_CONFIG.min_confidence_threshold:
            logger.info(f"🛡️ Low confidence ({decision.confidence:.3f}) - skipping execution")
            return False
        
        # Check risk level
        if decision.risk_level == "high":
            logger.warning("⚠️ High risk decision - skipping execution")
            return False
        
        # Check timing (don't execute too frequently)
        current_time = time.time()
        if current_time - self.last_decision_time < 60:  # 1 minute minimum
            return False
        
        # Check if we have sufficient balance
        if decision.decision_type == DecisionType.BUY and self.current_balances['usdc'] < 100:
            logger.info("🛡️ Insufficient USDC for buy")
            return False
        
        if decision.decision_type == DecisionType.SELL and self.current_balances['btc'] < 0.001:
            logger.info("🛡️ Insufficient BTC for sell")
            return False
        
        return True
    
    def _execute_decision(self, decision: TradingDecision) -> None:
        """Execute a trading decision."""
        try:
            # Simulate trade execution
            if decision.decision_type == DecisionType.BUY:
                # Buy BTC with USDC
                trade_amount_usdc = min(self.current_balances['usdc'] * 0.1, 1000)  # 10% or $1000 max
                btc_received = trade_amount_usdc / decision.market_data.btc_price
                
                self.current_balances['usdc'] -= trade_amount_usdc
                self.current_balances['btc'] += btc_received
                
                logger.info(f"💰 BUY: {btc_received:.6f} BTC for ${trade_amount_usdc:.2f}")
                
            elif decision.decision_type == DecisionType.SELL:
                # Sell BTC for USDC
                trade_amount_btc = min(self.current_balances['btc'] * 0.1, 0.01)  # 10% or 0.01 BTC max
                usdc_received = trade_amount_btc * decision.market_data.btc_price
                
                self.current_balances['btc'] -= trade_amount_btc
                self.current_balances['usdc'] += usdc_received
                
                logger.info(f"💰 SELL: {trade_amount_btc:.6f} BTC for ${usdc_received:.2f}")
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'decision': decision.decision_type.value,
                'confidence': decision.confidence,
                'btc_balance': self.current_balances['btc'],
                'usdc_balance': self.current_balances['usdc'],
                'btc_price': decision.market_data.btc_price,
                'expected_profit': decision.expected_profit
            }
            
            self.trade_history.append(trade_record)
            self.last_decision_time = time.time()
            
            # Calculate actual profit (simplified)
            total_value = (self.current_balances['btc'] * decision.market_data.btc_price) + self.current_balances['usdc']
            actual_profit = total_value - 10000.0  # Assuming starting with $10,000
            
            # Learn from outcome
            self.neural_core.learn_from_outcome(decision, actual_profit)
            
        except Exception as e:
            logger.error(f"❌ Error executing decision: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        if len(self.trade_history) < 2:
            return
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in self.trade_history if trade.get('expected_profit', 0) > 0)
        self.performance_metrics['win_rate'] = profitable_trades / len(self.trade_history)
        
        # Calculate average profit
        total_profit = sum(trade.get('expected_profit', 0) for trade in self.trade_history)
        self.performance_metrics['avg_profit'] = total_profit / len(self.trade_history)
        
        # Calculate max drawdown (simplified)
        balances = [trade['btc_balance'] * trade['btc_price'] + trade['usdc_balance'] for trade in self.trade_history]
        peak = max(balances)
        current = balances[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        self.performance_metrics['max_drawdown'] = max(self.performance_metrics['max_drawdown'], drawdown)
    
    def _calculate_sleep_time(self, clock_status: Dict[str, Any]) -> float:
        """Calculate sleep time based on clock system timing."""
        # Get timing from clock mechanism
        mechanisms = clock_status.get("mechanisms", {})
        if mechanisms:
            mechanism_id = list(mechanisms.keys())[0]
            mechanism_data = mechanisms[mechanism_id]
            escapement_timing = mechanism_data.get('escapement_timing', 1.0)
            return min(max(escapement_timing, 0.5), 10.0)  # Between 0.5 and 10 seconds
        
        return 2.0  # Default 2 seconds
    
    def _log_cycle_info(self, clock_status: Dict[str, Any], decision: TradingDecision) -> None:
        """Log cycle information."""
        if self.cycle_count % 10 == 0:  # Log every 10 cycles
            neural_stats = self.neural_core.get_neural_stats()
            
            log_message = f"🤖 Cycle {self.cycle_count} - " \
                         f"Decision: {decision.decision_type.value} " \
                         f"(conf: {decision.confidence:.3f}) - " \
                         f"BTC: {self.current_balances['btc']:.6f}, " \
                         f"USDC: ${self.current_balances['usdc']:.2f} - " \
                         f"Success Rate: {neural_stats['success_rate']:.3f}"
            
            logger.info(log_message)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        clock_status = self.clock_system.get_all_mechanisms_status()
        neural_stats = self.neural_core.get_neural_stats()
        
        return {
            "system_status": {
                "is_running": self.is_running,
                "cycle_count": self.cycle_count,
                "last_decision_time": self.last_decision_time
            },
            "balances": self.current_balances,
            "performance": self.performance_metrics,
            "trade_history_count": len(self.trade_history),
            "clock_system": clock_status,
            "neural_core": neural_stats,
            "safety_config": {
                "execution_mode": SAFETY_CONFIG.execution_mode.value,
                "max_position_size": SAFETY_CONFIG.max_position_size,
                "max_daily_loss": SAFETY_CONFIG.max_daily_loss,
                "emergency_stop_enabled": SAFETY_CONFIG.emergency_stop_enabled
            }
        }
    
    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        return self.trade_history[-count:] if self.trade_history else []
    
    def reset_system(self) -> None:
        """Reset the system to initial state."""
        # Stop if running
        if self.is_running:
            self.stop_system()
        
        # Reset balances
        self.current_balances = {
            'btc': 0.2,
            'usdc': 10000.0
        }
        
        # Reset performance
        self.total_profit = 0.0
        self.trade_history.clear()
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Reset neural core
        self.neural_core.reset_learning()
        
        # Reset cycle count
        self.cycle_count = 0
        
        logger.info("🤖 System reset to initial state")

def main():
    """Test the complete Schwabot Integrated System."""
    logger.info("🤖 Starting Schwabot Integrated System Test")
    
    # Create integrated system
    schwabot = SchwabotIntegratedSystem()
    
    # Start system
    if not schwabot.start_system():
        logger.error("❌ Failed to start Schwabot system")
        return
    
    # Run for a few minutes to see results
    logger.info("🤖 Running Schwabot system for 2 minutes...")
    time.sleep(120)  # 2 minutes
    
    # Get final status
    status = schwabot.get_system_status()
    logger.info(f"🤖 Final Status: {json.dumps(status, indent=2)}")
    
    # Get recent trades
    recent_trades = schwabot.get_recent_trades(5)
    logger.info(f"🤖 Recent Trades: {json.dumps(recent_trades, indent=2)}")
    
    # Stop system
    schwabot.stop_system()
    
    logger.info("🤖 Schwabot Integrated System Test Complete")

if __name__ == "__main__":
    main() 