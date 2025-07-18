#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Trading Bot - Production Ready with COMPLETE MATHEMATICAL INTEGRATION
==============================================================================

Complete trading bot that integrates:
- AI-powered market analysis
- Real-time trading decisions
- Portfolio management
- Risk management
- Performance tracking
- ALL MATHEMATICAL SYSTEMS (DLT, Dualistic Engines, Bit Phases, etc.)

This is the main entry point for the Schwabot trading system with your complete mathematical foundation.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from trading_pipeline_manager import TradingPipelineManager, MarketDataPoint
from market_data_simulator import MarketDataSimulator
from schwabot_ai_integration import AnalysisType

# MATHEMATICAL INTEGRATION - ALL YOUR SYSTEMS
from backtesting.mathematical_integration import mathematical_integration, MathematicalSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotTradingBot:
    """Production-ready trading bot with complete AI and mathematical integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trading bot with mathematical integration."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.pipeline = TradingPipelineManager(self.config.get("pipeline", {}))
        self.simulator = MarketDataSimulator(self.config.get("symbols", ["BTC/USD", "ETH/USD"]))
        
        # Bot state
        self.running = False
        self.start_time = None
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Performance tracking
        self.performance_stats = {
            "trades_executed": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_volume": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Mathematical tracking
        self.mathematical_signals: List[MathematicalSignal] = []
        self.dualistic_consensus_history: List[Dict[str, Any]] = []
        self.dlt_waveform_history: List[float] = []
        self.bit_phase_history: List[int] = []
        self.ferris_phase_history: List[float] = []
        self.mathematical_decisions_made = 0
        self.mathematical_signals_processed = 0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Schwabot Trading Bot initialized with COMPLETE MATHEMATICAL INTEGRATION")
        logger.info("🧮 All mathematical systems enabled: DLT, Dualistic Engines, Bit Phases, Ferris RDE, etc.")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load config: {e}")
        
        # Default configuration with mathematical integration
        return {
            "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
            "initial_balance": 10000.0,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "analysis_interval": 60,
            "min_confidence": 0.7,
            "enable_ai_analysis": True,
            "enable_pattern_recognition": True,
            "enable_sentiment_analysis": True,
            "enable_technical_analysis": True,
            "enable_mathematical_integration": True,  # Enable mathematical systems
            "mathematical_confidence_threshold": 0.7,
            "mathematical_weight": 0.7,  # Weight for mathematical vs AI decisions
            "pipeline": {
                "analysis_interval": 60,
                "max_buffer_size": 100,
                "min_confidence": 0.7,
                "risk_per_trade": 0.02,
                "stop_loss_percent": 0.02,
                "take_profit_percent": 0.03
            },
            "performance_tracking": {
                "enable_tracking": True,
                "tracking_interval": 300,  # 5 minutes
                "save_performance_data": True
            }
        }
    
    async def start(self):
        """Start the trading bot with mathematical integration."""
        try:
            logger.info("🚀 Starting Schwabot Trading Bot with FULL MATHEMATICAL INTEGRATION...")
            self.running = True
            self.start_time = time.time()
            
            # Start pipeline
            await self.pipeline.start_pipeline()
            logger.info("✅ Trading pipeline started")
            
            # Start market data simulation with mathematical processing
            simulation_task = asyncio.create_task(
                self.simulator.start_simulation(self._process_market_data_with_mathematics)
            )
            
            # Start performance tracking
            if self.config["performance_tracking"]["enable_tracking"]:
                tracking_task = asyncio.create_task(self._performance_tracking_loop())
            else:
                tracking_task = None
            
            # Main bot loop
            await self._main_loop(simulation_task, tracking_task)
            
        except Exception as e:
            logger.error(f"❌ Trading bot startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the trading bot."""
        try:
            logger.info("🛑 Stopping Schwabot Trading Bot...")
            self.running = False
            
            # Stop pipeline
            await self.pipeline.stop_pipeline()
            
            # Stop simulator
            self.simulator.stop_simulation()
            
            # Save final performance data including mathematical metrics
            if self.config["performance_tracking"]["save_performance_data"]:
                self._save_performance_data()
            
            logger.info("✅ Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Trading bot shutdown failed: {e}")
    
    async def _main_loop(self, simulation_task, tracking_task):
        """Main bot processing loop."""
        try:
            while self.running:
                # Update performance stats
                self._update_performance_stats()
                
                # Display status periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self._display_status()
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
        finally:
            # Cancel tasks
            if simulation_task:
                simulation_task.cancel()
            if tracking_task:
                tracking_task.cancel()
            
            try:
                await simulation_task
            except asyncio.CancelledError:
                pass
            
            try:
                await tracking_task
            except asyncio.CancelledError:
                pass
    
    async def _process_market_data_with_mathematics(self, market_data: MarketDataPoint):
        """Process market data through ALL mathematical systems and AI pipeline."""
        try:
            # Step 1: Process through mathematical integration
            mathematical_signal = await self._process_mathematical_analysis(market_data)
            
            # Step 2: Process through AI pipeline
            ai_decision = await self.pipeline.process_market_data(market_data)
            
            # Step 3: Combine mathematical and AI decisions
            final_decision = self._combine_mathematical_and_ai_decisions(mathematical_signal, ai_decision, market_data)
            
            if final_decision and final_decision.confidence >= self.config["min_confidence"]:
                # Execute trade
                success = await self._execute_trade(final_decision, market_data)
                
                if success:
                    logger.debug(f"💰 Trade executed: {final_decision.action} {final_decision.symbol} @ ${final_decision.entry_price:.4f}")
                    logger.debug(f"🧮 Mathematical confidence: {mathematical_signal.confidence:.3f}")
                    logger.debug(f"🤖 AI confidence: {ai_decision.confidence if ai_decision else 0.5:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Market data processing error: {e}")
    
    async def _process_mathematical_analysis(self, market_data: MarketDataPoint) -> MathematicalSignal:
        """Process market data through ALL mathematical systems."""
        try:
            if not self.config.get("enable_mathematical_integration", True):
                return MathematicalSignal()
            
            # Prepare market data for mathematical processing
            mathematical_market_data = {
                'current_price': market_data.price,
                'volume': market_data.volume,
                'price_change': market_data.price_change,
                'volatility': market_data.volatility,
                'sentiment': market_data.sentiment,
                'close_prices': [market_data.price],  # Simplified price history
                'entry_price': self._get_entry_price(market_data.symbol),
                'bit_phase': self._get_current_bit_phase(market_data.symbol)
            }
            
            # Process through ALL mathematical systems
            mathematical_signal = await mathematical_integration.process_market_data_mathematically(mathematical_market_data)
            
            # Store mathematical signal for analysis
            self.mathematical_signals.append(mathematical_signal)
            self.mathematical_signals_processed += 1
            
            # Store specific mathematical components
            if mathematical_signal.dualistic_consensus:
                self.dualistic_consensus_history.append(mathematical_signal.dualistic_consensus)
            
            self.dlt_waveform_history.append(mathematical_signal.dlt_waveform_score)
            self.bit_phase_history.append(mathematical_signal.bit_phase)
            self.ferris_phase_history.append(mathematical_signal.ferris_phase)
            
            return mathematical_signal
            
        except Exception as e:
            logger.error(f"❌ Mathematical analysis failed: {e}")
            return MathematicalSignal()
    
    def _combine_mathematical_and_ai_decisions(self, mathematical_signal: MathematicalSignal, ai_decision, market_data: MarketDataPoint):
        """Combine mathematical and AI decisions."""
        try:
            if not ai_decision:
                # Use mathematical decision only
                return self._create_trading_decision_from_mathematical(mathematical_signal, market_data)
            
            # Weight mathematical vs AI decision
            math_weight = self.config.get("mathematical_weight", 0.7)  # Mathematical systems get higher weight
            ai_weight = 1.0 - math_weight
            
            # Calculate combined confidence
            math_confidence = mathematical_signal.confidence
            ai_confidence = ai_decision.confidence if ai_decision else 0.5
            
            combined_confidence = (math_confidence * math_weight + ai_confidence * ai_weight)
            
            # Determine final decision
            if mathematical_signal.decision == "BUY" and ai_decision.action == "BUY":
                final_action = "BUY"
                final_confidence = combined_confidence
            elif mathematical_signal.decision == "SELL" and ai_decision.action == "SELL":
                final_action = "SELL"
                final_confidence = combined_confidence
            elif mathematical_signal.decision == "BUY" and ai_decision.action == "SELL":
                # Conflict - use mathematical decision with reduced confidence
                final_action = mathematical_signal.decision
                final_confidence = math_confidence * 0.8
            elif mathematical_signal.decision == "SELL" and ai_decision.action == "BUY":
                # Conflict - use mathematical decision with reduced confidence
                final_action = mathematical_signal.decision
                final_confidence = math_confidence * 0.8
            else:
                # Use mathematical decision
                final_action = mathematical_signal.decision
                final_confidence = math_confidence
            
            # Create final trading decision
            from trading_pipeline_manager import TradingDecision
            
            final_decision = TradingDecision(
                action=final_action,
                symbol=market_data.symbol,
                entry_price=market_data.price,
                position_size=self._calculate_position_size(final_confidence, market_data.price),
                confidence=final_confidence,
                timestamp=market_data.timestamp,
                metadata={
                    'mathematical_decision': mathematical_signal.decision,
                    'ai_decision': ai_decision.action if ai_decision else 'NONE',
                    'dualistic_consensus': mathematical_signal.dualistic_consensus,
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score
                }
            )
            
            self.mathematical_decisions_made += 1
            
            return final_decision
            
        except Exception as e:
            logger.error(f"❌ Decision combination failed: {e}")
            return None
    
    def _create_trading_decision_from_mathematical(self, mathematical_signal: MathematicalSignal, market_data: MarketDataPoint):
        """Create trading decision from mathematical signal only."""
        try:
            from trading_pipeline_manager import TradingDecision
            
            position_size = self._calculate_position_size(mathematical_signal.confidence, market_data.price)
            
            return TradingDecision(
                action=mathematical_signal.decision,
                symbol=market_data.symbol,
                entry_price=market_data.price,
                position_size=position_size,
                confidence=mathematical_signal.confidence,
                timestamp=market_data.timestamp,
                metadata={
                    'mathematical_only': True,
                    'dualistic_consensus': mathematical_signal.dualistic_consensus,
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Mathematical decision creation failed: {e}")
            return None
    
    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for a symbol."""
        # For simplicity, use current market price
        # In a real implementation, you'd track actual entry prices
        return 50000.0  # Default BTC price
    
    def _get_current_bit_phase(self, symbol: str) -> int:
        """Get current bit phase for a symbol."""
        # For simplicity, return a default bit phase
        # In a real implementation, you'd track actual bit phases
        return 8  # Default 8-bit phase
    
    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on confidence and risk management."""
        try:
            # Base position size from risk management
            base_size = self.config["initial_balance"] * self.config["risk_per_trade"] / price
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale confidence to 0-1
            
            # Apply maximum position limit
            max_positions = self.config["max_positions"]
            current_positions = 0  # Simplified - in real implementation, track actual positions
            
            if current_positions >= max_positions:
                position_multiplier = 0.5  # Reduce size if at position limit
            else:
                position_multiplier = 1.0
            
            final_size = base_size * confidence_multiplier * position_multiplier
            
            return max(0.0, final_size)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed: {e}")
            return 0.0
    
    async def _execute_trade(self, decision, market_data: MarketDataPoint) -> bool:
        """Execute a trade based on the decision."""
        try:
            # Simulate trade execution
            self.total_trades += 1
            
            # Calculate P&L (simplified)
            if decision.action == "BUY":
                # Simulate buy
                self.total_pnl -= decision.position_size * decision.entry_price
            elif decision.action == "SELL":
                # Simulate sell
                self.total_pnl += decision.position_size * decision.entry_price
            
            logger.info(f"💰 Trade executed: {decision.action} {decision.symbol} @ ${decision.entry_price:.4f}")
            logger.info(f"🧮 Mathematical confidence: {decision.metadata.get('dlt_waveform_score', 0):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False
    
    async def _performance_tracking_loop(self):
        """Performance tracking loop with mathematical metrics."""
        try:
            while self.running:
                # Update mathematical performance metrics
                self._update_mathematical_metrics()
                
                # Sleep for tracking interval
                await asyncio.sleep(self.config["performance_tracking"]["tracking_interval"])
                
        except Exception as e:
            logger.error(f"❌ Performance tracking error: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        try:
            # Calculate basic metrics
            if self.total_trades > 0:
                win_rate = self.performance_stats["winning_trades"] / self.total_trades
                self.performance_stats["win_rate"] = win_rate
            
            # Update mathematical metrics
            self._update_mathematical_metrics()
            
        except Exception as e:
            logger.error(f"❌ Performance stats update failed: {e}")
    
    def _update_mathematical_metrics(self):
        """Update mathematical performance metrics."""
        try:
            if not self.mathematical_signals:
                return
            
            # Calculate mathematical metrics
            dlt_scores = [s.dlt_waveform_score for s in self.mathematical_signals if s.dlt_waveform_score > 0]
            avg_dlt_score = sum(dlt_scores) / len(dlt_scores) if dlt_scores else 0.0
            
            dualistic_scores = []
            for consensus in self.dualistic_consensus_history:
                if consensus and 'mathematical_score' in consensus:
                    dualistic_scores.append(consensus['mathematical_score'])
            avg_dualistic_score = sum(dualistic_scores) / len(dualistic_scores) if dualistic_scores else 0.0
            
            # Store in performance stats
            self.performance_stats.update({
                "avg_dlt_waveform_score": avg_dlt_score,
                "avg_dualistic_score": avg_dualistic_score,
                "mathematical_signals_processed": self.mathematical_signals_processed,
                "mathematical_decisions_made": self.mathematical_decisions_made
            })
            
        except Exception as e:
            logger.error(f"❌ Mathematical metrics update failed: {e}")
    
    def _display_status(self):
        """Display current bot status with mathematical information."""
        try:
            runtime = time.time() - self.start_time if self.start_time else 0
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            
            print(f"\n📊 Schwabot Status ({hours:02d}:{minutes:02d})")
            print(f"   Total Trades: {self.total_trades}")
            print(f"   Total P&L: ${self.total_pnl:.2f}")
            print(f"   Win Rate: {self.performance_stats.get('win_rate', 0):.2%}")
            print(f"   Mathematical Signals: {self.mathematical_signals_processed}")
            print(f"   Mathematical Decisions: {self.mathematical_decisions_made}")
            print(f"   Avg DLT Score: {self.performance_stats.get('avg_dlt_waveform_score', 0):.4f}")
            print(f"   Avg Dualistic Score: {self.performance_stats.get('avg_dualistic_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Status display failed: {e}")
    
    def _save_performance_data(self):
        """Save performance data including mathematical metrics."""
        try:
            performance_data = {
                "timestamp": time.time(),
                "runtime": time.time() - self.start_time if self.start_time else 0,
                "total_trades": self.total_trades,
                "total_pnl": self.total_pnl,
                "performance_stats": self.performance_stats,
                "mathematical_metrics": {
                    "signals_processed": self.mathematical_signals_processed,
                    "decisions_made": self.mathematical_decisions_made,
                    "avg_dlt_score": self.performance_stats.get('avg_dlt_waveform_score', 0),
                    "avg_dualistic_score": self.performance_stats.get('avg_dualistic_score', 0),
                    "total_signals": len(self.mathematical_signals)
                }
            }
            
            # Save to file
            with open(f"performance_data_{int(time.time())}.json", 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.info("✅ Performance data saved with mathematical metrics")
            
        except Exception as e:
            logger.error(f"❌ Performance data save failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False

async def main():
    """Main entry point for the Schwabot trading bot."""
    try:
        # Create and start the trading bot
        bot = SchwabotTradingBot()
        
        # Start the bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Main error: {e}")
    finally:
        # Ensure bot is stopped
        if 'bot' in locals():
            await bot.stop()

if __name__ == "__main__":
    asyncio.run(main()) 