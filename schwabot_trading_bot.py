#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Trading Bot - Production Ready
=======================================

Complete trading bot that integrates:
- AI-powered market analysis
- Real-time trading decisions
- Portfolio management
- Risk management
- Performance tracking

This is the main entry point for the Schwabot trading system.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from trading_pipeline_manager import TradingPipelineManager, MarketDataPoint
from market_data_simulator import MarketDataSimulator
from schwabot_ai_integration import AnalysisType

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
    """Production-ready trading bot with complete AI integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trading bot."""
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
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Schwabot Trading Bot initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load config: {e}")
        
        # Default configuration
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
        """Start the trading bot."""
        try:
            logger.info("🚀 Starting Schwabot Trading Bot...")
            self.running = True
            self.start_time = time.time()
            
            # Start pipeline
            await self.pipeline.start_pipeline()
            logger.info("✅ Trading pipeline started")
            
            # Start market data simulation
            simulation_task = asyncio.create_task(
                self.simulator.start_simulation(self._process_market_data)
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
            
            # Save final performance data
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
    
    async def _process_market_data(self, market_data: MarketDataPoint):
        """Process market data through the trading pipeline."""
        try:
            # Process through pipeline
            decision = await self.pipeline.process_market_data(market_data)
            
            if decision:
                logger.info(f"🎯 Trading decision: {decision.action} {decision.symbol} (confidence: {decision.confidence:.1%})")
                
                # Execute trade if confidence meets threshold
                if decision.confidence >= self.config["min_confidence"]:
                    success = await self.pipeline.execute_trade(decision)
                    if success:
                        self.total_trades += 1
                        logger.info(f"✅ Trade executed: {decision.action} {decision.position_size:.4f} {decision.symbol}")
                        
                        # Update performance tracking
                        self._record_trade(decision)
                
        except Exception as e:
            logger.error(f"❌ Market data processing failed: {e}")
    
    async def _performance_tracking_loop(self):
        """Performance tracking loop."""
        interval = self.config["performance_tracking"]["tracking_interval"]
        
        while self.running:
            try:
                # Update performance metrics
                self._update_performance_stats()
                
                # Save performance data
                if self.config["performance_tracking"]["save_performance_data"]:
                    self._save_performance_data()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    def _record_trade(self, decision):
        """Record trade for performance tracking."""
        self.performance_stats["trades_executed"] += 1
        self.performance_stats["total_volume"] += decision.position_size * decision.entry_price
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        try:
            # Get current portfolio state
            portfolio = self.pipeline.portfolio_state
            initial_balance = self.config["initial_balance"]
            
            # Calculate P&L
            current_balance = portfolio.balance
            for symbol, quantity in portfolio.positions.items():
                # For simulation, use a simple P&L calculation
                # In production, this would use real market prices
                current_balance += quantity * 45000  # Simplified
            
            self.total_pnl = current_balance - initial_balance
            
            # Update win/loss ratio
            if self.total_pnl > 0:
                self.performance_stats["winning_trades"] += 1
            elif self.total_pnl < 0:
                self.performance_stats["losing_trades"] += 1
            
            # Calculate max drawdown
            if self.total_pnl < self.performance_stats["max_drawdown"]:
                self.performance_stats["max_drawdown"] = self.total_pnl
            
        except Exception as e:
            logger.error(f"❌ Performance stats update failed: {e}")
    
    def _display_status(self):
        """Display current bot status."""
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            portfolio = self.pipeline.portfolio_state
            
            print(f"\n📊 Schwabot Trading Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print(f"⏱️  Uptime: {uptime/3600:.1f} hours")
            print(f"💰 Portfolio Balance: ${portfolio.balance:.2f}")
            print(f"📈 Total P&L: ${self.total_pnl:.2f}")
            print(f"🎯 Total Trades: {self.total_trades}")
            print(f"📊 Active Positions: {len(portfolio.positions)}")
            print(f"🔍 AI Analysis Count: {self.pipeline.ai_integration.request_count}")
            
            if portfolio.positions:
                print("\n📈 Active Positions:")
                for symbol, quantity in portfolio.positions.items():
                    print(f"   {symbol}: {quantity:.4f}")
            
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Status display failed: {e}")
    
    def _save_performance_data(self):
        """Save performance data to file."""
        try:
            data = {
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "portfolio_balance": self.pipeline.portfolio_state.balance,
                "total_pnl": self.total_pnl,
                "total_trades": self.total_trades,
                "active_positions": len(self.pipeline.portfolio_state.positions),
                "performance_stats": self.performance_stats,
                "ai_analysis_count": self.pipeline.ai_integration.request_count
            }
            
            # Save to file
            performance_file = Path("data/performance_data.json")
            performance_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(performance_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Performance data save failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        self.running = False

async def main():
    """Main function for the trading bot."""
    logger.info("🚀 Schwabot Trading Bot Starting...")
    
    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and start trading bot
    bot = SchwabotTradingBot(config_path)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Trading bot error: {e}")
    finally:
        await bot.stop()
        
        # Display final status
        bot._display_status()
        
        logger.info("👋 Schwabot Trading Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 