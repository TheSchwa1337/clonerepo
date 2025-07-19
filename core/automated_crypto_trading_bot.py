"""
🤖 AUTOMATED CRYPTO TRADING BOT - SCHWABOT COMPLETE INTEGRATION
==============================================================

Complete automated crypto trading bot that integrates ALL existing Schwabot components
into a unified system for automated portfolio trading and rebalancing.

Features:
- Automated portfolio rebalancing with multiple strategies
- Real-time stop-loss and take-profit management
- Multi-exchange trading with smart order routing
- Advanced mathematical decision making
- Risk management with circuit breakers
- Performance monitoring and analytics
- Complete automation with minimal human intervention

This bot brings together all the sophisticated mathematical systems you've built
into a production-ready automated trading solution.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import numpy as np

# Import all existing Schwabot components
from .enhanced_portfolio_tracker import EnhancedPortfolioTracker, RebalancingAction
from .algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer
from .live_trading_system import LiveTradingSystem, TradingConfig
from .risk_manager import RiskManager
from .real_time_market_data_pipeline import RealTimeMarketDataPipeline
from .ensemble_decision_making_system import EnsembleDecisionMakingSystem
from .advanced_strategy_execution_engine import AdvancedStrategyExecutionEngine
from .btc_usdc_trading_integration import BTCUSDCTradingIntegration
from .profit_optimization_engine import PortfolioOptimizationEngine
from .bio_profit_vectorization import BioProfitVectorization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading modes for the automated bot."""
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"

class BotState(Enum):
    """Bot operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class AutomatedBotConfig:
    """Configuration for the automated trading bot."""
    # Trading mode
    trading_mode: TradingMode = TradingMode.DEMO
    
    # Portfolio settings
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    
    # Risk management
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.05  # 5%
    max_daily_loss: float = 0.05  # 5%
    max_drawdown: float = 0.15  # 15%
    
    # Rebalancing settings
    rebalancing_enabled: bool = True
    rebalancing_threshold: float = 0.05  # 5% deviation
    rebalancing_interval: int = 3600  # 1 hour
    target_allocation: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.4,
        "ETH": 0.3,
        "USDC": 0.3
    })
    
    # Trading pairs
    trading_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USDC", "ETH/USDC", "SOL/USDC"
    ])
    
    # Exchange settings
    exchanges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Mathematical settings
    math_confidence_threshold: float = 0.7
    math_risk_threshold: float = 0.8
    
    # Execution settings
    execution_timeout: int = 30
    max_retry_attempts: int = 3
    slippage_tolerance: float = 0.001  # 0.1%
    
    # Monitoring
    performance_tracking: bool = True
    enable_alerts: bool = True
    log_level: str = "INFO"

@dataclass
class BotPerformance:
    """Bot performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    average_trade_duration: float = 0.0
    rebalancing_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

class AutomatedCryptoTradingBot:
    """
    Complete automated crypto trading bot integrating all Schwabot components.
    
    This bot provides:
    - Automated portfolio rebalancing
    - Real-time stop-loss and take-profit management
    - Multi-exchange trading with smart order routing
    - Advanced mathematical decision making
    - Comprehensive risk management
    - Performance monitoring and analytics
    """
    
    def __init__(self, config: AutomatedBotConfig):
        """Initialize the automated trading bot."""
        self.config = config
        self.state = BotState.INITIALIZING
        self.performance = BotPerformance()
        
        # Initialize all components
        self._initialize_components()
        
        # Trading state
        self.active_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.rebalancing_history: List[Dict] = []
        
        # Callbacks
        self.trade_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
        # Control flags
        self.is_running = False
        self.emergency_stop_triggered = False
        
        logger.info("🤖 Automated Crypto Trading Bot initialized")
    
    def _initialize_components(self):
        """Initialize all trading system components."""
        try:
            # Initialize market data pipeline
            market_data_config = {
                'symbols': self.config.trading_pairs,
                'exchanges': self.config.exchanges,
                'cache_ttl': 60,
                'max_price_history': 2000,
                'quality_threshold': 0.5
            }
            self.market_data_pipeline = RealTimeMarketDataPipeline(market_data_config)
            
            # Initialize portfolio tracker
            portfolio_config = {
                'exchanges': self.config.exchanges,
                'tracked_symbols': self.config.trading_pairs,
                'price_update_interval': 5,
                'rebalancing': {
                    'enabled': self.config.rebalancing_enabled,
                    'threshold': self.config.rebalancing_threshold,
                    'interval': self.config.rebalancing_interval,
                    'target_allocation': self.config.target_allocation
                }
            }
            self.portfolio_tracker = EnhancedPortfolioTracker(portfolio_config)
            
            # Initialize portfolio balancer
            balancer_config = {
                'rebalancing_strategy': 'phantom_adaptive',
                'rebalance_threshold': self.config.rebalancing_threshold,
                'max_rebalance_frequency': self.config.rebalancing_interval,
                'risk_free_rate': 0.02
            }
            self.portfolio_balancer = AlgorithmicPortfolioBalancer(balancer_config)
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                risk_tolerance=self.config.max_daily_loss,
                max_portfolio_risk=self.config.max_drawdown
            )
            
            # Initialize trading system
            trading_config = TradingConfig(
                exchanges=self.config.exchanges,
                tracked_symbols=self.config.trading_pairs,
                price_update_interval=5,
                rebalancing_enabled=self.config.rebalancing_enabled,
                rebalancing_threshold=self.config.rebalancing_threshold,
                rebalancing_interval=self.config.rebalancing_interval,
                target_allocation=self.config.target_allocation,
                max_position_size=self.config.max_position_size,
                max_daily_trades=100,
                stop_loss_percentage=self.config.stop_loss_percentage,
                take_profit_percentage=self.config.take_profit_percentage,
                math_decision_enabled=True,
                math_confidence_threshold=self.config.math_confidence_threshold,
                math_risk_threshold=self.config.math_risk_threshold,
                live_trading_enabled=(self.config.trading_mode == TradingMode.LIVE),
                sandbox_mode=(self.config.trading_mode != TradingMode.LIVE),
                max_slippage=self.config.slippage_tolerance,
                enable_logging=True,
                enable_alerts=self.config.enable_alerts,
                performance_tracking=self.config.performance_tracking
            )
            self.trading_system = LiveTradingSystem(trading_config)
            
            # Initialize mathematical components
            self.ensemble_system = EnsembleDecisionMakingSystem()
            self.portfolio_optimizer = PortfolioOptimizationEngine()
            self.bio_profit_system = BioProfitVectorization()
            
            # Initialize execution engine
            self.execution_engine = AdvancedStrategyExecutionEngine(
                trading_integration=None,  # Will be set during startup
                ensemble_system=self.ensemble_system,
                market_data_pipeline=self.market_data_pipeline,
                risk_system=self.risk_manager,
                portfolio_engine=self.portfolio_optimizer
            )
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            self.state = BotState.ERROR
            raise
    
    async def start(self):
        """Start the automated trading bot."""
        if self.is_running:
            logger.warning("Bot is already running")
            return
        
        try:
            logger.info("🚀 Starting Automated Crypto Trading Bot...")
            self.state = BotState.INITIALIZING
            self.is_running = True
            
            # Start all components
            await self._start_components()
            
            # Setup callbacks
            self._setup_callbacks()
            
            # Start main trading loop
            asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            # Start rebalancing loop
            if self.config.rebalancing_enabled:
                asyncio.create_task(self._rebalancing_loop())
            
            self.state = BotState.RUNNING
            logger.info("✅ Automated Crypto Trading Bot started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            self.state = BotState.ERROR
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the automated trading bot."""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping Automated Crypto Trading Bot...")
        self.state = BotState.STOPPING
        self.is_running = False
        
        try:
            # Stop all components
            await self._stop_components()
            
            # Close all positions if in live mode
            if self.config.trading_mode == TradingMode.LIVE:
                await self._close_all_positions()
            
            logger.info("✅ Automated Crypto Trading Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping bot: {e}")
    
    async def emergency_stop(self):
        """Emergency stop - immediately close all positions and stop trading."""
        logger.warning("🚨 EMERGENCY STOP TRIGGERED")
        self.emergency_stop_triggered = True
        self.state = BotState.EMERGENCY_STOP
        
        try:
            # Immediately close all positions
            await self._close_all_positions()
            
            # Stop the bot
            await self.stop()
            
            logger.info("✅ Emergency stop completed")
            
        except Exception as e:
            logger.error(f"❌ Error during emergency stop: {e}")
    
    async def _start_components(self):
        """Start all trading components."""
        # Start market data pipeline
        await self.market_data_pipeline.start()
        
        # Start portfolio tracker
        await self.portfolio_tracker.start()
        
        # Start trading system
        await self.trading_system.start()
        
        # Initialize portfolio state
        await self._initialize_portfolio_state()
    
    async def _stop_components(self):
        """Stop all trading components."""
        # Stop trading system
        if hasattr(self, 'trading_system'):
            await self.trading_system.stop()
        
        # Stop portfolio tracker
        if hasattr(self, 'portfolio_tracker'):
            await self.portfolio_tracker.stop()
        
        # Stop market data pipeline
        if hasattr(self, 'market_data_pipeline'):
            await self.market_data_pipeline.stop()
    
    async def _initialize_portfolio_state(self):
        """Initialize portfolio state with current balances."""
        try:
            # Sync with exchanges
            await self.portfolio_tracker.sync_with_exchanges()
            
            # Update portfolio balancer state
            portfolio_summary = self.portfolio_tracker.get_enhanced_summary()
            await self.portfolio_balancer.update_portfolio_state({
                'balances': portfolio_summary.get('balances', {}),
                'total_value': portfolio_summary.get('total_value', 0)
            })
            
            logger.info("✅ Portfolio state initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize portfolio state: {e}")
    
    def _setup_callbacks(self):
        """Setup callbacks for various events."""
        # Portfolio tracker callbacks
        self.portfolio_tracker.add_rebalancing_callback(self._on_rebalancing_event)
        self.portfolio_tracker.add_price_update_callback(self._on_price_update)
        
        # Trading system callbacks
        self.trading_system.add_trade_callback(self._on_trade_event)
        self.trading_system.add_alert_callback(self._on_alert_event)
        self.trading_system.add_performance_callback(self._on_performance_event)
    
    async def _main_trading_loop(self):
        """Main trading loop - handles automated trading decisions."""
        logger.info("🔄 Starting main trading loop")
        
        while self.is_running and not self.emergency_stop_triggered:
            try:
                # Check if we can trade
                if not self._can_trade():
                    await asyncio.sleep(10)
                    continue
                
                # Get current market data
                market_data = await self._get_market_data()
                
                # Generate trading decisions using ensemble system
                decisions = await self._generate_trading_decisions(market_data)
                
                # Execute trading decisions
                if decisions:
                    await self._execute_trading_decisions(decisions)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check risk management
                await self._check_risk_management()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second intervals
                
            except asyncio.CancelledError:
                logger.info("Main trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _rebalancing_loop(self):
        """Portfolio rebalancing loop."""
        logger.info("⚖️ Starting portfolio rebalancing loop")
        
        while self.is_running and not self.emergency_stop_triggered:
            try:
                # Check if rebalancing is needed
                needs_rebalancing = await self.portfolio_balancer.check_rebalancing_needs()
                
                if needs_rebalancing:
                    logger.info("🔄 Portfolio rebalancing needed")
                    
                    # Get market data
                    market_data = await self._get_market_data()
                    
                    # Generate rebalancing decisions
                    rebalancing_decisions = await self.portfolio_balancer.generate_rebalancing_decisions(market_data)
                    
                    if rebalancing_decisions:
                        # Execute rebalancing
                        success = await self.portfolio_balancer.execute_rebalancing(rebalancing_decisions)
                        
                        if success:
                            self.performance.rebalancing_count += 1
                            logger.info(f"✅ Portfolio rebalancing completed: {len(rebalancing_decisions)} trades")
                        else:
                            logger.warning("⚠️ Portfolio rebalancing failed")
                
                # Wait before next check
                await asyncio.sleep(self.config.rebalancing_interval)
                
            except asyncio.CancelledError:
                logger.info("Rebalancing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Monitoring loop for system health and performance."""
        logger.info("📊 Starting monitoring loop")
        
        while self.is_running and not self.emergency_stop_triggered:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                # Check performance metrics
                await self._update_performance_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Log status
                if self.config.performance_tracking:
                    self._log_performance_summary()
                
                # Wait before next check
                await asyncio.sleep(30)  # 30-second intervals
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for all trading pairs."""
        try:
            market_data = {}
            
            for symbol in self.config.trading_pairs:
                data = await self.market_data_pipeline.get_market_data_packet(symbol)
                if data:
                    market_data[symbol] = data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    async def _generate_trading_decisions(self, market_data: Dict[str, Any]) -> List[Dict]:
        """Generate trading decisions using the ensemble system."""
        try:
            decisions = []
            
            for symbol, data in market_data.items():
                # Generate ensemble decision
                decision_vector = await self.ensemble_system.generate_decision_vector(
                    market_data=data,
                    portfolio_state=await self.portfolio_balancer.get_portfolio_metrics()
                )
                
                if decision_vector and decision_vector.get('confidence', 0) > self.config.math_confidence_threshold:
                    decisions.append({
                        'symbol': symbol,
                        'decision_vector': decision_vector,
                        'market_data': data,
                        'timestamp': time.time()
                    })
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error generating trading decisions: {e}")
            return []
    
    async def _execute_trading_decisions(self, decisions: List[Dict]):
        """Execute trading decisions."""
        try:
            for decision in decisions:
                symbol = decision['symbol']
                decision_vector = decision['decision_vector']
                market_data = decision['market_data']
                
                # Execute the decision
                result = await self.execution_engine.execute_strategy(
                    decision_vector=decision_vector['vector'],
                    market_data=market_data
                )
                
                if result.success:
                    logger.info(f"✅ Trade executed for {symbol}: {result.executed_size} @ {result.executed_price}")
                else:
                    logger.warning(f"⚠️ Trade failed for {symbol}: {result.error_message}")
                
        except Exception as e:
            logger.error(f"Error executing trading decisions: {e}")
    
    async def _check_risk_management(self):
        """Check risk management rules."""
        try:
            # Check portfolio risk
            portfolio_risk = self.risk_manager.assess_portfolio_risk(self.active_positions)
            
            # Check for emergency stop conditions
            if portfolio_risk.risk_level.value == 'critical':
                logger.warning("🚨 Critical risk level detected")
                await self.emergency_stop()
                return
            
            # Check daily loss limit
            if self.performance.total_pnl_percentage <= -self.config.max_daily_loss:
                logger.warning(f"🚨 Daily loss limit reached: {self.performance.total_pnl_percentage:.2%}")
                await self.emergency_stop()
                return
            
            # Check max drawdown
            if self.performance.max_drawdown >= self.config.max_drawdown:
                logger.warning(f"🚨 Max drawdown reached: {self.performance.max_drawdown:.2%}")
                await self.emergency_stop()
                return
                
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions."""
        try:
            for symbol, position in self.active_positions.items():
                await self.trading_system._close_position(symbol, 'emergency_stop')
                logger.info(f"🛑 Closed position for {symbol}")
            
            self.active_positions.clear()
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self.emergency_stop_triggered:
            return False
        
        if self.state != BotState.RUNNING:
            return False
        
        # Check if within trading hours (if configured)
        # Add any other trading restrictions here
        
        return True
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health status."""
        try:
            health_status = {
                'bot_state': self.state.value,
                'is_running': self.is_running,
                'emergency_stop': self.emergency_stop_triggered,
                'active_positions': len(self.active_positions),
                'pending_orders': len(self.pending_orders),
                'market_data_connected': self.market_data_pipeline.is_connected(),
                'portfolio_synced': self.portfolio_tracker.is_synced(),
                'risk_level': 'normal'  # Will be updated by risk manager
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {'error': str(e)}
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get current portfolio value
            portfolio_summary = self.portfolio_tracker.get_enhanced_summary()
            current_value = portfolio_summary.get('total_value', 0)
            
            # Calculate performance metrics
            if self.performance.start_time:
                elapsed_time = (datetime.now() - self.performance.start_time).total_seconds() / 3600  # hours
                if elapsed_time > 0:
                    self.performance.total_pnl_percentage = (current_value - self.config.initial_capital) / self.config.initial_capital
                    
                    # Calculate win rate
                    if self.performance.total_trades > 0:
                        self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
            
            self.performance.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            # Check for significant P&L changes
            if abs(self.performance.total_pnl_percentage) > 0.05:  # 5% change
                await self._trigger_alert('significant_pnl_change', {
                    'pnl_percentage': self.performance.total_pnl_percentage,
                    'total_pnl': self.performance.total_pnl
                })
            
            # Check for high drawdown
            if self.performance.max_drawdown > 0.10:  # 10% drawdown
                await self._trigger_alert('high_drawdown', {
                    'max_drawdown': self.performance.max_drawdown
                })
            
            # Check for low win rate
            if self.performance.total_trades > 10 and self.performance.win_rate < 0.4:  # 40% win rate
                await self._trigger_alert('low_win_rate', {
                    'win_rate': self.performance.win_rate,
                    'total_trades': self.performance.total_trades
                })
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert."""
        try:
            alert = {
                'type': alert_type,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'bot_state': self.state.value
            }
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"🚨 Alert triggered: {alert_type}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _log_performance_summary(self):
        """Log performance summary."""
        try:
            logger.info(f"📊 Performance Summary:")
            logger.info(f"   Total Trades: {self.performance.total_trades}")
            logger.info(f"   Win Rate: {self.performance.win_rate:.2%}")
            logger.info(f"   Total P&L: ${self.performance.total_pnl:.2f} ({self.performance.total_pnl_percentage:.2%})")
            logger.info(f"   Max Drawdown: {self.performance.max_drawdown:.2%}")
            logger.info(f"   Rebalancing Count: {self.performance.rebalancing_count}")
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")
    
    # Event handlers
    def _on_rebalancing_event(self, action: RebalancingAction, result: Dict[str, Any]):
        """Handle rebalancing events."""
        logger.info(f"🔄 Rebalancing event: {action.symbol} {action.action} {action.amount}")
        self.rebalancing_history.append({
            'action': action,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    def _on_price_update(self, price_update):
        """Handle price updates."""
        # Price updates are handled by the portfolio tracker
        pass
    
    def _on_trade_event(self, event_type: str, data: Any, metadata: Dict[str, Any] = None):
        """Handle trade events."""
        logger.info(f"💰 Trade event: {event_type}")
        
        if event_type == 'trade_executed':
            self.performance.total_trades += 1
            if data.get('pnl', 0) > 0:
                self.performance.winning_trades += 1
            self.performance.total_pnl += data.get('pnl', 0)
        
        self.trade_history.append({
            'event_type': event_type,
            'data': data,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        })
    
    def _on_alert_event(self, alert_type: str, data: Dict[str, Any]):
        """Handle alert events."""
        logger.warning(f"🚨 Alert: {alert_type}")
    
    def _on_performance_event(self, metrics: Dict[str, Any]):
        """Handle performance events."""
        # Performance metrics are updated in the monitoring loop
        pass
    
    # Public methods for monitoring and control
    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            'state': self.state.value,
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop_triggered,
            'performance': {
                'total_trades': self.performance.total_trades,
                'win_rate': self.performance.win_rate,
                'total_pnl': self.performance.total_pnl,
                'total_pnl_percentage': self.performance.total_pnl_percentage,
                'max_drawdown': self.performance.max_drawdown,
                'rebalancing_count': self.performance.rebalancing_count
            },
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'uptime': (datetime.now() - self.performance.start_time).total_seconds() if self.performance.start_time else 0
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history.copy()
    
    def get_rebalancing_history(self) -> List[Dict]:
        """Get rebalancing history."""
        return self.rebalancing_history.copy()
    
    def add_trade_callback(self, callback: Callable):
        """Add trade event callback."""
        self.trade_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add alert event callback."""
        self.alert_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable):
        """Add performance event callback."""
        self.performance_callbacks.append(callback)

# Factory function to create the bot
def create_automated_crypto_trading_bot(config: AutomatedBotConfig) -> AutomatedCryptoTradingBot:
    """Create and configure an automated crypto trading bot."""
    return AutomatedCryptoTradingBot(config)

# Test function
async def test_automated_crypto_trading_bot():
    """Test the automated crypto trading bot."""
    try:
        # Create test configuration
        config = AutomatedBotConfig(
            trading_mode=TradingMode.DEMO,
            initial_capital=10000.0,
            trading_pairs=["BTC/USDC", "ETH/USDC"],
            target_allocation={"BTC": 0.5, "ETH": 0.3, "USDC": 0.2}
        )
        
        # Create bot
        bot = create_automated_crypto_trading_bot(config)
        
        # Start bot
        await bot.start()
        
        # Run for a short time
        await asyncio.sleep(60)
        
        # Get status
        status = bot.get_bot_status()
        print(f"Bot Status: {status}")
        
        # Stop bot
        await bot.stop()
        
        print("✅ Automated crypto trading bot test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_automated_crypto_trading_bot()) 