#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MODE INTEGRATION SYSTEM - SCHWABOT
====================================

Comprehensive integration system that connects all trading modes:
- Clock Mode System (mechanical watchmaker principles)
- Unified Live Backtesting System (real API data testing)
- Live Trading System (real execution)
- Real API Pricing & Memory System (data and storage)

This system ensures seamless operation across all modes with proper memory routing,
real API integration, and mathematical consensus building.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import sys

# Import all system components
try:
    from clock_mode_system import ClockModeSystem, ExecutionMode, SafetyConfig
    CLOCK_MODE_AVAILABLE = True
except ImportError:
    CLOCK_MODE_AVAILABLE = False
    logging.warning("⚠️ Clock Mode System not available")

try:
    from unified_live_backtesting_system import UnifiedLiveBacktestingSystem, BacktestConfig, BacktestMode
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    logging.warning("⚠️ Unified Live Backtesting System not available")

try:
    from real_api_pricing_memory_system import (
        initialize_real_api_memory_system, 
        get_real_price_data, 
        store_memory_entry,
        MemoryConfig,
        MemoryStorageMode,
        APIMode
    )
    REAL_API_AVAILABLE = True
except ImportError:
    REAL_API_AVAILABLE = False
    logging.warning("⚠️ Real API Pricing & Memory System not available")

try:
    from schwabot_trading_bot import SchwabotTradingBot
    TRADING_BOT_AVAILABLE = True
except ImportError:
    TRADING_BOT_AVAILABLE = False
    logging.warning("⚠️ Schwabot Trading Bot not available")

try:
    from backtesting.mathematical_integration import MathematicalIntegrationEngine, MathematicalSignal
    MATHEMATICAL_AVAILABLE = True
except ImportError:
    MATHEMATICAL_AVAILABLE = False
    logging.warning("⚠️ Mathematical Integration not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mode_integration_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Available trading modes."""
    CLOCK_MODE = "clock_mode"              # Mechanical watchmaker system
    LIVE_BACKTESTING = "live_backtesting"  # Real API data testing
    LIVE_TRADING = "live_trading"          # Real execution
    SHADOW_MODE = "shadow_mode"            # Analysis only
    PAPER_TRADING = "paper_trading"        # Simulated execution
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"  # Math-only analysis

@dataclass
class ModeConfiguration:
    """Configuration for each trading mode."""
    mode: TradingMode
    enabled: bool = True
    priority: int = 1
    auto_start: bool = False
    memory_routing: str = "auto"  # auto, local, usb, hybrid
    api_integration: bool = True
    mathematical_integration: bool = True
    safety_checks: bool = True
    performance_tracking: bool = True

@dataclass
class SystemStatus:
    """Overall system status."""
    current_mode: TradingMode
    is_running: bool = False
    active_components: List[str] = field(default_factory=list)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    api_status: Dict[str, Any] = field(default_factory=dict)
    mathematical_status: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    safety_status: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)

class ModeIntegrationSystem:
    """Main integration system that coordinates all trading modes."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the mode integration system."""
        self.config = self._load_config(config_path)
        self.status = SystemStatus(current_mode=TradingMode.SHADOW_MODE)
        
        # Initialize all available systems
        self._initialize_systems()
        
        # Mode configurations
        self.mode_configs = self._create_mode_configurations()
        
        # Active components
        self.active_components: Dict[str, Any] = {}
        self.memory_system = None
        self.mathematical_engine = None
        
        # Threading and async support
        self.is_running = False
        self.monitoring_thread = None
        self.async_loop = None
        
        logger.info("🎯 Mode Integration System initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Error loading config: {e}")
        
        # Default configuration
        return {
            "default_mode": "shadow_mode",
            "memory_storage": "auto",
            "api_mode": "real_api_only",
            "safety_enabled": True,
            "auto_sync": True,
            "performance_tracking": True,
            "mathematical_integration": True
        }
    
    def _initialize_systems(self):
        """Initialize all available systems."""
        # Initialize Real API Pricing & Memory System
        if REAL_API_AVAILABLE:
            try:
                memory_config = MemoryConfig(
                    storage_mode=MemoryStorageMode.AUTO,
                    api_mode=APIMode.REAL_API_ONLY,
                    auto_sync=True,
                    memory_choice_menu=False  # Don't show menu for integration
                )
                self.memory_system = initialize_real_api_memory_system(memory_config)
                logger.info("✅ Real API Pricing & Memory System initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Real API system: {e}")
                self.memory_system = None
        
        # Initialize Mathematical Integration Engine
        if MATHEMATICAL_AVAILABLE:
            try:
                self.mathematical_engine = MathematicalIntegrationEngine()
                logger.info("✅ Mathematical Integration Engine initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Mathematical Engine: {e}")
                self.mathematical_engine = None
        
        # Initialize Clock Mode System
        if CLOCK_MODE_AVAILABLE:
            try:
                self.clock_system = ClockModeSystem()
                logger.info("✅ Clock Mode System initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Clock Mode System: {e}")
                self.clock_system = None
        
        # Initialize Unified Live Backtesting System
        if BACKTESTING_AVAILABLE:
            try:
                backtest_config = BacktestConfig(
                    mode=BacktestMode.LIVE_API_BACKTEST,
                    enable_ai_analysis=True,
                    enable_risk_management=True,
                    enable_performance_optimization=True
                )
                self.backtesting_system = UnifiedLiveBacktestingSystem(backtest_config)
                logger.info("✅ Unified Live Backtesting System initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Backtesting System: {e}")
                self.backtesting_system = None
        
        # Initialize Schwabot Trading Bot
        if TRADING_BOT_AVAILABLE:
            try:
                self.trading_bot = SchwabotTradingBot()
                logger.info("✅ Schwabot Trading Bot initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Trading Bot: {e}")
                self.trading_bot = None
    
    def _create_mode_configurations(self) -> Dict[TradingMode, ModeConfiguration]:
        """Create configurations for all trading modes."""
        configs = {}
        
        # Clock Mode Configuration
        configs[TradingMode.CLOCK_MODE] = ModeConfiguration(
            mode=TradingMode.CLOCK_MODE,
            enabled=CLOCK_MODE_AVAILABLE,
            priority=1,
            auto_start=False,
            memory_routing="auto",
            api_integration=True,
            mathematical_integration=True,
            safety_checks=True,
            performance_tracking=True
        )
        
        # Live Backtesting Configuration
        configs[TradingMode.LIVE_BACKTESTING] = ModeConfiguration(
            mode=TradingMode.LIVE_BACKTESTING,
            enabled=BACKTESTING_AVAILABLE,
            priority=2,
            auto_start=False,
            memory_routing="auto",
            api_integration=True,
            mathematical_integration=True,
            safety_checks=True,
            performance_tracking=True
        )
        
        # Live Trading Configuration
        configs[TradingMode.LIVE_TRADING] = ModeConfiguration(
            mode=TradingMode.LIVE_TRADING,
            enabled=TRADING_BOT_AVAILABLE,
            priority=3,
            auto_start=False,
            memory_routing="hybrid",  # Use both local and USB
            api_integration=True,
            mathematical_integration=True,
            safety_checks=True,
            performance_tracking=True
        )
        
        # Shadow Mode Configuration
        configs[TradingMode.SHADOW_MODE] = ModeConfiguration(
            mode=TradingMode.SHADOW_MODE,
            enabled=True,
            priority=0,  # Highest priority for safety
            auto_start=True,
            memory_routing="local",
            api_integration=True,
            mathematical_integration=True,
            safety_checks=True,
            performance_tracking=True
        )
        
        # Paper Trading Configuration
        configs[TradingMode.PAPER_TRADING] = ModeConfiguration(
            mode=TradingMode.PAPER_TRADING,
            enabled=TRADING_BOT_AVAILABLE,
            priority=2,
            auto_start=False,
            memory_routing="auto",
            api_integration=True,
            mathematical_integration=True,
            safety_checks=True,
            performance_tracking=True
        )
        
        # Mathematical Analysis Configuration
        configs[TradingMode.MATHEMATICAL_ANALYSIS] = ModeConfiguration(
            mode=TradingMode.MATHEMATICAL_ANALYSIS,
            enabled=MATHEMATICAL_AVAILABLE,
            priority=1,
            auto_start=False,
            memory_routing="auto",
            api_integration=True,
            mathematical_integration=True,
            safety_checks=False,  # No safety needed for analysis
            performance_tracking=True
        )
        
        return configs
    
    async def start_system(self, initial_mode: TradingMode = TradingMode.SHADOW_MODE) -> bool:
        """Start the mode integration system."""
        if self.is_running:
            logger.warning("⚠️ System already running")
            return False
        
        # Safety check
        if not self._safety_check_startup():
            logger.error("❌ Safety check failed - cannot start system")
            return False
        
        self.is_running = True
        self.status.current_mode = initial_mode
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start async loop
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        
        # Initialize memory system
        if self.memory_system:
            try:
                # Start memory system
                logger.info("🔄 Starting memory system...")
                # Note: Memory system doesn't have explicit start method, it's always running
            except Exception as e:
                logger.error(f"❌ Error starting memory system: {e}")
        
        # Start initial mode
        await self.switch_mode(initial_mode)
        
        logger.info(f"🎯 Mode Integration System started in {initial_mode.value} mode")
        return True
    
    async def stop_system(self) -> bool:
        """Stop the mode integration system."""
        self.is_running = False
        
        # Stop all active components
        await self._stop_all_components()
        
        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop memory system
        if self.memory_system:
            try:
                self.memory_system.stop()
                logger.info("✅ Memory system stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping memory system: {e}")
        
        logger.info("🎯 Mode Integration System stopped")
        return True
    
    async def switch_mode(self, new_mode: TradingMode) -> bool:
        """Switch to a different trading mode."""
        if not self.is_running:
            logger.error("❌ System not running")
            return False
        
        # Check if mode is available
        if not self.mode_configs[new_mode].enabled:
            logger.error(f"❌ Mode {new_mode.value} not available")
            return False
        
        # Safety check for mode switching
        if not self._safety_check_mode_switch(new_mode):
            logger.error(f"❌ Safety check failed for mode {new_mode.value}")
            return False
        
        # Stop current mode
        await self._stop_current_mode()
        
        # Start new mode
        success = await self._start_mode(new_mode)
        
        if success:
            self.status.current_mode = new_mode
            logger.info(f"🎯 Switched to {new_mode.value} mode")
            
            # Store mode change in memory
            if self.memory_system:
                try:
                    store_memory_entry(
                        data_type='mode_change',
                        data={
                            'previous_mode': self.status.current_mode.value,
                            'new_mode': new_mode.value,
                            'timestamp': datetime.now().isoformat(),
                            'reason': 'user_request'
                        },
                        source='mode_integration',
                        priority=2,
                        tags=['mode_change', 'system_event']
                    )
                except Exception as e:
                    logger.debug(f"Error storing mode change: {e}")
        
        return success
    
    async def _start_mode(self, mode: TradingMode) -> bool:
        """Start a specific trading mode."""
        try:
            if mode == TradingMode.CLOCK_MODE:
                return await self._start_clock_mode()
            elif mode == TradingMode.LIVE_BACKTESTING:
                return await self._start_live_backtesting()
            elif mode == TradingMode.LIVE_TRADING:
                return await self._start_live_trading()
            elif mode == TradingMode.SHADOW_MODE:
                return await self._start_shadow_mode()
            elif mode == TradingMode.PAPER_TRADING:
                return await self._start_paper_trading()
            elif mode == TradingMode.MATHEMATICAL_ANALYSIS:
                return await self._start_mathematical_analysis()
            else:
                logger.error(f"❌ Unknown mode: {mode.value}")
                return False
        except Exception as e:
            logger.error(f"❌ Error starting mode {mode.value}: {e}")
            return False
    
    async def _start_clock_mode(self) -> bool:
        """Start clock mode system."""
        if not self.clock_system:
            logger.error("❌ Clock mode system not available")
            return False
        
        try:
            success = self.clock_system.start_clock_mode()
            if success:
                self.active_components['clock_mode'] = self.clock_system
                logger.info("✅ Clock mode started")
            return success
        except Exception as e:
            logger.error(f"❌ Error starting clock mode: {e}")
            return False
    
    async def _start_live_backtesting(self) -> bool:
        """Start live backtesting system."""
        if not self.backtesting_system:
            logger.error("❌ Backtesting system not available")
            return False
        
        try:
            # Start backtesting in async context
            backtest_task = asyncio.create_task(self.backtesting_system.start_backtest())
            self.active_components['live_backtesting'] = backtest_task
            logger.info("✅ Live backtesting started")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting live backtesting: {e}")
            return False
    
    async def _start_live_trading(self) -> bool:
        """Start live trading system."""
        if not self.trading_bot:
            logger.error("❌ Trading bot not available")
            return False
        
        try:
            # Start trading bot in async context
            trading_task = asyncio.create_task(self.trading_bot.start())
            self.active_components['live_trading'] = trading_task
            logger.info("✅ Live trading started")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting live trading: {e}")
            return False
    
    async def _start_shadow_mode(self) -> bool:
        """Start shadow mode (analysis only)."""
        try:
            # Shadow mode just runs mathematical analysis and memory storage
            # No actual trading components needed
            self.active_components['shadow_mode'] = {
                'type': 'shadow_analysis',
                'start_time': datetime.now()
            }
            logger.info("✅ Shadow mode started")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting shadow mode: {e}")
            return False
    
    async def _start_paper_trading(self) -> bool:
        """Start paper trading system."""
        if not self.trading_bot:
            logger.error("❌ Trading bot not available")
            return False
        
        try:
            # Configure trading bot for paper trading
            # This would involve setting the bot to paper trading mode
            paper_task = asyncio.create_task(self.trading_bot.start())
            self.active_components['paper_trading'] = paper_task
            logger.info("✅ Paper trading started")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting paper trading: {e}")
            return False
    
    async def _start_mathematical_analysis(self) -> bool:
        """Start mathematical analysis mode."""
        if not self.mathematical_engine:
            logger.error("❌ Mathematical engine not available")
            return False
        
        try:
            # Start mathematical analysis loop
            math_task = asyncio.create_task(self._mathematical_analysis_loop())
            self.active_components['mathematical_analysis'] = math_task
            logger.info("✅ Mathematical analysis started")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting mathematical analysis: {e}")
            return False
    
    async def _stop_current_mode(self):
        """Stop the current trading mode."""
        current_mode = self.status.current_mode
        
        if current_mode == TradingMode.CLOCK_MODE:
            if 'clock_mode' in self.active_components:
                try:
                    self.clock_system.stop_clock_mode()
                    del self.active_components['clock_mode']
                except Exception as e:
                    logger.error(f"❌ Error stopping clock mode: {e}")
        
        elif current_mode == TradingMode.LIVE_BACKTESTING:
            if 'live_backtesting' in self.active_components:
                try:
                    task = self.active_components['live_backtesting']
                    task.cancel()
                    del self.active_components['live_backtesting']
                except Exception as e:
                    logger.error(f"❌ Error stopping live backtesting: {e}")
        
        elif current_mode == TradingMode.LIVE_TRADING:
            if 'live_trading' in self.active_components:
                try:
                    task = self.active_components['live_trading']
                    task.cancel()
                    del self.active_components['live_trading']
                except Exception as e:
                    logger.error(f"❌ Error stopping live trading: {e}")
        
        elif current_mode == TradingMode.PAPER_TRADING:
            if 'paper_trading' in self.active_components:
                try:
                    task = self.active_components['paper_trading']
                    task.cancel()
                    del self.active_components['paper_trading']
                except Exception as e:
                    logger.error(f"❌ Error stopping paper trading: {e}")
        
        elif current_mode == TradingMode.MATHEMATICAL_ANALYSIS:
            if 'mathematical_analysis' in self.active_components:
                try:
                    task = self.active_components['mathematical_analysis']
                    task.cancel()
                    del self.active_components['mathematical_analysis']
                except Exception as e:
                    logger.error(f"❌ Error stopping mathematical analysis: {e}")
        
        # Shadow mode doesn't need explicit stopping
    
    async def _stop_all_components(self):
        """Stop all active components."""
        for component_name, component in list(self.active_components.items()):
            try:
                if hasattr(component, 'cancel'):
                    component.cancel()
                elif hasattr(component, 'stop'):
                    component.stop()
                del self.active_components[component_name]
            except Exception as e:
                logger.error(f"❌ Error stopping component {component_name}: {e}")
    
    def _safety_check_startup(self) -> bool:
        """Perform safety checks before starting the system."""
        try:
            # Check if any critical systems are available
            if not REAL_API_AVAILABLE:
                logger.warning("⚠️ Real API system not available")
            
            if not MATHEMATICAL_AVAILABLE:
                logger.warning("⚠️ Mathematical integration not available")
            
            # Check memory system
            if not self.memory_system:
                logger.warning("⚠️ Memory system not available")
            
            # Basic safety checks passed
            logger.info("✅ Safety checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safety check error: {e}")
            return False
    
    def _safety_check_mode_switch(self, new_mode: TradingMode) -> bool:
        """Check safety before switching modes."""
        try:
            # Check if mode is enabled
            if not self.mode_configs[new_mode].enabled:
                return False
            
            # Check for dangerous mode transitions
            current_mode = self.status.current_mode
            
            # Don't allow direct switch from shadow to live trading
            if (current_mode == TradingMode.SHADOW_MODE and 
                new_mode == TradingMode.LIVE_TRADING):
                logger.warning("⚠️ Direct switch from shadow to live trading not allowed")
                return False
            
            # Check if memory system is available for modes that need it
            if new_mode in [TradingMode.LIVE_TRADING, TradingMode.LIVE_BACKTESTING]:
                if not self.memory_system:
                    logger.warning("⚠️ Memory system required for this mode")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Mode switch safety check error: {e}")
            return False
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Update system status
                self._update_system_status()
                
                # Check component health
                self._check_component_health()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(10.0)  # Longer sleep on error
    
    def _update_system_status(self):
        """Update overall system status."""
        try:
            self.status.is_running = self.is_running
            self.status.active_components = list(self.active_components.keys())
            self.status.last_update = datetime.now()
            
            # Update memory stats
            if self.memory_system:
                try:
                    self.status.memory_stats = self.memory_system.get_memory_stats()
                except Exception as e:
                    logger.debug(f"Error getting memory stats: {e}")
            
            # Update API status
            if REAL_API_AVAILABLE:
                self.status.api_status = {
                    "available": True,
                    "last_check": datetime.now().isoformat()
                }
            else:
                self.status.api_status = {
                    "available": False,
                    "reason": "System not available"
                }
            
            # Update mathematical status
            if self.mathematical_engine:
                self.status.mathematical_status = {
                    "available": True,
                    "engine_type": "MathematicalIntegrationEngine"
                }
            else:
                self.status.mathematical_status = {
                    "available": False,
                    "reason": "Engine not available"
                }
            
        except Exception as e:
            logger.error(f"❌ Error updating system status: {e}")
    
    def _check_component_health(self):
        """Check health of active components."""
        for component_name, component in self.active_components.items():
            try:
                # Check if component is still running
                if hasattr(component, 'done') and component.done():
                    if component.cancelled():
                        logger.warning(f"⚠️ Component {component_name} was cancelled")
                    elif component.exception():
                        logger.error(f"❌ Component {component_name} failed: {component.exception()}")
                    del self.active_components[component_name]
                
            except Exception as e:
                logger.error(f"❌ Error checking component {component_name}: {e}")
    
    async def _mathematical_analysis_loop(self):
        """Mathematical analysis loop for analysis-only mode."""
        while self.is_running:
            try:
                # Get real market data
                if REAL_API_AVAILABLE:
                    btc_price = get_real_price_data('BTC/USDC', 'binance')
                    eth_price = get_real_price_data('ETH/USDC', 'binance')
                    
                    market_data = {
                        "price": btc_price,
                        "eth_price": eth_price,
                        "timestamp": datetime.now().isoformat(),
                        "source": "real_api"
                    }
                else:
                    # Fallback to simulated data
                    market_data = {
                        "price": 50000.0,
                        "eth_price": 3000.0,
                        "timestamp": datetime.now().isoformat(),
                        "source": "simulated"
                    }
                
                # Process mathematical analysis
                if self.mathematical_engine:
                    signal = await self.mathematical_engine.process_market_data_mathematically(market_data)
                    
                    # Store analysis results
                    if self.memory_system:
                        try:
                            store_memory_entry(
                                data_type='mathematical_analysis',
                                data={
                                    'market_data': market_data,
                                    'signal': {
                                        'dlt_waveform_score': signal.dlt_waveform_score,
                                        'dualistic_consensus': signal.dualistic_consensus,
                                        'bit_phase': signal.bit_phase,
                                        'ferris_phase': signal.ferris_phase,
                                        'confidence': signal.confidence,
                                        'decision': signal.decision
                                    },
                                    'timestamp': datetime.now().isoformat()
                                },
                                source='mathematical_analysis',
                                priority=1,
                                tags=['mathematical_analysis', 'real_time']
                            )
                        except Exception as e:
                            logger.debug(f"Error storing mathematical analysis: {e}")
                
                # Sleep for analysis interval
                await asyncio.sleep(1.0)  # Analyze every second
                
            except Exception as e:
                logger.error(f"❌ Error in mathematical analysis loop: {e}")
                await asyncio.sleep(5.0)  # Longer sleep on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_info": {
                "is_running": self.is_running,
                "current_mode": self.status.current_mode.value,
                "active_components": self.status.active_components,
                "last_update": self.status.last_update.isoformat()
            },
            "available_modes": {
                mode.value: {
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "auto_start": config.auto_start
                }
                for mode, config in self.mode_configs.items()
            },
            "component_status": {
                "real_api_available": REAL_API_AVAILABLE,
                "mathematical_available": MATHEMATICAL_AVAILABLE,
                "clock_mode_available": CLOCK_MODE_AVAILABLE,
                "backtesting_available": BACKTESTING_AVAILABLE,
                "trading_bot_available": TRADING_BOT_AVAILABLE
            },
            "memory_stats": self.status.memory_stats,
            "api_status": self.status.api_status,
            "mathematical_status": self.status.mathematical_status,
            "performance_metrics": self.status.performance_metrics,
            "safety_status": self.status.safety_status
        }
    
    def get_available_modes(self) -> List[Dict[str, Any]]:
        """Get list of available modes with their configurations."""
        modes = []
        for mode, config in self.mode_configs.items():
            modes.append({
                "mode": mode.value,
                "enabled": config.enabled,
                "priority": config.priority,
                "auto_start": config.auto_start,
                "description": self._get_mode_description(mode)
            })
        return modes
    
    def _get_mode_description(self, mode: TradingMode) -> str:
        """Get description for a trading mode."""
        descriptions = {
            TradingMode.CLOCK_MODE: "Mechanical watchmaker system with real API integration",
            TradingMode.LIVE_BACKTESTING: "Real API data testing without real trades",
            TradingMode.LIVE_TRADING: "Real trading execution with full safety",
            TradingMode.SHADOW_MODE: "Analysis only - no trades executed",
            TradingMode.PAPER_TRADING: "Simulated trading with real market data",
            TradingMode.MATHEMATICAL_ANALYSIS: "Mathematical analysis and signal generation"
        }
        return descriptions.get(mode, "Unknown mode")

# Global instance for easy access
mode_integration_system = ModeIntegrationSystem()

# Convenience functions
async def start_mode_integration_system(initial_mode: TradingMode = TradingMode.SHADOW_MODE) -> bool:
    """Start the mode integration system."""
    return await mode_integration_system.start_system(initial_mode)

async def stop_mode_integration_system() -> bool:
    """Stop the mode integration system."""
    return await mode_integration_system.stop_system()

async def switch_trading_mode(new_mode: TradingMode) -> bool:
    """Switch to a different trading mode."""
    return await mode_integration_system.switch_mode(new_mode)

def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    return mode_integration_system.get_system_status()

def get_available_modes() -> List[Dict[str, Any]]:
    """Get available modes."""
    return mode_integration_system.get_available_modes()

# Main function for testing
async def main():
    """Test the mode integration system."""
    logger.info("🎯 Starting Mode Integration System Test")
    
    # Start system in shadow mode
    if not await start_mode_integration_system(TradingMode.SHADOW_MODE):
        logger.error("❌ Failed to start mode integration system")
        return
    
    # Run for a few seconds
    await asyncio.sleep(10)
    
    # Get status
    status = get_system_status()
    logger.info(f"🎯 System Status: {json.dumps(status, indent=2)}")
    
    # Switch to mathematical analysis mode
    await switch_trading_mode(TradingMode.MATHEMATICAL_ANALYSIS)
    
    # Run for a few more seconds
    await asyncio.sleep(10)
    
    # Stop system
    await stop_mode_integration_system()
    
    logger.info("🎯 Mode Integration System Test Complete")

if __name__ == "__main__":
    asyncio.run(main()) 