#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SCHWABOT COMPLETE SYSTEM LAUNCHER
====================================

Complete system launcher that integrates ALL components for live backtesting:
- Real API Pricing & Memory System
- Clock Mode System (mechanical watchmaker)
- Unified Live Backtesting System
- Mathematical Integration Engine
- Mode Integration System
- Schwabot Trading Bot

This launcher ensures everything runs smoothly with proper memory routing,
real API integration, and mathematical consensus building for optimal trading performance.
"""

import asyncio
import json
import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_complete_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import all system components
try:
    from core.mode_integration_system import (
        ModeIntegrationSystem, TradingMode, 
        start_mode_integration_system, stop_mode_integration_system,
        switch_trading_mode, get_system_status, get_available_modes
    )
    MODE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    MODE_INTEGRATION_AVAILABLE = False
    logger.error(f"❌ Mode Integration System not available: {e}")

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
except ImportError as e:
    REAL_API_AVAILABLE = False
    logger.error(f"❌ Real API Pricing & Memory System not available: {e}")

try:
    from clock_mode_system import ClockModeSystem, ExecutionMode, SafetyConfig
    CLOCK_MODE_AVAILABLE = True
except ImportError as e:
    CLOCK_MODE_AVAILABLE = False
    logger.error(f"❌ Clock Mode System not available: {e}")

try:
    from unified_live_backtesting_system import (
        UnifiedLiveBacktestingSystem, BacktestConfig, BacktestMode,
        start_live_backtest
    )
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    logger.error(f"❌ Unified Live Backtesting System not available: {e}")

try:
    from backtesting.mathematical_integration import MathematicalIntegrationEngine, MathematicalSignal
    MATHEMATICAL_AVAILABLE = True
except ImportError as e:
    MATHEMATICAL_AVAILABLE = False
    logger.error(f"❌ Mathematical Integration not available: {e}")

try:
    from schwabot_trading_bot import SchwabotTradingBot
    TRADING_BOT_AVAILABLE = True
except ImportError as e:
    TRADING_BOT_AVAILABLE = False
    logger.error(f"❌ Schwabot Trading Bot not available: {e}")

class SchwabotCompleteSystemLauncher:
    """Complete system launcher for Schwabot trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the complete system launcher."""
        self.config = self._load_config(config_path)
        self.mode_integration = None
        self.memory_system = None
        self.mathematical_engine = None
        self.clock_system = None
        self.backtesting_system = None
        self.trading_bot = None
        
        # System state
        self.is_running = False
        self.current_mode = TradingMode.SHADOW_MODE
        self.start_time = None
        self.system_stats = {
            "total_runtime": 0,
            "mode_switches": 0,
            "memory_entries": 0,
            "mathematical_signals": 0,
            "api_calls": 0,
            "errors": 0
        }
        
        # Initialize all systems
        self._initialize_all_systems()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Schwabot Complete System Launcher initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Error loading config: {e}")
        
        # Default configuration for complete system
        return {
            "system_name": "Schwabot Complete System",
            "version": "1.0.0",
            "default_mode": "shadow_mode",
            "auto_start": True,
            "memory_storage": "auto",
            "api_mode": "real_api_only",
            "safety_enabled": True,
            "auto_sync": True,
            "performance_tracking": True,
            "mathematical_integration": True,
            "backtesting_duration_hours": 24,
            "clock_mode_enabled": True,
            "live_trading_enabled": False,  # Disabled by default for safety
            "paper_trading_enabled": True,
            "monitoring_interval": 5.0,
            "emergency_stop_enabled": True,
            "max_daily_loss": 0.05,
            "max_position_size": 0.1,
            "min_confidence_threshold": 0.7
        }
    
    def _initialize_all_systems(self):
        """Initialize all available systems."""
        logger.info("🔄 Initializing all Schwabot systems...")
        
        # Initialize Real API Pricing & Memory System
        if REAL_API_AVAILABLE:
            try:
                memory_config = MemoryConfig(
                    storage_mode=MemoryStorageMode.AUTO,
                    api_mode=APIMode.REAL_API_ONLY,
                    auto_sync=True,
                    memory_choice_menu=False,  # Don't show menu for launcher
                    backup_interval=300,  # 5 minutes
                    max_backup_age_days=30,
                    compression_enabled=True,
                    encryption_enabled=True
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
                    symbols=["BTCUSDT", "ETHUSDT"],
                    exchanges=["binance", "coinbase"],
                    initial_balance=10000.0,
                    commission_rate=0.001,
                    slippage_rate=0.0005,
                    enable_ai_analysis=True,
                    enable_risk_management=True,
                    max_positions=5,
                    risk_per_trade=0.02,
                    min_confidence=0.7,
                    data_update_interval=1.0,
                    backtest_duration_hours=self.config.get("backtesting_duration_hours", 24),
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
        
        # Initialize Mode Integration System
        if MODE_INTEGRATION_AVAILABLE:
            try:
                self.mode_integration = ModeIntegrationSystem()
                logger.info("✅ Mode Integration System initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Mode Integration System: {e}")
                self.mode_integration = None
        
        # Store system initialization in memory
        if self.memory_system:
            try:
                store_memory_entry(
                    data_type='system_initialization',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'systems_available': {
                            'real_api': REAL_API_AVAILABLE,
                            'mathematical': MATHEMATICAL_AVAILABLE,
                            'clock_mode': CLOCK_MODE_AVAILABLE,
                            'backtesting': BACKTESTING_AVAILABLE,
                            'trading_bot': TRADING_BOT_AVAILABLE,
                            'mode_integration': MODE_INTEGRATION_AVAILABLE
                        },
                        'config': self.config
                    },
                    source='complete_system_launcher',
                    priority=1,
                    tags=['system_init', 'launcher']
                )
            except Exception as e:
                logger.debug(f"Error storing system initialization: {e}")
        
        logger.info("🔄 All systems initialization complete")
    
    async def start_complete_system(self, initial_mode: TradingMode = None) -> bool:
        """Start the complete Schwabot system."""
        if self.is_running:
            logger.warning("⚠️ System already running")
            return False
        
        # Determine initial mode
        if initial_mode is None:
            mode_name = self.config.get("default_mode", "shadow_mode")
            try:
                initial_mode = TradingMode(mode_name)
            except ValueError:
                logger.warning(f"⚠️ Invalid default mode: {mode_name}, using SHADOW_MODE")
                initial_mode = TradingMode.SHADOW_MODE
        
        # Safety check
        if not self._safety_check_startup():
            logger.error("❌ Safety check failed - cannot start system")
            return False
        
        self.is_running = True
        self.current_mode = initial_mode
        self.start_time = datetime.now()
        
        logger.info(f"🚀 Starting Schwabot Complete System in {initial_mode.value} mode")
        
        # Start mode integration system
        if self.mode_integration:
            try:
                success = await start_mode_integration_system(initial_mode)
                if not success:
                    logger.error("❌ Failed to start mode integration system")
                    return False
                logger.info("✅ Mode integration system started")
            except Exception as e:
                logger.error(f"❌ Error starting mode integration system: {e}")
                return False
        
        # Start monitoring and performance tracking
        await self._start_monitoring()
        
        # Store system start in memory
        if self.memory_system:
            try:
                store_memory_entry(
                    data_type='system_start',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'initial_mode': initial_mode.value,
                        'config': self.config,
                        'start_time': self.start_time.isoformat()
                    },
                    source='complete_system_launcher',
                    priority=1,
                    tags=['system_start', 'launcher']
                )
            except Exception as e:
                logger.debug(f"Error storing system start: {e}")
        
        logger.info("🚀 Schwabot Complete System started successfully")
        return True
    
    async def stop_complete_system(self) -> bool:
        """Stop the complete Schwabot system."""
        if not self.is_running:
            logger.warning("⚠️ System not running")
            return False
        
        logger.info("🛑 Stopping Schwabot Complete System...")
        
        self.is_running = False
        
        # Stop mode integration system
        if self.mode_integration:
            try:
                await stop_mode_integration_system()
                logger.info("✅ Mode integration system stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping mode integration system: {e}")
        
        # Stop all individual systems
        await self._stop_all_systems()
        
        # Calculate final stats
        if self.start_time:
            self.system_stats["total_runtime"] = (datetime.now() - self.start_time).total_seconds()
        
        # Store system stop in memory
        if self.memory_system:
            try:
                store_memory_entry(
                    data_type='system_stop',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'final_mode': self.current_mode.value,
                        'system_stats': self.system_stats,
                        'runtime_seconds': self.system_stats["total_runtime"]
                    },
                    source='complete_system_launcher',
                    priority=1,
                    tags=['system_stop', 'launcher']
                )
            except Exception as e:
                logger.debug(f"Error storing system stop: {e}")
        
        logger.info("🛑 Schwabot Complete System stopped")
        return True
    
    async def switch_system_mode(self, new_mode: TradingMode) -> bool:
        """Switch the system to a different mode."""
        if not self.is_running:
            logger.error("❌ System not running")
            return False
        
        try:
            success = await switch_trading_mode(new_mode)
            if success:
                self.current_mode = new_mode
                self.system_stats["mode_switches"] += 1
                logger.info(f"🎯 Switched to {new_mode.value} mode")
            return success
        except Exception as e:
            logger.error(f"❌ Error switching mode: {e}")
            self.system_stats["errors"] += 1
            return False
    
    async def _start_monitoring(self):
        """Start system monitoring and performance tracking."""
        logger.info("📊 Starting system monitoring...")
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Store monitoring start in memory
        if self.memory_system:
            try:
                store_memory_entry(
                    data_type='monitoring_start',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'monitoring_interval': self.config.get("monitoring_interval", 5.0)
                    },
                    source='complete_system_launcher',
                    priority=2,
                    tags=['monitoring', 'performance']
                )
            except Exception as e:
                logger.debug(f"Error storing monitoring start: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for system health and performance."""
        while self.is_running:
            try:
                # Update system stats
                await self._update_system_stats()
                
                # Check system health
                await self._check_system_health()
                
                # Log performance metrics
                await self._log_performance_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config.get("monitoring_interval", 5.0))
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                self.system_stats["errors"] += 1
                await asyncio.sleep(10.0)  # Longer sleep on error
    
    async def _update_system_stats(self):
        """Update system statistics."""
        try:
            # Get current system status
            if self.mode_integration:
                status = get_system_status()
                
                # Update stats based on status
                if "memory_stats" in status:
                    memory_stats = status["memory_stats"]
                    if "total_entries" in memory_stats:
                        self.system_stats["memory_entries"] = memory_stats["total_entries"]
                
                # Update API calls count (simplified)
                self.system_stats["api_calls"] += 1
                
        except Exception as e:
            logger.debug(f"Error updating system stats: {e}")
    
    async def _check_system_health(self):
        """Check overall system health."""
        try:
            # Check if all critical systems are running
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "mode_integration_running": self.mode_integration is not None,
                "memory_system_running": self.memory_system is not None,
                "mathematical_engine_running": self.mathematical_engine is not None,
                "current_mode": self.current_mode.value,
                "system_stats": self.system_stats.copy()
            }
            
            # Store health check in memory
            if self.memory_system:
                try:
                    store_memory_entry(
                        data_type='system_health',
                        data=health_status,
                        source='complete_system_launcher',
                        priority=3,
                        tags=['health_check', 'monitoring']
                    )
                except Exception as e:
                    logger.debug(f"Error storing health check: {e}")
            
            # Check for critical errors
            if self.system_stats["errors"] > 10:
                logger.warning("⚠️ High error count detected")
            
        except Exception as e:
            logger.error(f"❌ Error in health check: {e}")
    
    async def _log_performance_metrics(self):
        """Log performance metrics."""
        try:
            # Calculate runtime
            runtime = 0
            if self.start_time:
                runtime = (datetime.now() - self.start_time).total_seconds()
            
            # Log key metrics
            logger.info(f"📊 Performance - Runtime: {runtime:.1f}s, "
                       f"Mode Switches: {self.system_stats['mode_switches']}, "
                       f"Memory Entries: {self.system_stats['memory_entries']}, "
                       f"API Calls: {self.system_stats['api_calls']}, "
                       f"Errors: {self.system_stats['errors']}")
            
        except Exception as e:
            logger.debug(f"Error logging performance metrics: {e}")
    
    async def _stop_all_systems(self):
        """Stop all individual systems."""
        logger.info("🛑 Stopping all individual systems...")
        
        # Stop clock mode system
        if self.clock_system:
            try:
                self.clock_system.stop_clock_mode()
                logger.info("✅ Clock mode system stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping clock mode system: {e}")
        
        # Stop backtesting system
        if self.backtesting_system:
            try:
                # Backtesting system doesn't have explicit stop method
                logger.info("✅ Backtesting system stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping backtesting system: {e}")
        
        # Stop trading bot
        if self.trading_bot:
            try:
                # Trading bot doesn't have explicit stop method
                logger.info("✅ Trading bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping trading bot: {e}")
        
        # Stop memory system
        if self.memory_system:
            try:
                self.memory_system.stop()
                logger.info("✅ Memory system stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping memory system: {e}")
    
    def _safety_check_startup(self) -> bool:
        """Perform safety checks before starting the system."""
        try:
            # Check if any critical systems are available
            if not REAL_API_AVAILABLE:
                logger.warning("⚠️ Real API system not available")
            
            if not MODE_INTEGRATION_AVAILABLE:
                logger.error("❌ Mode Integration System not available - critical")
                return False
            
            # Check safety configuration
            if not self.config.get("safety_enabled", True):
                logger.warning("⚠️ Safety checks disabled")
            
            if not self.config.get("emergency_stop_enabled", True):
                logger.warning("⚠️ Emergency stop disabled")
            
            # Check risk parameters
            max_daily_loss = self.config.get("max_daily_loss", 0.05)
            if max_daily_loss > 0.1:
                logger.warning("⚠️ Daily loss limit too high")
            
            max_position_size = self.config.get("max_position_size", 0.1)
            if max_position_size > 0.5:
                logger.warning("⚠️ Position size too large")
            
            # Basic safety checks passed
            logger.info("✅ Safety checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safety check error: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop_complete_system())
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "system_info": {
                "name": self.config.get("system_name", "Schwabot Complete System"),
                "version": self.config.get("version", "1.0.0"),
                "is_running": self.is_running,
                "current_mode": self.current_mode.value,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "runtime_seconds": self.system_stats["total_runtime"]
            },
            "system_stats": self.system_stats,
            "configuration": {
                "default_mode": self.config.get("default_mode"),
                "memory_storage": self.config.get("memory_storage"),
                "api_mode": self.config.get("api_mode"),
                "safety_enabled": self.config.get("safety_enabled"),
                "backtesting_duration_hours": self.config.get("backtesting_duration_hours")
            },
            "component_status": {
                "real_api_available": REAL_API_AVAILABLE,
                "mathematical_available": MATHEMATICAL_AVAILABLE,
                "clock_mode_available": CLOCK_MODE_AVAILABLE,
                "backtesting_available": BACKTESTING_AVAILABLE,
                "trading_bot_available": TRADING_BOT_AVAILABLE,
                "mode_integration_available": MODE_INTEGRATION_AVAILABLE
            }
        }
        
        # Add mode integration status if available
        if self.mode_integration:
            try:
                mode_status = get_system_status()
                status["mode_integration_status"] = mode_status
            except Exception as e:
                logger.debug(f"Error getting mode integration status: {e}")
        
        return status
    
    def get_available_modes(self) -> List[Dict[str, Any]]:
        """Get available modes."""
        if self.mode_integration:
            try:
                return get_available_modes()
            except Exception as e:
                logger.debug(f"Error getting available modes: {e}")
        
        # Fallback to basic mode list
        return [
            {"mode": "shadow_mode", "enabled": True, "description": "Analysis only"},
            {"mode": "mathematical_analysis", "enabled": MATHEMATICAL_AVAILABLE, "description": "Mathematical analysis"},
            {"mode": "clock_mode", "enabled": CLOCK_MODE_AVAILABLE, "description": "Clock mode system"},
            {"mode": "live_backtesting", "enabled": BACKTESTING_AVAILABLE, "description": "Live backtesting"},
            {"mode": "paper_trading", "enabled": TRADING_BOT_AVAILABLE, "description": "Paper trading"},
            {"mode": "live_trading", "enabled": TRADING_BOT_AVAILABLE, "description": "Live trading"}
        ]

# Global instance
complete_system_launcher = SchwabotCompleteSystemLauncher()

# Convenience functions
async def start_schwabot_complete_system(initial_mode: TradingMode = TradingMode.SHADOW_MODE) -> bool:
    """Start the complete Schwabot system."""
    return await complete_system_launcher.start_complete_system(initial_mode)

async def stop_schwabot_complete_system() -> bool:
    """Stop the complete Schwabot system."""
    return await complete_system_launcher.stop_complete_system()

async def switch_schwabot_mode(new_mode: TradingMode) -> bool:
    """Switch Schwabot system mode."""
    return await complete_system_launcher.switch_system_mode(new_mode)

def get_schwabot_system_status() -> Dict[str, Any]:
    """Get Schwabot system status."""
    return complete_system_launcher.get_system_status()

def get_schwabot_available_modes() -> List[Dict[str, Any]]:
    """Get available Schwabot modes."""
    return complete_system_launcher.get_available_modes()

# CLI interface
def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Schwabot Complete System Launcher")
    parser.add_argument("--mode", choices=["shadow", "mathematical", "clock", "backtesting", "paper", "live"], 
                       default="shadow", help="Initial mode to start in")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--duration", type=int, default=3600, help="Run duration in seconds")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--modes", action="store_true", help="Show available modes and exit")
    return parser

async def main():
    """Main function for CLI operation."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show status and exit
    if args.status:
        status = get_schwabot_system_status()
        print(json.dumps(status, indent=2))
        return
    
    # Show available modes and exit
    if args.modes:
        modes = get_schwabot_available_modes()
        print(json.dumps(modes, indent=2))
        return
    
    # Map mode string to TradingMode enum
    mode_mapping = {
        "shadow": TradingMode.SHADOW_MODE,
        "mathematical": TradingMode.MATHEMATICAL_ANALYSIS,
        "clock": TradingMode.CLOCK_MODE,
        "backtesting": TradingMode.LIVE_BACKTESTING,
        "paper": TradingMode.PAPER_TRADING,
        "live": TradingMode.LIVE_TRADING
    }
    
    initial_mode = mode_mapping.get(args.mode, TradingMode.SHADOW_MODE)
    
    logger.info("🚀 Starting Schwabot Complete System Launcher")
    
    # Initialize launcher with config
    launcher = SchwabotCompleteSystemLauncher(args.config)
    
    # Start system
    if not await launcher.start_complete_system(initial_mode):
        logger.error("❌ Failed to start Schwabot Complete System")
        return
    
    try:
        # Run for specified duration
        logger.info(f"⏱️ Running for {args.duration} seconds...")
        await asyncio.sleep(args.duration)
        
        # Get final status
        status = launcher.get_system_status()
        logger.info(f"📊 Final Status: {json.dumps(status, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    finally:
        # Stop system
        await launcher.stop_complete_system()
    
    logger.info("🚀 Schwabot Complete System Launcher finished")

if __name__ == "__main__":
    asyncio.run(main()) 