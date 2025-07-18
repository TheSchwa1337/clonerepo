#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Market Data Bridge - Schwabot Integration Layer
===================================================

This bridge connects the live market data integration with the unified interface
and visual layer controller, ensuring complete end-to-end data flow from real
API feeds to AI-powered analysis and visualization.

Features:
- Real-time market data integration with multiple exchanges
- Seamless connection to unified interface
- Visual layer controller integration
- KoboldCPP AI analysis integration
- Hardware-optimized performance
- Complete trading pipeline integration
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

# Import our core components
from .live_market_data_integration import LiveMarketDataIntegration, MarketData, TradingSignal
from .schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
from .visual_layer_controller import VisualLayerController, VisualizationType, ChartTimeframe
from .koboldcpp_integration import KoboldCPPIntegration, AnalysisType, KoboldRequest
from .tick_loader import TickLoader, TickPriority
from .signal_cache import SignalCache, SignalType, SignalPriority
from .registry_writer import RegistryWriter, ArchivePriority
from .json_server import JSONServer, PacketPriority
from .hardware_auto_detector import HardwareAutoDetector

logger = logging.getLogger(__name__)

class BridgeMode(Enum):
    """Bridge operation modes."""
    FULL_INTEGRATION = "full_integration"
    MARKET_DATA_ONLY = "market_data_only"
    VISUAL_ONLY = "visual_only"
    AI_ANALYSIS_ONLY = "ai_analysis_only"

@dataclass
class BridgeStatus:
    """Bridge status information."""
    mode: BridgeMode
    market_data_running: bool
    unified_interface_running: bool
    visual_layer_running: bool
    kobold_integration_running: bool
    data_flow_active: bool
    uptime_seconds: float
    total_data_points: int
    total_analyses: int
    total_visualizations: int
    system_health: str

class LiveMarketDataBridge:
    """Bridge connecting live market data with unified interface and visual layer."""
    
    def __init__(self, mode: BridgeMode = BridgeMode.FULL_INTEGRATION):
        """Initialize the live market data bridge."""
        self.mode = mode
        
        # Core system components
        self.hardware_detector = HardwareAutoDetector()
        
        # Market data integration
        self.market_data_integration = None
        
        # Unified interface
        self.unified_interface = None
        
        # Visual layer controller
        self.visual_controller = None
        
        # KoboldCPP integration
        self.kobold_integration = None
        
        # Trading system components
        self.tick_loader = None
        self.signal_cache = None
        self.registry_writer = None
        self.json_server = None
        
        # System state
        self.running = False
        self.initialized = False
        self.system_info = None
        
        # Performance tracking
        self.stats = {
            "start_time": 0.0,
            "uptime_seconds": 0.0,
            "total_data_points": 0,
            "total_analyses": 0,
            "total_visualizations": 0,
            "data_flow_errors": 0
        }
        
        # Data flow control
        self.data_flow_active = False
        self.data_processing_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize system
        self._initialize_bridge()
    
    def _initialize_bridge(self):
        """Initialize the complete bridge system."""
        try:
            logger.info("🚀 Initializing Live Market Data Bridge...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            
            logger.info(f"✅ Hardware detected: {self.system_info.platform}")
            logger.info(f"   RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
            # Load or create bridge configuration
            self._load_bridge_configuration()
            
            # Initialize core components based on mode
            self._initialize_core_components()
            
            self.initialized = True
            logger.info("✅ Live Market Data Bridge initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Bridge initialization failed: {e}")
            raise
    
    def _load_bridge_configuration(self):
        """Load or create bridge configuration."""
        config_path = Path("config/live_market_data_bridge_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Loaded existing bridge configuration")
            else:
                self.config = self._create_default_bridge_config()
                self._save_bridge_configuration()
                logger.info("✅ Created new bridge configuration")
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.config = self._create_default_bridge_config()
    
    def _create_default_bridge_config(self) -> Dict[str, Any]:
        """Create default bridge configuration."""
        return {
            "version": "1.0.0",
            "system_name": "Live Market Data Bridge",
            "mode": self.mode.value,
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "market_data_integration": {
                "enabled": True,
                "exchanges": {
                    "coinbase": {
                        "enabled": True,
                        "api_key": "",
                        "secret": "",
                        "password": "",
                        "sandbox": True
                    },
                    "kraken": {
                        "enabled": True,
                        "api_key": "",
                        "secret": "",
                        "sandbox": True
                    },
                    "binance": {
                        "enabled": True,
                        "api_key": "",
                        "secret": "",
                        "sandbox": True
                    },
                    "finance_api": {
                        "enabled": True,
                        "api_key": "",
                        "provider": "alphavantage"
                    }
                },
                "symbols": ["BTC/USDC", "ETH/USDC", "XRP/USDC", "SOL/USDC"],
                "update_interval": 1.0,
                "enable_technical_indicators": True,
                "enable_memory_keys": True
            },
            "unified_interface": {
                "enabled": True,
                "mode": "full_integration",
                "auto_start": True,
                "enable_visual_takeover": True
            },
            "visual_layer": {
                "enabled": True,
                "output_dir": "visualizations",
                "enable_ai_analysis": True,
                "enable_pattern_recognition": True,
                "enable_real_time_rendering": True,
                "chart_update_interval": 5.0
            },
            "kobold_integration": {
                "enabled": True,
                "kobold_path": "koboldcpp",
                "model_path": "",
                "port": 5001,
                "auto_start": True,
                "enable_ai_analysis": True
            },
            "trading_system": {
                "enabled": True,
                "tick_loader": {
                    "max_queue_size": 10000,
                    "enable_compression": True,
                    "enable_encryption": True
                },
                "signal_cache": {
                    "cache_size": 10000,
                    "enable_similarity_matching": True,
                    "enable_confidence_scoring": True
                },
                "registry_writer": {
                    "base_path": "data/registry",
                    "enable_compression": True,
                    "enable_encryption": True,
                    "enable_backup_rotation": True
                }
            },
            "data_flow": {
                "enable_real_time_processing": True,
                "enable_ai_analysis": True,
                "enable_visualization": True,
                "enable_signal_generation": True,
                "enable_memory_storage": True,
                "processing_interval": 1.0
            }
        }
    
    def _save_bridge_configuration(self):
        """Save bridge configuration to file."""
        try:
            config_path = Path("config/live_market_data_bridge_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save bridge configuration: {e}")
    
    def _initialize_core_components(self):
        """Initialize all core system components."""
        try:
            logger.info("🔧 Initializing core components...")
            
            # Initialize market data integration
            if self.config["market_data_integration"]["enabled"]:
                self.market_data_integration = LiveMarketDataIntegration(self.config["market_data_integration"])
                logger.info("✅ Market data integration initialized")
            
            # Initialize unified interface
            if self.config["unified_interface"]["enabled"]:
                interface_mode = InterfaceMode.FULL_INTEGRATION if self.config["unified_interface"]["mode"] == "full_integration" else InterfaceMode.API_ONLY
                self.unified_interface = SchwabotUnifiedInterface(interface_mode)
                logger.info("✅ Unified interface initialized")
            
            # Initialize visual layer controller
            if self.config["visual_layer"]["enabled"]:
                self.visual_controller = VisualLayerController(
                    output_dir=self.config["visual_layer"]["output_dir"]
                )
                logger.info("✅ Visual layer controller initialized")
            
            # Initialize KoboldCPP integration
            if self.config["kobold_integration"]["enabled"]:
                self.kobold_integration = KoboldCPPIntegration(
                    kobold_path=self.config["kobold_integration"]["kobold_path"],
                    model_path=self.config["kobold_integration"]["model_path"],
                    port=self.config["kobold_integration"]["port"]
                )
                logger.info("✅ KoboldCPP integration initialized")
            
            # Initialize trading system components
            if self.config["trading_system"]["enabled"]:
                self.tick_loader = TickLoader()
                self.signal_cache = SignalCache()
                self.registry_writer = RegistryWriter(
                    base_path=self.config["trading_system"]["registry_writer"]["base_path"]
                )
                logger.info("✅ Trading system components initialized")
            
            logger.info("✅ All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Core component initialization failed: {e}")
            raise
    
    async def start_bridge(self):
        """Start the complete live market data bridge."""
        try:
            if self.running:
                logger.warning("⚠️ Bridge already running")
                return
            
            logger.info("🚀 Starting Live Market Data Bridge...")
            self.running = True
            self.stats["start_time"] = time.time()
            
            # Start all core components
            await self._start_core_components()
            
            # Start data flow processing
            if self.config["data_flow"]["enable_real_time_processing"]:
                self._start_data_flow_processing()
            
            logger.info("✅ Live Market Data Bridge started successfully")
            
            # Main bridge loop
            await self._main_bridge_loop()
            
        except Exception as e:
            logger.error(f"❌ Bridge startup failed: {e}")
            await self.stop_bridge()
            raise
    
    async def _start_core_components(self):
        """Start all core system components."""
        try:
            # Start market data integration
            if self.market_data_integration:
                self.market_data_integration.start_data_feed()
                logger.info("✅ Market data integration started")
            
            # Start unified interface
            if self.unified_interface:
                await self.unified_interface.start_unified_system()
                logger.info("✅ Unified interface started")
            
            # Start visual layer controller
            if self.visual_controller:
                await self.visual_controller.start_processing()
                logger.info("✅ Visual layer controller started")
            
            # Start KoboldCPP integration
            if self.kobold_integration:
                if self.config["kobold_integration"]["auto_start"]:
                    await self.kobold_integration.start_kobold_server()
                logger.info("✅ KoboldCPP integration started")
            
            # Start trading system components
            if self.tick_loader:
                await self.tick_loader.start_processing()
                logger.info("✅ Tick loader started")
            
            if self.signal_cache:
                await self.signal_cache.start_processing()
                logger.info("✅ Signal cache started")
            
            if self.registry_writer:
                await self.registry_writer.start_writing()
                logger.info("✅ Registry writer started")
            
        except Exception as e:
            logger.error(f"❌ Core component startup failed: {e}")
            raise
    
    def _start_data_flow_processing(self):
        """Start data flow processing thread."""
        try:
            self.data_processing_thread = threading.Thread(
                target=self._data_flow_loop,
                daemon=True,
                name="DataFlowProcessor"
            )
            self.data_processing_thread.start()
            self.data_flow_active = True
            logger.info("✅ Data flow processing started")
            
        except Exception as e:
            logger.error(f"❌ Data flow processing startup failed: {e}")
    
    def _data_flow_loop(self):
        """Main data flow processing loop."""
        try:
            interval = self.config["data_flow"]["processing_interval"]
            
            while self.running and not self.shutdown_event.is_set():
                # Process market data
                if self.market_data_integration and self.market_data_integration.running:
                    self._process_market_data_flow()
                
                # Process AI analysis
                if self.config["data_flow"]["enable_ai_analysis"]:
                    self._process_ai_analysis_flow()
                
                # Process visualization
                if self.config["data_flow"]["enable_visualization"]:
                    self._process_visualization_flow()
                
                # Process signal generation
                if self.config["data_flow"]["enable_signal_generation"]:
                    self._process_signal_generation_flow()
                
                # Update statistics
                self._update_bridge_statistics()
                
                # Sleep between processing cycles
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"❌ Data flow loop error: {e}")
            self.stats["data_flow_errors"] += 1
    
    def _process_market_data_flow(self):
        """Process market data flow through the bridge."""
        try:
            # Get latest market data from all exchanges
            for exchange_name in self.config["market_data_integration"]["exchanges"]:
                if self.config["market_data_integration"]["exchanges"][exchange_name]["enabled"]:
                    for symbol in self.config["market_data_integration"]["symbols"]:
                        market_data = self.market_data_integration.get_latest_market_data(symbol, exchange_name)
                        
                        if market_data:
                            # Cache market data
                            if self.signal_cache:
                                asyncio.run(self.signal_cache.cache_signal(
                                    SignalType.PRICE,
                                    symbol,
                                    {
                                        "price": market_data.price,
                                        "volume": market_data.volume,
                                        "rsi": market_data.rsi,
                                        "timestamp": market_data.timestamp,
                                        "exchange": market_data.exchange
                                    },
                                    SignalPriority.HIGH
                                ))
                            
                            # Archive market data
                            if self.registry_writer:
                                asyncio.run(self.registry_writer.write_entry(
                                    "market_data",
                                    {
                                        "symbol": symbol,
                                        "exchange": exchange_name,
                                        "price": market_data.price,
                                        "volume": market_data.volume,
                                        "rsi": market_data.rsi,
                                        "timestamp": market_data.timestamp,
                                        "phase": market_data.phase.value,
                                        "strategy_tier": market_data.strategy_tier.value
                                    },
                                    ArchivePriority.HIGH
                                ))
                            
                            self.stats["total_data_points"] += 1
            
        except Exception as e:
            logger.error(f"❌ Market data flow processing failed: {e}")
    
    def _process_ai_analysis_flow(self):
        """Process AI analysis flow through KoboldCPP."""
        try:
            if self.kobold_integration and self.kobold_integration.kobold_running:
                # Get recent market data for analysis
                if self.signal_cache:
                    signals = asyncio.run(self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=10))
                    
                    for signal in signals:
                        # Create AI analysis request
                        request = KoboldRequest(
                            prompt=f"Analyze trading data for {signal.symbol}: price=${signal.data['price']:.2f}, volume={signal.data['volume']:.0f}, RSI={signal.data['rsi']:.2f}. Provide technical analysis and trading recommendations.",
                            max_length=512,
                            temperature=0.7,
                            analysis_type=AnalysisType.TECHNICAL_ANALYSIS
                        )
                        
                        # Perform AI analysis
                        response = asyncio.run(self.kobold_integration.analyze_trading_data(request))
                        
                        if response:
                            # Cache AI analysis result
                            asyncio.run(self.signal_cache.cache_signal(
                                SignalType.COMPOSITE,
                                signal.symbol,
                                {
                                    "ai_analysis": response.text,
                                    "confidence": response.confidence_score,
                                    "results": response.analysis_results,
                                    "timestamp": time.time()
                                },
                                SignalPriority.HIGH
                            ))
                            
                            self.stats["total_analyses"] += 1
            
        except Exception as e:
            logger.error(f"❌ AI analysis flow processing failed: {e}")
    
    def _process_visualization_flow(self):
        """Process visualization flow through visual layer controller."""
        try:
            if self.visual_controller and self.visual_controller.running:
                # Get recent signals for visualization
                if self.signal_cache:
                    signals = asyncio.run(self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=100))
                    
                    if len(signals) >= 20:
                        # Group signals by symbol
                        symbol_data = {}
                        for signal in signals:
                            if signal.symbol not in symbol_data:
                                symbol_data[signal.symbol] = []
                            symbol_data[signal.symbol].append(signal.data)
                        
                        # Generate visualizations for each symbol
                        for symbol, data in symbol_data.items():
                            if len(data) >= 20:
                                # Generate price chart
                                visual_analysis = asyncio.run(self.visual_controller.generate_price_chart(
                                    data, symbol, ChartTimeframe.MINUTE_5
                                ))
                                
                                if visual_analysis:
                                    # Perform AI analysis on chart
                                    visual_analysis = asyncio.run(self.visual_controller.perform_ai_analysis(visual_analysis))
                                    
                                    # Save visualization
                                    asyncio.run(self.visual_controller.save_visualization(visual_analysis))
                                    
                                    self.stats["total_visualizations"] += 1
            
        except Exception as e:
            logger.error(f"❌ Visualization flow processing failed: {e}")
    
    def _process_signal_generation_flow(self):
        """Process trading signal generation flow."""
        try:
            if self.market_data_integration and self.market_data_integration.running:
                # Get trading signals from market data integration
                signals = self.market_data_integration.get_trading_signals(limit=10)
                
                for signal in signals:
                    # Cache trading signal
                    if self.signal_cache:
                        asyncio.run(self.signal_cache.cache_signal(
                            SignalType.COMPOSITE,
                            signal.symbol,
                            {
                                "action": signal.action,
                                "confidence": signal.confidence,
                                "price": signal.price,
                                "amount": signal.amount,
                                "strategy_tier": signal.strategy_tier.value,
                                "phase": signal.phase.value,
                                "priority": signal.priority,
                                "timestamp": signal.timestamp
                            },
                            SignalPriority.HIGH if signal.priority == "critical" else SignalPriority.MEDIUM
                        ))
            
        except Exception as e:
            logger.error(f"❌ Signal generation flow processing failed: {e}")
    
    def _update_bridge_statistics(self):
        """Update bridge statistics."""
        try:
            self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
            
        except Exception as e:
            logger.error(f"❌ Statistics update failed: {e}")
    
    async def _main_bridge_loop(self):
        """Main bridge processing loop."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Update uptime
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Main bridge loop error: {e}")
    
    async def stop_bridge(self):
        """Stop the complete live market data bridge."""
        try:
            if not self.running:
                return
            
            logger.info("🛑 Stopping Live Market Data Bridge...")
            self.running = False
            self.shutdown_event.set()
            
            # Stop all core components
            await self._stop_core_components()
            
            logger.info("✅ Live Market Data Bridge stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Bridge shutdown failed: {e}")
    
    async def _stop_core_components(self):
        """Stop all core system components."""
        try:
            # Stop market data integration
            if self.market_data_integration:
                self.market_data_integration.shutdown()
            
            # Stop unified interface
            if self.unified_interface:
                await self.unified_interface.stop_unified_system()
            
            # Stop visual layer controller
            if self.visual_controller:
                self.visual_controller.stop_processing()
            
            # Stop KoboldCPP integration
            if self.kobold_integration:
                self.kobold_integration.stop_processing()
            
            # Stop trading system components
            if self.tick_loader:
                self.tick_loader.stop_processing()
            
            if self.signal_cache:
                self.signal_cache.stop_processing()
            
            if self.registry_writer:
                self.registry_writer.stop_writing()
            
        except Exception as e:
            logger.error(f"❌ Core component shutdown failed: {e}")
    
    def get_bridge_status(self) -> BridgeStatus:
        """Get complete bridge status."""
        try:
            return BridgeStatus(
                mode=self.mode,
                market_data_running=self.market_data_integration.running if self.market_data_integration else False,
                unified_interface_running=self.unified_interface.running if self.unified_interface else False,
                visual_layer_running=self.visual_controller.running if self.visual_controller else False,
                kobold_integration_running=self.kobold_integration.kobold_running if self.kobold_integration else False,
                data_flow_active=self.data_flow_active,
                uptime_seconds=self.stats["uptime_seconds"],
                total_data_points=self.stats["total_data_points"],
                total_analyses=self.stats["total_analyses"],
                total_visualizations=self.stats["total_visualizations"],
                system_health=self._assess_system_health()
            )
            
        except Exception as e:
            logger.error(f"❌ Status collection failed: {e}")
            return BridgeStatus(
                mode=self.mode,
                market_data_running=False,
                unified_interface_running=False,
                visual_layer_running=False,
                kobold_integration_running=False,
                data_flow_active=False,
                uptime_seconds=0.0,
                total_data_points=0,
                total_analyses=0,
                total_visualizations=0,
                system_health="unknown"
            )
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        try:
            health_checks = []
            
            # Check market data integration
            if self.market_data_integration:
                health_checks.append(self.market_data_integration.running)
            
            # Check unified interface
            if self.unified_interface:
                health_checks.append(self.unified_interface.running)
            
            # Check visual controller
            if self.visual_controller:
                health_checks.append(self.visual_controller.running)
            
            # Check KoboldCPP integration
            if self.kobold_integration:
                health_checks.append(self.kobold_integration.kobold_running)
            
            # Check data flow
            health_checks.append(self.data_flow_active)
            
            if all(health_checks):
                return "healthy"
            elif any(health_checks):
                return "degraded"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"❌ Health assessment failed: {e}")
            return "unknown"

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for Live Market Data Bridge."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('live_market_data_bridge.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Live Market Data Bridge...")
    
    # Create bridge
    bridge = LiveMarketDataBridge(BridgeMode.FULL_INTEGRATION)
    
    try:
        # Start the bridge
        await bridge.start_bridge()
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Bridge error: {e}")
    finally:
        # Stop the bridge
        await bridge.stop_bridge()
        
        # Print final status
        status = bridge.get_bridge_status()
        logger.info(f"📊 Final bridge status: {status}")
        
        logger.info("👋 Live Market Data Bridge shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 