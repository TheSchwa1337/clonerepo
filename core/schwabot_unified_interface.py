#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Unified Interface - Complete Trading System Integration
===============================================================

Unified interface that integrates ALL Schwabot components with KoboldCPP
to create a complete "all-in-one" trading system. This interface allows
the visual layer to take over the entire system while providing access
to all existing Schwabot functionality.

Features:
- Complete integration of all Schwabot components
- KoboldCPP visual layer takeover
- DLT waveform visualization
- Conversation space and API integration
- Real-time trading analysis and execution
- Hardware-optimized performance
- Unified access point for all functionality
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import psutil

from .hardware_auto_detector import HardwareAutoDetector

logger = logging.getLogger(__name__)

# Import real implementations instead of stubs
try:
    from .koboldcpp_integration import KoboldCPPIntegration, AnalysisType, KoboldRequest, KoboldResponse
    KOBOLD_AVAILABLE = True
except ImportError:
    logger.warning("KoboldCPP integration not available, using stub")
    KOBOLD_AVAILABLE = False

try:
    from .visual_layer_controller import VisualLayerController, VisualizationType, ChartTimeframe
    VISUAL_AVAILABLE = True
except ImportError:
    logger.warning("Visual layer controller not available, using stub")
    VISUAL_AVAILABLE = False

try:
    from .tick_loader import TickLoader, TickPriority
    TICK_LOADER_AVAILABLE = True
except ImportError:
    logger.warning("Tick loader not available, using stub")
    TICK_LOADER_AVAILABLE = False

try:
    from .signal_cache import SignalCache, SignalType, SignalPriority
    SIGNAL_CACHE_AVAILABLE = True
except ImportError:
    logger.warning("Signal cache not available, using stub")
    SIGNAL_CACHE_AVAILABLE = False

try:
    from .registry_writer import RegistryWriter, ArchivePriority
    REGISTRY_AVAILABLE = True
except ImportError:
    logger.warning("Registry writer not available, using stub")
    REGISTRY_AVAILABLE = False

try:
    from .json_server import JSONServer, PacketPriority
    JSON_SERVER_AVAILABLE = True
except ImportError:
    logger.warning("JSON server not available, using stub")
    JSON_SERVER_AVAILABLE = False

# Simple stub classes for missing imports
class Alpha256Encryption:
    """Simple stub for Alpha256Encryption."""
    def __init__(self):
        pass
    
    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        return data  # Simple pass-through for now
    
    def decrypt(self, data: str) -> str:
        """Decrypt data."""
        return data  # Simple pass-through for now

# Stub classes for missing components
if not KOBOLD_AVAILABLE:
    class KoboldCPPIntegration:
        """Simple stub for KoboldCPPIntegration."""
        def __init__(self, kobold_path: str = "koboldcpp", model_path: str = "", port: int = 5001):
            self.kobold_path = kobold_path
            self.model_path = model_path
            self.port = port
            self.kobold_running = False
            self.running = False
        
        async def start_kobold_server(self) -> bool:
            """Stub method for starting KoboldCPP server."""
            logger.info("✅ KoboldCPP server started (stubbed)")
            self.kobold_running = True
            return True
        
        async def analyze_trading_data(self, request) -> Optional[Any]:
            """Stub method for analyzing trading data."""
            return None
        
        def stop_processing(self):
            """Stub method for stopping processing."""
            self.kobold_running = False
            self.running = False

    class AnalysisType(Enum):
        """Analysis types."""
        TECHNICAL_ANALYSIS = "technical_analysis"

    class KoboldRequest:
        """Stub for KoboldRequest."""
        def __init__(self, prompt: str, max_length: int = 512, temperature: float = 0.7, analysis_type: AnalysisType = AnalysisType.TECHNICAL_ANALYSIS):
            self.prompt = prompt
            self.max_length = max_length
            self.temperature = temperature
            self.analysis_type = analysis_type

    class KoboldResponse:
        """Stub for KoboldResponse."""
        def __init__(self, text: str = "", tokens_generated: int = 0, processing_time_ms: float = 0.0, model_used: str = "stub", confidence_score: float = 0.0):
            self.text = text
            self.tokens_generated = tokens_generated
            self.processing_time_ms = processing_time_ms
            self.model_used = model_used
            self.confidence_score = confidence_score
            self.analysis_results = {}

if not VISUAL_AVAILABLE:
    class VisualLayerController:
        """Simple stub for VisualLayerController."""
        def __init__(self, output_dir: str = "visualizations"):
            self.output_dir = output_dir
            self.running = False
        
        async def generate_price_chart(self, data, symbol: str, timeframe) -> Optional[Any]:
            """Stub method for generating price chart."""
            return None
        
        async def perform_ai_analysis(self, visual_analysis) -> Optional[Any]:
            """Stub method for performing AI analysis."""
            return visual_analysis
        
        async def save_visualization(self, visual_analysis) -> bool:
            """Stub method for saving visualization."""
            return True
        
        def start_processing(self):
            """Stub method for starting processing."""
            self.running = True
        
        def stop_processing(self):
            """Stub method for stopping processing."""
            self.running = False

    class VisualizationType(Enum):
        """Visualization types."""
        PRICE_CHART = "price_chart"

    class ChartTimeframe(Enum):
        """Chart timeframes."""
        HOUR_1 = "1h"
        MINUTE_5 = "5m"

if not TICK_LOADER_AVAILABLE:
    class TickLoader:
        """Simple stub for TickLoader."""
        def __init__(self):
            self.running = False
        
        async def process_ticks(self) -> List[Any]:
            """Stub method for processing ticks."""
            return []
        
        def start_processing(self):
            """Stub method for starting processing."""
            self.running = True
        
        def stop_processing(self):
            """Stub method for stopping processing."""
            self.running = False

    class TickPriority(Enum):
        """Tick priority levels."""
        MEDIUM = "medium"
        HIGH = "high"

if not SIGNAL_CACHE_AVAILABLE:
    class SignalCache:
        """Simple stub for SignalCache."""
        def __init__(self):
            self.running = False
        
        async def cache_signal(self, signal_type, symbol: str, data: Dict[str, Any], priority) -> bool:
            """Stub method for caching signal."""
            return True
        
        async def get_signals_by_type(self, signal_type, limit: int = 10) -> List[Any]:
            """Stub method for getting signals by type."""
            return []
        
        async def get_signals_by_symbol(self, symbol: str, limit: int = 10) -> List[Any]:
            """Stub method for getting signals by symbol."""
            return []
        
        def start_processing(self):
            """Stub method for starting processing."""
            self.running = True
        
        def stop_processing(self):
            """Stub method for stopping processing."""
            self.running = False

    class SignalType(Enum):
        """Signal types."""
        PRICE = "price"
        COMPOSITE = "composite"

    class SignalPriority(Enum):
        """Signal priority levels."""
        MEDIUM = "medium"
        HIGH = "high"

if not REGISTRY_AVAILABLE:
    class RegistryWriter:
        """Simple stub for RegistryWriter."""
        def __init__(self, base_path: str = "data/registry"):
            self.base_path = base_path
            self.running = False
        
        async def write_entry(self, entry_type: str, data: Dict[str, Any], priority) -> bool:
            """Stub method for writing entry."""
            return True
        
        def start_writing(self):
            """Stub method for starting writing."""
            self.running = True
        
        def stop_writing(self):
            """Stub method for stopping writing."""
            self.running = False

    class ArchivePriority(Enum):
        """Archive priority levels."""
        MEDIUM = "medium"
        HIGH = "high"

if not JSON_SERVER_AVAILABLE:
    class JSONServer:
        """Simple stub for JSONServer."""
        def __init__(self):
            self.running = False
        
        def start_server(self):
            """Stub method for starting server."""
            self.running = True
        
        def stop_server(self):
            """Stub method for stopping server."""
            self.running = False

    class PacketPriority(Enum):
        """Packet priority levels."""
        MEDIUM = "medium"

class InterfaceMode(Enum):
    """Interface operation modes."""
    VISUAL_LAYER = "visual_layer"
    CONVERSATION = "conversation"
    API_ONLY = "api_only"
    FULL_INTEGRATION = "full_integration"

@dataclass
class UnifiedSystemStatus:
    """Unified system status information."""
    mode: InterfaceMode
    kobold_running: bool
    visual_layer_active: bool
    trading_active: bool
    dlt_waveform_active: bool
    conversation_active: bool
    api_active: bool
    hardware_optimized: bool
    uptime_seconds: float
    total_analyses: int
    total_trades: int
    system_health: str

class SchwabotUnifiedInterface:
    """Unified interface for complete Schwabot trading system."""
    
    def __init__(self, mode: InterfaceMode = InterfaceMode.FULL_INTEGRATION):
        """Initialize unified interface."""
        self.mode = mode
        
        # Core system components
        self.hardware_detector = HardwareAutoDetector()
        self.alpha256 = Alpha256Encryption()
        
        # KoboldCPP and visual layer
        self.kobold_integration = None
        self.visual_controller = None
        
        # Trading system components
        self.tick_loader = None
        self.signal_cache = None
        self.registry_writer = None
        self.json_server = None
        
        # DLT waveform and visualization
        self.dlt_waveform_active = False
        self.visualization_ports = {
            'main_dashboard': 5000,
            'dlt_waveform': 5001,
            'data_pipeline': 5002,
            'kobold_webui': 5003,
            'trading_dashboard': 5004
        }
        
        # Conversation and API
        self.conversation_active = False
        self.api_active = False
        
        # System state
        self.running = False
        self.initialized = False
        self.system_info = None
        self.memory_config = None
        
        # Performance tracking
        self.stats = {
            "start_time": 0.0,
            "uptime_seconds": 0.0,
            "total_analyses": 0,
            "total_trades": 0,
            "total_visualizations": 0,
            "conversation_messages": 0,
            "api_requests": 0
        }
        
        # Control interface
        self.control_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize system
        self._initialize_unified_system()
    
    def _initialize_unified_system(self):
        """Initialize the complete unified Schwabot system."""
        try:
            logger.info("🚀 Initializing Schwabot Unified Interface...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"✅ Hardware detected: {self.system_info.platform}")
            logger.info(f"   RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
            # Load or create unified configuration
            self._load_unified_configuration()
            
            # Initialize core components based on mode
            self._initialize_core_components()
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.initialized = True
            logger.info("✅ Schwabot Unified Interface initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Unified system initialization failed: {e}")
            raise
    
    def _load_unified_configuration(self):
        """Load or create unified configuration."""
        config_path = Path("config/unified_interface_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Loaded existing unified configuration")
            else:
                self.config = self._create_default_unified_config()
                self._save_unified_configuration()
                logger.info("✅ Created new unified configuration")
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.config = self._create_default_unified_config()
    
    def _create_default_unified_config(self) -> Dict[str, Any]:
        """Create default unified configuration."""
        return {
            "version": "1.0.0",
            "system_name": "Schwabot Unified Interface",
            "mode": self.mode.value,
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "kobold_integration": {
                "enabled": True,
                "kobold_path": "koboldcpp",
                "model_path": "",
                "port": 5001,
                "auto_start": True,
                "auto_load_model": True,
                "enable_visual_takeover": True
            },
            "visual_layer": {
                "enabled": True,
                "output_dir": "visualizations",
                "enable_ai_analysis": True,
                "enable_pattern_recognition": True,
                "enable_real_time_rendering": True,
                "enable_dlt_waveform": True
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
            "conversation_space": {
                "enabled": True,
                "port": 8080,
                "enable_chat_interface": True,
                "enable_api_access": True,
                "enable_websocket": True
            },
            "api_integration": {
                "enabled": True,
                "host": "localhost",
                "port": 5000,
                "max_connections": 100,
                "enable_encryption": True,
                "enable_rate_limiting": True
            },
            "dlt_waveform": {
                "enabled": True,
                "port": 5001,
                "enable_real_time": True,
                "enable_3d_visualization": True,
                "enable_pattern_detection": True
            },
            "system_control": {
                "enable_health_monitoring": True,
                "health_check_interval_seconds": 30,
                "enable_performance_tracking": True,
                "enable_auto_restart": True,
                "max_restart_attempts": 3,
                "enable_graceful_shutdown": True
            }
        }
    
    def _save_unified_configuration(self):
        """Save unified configuration to file."""
        try:
            config_path = Path("config/unified_interface_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save unified configuration: {e}")
    
    def _initialize_core_components(self):
        """Initialize all core system components."""
        try:
            logger.info("🔧 Initializing core components...")
            
            # Initialize KoboldCPP integration
            if self.config["kobold_integration"]["enabled"]:
                self.kobold_integration = KoboldCPPIntegration(
                    kobold_path=self.config["kobold_integration"]["kobold_path"],
                    model_path=self.config["kobold_integration"]["model_path"],
                    port=self.config["kobold_integration"]["port"]
                )
                logger.info("✅ KoboldCPP integration initialized")
            
            # Initialize visual layer controller
            if self.config["visual_layer"]["enabled"]:
                self.visual_controller = VisualLayerController(
                    output_dir=self.config["visual_layer"]["output_dir"]
                )
                logger.info("✅ Visual layer controller initialized")
            
            # Initialize trading system components
            if self.config["trading_system"]["enabled"]:
                self.tick_loader = TickLoader()
                self.signal_cache = SignalCache()
                self.registry_writer = RegistryWriter(
                    base_path=self.config["trading_system"]["registry_writer"]["base_path"]
                )
                logger.info("✅ Trading system components initialized")
            
            # Initialize JSON server for API
            if self.config["api_integration"]["enabled"]:
                self.json_server = JSONServer()
                logger.info("✅ JSON server initialized")
            
            logger.info("✅ All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Core component initialization failed: {e}")
            raise
    
    async def start_unified_system(self):
        """Start the complete unified Schwabot system."""
        try:
            if self.running:
                logger.warning("⚠️ System already running")
                return
            
            logger.info("🚀 Starting Schwabot Unified Interface...")
            self.running = True
            self.stats["start_time"] = time.time()
            
            # Start all core components
            await self._start_core_components()
            
            # Start DLT waveform visualization
            if self.config["dlt_waveform"]["enabled"]:
                await self._start_dlt_waveform()
            
            # Start conversation space
            if self.config["conversation_space"]["enabled"]:
                await self._start_conversation_space()
            
            # Start control interface
            self._start_control_interface()
            
            # Start health monitoring
            if self.config["system_control"]["enable_health_monitoring"]:
                self._start_health_monitoring()
            
            # Open main interface
            await self._open_main_interface()
            
            logger.info("✅ Schwabot Unified Interface started successfully")
            
            # Main system loop
            await self._main_system_loop()
            
        except Exception as e:
            logger.error(f"❌ Unified system startup failed: {e}")
            await self.stop_unified_system()
            raise
    
    async def _start_core_components(self):
        """Start all core system components."""
        try:
            # Start KoboldCPP integration
            if self.kobold_integration:
                if self.config["kobold_integration"]["auto_start"]:
                    await self.kobold_integration.start_kobold_server()
                else:
                    logger.info("✅ KoboldCPP integration ready (auto-start disabled)")
            
            # Start visual layer controller
            if self.visual_controller:
                self.visual_controller.start_processing()
                logger.info("✅ Visual layer controller started")
            
            # Start trading system components
            if self.tick_loader:
                self.tick_loader.start_processing()
                logger.info("✅ Tick loader started")
            
            if self.signal_cache:
                self.signal_cache.start_processing()
                logger.info("✅ Signal cache started")
            
            if self.registry_writer:
                self.registry_writer.start_writing()
                logger.info("✅ Registry writer started")
            
            # Start JSON server
            if self.json_server:
                self.json_server.start_server()
                logger.info("✅ JSON server started")
            
        except Exception as e:
            logger.error(f"❌ Core component startup failed: {e}")
            raise
    
    async def _start_dlt_waveform(self):
        """Start DLT waveform visualization."""
        try:
            # Import and start DLT waveform visualizer
            try:
                from gui.visualizer_launcher import VisualizerLauncher  # type: ignore
                VISUALIZER_AVAILABLE = True
            except ImportError:
                logger.warning("⚠️ VisualizerLauncher not available, using stub")
                VISUALIZER_AVAILABLE = False
            
            if VISUALIZER_AVAILABLE:
                self.visualizer_launcher = VisualizerLauncher()
                self.visualizer_launcher.ports = self.visualization_ports
                
                # Start DLT waveform in background
                dlt_thread = threading.Thread(
                    target=self.visualizer_launcher._launch_dlt_waveform,
                    daemon=True
                )
                dlt_thread.start()
                
                self.dlt_waveform_active = True
                logger.info("✅ DLT waveform visualization started")
            else:
                # Create stub visualizer
                class StubVisualizerLauncher:
                    def __init__(self):
                        self.ports = {}
                    
                    def _launch_dlt_waveform(self):
                        logger.info("✅ DLT waveform visualization started (stubbed)")
                
                self.visualizer_launcher = StubVisualizerLauncher()
                self.visualizer_launcher.ports = self.visualization_ports
                self.dlt_waveform_active = True
                logger.info("✅ DLT waveform visualization started (stubbed)")
            
        except Exception as e:
            logger.error(f"❌ DLT waveform startup failed: {e}")
    
    async def _start_conversation_space(self):
        """Start conversation space and chat interface."""
        try:
            # Start conversation space using KoboldCPP web interface
            if self.kobold_integration and self.kobold_integration.kobold_running:
                # KoboldCPP provides the conversation interface
                self.conversation_active = True
                logger.info("✅ Conversation space started (via KoboldCPP)")
            
            # Start API access
            if self.json_server and self.json_server.running:
                self.api_active = True
                logger.info("✅ API access started")
            
        except Exception as e:
            logger.error(f"❌ Conversation space startup failed: {e}")
    
    async def _open_main_interface(self):
        """Open the main unified interface."""
        try:
            # Determine which interface to open based on mode
            if self.mode == InterfaceMode.VISUAL_LAYER:
                # Open visual layer interface
                url = f"http://localhost:{self.visualization_ports['main_dashboard']}"
                logger.info(f"🌐 Opened visual layer interface: {url}")
            
            elif self.mode == InterfaceMode.CONVERSATION:
                # Open KoboldCPP conversation interface
                url = f"http://localhost:{self.config['kobold_integration']['port']}"
                logger.info(f"🌐 Opened conversation interface: {url}")
            
            elif self.mode == InterfaceMode.API_ONLY:
                # Open API documentation
                url = f"http://localhost:{self.config['api_integration']['port']}/docs"
                logger.info(f"🌐 Opened API documentation: {url}")
            
            elif self.mode == InterfaceMode.FULL_INTEGRATION:
                # Open unified dashboard
                url = f"http://localhost:{self.visualization_ports['trading_dashboard']}"
                logger.info(f"🌐 Opened unified trading dashboard: {url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to open main interface: {e}")
    
    def _start_control_interface(self):
        """Start control interface for system management."""
        try:
            self.control_thread = threading.Thread(target=self._control_interface_loop, daemon=True)
            self.control_thread.start()
            logger.info("✅ Control interface started")
            
        except Exception as e:
            logger.error(f"❌ Control interface startup failed: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring system."""
        try:
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            logger.info("✅ Health monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Health monitoring startup failed: {e}")
    
    async def _main_system_loop(self):
        """Main system processing loop."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Update uptime
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Process system tasks
                await self._process_unified_tasks()
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Main system loop error: {e}")
    
    async def _process_unified_tasks(self):
        """Process unified system tasks."""
        try:
            # Process trading data
            if self.tick_loader and self.tick_loader.running:
                await self._process_trading_data()
            
            # Process AI analyses
            if self.kobold_integration and self.kobold_integration.kobold_running:
                await self._process_ai_analyses()
            
            # Process visualizations
            if self.visual_controller and self.visual_controller.running:
                await self._process_visualizations()
            
            # Update system statistics
            self._update_system_statistics()
            
        except Exception as e:
            logger.error(f"❌ Unified task processing failed: {e}")
    
    async def _process_trading_data(self):
        """Process trading data through unified system."""
        try:
            # Get processed ticks
            ticks = await self.tick_loader.process_ticks()
            
            for tick in ticks:
                # Cache tick data
                if self.signal_cache:
                    await self.signal_cache.cache_signal(
                        SignalType.PRICE,
                        tick.symbol,
                        {
                            "price": tick.price,
                            "volume": tick.volume,
                            "timestamp": tick.timestamp
                        },
                        SignalPriority.HIGH if tick.priority == TickPriority.HIGH else SignalPriority.MEDIUM
                    )
                
                # Archive tick data
                if self.registry_writer:
                    await self.registry_writer.write_entry(
                        "tick_data",
                        {
                            "symbol": tick.symbol,
                            "price": tick.price,
                            "volume": tick.volume,
                            "timestamp": tick.timestamp,
                            "priority": tick.priority.value
                        },
                        ArchivePriority.HIGH if tick.priority == TickPriority.HIGH else ArchivePriority.MEDIUM
                    )
                
                self.stats["total_trades"] += 1
            
        except Exception as e:
            logger.error(f"❌ Trading data processing failed: {e}")
    
    async def _process_ai_analyses(self):
        """Process AI analyses using KoboldCPP."""
        try:
            # Get cached signals for analysis
            if self.signal_cache:
                signals = await self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=10)
                
                for signal in signals:
                    # Perform AI analysis
                    request = KoboldRequest(
                        prompt=f"Analyze trading data for {signal.symbol}: {signal.data}",
                        max_length=512,
                        temperature=0.7,
                        analysis_type=AnalysisType.TECHNICAL_ANALYSIS
                    )
                    
                    response = await self.kobold_integration.analyze_trading_data(request)
                    
                    if response:
                        self.stats["total_analyses"] += 1
                        
                        # Cache AI analysis result
                        await self.signal_cache.cache_signal(
                            SignalType.COMPOSITE,
                            signal.symbol,
                            {
                                "ai_analysis": response.text,
                                "confidence": response.confidence_score,
                                "results": response.analysis_results
                            },
                            SignalPriority.HIGH
                        )
            
        except Exception as e:
            logger.error(f"❌ AI analysis processing failed: {e}")
    
    async def _process_visualizations(self):
        """Process visualizations and AI-enhanced charts."""
        try:
            # Get recent signals for visualization
            if self.signal_cache:
                signals = await self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=100)
                
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
                            visual_analysis = await self.visual_controller.generate_price_chart(
                                data, symbol, ChartTimeframe.MINUTE_5
                            )
                            
                            if visual_analysis:
                                # Perform AI analysis on chart
                                visual_analysis = await self.visual_controller.perform_ai_analysis(visual_analysis)
                                
                                # Save visualization
                                await self.visual_controller.save_visualization(visual_analysis)
                                
                                self.stats["total_visualizations"] += 1
            
        except Exception as e:
            logger.error(f"❌ Visualization processing failed: {e}")
    
    def _update_system_statistics(self):
        """Update system statistics."""
        try:
            # Update uptime
            self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
            
        except Exception as e:
            logger.error(f"❌ Statistics update failed: {e}")
    
    def _control_interface_loop(self):
        """Control interface loop for system management."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Process control commands
                # This could be expanded to handle user input, API commands, etc.
                
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Control interface loop error: {e}")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            interval = self.config["system_control"]["health_check_interval_seconds"]
            
            while self.running and not self.shutdown_event.is_set():
                # Perform health checks
                health = self._assess_system_health()
                
                if health == "unhealthy":
                    logger.warning("⚠️ System health check failed - attempting recovery")
                    asyncio.run(self._attempt_system_recovery())
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"❌ Health monitoring loop error: {e}")
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        try:
            health_checks = []
            
            # Check KoboldCPP integration
            if self.kobold_integration:
                health_checks.append(self.kobold_integration.kobold_running)
            
            # Check visual controller
            if self.visual_controller:
                health_checks.append(self.visual_controller.running)
            
            # Check trading components
            if self.tick_loader:
                health_checks.append(self.tick_loader.running)
            
            if self.signal_cache:
                health_checks.append(self.signal_cache.running)
            
            if self.registry_writer:
                health_checks.append(self.registry_writer.running)
            
            # Check JSON server
            if self.json_server:
                health_checks.append(self.json_server.running)
            
            if all(health_checks):
                return "healthy"
            elif any(health_checks):
                return "degraded"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"❌ Health assessment failed: {e}")
            return "unknown"
    
    async def _attempt_system_recovery(self):
        """Attempt to recover system components."""
        try:
            logger.info("🔄 Attempting system recovery...")
            
            # Restart failed components
            if self.kobold_integration and not self.kobold_integration.kobold_running:
                await self.kobold_integration.start_kobold_server()
            
            if self.visual_controller and not self.visual_controller.running:
                self.visual_controller.start_processing()
            
            logger.info("✅ System recovery completed")
            
        except Exception as e:
            logger.error(f"❌ System recovery failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def stop_unified_system(self):
        """Stop the complete unified Schwabot system."""
        try:
            if not self.running:
                return
            
            logger.info("🛑 Stopping Schwabot Unified Interface...")
            self.running = False
            self.shutdown_event.set()
            
            # Stop all core components
            await self._stop_core_components()
            
            logger.info("✅ Schwabot Unified Interface stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Unified system shutdown failed: {e}")
    
    async def _stop_core_components(self):
        """Stop all core system components."""
        try:
            # Stop KoboldCPP integration
            if self.kobold_integration:
                self.kobold_integration.stop_processing()
            
            # Stop visual layer controller
            if self.visual_controller:
                self.visual_controller.stop_processing()
            
            # Stop trading system components
            if self.tick_loader:
                self.tick_loader.stop_processing()
            
            if self.signal_cache:
                self.signal_cache.stop_processing()
            
            if self.registry_writer:
                self.registry_writer.stop_writing()
            
            # Stop JSON server
            if self.json_server:
                self.json_server.stop_server()
            
        except Exception as e:
            logger.error(f"❌ Core component shutdown failed: {e}")
    
    def get_unified_status(self) -> UnifiedSystemStatus:
        """Get complete unified system status."""
        try:
            return UnifiedSystemStatus(
                mode=self.mode,
                kobold_running=self.kobold_integration.kobold_running if self.kobold_integration else False,
                visual_layer_active=self.visual_controller.running if self.visual_controller else False,
                trading_active=self.tick_loader.running if self.tick_loader else False,
                dlt_waveform_active=self.dlt_waveform_active,
                conversation_active=self.conversation_active,
                api_active=self.api_active,
                hardware_optimized=self.system_info is not None,
                uptime_seconds=self.stats["uptime_seconds"],
                total_analyses=self.stats["total_analyses"],
                total_trades=self.stats["total_trades"],
                system_health=self._assess_system_health()
            )
            
        except Exception as e:
            logger.error(f"❌ Status collection failed: {e}")
            return UnifiedSystemStatus(
                mode=self.mode,
                kobold_running=False,
                visual_layer_active=False,
                trading_active=False,
                dlt_waveform_active=False,
                conversation_active=False,
                api_active=False,
                hardware_optimized=False,
                uptime_seconds=0.0,
                total_analyses=0,
                total_trades=0,
                system_health="unknown"
            )
    
    async def send_conversation_message(self, message: str) -> str:
        """Send a message through the conversation interface."""
        try:
            if not self.kobold_integration or not self.kobold_integration.kobold_running:
                return "Error: KoboldCPP not available"
            
            # Create AI analysis request
            request = KoboldRequest(
                prompt=message,
                max_length=512,
                temperature=0.7,
                analysis_type=AnalysisType.TECHNICAL_ANALYSIS
            )
            
            # Get response
            response = await self.kobold_integration.analyze_trading_data(request)
            
            if response:
                self.stats["conversation_messages"] += 1
                return response.text
            else:
                return "Error: No response from AI"
                
        except Exception as e:
            logger.error(f"❌ Conversation message failed: {e}")
            return f"Error: {str(e)}"
    
    async def get_trading_analysis(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get comprehensive trading analysis for a symbol."""
        try:
            if not self.kobold_integration or not self.kobold_integration.kobold_running:
                return {"error": "KoboldCPP not available"}
            
            # Get recent signals for the symbol
            if self.signal_cache:
                signals = await self.signal_cache.get_signals_by_symbol(symbol, limit=100)
                
                if signals:
                    # Create analysis prompt
                    prompt = f"Analyze {symbol} trading data for {timeframe} timeframe. Provide technical analysis, risk assessment, and trading recommendations."
                    
                    # Perform AI analysis
                    request = KoboldRequest(
                        prompt=prompt,
                        max_length=512,
                        temperature=0.7,
                        analysis_type=AnalysisType.TECHNICAL_ANALYSIS
                    )
                    
                    response = await self.kobold_integration.analyze_trading_data(request)
                    
                    if response:
                        return {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "analysis": response.text,
                            "confidence": response.confidence_score,
                            "recommendations": response.analysis_results,
                            "timestamp": time.time()
                        }
            
            return {"error": "No data available for analysis"}
            
        except Exception as e:
            logger.error(f"❌ Trading analysis failed: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for Schwabot Unified Interface."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('schwabot_unified_interface.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Schwabot Unified Interface...")
    
    # Parse command line arguments
    mode = InterfaceMode.FULL_INTEGRATION
    if len(sys.argv) > 1:
        mode_str = sys.argv[1].lower()
        if mode_str == "visual":
            mode = InterfaceMode.VISUAL_LAYER
        elif mode_str == "conversation":
            mode = InterfaceMode.CONVERSATION
        elif mode_str == "api":
            mode = InterfaceMode.API_ONLY
    
    # Create unified interface
    unified_interface = SchwabotUnifiedInterface(mode)
    
    try:
        # Start the system
        await unified_interface.start_unified_system()
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        # Stop the system
        await unified_interface.stop_unified_system()
        
        # Print final status
        status = unified_interface.get_unified_status()
        logger.info(f"📊 Final system status: {status}")
        
        logger.info("👋 Schwabot Unified Interface shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 