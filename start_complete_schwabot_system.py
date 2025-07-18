#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Schwabot System Launcher
=================================

This launcher starts the complete Schwabot trading system with:
- Live market data integration from real APIs
- Unified interface with Schwabot AI integration
- Visual layer controller with AI-powered analysis
- Complete trading pipeline integration
- Hardware auto-detection and optimization

This is the ONE command to start everything!
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

# Import our core components
from core.live_market_data_bridge import LiveMarketDataBridge, BridgeMode
from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
from core.visual_layer_controller import VisualLayerController
from core.schwabot_ai_integration import SchwabotAIIntegration
from core.hardware_auto_detector import HardwareAutoDetector

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """System operation modes."""
    COMPLETE = "complete"           # Full system with live data
    UNIFIED = "unified"            # Unified interface only
    VISUAL = "visual"              # Visual layer only
    KOBOLD = "kobold"              # KoboldCPP only
    BRIDGE = "bridge"              # Live market data bridge only
    DEMO = "demo"                  # Demo mode with simulated data

class CompleteSchwabotSystem:
    """Complete Schwabot trading system launcher."""
    
    def __init__(self, mode: SystemMode = SystemMode.COMPLETE):
        """Initialize the complete system."""
        self.mode = mode
        
        # Core system components
        self.hardware_detector = HardwareAutoDetector()
        
        # Main system components
        self.live_market_bridge = None
        self.unified_interface = None
        self.visual_controller = None
        self.schwabot_ai_integration = None
        
        # System state
        self.running = False
        self.initialized = False
        self.system_info = None
        
        # Performance tracking
        self.stats = {
            "start_time": 0.0,
            "uptime_seconds": 0.0,
            "total_components": 0,
            "active_components": 0,
            "system_health": "unknown"
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete system."""
        try:
            logger.info("🚀 Initializing Complete Schwabot System...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            
            logger.info(f"✅ Hardware detected: {self.system_info.platform}")
            logger.info(f"   RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
            # Load or create system configuration
            self._load_system_configuration()
            
            # Initialize components based on mode
            self._initialize_components()
            
            self.initialized = True
            logger.info("✅ Complete Schwabot System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _load_system_configuration(self):
        """Load or create system configuration."""
        config_path = Path("config/complete_system_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Loaded existing system configuration")
            else:
                self.config = self._create_default_system_config()
                self._save_system_configuration()
                logger.info("✅ Created new system configuration")
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.config = self._create_default_system_config()
    
    def _create_default_system_config(self) -> Dict[str, Any]:
        """Create default system configuration."""
        return {
            "version": "1.0.0",
            "system_name": "Complete Schwabot System",
            "mode": self.mode.value,
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "live_market_bridge": {
                "enabled": True,
                "mode": "full_integration",
                "auto_start": True,
                "enable_real_time_data": True,
                "enable_ai_analysis": True,
                "enable_visualization": True
            },
            "unified_interface": {
                "enabled": True,
                "mode": "full_integration",
                "auto_start": True,
                "enable_visual_takeover": True,
                "enable_conversation": True,
                "enable_api_access": True
            },
            "visual_layer": {
                "enabled": True,
                "output_dir": "visualizations",
                "enable_ai_analysis": True,
                "enable_pattern_recognition": True,
                "enable_real_time_rendering": True,
                "enable_dlt_waveform": True
            },
            "schwabot_ai_integration": {
                "enabled": True,
                "schwabot_ai_path": "schwabot_ai",
                "model_path": "",
                "port": 5001,
                "auto_start": True,
                "enable_ai_analysis": True,
                "enable_conversation": True
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
    
    def _save_system_configuration(self):
        """Save system configuration to file."""
        try:
            config_path = Path("config/complete_system_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save system configuration: {e}")
    
    def _initialize_components(self):
        """Initialize system components based on mode."""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Initialize live market data bridge
            if self.config["live_market_bridge"]["enabled"]:
                bridge_mode = BridgeMode.FULL_INTEGRATION if self.config["live_market_bridge"]["mode"] == "full_integration" else BridgeMode.MARKET_DATA_ONLY
                self.live_market_bridge = LiveMarketDataBridge(bridge_mode)
                logger.info("✅ Live market data bridge initialized")
            
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
            
            # Initialize Schwabot AI integration
            if self.config["schwabot_ai_integration"]["enabled"]:
                self.schwabot_ai_integration = SchwabotAIIntegration(
                    schwabot_ai_path=self.config["schwabot_ai_integration"]["schwabot_ai_path"],
                    model_path=self.config["schwabot_ai_integration"]["model_path"],
                    port=self.config["schwabot_ai_integration"]["port"]
                )
                logger.info("✅ Schwabot AI integration initialized")
            
            # Count total components
            self.stats["total_components"] = sum([
                bool(self.live_market_bridge),
                bool(self.unified_interface),
                bool(self.visual_controller),
                bool(self.schwabot_ai_integration)
            ])
            
            logger.info(f"✅ All system components initialized ({self.stats['total_components']} components)")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def start_system(self):
        """Start the complete Schwabot system."""
        try:
            if self.running:
                logger.warning("⚠️ System already running")
                return
            
            logger.info("🚀 Starting Complete Schwabot System...")
            self.running = True
            self.stats["start_time"] = time.time()
            
            # Start all components
            await self._start_components()
            
            # Start health monitoring
            if self.config["system_control"]["enable_health_monitoring"]:
                self._start_health_monitoring()
            
            logger.info("✅ Complete Schwabot System started successfully")
            
            # Print system status
            self._print_system_status()
            
            # Main system loop
            await self._main_system_loop()
            
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            await self.stop_system()
            raise
    
    async def _start_components(self):
        """Start all system components."""
        try:
            active_components = 0
            
            # Start live market data bridge
            if self.live_market_bridge and self.config["live_market_bridge"]["auto_start"]:
                await self.live_market_bridge.start_bridge()
                active_components += 1
                logger.info("✅ Live market data bridge started")
            
            # Start unified interface
            if self.unified_interface and self.config["unified_interface"]["auto_start"]:
                await self.unified_interface.start_unified_system()
                active_components += 1
                logger.info("✅ Unified interface started")
            
            # Start visual layer controller
            if self.visual_controller:
                await self.visual_controller.start_processing()
                active_components += 1
                logger.info("✅ Visual layer controller started")
            
            # Start Schwabot AI integration
            if self.schwabot_ai_integration and self.config["schwabot_ai_integration"]["auto_start"]:
                await self.schwabot_ai_integration.start_schwabot_ai_server()
                active_components += 1
                logger.info("✅ Schwabot AI integration started")
            
            self.stats["active_components"] = active_components
            
        except Exception as e:
            logger.error(f"❌ Component startup failed: {e}")
            raise
    
    def _start_health_monitoring(self):
        """Start health monitoring system."""
        try:
            import threading
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            logger.info("✅ Health monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Health monitoring startup failed: {e}")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            interval = self.config["system_control"]["health_check_interval_seconds"]
            
            while self.running:
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
            
            # Check live market bridge
            if self.live_market_bridge:
                bridge_status = self.live_market_bridge.get_bridge_status()
                health_checks.append(bridge_status.system_health == "healthy")
            
            # Check unified interface
            if self.unified_interface:
                health_checks.append(self.unified_interface.running)
            
            # Check visual controller
            if self.visual_controller:
                health_checks.append(self.visual_controller.running)
            
            # Check Schwabot AI integration
            if self.schwabot_ai_integration:
                health_checks.append(self.schwabot_ai_integration.schwabot_ai_running)
            
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
            if self.live_market_bridge:
                bridge_status = self.live_market_bridge.get_bridge_status()
                if bridge_status.system_health != "healthy":
                    await self.live_market_bridge.stop_bridge()
                    await self.live_market_bridge.start_bridge()
            
            if self.unified_interface and not self.unified_interface.running:
                await self.unified_interface.start_unified_system()
            
            if self.visual_controller and not self.visual_controller.running:
                await self.visual_controller.start_processing()
            
            if self.schwabot_ai_integration and not self.schwabot_ai_integration.schwabot_ai_running:
                await self.schwabot_ai_integration.start_schwabot_ai_server()
            
            logger.info("✅ System recovery completed")
            
        except Exception as e:
            logger.error(f"❌ System recovery failed: {e}")
    
    def _print_system_status(self):
        """Print current system status."""
        try:
            logger.info("\n" + "="*60)
            logger.info("🎯 COMPLETE SCHWABOT SYSTEM STATUS")
            logger.info("="*60)
            
            # System info
            logger.info(f"📊 Mode: {self.mode.value.upper()}")
            logger.info(f"🖥️  Platform: {self.system_info.platform}")
            logger.info(f"💾 RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"⚡ Optimization: {self.system_info.optimization_mode.value}")
            
            # Component status
            logger.info(f"\n🔧 Components: {self.stats['active_components']}/{self.stats['total_components']} active")
            
            if self.live_market_bridge:
                bridge_status = self.live_market_bridge.get_bridge_status()
                logger.info(f"📡 Live Market Bridge: {'✅ RUNNING' if bridge_status.data_flow_active else '❌ STOPPED'}")
                logger.info(f"   Data Points: {bridge_status.total_data_points}")
                logger.info(f"   AI Analyses: {bridge_status.total_analyses}")
                logger.info(f"   Visualizations: {bridge_status.total_visualizations}")
            
            if self.unified_interface:
                logger.info(f"🌐 Unified Interface: {'✅ RUNNING' if self.unified_interface.running else '❌ STOPPED'}")
            
            if self.visual_controller:
                logger.info(f"🎨 Visual Layer: {'✅ RUNNING' if self.visual_controller.running else '❌ STOPPED'}")
            
            if self.schwabot_ai_integration:
                logger.info(f"🤖 KoboldCPP: {'✅ RUNNING' if self.schwabot_ai_integration.schwabot_ai_running else '❌ STOPPED'}")
            
            # System health
            health = self._assess_system_health()
            health_emoji = "✅" if health == "healthy" else "⚠️" if health == "degraded" else "❌"
            logger.info(f"\n🏥 System Health: {health_emoji} {health.upper()}")
            
            # Access URLs
            logger.info(f"\n🌐 Access URLs:")
            logger.info(f"   Schwabot AI Web UI: http://localhost:5001")
            logger.info(f"   Unified Dashboard: http://localhost:5004")
            logger.info(f"   Visual Layer: http://localhost:5000")
            logger.info(f"   DLT Waveform: http://localhost:5001")
            
            logger.info("="*60)
            logger.info("🎉 System ready for trading! Press Ctrl+C to stop.")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Status display failed: {e}")
    
    async def _main_system_loop(self):
        """Main system processing loop."""
        try:
            while self.running:
                # Update uptime
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Update system health
                self.stats["system_health"] = self._assess_system_health()
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Main system loop error: {e}")
    
    async def stop_system(self):
        """Stop the complete Schwabot system."""
        try:
            if not self.running:
                return
            
            logger.info("🛑 Stopping Complete Schwabot System...")
            self.running = False
            
            # Stop all components
            await self._stop_components()
            
            logger.info("✅ Complete Schwabot System stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ System shutdown failed: {e}")
    
    async def _stop_components(self):
        """Stop all system components."""
        try:
            # Stop live market data bridge
            if self.live_market_bridge:
                await self.live_market_bridge.stop_bridge()
            
            # Stop unified interface
            if self.unified_interface:
                await self.unified_interface.stop_unified_system()
            
            # Stop visual layer controller
            if self.visual_controller:
                self.visual_controller.stop_processing()
            
            # Stop Schwabot AI integration
            if self.schwabot_ai_integration:
                self.schwabot_ai_integration.stop_processing()
            
        except Exception as e:
            logger.error(f"❌ Component shutdown failed: {e}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for Complete Schwabot System."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('complete_schwabot_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Complete Schwabot Trading System")
    parser.add_argument("mode", nargs="?", default="complete", 
                       choices=["complete", "unified", "visual", "kobold", "bridge", "demo"],
                       help="System operation mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Map mode string to enum
    mode_map = {
        "complete": SystemMode.COMPLETE,
        "unified": SystemMode.UNIFIED,
        "visual": SystemMode.VISUAL,
        "kobold": SystemMode.KOBOLD,
        "bridge": SystemMode.BRIDGE,
        "demo": SystemMode.DEMO
    }
    
    mode = mode_map.get(args.mode, SystemMode.COMPLETE)
    
    logger.info("🚀 Starting Complete Schwabot Trading System...")
    logger.info(f"📊 Mode: {mode.value.upper()}")
    
    # Create system
    system = CompleteSchwabotSystem(mode)
    
    try:
        # Start the system
        await system.start_system()
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        # Stop the system
        await system.stop_system()
        
        # Print final statistics
        logger.info(f"📊 Final system statistics:")
        logger.info(f"   Uptime: {system.stats['uptime_seconds']:.1f} seconds")
        logger.info(f"   Active Components: {system.stats['active_components']}/{system.stats['total_components']}")
        logger.info(f"   System Health: {system.stats['system_health']}")
        
        logger.info("👋 Complete Schwabot System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 