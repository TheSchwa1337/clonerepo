#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Integration Script - Schwabot + KoboldCPP Complete System
===============================================================

This script orchestrates the complete integration between Schwabot's
unified trading system and KoboldCPP's AI interface, providing:

1. **Bridge Layer**: Connects Schwabot to KoboldCPP's existing Flask/HTTP interface
2. **Enhanced Interface**: Extends KoboldCPP with trading-specific functionality
3. **Unified System**: Coordinates all trading components
4. **Visual Layer**: Provides real-time charts and visualizations
5. **Memory Stack**: Manages AI command sequencing and execution
6. **Real-time Integration**: Seamless conversation-to-trading workflow

Usage:
    python master_integration.py [mode] [options]

Modes:
    - full: Complete integration (default)
    - bridge: Bridge only
    - enhanced: Enhanced interface only
    - visual: Visual layer only
    - conversation: Conversation mode only
    - api: API only mode
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
from typing import Any, Dict, List, Optional
from pathlib import Path

# Import our core components
from core.koboldcpp_bridge import KoboldCPPBridge, start_bridge, stop_bridge
from core.koboldcpp_enhanced_interface import KoboldCPPEnhancedInterface, start_enhanced_interface, stop_enhanced_interface
from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
from core.visual_layer_controller import VisualLayerController
from core.koboldcpp_integration import KoboldCPPIntegration
from core.tick_loader import TickLoader
from core.signal_cache import SignalCache
from core.registry_writer import RegistryWriter
from core.json_server import JSONServer

# Import memory stack components
from core.memory_stack.ai_command_sequencer import AICommandSequencer
from core.memory_stack.execution_validator import ExecutionValidator
from core.memory_stack.memory_key_allocator import MemoryKeyAllocator

logger = logging.getLogger(__name__)

class IntegrationMode:
    """Integration modes for the master system."""
    FULL = "full"
    BRIDGE = "bridge"
    ENHANCED = "enhanced"
    VISUAL = "visual"
    CONVERSATION = "conversation"
    API = "api"

class MasterIntegration:
    """Master integration orchestrator for Schwabot + KoboldCPP."""
    
    def __init__(self, mode: str = IntegrationMode.FULL):
        """Initialize the master integration system."""
        self.mode = mode
        self.start_time = datetime.now()
        
        # System state
        self.running = False
        self.components = {}
        self.ports = {
            'kobold': 5001,
            'bridge': 5005,
            'enhanced': 5006,
            'visual': 5007,
            'api': 5008
        }
        
        # Initialize components based on mode
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🔧 Master Integration initialized in {mode} mode")
    
    def _initialize_components(self):
        """Initialize components based on integration mode."""
        try:
            if self.mode in [IntegrationMode.FULL, IntegrationMode.BRIDGE, IntegrationMode.ENHANCED]:
                # Initialize bridge
                self.components['bridge'] = KoboldCPPBridge(
                    kobold_port=self.ports['kobold'],
                    bridge_port=self.ports['bridge']
                )
                logger.info("✅ Bridge component initialized")
            
            if self.mode in [IntegrationMode.FULL, IntegrationMode.ENHANCED]:
                # Initialize enhanced interface
                self.components['enhanced'] = KoboldCPPEnhancedInterface(
                    kobold_port=self.ports['kobold'],
                    enhanced_port=self.ports['enhanced']
                )
                logger.info("✅ Enhanced interface component initialized")
            
            if self.mode in [IntegrationMode.FULL, IntegrationMode.VISUAL]:
                # Initialize visual layer
                self.components['visual'] = VisualLayerController()
                logger.info("✅ Visual layer component initialized")
            
            if self.mode in [IntegrationMode.FULL, IntegrationMode.API]:
                # Initialize unified interface
                interface_mode = InterfaceMode.FULL_INTEGRATION if self.mode == IntegrationMode.FULL else InterfaceMode.API_ONLY
                self.components['unified'] = SchwabotUnifiedInterface(interface_mode)
                logger.info("✅ Unified interface component initialized")
            
            if self.mode in [IntegrationMode.FULL, IntegrationMode.CONVERSATION]:
                # Initialize KoboldCPP integration
                self.components['kobold'] = KoboldCPPIntegration(port=self.ports['kobold'])
                logger.info("✅ KoboldCPP integration component initialized")
            
            # Initialize core components for all modes
            self.components['tick_loader'] = TickLoader()
            self.components['signal_cache'] = SignalCache()
            self.components['registry_writer'] = RegistryWriter()
            self.components['json_server'] = JSONServer()
            
            # Initialize memory stack components
            self.components['command_sequencer'] = AICommandSequencer()
            self.components['execution_validator'] = ExecutionValidator()
            self.components['memory_allocator'] = MemoryKeyAllocator()
            
            logger.info("✅ All core components initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def _start_components(self):
        """Start all initialized components."""
        try:
            logger.info("🚀 Starting components...")
            
            # Start bridge if available
            if 'bridge' in self.components:
                logger.info("Starting bridge...")
                # Note: Bridge starts its own Flask server, so we don't call start_bridge() here
                # The bridge will be started when its Flask app runs
            
            # Start enhanced interface if available
            if 'enhanced' in self.components:
                logger.info("Starting enhanced interface...")
                # Enhanced interface starts its own Flask server
            
            # Start visual layer if available
            if 'visual' in self.components:
                logger.info("Starting visual layer...")
                await self.components['visual'].initialize()
            
            # Start unified interface if available
            if 'unified' in self.components:
                logger.info("Starting unified interface...")
                await self.components['unified'].initialize()
            
            # Start JSON server
            logger.info("Starting JSON server...")
            await self.components['json_server'].start()
            
            # Start tick loader
            logger.info("Starting tick loader...")
            await self.components['tick_loader'].start()
            
            # Start signal cache
            logger.info("Starting signal cache...")
            await self.components['signal_cache'].start()
            
            # Start registry writer
            logger.info("Starting registry writer...")
            await self.components['registry_writer'].start()
            
            logger.info("✅ All components started successfully")
            
        except Exception as e:
            logger.error(f"❌ Component startup failed: {e}")
            raise
    
    async def _stop_components(self):
        """Stop all running components."""
        try:
            logger.info("🛑 Stopping components...")
            
            # Stop components in reverse order
            if 'registry_writer' in self.components:
                await self.components['registry_writer'].stop()
            
            if 'signal_cache' in self.components:
                await self.components['signal_cache'].stop()
            
            if 'tick_loader' in self.components:
                await self.components['tick_loader'].stop()
            
            if 'json_server' in self.components:
                await self.components['json_server'].stop()
            
            if 'unified' in self.components:
                await self.components['unified'].shutdown()
            
            if 'visual' in self.components:
                await self.components['visual'].shutdown()
            
            if 'enhanced' in self.components:
                self.components['enhanced'].stop_enhanced_interface()
            
            if 'bridge' in self.components:
                self.components['bridge'].stop_bridge()
            
            logger.info("✅ All components stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Component shutdown failed: {e}")
    
    async def _health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'mode': self.mode,
                'components': {}
            }
            
            # Check each component
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'get_status'):
                        status = await component.get_status()
                    elif hasattr(component, 'status'):
                        status = component.status
                    else:
                        status = 'unknown'
                    
                    health_status['components'][name] = {
                        'status': status,
                        'healthy': status in ['running', 'active', 'connected']
                    }
                    
                except Exception as e:
                    health_status['components'][name] = {
                        'status': 'error',
                        'healthy': False,
                        'error': str(e)
                    }
            
            # Overall health
            healthy_components = sum(1 for c in health_status['components'].values() if c['healthy'])
            total_components = len(health_status['components'])
            
            health_status['overall'] = {
                'healthy': healthy_components == total_components,
                'healthy_components': healthy_components,
                'total_components': total_components,
                'health_percentage': (healthy_components / total_components * 100) if total_components > 0 else 0
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall': {'healthy': False}
            }
    
    async def _monitor_system(self):
        """Monitor system health and performance."""
        try:
            while self.running:
                # Perform health check
                health = await self._health_check()
                
                # Log health status
                if health['overall']['healthy']:
                    logger.info(f"✅ System healthy: {health['overall']['health_percentage']:.1f}% components healthy")
                else:
                    logger.warning(f"⚠️ System issues: {health['overall']['health_percentage']:.1f}% components healthy")
                    
                    # Log unhealthy components
                    for name, status in health['components'].items():
                        if not status['healthy']:
                            logger.warning(f"⚠️ {name}: {status.get('error', 'Unknown error')}")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"❌ System monitoring failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self):
        """Start the master integration system."""
        try:
            logger.info(f"🚀 Starting Master Integration in {self.mode} mode...")
            
            # Start components
            await self._start_components()
            
            # Set running flag
            self.running = True
            
            # Start monitoring
            monitor_task = asyncio.create_task(self._monitor_system())
            
            # Start Flask servers for bridge and enhanced interface
            if 'bridge' in self.components:
                logger.info("Starting bridge Flask server...")
                # Start bridge in a separate thread
                import threading
                bridge_thread = threading.Thread(
                    target=self.components['bridge'].start_bridge,
                    daemon=True
                )
                bridge_thread.start()
            
            if 'enhanced' in self.components:
                logger.info("Starting enhanced interface Flask server...")
                # Start enhanced interface in a separate thread
                import threading
                enhanced_thread = threading.Thread(
                    target=self.components['enhanced'].start_enhanced_interface,
                    daemon=True
                )
                enhanced_thread.start()
            
            logger.info("✅ Master Integration started successfully")
            logger.info(f"📊 System Status:")
            logger.info(f"   - Mode: {self.mode}")
            logger.info(f"   - Bridge Port: {self.ports['bridge']}")
            logger.info(f"   - Enhanced Port: {self.ports['enhanced']}")
            logger.info(f"   - Visual Port: {self.ports['visual']}")
            logger.info(f"   - API Port: {self.ports['api']}")
            logger.info(f"   - KoboldCPP Port: {self.ports['kobold']}")
            
            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)
            
            # Cancel monitoring task
            monitor_task.cancel()
            
        except Exception as e:
            logger.error(f"❌ Master Integration startup failed: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the master integration system."""
        try:
            logger.info("🛑 Stopping Master Integration...")
            
            # Stop components
            await self._stop_components()
            
            # Set running flag
            self.running = False
            
            logger.info("✅ Master Integration stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Master Integration shutdown failed: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            health = await self._health_check()
            
            status = {
                'system': {
                    'mode': self.mode,
                    'running': self.running,
                    'start_time': self.start_time.isoformat(),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                },
                'ports': self.ports,
                'health': health,
                'components': list(self.components.keys())
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Get status failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('master_integration.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Master Integration Script for Schwabot + KoboldCPP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_integration.py                    # Full integration
  python master_integration.py full              # Full integration
  python master_integration.py bridge            # Bridge only
  python master_integration.py enhanced          # Enhanced interface only
  python master_integration.py visual            # Visual layer only
  python master_integration.py conversation      # Conversation mode only
  python master_integration.py api               # API only mode
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        default=IntegrationMode.FULL,
        choices=[
            IntegrationMode.FULL,
            IntegrationMode.BRIDGE,
            IntegrationMode.ENHANCED,
            IntegrationMode.VISUAL,
            IntegrationMode.CONVERSATION,
            IntegrationMode.API
        ],
        help='Integration mode to run'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--kobold-port',
        type=int,
        default=5001,
        help='KoboldCPP port (default: 5001)'
    )
    
    parser.add_argument(
        '--bridge-port',
        type=int,
        default=5005,
        help='Bridge port (default: 5005)'
    )
    
    parser.add_argument(
        '--enhanced-port',
        type=int,
        default=5006,
        help='Enhanced interface port (default: 5006)'
    )
    
    parser.add_argument(
        '--visual-port',
        type=int,
        default=5007,
        help='Visual layer port (default: 5007)'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=5008,
        help='API port (default: 5008)'
    )
    
    return parser.parse_args()

async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        
        logger.info("🔧 Starting Master Integration Script")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Log Level: {args.log_level}")
        
        # Create and start master integration
        integration = MasterIntegration(mode=args.mode)
        
        # Update ports if specified
        integration.ports['kobold'] = args.kobold_port
        integration.ports['bridge'] = args.bridge_port
        integration.ports['enhanced'] = args.enhanced_port
        integration.ports['visual'] = args.visual_port
        integration.ports['api'] = args.api_port
        
        # Start the system
        await integration.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Master Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 