#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Cross-Chain Mode Launcher - Unified Trading System Interface
==============================================================

Unified launcher that integrates:
- Cross-Chain Mode System
- Existing trading modes (Clock Mode, Ferris Ride, etc.)
- Real API pricing and memory storage
- Kraken real-time data integration
- USB memory management
- Shadow mode test suite

This launcher provides a single entry point for all cross-chain operations
and integrates seamlessly with existing Schwabot systems.

⚠️ SAFETY NOTICE: This system is for analysis and timing only.
    Real trading execution requires additional safety layers.
"""

import sys
import os
import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cross_chain_launcher.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import cross-chain system
try:
    from cross_chain_mode_system import (
        CrossChainModeSystem, 
        StrategyType, 
        ChainType, 
        CrossChainExecutionMode,
        CROSS_CHAIN_SAFETY_CONFIG
    )
    CROSS_CHAIN_AVAILABLE = True
except ImportError as e:
    CROSS_CHAIN_AVAILABLE = False
    logger.warning(f"⚠️ Cross-Chain Mode System not available: {e}")

# Import existing systems
try:
    from clock_mode_system import ClockModeSystem, ExecutionMode as ClockExecutionMode
    CLOCK_MODE_AVAILABLE = True
except ImportError:
    CLOCK_MODE_AVAILABLE = False
    logger.warning("⚠️ Clock Mode System not available")

try:
    from AOI_Base_Files_Schwabot.core.ferris_ride_manager import FerrisRideManager
    FERRIS_RIDE_AVAILABLE = True
except ImportError:
    FERRIS_RIDE_AVAILABLE = False
    logger.warning("⚠️ Ferris Ride Manager not available")

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
    logger.warning("⚠️ Real API pricing system not available")

class CrossChainLauncher:
    """Unified launcher for Cross-Chain Mode System."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the cross-chain launcher."""
        self.config_path = config_path or "config/cross_chain_config.yaml"
        self.config = self._load_config()
        
        # Initialize cross-chain system
        if CROSS_CHAIN_AVAILABLE:
            self.cross_chain_system = CrossChainModeSystem()
            logger.info("✅ Cross-Chain Mode System initialized")
        else:
            self.cross_chain_system = None
            logger.error("❌ Cross-Chain Mode System not available")
        
        # Initialize existing systems
        self.clock_system = None
        self.ferris_ride_manager = None
        self.real_api_system = None
        
        self._initialize_existing_systems()
        
        # System state
        self.is_running = False
        self.active_modes: List[str] = []
        self.system_threads: Dict[str, threading.Thread] = {}
        
        logger.info("🔗 Cross-Chain Mode Launcher initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import yaml
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"⚠️ Configuration file not found: {self.config_path}")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "safety": {
                "execution_mode": "shadow",
                "max_position_size": 0.1,
                "max_daily_loss": 0.05
            },
            "cross_chain": {
                "max_chains": 3,
                "chain_sync_interval": 0.1
            },
            "usb_memory": {
                "enabled": True,
                "auto_backup": True
            }
        }
    
    def _initialize_existing_systems(self):
        """Initialize existing trading systems."""
        # Initialize Clock Mode System
        if CLOCK_MODE_AVAILABLE:
            try:
                self.clock_system = ClockModeSystem()
                logger.info("✅ Clock Mode System initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Clock Mode System: {e}")
        
        # Initialize Ferris Ride Manager
        if FERRIS_RIDE_AVAILABLE:
            try:
                self.ferris_ride_manager = FerrisRideManager()
                logger.info("✅ Ferris Ride Manager initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Ferris Ride Manager: {e}")
        
        # Initialize Real API System
        if REAL_API_AVAILABLE:
            try:
                memory_config = MemoryConfig(
                    storage_mode=MemoryStorageMode.AUTO,
                    api_mode=APIMode.REAL_API_ONLY,
                    memory_choice_menu=False,
                    auto_sync=True
                )
                self.real_api_system = initialize_real_api_memory_system(memory_config)
                logger.info("✅ Real API pricing and memory storage system initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Real API System: {e}")
    
    def start_cross_chain_mode(self) -> bool:
        """Start the cross-chain mode system."""
        try:
            if not self.cross_chain_system:
                logger.error("❌ Cross-Chain Mode System not available")
                return False
            
            # Enable shadow mode
            self.cross_chain_system.enable_shadow_mode()
            
            # Initialize default strategies
            self._initialize_default_strategies()
            
            # Create default cross-chains
            self._create_default_chains()
            
            self.is_running = True
            logger.info("🔗 Cross-Chain Mode System started")
            
            # Store startup data in USB memory
            if REAL_API_AVAILABLE:
                store_memory_entry(
                    data_type='cross_chain_startup',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'system_status': 'started',
                        'active_strategies': self.cross_chain_system.active_strategies,
                        'active_chains': len(self.cross_chain_system.active_chains),
                        'shadow_mode_active': self.cross_chain_system.shadow_mode_active,
                        'usb_memory_enabled': self.cross_chain_system.usb_memory_enabled
                    },
                    source='cross_chain_launcher',
                    priority=3,
                    tags=['cross_chain', 'startup', 'system_init']
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting cross-chain mode: {e}")
            return False
    
    def _initialize_default_strategies(self):
        """Initialize default strategies."""
        try:
            if not self.cross_chain_system:
                return
            
            # Enable default strategies
            default_strategies = ["clock_mode_001", "ferris_ride_001"]
            
            for strategy_id in default_strategies:
                if strategy_id in self.cross_chain_system.strategies:
                    success = self.cross_chain_system.toggle_strategy(strategy_id, True)
                    if success:
                        logger.info(f"✅ Enabled default strategy: {strategy_id}")
                    else:
                        logger.warning(f"⚠️ Failed to enable strategy: {strategy_id}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing default strategies: {e}")
    
    def _create_default_chains(self):
        """Create default cross-chains."""
        try:
            if not self.cross_chain_system:
                return
            
            # Create dual chain (Clock + Ferris)
            success = self.cross_chain_system.create_cross_chain(
                "dual_clock_ferris",
                ChainType.DUAL,
                ["clock_mode_001", "ferris_ride_001"],
                {"clock_mode_001": 0.6, "ferris_ride_001": 0.4}
            )
            
            if success:
                # Activate the chain
                activate_success = self.cross_chain_system.activate_cross_chain("dual_clock_ferris")
                if activate_success:
                    logger.info("✅ Created and activated default dual chain: dual_clock_ferris")
                else:
                    logger.warning("⚠️ Created dual chain but failed to activate")
            else:
                logger.warning("⚠️ Failed to create default dual chain")
            
        except Exception as e:
            logger.error(f"❌ Error creating default chains: {e}")
    
    def toggle_strategy(self, strategy_id: str, enable: bool) -> bool:
        """Toggle a strategy on/off."""
        try:
            if not self.cross_chain_system:
                return False
            
            success = self.cross_chain_system.toggle_strategy(strategy_id, enable)
            
            if success:
                action = "enabled" if enable else "disabled"
                logger.info(f"✅ Strategy {strategy_id} {action}")
                
                # Store in USB memory
                if REAL_API_AVAILABLE:
                    store_memory_entry(
                        data_type='strategy_toggle',
                        data={
                            'strategy_id': strategy_id,
                            'action': action,
                            'timestamp': datetime.now().isoformat(),
                            'active_strategies': self.cross_chain_system.active_strategies
                        },
                        source='cross_chain_launcher',
                        priority=2,
                        tags=['strategy', 'toggle', action]
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error toggling strategy {strategy_id}: {e}")
            return False
    
    def create_cross_chain(self, chain_id: str, chain_type: ChainType, 
                          strategies: List[str], weights: Optional[Dict[str, float]] = None) -> bool:
        """Create a new cross-chain."""
        try:
            if not self.cross_chain_system:
                return False
            
            success = self.cross_chain_system.create_cross_chain(chain_id, chain_type, strategies, weights)
            
            if success:
                # Activate the chain
                activate_success = self.cross_chain_system.activate_cross_chain(chain_id)
                
                if activate_success:
                    logger.info(f"✅ Created and activated cross-chain: {chain_id}")
                    
                    # Store in USB memory
                    if REAL_API_AVAILABLE:
                        store_memory_entry(
                            data_type='cross_chain_created',
                            data={
                                'chain_id': chain_id,
                                'chain_type': chain_type.value,
                                'strategies': strategies,
                                'weights': weights,
                                'timestamp': datetime.now().isoformat(),
                                'active_chains': len(self.cross_chain_system.active_chains)
                            },
                            source='cross_chain_launcher',
                            priority=3,
                            tags=['cross_chain', 'created', 'activated']
                        )
                else:
                    logger.warning(f"⚠️ Created cross-chain {chain_id} but failed to activate")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error creating cross-chain {chain_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "launcher_status": "running" if self.is_running else "stopped",
                "cross_chain_available": CROSS_CHAIN_AVAILABLE,
                "clock_mode_available": CLOCK_MODE_AVAILABLE,
                "ferris_ride_available": FERRIS_RIDE_AVAILABLE,
                "real_api_available": REAL_API_AVAILABLE,
                "active_modes": self.active_modes,
                "config_loaded": bool(self.config)
            }
            
            # Add cross-chain system status
            if self.cross_chain_system:
                cross_chain_status = self.cross_chain_system.get_system_status()
                status["cross_chain_system"] = cross_chain_status
            
            # Add existing systems status
            if self.clock_system:
                status["clock_system"] = {
                    "available": True,
                    "status": "initialized"
                }
            
            if self.ferris_ride_manager:
                status["ferris_ride_manager"] = {
                    "available": True,
                    "status": "initialized"
                }
            
            if self.real_api_system:
                status["real_api_system"] = {
                    "available": True,
                    "status": "initialized"
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {"error": str(e)}
    
    def run_interactive_mode(self):
        """Run interactive command-line mode."""
        try:
            print("🔗 Cross-Chain Mode Launcher - Interactive Mode")
            print("=" * 50)
            
            # Start cross-chain system
            if self.start_cross_chain_mode():
                print("✅ Cross-Chain Mode System started")
            else:
                print("❌ Failed to start Cross-Chain Mode System")
                return
            
            print("\nAvailable commands:")
            print("  status          - Show system status")
            print("  toggle <id>     - Toggle strategy on/off")
            print("  create <id>     - Create cross-chain")
            print("  chains          - Show active chains")
            print("  quit            - Exit")
            print()
            
            while True:
                try:
                    command = input("🔗 > ").strip().lower()
                    
                    if command == "quit":
                        break
                    elif command == "status":
                        self._show_status()
                    elif command.startswith("toggle "):
                        strategy_id = command.split(" ", 1)[1]
                        self._interactive_toggle_strategy(strategy_id)
                    elif command.startswith("create "):
                        chain_id = command.split(" ", 1)[1]
                        self._interactive_create_chain(chain_id)
                    elif command == "chains":
                        self._show_active_chains()
                    else:
                        print("❌ Unknown command. Type 'quit' to exit.")
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error in interactive mode: {e}")
    
    def _show_status(self):
        """Show system status."""
        try:
            status = self.get_system_status()
            print("\n📊 System Status:")
            print(f"  Launcher: {status['launcher_status']}")
            print(f"  Cross-Chain: {'✅' if status['cross_chain_available'] else '❌'}")
            print(f"  Clock Mode: {'✅' if status['clock_mode_available'] else '❌'}")
            print(f"  Ferris Ride: {'✅' if status['ferris_ride_available'] else '❌'}")
            print(f"  Real API: {'✅' if status['real_api_available'] else '❌'}")
            
            if "cross_chain_system" in status:
                cs = status["cross_chain_system"]
                print(f"  Active Strategies: {cs['active_strategies']}")
                print(f"  Active Chains: {cs['active_chains']}")
                print(f"  Shadow Mode: {'✅' if cs['shadow_mode_active'] else '❌'}")
                print(f"  USB Memory: {'✅' if cs['usb_memory_enabled'] else '❌'}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error showing status: {e}")
    
    def _interactive_toggle_strategy(self, strategy_id: str):
        """Interactive strategy toggle."""
        try:
            if not self.cross_chain_system:
                print("❌ Cross-Chain System not available")
                return
            
            # Get current state
            strategy = self.cross_chain_system.strategies.get(strategy_id)
            if not strategy:
                print(f"❌ Strategy {strategy_id} not found")
                return
            
            # Toggle state
            new_state = not strategy.is_active
            success = self.toggle_strategy(strategy_id, new_state)
            
            if success:
                action = "enabled" if new_state else "disabled"
                print(f"✅ Strategy {strategy_id} {action}")
            else:
                print(f"❌ Failed to toggle strategy {strategy_id}")
                
        except Exception as e:
            print(f"❌ Error toggling strategy: {e}")
    
    def _interactive_create_chain(self, chain_id: str):
        """Interactive chain creation."""
        try:
            print(f"Creating cross-chain: {chain_id}")
            
            # Get available strategies
            if not self.cross_chain_system:
                print("❌ Cross-Chain System not available")
                return
            
            available_strategies = list(self.cross_chain_system.strategies.keys())
            print(f"Available strategies: {', '.join(available_strategies)}")
            
            # Get strategy selection
            strategy_input = input("Enter strategies (comma-separated): ").strip()
            selected_strategies = [s.strip() for s in strategy_input.split(",")]
            
            # Validate strategies
            for strategy in selected_strategies:
                if strategy not in available_strategies:
                    print(f"❌ Strategy {strategy} not found")
                    return
            
            # Create chain
            success = self.create_cross_chain(chain_id, ChainType.DUAL, selected_strategies)
            
            if success:
                print(f"✅ Cross-chain {chain_id} created and activated")
            else:
                print(f"❌ Failed to create cross-chain {chain_id}")
                
        except Exception as e:
            print(f"❌ Error creating chain: {e}")
    
    def _show_active_chains(self):
        """Show active chains."""
        try:
            if not self.cross_chain_system:
                print("❌ Cross-Chain System not available")
                return
            
            active_chains = [
                chain_id for chain_id in self.cross_chain_system.chains
                if self.cross_chain_system.chains[chain_id].is_active
            ]
            
            if not active_chains:
                print("No active cross-chains")
                return
            
            print("\n🔗 Active Cross-Chains:")
            for chain_id in active_chains:
                chain = self.cross_chain_system.chains[chain_id]
                print(f"  {chain_id} ({chain.chain_type.value})")
                print(f"    Strategies: {', '.join(chain.strategies)}")
                print(f"    Sync Count: {chain.sync_count}")
                print()
                
        except Exception as e:
            print(f"❌ Error showing chains: {e}")
    
    def stop_system(self):
        """Stop the cross-chain system."""
        try:
            self.is_running = False
            
            # Stop cross-chain system
            if self.cross_chain_system:
                # Stop background threads
                for thread in self.system_threads.values():
                    if thread.is_alive():
                        thread.join(timeout=5.0)
                
                logger.info("🔗 Cross-Chain Mode System stopped")
            
            # Store shutdown data
            if REAL_API_AVAILABLE:
                store_memory_entry(
                    data_type='cross_chain_shutdown',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'system_status': 'stopped',
                        'uptime': time.time() - getattr(self, '_start_time', time.time())
                    },
                    source='cross_chain_launcher',
                    priority=2,
                    tags=['cross_chain', 'shutdown']
                )
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-Chain Mode Launcher")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch GUI")
    parser.add_argument("--status", "-s", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    try:
        # Create launcher
        launcher = CrossChainLauncher(args.config)
        
        if args.status:
            # Show status and exit
            status = launcher.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.gui:
            # Launch GUI
            try:
                from cross_chain_gui import CrossChainGUI
                gui = CrossChainGUI()
                gui.run()
            except ImportError:
                print("❌ GUI not available")
                
        elif args.interactive:
            # Run interactive mode
            launcher.run_interactive_mode()
            
        else:
            # Default: start system and show status
            if launcher.start_cross_chain_mode():
                print("✅ Cross-Chain Mode System started")
                status = launcher.get_system_status()
                print(json.dumps(status, indent=2))
            else:
                print("❌ Failed to start Cross-Chain Mode System")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 