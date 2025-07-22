#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified System CLI - Schwabot Complete Control Interface
========================================================

Provides complete command-line control over the entire Schwabot system including:
- Distributed node management
- AI integration and decision making
- High-volume trading system
- Real-time context ingestion
- System monitoring and health
- Hardware optimization

Features:
- Hardware-agnostic optimization
- Complete system control
- Real-time monitoring
- AI decision integration
- Distributed system management
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distributed_system.distributed_node_manager import get_distributed_manager, start_distributed_system
from core.distributed_system.real_time_context_ingestion import get_context_ingestion, start_context_ingestion
from core.distributed_system.ai_integration_bridge import get_ai_bridge, start_ai_integration
from core.high_volume_trading_manager import HighVolumeTradingManager
from AOI_Base_Files_Schwabot.cli.hardware_optimization_cli import HardwareOptimizationCLI

logger = logging.getLogger(__name__)

class UnifiedSystemCLI:
    """Unified CLI for complete Schwabot system control."""
    
    def __init__(self):
        self.distributed_manager = None
        self.context_ingestion = None
        self.ai_bridge = None
        self.trading_manager = None
        self.hardware_cli = None
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Initialized UnifiedSystemCLI")
    
    async def start_system(self):
        """Start the complete Schwabot system."""
        logger.info("Starting complete Schwabot system...")
        
        try:
            # Start distributed system
            self.distributed_manager = await start_distributed_system()
            logger.info("✓ Distributed system started")
            
            # Start context ingestion
            self.context_ingestion = await start_context_ingestion()
            logger.info("✓ Context ingestion started")
            
            # Start AI integration
            self.ai_bridge = await start_ai_integration()
            logger.info("✓ AI integration started")
            
            # Initialize hardware optimization
            self.hardware_cli = HardwareOptimizationCLI()
            await self.hardware_cli.initialize()
            logger.info("✓ Hardware optimization initialized")
            
            # Initialize trading manager
            self.trading_manager = HighVolumeTradingManager()
            await self.trading_manager.initialize()
            logger.info("✓ Trading manager initialized")
            
            self.is_running = True
            logger.info("✓ Complete Schwabot system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """Stop the complete Schwabot system."""
        logger.info("Stopping complete Schwabot system...")
        
        self.is_running = False
        
        # Stop components in reverse order
        if self.trading_manager:
            await self.trading_manager.stop()
            logger.info("✓ Trading manager stopped")
        
        if self.hardware_cli:
            await self.hardware_cli.cleanup()
            logger.info("✓ Hardware optimization stopped")
        
        if self.ai_bridge:
            await self.ai_bridge.stop()
            logger.info("✓ AI integration stopped")
        
        if self.context_ingestion:
            await self.context_ingestion.stop()
            logger.info("✓ Context ingestion stopped")
        
        if self.distributed_manager:
            await self.distributed_manager.stop()
            logger.info("✓ Distributed system stopped")
        
        logger.info("✓ Complete Schwabot system stopped")
    
    async def show_system_status(self):
        """Show comprehensive system status."""
        if not self.is_running:
            print("❌ System is not running")
            return
        
        print("\n" + "="*60)
        print("SCHWABOT SYSTEM STATUS")
        print("="*60)
        
        # Distributed system status
        if self.distributed_manager:
            node_status = self.distributed_manager.get_node_status()
            print(f"\n🌐 DISTRIBUTED SYSTEM:")
            print(f"   Flask Node: {node_status['current_flask_node']}")
            print(f"   Is Flask Node: {node_status['is_flask_node']}")
            print(f"   Total Nodes: {node_status['total_nodes']}")
            print(f"   Active Nodes: {node_status['active_nodes']}")
        
        # Context ingestion status
        if self.context_ingestion:
            context_summary = self.context_ingestion.get_context_summary()
            print(f"\n📊 CONTEXT INGESTION:")
            print(f"   Total Context Items: {context_summary['total_context_items']}")
            print(f"   Tensor Cache Size: {context_summary['tensor_cache_size']}")
            print(f"   Hash Context Size: {context_summary['hash_context_size']}")
            print(f"   Recent Types: {context_summary['recent_context_types']}")
        
        # AI integration status
        if self.ai_bridge:
            ai_status = self.ai_bridge.get_ai_status()
            print(f"\n🤖 AI INTEGRATION:")
            print(f"   Total Models: {ai_status['total_models']}")
            print(f"   Active Models: {ai_status['active_models']}")
            print(f"   Recent Decisions: {ai_status['recent_decisions']}")
            print(f"   Consensus Decisions: {ai_status['consensus_decisions']}")
        
        # Trading system status
        if self.trading_manager:
            trading_status = self.trading_manager.get_status()
            print(f"\n💰 TRADING SYSTEM:")
            print(f"   Status: {trading_status.get('status', 'unknown')}")
            print(f"   Active Exchanges: {len(trading_status.get('exchanges', {}))}")
            print(f"   Total P&L: {trading_status.get('total_pnl', 0):.2f}")
            print(f"   Win Rate: {trading_status.get('win_rate', 0):.1f}%")
        
        # Hardware optimization status
        if self.hardware_cli:
            hw_status = await self.hardware_cli.get_system_status()
            print(f"\n⚡ HARDWARE OPTIMIZATION:")
            print(f"   CPU Usage: {hw_status.get('cpu_usage', 0):.1f}%")
            print(f"   Memory Usage: {hw_status.get('memory_usage', 0):.1f}%")
            print(f"   GPU Available: {hw_status.get('gpu_available', False)}")
            print(f"   Optimization Level: {hw_status.get('optimization_level', 'unknown')}")
        
        print("\n" + "="*60)
    
    async def request_ai_decision(self, symbols: List[str] = None):
        """Request an AI decision for trading."""
        if not self.is_running or not self.ai_bridge:
            print("❌ AI system is not available")
            return
        
        print(f"\n🤖 Requesting AI decision for symbols: {symbols or 'all'}")
        
        # Get current context
        context_data = {
            "timestamp": time.time(),
            "request_type": "ai_decision",
            "symbols": symbols
        }
        
        try:
            decision = await self.ai_bridge.request_decision(context_data, symbols)
            
            print(f"\n📋 AI DECISION:")
            print(f"   Decision: {decision.final_decision.value.upper()}")
            print(f"   Confidence: {decision.confidence:.2f}")
            print(f"   Models Participated: {len(decision.model_votes)}")
            print(f"   Reasoning: {decision.consensus_reasoning}")
            
            # Show individual model decisions
            if decision.model_votes:
                print(f"\n   Individual Model Decisions:")
                for model_type, model_decision in decision.model_votes.items():
                    print(f"     {model_type.value}: {model_decision.decision_type.value} "
                          f"(confidence: {model_decision.confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error requesting AI decision: {e}")
    
    async def start_trading(self, mode: str = "high_volume"):
        """Start the trading system."""
        if not self.is_running or not self.trading_manager:
            print("❌ Trading system is not available")
            return
        
        print(f"\n💰 Starting trading system in {mode} mode...")
        
        try:
            if mode == "high_volume":
                await self.trading_manager.activate_high_volume_trading()
            else:
                await self.trading_manager.activate_trading()
            
            print("✓ Trading system activated successfully")
            
        except Exception as e:
            print(f"❌ Error starting trading: {e}")
    
    async def stop_trading(self):
        """Stop the trading system."""
        if not self.is_running or not self.trading_manager:
            print("❌ Trading system is not available")
            return
        
        print("\n💰 Stopping trading system...")
        
        try:
            await self.trading_manager.deactivate_trading()
            print("✓ Trading system stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping trading: {e}")
    
    async def emergency_stop(self):
        """Emergency stop all trading."""
        if not self.is_running:
            print("❌ System is not running")
            return
        
        print("\n🚨 EMERGENCY STOP - Stopping all trading activities...")
        
        try:
            # Stop trading
            if self.trading_manager:
                await self.trading_manager.emergency_stop()
            
            # Stop AI decisions
            if self.ai_bridge:
                await self.ai_bridge.stop()
            
            print("✓ Emergency stop completed")
            
        except Exception as e:
            print(f"❌ Error during emergency stop: {e}")
    
    async def optimize_hardware(self, target: str = "auto"):
        """Optimize hardware for current workload."""
        if not self.is_running or not self.hardware_cli:
            print("❌ Hardware optimization is not available")
            return
        
        print(f"\n⚡ Optimizing hardware for target: {target}")
        
        try:
            if target == "auto":
                await self.hardware_cli.auto_optimize()
            elif target == "trading":
                await self.hardware_cli.optimize_for_trading()
            elif target == "ai":
                await self.hardware_cli.optimize_for_ai()
            else:
                await self.hardware_cli.optimize_for_workload(target)
            
            print("✓ Hardware optimization completed")
            
        except Exception as e:
            print(f"❌ Error optimizing hardware: {e}")
    
    async def ingest_test_data(self):
        """Ingest test data into the context system."""
        if not self.is_running or not self.context_ingestion:
            print("❌ Context ingestion is not available")
            return
        
        print("\n📊 Ingesting test data...")
        
        try:
            # Test trading data
            await self.context_ingestion.ingest_trading_data({
                "symbol": "BTC/USD",
                "price": 50000.0,
                "volume": 1000.0,
                "timestamp": time.time()
            })
            
            # Test tensor math
            from core.distributed_system.real_time_context_ingestion import TensorMathResult
            await self.context_ingestion.ingest_tensor_math(TensorMathResult(
                calculation_id="test_calc_001",
                input_data={"price": 50000.0, "volume": 1000.0},
                result={"prediction": "buy", "confidence": 0.85},
                hash_value="abc123",
                context_meaning="Strong buy signal based on volume analysis",
                confidence=0.85,
                timestamp=time.time()
            ))
            
            # Test system health
            await self.context_ingestion.ingest_system_health({
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "network_io": {"bytes_sent": 1024, "bytes_recv": 2048}
            })
            
            print("✓ Test data ingested successfully")
            
        except Exception as e:
            print(f"❌ Error ingesting test data: {e}")
    
    async def show_ai_history(self, limit: int = 10):
        """Show recent AI decision history."""
        if not self.is_running or not self.ai_bridge:
            print("❌ AI system is not available")
            return
        
        print(f"\n📋 Recent AI Decision History (last {limit}):")
        
        try:
            history = self.ai_bridge.get_decision_history(limit)
            
            if not history:
                print("   No recent decisions")
                return
            
            for i, decision in enumerate(reversed(history), 1):
                timestamp = datetime.fromtimestamp(decision.timestamp).strftime("%H:%M:%S")
                print(f"   {i}. [{timestamp}] {decision.final_decision.value.upper()} "
                      f"(confidence: {decision.confidence:.2f})")
                print(f"      Reasoning: {decision.consensus_reasoning[:100]}...")
                print()
            
        except Exception as e:
            print(f"❌ Error showing AI history: {e}")
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🎮 Starting interactive mode...")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while self.is_running:
            try:
                command = input("\nschwabot> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "help":
                    self._show_help()
                elif command == "status":
                    await self.show_system_status()
                elif command == "ai_decision":
                    symbols = input("Enter symbols (comma-separated, or press Enter for all): ").strip()
                    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
                    await self.request_ai_decision(symbol_list)
                elif command == "start_trading":
                    mode = input("Enter trading mode (high_volume/standard): ").strip() or "high_volume"
                    await self.start_trading(mode)
                elif command == "stop_trading":
                    await self.stop_trading()
                elif command == "emergency_stop":
                    confirm = input("Are you sure? Type 'YES' to confirm: ")
                    if confirm == "YES":
                        await self.emergency_stop()
                    else:
                        print("Emergency stop cancelled")
                elif command == "optimize":
                    target = input("Enter optimization target (auto/trading/ai): ").strip() or "auto"
                    await self.optimize_hardware(target)
                elif command == "test_data":
                    await self.ingest_test_data()
                elif command == "ai_history":
                    limit = input("Enter number of decisions to show (default 10): ").strip()
                    limit = int(limit) if limit.isdigit() else 10
                    await self.show_ai_history(limit)
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show available commands."""
        print("\n📖 Available Commands:")
        print("  status          - Show system status")
        print("  ai_decision     - Request AI decision")
        print("  start_trading   - Start trading system")
        print("  stop_trading    - Stop trading system")
        print("  emergency_stop  - Emergency stop all trading")
        print("  optimize        - Optimize hardware")
        print("  test_data       - Ingest test data")
        print("  ai_history      - Show AI decision history")
        print("  help            - Show this help")
        print("  quit/exit       - Exit interactive mode")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Schwabot System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/unified_system_cli.py --start                    # Start system and enter interactive mode
  python cli/unified_system_cli.py --status                   # Show system status
  python cli/unified_system_cli.py --ai-decision BTC/USD     # Request AI decision
  python cli/unified_system_cli.py --start-trading            # Start trading
  python cli/unified_system_cli.py --emergency-stop           # Emergency stop
        """
    )
    
    parser.add_argument("--start", action="store_true", 
                       help="Start the complete system")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status")
    parser.add_argument("--ai-decision", nargs="*", metavar="SYMBOL",
                       help="Request AI decision for symbols")
    parser.add_argument("--start-trading", choices=["high_volume", "standard"],
                       default="high_volume", help="Start trading system")
    parser.add_argument("--stop-trading", action="store_true",
                       help="Stop trading system")
    parser.add_argument("--emergency-stop", action="store_true",
                       help="Emergency stop all trading")
    parser.add_argument("--optimize", choices=["auto", "trading", "ai"],
                       default="auto", help="Optimize hardware")
    parser.add_argument("--test-data", action="store_true",
                       help="Ingest test data")
    parser.add_argument("--ai-history", type=int, metavar="LIMIT",
                       help="Show AI decision history")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = UnifiedSystemCLI()
    
    async def run_cli():
        try:
            # Start system if requested
            if args.start or args.interactive:
                await cli.start_system()
            
            # Execute commands
            if args.status:
                await cli.show_system_status()
            
            if args.ai_decision is not None:
                symbols = args.ai_decision if args.ai_decision else None
                await cli.request_ai_decision(symbols)
            
            if args.start_trading:
                await cli.start_trading(args.start_trading)
            
            if args.stop_trading:
                await cli.stop_trading()
            
            if args.emergency_stop:
                await cli.emergency_stop()
            
            if args.optimize:
                await cli.optimize_hardware(args.optimize)
            
            if args.test_data:
                await cli.ingest_test_data()
            
            if args.ai_history is not None:
                await cli.show_ai_history(args.ai_history)
            
            # Run interactive mode if requested
            if args.interactive:
                await cli.interactive_mode()
            
            # Stop system if we started it
            if args.start or args.interactive:
                await cli.stop_system()
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            await cli.stop_system()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await cli.stop_system()
            sys.exit(1)
    
    # Run the CLI
    asyncio.run(run_cli())

if __name__ == "__main__":
    main() 