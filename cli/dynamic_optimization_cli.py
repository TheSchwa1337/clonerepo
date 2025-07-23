#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Worker Optimization CLI Integration
==========================================

Integrates dynamic worker optimization commands into the existing Schwabot CLI system.
Provides commands for managing the sophisticated downtime optimization system.

Usage:
    python cli/dynamic_optimization_cli.py --start                    # Start optimization
    python cli/dynamic_optimization_cli.py --status                   # Show status
    python cli/dynamic_optimization_cli.py --workers                  # Show workers
    python cli/dynamic_optimization_cli.py --lattice                  # Show profit lattice
    python cli/dynamic_optimization_cli.py --add-worker ID HOST IP    # Add worker
    python cli/dynamic_optimization_cli.py --demo                     # Run demo
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.dynamic_worker_optimization import DynamicWorkerOptimization
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dynamic Worker Optimization not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicOptimizationCLI:
    """CLI interface for Dynamic Worker Optimization system."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.optimization = None
        self.running = False
    
    def show_banner(self):
        """Show CLI banner."""
        print("🚀 SCHWABOT DYNAMIC WORKER OPTIMIZATION CLI")
        print("=" * 50)
        print("Advanced downtime optimization with profit lattice mapping")
        print("Flask AI agent communication and memory registry updates")
        print("=" * 50)
    
    def show_help(self):
        """Show help information."""
        print("\n📋 AVAILABLE COMMANDS:")
        print("  --start                    Start optimization system")
        print("  --stop                     Stop optimization system")
        print("  --status                   Show optimization status")
        print("  --workers                  Show detailed worker status")
        print("  --lattice                  Show profit lattice summary")
        print("  --add-worker ID HOST IP    Add new worker node")
        print("  --remove-worker ID         Remove worker node")
        print("  --reset-metrics            Reset performance metrics")
        print("  --export-config FILE       Export configuration")
        print("  --import-config FILE       Import configuration")
        print("  --demo                     Run demonstration mode")
        print("  --interactive              Run interactive mode")
        print("  --help                     Show this help")
        
        print("\n🎯 FEATURES:")
        print("  • Downtime optimization (1 AM - 4 AM)")
        print("  • Profit lattice mapping")
        print("  • Flask AI agent communication")
        print("  • Memory registry updates")
        print("  • Dynamic ticker assignment")
        print("  • Orbital performance tracking")
        print("  • Swing timing optimization")
        
        print("\n💡 EXAMPLES:")
        print("  python cli/dynamic_optimization_cli.py --start")
        print("  python cli/dynamic_optimization_cli.py --status")
        print("  python cli/dynamic_optimization_cli.py --add-worker worker1 localhost 192.168.1.100")
        print("  python cli/dynamic_optimization_cli.py --demo")
    
    async def initialize_optimization(self):
        """Initialize the optimization system."""
        if not OPTIMIZATION_AVAILABLE:
            print("❌ Dynamic Worker Optimization not available")
            return False
        
        try:
            self.optimization = DynamicWorkerOptimization()
            
            # Add some demo workers
            self.optimization.add_worker(
                "worker_01", "schwabot-node-1", "192.168.1.101",
                {"cpu_cores": 8, "gpu": True, "memory_gb": 16}
            )
            self.optimization.add_worker(
                "worker_02", "schwabot-node-2", "192.168.1.102",
                {"cpu_cores": 4, "gpu": False, "memory_gb": 8}
            )
            self.optimization.add_worker(
                "worker_03", "schwabot-node-3", "192.168.1.103",
                {"cpu_cores": 6, "gpu": True, "memory_gb": 12}
            )
            
            print("✅ Dynamic Worker Optimization initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize optimization: {e}")
            return False
    
    async def start_optimization(self):
        """Start the optimization system."""
        if not await self.initialize_optimization():
            return
        
        try:
            await self.optimization.start_optimization()
            self.running = True
            print("🚀 Dynamic Worker Optimization started")
            print("   • Downtime optimization: 1 AM - 4 AM")
            print("   • Active optimization: During trading hours")
            print("   • Profit lattice mapping: Continuous")
            print("   • AI agent communication: Enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to start optimization: {e}")
    
    def stop_optimization(self):
        """Stop the optimization system."""
        if self.optimization:
            self.optimization.stop_optimization()
            self.running = False
            print("🛑 Dynamic Worker Optimization stopped")
    
    def show_status(self):
        """Show optimization status."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        status = self.optimization.get_optimization_status()
        
        print("\n📊 OPTIMIZATION STATUS:")
        print(f"  Running: {'✅ Yes' if status['running'] else '❌ No'}")
        print(f"  Downtime Mode: {'🌙 Yes' if status['downtime_mode'] else '☀️ No'}")
        print(f"  Workers: {status['workers_count']}")
        print(f"  Assets: {status['assets_count']}")
        print(f"  Profit Lattice Size: {status['profit_lattice_size']}")
        print(f"  Memory Keys: {status['memory_keys_count']}")
        print(f"  Registry Updates: {status['registry_updates_count']}")
        
        print("\n📈 PERFORMANCE METRICS:")
        metrics = status['performance_metrics']
        print(f"  Optimization Cycles: {metrics['optimization_cycles']}")
        print(f"  Assets Mapped: {metrics['assets_mapped']}")
        print(f"  AI Analyses: {metrics['ai_analyses']}")
        print(f"  Memory Updates: {metrics['memory_updates']}")
        print(f"  Worker Reassignments: {metrics['worker_reassignments']}")
        print(f"  Profit Improvements: {metrics['profit_improvements']:.2f}%")
    
    def show_workers(self):
        """Show detailed worker status."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        worker_status = self.optimization.get_worker_status()
        
        print("\n👥 WORKER STATUS:")
        for worker_id, worker in worker_status.items():
            print(f"\n  🔧 {worker_id} ({worker['hostname']})")
            print(f"    IP: {worker['ip_address']}")
            print(f"    Usage: {worker['current_usage']:.1%}")
            print(f"    Mode: {worker['optimization_mode']}")
            print(f"    Assigned Assets: {', '.join(worker['assigned_assets']) if worker['assigned_assets'] else 'None'}")
            print(f"    Last Activity: {time.strftime('%H:%M:%S', time.localtime(worker['last_activity']))}")
    
    def show_lattice(self):
        """Show profit lattice summary."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        lattice_summary = self.optimization.get_profit_lattice_summary()
        
        if 'message' in lattice_summary:
            print(f"ℹ️  {lattice_summary['message']}")
            return
        
        print("\n📊 PROFIT LATTICE SUMMARY:")
        print(f"  Total Assets: {lattice_summary['total_assets']}")
        print(f"  Avg Profit Potential: {lattice_summary['avg_profit_potential']:.2%}")
        print(f"  Max Profit Potential: {lattice_summary['max_profit_potential']:.2%}")
        print(f"  Avg Confidence Score: {lattice_summary['avg_confidence_score']:.2f}")
        print(f"  Swing Opportunities: {lattice_summary['swing_opportunities']}")
        
        print("\n🌌 ORBITAL DISTRIBUTION:")
        for orbital, count in lattice_summary['orbital_distribution'].items():
            print(f"  {orbital.replace('_', ' ').title()}: {count}")
    
    def add_worker(self, worker_id: str, hostname: str, ip_address: str):
        """Add a new worker."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        try:
            capabilities = {
                "cpu_cores": 4,
                "gpu": False,
                "memory_gb": 8
            }
            
            self.optimization.add_worker(worker_id, hostname, ip_address, capabilities)
            print(f"✅ Added worker {worker_id} ({hostname})")
            
        except Exception as e:
            logger.error(f"❌ Failed to add worker: {e}")
    
    def remove_worker(self, worker_id: str):
        """Remove a worker."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        try:
            self.optimization.remove_worker(worker_id)
            print(f"✅ Removed worker {worker_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to remove worker: {e}")
    
    def reset_metrics(self):
        """Reset performance metrics."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        self.optimization.reset_performance_metrics()
        print("🔄 Performance metrics reset")
    
    def export_config(self, filename: str):
        """Export configuration."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        try:
            config_data = self.optimization.export_configuration()
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration exported to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export configuration: {e}")
    
    def import_config(self, filename: str):
        """Import configuration."""
        if not self.optimization:
            print("❌ Optimization system not initialized")
            return
        
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            self.optimization.import_configuration(config_data)
            print(f"✅ Configuration imported from {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to import configuration: {e}")
    
    async def run_demo(self):
        """Run demonstration mode."""
        print("🎮 DYNAMIC WORKER OPTIMIZATION DEMO")
        print("=" * 40)
        
        if not await self.initialize_optimization():
            return
        
        print("🚀 Starting optimization system...")
        await self.optimization.start_optimization()
        
        print("⏳ Running demo for 30 seconds...")
        print("   • Simulating downtime optimization")
        print("   • Mapping assets to profit lattice")
        print("   • Communicating with Flask AI agents")
        print("   • Updating memory registry")
        print("   • Applying dynamic ticker assignment")
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        print("🛑 Stopping demo...")
        self.optimization.stop_optimization()
        
        # Show results
        print("\n📊 DEMO RESULTS:")
        self.show_status()
        print()
        self.show_lattice()
        
        print("\n✅ Demo completed!")
    
    async def run_interactive_mode(self):
        """Run interactive mode."""
        if not await self.initialize_optimization():
            return
        
        self.show_banner()
        print("🎮 INTERACTIVE MODE")
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n🚀 optimization> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'start':
                    await self.start_optimization()
                elif command == 'stop':
                    self.stop_optimization()
                elif command == 'status':
                    self.show_status()
                elif command == 'workers':
                    self.show_workers()
                elif command == 'lattice':
                    self.show_lattice()
                elif command == 'reset':
                    self.reset_metrics()
                elif command.startswith('add '):
                    parts = command.split()
                    if len(parts) >= 4:
                        self.add_worker(parts[1], parts[2], parts[3])
                    else:
                        print("Usage: add <id> <hostname> <ip>")
                elif command.startswith('remove '):
                    parts = command.split()
                    if len(parts) >= 2:
                        self.remove_worker(parts[1])
                    else:
                        print("Usage: remove <id>")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        if self.running:
            self.stop_optimization()
        print("👋 Goodbye!")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Schwabot Dynamic Worker Optimization CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/dynamic_optimization_cli.py --start                    # Start optimization
  python cli/dynamic_optimization_cli.py --status                   # Show status
  python cli/dynamic_optimization_cli.py --workers                  # Show workers
  python cli/dynamic_optimization_cli.py --lattice                  # Show profit lattice
  python cli/dynamic_optimization_cli.py --add-worker worker1 localhost 192.168.1.100
  python cli/dynamic_optimization_cli.py --demo                     # Run demo
  python cli/dynamic_optimization_cli.py --interactive              # Interactive mode
        """
    )
    
    parser.add_argument('--start', action='store_true', help='Start optimization system')
    parser.add_argument('--stop', action='store_true', help='Stop optimization system')
    parser.add_argument('--status', action='store_true', help='Show optimization status')
    parser.add_argument('--workers', action='store_true', help='Show worker status')
    parser.add_argument('--lattice', action='store_true', help='Show profit lattice summary')
    parser.add_argument('--add-worker', nargs=3, metavar=('ID', 'HOSTNAME', 'IP'), help='Add worker')
    parser.add_argument('--remove-worker', metavar='ID', help='Remove worker')
    parser.add_argument('--reset-metrics', action='store_true', help='Reset performance metrics')
    parser.add_argument('--export-config', metavar='FILE', help='Export configuration')
    parser.add_argument('--import-config', metavar='FILE', help='Import configuration')
    parser.add_argument('--demo', action='store_true', help='Run demonstration mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    
    args = parser.parse_args()
    
    cli = DynamicOptimizationCLI()
    
    try:
        if args.start:
            await cli.start_optimization()
        elif args.stop:
            cli.stop_optimization()
        elif args.status:
            cli.show_status()
        elif args.workers:
            cli.show_workers()
        elif args.lattice:
            cli.show_lattice()
        elif args.add_worker:
            cli.add_worker(args.add_worker[0], args.add_worker[1], args.add_worker[2])
        elif args.remove_worker:
            cli.remove_worker(args.remove_worker)
        elif args.reset_metrics:
            cli.reset_metrics()
        elif args.export_config:
            cli.export_config(args.export_config)
        elif args.import_config:
            cli.import_config(args.import_config)
        elif args.demo:
            await cli.run_demo()
        elif args.interactive:
            await cli.run_interactive_mode()
        else:
            cli.show_banner()
            cli.show_help()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        if cli.running:
            cli.stop_optimization()
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 