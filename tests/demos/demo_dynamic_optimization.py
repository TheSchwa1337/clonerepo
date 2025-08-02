#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Worker Optimization Demo
================================

Comprehensive demonstration of the sophisticated downtime optimization system
with profit lattice mapping, Flask AI agent communication, and memory registry updates.

Features demonstrated:
- Downtime optimization during low-trading hours
- Profit lattice mapping for asset relationships
- Flask AI agent communication
- Memory registry updates
- Dynamic ticker assignment with weighted randomization
- Orbital performance tracking
- Swing timing optimization
- CLI and GUI integration

Usage:
    python demo_dynamic_optimization.py --cli      # Run CLI demo
    python demo_dynamic_optimization.py --gui      # Run GUI demo
    python demo_dynamic_optimization.py --demo     # Run automated demo
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from core.dynamic_worker_optimization import DynamicWorkerOptimization
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dynamic Worker Optimization not available: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    from cli.dynamic_optimization_cli import DynamicOptimizationCLI
    CLI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dynamic Optimization CLI not available: {e}")
    CLI_AVAILABLE = False

try:
    from enhanced_advanced_options_gui import show_enhanced_advanced_options
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced Advanced Options GUI not available: {e}")
    GUI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicOptimizationDemo:
    """Comprehensive demo of the Dynamic Worker Optimization system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.optimization = None
        self.cli = None
        self.running = False
    
    def show_banner(self):
        """Show demo banner."""
        print("🚀 SCHWABOT DYNAMIC WORKER OPTIMIZATION DEMO")
        print("=" * 60)
        print("Advanced downtime optimization with profit lattice mapping")
        print("Flask AI agent communication and memory registry updates")
        print("Dynamic ticker assignment with weighted randomization")
        print("=" * 60)
    
    async def initialize_system(self):
        """Initialize the optimization system."""
        if not OPTIMIZATION_AVAILABLE:
            print("❌ Dynamic Worker Optimization not available")
            return False
        
        try:
            # Initialize optimization system
            self.optimization = DynamicWorkerOptimization()
            
            # Initialize CLI
            self.cli = DynamicOptimizationCLI()
            await self.cli.initialize_optimization()
            
            print("✅ Dynamic Worker Optimization system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    async def run_cli_demo(self):
        """Run CLI demonstration."""
        print("\n💻 CLI DEMONSTRATION")
        print("=" * 30)
        
        if not await self.initialize_system():
            return
        
        # Show initial status
        print("\n📊 Initial System Status:")
        self.cli.show_status()
        
        # Add workers
        print("\n👥 Adding Workers:")
        self.cli.add_worker("demo_worker_1", "demo-node-1", "192.168.1.201")
        self.cli.add_worker("demo_worker_2", "demo-node-2", "192.168.1.202")
        self.cli.add_worker("demo_worker_3", "demo-node-3", "192.168.1.203")
        
        # Show worker status
        print("\n👥 Worker Status:")
        self.cli.show_workers()
        
        # Start optimization
        print("\n🚀 Starting Optimization:")
        await self.cli.start_optimization()
        
        # Run for a short time
        print("\n⏳ Running optimization for 10 seconds...")
        await asyncio.sleep(10)
        
        # Show results
        print("\n📊 Optimization Results:")
        self.cli.show_status()
        self.cli.show_lattice()
        
        # Stop optimization
        print("\n🛑 Stopping Optimization:")
        self.cli.stop_optimization()
        
        print("\n✅ CLI Demo completed!")
    
    def run_gui_demo(self):
        """Run GUI demonstration."""
        print("\n🎨 GUI DEMONSTRATION")
        print("=" * 30)
        
        print("🚀 Launching Enhanced Advanced Options GUI...")
        print("   • Navigate to the '⚙️ Advanced Settings' tab")
        print("   • Find the '🚀 Dynamic Worker Optimization System' section")
        print("   • Configure optimization settings")
        print("   • Test the system functionality")
        
        try:
            show_enhanced_advanced_options()
        except Exception as e:
            logger.error(f"❌ GUI demo failed: {e}")
    
    async def run_automated_demo(self):
        """Run automated demonstration."""
        print("\n🤖 AUTOMATED DEMONSTRATION")
        print("=" * 35)
        
        if not await self.initialize_system():
            return
        
        print("🚀 Starting automated optimization demo...")
        
        # Start optimization
        await self.optimization.start_optimization()
        self.running = True
        
        print("⏳ Running automated demo for 60 seconds...")
        print("   • Simulating downtime optimization")
        print("   • Mapping assets to profit lattice")
        print("   • Communicating with Flask AI agents")
        print("   • Updating memory registry")
        print("   • Applying dynamic ticker assignment")
        print("   • Tracking orbital performance")
        print("   • Optimizing swing timing")
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 60:
            elapsed = int(time.time() - start_time)
            status = self.optimization.get_optimization_status()
            
            print(f"\r⏱️  Elapsed: {elapsed}s | "
                  f"Cycles: {status['performance_metrics']['optimization_cycles']} | "
                  f"Assets: {status['assets_count']} | "
                  f"AI Analyses: {status['performance_metrics']['ai_analyses']}", end="")
            
            await asyncio.sleep(1)
        
        print("\n🛑 Stopping automated demo...")
        self.optimization.stop_optimization()
        self.running = False
        
        # Show final results
        print("\n📊 FINAL RESULTS:")
        final_status = self.optimization.get_optimization_status()
        
        print(f"  Optimization Cycles: {final_status['performance_metrics']['optimization_cycles']}")
        print(f"  Assets Mapped: {final_status['performance_metrics']['assets_mapped']}")
        print(f"  AI Analyses: {final_status['performance_metrics']['ai_analyses']}")
        print(f"  Memory Updates: {final_status['performance_metrics']['memory_updates']}")
        print(f"  Worker Reassignments: {final_status['performance_metrics']['worker_reassignments']}")
        print(f"  Profit Improvements: {final_status['performance_metrics']['profit_improvements']:.2f}%")
        
        # Show profit lattice summary
        lattice_summary = self.optimization.get_profit_lattice_summary()
        if 'message' not in lattice_summary:
            print(f"\n📊 Profit Lattice Summary:")
            print(f"  Total Assets: {lattice_summary['total_assets']}")
            print(f"  Avg Profit Potential: {lattice_summary['avg_profit_potential']:.2%}")
            print(f"  Max Profit Potential: {lattice_summary['max_profit_potential']:.2%}")
            print(f"  Swing Opportunities: {lattice_summary['swing_opportunities']}")
        
        print("\n✅ Automated Demo completed!")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration."""
        print("\n🌟 COMPREHENSIVE DEMONSTRATION")
        print("=" * 40)
        
        if not await self.initialize_system():
            return
        
        print("🚀 Starting comprehensive demo...")
        
        # Phase 1: System Initialization
        print("\n📋 Phase 1: System Initialization")
        print("   • Initializing Dynamic Worker Optimization")
        print("   • Setting up worker nodes")
        print("   • Configuring profit lattice mapping")
        print("   • Establishing Flask AI agent communication")
        print("   • Preparing memory registry")
        
        # Add demo workers
        self.optimization.add_worker(
            "comprehensive_worker_1", "comp-node-1", "192.168.1.101",
            {"cpu_cores": 8, "gpu": True, "memory_gb": 16}
        )
        self.optimization.add_worker(
            "comprehensive_worker_2", "comp-node-2", "192.168.1.102",
            {"cpu_cores": 4, "gpu": False, "memory_gb": 8}
        )
        self.optimization.add_worker(
            "comprehensive_worker_3", "comp-node-3", "192.168.1.103",
            {"cpu_cores": 6, "gpu": True, "memory_gb": 12}
        )
        
        print("✅ Phase 1 completed")
        
        # Phase 2: Optimization Execution
        print("\n📋 Phase 2: Optimization Execution")
        print("   • Starting optimization system")
        print("   • Running downtime optimization")
        print("   • Mapping assets to profit lattice")
        print("   • Communicating with AI agents")
        print("   • Updating memory registry")
        
        await self.optimization.start_optimization()
        self.running = True
        
        # Run optimization for 30 seconds
        await asyncio.sleep(30)
        
        print("✅ Phase 2 completed")
        
        # Phase 3: Analysis and Results
        print("\n📋 Phase 3: Analysis and Results")
        print("   • Analyzing optimization results")
        print("   • Calculating performance metrics")
        print("   • Generating profit lattice summary")
        print("   • Evaluating worker assignments")
        print("   • Assessing AI agent performance")
        
        # Stop optimization
        self.optimization.stop_optimization()
        self.running = False
        
        # Show comprehensive results
        status = self.optimization.get_optimization_status()
        worker_status = self.optimization.get_worker_status()
        lattice_summary = self.optimization.get_profit_lattice_summary()
        
        print("\n📊 COMPREHENSIVE RESULTS:")
        print(f"  System Running: {'✅ Yes' if status['running'] else '❌ No'}")
        print(f"  Downtime Mode: {'🌙 Yes' if status['downtime_mode'] else '☀️ No'}")
        print(f"  Workers Active: {status['workers_count']}")
        print(f"  Assets Processed: {status['assets_count']}")
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
        
        print("\n👥 WORKER STATUS:")
        for worker_id, worker in worker_status.items():
            print(f"  {worker_id}: {worker['hostname']} - Usage: {worker['current_usage']:.1%}")
        
        if 'message' not in lattice_summary:
            print("\n📊 PROFIT LATTICE SUMMARY:")
            print(f"  Total Assets: {lattice_summary['total_assets']}")
            print(f"  Avg Profit Potential: {lattice_summary['avg_profit_potential']:.2%}")
            print(f"  Max Profit Potential: {lattice_summary['max_profit_potential']:.2%}")
            print(f"  Swing Opportunities: {lattice_summary['swing_opportunities']}")
            
            print("\n🌌 ORBITAL DISTRIBUTION:")
            for orbital, count in lattice_summary['orbital_distribution'].items():
                print(f"  {orbital.replace('_', ' ').title()}: {count}")
        
        print("\n✅ Phase 3 completed")
        print("\n🎉 COMPREHENSIVE DEMO COMPLETED!")
    
    def show_help(self):
        """Show demo help information."""
        print("\n📋 DEMO OPTIONS:")
        print("  --cli           Run CLI demonstration")
        print("  --gui           Run GUI demonstration")
        print("  --demo          Run automated demonstration")
        print("  --comprehensive Run comprehensive demonstration")
        print("  --help          Show this help")
        
        print("\n🎯 DEMO FEATURES:")
        print("  • Downtime optimization (1 AM - 4 AM)")
        print("  • Profit lattice mapping")
        print("  • Flask AI agent communication")
        print("  • Memory registry updates")
        print("  • Dynamic ticker assignment")
        print("  • Orbital performance tracking")
        print("  • Swing timing optimization")
        print("  • CLI and GUI integration")
        
        print("\n💡 EXAMPLES:")
        print("  python demo_dynamic_optimization.py --cli")
        print("  python demo_dynamic_optimization.py --gui")
        print("  python demo_dynamic_optimization.py --demo")
        print("  python demo_dynamic_optimization.py --comprehensive")


async def main():
    """Main demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Schwabot Dynamic Worker Optimization Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_dynamic_optimization.py --cli           # Run CLI demo
  python demo_dynamic_optimization.py --gui           # Run GUI demo
  python demo_dynamic_optimization.py --demo          # Run automated demo
  python demo_dynamic_optimization.py --comprehensive # Run comprehensive demo
        """
    )
    
    parser.add_argument('--cli', action='store_true', help='Run CLI demonstration')
    parser.add_argument('--gui', action='store_true', help='Run GUI demonstration')
    parser.add_argument('--demo', action='store_true', help='Run automated demonstration')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive demonstration')
    
    args = parser.parse_args()
    
    demo = DynamicOptimizationDemo()
    demo.show_banner()
    
    try:
        if args.cli:
            await demo.run_cli_demo()
        elif args.gui:
            demo.run_gui_demo()
        elif args.demo:
            await demo.run_automated_demo()
        elif args.comprehensive:
            await demo.run_comprehensive_demo()
        else:
            demo.show_help()
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        if demo.running:
            demo.optimization.stop_optimization()
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 