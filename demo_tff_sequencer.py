#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFF Sequencer Demo - Natural Rebalancing & Organic Strategy Evolution
====================================================================

Demonstrates the sophisticated TensorFlow Federated sequencer system
with natural rebalancing, bit flipping detection, and organic strategy evolution.

Features:
- TFF sequencer role assignment
- Natural rebalancing without forced rebalancing
- Volume and indicator-based strategy generation
- Bit flipping detection
- Organic strategy evolution
- Federated learning rounds
- Flask AI agent communication
- Memory registry updates

Usage:
    python demo_tff_sequencer.py --demo     # Run comprehensive demo
    python demo_tff_sequencer.py --status   # Show TFF sequencer status
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from core.dynamic_worker_optimization import DynamicWorkerOptimization
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ TFF Sequencer not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TFFSequencerDemo:
    """Comprehensive demo of the TFF sequencer system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.optimization = None
        self.running = False
    
    def show_banner(self):
        """Show demo banner."""
        print("🧠 TFF SEQUENCER DEMO - NATURAL REBALANCING & ORGANIC STRATEGY EVOLUTION")
        print("=" * 80)
        print("Sophisticated TensorFlow Federated sequencer with natural rebalancing")
        print("Volume and indicator-based strategy generation with bit flipping detection")
        print("Organic strategy evolution without forced rebalancing")
        print("=" * 80)
    
    async def initialize_system(self):
        """Initialize the TFF sequencer system."""
        if not OPTIMIZATION_AVAILABLE:
            print("❌ TFF Sequencer not available")
            return False
        
        try:
            self.optimization = DynamicWorkerOptimization()
            
            # Add specialized workers for TFF sequencer roles
            print("👥 Adding specialized TFF sequencer workers...")
            
            # TFF Sequencer worker
            self.optimization.add_worker(
                "tff_sequencer_01", "tff-node-1", "192.168.1.101",
                {"cpu_cores": 8, "gpu": True, "memory_gb": 16, "tff_enabled": True}
            )
            
            # Strategy Optimizer worker
            self.optimization.add_worker(
                "strategy_optimizer_01", "strategy-node-1", "192.168.1.102",
                {"cpu_cores": 6, "gpu": True, "memory_gb": 12, "strategy_enabled": True}
            )
            
            # Volume Analyzer worker
            self.optimization.add_worker(
                "volume_analyzer_01", "volume-node-1", "192.168.1.103",
                {"cpu_cores": 4, "gpu": False, "memory_gb": 8, "volume_enabled": True}
            )
            
            # Indicator Processor worker
            self.optimization.add_worker(
                "indicator_processor_01", "indicator-node-1", "192.168.1.104",
                {"cpu_cores": 4, "gpu": False, "memory_gb": 8, "indicator_enabled": True}
            )
            
            # Profit Lattice Mapper worker
            self.optimization.add_worker(
                "lattice_mapper_01", "lattice-node-1", "192.168.1.105",
                {"cpu_cores": 6, "gpu": True, "memory_gb": 12, "lattice_enabled": True}
            )
            
            print("✅ TFF Sequencer system initialized with 5 specialized workers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TFF sequencer: {e}")
            return False
    
    async def run_comprehensive_demo(self):
        """Run comprehensive TFF sequencer demonstration."""
        print("\n🌟 COMPREHENSIVE TFF SEQUENCER DEMONSTRATION")
        print("=" * 50)
        
        if not await self.initialize_system():
            return
        
        print("🚀 Starting TFF sequencer optimization...")
        
        # Start optimization
        await self.optimization.start_optimization()
        self.running = True
        
        print("⏳ Running TFF sequencer demo for 60 seconds...")
        print("   • Assigning TFF sequencer roles to least-used workers")
        print("   • Executing federated learning rounds")
        print("   • Generating organic strategies based on volume and indicators")
        print("   • Detecting natural rebalancing triggers")
        print("   • Monitoring bit flipping detection")
        print("   • Tracking organic strategy evolution")
        print("   • Updating memory registry with AI insights")
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 60:
            elapsed = int(time.time() - start_time)
            status = self.optimization.get_optimization_status()
            tff_status = status.get('tff_sequencer_status', {})
            
            print(f"\r⏱️  Elapsed: {elapsed}s | "
                  f"TFF Rounds: {tff_status.get('sequencer_rounds', 0)} | "
                  f"Active Workers: {tff_status.get('active_workers', 0)} | "
                  f"Rebalancing Events: {status['performance_metrics']['natural_rebalancing_events']} | "
                  f"Bit Flipping: {status['performance_metrics']['bit_flipping_detections']} | "
                  f"Strategy Evolutions: {status['performance_metrics']['organic_strategy_evolutions']}", end="")
            
            await asyncio.sleep(1)
        
        print("\n🛑 Stopping TFF sequencer demo...")
        self.optimization.stop_optimization()
        self.running = False
        
        # Show comprehensive results
        await self._show_tff_sequencer_results()
        
        print("\n✅ TFF Sequencer Demo completed!")
    
    async def _show_tff_sequencer_results(self):
        """Show comprehensive TFF sequencer results."""
        print("\n📊 TFF SEQUENCER RESULTS:")
        print("=" * 40)
        
        status = self.optimization.get_optimization_status()
        tff_status = status.get('tff_sequencer_status', {})
        
        print(f"🧠 TFF Sequencer Status:")
        print(f"  Active Workers: {tff_status.get('active_workers', 0)}")
        print(f"  Sequencer Rounds: {tff_status.get('sequencer_rounds', 0)}")
        print(f"  Federated Learning Data: {tff_status.get('federated_learning_data_count', 0)}")
        print(f"  Natural Rebalancing Triggers: {tff_status.get('natural_rebalancing_triggers', 0)}")
        print(f"  Strategy Evolution History: {tff_status.get('strategy_evolution_history_count', 0)}")
        
        print(f"\n🎭 Role Distribution:")
        role_distribution = tff_status.get('role_distribution', {})
        for role, count in role_distribution.items():
            print(f"  {role.replace('_', ' ').title()}: {count}")
        
        print(f"\n🔄 Rebalancing Statistics:")
        rebalancing_stats = tff_status.get('rebalancing_statistics', {})
        print(f"  Workers with Rebalancing: {rebalancing_stats.get('workers_with_rebalancing', 0)}")
        print(f"  Workers with Bit Flipping: {rebalancing_stats.get('workers_with_bit_flipping', 0)}")
        print(f"  Average Rebalancing Score: {rebalancing_stats.get('avg_rebalancing_score', 0.0):.3f}")
        print(f"  Total Evolution Events: {rebalancing_stats.get('total_evolution_events', 0)}")
        
        print(f"\n📈 Performance Metrics:")
        metrics = status['performance_metrics']
        print(f"  TFF Sequencer Cycles: {metrics['tff_sequencer_cycles']}")
        print(f"  Natural Rebalancing Events: {metrics['natural_rebalancing_events']}")
        print(f"  Bit Flipping Detections: {metrics['bit_flipping_detections']}")
        print(f"  Organic Strategy Evolutions: {metrics['organic_strategy_evolutions']}")
        print(f"  Optimization Cycles: {metrics['optimization_cycles']}")
        print(f"  Assets Mapped: {metrics['assets_mapped']}")
        print(f"  AI Analyses: {metrics['ai_analyses']}")
        print(f"  Memory Updates: {metrics['memory_updates']}")
        print(f"  Worker Reassignments: {metrics['worker_reassignments']}")
        
        # Show detailed worker status
        await self._show_detailed_worker_status()
    
    async def _show_detailed_worker_status(self):
        """Show detailed worker status with TFF sequencer information."""
        print(f"\n👥 DETAILED WORKER STATUS:")
        print("=" * 40)
        
        worker_status = self.optimization.get_worker_status()
        
        for worker_id, worker in worker_status.items():
            print(f"\n🔧 {worker_id} ({worker['hostname']})")
            print(f"  Role: {worker['worker_role']}")
            print(f"  Mode: {worker['optimization_mode']}")
            print(f"  Usage: {worker['current_usage']:.1%}")
            print(f"  Natural Rebalancing Score: {worker['natural_rebalancing_score']:.3f}")
            print(f"  Bit Flipping Detection: {'✅ Yes' if worker['bit_flipping_detection'] else '❌ No'}")
            print(f"  Organic Strategy Evolutions: {worker['organic_strategy_evolution_count']}")
            print(f"  Assigned Assets: {', '.join(worker['assigned_assets']) if worker['assigned_assets'] else 'None'}")
            
            # Show role-specific data
            if worker['worker_role'] == 'tff_sequencer':
                tff_data = worker['tff_sequencer_data']
                print(f"  TFF Data: Rounds={tff_data.get('federated_rounds', 0)}, "
                      f"Updates={tff_data.get('model_updates', 0)}, "
                      f"Convergence={tff_data.get('convergence_score', 0.0):.3f}")
            
            elif worker['worker_role'] == 'strategy_optimizer':
                strategy_data = worker['strategy_optimization_data']
                print(f"  Strategy Data: Rounds={strategy_data.get('optimization_rounds', 0)}, "
                      f"Strategies={strategy_data.get('strategy_count', 0)}")
            
            elif worker['worker_role'] == 'volume_analyzer':
                volume_data = worker['volume_analysis_data']
                print(f"  Volume Data: Patterns={len(volume_data.get('volume_patterns', {}))}, "
                      f"Triggers={len(volume_data.get('rebalancing_triggers', []))}")
            
            elif worker['worker_role'] == 'indicator_processor':
                indicator_data = worker['indicator_processing_data']
                print(f"  Indicator Data: Cache={len(indicator_data.get('indicator_cache', {}))}, "
                      f"Processed={len(indicator_data.get('processed_indicators', []))}")
    
    def show_status(self):
        """Show current TFF sequencer status."""
        if not self.optimization:
            print("❌ TFF Sequencer not initialized")
            return
        
        status = self.optimization.get_optimization_status()
        tff_status = status.get('tff_sequencer_status', {})
        
        print("\n🧠 TFF SEQUENCER STATUS:")
        print("=" * 30)
        print(f"Running: {'✅ Yes' if status['running'] else '❌ No'}")
        print(f"Downtime Mode: {'🌙 Yes' if status['downtime_mode'] else '☀️ No'}")
        print(f"Workers: {status['workers_count']}")
        print(f"Active TFF Workers: {tff_status.get('active_workers', 0)}")
        print(f"Sequencer Rounds: {tff_status.get('sequencer_rounds', 0)}")
        print(f"Natural Rebalancing Events: {status['performance_metrics']['natural_rebalancing_events']}")
        print(f"Bit Flipping Detections: {status['performance_metrics']['bit_flipping_detections']}")
        print(f"Organic Strategy Evolutions: {status['performance_metrics']['organic_strategy_evolutions']}")
        
        print(f"\n🎭 Role Distribution:")
        role_distribution = tff_status.get('role_distribution', {})
        for role, count in role_distribution.items():
            print(f"  {role.replace('_', ' ').title()}: {count}")
    
    def show_help(self):
        """Show demo help information."""
        print("\n📋 TFF SEQUENCER DEMO OPTIONS:")
        print("  --demo     Run comprehensive TFF sequencer demonstration")
        print("  --status   Show current TFF sequencer status")
        print("  --help     Show this help")
        
        print("\n🧠 TFF SEQUENCER FEATURES:")
        print("  • TensorFlow Federated sequencer roles")
        print("  • Natural rebalancing without forced rebalancing")
        print("  • Volume and indicator-based strategy generation")
        print("  • Bit flipping detection")
        print("  • Organic strategy evolution")
        print("  • Federated learning rounds")
        print("  • Flask AI agent communication")
        print("  • Memory registry updates")
        print("  • Role-based worker optimization")
        
        print("\n💡 EXAMPLES:")
        print("  python demo_tff_sequencer.py --demo")
        print("  python demo_tff_sequencer.py --status")


async def main():
    """Main demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TFF Sequencer Demo - Natural Rebalancing & Organic Strategy Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_tff_sequencer.py --demo     # Run comprehensive demo
  python demo_tff_sequencer.py --status   # Show TFF sequencer status
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run comprehensive TFF sequencer demonstration')
    parser.add_argument('--status', action='store_true', help='Show current TFF sequencer status')
    
    args = parser.parse_args()
    
    demo = TFFSequencerDemo()
    demo.show_banner()
    
    try:
        if args.demo:
            await demo.run_comprehensive_demo()
        elif args.status:
            demo.show_status()
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