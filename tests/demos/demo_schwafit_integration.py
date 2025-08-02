#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwafit Integration Demo - Anti-Overfitting with TFF Sequencer
==============================================================

Demonstrates the integration of Schwafit anti-overfitting system
with the TFF sequencer for maintaining profit trajectory integrity
and preventing overfitting to trading data.

Features:
- Schwafit anti-overfitting monitoring
- TFF sequencer with natural rebalancing
- Profit trajectory integrity maintenance
- Volume spike analysis for profit creation
- Overfitting risk assessment and prevention
- Real-time intervention and recovery measures

Usage:
    python demo_schwafit_integration.py --demo     # Run comprehensive demo
    python demo_schwafit_integration.py --status   # Show integration status
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
    from core.schwafit_anti_overfitting import SchwafitAntiOverfitting
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Schwafit Integration not available: {e}")
    INTEGRATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchwafitIntegrationDemo:
    """Comprehensive demo of Schwafit integration with TFF sequencer."""
    
    def __init__(self):
        """Initialize the demo."""
        self.optimization = None
        self.schwafit = None
        self.running = False
    
    def show_banner(self):
        """Show demo banner."""
        print("🧠 SCHWAFIT INTEGRATION DEMO - ANTI-OVERFITTING WITH TFF SEQUENCER")
        print("=" * 80)
        print("Advanced anti-overfitting system integrated with TFF sequencer")
        print("Maintains profit trajectory integrity and prevents overfitting")
        print("Volume spike analysis for fast profit creation")
        print("Real-time intervention and recovery measures")
        print("=" * 80)
    
    async def initialize_system(self):
        """Initialize the Schwafit integration system."""
        if not INTEGRATION_AVAILABLE:
            print("❌ Schwafit Integration not available")
            return False
        
        try:
            # Initialize both systems
            self.optimization = DynamicWorkerOptimization()
            self.schwafit = SchwafitAntiOverfitting()
            
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
            
            # Orbital Tracker worker
            self.optimization.add_worker(
                "orbital_tracker_01", "orbital-node-1", "192.168.1.105",
                {"cpu_cores": 6, "gpu": True, "memory_gb": 12, "orbital_enabled": True}
            )
            
            print("✅ Schwafit Integration system initialized with 5 specialized workers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Schwafit integration: {e}")
            return False
    
    async def run_comprehensive_demo(self):
        """Run comprehensive Schwafit integration demonstration."""
        print("\n🌟 COMPREHENSIVE SCHWAFIT INTEGRATION DEMONSTRATION")
        print("=" * 60)
        
        if not await self.initialize_system():
            return
        
        print("🚀 Starting Schwafit integration optimization...")
        
        # Start both systems
        await self.optimization.start_optimization()
        await self.schwafit.start_monitoring()
        self.running = True
        
        print("⏳ Running Schwafit integration demo for 90 seconds...")
        print("   • TFF sequencer with natural rebalancing")
        print("   • Schwafit anti-overfitting monitoring")
        print("   • Profit trajectory integrity validation")
        print("   • Volume spike analysis for profit creation")
        print("   • Overfitting risk assessment and prevention")
        print("   • Real-time intervention and recovery measures")
        print("   • Stable pair mapping optimization")
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 90:
            elapsed = int(time.time() - start_time)
            
            # Get status from both systems
            optimization_status = self.optimization.get_optimization_status()
            schwafit_status = self.schwafit.get_schwafit_status()
            
            # Extract key metrics
            tff_status = optimization_status.get('tff_sequencer_status', {})
            schwafit_metrics = schwafit_status.get('overfitting_metrics', {})
            profit_trajectory = schwafit_status.get('profit_trajectory', {})
            
            print(f"\r⏱️  Elapsed: {elapsed}s | "
                  f"TFF Rounds: {tff_status.get('sequencer_rounds', 0)} | "
                  f"Overfitting Risk: {schwafit_metrics.get('risk_level', 'none')} | "
                  f"Trajectory: {profit_trajectory.get('state', 'stable')} | "
                  f"Integrity: {profit_trajectory.get('integrity', 0.0):.3f} | "
                  f"Volume Efficiency: {profit_trajectory.get('volume_efficiency', 0.0):.3f} | "
                  f"Profit Spikes: {profit_trajectory.get('profit_spikes', 0)}", end="")
            
            await asyncio.sleep(1)
        
        print("\n🛑 Stopping Schwafit integration demo...")
        self.optimization.stop_optimization()
        self.schwafit.stop_monitoring()
        self.running = False
        
        # Show comprehensive results
        await self._show_integration_results()
        
        print("\n✅ Schwafit Integration Demo completed!")
    
    async def _show_integration_results(self):
        """Show comprehensive Schwafit integration results."""
        print("\n📊 SCHWAFIT INTEGRATION RESULTS:")
        print("=" * 50)
        
        # Get final status from both systems
        optimization_status = self.optimization.get_optimization_status()
        schwafit_status = self.schwafit.get_schwafit_status()
        
        print(f"🧠 Schwafit Anti-Overfitting Status:")
        schwafit_metrics = schwafit_status.get('overfitting_metrics', {})
        print(f"  Risk Level: {schwafit_metrics.get('risk_level', 'none')}")
        print(f"  Overfitting Score: {schwafit_metrics.get('overfitting_score', 0.0):.3f}")
        print(f"  Generalization Gap: {schwafit_metrics.get('generalization_gap', 0.0):.3f}")
        print(f"  External Validation: {schwafit_metrics.get('external_validation', 0.0):.3f}")
        
        print(f"\n📈 Profit Trajectory Status:")
        profit_trajectory = schwafit_status.get('profit_trajectory', {})
        print(f"  State: {profit_trajectory.get('state', 'stable')}")
        print(f"  Score: {profit_trajectory.get('score', 0.0):.3f}")
        print(f"  Growth Rate: {profit_trajectory.get('growth_rate', 0.0):.3f}")
        print(f"  Volatility: {profit_trajectory.get('volatility', 0.0):.3f}")
        print(f"  Orbital Integrity: {profit_trajectory.get('integrity', 0.0):.3f}")
        print(f"  Volume Efficiency: {profit_trajectory.get('volume_efficiency', 0.0):.3f}")
        print(f"  Profit Spikes: {profit_trajectory.get('profit_spikes', 0)}")
        
        print(f"\n📊 Volume Analysis:")
        volume_analysis = schwafit_status.get('volume_analysis', {})
        print(f"  Recent Spikes: {volume_analysis.get('recent_spikes', 0)}")
        print(f"  Stable Pairs: {volume_analysis.get('stable_pairs', 0)}")
        print(f"  Avg Profit Potential: {volume_analysis.get('avg_profit_potential', 0.0):.3f}")
        
        print(f"\n🧠 TFF Sequencer Status:")
        tff_status = optimization_status.get('tff_sequencer_status', {})
        print(f"  Active Workers: {tff_status.get('active_workers', 0)}")
        print(f"  Sequencer Rounds: {tff_status.get('sequencer_rounds', 0)}")
        print(f"  Federated Learning Data: {tff_status.get('federated_learning_data_count', 0)}")
        print(f"  Natural Rebalancing Triggers: {tff_status.get('natural_rebalancing_triggers', 0)}")
        
        print(f"\n🎭 Role Distribution:")
        role_distribution = tff_status.get('role_distribution', {})
        for role, count in role_distribution.items():
            print(f"  {role.replace('_', ' ').title()}: {count}")
        
        print(f"\n📈 Performance Metrics:")
        opt_metrics = optimization_status['performance_metrics']
        schwafit_metrics = schwafit_status.get('performance_metrics', {})
        
        print(f"  TFF Sequencer Cycles: {opt_metrics['tff_sequencer_cycles']}")
        print(f"  Natural Rebalancing Events: {opt_metrics['natural_rebalancing_events']}")
        print(f"  Bit Flipping Detections: {opt_metrics['bit_flipping_detections']}")
        print(f"  Organic Strategy Evolutions: {opt_metrics['organic_strategy_evolutions']}")
        print(f"  Overfitting Assessments: {schwafit_metrics.get('overfitting_assessments', 0)}")
        print(f"  Trajectory Validations: {schwafit_metrics.get('trajectory_validations', 0)}")
        print(f"  Volume Spike Analyses: {schwafit_metrics.get('volume_spike_analyses', 0)}")
        print(f"  Anti-Overfitting Interventions: {schwafit_metrics.get('anti_overfitting_interventions', 0)}")
        print(f"  Profit Trajectory Corrections: {schwafit_metrics.get('profit_trajectory_corrections', 0)}")
        
        # Show alerts and interventions
        alerts_count = schwafit_status.get('alerts_count', 0)
        last_intervention = schwafit_status.get('last_intervention', 0)
        
        print(f"\n⚠️  System Alerts & Interventions:")
        print(f"  Overfitting Alerts: {alerts_count}")
        if last_intervention > 0:
            intervention_time = time.strftime('%H:%M:%S', time.localtime(last_intervention))
            print(f"  Last Intervention: {intervention_time}")
        else:
            print(f"  Last Intervention: None")
        
        # Show detailed worker status
        await self._show_detailed_worker_status()
    
    async def _show_detailed_worker_status(self):
        """Show detailed worker status with Schwafit integration information."""
        print(f"\n👥 DETAILED WORKER STATUS:")
        print("=" * 50)
        
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
            
            elif worker['worker_role'] == 'orbital_tracker':
                print(f"  Orbital Tracking: Active for trajectory integrity")
    
    def show_status(self):
        """Show current Schwafit integration status."""
        if not self.optimization or not self.schwafit:
            print("❌ Schwafit Integration not initialized")
            return
        
        optimization_status = self.optimization.get_optimization_status()
        schwafit_status = self.schwafit.get_schwafit_status()
        
        print("\n🧠 SCHWAFIT INTEGRATION STATUS:")
        print("=" * 40)
        print(f"Running: {'✅ Yes' if optimization_status['running'] else '❌ No'}")
        print(f"Downtime Mode: {'🌙 Yes' if optimization_status['downtime_mode'] else '☀️ No'}")
        print(f"Workers: {optimization_status['workers_count']}")
        
        # Schwafit metrics
        schwafit_metrics = schwafit_status.get('overfitting_metrics', {})
        print(f"Overfitting Risk: {schwafit_metrics.get('risk_level', 'none')}")
        print(f"Overfitting Score: {schwafit_metrics.get('overfitting_score', 0.0):.3f}")
        
        # Profit trajectory
        profit_trajectory = schwafit_status.get('profit_trajectory', {})
        print(f"Trajectory State: {profit_trajectory.get('state', 'stable')}")
        print(f"Orbital Integrity: {profit_trajectory.get('integrity', 0.0):.3f}")
        print(f"Volume Efficiency: {profit_trajectory.get('volume_efficiency', 0.0):.3f}")
        
        # TFF sequencer
        tff_status = optimization_status.get('tff_sequencer_status', {})
        print(f"TFF Rounds: {tff_status.get('sequencer_rounds', 0)}")
        print(f"Active Workers: {tff_status.get('active_workers', 0)}")
        print(f"Rebalancing Events: {optimization_status['performance_metrics']['natural_rebalancing_events']}")
        print(f"Bit Flipping: {optimization_status['performance_metrics']['bit_flipping_detections']}")
        print(f"Strategy Evolutions: {optimization_status['performance_metrics']['organic_strategy_evolutions']}")
        print(f"Interventions: {schwafit_status.get('performance_metrics', {}).get('anti_overfitting_interventions', 0)}")
    
    def show_help(self):
        """Show demo help information."""
        print("\n📋 SCHWAFIT INTEGRATION DEMO OPTIONS:")
        print("  --demo     Run comprehensive Schwafit integration demonstration")
        print("  --status   Show current integration status")
        print("  --help     Show this help")
        
        print("\n🧠 SCHWAFIT INTEGRATION FEATURES:")
        print("  • Schwafit anti-overfitting monitoring")
        print("  • TFF sequencer with natural rebalancing")
        print("  • Profit trajectory integrity maintenance")
        print("  • Volume spike analysis for profit creation")
        print("  • Overfitting risk assessment and prevention")
        print("  • Real-time intervention and recovery measures")
        print("  • Stable pair mapping optimization")
        print("  • Orbital mapping preservation")
        print("  • Cross-validation with external data sources")
        
        print("\n💡 EXAMPLES:")
        print("  python demo_schwafit_integration.py --demo")
        print("  python demo_schwafit_integration.py --status")


async def main():
    """Main demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Schwafit Integration Demo - Anti-Overfitting with TFF Sequencer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_schwafit_integration.py --demo     # Run comprehensive demo
  python demo_schwafit_integration.py --status   # Show integration status
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run comprehensive Schwafit integration demonstration')
    parser.add_argument('--status', action='store_true', help='Show current integration status')
    
    args = parser.parse_args()
    
    demo = SchwafitIntegrationDemo()
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
            demo.schwafit.stop_monitoring()
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 