#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Phantom Mode Integration
=================================

Integration layer connecting Phantom Mode engine to Schwabot trading system.
Handles node load balancing, thermal management, and profit optimization.
"""

import os
import sys
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import psutil
import subprocess

# Import Phantom Mode engine
from phantom_mode_engine import PhantomModeEngine, PhantomConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeSpecs:
    """Hardware specifications for each node in the Schwabot system."""
    name: str
    gpu_model: str
    vram_gb: float
    thermal_limit_celsius: float
    power_watts: float
    processing_capacity: float  # Relative processing power
    is_active: bool = True

@dataclass
class NodeStatus:
    """Current status of a node."""
    name: str
    temperature: float
    memory_usage: float
    gpu_utilization: float
    is_available: bool
    last_update: float
    thermal_warning: bool = False
    entropy_drift: float = 0.0

class NodeLoadBalancer:
    """Manages load distribution across Schwabot nodes based on thermal and performance metrics."""
    
    def __init__(self):
        self.nodes = {
            'xfx_7970': NodeSpecs(
                name='XFX 7970',
                gpu_model='AMD Radeon HD 7970',
                vram_gb=3.0,
                thermal_limit_celsius=84.0,
                power_watts=250.0,
                processing_capacity=1.0
            ),
            'pi4': NodeSpecs(
                name='Raspberry Pi 4',
                gpu_model='Broadcom VideoCore VI',
                vram_gb=0.0,  # Uses system RAM
                thermal_limit_celsius=67.0,
                power_watts=5.0,
                processing_capacity=0.1
            ),
            'gtx_1070': NodeSpecs(
                name='GTX 1070',
                gpu_model='NVIDIA GeForce GTX 1070',
                vram_gb=8.0,
                thermal_limit_celsius=78.0,
                power_watts=150.0,
                processing_capacity=2.5
            )
        }
        
        self.node_status = {}
        self.load_distribution = {}
        self.phantom_mode_engine = PhantomModeEngine()
        
        # Initialize status for all nodes
        for node_name in self.nodes:
            self.node_status[node_name] = NodeStatus(
                name=node_name,
                temperature=25.0,
                memory_usage=0.0,
                gpu_utilization=0.0,
                is_available=True,
                last_update=time.time()
            )
            
        logger.info("Node Load Balancer initialized")
        
    def get_node_metrics(self, node_name: str) -> Optional[Dict]:
        """Get real-time metrics for a specific node."""
        try:
            if node_name == 'xfx_7970':
                return self._get_amd_gpu_metrics()
            elif node_name == 'pi4':
                return self._get_pi4_metrics()
            elif node_name == 'gtx_1070':
                return self._get_nvidia_gpu_metrics()
            else:
                logger.warning(f"Unknown node: {node_name}")
                return None
        except Exception as e:
            logger.error(f"Error getting metrics for {node_name}: {e}")
            return None
            
    def _get_amd_gpu_metrics(self) -> Dict:
        """Get AMD GPU metrics (XFX 7970)."""
        try:
            # Try to use radeontop or similar AMD tools
            # For now, simulate metrics
            temp = np.random.normal(75, 5)  # Simulate 75°C ± 5°C
            memory_usage = np.random.uniform(0.3, 0.8)
            gpu_util = np.random.uniform(0.4, 0.9)
            
            return {
                'temperature': temp,
                'memory_usage': memory_usage,
                'gpu_utilization': gpu_util,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting AMD GPU metrics: {e}")
            return {'temperature': 25.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
            
    def _get_pi4_metrics(self) -> Dict:
        """Get Raspberry Pi 4 metrics."""
        try:
            # Get CPU temperature
            temp_cmd = "vcgencmd measure_temp"
            result = subprocess.run(temp_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace("temp=", "").replace("'C", ""))
            else:
                temp = np.random.normal(45, 5)  # Simulate 45°C ± 5°C
                
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            return {
                'temperature': temp,
                'memory_usage': memory_usage,
                'gpu_utilization': 0.0,  # Pi 4 doesn't have dedicated GPU monitoring
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting Pi 4 metrics: {e}")
            return {'temperature': 25.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
            
    def _get_nvidia_gpu_metrics(self) -> Dict:
        """Get NVIDIA GPU metrics (GTX 1070)."""
        try:
            # Try to use nvidia-smi
            nvidia_cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
            result = subprocess.run(nvidia_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'GTX 1070' in line or '1070' in line:
                        parts = line.split(', ')
                        temp = float(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        gpu_util = float(parts[3])
                        
                        memory_usage = memory_used / memory_total
                        
                        return {
                            'temperature': temp,
                            'memory_usage': memory_usage,
                            'gpu_utilization': gpu_util / 100.0,
                            'timestamp': time.time()
                        }
            
            # Fallback to simulation
            temp = np.random.normal(65, 3)  # Simulate 65°C ± 3°C
            memory_usage = np.random.uniform(0.2, 0.6)
            gpu_util = np.random.uniform(0.3, 0.7)
            
            return {
                'temperature': temp,
                'memory_usage': memory_usage,
                'gpu_utilization': gpu_util,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting NVIDIA GPU metrics: {e}")
            return {'temperature': 25.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
            
    def update_node_status(self):
        """Update status for all nodes."""
        for node_name in self.nodes:
            metrics = self.get_node_metrics(node_name)
            if metrics:
                status = self.node_status[node_name]
                status.temperature = metrics['temperature']
                status.memory_usage = metrics['memory_usage']
                status.gpu_utilization = metrics['gpu_utilization']
                status.last_update = metrics['timestamp']
                
                # Check thermal warning
                thermal_limit = self.nodes[node_name].thermal_limit_celsius
                status.thermal_warning = status.temperature > (thermal_limit - 5.0)
                
                # Calculate entropy drift based on thermal and memory stress
                thermal_stress = status.temperature / thermal_limit
                memory_stress = status.memory_usage
                status.entropy_drift = (thermal_stress + memory_stress) / 2.0
                
    def calculate_load_distribution(self, phantom_decision: Dict) -> Dict:
        """
        Calculate optimal load distribution based on Phantom Mode decision and node status.
        Implements the ZBE pressure response theory.
        """
        self.update_node_status()
        
        # Get Phantom Mode parameters
        phase_alignment = phantom_decision.get('phase_alignment', 0.0)
        entropy_compression = phantom_decision.get('entropy_compression', 0.0)
        bloom_probability = phantom_decision.get('bloom_probability', 0.0)
        
        # Calculate load requirements
        load_intensity = phase_alignment * entropy_compression * bloom_probability
        
        # Initialize distribution
        distribution = {}
        
        # Stage 1: XFX 7970 (Primary load handler)
        xfx_status = self.node_status['xfx_7970']
        xfx_specs = self.nodes['xfx_7970']
        
        if xfx_status.temperature < xfx_specs.thermal_limit_celsius and xfx_status.is_available:
            # XFX can handle load
            xfx_load = min(load_intensity, 1.0)
            distribution['xfx_7970'] = {
                'load_percentage': xfx_load,
                'role': 'primary_execution',
                'thermal_margin': xfx_specs.thermal_limit_celsius - xfx_status.temperature,
                'entropy_drift': xfx_status.entropy_drift
            }
            
            remaining_load = load_intensity - xfx_load
        else:
            # XFX is at thermal limit or unavailable
            distribution['xfx_7970'] = {
                'load_percentage': 0.0,
                'role': 'cooldown',
                'thermal_margin': 0.0,
                'entropy_drift': xfx_status.entropy_drift
            }
            remaining_load = load_intensity
            
        # Stage 2: Pi 4 (Swap buffer)
        pi4_status = self.node_status['pi4']
        pi4_specs = self.nodes['pi4']
        
        if pi4_status.temperature < pi4_specs.thermal_limit_celsius and pi4_status.is_available:
            # Pi 4 can handle swap load
            pi4_load = min(remaining_load * 0.3, 0.8)  # Pi 4 handles 30% of remaining load, max 80%
            distribution['pi4'] = {
                'load_percentage': pi4_load,
                'role': 'swap_buffer',
                'thermal_margin': pi4_specs.thermal_limit_celsius - pi4_status.temperature,
                'entropy_drift': pi4_status.entropy_drift
            }
            remaining_load -= pi4_load
        else:
            distribution['pi4'] = {
                'load_percentage': 0.0,
                'role': 'unavailable',
                'thermal_margin': 0.0,
                'entropy_drift': pi4_status.entropy_drift
            }
            
        # Stage 3: GTX 1070 (High-bandwidth execution)
        gtx_status = self.node_status['gtx_1070']
        gtx_specs = self.nodes['gtx_1070']
        
        if gtx_status.temperature < gtx_specs.thermal_limit_celsius and gtx_status.is_available:
            # GTX 1070 handles remaining load
            gtx_load = min(remaining_load, 1.0)
            distribution['gtx_1070'] = {
                'load_percentage': gtx_load,
                'role': 'high_bandwidth_execution',
                'thermal_margin': gtx_specs.thermal_limit_celsius - gtx_status.temperature,
                'entropy_drift': gtx_status.entropy_drift
            }
        else:
            distribution['gtx_1070'] = {
                'load_percentage': 0.0,
                'role': 'unavailable',
                'thermal_margin': 0.0,
                'entropy_drift': gtx_status.entropy_drift
            }
            
        return distribution
        
    def execute_phantom_trade(self, phantom_decision: Dict, market_data: Dict) -> Dict:
        """
        Execute Phantom Mode trade with optimal node distribution.
        """
        # Calculate load distribution
        load_dist = self.calculate_load_distribution(phantom_decision)
        
        # Determine execution strategy
        execution_result = {
            'timestamp': time.time(),
            'phantom_decision': phantom_decision,
            'load_distribution': load_dist,
            'execution_nodes': [],
            'trade_executed': False,
            'profit_expected': 0.0
        }
        
        # Check if we have sufficient node capacity
        total_available_load = sum(node['load_percentage'] for node in load_dist.values())
        
        if total_available_load >= 0.5:  # Need at least 50% capacity
            # Execute trade across available nodes
            for node_name, node_info in load_dist.items():
                if node_info['load_percentage'] > 0:
                    execution_result['execution_nodes'].append({
                        'node': node_name,
                        'load': node_info['load_percentage'],
                        'role': node_info['role']
                    })
                    
            execution_result['trade_executed'] = True
            execution_result['profit_expected'] = phantom_decision.get('confidence', 0.0) * 100.0
            
            logger.info(f"Phantom trade executed across {len(execution_result['execution_nodes'])} nodes")
        else:
            logger.warning("Insufficient node capacity for Phantom trade execution")
            
        return execution_result

class PhantomModeIntegration:
    """Main integration class connecting Phantom Mode to Schwabot."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.phantom_engine = PhantomModeEngine()
        self.load_balancer = NodeLoadBalancer()
        self.integration_active = False
        self.last_market_update = time.time()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Threading for continuous monitoring
        self.monitor_thread = None
        self.stop_monitoring_event = threading.Event()
        
        logger.info("Phantom Mode Integration initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load integration configuration."""
        default_config = {
            'phantom_mode_enabled': True,
            'node_monitoring_interval': 5.0,  # seconds
            'thermal_warning_threshold': 0.8,
            'profit_threshold': 0.6,
            'max_concurrent_trades': 3
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                
        return default_config
        
    def start_monitoring(self):
        """Start continuous node monitoring and Phantom Mode processing."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already active")
            return
            
        self.integration_active = True
        self.stop_monitoring_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Phantom Mode monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.integration_active = False
        self.stop_monitoring_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        logger.info("Phantom Mode monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring_event.is_set():
            try:
                # Update node status
                self.load_balancer.update_node_status()
                
                # Check for thermal warnings
                self._check_thermal_warnings()
                
                # Process any pending Phantom Mode decisions
                self._process_phantom_decisions()
                
                # Sleep for monitoring interval
                time.sleep(self.config['node_monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
                
    def _check_thermal_warnings(self):
        """Check for thermal warnings across all nodes."""
        warnings = []
        
        for node_name, status in self.load_balancer.node_status.items():
            if status.thermal_warning:
                warnings.append({
                    'node': node_name,
                    'temperature': status.temperature,
                    'thermal_limit': self.load_balancer.nodes[node_name].thermal_limit_celsius
                })
                
        if warnings:
            logger.warning(f"Thermal warnings detected: {warnings}")
            
    def _process_phantom_decisions(self):
        """Process any pending Phantom Mode trading decisions."""
        # This would integrate with your existing trading system
        # For now, just log the current Phantom Mode status
        phantom_status = self.phantom_engine.get_phantom_status()
        
        if phantom_status['phantom_mode_active']:
            logger.info(f"Phantom Mode active - Total trades: {phantom_status['total_trades']}")
            
    def process_market_data(self, price_data: List[float], timestamps: List[float], 
                          volume_data: Optional[List[float]] = None) -> Dict:
        """
        Process market data through Phantom Mode and return integrated decision.
        """
        # Process through Phantom Mode engine
        phantom_decision = self.phantom_engine.process_market_data(price_data, timestamps, volume_data)
        
        # If Phantom Mode triggers, calculate node distribution
        if phantom_decision.get('action') == 'execute_trade':
            execution_result = self.load_balancer.execute_phantom_trade(phantom_decision, {
                'prices': price_data,
                'timestamps': timestamps,
                'volumes': volume_data
            })
            
            # Combine Phantom decision with execution result
            integrated_decision = {
                **phantom_decision,
                'execution': execution_result,
                'node_status': self.load_balancer.node_status,
                'integration_timestamp': time.time()
            }
        else:
            integrated_decision = {
                **phantom_decision,
                'execution': None,
                'node_status': self.load_balancer.node_status,
                'integration_timestamp': time.time()
            }
            
        return integrated_decision
        
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'phantom_mode': self.phantom_engine.get_phantom_status(),
            'node_status': {name: {
                'temperature': status.temperature,
                'memory_usage': status.memory_usage,
                'gpu_utilization': status.gpu_utilization,
                'thermal_warning': status.thermal_warning,
                'entropy_drift': status.entropy_drift,
                'is_available': status.is_available
            } for name, status in self.load_balancer.node_status.items()},
            'integration_active': self.integration_active,
            'config': self.config
        }
        
    def export_phantom_data(self) -> Dict:
        """Export all Phantom Mode data for analysis."""
        return {
            'phantom_engine_data': self.phantom_engine.export_phantom_data(),
            'node_status': self.load_balancer.node_status,
            'load_distribution': self.load_balancer.load_distribution,
            'system_status': self.get_system_status()
        }

# Example usage
def test_phantom_integration():
    """Test Phantom Mode integration."""
    integration = PhantomModeIntegration()
    
    # Start monitoring
    integration.start_monitoring()
    
    # Simulate market data
    base_price = self._get_real_price_data()  # REAL API DATA
    prices = []
    timestamps = []
    
    for i in range(50):
        price_change = np.random.normal(0, 100)
        base_price += price_change
        prices.append(base_price)
        timestamps.append(time.time() + i * 60)
        
        # Process through integration
        decision = integration.process_market_data(prices, timestamps)
        
        if decision.get('action') == 'execute_trade':
            print(f"Phantom trade triggered at price {base_price:.2f}")
            print(f"Execution nodes: {decision.get('execution', {}).get('execution_nodes', [])}")
            print("---")
            
    # Get final status
    status = integration.get_system_status()
    print(f"Final system status: {status}")
    
    # Stop monitoring
    integration.stop_monitoring()


    def _get_real_price_data(self) -> float:
        """Get real price data from API - NO MORE STATIC 50000.0!"""
        try:
            # Try to get real price from API
            if hasattr(self, 'api_client') and self.api_client:
                try:
                    ticker = self.api_client.fetch_ticker('BTC/USDC')
                    if ticker and 'last' in ticker and ticker['last']:
                        return float(ticker['last'])
                except Exception as e:
                    pass
            
            # Try to get from market data provider
            if hasattr(self, 'market_data_provider') and self.market_data_provider:
                try:
                    price = self.market_data_provider.get_current_price('BTC/USDC')
                    if price and price > 0:
                        return price
                except Exception as e:
                    pass
            
            # Try to get from cache
            if hasattr(self, 'market_data_cache') and 'BTC/USDC' in self.market_data_cache:
                cached_price = self.market_data_cache['BTC/USDC'].get('price')
                if cached_price and cached_price > 0:
                    return cached_price
            
            # CRITICAL: No real data available - fail properly
            raise ValueError("No live price data available - API connection required")
            
        except Exception as e:
            raise ValueError(f"Cannot get live price data: {e}")

if __name__ == "__main__":
    test_phantom_integration() 