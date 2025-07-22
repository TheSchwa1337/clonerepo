#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ WORKER NODE - DISTRIBUTED TRADING SYSTEM EXECUTOR
====================================================

Worker node that runs on slave machines (GPU/CPU nodes) to execute tasks
assigned by the master node.

Features:
- Auto-detection of hardware capabilities (GPU/CPU)
- Task execution and result reporting
- Heartbeat communication with master
- Performance monitoring
- Automatic registration with master node
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import platform
import socket
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# System imports
import psutil
import requests
import yaml
from dotenv import load_dotenv

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Schwabot imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from core.unified_advanced_calculations import unified_advanced_calculations
    SCHWABOT_AVAILABLE = True
except ImportError:
    SCHWABOT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_node.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """Hardware capabilities information."""
    cpu_count: int
    cpu_frequency: float
    memory_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    platform: str
    architecture: str

@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    status: str  # 'completed', 'failed', 'timeout'
    result_data: Dict[str, Any]
    execution_time: float
    error_message: str
    timestamp: float

class WorkerNode:
    """
    Worker node that executes tasks assigned by the master node.
    """
    
    def __init__(self, config_path: str = "config/worker_config.yaml"):
        """Initialize the worker node."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Node identification
        self.node_id = self._generate_node_id()
        self.hardware_info = self._detect_hardware()
        self.node_type = self._determine_node_type()
        self.capabilities = self._determine_capabilities()
        
        # Task management
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[TaskResult] = []
        self.task_queue = []
        
        # Communication
        self.master_host = self.config.get('master_host', 'localhost')
        self.master_port = self.config.get('master_port', 5000)
        self.worker_port = self.config.get('worker_port', 5001)
        
        # Performance monitoring
        self.performance_metrics = {}
        
        # Threading
        self.running = False
        self.heartbeat_thread = None
        self.task_thread = None
        self.monitoring_thread = None
        
        # Shared data directory
        self.shared_data_dir = Path("shared_data")
        self.shared_data_dir.mkdir(exist_ok=True)
        
        logger.info(f"🖥️ Worker Node {self.node_id} initialized")
        logger.info(f"Hardware: {self.hardware_info}")
        logger.info(f"Capabilities: {self.capabilities}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default configuration
                config = {
                    'worker_port': 5001,
                    'master_host': os.getenv('MASTER_HOST', 'localhost'),
                    'master_port': int(os.getenv('MASTER_PORT', 5000)),
                    'heartbeat_interval': 30,
                    'task_poll_interval': 10,
                    'max_concurrent_tasks': 4,
                    'task_timeout': 300,  # 5 minutes
                    'auto_register': True
                }
                
                # Save default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                return config
                
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        hostname = socket.gethostname()
        return f"{hostname}_{platform.machine()}_{uuid.uuid4().hex[:8]}"
    
    def _detect_hardware(self) -> HardwareInfo:
        """Detect hardware capabilities."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current if cpu_freq else 0.0
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # GPU detection
            gpu_available = False
            gpu_name = "None"
            gpu_memory_gb = 0.0
            
            # Try NVIDIA GPU detection
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        gpu_available = True
                        gpu_info = lines[0].split(', ')
                        gpu_name = gpu_info[0]
                        gpu_memory_gb = float(gpu_info[1]) / 1024  # Convert MB to GB
            except:
                pass
            
            # Try PyTorch GPU detection
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_available = True
                if gpu_name == "None":
                    gpu_name = torch.cuda.get_device_name(0)
                if gpu_memory_gb == 0.0:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return HardwareInfo(
                cpu_count=cpu_count,
                cpu_frequency=cpu_frequency,
                memory_gb=memory_gb,
                gpu_available=gpu_available,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                platform=platform.system(),
                architecture=platform.machine()
            )
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            return HardwareInfo(
                cpu_count=1,
                cpu_frequency=0.0,
                memory_gb=0.0,
                gpu_available=False,
                gpu_name="Unknown",
                gpu_memory_gb=0.0,
                platform=platform.system(),
                architecture=platform.machine()
            )
    
    def _determine_node_type(self) -> str:
        """Determine node type based on hardware."""
        if self.hardware_info.gpu_available:
            if self.hardware_info.gpu_memory_gb >= 8:
                return "gpu_high"
            elif self.hardware_info.gpu_memory_gb >= 4:
                return "gpu_medium"
            else:
                return "gpu_low"
        elif self.hardware_info.cpu_count >= 8:
            return "cpu_high"
        elif self.hardware_info.cpu_count >= 4:
            return "cpu_medium"
        else:
            return "cpu_low"
    
    def _determine_capabilities(self) -> List[str]:
        """Determine node capabilities."""
        capabilities = []
        
        # Basic capabilities
        capabilities.append("basic_computation")
        capabilities.append("data_processing")
        
        # CPU capabilities
        if self.hardware_info.cpu_count >= 4:
            capabilities.append("parallel_processing")
        if self.hardware_info.cpu_count >= 8:
            capabilities.append("high_performance_computing")
        
        # GPU capabilities
        if self.hardware_info.gpu_available:
            capabilities.append("gpu_computation")
            capabilities.append("tensor_operations")
            capabilities.append("quantum_calculations")
            
            if self.hardware_info.gpu_memory_gb >= 8:
                capabilities.append("large_model_training")
                capabilities.append("batch_processing")
        
        # Memory capabilities
        if self.hardware_info.memory_gb >= 16:
            capabilities.append("large_dataset_processing")
        
        # Platform-specific capabilities
        if self.hardware_info.platform == "Linux":
            capabilities.append("linux_optimized")
        elif self.hardware_info.platform == "Windows":
            capabilities.append("windows_compatible")
        
        # Schwabot capabilities
        if SCHWABOT_AVAILABLE:
            capabilities.append("schwabot_analysis")
            capabilities.append("advanced_calculations")
        
        return capabilities
    
    def setup_flask_routes(self):
        """Setup Flask API routes."""
        
        @self.app.route('/')
        def index():
            """Worker node status."""
            return jsonify({
                'node_id': self.node_id,
                'node_type': self.node_type,
                'capabilities': self.capabilities,
                'status': 'online',
                'current_tasks': len(self.current_tasks),
                'completed_tasks': len(self.completed_tasks)
            })
        
        @self.app.route('/heartbeat', methods=['POST'])
        def heartbeat():
            """Receive heartbeat from master."""
            data = request.json
            # Process heartbeat data if needed
            return jsonify({'status': 'received'})
        
        @self.app.route('/tasks/<task_id>/result', methods=['POST'])
        def submit_task_result(task_id):
            """Submit task result to master."""
            data = request.json
            
            if task_id in self.current_tasks:
                # Create task result
                result = TaskResult(
                    task_id=task_id,
                    status=data.get('status', 'completed'),
                    result_data=data.get('result_data', {}),
                    execution_time=data.get('execution_time', 0.0),
                    error_message=data.get('error_message', ''),
                    timestamp=time.time()
                )
                
                # Remove from current tasks
                del self.current_tasks[task_id]
                
                # Add to completed tasks
                self.completed_tasks.append(result)
                
                logger.info(f"Task {task_id} completed with status: {result.status}")
                
                return jsonify({'status': 'received'})
            else:
                return jsonify({'error': 'Task not found'}), 404
        
        @self.app.route('/tasks/<task_id>/status')
        def get_task_status(task_id):
            """Get task status."""
            if task_id in self.current_tasks:
                return jsonify(self.current_tasks[task_id])
            else:
                return jsonify({'error': 'Task not found'}), 404
        
        @self.app.route('/performance')
        def get_performance():
            """Get performance metrics."""
            return jsonify(self.performance_metrics)
    
    def register_with_master(self):
        """Register this worker node with the master."""
        try:
            registration_data = {
                'node_id': self.node_id,
                'host': socket.gethostbyname(socket.gethostname()),
                'port': self.worker_port,
                'node_type': self.node_type,
                'capabilities': self.capabilities,
                'performance_metrics': self.performance_metrics
            }
            
            response = requests.post(
                f"http://{self.master_host}:{self.master_port}/api/nodes/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully registered with master node")
                return True
            else:
                logger.error(f"Failed to register with master: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def send_heartbeat(self):
        """Send heartbeat to master node."""
        try:
            # Update performance metrics
            self._update_performance_metrics()
            
            heartbeat_data = {
                'status': 'online',
                'performance_metrics': self.performance_metrics,
                'current_tasks': len(self.current_tasks),
                'completed_tasks': len(self.completed_tasks)
            }
            
            response = requests.post(
                f"http://{self.master_host}:{self.master_port}/api/nodes/{self.node_id}/heartbeat",
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU usage (if available)
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            
            if self.hardware_info.gpu_available:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            gpu_info = lines[0].split(', ')
                            gpu_percent = float(gpu_info[0])
                            gpu_memory_used = float(gpu_info[1])
                            gpu_memory_total = float(gpu_info[2])
                            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                except:
                    pass
            
            # Task metrics
            task_metrics = {
                'current_tasks': len(self.current_tasks),
                'completed_tasks': len(self.completed_tasks),
                'queue_length': len(self.task_queue)
            }
            
            self.performance_metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'gpu_usage': gpu_percent,
                'gpu_memory_usage': gpu_memory_percent,
                'task_metrics': task_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def poll_for_tasks(self):
        """Poll master node for new tasks."""
        try:
            # Check if we can accept more tasks
            if len(self.current_tasks) >= self.config.get('max_concurrent_tasks', 4):
                return
            
            # Request task from master
            response = requests.post(
                f"http://{self.master_host}:{self.master_port}/api/tasks/assign",
                json={
                    'task_type': 'any',  # Accept any task type
                    'task_data': {
                        'worker_capabilities': self.capabilities,
                        'worker_type': self.node_type
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                task = response.json()
                self._execute_task(task)
            elif response.status_code != 503:  # 503 means no tasks available
                logger.warning(f"Task polling failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Task polling error: {e}")
    
    def _execute_task(self, task: Dict[str, Any]):
        """Execute a task."""
        task_id = task['task_id']
        task_type = task['task_type']
        task_data = task['task_data']
        
        logger.info(f"Executing task {task_id}: {task_type}")
        
        # Add to current tasks
        self.current_tasks[task_id] = {
            'task_type': task_type,
            'start_time': time.time(),
            'status': 'running'
        }
        
        # Execute task in thread
        def execute():
            try:
                start_time = time.time()
                result = self._run_task(task_type, task_data)
                execution_time = time.time() - start_time
                
                # Submit result
                self._submit_task_result(task_id, 'completed', result, execution_time)
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Task {task_id} failed: {e}")
                self._submit_task_result(task_id, 'failed', {}, execution_time, str(e))
        
        task_thread = threading.Thread(target=execute, daemon=True)
        task_thread.start()
    
    def _run_task(self, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific task type."""
        try:
            if task_type == 'fractal_analysis':
                return self._run_fractal_analysis(task_data)
            elif task_type == 'quantum_calculation':
                return self._run_quantum_calculation(task_data)
            elif task_type == 'waveform_processing':
                return self._run_waveform_processing(task_data)
            elif task_type == 'entropy_analysis':
                return self._run_entropy_analysis(task_data)
            elif task_type == 'temporal_analysis':
                return self._run_temporal_analysis(task_data)
            elif task_type == 'backtest':
                return self._run_backtest(task_data)
            else:
                return {'error': f'Unknown task type: {task_type}'}
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {'error': str(e)}
    
    def _run_fractal_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run fractal analysis task."""
        if not SCHWABOT_AVAILABLE:
            return {'error': 'Schwabot not available'}
        
        try:
            # Generate sample data if not provided
            data = task_data.get('data')
            if not data:
                import numpy as np
                data = np.random.randn(1000)
            
            # Run fractal analysis
            result = unified_advanced_calculations.comprehensive_analysis(
                data=data,
                analysis_types=['statistical']
            )
            
            return {
                'fractal_dimension': result.unified_features.get('fractal_dimension', 0.0),
                'hurst_exponent': result.unified_features.get('hurst_exponent', 0.0),
                'lyapunov_exponent': result.unified_features.get('lyapunov_exponent', 0.0),
                'complexity_score': result.unified_features.get('complexity_score', 0.0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_quantum_calculation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum calculation task."""
        if not SCHWABOT_AVAILABLE:
            return {'error': 'Schwabot not available'}
        
        try:
            # Generate sample data
            import numpy as np
            data = np.random.randn(1000)
            
            # Run quantum-inspired calculations
            result = unified_advanced_calculations.comprehensive_analysis(
                data=data,
                analysis_types=['entropy']
            )
            
            return {
                'shannon_entropy': result.unified_features.get('shannon_entropy', 0.0),
                'fisher_information': result.unified_features.get('fisher_information', 0.0),
                'quantum_coherence': result.unified_features.get('predictability_score', 0.0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_waveform_processing(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run waveform processing task."""
        if not SCHWABOT_AVAILABLE:
            return {'error': 'Schwabot not available'}
        
        try:
            # Generate sample data
            import numpy as np
            data = np.random.randn(1000)
            reference_data = np.random.randn(1000)
            
            # Run waveform analysis
            result = unified_advanced_calculations.comprehensive_analysis(
                data=data,
                reference_data=reference_data,
                analysis_types=['waveform']
            )
            
            return {
                'phase_synchronization': result.unified_features.get('phase_synchronization', 0.0),
                'dominant_frequency_count': result.unified_features.get('dominant_frequency_count', 0.0),
                'waveform_complexity': result.unified_features.get('complexity_score', 0.0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_entropy_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run entropy analysis task."""
        if not SCHWABOT_AVAILABLE:
            return {'error': 'Schwabot not available'}
        
        try:
            # Generate sample data
            import numpy as np
            data = np.random.randn(1000)
            
            # Run entropy analysis
            result = unified_advanced_calculations.comprehensive_analysis(
                data=data,
                analysis_types=['entropy']
            )
            
            return {
                'shannon_entropy': result.unified_features.get('shannon_entropy', 0.0),
                'renyi_entropy_alpha_2': result.unified_features.get('renyi_entropy_alpha_2', 0.0),
                'tsallis_entropy_q_2': result.unified_features.get('tsallis_entropy_q_2', 0.0),
                'fisher_information': result.unified_features.get('fisher_information', 0.0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_temporal_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run temporal analysis task."""
        if not SCHWABOT_AVAILABLE:
            return {'error': 'Schwabot not available'}
        
        try:
            # Generate sample data
            import numpy as np
            data = np.random.randn(1000)
            reference_data = np.random.randn(1000)
            
            # Run temporal analysis
            result = unified_advanced_calculations.comprehensive_analysis(
                data=data,
                reference_data=reference_data,
                analysis_types=['temporal']
            )
            
            return {
                'granger_causality': result.unified_features.get('granger_causality', 0.0),
                'dtw_distance': result.unified_features.get('dtw_distance', 0.0),
                'regime_count': result.unified_features.get('regime_count', 0.0),
                'temporal_stability': result.unified_features.get('stability_score', 0.0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_backtest(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest task."""
        try:
            # Simulate backtest
            import numpy as np
            
            # Generate simulated trading data
            days = task_data.get('days', 30)
            initial_capital = task_data.get('initial_capital', 10000)
            
            # Simulate price movements
            returns = np.random.normal(0.001, 0.02, days)  # Daily returns
            prices = np.cumprod(1 + returns)
            
            # Simulate trading strategy
            position = 0
            capital = initial_capital
            trades = []
            
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1] * 1.01:  # Buy signal
                    if position == 0:
                        position = capital / prices[i]
                        capital = 0
                        trades.append({'day': i, 'action': 'buy', 'price': prices[i]})
                elif prices[i] < prices[i-1] * 0.99:  # Sell signal
                    if position > 0:
                        capital = position * prices[i]
                        position = 0
                        trades.append({'day': i, 'action': 'sell', 'price': prices[i]})
            
            # Final capital
            if position > 0:
                capital = position * prices[-1]
            
            profit = capital - initial_capital
            profit_percent = (profit / initial_capital) * 100
            
            return {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'profit': profit,
                'profit_percent': profit_percent,
                'total_trades': len(trades),
                'trades': trades
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _submit_task_result(self, task_id: str, status: str, result_data: Dict[str, Any], 
                           execution_time: float, error_message: str = ""):
        """Submit task result to master node."""
        try:
            result = {
                'status': status,
                'result_data': result_data,
                'execution_time': execution_time,
                'error_message': error_message
            }
            
            response = requests.post(
                f"http://{self.master_host}:{self.master_port}/tasks/{task_id}/result",
                json=result,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Task {task_id} result submitted successfully")
            else:
                logger.error(f"Failed to submit task result: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Task result submission failed: {e}")
    
    def start_heartbeat_loop(self):
        """Start heartbeat loop."""
        def heartbeat_loop():
            while self.running:
                self.send_heartbeat()
                time.sleep(self.config.get('heartbeat_interval', 30))
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Heartbeat loop started")
    
    def start_task_polling(self):
        """Start task polling loop."""
        def polling_loop():
            while self.running:
                self.poll_for_tasks()
                time.sleep(self.config.get('task_poll_interval', 10))
        
        self.task_thread = threading.Thread(target=polling_loop, daemon=True)
        self.task_thread.start()
        logger.info("Task polling started")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        def monitoring_loop():
            while self.running:
                self._update_performance_metrics()
                time.sleep(10)  # Update every 10 seconds
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def start(self):
        """Start the worker node."""
        logger.info(f"🚀 Starting Worker Node {self.node_id}...")
        
        # Setup Flask routes
        self.setup_flask_routes()
        
        # Register with master
        if self.config.get('auto_register', True):
            if not self.register_with_master():
                logger.warning("Failed to register with master - continuing anyway")
        
        # Start background threads
        self.running = True
        self.start_heartbeat_loop()
        self.start_task_polling()
        self.start_monitoring()
        
        logger.info(f"🖥️ Worker Node running on port {self.worker_port}")
        
        # Start Flask app
        self.app.run(
            host='0.0.0.0',
            port=self.worker_port,
            debug=False
        )
    
    def stop(self):
        """Stop the worker node."""
        logger.info("🛑 Stopping Worker Node...")
        self.running = False

def main():
    """Main entry point."""
    worker = WorkerNode()
    
    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        worker.stop()

if __name__ == "__main__":
    main() 