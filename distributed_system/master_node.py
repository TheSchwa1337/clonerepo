#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 MASTER NODE - DISTRIBUTED TRADING SYSTEM BRAIN
=================================================

Master node that serves as the central brain and bootstrap for the entire
distributed Schwabot trading system.

Features:
- Flask API Server with all endpoints
- Discord bot integration
- Multi-node coordination and task routing
- Shared memory and registry management
- Auto-detection and configuration
- Real-time monitoring and control
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Flask and API imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Discord integration
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# System imports
import psutil
import requests
import yaml
from dotenv import load_dotenv

# Schwabot imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from core.unified_advanced_calculations import unified_advanced_calculations
    from schwabot_trading_engine import SchwabotTradingEngine
    from schwabot_monitoring_system import SchwabotMonitoringSystem
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
        logging.FileHandler('master_node.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a connected node."""
    node_id: str
    host: str
    port: int
    node_type: str  # 'gpu', 'cpu', 'pi'
    capabilities: List[str]
    status: str  # 'online', 'offline', 'busy'
    last_heartbeat: float
    performance_metrics: Dict[str, float]
    assigned_tasks: List[str]

@dataclass
class SystemState:
    """Current system state."""
    master_status: str
    connected_nodes: int
    active_trades: int
    total_profit: float
    system_health: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    last_update: float

class MasterNode:
    """
    Master node that coordinates the entire distributed trading system.
    """
    
    def __init__(self, config_path: str = "config/master_config.yaml"):
        """Initialize the master node."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # System state
        self.nodes: Dict[str, NodeInfo] = {}
        self.system_state = SystemState(
            master_status="initializing",
            connected_nodes=0,
            active_trades=0,
            total_profit=0.0,
            system_health=100.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            gpu_usage=0.0,
            last_update=time.time()
        )
        
        # Trading system
        self.trading_engine = None
        self.monitoring_system = None
        
        # Discord bot
        self.discord_bot = None
        
        # Threading
        self.running = False
        self.heartbeat_thread = None
        self.monitoring_thread = None
        
        # Shared data directory
        self.shared_data_dir = Path("shared_data")
        self.shared_data_dir.mkdir(exist_ok=True)
        
        logger.info("🧠 Master Node initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default configuration
                config = {
                    'master': {
                        'host': '0.0.0.0',
                        'port': 5000,
                        'debug': False
                    },
                    'discord': {
                        'enabled': True,
                        'token': os.getenv('DISCORD_TOKEN', ''),
                        'channel_id': os.getenv('DISCORD_CHANNEL_ID', '')
                    },
                    'trading': {
                        'observation_mode': True,
                        'auto_trade': False,
                        'backtest_duration': 72,  # hours
                        'min_performance': 1.5  # percent
                    },
                    'nodes': {
                        'heartbeat_interval': 30,
                        'timeout': 120
                    }
                }
                
                # Save default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                return config
                
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def setup_flask_routes(self):
        """Setup Flask API routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard."""
            return render_template('dashboard.html', state=self.system_state)
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status."""
            return jsonify(asdict(self.system_state))
        
        @self.app.route('/api/nodes')
        def get_nodes():
            """Get connected nodes."""
            return jsonify([asdict(node) for node in self.nodes.values()])
        
        @self.app.route('/api/nodes/register', methods=['POST'])
        def register_node():
            """Register a new node."""
            data = request.json
            node_id = data.get('node_id')
            
            if node_id in self.nodes:
                return jsonify({'error': 'Node already registered'}), 400
            
            node_info = NodeInfo(
                node_id=node_id,
                host=data.get('host'),
                port=data.get('port'),
                node_type=data.get('node_type', 'cpu'),
                capabilities=data.get('capabilities', []),
                status='online',
                last_heartbeat=time.time(),
                performance_metrics=data.get('performance_metrics', {}),
                assigned_tasks=[]
            )
            
            self.nodes[node_id] = node_info
            self.system_state.connected_nodes = len(self.nodes)
            
            logger.info(f"Node registered: {node_id}")
            return jsonify({'status': 'registered', 'node_id': node_id})
        
        @self.app.route('/api/nodes/<node_id>/heartbeat', methods=['POST'])
        def node_heartbeat(node_id):
            """Update node heartbeat."""
            if node_id not in self.nodes:
                return jsonify({'error': 'Node not found'}), 404
            
            data = request.json
            self.nodes[node_id].last_heartbeat = time.time()
            self.nodes[node_id].status = data.get('status', 'online')
            self.nodes[node_id].performance_metrics = data.get('performance_metrics', {})
            
            return jsonify({'status': 'updated'})
        
        @self.app.route('/api/tasks/assign', methods=['POST'])
        def assign_task():
            """Assign task to available node."""
            data = request.json
            task_type = data.get('task_type')
            task_data = data.get('task_data', {})
            
            # Find best available node
            best_node = self._find_best_node(task_type)
            if not best_node:
                return jsonify({'error': 'No available nodes'}), 503
            
            # Create task
            task_id = f"task_{int(time.time())}_{task_type}"
            task = {
                'task_id': task_id,
                'task_type': task_type,
                'task_data': task_data,
                'assigned_to': best_node.node_id,
                'status': 'assigned',
                'created_at': time.time()
            }
            
            # Assign to node
            best_node.assigned_tasks.append(task_id)
            
            # Save task to shared data
            self._save_task(task)
            
            logger.info(f"Task assigned: {task_id} -> {best_node.node_id}")
            return jsonify(task)
        
        @self.app.route('/api/trading/status')
        def trading_status():
            """Get trading system status."""
            if not self.trading_engine:
                return jsonify({'error': 'Trading engine not initialized'}), 503
            
            return jsonify({
                'observation_mode': self.config['trading']['observation_mode'],
                'auto_trade': self.config['trading']['auto_trade'],
                'active_trades': self.system_state.active_trades,
                'total_profit': self.system_state.total_profit
            })
        
        @self.app.route('/api/trading/control', methods=['POST'])
        def trading_control():
            """Control trading system."""
            data = request.json
            action = data.get('action')
            
            if action == 'start_observation':
                self.config['trading']['observation_mode'] = True
                self.config['trading']['auto_trade'] = False
                logger.info("Trading observation mode started")
                
            elif action == 'start_trading':
                if self._can_start_trading():
                    self.config['trading']['observation_mode'] = False
                    self.config['trading']['auto_trade'] = True
                    logger.info("Live trading started")
                else:
                    return jsonify({'error': 'Cannot start trading - requirements not met'}), 400
                    
            elif action == 'stop_trading':
                self.config['trading']['auto_trade'] = False
                logger.info("Trading stopped")
            
            # Save config
            self._save_config()
            return jsonify({'status': 'success', 'action': action})
        
        @self.app.route('/api/ai/consult', methods=['POST'])
        def ai_consult():
            """AI consultation endpoint."""
            data = request.json
            query = data.get('query')
            context = data.get('context', {})
            
            # Get system context
            system_context = {
                'nodes': len(self.nodes),
                'trades': self.system_state.active_trades,
                'profit': self.system_state.total_profit,
                'health': self.system_state.system_health
            }
            
            # Generate AI response (placeholder)
            response = self._generate_ai_response(query, {**context, **system_context})
            
            return jsonify({
                'query': query,
                'response': response,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/system/health')
        def system_health():
            """Get detailed system health."""
            return jsonify({
                'master': {
                    'status': self.system_state.master_status,
                    'memory_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent(),
                    'uptime': time.time() - self.system_state.last_update
                },
                'nodes': {
                    'total': len(self.nodes),
                    'online': len([n for n in self.nodes.values() if n.status == 'online']),
                    'busy': len([n for n in self.nodes.values() if n.status == 'busy'])
                },
                'trading': {
                    'observation_mode': self.config['trading']['observation_mode'],
                    'auto_trade': self.config['trading']['auto_trade'],
                    'active_trades': self.system_state.active_trades
                }
            })
    
    def _find_best_node(self, task_type: str) -> Optional[NodeInfo]:
        """Find the best available node for a task."""
        available_nodes = [n for n in self.nodes.values() if n.status == 'online']
        
        if not available_nodes:
            return None
        
        # Simple selection logic
        if task_type in ['fractal_analysis', 'quantum_calculation', 'gpu_intensive']:
            # Prefer GPU nodes
            gpu_nodes = [n for n in available_nodes if 'gpu' in n.capabilities]
            if gpu_nodes:
                return min(gpu_nodes, key=lambda x: len(x.assigned_tasks))
        
        # Default to least busy node
        return min(available_nodes, key=lambda x: len(x.assigned_tasks))
    
    def _save_task(self, task: Dict[str, Any]):
        """Save task to shared data directory."""
        task_file = self.shared_data_dir / f"tasks/{task['task_id']}.json"
        task_file.parent.mkdir(exist_ok=True)
        
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)
    
    def _save_config(self):
        """Save current configuration."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _can_start_trading(self) -> bool:
        """Check if system can start live trading."""
        # Check observation time
        if self.config['trading']['observation_mode']:
            # Would need to track observation start time
            pass
        
        # Check performance
        if self.system_state.total_profit < self.config['trading']['min_performance']:
            return False
        
        # Check system health
        if self.system_state.system_health < 80.0:
            return False
        
        return True
    
    def _generate_ai_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate AI response (placeholder)."""
        # This would integrate with actual AI models
        return f"AI Response to '{query}' with context: {context}"
    
    def setup_discord_bot(self):
        """Setup Discord bot integration."""
        if not DISCORD_AVAILABLE or not self.config['discord']['enabled']:
            logger.warning("Discord integration not available")
            return
        
        try:
            intents = discord.Intents.default()
            intents.message_content = True
            
            self.discord_bot = commands.Bot(command_prefix='/', intents=intents)
            
            @self.discord_bot.event
            async def on_ready():
                logger.info(f"Discord bot logged in as {self.discord_bot.user}")
            
            @self.discord_bot.command(name='status')
            async def status(ctx):
                """Get system status."""
                embed = discord.Embed(title="🧠 Schwabot Master Node Status", color=0x00ff00)
                embed.add_field(name="Master Status", value=self.system_state.master_status, inline=True)
                embed.add_field(name="Connected Nodes", value=self.system_state.connected_nodes, inline=True)
                embed.add_field(name="Active Trades", value=self.system_state.active_trades, inline=True)
                embed.add_field(name="Total Profit", value=f"${self.system_state.total_profit:.2f}", inline=True)
                embed.add_field(name="System Health", value=f"{self.system_state.system_health:.1f}%", inline=True)
                embed.add_field(name="Memory Usage", value=f"{self.system_state.memory_usage:.1f}%", inline=True)
                
                await ctx.send(embed=embed)
            
            @self.discord_bot.command(name='nodes')
            async def nodes(ctx):
                """List connected nodes."""
                if not self.nodes:
                    await ctx.send("No nodes connected")
                    return
                
                embed = discord.Embed(title="🖥️ Connected Nodes", color=0x0099ff)
                for node_id, node in self.nodes.items():
                    status_emoji = "🟢" if node.status == "online" else "🔴"
                    embed.add_field(
                        name=f"{status_emoji} {node_id}",
                        value=f"Type: {node.node_type}\nStatus: {node.status}\nTasks: {len(node.assigned_tasks)}",
                        inline=True
                    )
                
                await ctx.send(embed=embed)
            
            @self.discord_bot.command(name='trading')
            async def trading(ctx):
                """Get trading status."""
                embed = discord.Embed(title="💰 Trading Status", color=0xff9900)
                embed.add_field(name="Observation Mode", value=self.config['trading']['observation_mode'], inline=True)
                embed.add_field(name="Auto Trade", value=self.config['trading']['auto_trade'], inline=True)
                embed.add_field(name="Active Trades", value=self.system_state.active_trades, inline=True)
                embed.add_field(name="Total Profit", value=f"${self.system_state.total_profit:.2f}", inline=True)
                
                await ctx.send(embed=embed)
            
            @self.discord_bot.command(name='ai')
            async def ai_consult(ctx, *, query):
                """Consult AI with system context."""
                # Send to Flask API
                response = requests.post(
                    f"http://localhost:{self.config['master']['port']}/api/ai/consult",
                    json={'query': query, 'context': {'source': 'discord'}}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embed = discord.Embed(title="🤖 AI Consultation", color=0x9932cc)
                    embed.add_field(name="Query", value=query, inline=False)
                    embed.add_field(name="Response", value=data['response'], inline=False)
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("❌ Failed to get AI response")
            
            # Start Discord bot in thread
            def run_discord():
                self.discord_bot.run(self.config['discord']['token'])
            
            discord_thread = threading.Thread(target=run_discord, daemon=True)
            discord_thread.start()
            
            logger.info("Discord bot started")
            
        except Exception as e:
            logger.error(f"Failed to setup Discord bot: {e}")
    
    def start_monitoring(self):
        """Start system monitoring."""
        def monitor_loop():
            while self.running:
                try:
                    # Update system metrics
                    self.system_state.memory_usage = psutil.virtual_memory().percent
                    self.system_state.cpu_usage = psutil.cpu_percent()
                    self.system_state.last_update = time.time()
                    
                    # Check node timeouts
                    current_time = time.time()
                    for node_id, node in list(self.nodes.items()):
                        if current_time - node.last_heartbeat > self.config['nodes']['timeout']:
                            node.status = 'offline'
                            logger.warning(f"Node {node_id} timed out")
                    
                    # Update system health
                    online_nodes = len([n for n in self.nodes.values() if n.status == 'online'])
                    total_nodes = len(self.nodes)
                    if total_nodes > 0:
                        self.system_state.system_health = (online_nodes / total_nodes) * 100
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def start_heartbeat(self):
        """Start heartbeat broadcasting."""
        def heartbeat_loop():
            while self.running:
                try:
                    # Broadcast heartbeat to all nodes
                    for node in self.nodes.values():
                        if node.status == 'online':
                            try:
                                requests.post(
                                    f"http://{node.host}:{node.port}/heartbeat",
                                    json={'master_heartbeat': time.time()},
                                    timeout=5
                                )
                            except:
                                pass  # Node might be offline
                    
                    time.sleep(self.config['nodes']['heartbeat_interval'])
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(self.config['nodes']['heartbeat_interval'])
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Heartbeat broadcasting started")
    
    def initialize_trading_system(self):
        """Initialize trading engine and monitoring."""
        if not SCHWABOT_AVAILABLE:
            logger.warning("Schwabot modules not available - trading system disabled")
            return
        
        try:
            # Initialize trading engine
            self.trading_engine = SchwabotTradingEngine()
            self.monitoring_system = SchwabotMonitoringSystem()
            
            logger.info("Trading system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
    
    def start(self):
        """Start the master node."""
        logger.info("🚀 Starting Master Node...")
        
        # Setup Flask routes
        self.setup_flask_routes()
        
        # Initialize trading system
        self.initialize_trading_system()
        
        # Setup Discord bot
        self.setup_discord_bot()
        
        # Start monitoring and heartbeat
        self.running = True
        self.start_monitoring()
        self.start_heartbeat()
        
        # Update status
        self.system_state.master_status = "running"
        
        logger.info(f"🧠 Master Node running on {self.config['master']['host']}:{self.config['master']['port']}")
        
        # Start Flask app
        self.socketio.run(
            self.app,
            host=self.config['master']['host'],
            port=self.config['master']['port'],
            debug=self.config['master']['debug']
        )
    
    def stop(self):
        """Stop the master node."""
        logger.info("🛑 Stopping Master Node...")
        self.running = False
        self.system_state.master_status = "stopped"

def main():
    """Main entry point."""
    master = MasterNode()
    
    try:
        master.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        master.stop()

if __name__ == "__main__":
    main() 