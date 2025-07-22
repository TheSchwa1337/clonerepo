#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Node Manager - Schwabot Distributed Real-time Context System
=======================================================================

Manages distributed nodes in the Schwabot system where any machine can become
the dedicated Flask node for real-time context ingestion and AI integration.

Features:
- Node election and management
- Distributed storage coordination
- Real-time context streaming
- Hardware-aware optimization
- Cross-machine communication
"""

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

import psutil
import requests

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """Node roles in the distributed system."""
    FLASK_NODE = "flask_node"
    WORKER_NODE = "worker_node"
    STORAGE_NODE = "storage_node"
    AI_NODE = "ai_node"

@dataclass
class NodeInfo:
    """Information about a node in the distributed system."""
    node_id: str
    hostname: str
    ip_address: str
    role: NodeRole
    capabilities: Dict[str, Any]
    resources: Dict[str, Any]
    last_heartbeat: float
    is_active: bool = True

@dataclass
class DistributedConfig:
    """Configuration for the distributed system."""
    election_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    context_update_interval: float = 30.0  # 30-second latency
    storage_sync_interval: float = 60.0
    max_nodes: int = 10
    shared_storage_path: str = "./shared_storage"
    flask_node_port: int = 5000

class DistributedNodeManager:
    """Manages distributed nodes in the Schwabot system."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.node_id = str(uuid.uuid4())
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        self.nodes: Dict[str, NodeInfo] = {}
        self.current_flask_node: Optional[str] = None
        self.is_flask_node = False
        self.election_in_progress = False
        self.last_election_time = 0
        
        # Initialize shared storage
        self.shared_storage_path = Path(config.shared_storage_path)
        self.shared_storage_path.mkdir(exist_ok=True)
        
        # Context streaming
        self.context_queue = asyncio.Queue()
        self.context_subscribers: Set[str] = set()
        
        logger.info(f"Initialized DistributedNodeManager on {self.hostname} ({self.ip_address})")
    
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def start(self):
        """Start the distributed node manager."""
        logger.info("Starting DistributedNodeManager...")
        
        # Register this node
        await self._register_node()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._election_monitor())
        asyncio.create_task(self._context_streaming_loop())
        asyncio.create_task(self._storage_sync_loop())
        
        logger.info("DistributedNodeManager started successfully")
    
    async def _register_node(self):
        """Register this node in the distributed system."""
        node_info = NodeInfo(
            node_id=self.node_id,
            hostname=self.hostname,
            ip_address=self.ip_address,
            role=NodeRole.WORKER_NODE,
            capabilities=self._get_capabilities(),
            resources=self._get_resources(),
            last_heartbeat=time.time()
        )
        
        self.nodes[self.node_id] = node_info
        await self._save_node_registry()
        
        logger.info(f"Registered node {self.node_id} as {node_info.role.value}")
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this node."""
        return {
            "can_be_flask_node": True,
            "has_gpu": self._has_gpu(),
            "has_high_memory": self._has_high_memory(),
            "has_fast_storage": self._has_fast_storage(),
            "ai_models": self._get_ai_models(),
            "trading_systems": self._get_trading_systems()
        }
    
    def _get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }
    
    def _has_gpu(self) -> bool:
        """Check if this node has GPU capabilities."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _has_high_memory(self) -> bool:
        """Check if this node has high memory."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        return memory_gb >= 8  # 8GB or more
    
    def _has_fast_storage(self) -> bool:
        """Check if this node has fast storage."""
        # Simple heuristic - could be enhanced
        return True
    
    def _get_ai_models(self) -> List[str]:
        """Get available AI models on this node."""
        models = []
        if os.path.exists("koboldcpp"):
            models.append("koboldcpp")
        if os.path.exists("AOI_Base_Files_Schwabot"):
            models.append("schwabot_ai")
        return models
    
    def _get_trading_systems(self) -> List[str]:
        """Get available trading systems on this node."""
        systems = []
        if os.path.exists("core/high_volume_trading_manager.py"):
            systems.append("high_volume_trading")
        if os.path.exists("core/trading_engine.py"):
            systems.append("trading_engine")
        return systems
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain node status."""
        while True:
            try:
                # Update our own heartbeat
                if self.node_id in self.nodes:
                    self.nodes[self.node_id].last_heartbeat = time.time()
                    self.nodes[self.node_id].resources = self._get_resources()
                
                # Check for stale nodes
                current_time = time.time()
                stale_nodes = []
                for node_id, node_info in self.nodes.items():
                    if current_time - node_info.last_heartbeat > self.config.election_timeout * 2:
                        stale_nodes.append(node_id)
                
                for node_id in stale_nodes:
                    del self.nodes[node_id]
                    logger.warning(f"Removed stale node {node_id}")
                
                await self._save_node_registry()
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _election_monitor(self):
        """Monitor and participate in Flask node elections."""
        while True:
            try:
                current_time = time.time()
                
                # Check if we need a new Flask node
                if (self.current_flask_node is None or 
                    self.current_flask_node not in self.nodes or
                    current_time - self.last_election_time > self.config.election_timeout):
                    
                    await self._run_election()
                
                await asyncio.sleep(self.config.election_timeout)
                
            except Exception as e:
                logger.error(f"Error in election monitor: {e}")
                await asyncio.sleep(self.config.election_timeout)
    
    async def _run_election(self):
        """Run a Flask node election."""
        if self.election_in_progress:
            return
        
        self.election_in_progress = True
        logger.info("Starting Flask node election...")
        
        try:
            # Find the best candidate for Flask node
            candidates = []
            for node_id, node_info in self.nodes.items():
                if (node_info.is_active and 
                    node_info.capabilities.get("can_be_flask_node", False)):
                    candidates.append((node_id, node_info))
            
            if not candidates:
                logger.warning("No suitable Flask node candidates found")
                return
            
            # Score candidates based on capabilities and resources
            scored_candidates = []
            for node_id, node_info in candidates:
                score = self._calculate_flask_node_score(node_info)
                scored_candidates.append((score, node_id, node_info))
            
            # Sort by score (highest first)
            scored_candidates.sort(reverse=True)
            
            # Select the best candidate
            best_score, best_node_id, best_node_info = scored_candidates[0]
            
            if best_node_id == self.node_id:
                await self._become_flask_node()
            else:
                self.current_flask_node = best_node_id
                self.is_flask_node = False
                logger.info(f"Selected {best_node_info.hostname} as Flask node")
            
            self.last_election_time = time.time()
            
        except Exception as e:
            logger.error(f"Error in election: {e}")
        finally:
            self.election_in_progress = False
    
    def _calculate_flask_node_score(self, node_info: NodeInfo) -> float:
        """Calculate a score for Flask node candidacy."""
        score = 0.0
        
        # Resource availability (lower usage = higher score)
        resources = node_info.resources
        score += (100 - resources.get("cpu_percent", 50)) * 0.3
        score += (100 - resources.get("memory_percent", 50)) * 0.3
        score += (100 - resources.get("disk_percent", 50)) * 0.2
        
        # Capabilities bonus
        capabilities = node_info.capabilities
        if capabilities.get("has_gpu", False):
            score += 50
        if capabilities.get("has_high_memory", False):
            score += 30
        if capabilities.get("has_fast_storage", False):
            score += 20
        
        # AI models bonus
        ai_models = capabilities.get("ai_models", [])
        score += len(ai_models) * 10
        
        return score
    
    async def _become_flask_node(self):
        """This node becomes the Flask node."""
        logger.info("This node is becoming the Flask node...")
        
        self.is_flask_node = True
        self.current_flask_node = self.node_id
        
        # Update node role
        if self.node_id in self.nodes:
            self.nodes[self.node_id].role = NodeRole.FLASK_NODE
        
        # Start Flask server
        await self._start_flask_server()
        
        logger.info("Successfully became Flask node")
    
    async def _start_flask_server(self):
        """Start the Flask server for this node."""
        try:
            # Import and start the Flask app
            from AOI_Base_Files_Schwabot.api.flask_app import create_app
            app = create_app()
            
            # Start Flask in a separate thread
            import threading
            def run_flask():
                app.run(
                    host='0.0.0.0',
                    port=self.config.flask_node_port,
                    debug=False,
                    use_reloader=False
                )
            
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
            logger.info(f"Flask server started on port {self.config.flask_node_port}")
            
        except Exception as e:
            logger.error(f"Failed to start Flask server: {e}")
    
    async def _context_streaming_loop(self):
        """Stream context data to subscribers."""
        while True:
            try:
                # Process context updates
                while not self.context_queue.empty():
                    context_data = await self.context_queue.get()
                    await self._broadcast_context(context_data)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in context streaming: {e}")
                await asyncio.sleep(1)
    
    async def _broadcast_context(self, context_data: Dict[str, Any]):
        """Broadcast context data to all subscribers."""
        if not self.context_subscribers:
            return
        
        # Save to shared storage
        await self._save_context_data(context_data)
        
        # Broadcast to subscribers (in a real implementation, this would use WebSockets)
        logger.debug(f"Broadcasting context to {len(self.context_subscribers)} subscribers")
    
    async def _save_context_data(self, context_data: Dict[str, Any]):
        """Save context data to shared storage."""
        try:
            timestamp = context_data.get("timestamp", time.time())
            filename = f"context_{timestamp}.json"
            filepath = self.shared_storage_path / "context" / filename
            
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(context_data, f, indent=2)
            
            # Keep only recent context files (last 1000)
            await self._cleanup_old_context_files()
            
        except Exception as e:
            logger.error(f"Error saving context data: {e}")
    
    async def _cleanup_old_context_files(self):
        """Clean up old context files to prevent storage bloat."""
        try:
            context_dir = self.shared_storage_path / "context"
            if not context_dir.exists():
                return
            
            files = list(context_dir.glob("context_*.json"))
            if len(files) > 1000:
                # Sort by modification time and remove oldest
                files.sort(key=lambda f: f.stat().st_mtime)
                for old_file in files[:-1000]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.error(f"Error cleaning up context files: {e}")
    
    async def _storage_sync_loop(self):
        """Synchronize storage across nodes."""
        while True:
            try:
                await self._sync_storage()
                await asyncio.sleep(self.config.storage_sync_interval)
                
            except Exception as e:
                logger.error(f"Error in storage sync: {e}")
                await asyncio.sleep(self.config.storage_sync_interval)
    
    async def _sync_storage(self):
        """Synchronize storage with other nodes."""
        # In a real implementation, this would sync files across nodes
        # For now, just ensure our local storage is organized
        pass
    
    async def _save_node_registry(self):
        """Save the node registry to shared storage."""
        try:
            registry_file = self.shared_storage_path / "node_registry.json"
            
            # Convert to serializable format
            registry_data = {}
            for node_id, node_info in self.nodes.items():
                registry_data[node_id] = {
                    "node_id": node_info.node_id,
                    "hostname": node_info.hostname,
                    "ip_address": node_info.ip_address,
                    "role": node_info.role.value,
                    "capabilities": node_info.capabilities,
                    "resources": node_info.resources,
                    "last_heartbeat": node_info.last_heartbeat,
                    "is_active": node_info.is_active
                }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving node registry: {e}")
    
    async def add_context_data(self, data_type: str, data: Any, source: str = None):
        """Add context data to the streaming queue."""
        context_data = {
            "timestamp": time.time(),
            "data_type": data_type,
            "data": data,
            "source": source or self.node_id,
            "node_id": self.node_id
        }
        
        await self.context_queue.put(context_data)
        logger.debug(f"Added context data: {data_type} from {source}")
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get the current status of all nodes."""
        return {
            "current_flask_node": self.current_flask_node,
            "is_flask_node": self.is_flask_node,
            "total_nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.is_active),
            "nodes": {
                node_id: {
                    "hostname": node_info.hostname,
                    "role": node_info.role.value,
                    "is_active": node_info.is_active,
                    "last_heartbeat": node_info.last_heartbeat
                }
                for node_id, node_info in self.nodes.items()
            }
        }
    
    async def stop(self):
        """Stop the distributed node manager."""
        logger.info("Stopping DistributedNodeManager...")
        
        # Mark this node as inactive
        if self.node_id in self.nodes:
            self.nodes[self.node_id].is_active = False
            await self._save_node_registry()
        
        logger.info("DistributedNodeManager stopped")

# Global instance
_distributed_manager: Optional[DistributedNodeManager] = None

def get_distributed_manager() -> DistributedNodeManager:
    """Get the global distributed manager instance."""
    global _distributed_manager
    if _distributed_manager is None:
        config = DistributedConfig()
        _distributed_manager = DistributedNodeManager(config)
    return _distributed_manager

async def start_distributed_system():
    """Start the distributed system."""
    manager = get_distributed_manager()
    await manager.start()
    return manager

if __name__ == "__main__":
    # Test the distributed node manager
    async def test():
        manager = get_distributed_manager()
        await manager.start()
        
        # Add some test context data
        await manager.add_context_data("test", {"message": "Hello from distributed system"})
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await manager.stop()
    
    asyncio.run(test())
