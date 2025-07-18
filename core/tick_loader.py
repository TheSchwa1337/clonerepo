#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick Loader for Schwabot Trading System
=======================================

Intelligent tick loading and management system with hardware-optimized
performance and memory management. This system provides real-time tick
processing with priority-based queuing and intelligent memory allocation.

Features:
- Hardware-optimized tick processing and memory management
- Priority-based queuing system for different tick types
- Intelligent memory pool allocation based on system capabilities
- Real-time tick data processing and analysis
- Integration with existing mathematical framework
- Secure tick data handling with encryption
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import hashlib
import psutil

from .hardware_auto_detector import HardwareAutoDetector

logger = logging.getLogger(__name__)

# Import real implementations instead of stubs
try:
    from .hash_config_manager import HashConfigManager
    HASH_CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ HashConfigManager not available, using stub")
    HASH_CONFIG_AVAILABLE = False

try:
    from .alpha256_encryption import Alpha256Encryption
    ALPHA256_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Alpha256Encryption not available, using stub")
    ALPHA256_AVAILABLE = False

# Stub classes for missing components
if not HASH_CONFIG_AVAILABLE:
    class HashConfigManager:
        """Simple stub for HashConfigManager."""
        def __init__(self):
            self.config = {}
        
        def initialize(self):
            """Initialize the hash config manager."""
            pass
        
        def get_config(self, key: str, default: Any = None) -> Any:
            """Get configuration value."""
            return self.config.get(key, default)
        
        def set_config(self, key: str, value: Any):
            """Set configuration value."""
            self.config[key] = value

if not ALPHA256_AVAILABLE:
    class Alpha256Encryption:
        """Simple stub for Alpha256Encryption."""
        def __init__(self):
            pass
        
        def encrypt(self, data: str) -> str:
            """Encrypt data."""
            return data  # Simple pass-through for now
        
        def decrypt(self, data: str) -> str:
            """Decrypt data."""
            return data  # Simple pass-through for now

class TickPriority(Enum):
    """Tick processing priority levels."""
    CRITICAL = "critical"      # Real-time execution signals
    HIGH = "high"             # Short-term strategy signals
    MEDIUM = "medium"         # Pattern recognition
    LOW = "low"               # Historical analysis
    BACKGROUND = "background" # Deep analysis

@dataclass
class TickData:
    """Tick data structure with metadata."""
    timestamp: float
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    priority: TickPriority = TickPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: str = ""

class TickLoader:
    """Intelligent tick loading and management system."""
    
    def __init__(self, config_path: str = "config/tick_loader_config.json"):
        """Initialize tick loader with hardware auto-detection."""
        self.config_path = Path(config_path)
        self.hardware_detector = HardwareAutoDetector()
        self.hash_config = HashConfigManager()
        self.alpha256 = Alpha256Encryption()
        
        # Hardware-aware configuration
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # Tick processing queues
        self.tick_queues: Dict[TickPriority, deque] = {
            priority: deque(maxlen=self._get_queue_size(priority))
            for priority in TickPriority
        }
        
        # Memory management
        self.memory_pools: Dict[str, List[TickData]] = {
            "high_frequency": [],
            "pattern_recognition": [],
            "deep_analysis": []
        }
        
        # Performance tracking
        self.stats = {
            "ticks_processed": 0,
            "ticks_dropped": 0,
            "memory_usage_mb": 0.0,
            "processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Initialize with hardware detection
        self._initialize_with_hardware_detection()
    
    def _initialize_with_hardware_detection(self):
        """Initialize tick loader using hardware auto-detection."""
        try:
            logger.info("🔍 Initializing tick loader with hardware auto-detection...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            self.auto_detected = True
            
            # Initialize hash configuration
            self.hash_config.initialize()
            
            # Configure memory pools based on hardware
            self._configure_memory_pools()
            
            # Load or create configuration
            self._load_configuration()
            
            logger.info(f"✅ Tick loader initialized for {self.system_info.platform}")
            logger.info(f"   Memory: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
        except Exception as e:
            logger.error(f"❌ Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _configure_memory_pools(self):
        """Configure memory pools based on hardware capabilities."""
        if not self.memory_config:
            return
            
        # Configure pool sizes based on hardware tier
        for pool_name, pool_config in self.memory_config.memory_pools.items():
            max_size = pool_config['size_mb'] * 1024 * 1024  # Convert to bytes
            max_ticks = int(max_size / 1024)  # Approximate tick size
            
            # Set pool limits
            if pool_name in self.memory_pools:
                # Use deque with hardware-aware limits
                self.memory_pools[pool_name] = deque(maxlen=max_ticks)
    
    def _get_queue_size(self, priority: TickPriority) -> int:
        """Get queue size based on priority and hardware capabilities."""
        if not self.memory_config:
            return 1000  # Default fallback
            
        base_sizes = {
            TickPriority.CRITICAL: 100,
            TickPriority.HIGH: 500,
            TickPriority.MEDIUM: 1000,
            TickPriority.LOW: 2000,
            TickPriority.BACKGROUND: 5000
        }
        
        base_size = base_sizes[priority]
        
        # Scale based on memory tier
        memory_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "ultra": 4.0
        }.get(self.system_info.ram_tier.value, 1.0)
        
        return int(base_size * memory_multiplier)
    
    def _load_configuration(self):
        """Load or create tick loader configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Loaded existing tick loader configuration")
            else:
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info("✅ Created new tick loader configuration")
                
            # Apply configuration
            self._apply_configuration(config)
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self._apply_configuration(self._create_default_config())
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration based on hardware."""
        return {
            "version": "1.0.0",
            "hardware_auto_detected": self.auto_detected,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "tick_processing": {
                "batch_size": 100,
                "max_queue_size": 10000,
                "enable_compression": True,
                "enable_encryption": True,
                "enable_caching": True,
                "cache_duration_minutes": 30
            },
            "memory_management": {
                "max_memory_usage_mb": self.system_info.ram_gb * 100 if self.system_info else 800,
                "cleanup_interval_seconds": 60,
                "enable_auto_cleanup": True,
                "enable_memory_monitoring": True
            },
            "performance": {
                "enable_parallel_processing": True,
                "max_worker_threads": min(self.system_info.cpu_cores, 8) if self.system_info else 4,
                "enable_performance_tracking": True,
                "enable_health_monitoring": True
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        self.config = config
        
        # Apply tick processing settings
        tick_settings = config["tick_processing"]
        self.batch_size = tick_settings["batch_size"]
        self.max_queue_size = tick_settings["max_queue_size"]
        self.enable_compression = tick_settings["enable_compression"]
        self.enable_encryption = tick_settings["enable_encryption"]
        self.enable_caching = tick_settings["enable_caching"]
        self.cache_duration_minutes = tick_settings["cache_duration_minutes"]
        
        # Apply memory management settings
        memory_settings = config["memory_management"]
        self.max_memory_usage_mb = memory_settings["max_memory_usage_mb"]
        self.cleanup_interval_seconds = memory_settings["cleanup_interval_seconds"]
        self.enable_auto_cleanup = memory_settings["enable_auto_cleanup"]
        self.enable_memory_monitoring = memory_settings["enable_memory_monitoring"]
        
        # Apply performance settings
        performance_settings = config["performance"]
        self.enable_parallel_processing = performance_settings["enable_parallel_processing"]
        self.max_worker_threads = performance_settings["max_worker_threads"]
        self.enable_performance_tracking = performance_settings["enable_performance_tracking"]
        self.enable_health_monitoring = performance_settings["enable_health_monitoring"]
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def _initialize_fallback_config(self):
        """Initialize with fallback configuration."""
        logger.warning("Using fallback configuration")
        
        # Use default configuration
        self.config = self._create_default_config()
        self._apply_configuration(self.config)
        
        logger.info("Tick loader initialized with fallback configuration")
    
    async def load_tick(self, tick_data: Dict[str, Any], priority: TickPriority = TickPriority.MEDIUM) -> bool:
        """Load a tick into the processing system."""
        try:
            start_time = time.time()
            
            # Create tick data object
            tick = TickData(
                timestamp=tick_data.get("timestamp", time.time()),
                symbol=tick_data.get("symbol", "unknown"),
                price=tick_data.get("price", 0.0),
                volume=tick_data.get("volume", 0.0),
                bid=tick_data.get("bid", tick_data.get("price", 0.0)),
                ask=tick_data.get("ask", tick_data.get("price", 0.0)),
                priority=priority,
                metadata=tick_data.get("metadata", {})
            )
            
            # Generate hash ID
            tick.hash_id = self._generate_tick_hash(tick)
            
            # Check memory constraints
            if not self._check_memory_constraints():
                logger.warning("⚠️ Memory constraints exceeded, dropping tick")
                self.stats["ticks_dropped"] += 1
                return False
            
            # Add to appropriate queue
            with self.lock:
                queue = self.tick_queues[priority]
                if len(queue) < queue.maxlen:
                    queue.append(tick)
                    self.stats["ticks_processed"] += 1
                    
                    # Add to memory pool
                    self._add_to_memory_pool(tick)
                    
                    # Update processing time
                    processing_time = (time.time() - start_time) * 1000
                    self.stats["processing_time_ms"] = processing_time
                    
                    return True
                else:
                    logger.warning(f"⚠️ Queue full for priority {priority.value}")
                    self.stats["ticks_dropped"] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to load tick: {e}")
            self.stats["ticks_dropped"] += 1
            return False
    
    def _generate_tick_hash(self, tick: TickData) -> str:
        """Generate unique hash for tick data."""
        try:
            # Create hash from tick data
            hash_data = f"{tick.timestamp}_{tick.symbol}_{tick.price}_{tick.volume}"
            return hashlib.sha256(hash_data.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _check_memory_constraints(self) -> bool:
        """Check if memory usage is within constraints."""
        try:
            if not self.enable_memory_monitoring:
                return True
            
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.max_memory_usage_mb:
                logger.warning(f"⚠️ Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_usage_mb}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory constraint check failed: {e}")
            return True  # Allow processing if check fails
    
    async def process_ticks(self, batch_size: Optional[int] = None) -> List[TickData]:
        """Process ticks from queues in priority order."""
        try:
            batch_size = batch_size or self.batch_size
            processed_ticks = []
            
            # Process ticks in priority order
            for priority in [TickPriority.CRITICAL, TickPriority.HIGH, TickPriority.MEDIUM, TickPriority.LOW, TickPriority.BACKGROUND]:
                with self.lock:
                    queue = self.tick_queues[priority]
                    
                    while queue and len(processed_ticks) < batch_size:
                        tick = queue.popleft()
                        processed_ticks.append(tick)
                        
                        # Update statistics
                        self.stats["ticks_processed"] += 1
            
            logger.debug(f"✅ Processed {len(processed_ticks)} ticks")
            return processed_ticks
            
        except Exception as e:
            logger.error(f"❌ Tick processing failed: {e}")
            return []
    
    def _add_to_memory_pool(self, tick: TickData):
        """Add tick to appropriate memory pool."""
        try:
            # Determine appropriate pool based on priority
            if tick.priority in [TickPriority.CRITICAL, TickPriority.HIGH]:
                pool = "high_frequency"
            elif tick.priority == TickPriority.MEDIUM:
                pool = "pattern_recognition"
            else:
                pool = "deep_analysis"
            
            # Add to pool
            if pool in self.memory_pools:
                self.memory_pools[pool].append(tick)
                
                # Update memory usage
                self.stats["memory_usage_mb"] = len(self.memory_pools[pool]) * 0.001  # Approximate
            
        except Exception as e:
            logger.error(f"❌ Failed to add tick to memory pool: {e}")
    
    async def start_processing(self):
        """Start the tick processing system."""
        try:
            if self.running:
                logger.warning("⚠️ Tick processing already running")
                return
            
            logger.info("🚀 Starting tick processing system...")
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ Tick processing system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start tick processing: {e}")
    
    def stop_processing(self):
        """Stop the tick processing system."""
        try:
            logger.info("🛑 Stopping tick processing system...")
            self.running = False
            
            logger.info("✅ Tick processing system stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop tick processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop for tick loader."""
        try:
            while self.running:
                # Process ticks
                asyncio.run(self.process_ticks())
                
                # Cleanup memory if needed
                if self.enable_auto_cleanup:
                    self._cleanup_memory()
                
                time.sleep(0.1)  # 100ms interval
                
        except Exception as e:
            logger.error(f"❌ Processing loop error: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory pools to prevent excessive usage."""
        try:
            current_time = time.time()
            
            for pool_name, pool in self.memory_pools.items():
                if isinstance(pool, deque):
                    # Remove old ticks based on cache duration
                    cache_duration_seconds = self.cache_duration_minutes * 60
                    
                    # Keep only recent ticks
                    while pool and (current_time - pool[0].timestamp) > cache_duration_seconds:
                        pool.popleft()
                        
                    # Update memory usage
                    self.stats["memory_usage_mb"] = len(pool) * 0.001  # Approximate
            
            logger.debug(f"✅ Memory cleanup completed, usage: {self.stats['memory_usage_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                "running": self.running,
                "stats": self.stats,
                "system_info": {
                    "platform": self.system_info.platform if self.system_info else "unknown",
                    "ram_gb": self.system_info.ram_gb if self.system_info else 0.0,
                    "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "unknown"
                },
                "queue_sizes": {
                    priority.value: len(queue)
                    for priority, queue in self.tick_queues.items()
                },
                "memory_pools": {
                    pool_name: len(pool)
                    for pool_name, pool in self.memory_pools.items()
                },
                "configuration": {
                    "batch_size": self.batch_size,
                    "max_queue_size": self.max_queue_size,
                    "enable_compression": self.enable_compression,
                    "enable_encryption": self.enable_encryption,
                    "max_memory_usage_mb": self.max_memory_usage_mb
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics collection failed: {e}")
            return {"error": str(e)}
    
    def encrypt_tick_data(self, tick_data: Dict[str, Any]) -> str:
        """Encrypt tick data for secure transmission."""
        try:
            if not self.enable_encryption:
                return json.dumps(tick_data)
            
            data_str = json.dumps(tick_data)
            return self.alpha256.encrypt(data_str)
            
        except Exception as e:
            logger.error(f"❌ Tick data encryption failed: {e}")
            return json.dumps(tick_data)
    
    def decrypt_tick_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt tick data."""
        try:
            if not self.enable_encryption:
                return json.loads(encrypted_data)
            
            decrypted_str = self.alpha256.decrypt(encrypted_data)
            return json.loads(decrypted_str)
            
        except Exception as e:
            logger.error(f"❌ Tick data decryption failed: {e}")
            return {}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for tick loader testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Testing Tick Loader...")
    
    # Create tick loader instance
    tick_loader = TickLoader()
    
    try:
        # Start the system
        await tick_loader.start_processing()
        
        # Generate test ticks
        for i in range(100):
            tick_data = {
                "timestamp": time.time() + i,
                "symbol": "BTC/USDT",
                "price": 45000.0 + i * 10,
                "volume": 1000000.0 + i * 1000,
                "bid": 44999.0 + i * 10,
                "ask": 45001.0 + i * 10,
                "metadata": {"test": True, "index": i}
            }
            
            # Load tick with different priorities
            priority = TickPriority.MEDIUM if i % 3 == 0 else TickPriority.HIGH
            success = await tick_loader.load_tick(tick_data, priority)
            
            if success:
                logger.debug(f"✅ Loaded tick {i}")
            else:
                logger.warning(f"⚠️ Failed to load tick {i}")
        
        # Process ticks
        processed_ticks = await tick_loader.process_ticks(batch_size=50)
        logger.info(f"✅ Processed {len(processed_ticks)} ticks")
        
        # Get statistics
        stats = tick_loader.get_statistics()
        logger.info(f"📊 Statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        # Stop the system
        tick_loader.stop_processing()
        
        logger.info("👋 Tick Loader test complete")

if __name__ == "__main__":
    asyncio.run(main()) 