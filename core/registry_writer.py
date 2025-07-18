#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry Writer for Schwabot Trading System
==========================================

Smart archiving and state persistence system with hardware-optimized
performance and intelligent storage management. This system provides
real-time data archiving with priority-based processing and advanced
compression and encryption capabilities.

Features:
- Hardware-optimized data archiving and storage management
- Priority-based queuing system for different data types
- Intelligent compression and encryption for data security
- Real-time data persistence and backup management
- Integration with existing mathematical framework
- Secure data handling with Alpha256 encryption
"""

import asyncio
import json
import logging
import threading
import time
import shutil
import gzip
import pickle
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

class ArchivePriority(Enum):
    """Archive priority levels."""
    CRITICAL = "critical"      # System state, configurations
    HIGH = "high"             # Trading signals, performance data
    MEDIUM = "medium"         # Historical data, patterns
    LOW = "low"               # Debug logs, temporary data
    BACKGROUND = "background" # Deep analysis results

@dataclass
class ArchiveEntry:
    """Archive entry structure."""
    timestamp: float
    data_type: str
    data: Any
    priority: ArchivePriority = ArchivePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: str = ""
    compressed: bool = False
    encrypted: bool = False

class RegistryWriter:
    """Smart archiving and state persistence system."""
    
    def __init__(self, base_path: str = "data/registry"):
        """Initialize registry writer with hardware auto-detection."""
        self.base_path = Path(base_path)
        self.hardware_detector = HardwareAutoDetector()
        self.hash_config = HashConfigManager()
        self.alpha256 = Alpha256Encryption()
        
        # Hardware-aware configuration
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # Archive queues
        self.archive_queues: Dict[ArchivePriority, deque] = {
            priority: deque(maxlen=self._get_queue_size(priority))
            for priority in ArchivePriority
        }
        
        # Storage management
        self.storage_pools: Dict[str, List[ArchiveEntry]] = {
            "system_state": [],
            "trading_data": [],
            "historical_data": [],
            "analysis_results": []
        }
        
        # Performance tracking
        self.stats = {
            "entries_written": 0,
            "entries_dropped": 0,
            "storage_usage_mb": 0.0,
            "compression_ratio": 0.0,
            "backup_count": 0,
            "cleanup_count": 0
        }
        
        # Threading
        self.writing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Initialize with hardware detection
        self._initialize_with_hardware_detection()
    
    def _initialize_with_hardware_detection(self):
        """Initialize with hardware auto-detection."""
        try:
            logger.info("Initializing registry writer with hardware auto-detection...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"Hardware detected: {self.system_info.platform}")
            logger.info(f"RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"Optimization: {self.system_info.optimization_mode.value}")
            
            # Configure storage pools based on hardware
            self._configure_storage_pools()
            
            # Load or create configuration
            self._load_configuration()
            
            logger.info("Registry writer initialized with hardware optimization")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _configure_storage_pools(self):
        """Configure storage pools based on hardware capabilities."""
        try:
            # Check if memory_config has memory_pools attribute
            if hasattr(self.memory_config, 'memory_pools'):
                for pool_name, pool_config in self.memory_config.memory_pools.items():
                    self.storage_pools[pool_name] = {
                        'path': pool_config.get('path', f"data/registry/{pool_name}"),
                        'max_size_mb': pool_config.get('max_size_mb', 1000),
                        'compression': pool_config.get('compression', True),
                        'encryption': pool_config.get('encryption', True)
                    }
            else:
                # Use default storage pools
                self.storage_pools = {
                    'high_priority': {
                        'path': 'data/registry/high_priority',
                        'max_size_mb': 1000,
                        'compression': True,
                        'encryption': True
                    },
                    'medium_priority': {
                        'path': 'data/registry/medium_priority',
                        'max_size_mb': 2000,
                        'compression': True,
                        'encryption': False
                    },
                    'low_priority': {
                        'path': 'data/registry/low_priority',
                        'max_size_mb': 5000,
                        'compression': False,
                        'encryption': False
                    }
                }
            
            logger.info(f"Configured {len(self.storage_pools)} storage pools")
            
        except Exception as e:
            logger.error(f"Storage pool configuration failed: {e}")
            # Use fallback configuration
            self._initialize_fallback_config()
    
    def _get_queue_size(self, priority: ArchivePriority) -> int:
        """Get queue size based on priority and hardware capabilities."""
        if not self.memory_config:
            return 1000  # Default fallback
            
        base_sizes = {
            ArchivePriority.CRITICAL: 50,
            ArchivePriority.HIGH: 200,
            ArchivePriority.MEDIUM: 500,
            ArchivePriority.LOW: 1000,
            ArchivePriority.BACKGROUND: 2000
        }
        
        base_size = base_sizes[priority]
        
        # Scale based on storage capacity
        storage_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "ultra": 4.0
        }.get(self.system_info.ram_tier.value, 1.0)
        
        return int(base_size * storage_multiplier)
    
    def _create_directory_structure(self):
        """Create directory structure for registry."""
        try:
            # Create base directories
            directories = [
                self.base_path,
                self.base_path / "system_state",
                self.base_path / "trading_data",
                self.base_path / "historical_data",
                self.base_path / "analysis_results",
                self.base_path / "backups",
                self.base_path / "temp"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            logger.info("✅ Created registry directory structure")
            
        except Exception as e:
            logger.error(f"❌ Failed to create directory structure: {e}")
    
    def _get_storage_info(self) -> str:
        """Get storage information."""
        try:
            disk_usage = shutil.disk_usage(self.base_path)
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            
            return f"{total_gb:.1f}GB total, {free_gb:.1f}GB free, {used_gb:.1f}GB used"
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage info: {e}")
            return "unknown"
    
    def _load_configuration(self):
        """Load or create registry writer configuration."""
        config_path = Path("config/registry_writer_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Loaded existing registry writer configuration")
            else:
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info("✅ Created new registry writer configuration")
                
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
            "storage_settings": {
                "base_path": str(self.base_path),
                "enable_compression": True,
                "enable_encryption": True,
                "max_file_size_mb": 100,
                "backup_retention_days": 30,
                "cleanup_interval_hours": 24
            },
            "processing_settings": {
                "batch_size": 50,
                "max_concurrent_writes": 10,
                "enable_parallel_processing": True,
                "enable_performance_tracking": True
            },
            "memory_management": {
                "max_memory_usage_mb": self.system_info.ram_gb * 25 if self.system_info else 200,
                "cleanup_interval_seconds": 3600,
                "enable_auto_cleanup": True,
                "enable_memory_monitoring": True
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        self.config = config
        
        # Apply storage settings
        storage_settings = config["storage_settings"]
        self.base_path = Path(storage_settings["base_path"])
        self.enable_compression = storage_settings["enable_compression"]
        self.enable_encryption = storage_settings["enable_encryption"]
        self.max_file_size_mb = storage_settings["max_file_size_mb"]
        self.backup_retention_days = storage_settings["backup_retention_days"]
        self.cleanup_interval_hours = storage_settings["cleanup_interval_hours"]
        
        # Apply processing settings
        processing_settings = config["processing_settings"]
        self.batch_size = processing_settings["batch_size"]
        self.max_concurrent_writes = processing_settings["max_concurrent_writes"]
        self.enable_parallel_processing = processing_settings["enable_parallel_processing"]
        self.enable_performance_tracking = processing_settings["enable_performance_tracking"]
        
        # Apply memory management settings
        memory_settings = config["memory_management"]
        self.max_memory_usage_mb = memory_settings["max_memory_usage_mb"]
        self.cleanup_interval_seconds = memory_settings["cleanup_interval_seconds"]
        self.enable_auto_cleanup = memory_settings["enable_auto_cleanup"]
        self.enable_memory_monitoring = memory_settings["enable_memory_monitoring"]
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            config_path = Path("config/registry_writer_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def _initialize_fallback_config(self):
        """Initialize with fallback configuration."""
        logger.warning("Using fallback configuration")
        
        # Use default configuration
        self.config = self._create_default_config()
        self._apply_configuration(self.config)
        
        logger.info("Registry writer initialized with fallback configuration")
    
    async def write_entry(self, data_type: str, data: Any, priority: ArchivePriority = ArchivePriority.MEDIUM, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write an entry to the registry."""
        try:
            start_time = time.time()
            
            # Create archive entry
            entry = ArchiveEntry(
                timestamp=time.time(),
                data_type=data_type,
                data=data,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Generate hash ID
            entry.hash_id = self._generate_entry_hash(entry)
            
            # Check storage constraints
            if not self._check_storage_constraints():
                logger.warning("⚠️ Storage constraints exceeded, dropping entry")
                self.stats["entries_dropped"] += 1
                return False
            
            # Add to queue
            with self.lock:
                queue = self.archive_queues[priority]
                if len(queue) < queue.maxlen:
                    queue.append(entry)
                    
                    # Add to storage pool
                    self._add_to_storage_pool(entry)
                    
                    # Update statistics
                    self.stats["entries_written"] += 1
                    processing_time = (time.time() - start_time) * 1000
                    
                    return True
                else:
                    logger.warning(f"⚠️ Queue full for priority {priority.value}")
                    self.stats["entries_dropped"] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to write entry: {e}")
            self.stats["entries_dropped"] += 1
            return False
    
    def _generate_entry_hash(self, entry: ArchiveEntry) -> str:
        """Generate unique hash for archive entry."""
        try:
            # Create hash from entry data
            hash_data = f"{entry.timestamp}_{entry.data_type}_{str(entry.data)}"
            return hashlib.sha256(hash_data.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _check_storage_constraints(self) -> bool:
        """Check if storage usage is within constraints."""
        try:
            if not self.enable_memory_monitoring:
                return True
            
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.max_memory_usage_mb:
                logger.warning(f"⚠️ Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_usage_mb}MB")
                return False
            
            # Check disk space
            disk_usage = shutil.disk_usage(self.base_path)
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB free")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage constraint check failed: {e}")
            return True  # Allow processing if check fails
    
    async def process_entries(self, batch_size: Optional[int] = None) -> List[ArchiveEntry]:
        """Process entries from queues in priority order."""
        try:
            batch_size = batch_size or self.batch_size
            processed_entries = []
            
            # Process entries in priority order
            for priority in [ArchivePriority.CRITICAL, ArchivePriority.HIGH, ArchivePriority.MEDIUM, ArchivePriority.LOW, ArchivePriority.BACKGROUND]:
                with self.lock:
                    queue = self.archive_queues[priority]
                    
                    while queue and len(processed_entries) < batch_size:
                        entry = queue.popleft()
                        processed_entries.append(entry)
            
            logger.debug(f"✅ Processed {len(processed_entries)} entries")
            return processed_entries
            
        except Exception as e:
            logger.error(f"❌ Entry processing failed: {e}")
            return []
    
    def _add_to_storage_pool(self, entry: ArchiveEntry):
        """Add entry to appropriate storage pool."""
        try:
            # Determine appropriate pool based on data type
            if entry.data_type in ["system_state", "config"]:
                pool = "system_state"
            elif entry.data_type in ["tick", "signal", "trade"]:
                pool = "trading_data"
            elif entry.data_type in ["historical", "backtest"]:
                pool = "historical_data"
            else:
                pool = "analysis_results"
            
            # Add to pool
            if pool in self.storage_pools:
                self.storage_pools[pool].append(entry)
                
                # Update storage usage
                self.stats["storage_usage_mb"] = len(self.storage_pools[pool]) * 0.002  # Approximate
            
        except Exception as e:
            logger.error(f"❌ Failed to add entry to storage pool: {e}")
    
    async def write_to_disk(self, entry: ArchiveEntry) -> bool:
        """Write entry to disk with compression and encryption."""
        try:
            # Prepare data
            data_to_write = {
                "timestamp": entry.timestamp,
                "data_type": entry.data_type,
                "data": entry.data,
                "priority": entry.priority.value,
                "metadata": entry.metadata,
                "hash_id": entry.hash_id
            }
            
            # Serialize data
            serialized_data = json.dumps(data_to_write, default=str)
            
            # Compress if enabled
            if self.enable_compression:
                compressed_data = gzip.compress(serialized_data.encode('utf-8'))
                entry.compressed = True
                data_to_write = compressed_data
            else:
                data_to_write = serialized_data.encode('utf-8')
            
            # Encrypt if enabled
            if self.enable_encryption:
                encrypted_data = self.alpha256.encrypt(data_to_write.decode('latin-1'))
                entry.encrypted = True
                data_to_write = encrypted_data.encode('utf-8')
            
            # Get file path
            file_path = self._get_file_path(entry)
            
            # Write to disk
            with open(file_path, 'wb') as f:
                f.write(data_to_write)
            
            logger.debug(f"✅ Wrote entry to disk: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to write entry to disk: {e}")
            return False
    
    def _get_file_path(self, entry: ArchiveEntry) -> Path:
        """Get file path for archive entry."""
        try:
            # Create filename
            timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"{entry.data_type}_{entry.hash_id}_{timestamp}.dat"
            
            # Determine directory based on priority
            if entry.priority == ArchivePriority.CRITICAL:
                directory = self.base_path / "system_state"
            elif entry.priority == ArchivePriority.HIGH:
                directory = self.base_path / "trading_data"
            elif entry.priority == ArchivePriority.MEDIUM:
                directory = self.base_path / "historical_data"
            else:
                directory = self.base_path / "analysis_results"
            
            return directory / filename
            
        except Exception as e:
            logger.error(f"❌ Failed to get file path: {e}")
            return self.base_path / "temp" / f"entry_{entry.hash_id}.dat"
    
    async def start_writing(self):
        """Start the registry writing system."""
        try:
            if self.running:
                logger.warning("⚠️ Registry writing already running")
                return
            
            logger.info("🚀 Starting registry writing system...")
            self.running = True
            
            # Start writing thread
            self.writing_thread = threading.Thread(target=self._writing_loop, daemon=True)
            self.writing_thread.start()
            
            logger.info("✅ Registry writing system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start registry writing: {e}")
    
    def stop_writing(self):
        """Stop the registry writing system."""
        try:
            logger.info("🛑 Stopping registry writing system...")
            self.running = False
            
            logger.info("✅ Registry writing system stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop registry writing: {e}")
    
    def _writing_loop(self):
        """Main writing loop for registry."""
        try:
            while self.running:
                # Process and write entries
                asyncio.run(self._process_and_write_batch())
                
                # Cleanup storage if needed
                if self.enable_auto_cleanup:
                    self._cleanup_storage()
                
                time.sleep(1.0)  # 1 second interval
                
        except Exception as e:
            logger.error(f"❌ Writing loop error: {e}")
    
    async def _process_and_write_batch(self):
        """Process and write a batch of entries."""
        try:
            # Process entries
            entries = await self.process_entries()
            
            # Write to disk
            for entry in entries:
                success = await self.write_to_disk(entry)
                if success:
                    self.stats["entries_written"] += 1
                else:
                    self.stats["entries_dropped"] += 1
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
    
    def _cleanup_storage(self):
        """Clean up storage to prevent excessive usage."""
        try:
            current_time = time.time()
            
            # Remove old entries from storage pools
            for pool_name, pool in self.storage_pools.items():
                if isinstance(pool, deque):
                    # Remove entries older than retention period
                    retention_seconds = self.backup_retention_days * 24 * 3600
                    
                    while pool and (current_time - pool[0].timestamp) > retention_seconds:
                        pool.popleft()
                    
                    # Update storage usage
                    self.stats["storage_usage_mb"] = len(pool) * 0.002  # Approximate
            
            # Rotate backups
            self._rotate_backups()
            
            logger.debug(f"✅ Storage cleanup completed, usage: {self.stats['storage_usage_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Storage cleanup failed: {e}")
    
    def _rotate_backups(self):
        """Rotate old backup files."""
        try:
            backup_dir = self.base_path / "backups"
            if not backup_dir.exists():
                return
            
            current_time = time.time()
            retention_seconds = self.backup_retention_days * 24 * 3600
            
            # Remove old backup files
            for backup_file in backup_dir.glob("*.dat"):
                if (current_time - backup_file.stat().st_mtime) > retention_seconds:
                    backup_file.unlink()
                    self.stats["backup_count"] += 1
            
        except Exception as e:
            logger.error(f"❌ Backup rotation failed: {e}")
    
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
                "storage_info": {
                    "base_path": str(self.base_path),
                    "storage_usage_mb": self.stats["storage_usage_mb"],
                    "queue_sizes": {
                        priority.value: len(queue)
                        for priority, queue in self.archive_queues.items()
                    }
                },
                "configuration": {
                    "enable_compression": self.enable_compression,
                    "enable_encryption": self.enable_encryption,
                    "max_file_size_mb": self.max_file_size_mb,
                    "backup_retention_days": self.backup_retention_days,
                    "max_memory_usage_mb": self.max_memory_usage_mb
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics collection failed: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for registry writer testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Testing Registry Writer...")
    
    # Create registry writer instance
    registry_writer = RegistryWriter()
    
    try:
        # Start the system
        await registry_writer.start_writing()
        
        # Generate test entries
        for i in range(50):
            entry_data = {
                "test_id": i,
                "message": f"Test entry {i}",
                "value": 100.0 + i * 10,
                "timestamp": time.time()
            }
            
            # Write entry with different types and priorities
            data_type = "system_state" if i % 4 == 0 else "trading_data"
            priority = ArchivePriority.HIGH if i % 2 == 0 else ArchivePriority.MEDIUM
            
            success = await registry_writer.write_entry(
                data_type, entry_data, priority
            )
            
            if success:
                logger.debug(f"✅ Wrote entry {i}")
            else:
                logger.warning(f"⚠️ Failed to write entry {i}")
        
        # Process entries
        processed_entries = await registry_writer.process_entries(batch_size=20)
        logger.info(f"✅ Processed {len(processed_entries)} entries")
        
        # Get statistics
        stats = registry_writer.get_statistics()
        logger.info(f"📊 Statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        # Stop the system
        registry_writer.stop_writing()
        
        logger.info("👋 Registry Writer test complete")

if __name__ == "__main__":
    asyncio.run(main()) 