#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Cache for Schwabot Trading System
========================================

Optimized signal processing and caching system with hardware-optimized
performance and intelligent similarity matching. This system provides
real-time signal caching with priority-based processing and advanced
pattern recognition capabilities.

Features:
- Hardware-optimized signal caching and processing
- Priority-based queuing system for different signal types
- Intelligent similarity matching and pattern recognition
- Real-time signal data processing and analysis
- Integration with existing mathematical framework
- Secure signal data handling with encryption
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import hashlib
import psutil
import collections

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

class SignalType(Enum):
    """Signal type enumeration."""
    PRICE = "price"           # Price-based signals
    VOLUME = "volume"         # Volume-based signals
    PATTERN = "pattern"       # Pattern recognition signals
    INDICATOR = "indicator"   # Technical indicator signals
    COMPOSITE = "composite"   # Composite signals
    GHOST = "ghost"          # Ghost strategy signals

class SignalPriority(Enum):
    """Signal processing priority levels."""
    CRITICAL = "critical"      # Real-time execution signals
    HIGH = "high"             # Short-term strategy signals
    MEDIUM = "medium"         # Pattern recognition
    LOW = "low"               # Historical analysis
    BACKGROUND = "background" # Deep analysis

@dataclass
class SignalData:
    """Signal data structure."""
    timestamp: float
    signal_type: SignalType
    symbol: str
    data: Any
    priority: SignalPriority = SignalPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: str = ""
    similarity_score: float = 0.0
    confidence: float = 0.0

class SignalCache:
    """Optimized signal processing and caching system."""
    
    def __init__(self, cache_size: int = 10000):
        """Initialize signal cache with hardware auto-detection."""
        self.cache_size = cache_size
        
        self.hardware_detector = HardwareAutoDetector()
        self.hash_config = HashConfigManager()
        self.alpha256 = Alpha256Encryption()
        
        # Hardware-aware configuration
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # Signal storage
        self.signal_cache: Dict[str, SignalData] = {}
        self.signal_queues: Dict[SignalPriority, deque] = {
            priority: deque(maxlen=self._get_queue_size(priority))
            for priority in SignalPriority
        }
        
        # Similarity matching
        self.similarity_cache: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.similarity_threshold = 0.8
        
        # Performance tracking
        self.stats = {
            "signals_cached": 0,
            "signals_processed": 0,
            "signals_dropped": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "similarity_matches": 0,
            "processing_time_ms": 0.0
        }
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Initialize with hardware detection
        self._initialize_with_hardware_detection()
    
    def _initialize_with_hardware_detection(self):
        """Initialize with hardware auto-detection."""
        try:
            logger.info("Initializing signal cache with hardware auto-detection...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"Hardware detected: {self.system_info.platform}")
            logger.info(f"RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"Optimization: {self.system_info.optimization_mode.value}")
            
            # Configure cache based on hardware
            self._configure_cache_size()
            
            # Load or create configuration
            self._load_configuration()
            
            logger.info("Signal cache initialized with hardware optimization")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _configure_cache_size(self):
        """Configure cache size based on hardware capabilities."""
        try:
            # Get queue sizes based on priority
            for priority in SignalPriority:
                queue_size = self._get_queue_size(priority)
                # Create new deque with correct maxlen instead of modifying existing
                self.signal_queues[priority] = collections.deque(maxlen=queue_size)
            
            logger.info(f"Configured cache sizes for {len(self.signal_queues)} priority levels")
            
        except Exception as e:
            logger.error(f"Cache size configuration failed: {e}")
            # Use fallback sizes
            for priority in SignalPriority:
                self.signal_queues[priority] = collections.deque(maxlen=1000)
    
    def _get_queue_size(self, priority: SignalPriority) -> int:
        """Get queue size based on priority and hardware capabilities."""
        if not self.memory_config:
            return 1000  # Default fallback
            
        base_sizes = {
            SignalPriority.CRITICAL: 100,
            SignalPriority.HIGH: 500,
            SignalPriority.MEDIUM: 1000,
            SignalPriority.LOW: 2000,
            SignalPriority.BACKGROUND: 5000
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
        """Load or create signal cache configuration."""
        config_path = Path("config/signal_cache_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Loaded existing signal cache configuration")
            else:
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info("✅ Created new signal cache configuration")
                
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
            "cache_settings": {
                "cache_size": self.cache_size,
                "enable_compression": True,
                "enable_encryption": True,
                "enable_similarity_matching": True,
                "similarity_threshold": 0.8,
                "cache_duration_minutes": 60
            },
            "processing_settings": {
                "batch_size": 100,
                "max_concurrent_signals": 50,
                "enable_parallel_processing": True,
                "enable_performance_tracking": True
            },
            "memory_management": {
                "max_memory_usage_mb": self.system_info.ram_gb * 50 if self.system_info else 400,
                "cleanup_interval_seconds": 300,
                "enable_auto_cleanup": True,
                "enable_memory_monitoring": True
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        self.config = config
        
        # Apply cache settings
        cache_settings = config["cache_settings"]
        self.cache_size = cache_settings["cache_size"]
        self.enable_compression = cache_settings["enable_compression"]
        self.enable_encryption = cache_settings["enable_encryption"]
        self.enable_similarity_matching = cache_settings["enable_similarity_matching"]
        self.similarity_threshold = cache_settings["similarity_threshold"]
        self.cache_duration_minutes = cache_settings["cache_duration_minutes"]
        
        # Apply processing settings
        processing_settings = config["processing_settings"]
        self.batch_size = processing_settings["batch_size"]
        self.max_concurrent_signals = processing_settings["max_concurrent_signals"]
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
            config_path = Path("config/signal_cache_config.json")
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
        
        logger.info("Signal cache initialized with fallback configuration")
    
    async def cache_signal(self, signal_type: SignalType, symbol: str, data: Any, priority: SignalPriority = SignalPriority.MEDIUM, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Cache a signal in the system."""
        try:
            start_time = time.time()
            
            # Create signal data object
            signal = SignalData(
                timestamp=time.time(),
                signal_type=signal_type,
                symbol=symbol,
                data=data,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Generate hash ID
            signal.hash_id = self._generate_signal_hash(signal)
            
            # Check memory constraints
            if not self._check_memory_constraints():
                logger.warning("⚠️ Memory constraints exceeded, dropping signal")
                self.stats["signals_dropped"] += 1
                return False
            
            # Add to cache and queue
            with self.lock:
                # Add to cache
                if len(self.signal_cache) >= self.cache_size:
                    # Remove oldest signal
                    oldest_hash = min(self.signal_cache.keys(), key=lambda k: self.signal_cache[k].timestamp)
                    del self.signal_cache[oldest_hash]
                
                self.signal_cache[signal.hash_id] = signal
                
                # Add to priority queue
                queue = self.signal_queues[priority]
                if len(queue) < queue.maxlen:
                    queue.append(signal)
                    
                    # Update similarity cache if enabled
                    if self.enable_similarity_matching:
                        self._update_similarity_cache(signal)
                    
                    # Update statistics
                    self.stats["signals_cached"] += 1
                    processing_time = (time.time() - start_time) * 1000
                    self.stats["processing_time_ms"] = processing_time
                    
                    return True
                else:
                    logger.warning(f"⚠️ Queue full for priority {priority.value}")
                    self.stats["signals_dropped"] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to cache signal: {e}")
            self.stats["signals_dropped"] += 1
            return False
    
    def _generate_signal_hash(self, signal: SignalData) -> str:
        """Generate unique hash for signal data."""
        try:
            # Create hash from signal data
            hash_data = f"{signal.timestamp}_{signal.signal_type.value}_{signal.symbol}_{str(signal.data)}"
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
    
    async def find_similar_signals(self, signal_data: Any, signal_type: SignalType, threshold: Optional[float] = None) -> List[Tuple[SignalData, float]]:
        """Find signals similar to the given signal data."""
        try:
            if not self.enable_similarity_matching:
                return []
            
            threshold = threshold or self.similarity_threshold
            similar_signals = []
            
            # Search through cached signals
            for signal in self.signal_cache.values():
                if signal.signal_type == signal_type:
                    similarity = self._calculate_similarity(signal_data, signal.data)
                    if similarity >= threshold:
                        similar_signals.append((signal, similarity))
            
            # Sort by similarity score
            similar_signals.sort(key=lambda x: x[1], reverse=True)
            
            self.stats["similarity_matches"] += len(similar_signals)
            return similar_signals
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    def _calculate_similarity(self, data1: Any, data2: Any) -> float:
        """Calculate similarity between two signal data objects."""
        try:
            # Simple similarity calculation based on data type
            if isinstance(data1, dict) and isinstance(data2, dict):
                # Dictionary similarity
                keys1 = set(data1.keys())
                keys2 = set(data2.keys())
                
                if not keys1 and not keys2:
                    return 1.0
                
                intersection = keys1.intersection(keys2)
                union = keys1.union(keys2)
                
                if not union:
                    return 0.0
                
                key_similarity = len(intersection) / len(union)
                
                # Value similarity for common keys
                value_similarity = 0.0
                if intersection:
                    for key in intersection:
                        if data1[key] == data2[key]:
                            value_similarity += 1.0
                    value_similarity /= len(intersection)
                
                return (key_similarity + value_similarity) / 2.0
                
            elif isinstance(data1, (list, tuple)) and isinstance(data2, (list, tuple)):
                # List/tuple similarity
                if len(data1) != len(data2):
                    return 0.0
                
                matches = sum(1 for a, b in zip(data1, data2) if a == b)
                return matches / len(data1) if data1 else 0.0
                
            else:
                # Direct comparison
                return 1.0 if data1 == data2 else 0.0
                
        except Exception:
            return 0.0
    
    async def get_signal(self, signal_hash: str) -> Optional[SignalData]:
        """Get a specific signal by hash."""
        try:
            with self.lock:
                signal = self.signal_cache.get(signal_hash)
                if signal:
                    self.stats["cache_hits"] += 1
                    return signal
                else:
                    self.stats["cache_misses"] += 1
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to get signal: {e}")
            return None
    
    async def get_signals_by_type(self, signal_type: SignalType, limit: Optional[int] = None) -> List[SignalData]:
        """Get signals of a specific type."""
        try:
            signals = []
            
            with self.lock:
                for signal in self.signal_cache.values():
                    if signal.signal_type == signal_type:
                        signals.append(signal)
                        if limit and len(signals) >= limit:
                            break
            
            # Sort by timestamp (newest first)
            signals.sort(key=lambda x: x.timestamp, reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to get signals by type: {e}")
            return []
    
    async def get_signals_by_symbol(self, symbol: str, limit: Optional[int] = None) -> List[SignalData]:
        """Get signals for a specific symbol."""
        try:
            signals = []
            
            with self.lock:
                for signal in self.signal_cache.values():
                    if signal.symbol == symbol:
                        signals.append(signal)
                        if limit and len(signals) >= limit:
                            break
            
            # Sort by timestamp (newest first)
            signals.sort(key=lambda x: x.timestamp, reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to get signals by symbol: {e}")
            return []
    
    async def process_signals(self, batch_size: Optional[int] = None) -> List[SignalData]:
        """Process signals from queues in priority order."""
        try:
            batch_size = batch_size or self.batch_size
            processed_signals = []
            
            # Process signals in priority order
            for priority in [SignalPriority.CRITICAL, SignalPriority.HIGH, SignalPriority.MEDIUM, SignalPriority.LOW, SignalPriority.BACKGROUND]:
                with self.lock:
                    queue = self.signal_queues[priority]
                    
                    while queue and len(processed_signals) < batch_size:
                        signal = queue.popleft()
                        processed_signals.append(signal)
                        
                        # Update statistics
                        self.stats["signals_processed"] += 1
            
            logger.debug(f"✅ Processed {len(processed_signals)} signals")
            return processed_signals
            
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
            return []
    
    def _add_to_cache(self, signal: SignalData):
        """Add signal to cache."""
        try:
            with self.lock:
                if len(self.signal_cache) >= self.cache_size:
                    # Remove oldest signal
                    oldest_hash = min(self.signal_cache.keys(), key=lambda k: self.signal_cache[k].timestamp)
                    del self.signal_cache[oldest_hash]
                
                self.signal_cache[signal.hash_id] = signal
                
        except Exception as e:
            logger.error(f"❌ Failed to add signal to cache: {e}")
    
    def _update_similarity_cache(self, signal: SignalData):
        """Update similarity cache with new signal."""
        try:
            # Find similar signals and update cache
            similar_signals = []
            
            for cached_signal in self.signal_cache.values():
                if cached_signal.hash_id != signal.hash_id and cached_signal.signal_type == signal.signal_type:
                    similarity = self._calculate_similarity(signal.data, cached_signal.data)
                    if similarity >= self.similarity_threshold:
                        similar_signals.append((cached_signal.hash_id, similarity))
            
            # Update similarity cache
            self.similarity_cache[signal.hash_id] = similar_signals
            
        except Exception as e:
            logger.error(f"❌ Failed to update similarity cache: {e}")
    
    async def start_processing(self):
        """Start the signal processing system."""
        try:
            if self.running:
                logger.warning("⚠️ Signal processing already running")
                return
            
            logger.info("🚀 Starting signal processing system...")
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ Signal processing system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start signal processing: {e}")
    
    def stop_processing(self):
        """Stop the signal processing system."""
        try:
            logger.info("🛑 Stopping signal processing system...")
            self.running = False
            
            logger.info("✅ Signal processing system stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop signal processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop for signal cache."""
        try:
            while self.running:
                # Process signals
                asyncio.run(self.process_signals())
                
                # Cleanup memory if needed
                if self.enable_auto_cleanup:
                    self._cleanup_memory()
                
                time.sleep(0.1)  # 100ms interval
                
        except Exception as e:
            logger.error(f"❌ Processing loop error: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory to prevent excessive usage."""
        try:
            current_time = time.time()
            cache_duration_seconds = self.cache_duration_minutes * 60
            
            # Remove old signals from cache
            with self.lock:
                old_signals = [
                    hash_id for hash_id, signal in self.signal_cache.items()
                    if (current_time - signal.timestamp) > cache_duration_seconds
                ]
                
                for hash_id in old_signals:
                    del self.signal_cache[hash_id]
                    if hash_id in self.similarity_cache:
                        del self.similarity_cache[hash_id]
            
            logger.debug(f"✅ Memory cleanup completed, cache size: {len(self.signal_cache)}")
            
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
                "cache_info": {
                    "cache_size": len(self.signal_cache),
                    "max_cache_size": self.cache_size,
                    "queue_sizes": {
                        priority.value: len(queue)
                        for priority, queue in self.signal_queues.items()
                    }
                },
                "configuration": {
                    "enable_compression": self.enable_compression,
                    "enable_encryption": self.enable_encryption,
                    "enable_similarity_matching": self.enable_similarity_matching,
                    "similarity_threshold": self.similarity_threshold,
                    "max_memory_usage_mb": self.max_memory_usage_mb
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics collection failed: {e}")
            return {"error": str(e)}
    
    def encrypt_signal_data(self, signal_data: Dict[str, Any]) -> str:
        """Encrypt signal data for secure transmission."""
        try:
            if not self.enable_encryption:
                return json.dumps(signal_data)
            
            data_str = json.dumps(signal_data)
            return self.alpha256.encrypt(data_str)
            
        except Exception as e:
            logger.error(f"❌ Signal data encryption failed: {e}")
            return json.dumps(signal_data)
    
    def decrypt_signal_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt signal data."""
        try:
            if not self.enable_encryption:
                return json.loads(encrypted_data)
            
            decrypted_str = self.alpha256.decrypt(encrypted_data)
            return json.loads(decrypted_str)
            
        except Exception as e:
            logger.error(f"❌ Signal data decryption failed: {e}")
            return {}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for signal cache testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Testing Signal Cache...")
    
    # Create signal cache instance
    signal_cache = SignalCache()
    
    try:
        # Start the system
        await signal_cache.start_processing()
        
        # Generate test signals
        for i in range(50):
            signal_data = {
                "price": 45000.0 + i * 10,
                "volume": 1000000.0 + i * 1000,
                "rsi": 50.0 + (i % 20),
                "trend": "bullish" if i % 2 == 0 else "bearish"
            }
            
            # Cache signal with different types and priorities
            signal_type = SignalType.PRICE if i % 3 == 0 else SignalType.INDICATOR
            priority = SignalPriority.HIGH if i % 2 == 0 else SignalPriority.MEDIUM
            
            success = await signal_cache.cache_signal(
                signal_type, "BTC/USDT", signal_data, priority
            )
            
            if success:
                logger.debug(f"✅ Cached signal {i}")
            else:
                logger.warning(f"⚠️ Failed to cache signal {i}")
        
        # Test similarity matching
        test_data = {"price": 45000.0, "volume": 1000000.0, "rsi": 55.0}
        similar_signals = await signal_cache.find_similar_signals(
            test_data, SignalType.PRICE, threshold=0.7
        )
        logger.info(f"✅ Found {len(similar_signals)} similar signals")
        
        # Get signals by type
        price_signals = await signal_cache.get_signals_by_type(SignalType.PRICE, limit=10)
        logger.info(f"✅ Retrieved {len(price_signals)} price signals")
        
        # Process signals
        processed_signals = await signal_cache.process_signals(batch_size=20)
        logger.info(f"✅ Processed {len(processed_signals)} signals")
        
        # Get statistics
        stats = signal_cache.get_statistics()
        logger.info(f"📊 Statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        # Stop the system
        signal_cache.stop_processing()
        
        logger.info("👋 Signal Cache test complete")

if __name__ == "__main__":
    asyncio.run(main()) 