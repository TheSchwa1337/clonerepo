#!/usr/bin/env python3
"""
Unified Hardware Detector - Single source of truth for hardware detection
"""

import psutil
import platform
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class OSType(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"

class GPUTier(Enum):
    INTEGRATED = "integrated"
    LOW_END = "low_end"
    MID_RANGE = "mid_range"
    HIGH_END = "high_end"
    ULTRA = "ultra"
    EXTREME = "extreme"

class MemoryTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class OptimizationMode(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MAXIMUM = "maximum"

@dataclass
class GPUInfo:
    name: str = ""
    memory_gb: float = 0.0
    compute_capability: str = ""
    driver_version: str = ""
    tier: GPUTier = GPUTier.INTEGRATED
    cuda_cores: int = 0
    memory_bandwidth_gbps: float = 0.0
    boost_clock_mhz: float = 0.0

@dataclass
class SystemInfo:
    """Unified system information structure."""
    platform: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_frequency_mhz: float = 0.0
    ram_gb: float = 0.0
    ram_tier: MemoryTier = MemoryTier.LOW
    storage_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
    memory_pools: Dict[str, Any] = field(default_factory=dict)
    
    # Additional attributes for compatibility
    gpu_name: str = ""
    cuda_version: str = ""
    vram_gb: float = 0.0

class UnifiedHardwareDetector:
    """Unified hardware detector for all Schwabot components."""
    
    def __init__(self):
        self._system_info = None
        logger.info("🔍 Unified Hardware Detector initialized")
    
    def detect_hardware(self) -> SystemInfo:
        """Detect hardware capabilities."""
        if self._system_info is None:
            try:
                logger.info("🔍 Detecting hardware capabilities...")
                
                # Detect CPU
                cpu_cores = psutil.cpu_count(logical=True)
                cpu_freq = psutil.cpu_freq()
                cpu_frequency_mhz = cpu_freq.current if cpu_freq else 0.0
                
                # Detect RAM
                memory = psutil.virtual_memory()
                ram_gb = memory.total / (1024**3)
                
                # Determine RAM tier
                if ram_gb < 8:
                    ram_tier = MemoryTier.LOW
                elif ram_gb < 16:
                    ram_tier = MemoryTier.MEDIUM
                elif ram_gb < 32:
                    ram_tier = MemoryTier.HIGH
                else:
                    ram_tier = MemoryTier.ULTRA
                
                # Detect GPU (simplified)
                gpu_info = GPUInfo()
                try:
                    # Try to detect GPU (this is simplified)
                    gpu_info.name = "Unknown GPU"
                    gpu_info.memory_gb = 0.0
                    gpu_info.compute_capability = ""
                except:
                    pass
                
                # Determine optimization mode
                if ram_gb >= 32 and cpu_cores >= 16:
                    optimization_mode = OptimizationMode.MAXIMUM
                elif ram_gb >= 16 and cpu_cores >= 8:
                    optimization_mode = OptimizationMode.PERFORMANCE
                elif ram_gb >= 8 and cpu_cores >= 4:
                    optimization_mode = OptimizationMode.BALANCED
                else:
                    optimization_mode = OptimizationMode.CONSERVATIVE
                
                # Generate memory pools
                memory_pools = self._generate_memory_pools(ram_gb, cpu_cores)
                
                self._system_info = SystemInfo(
                    platform=platform.platform(),
                    cpu_model=platform.processor(),
                    cpu_cores=cpu_cores,
                    cpu_frequency_mhz=cpu_frequency_mhz,
                    ram_gb=ram_gb,
                    ram_tier=ram_tier,
                    storage_gb=100.0,  # Simplified
                    gpu=gpu_info,
                    optimization_mode=optimization_mode,
                    memory_pools=memory_pools,
                    gpu_name=gpu_info.name,
                    cuda_version=gpu_info.compute_capability,
                    vram_gb=gpu_info.memory_gb
                )
                
                logger.info(f"✅ Hardware detected: {self._system_info.platform}")
                logger.info(f"   CPU: {self._system_info.cpu_cores} cores @ {self._system_info.cpu_frequency_mhz:.0f} MHz")
                logger.info(f"   RAM: {self._system_info.ram_gb:.1f} GB ({self._system_info.ram_tier.value})")
                logger.info(f"   GPU: {self._system_info.gpu_name}")
                logger.info(f"   Optimization: {self._system_info.optimization_mode.value}")
                
            except Exception as e:
                logger.error(f"❌ Hardware detection failed: {e}")
                # Fallback to default values
                self._system_info = SystemInfo(
                    platform=platform.platform(),
                    cpu_model="Unknown",
                    cpu_cores=4,
                    cpu_frequency_mhz=2000.0,
                    ram_gb=8.0,
                    ram_tier=MemoryTier.MEDIUM,
                    storage_gb=100.0,
                    gpu=GPUInfo(),
                    optimization_mode=OptimizationMode.BALANCED,
                    memory_pools={},
                    gpu_name="Unknown",
                    cuda_version="",
                    vram_gb=0.0
                )
                logger.warning("⚠️ Using fallback hardware configuration")
        
        return self._system_info
    
    def _generate_memory_pools(self, ram_gb: float, cpu_cores: int) -> Dict[str, Any]:
        """Generate memory pool configuration."""
        if ram_gb >= 32:
            config = {
                'max_workers': 16,
                'memory_pool_size': int(ram_gb * 0.7 * 1024**3),
                'cache_size': int(ram_gb * 0.2 * 1024**3),
                'batch_size': 1000,
                'queue_size': 10000,
                'optimization_level': 'high'
            }
        elif ram_gb >= 16:
            config = {
                'max_workers': 8,
                'memory_pool_size': int(ram_gb * 0.6 * 1024**3),
                'cache_size': int(ram_gb * 0.15 * 1024**3),
                'batch_size': 500,
                'queue_size': 5000,
                'optimization_level': 'medium'
            }
        elif ram_gb >= 8:
            config = {
                'max_workers': 4,
                'memory_pool_size': int(ram_gb * 0.5 * 1024**3),
                'cache_size': int(ram_gb * 0.1 * 1024**3),
                'batch_size': 250,
                'queue_size': 2500,
                'optimization_level': 'low'
            }
        else:
            config = {
                'max_workers': 2,
                'memory_pool_size': int(ram_gb * 0.4 * 1024**3),
                'cache_size': int(ram_gb * 0.05 * 1024**3),
                'batch_size': 100,
                'queue_size': 1000,
                'optimization_level': 'minimal'
            }
        
        config.update({
            'total_memory_gb': ram_gb,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'cpu_cores': cpu_cores
        })
        
        return config
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except:
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except:
            return {
                'total_gb': 0.0,
                'available_gb': 0.0,
                'used_gb': 0.0,
                'percent': 0.0
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                'platform': platform.platform(),
                'cpu_cores': psutil.cpu_count(logical=True),
                'cpu_usage': self.get_cpu_usage(),
                'memory': self.get_memory_usage(),
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_used_gb': disk.used / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                'platform': platform.platform(),
                'cpu_cores': 4,
                'cpu_usage': 0.0,
                'memory': self.get_memory_usage(),
                'disk_total_gb': 100.0,
                'disk_free_gb': 50.0,
                'disk_used_gb': 50.0
            }
    
    def generate_memory_config(self) -> Dict[str, Any]:
        """Generate memory configuration for the system."""
        try:
            system_info = self.detect_hardware()
            memory_pools = system_info.memory_pools
            
            # Create a comprehensive memory configuration
            config = {
                'memory_pools': memory_pools,
                'tic_map_sizes': {
                    'critical': memory_pools.get('max_workers', 4) * 1000,
                    'high': memory_pools.get('max_workers', 4) * 500,
                    'medium': memory_pools.get('max_workers', 4) * 250,
                    'low': memory_pools.get('max_workers', 4) * 100,
                    'background': memory_pools.get('max_workers', 4) * 50
                },
                'cache_sizes': {
                    'price_cache': memory_pools.get('cache_size', 100 * 1024**3) // 4,
                    'signal_cache': memory_pools.get('cache_size', 100 * 1024**3) // 4,
                    'pattern_cache': memory_pools.get('cache_size', 100 * 1024**3) // 4,
                    'ai_cache': memory_pools.get('cache_size', 100 * 1024**3) // 4
                },
                'bit_depth_limits': {
                    'conservative': {'max_bits': 32, 'max_tensors': 100},
                    'balanced': {'max_bits': 64, 'max_tensors': 500},
                    'performance': {'max_bits': 128, 'max_tensors': 1000},
                    'maximum': {'max_bits': 256, 'max_tensors': 2000}
                },
                'optimization_settings': {
                    'level': system_info.optimization_mode.value,
                    'max_concurrent_trades': memory_pools.get('max_workers', 4) * 10,
                    'max_charts_per_device': memory_pools.get('max_workers', 4) * 5,
                    'data_processing_latency_ms': 50 if system_info.optimization_mode.value in ['performance', 'maximum'] else 100
                }
            }
            
            logger.info(f"✅ Generated memory configuration for {system_info.optimization_mode.value} optimization")
            return config
            
        except Exception as e:
            logger.error(f"❌ Error generating memory config: {e}")
            # Return fallback configuration
            return {
                'memory_pools': {},
                'tic_map_sizes': {'critical': 1000, 'high': 500, 'medium': 250, 'low': 100, 'background': 50},
                'cache_sizes': {'price_cache': 100 * 1024**3, 'signal_cache': 100 * 1024**3, 'pattern_cache': 100 * 1024**3, 'ai_cache': 100 * 1024**3},
                'bit_depth_limits': {'balanced': {'max_bits': 64, 'max_tensors': 500}},
                'optimization_settings': {'level': 'balanced', 'max_concurrent_trades': 40, 'max_charts_per_device': 20, 'data_processing_latency_ms': 100}
            }

# Global instance
unified_hardware_detector = UnifiedHardwareDetector()

# For backward compatibility
HardwareAutoDetector = UnifiedHardwareDetector 