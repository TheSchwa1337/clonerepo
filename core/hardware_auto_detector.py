"""
Hardware Auto Detector Module

This module provides hardware detection and system information gathering
for the quantum auto scaling system.
"""

import psutil
import platform
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
    """System type enumeration."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class OptimizationMode(Enum):
    """Optimization mode enumeration."""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVING = "power_saving"


class RAMTier(Enum):
    """RAM tier enumeration."""
    LOW = "low"      # < 8GB
    MEDIUM = "medium"  # 8-16GB
    HIGH = "high"    # 16-32GB
    EXTREME = "extreme"  # > 32GB


@dataclass
class SystemInfo:
    """System information data class."""
    system_type: SystemType
    cpu_count: int
    cpu_freq: float
    memory_total: int
    memory_available: int
    disk_total: int
    disk_free: int
    python_version: str
    platform: str


@dataclass
class HardwareInfo:
    """Hardware information data class."""
    platform: str
    ram_gb: float
    ram_tier: RAMTier
    optimization_mode: OptimizationMode
    cpu_count: int
    cpu_freq_mhz: float
    system_type: str


@dataclass
class MemoryConfig:
    """Memory configuration data class."""
    total_memory: int
    available_memory: int
    memory_usage_percent: float
    swap_total: int
    swap_used: int
    swap_free: int


class HardwareAutoDetector:
    """Hardware auto detection and system information gathering."""
    
    def __init__(self):
        """Initialize the hardware auto detector."""
        self._system_info = None
        self._memory_config = None
        self._hardware_info = None
    
    def detect_system_type(self) -> SystemType:
        """Detect the operating system type."""
        system = platform.system().lower()
        if system == "windows":
            return SystemType.WINDOWS
        elif system == "linux":
            return SystemType.LINUX
        elif system == "darwin":
            return SystemType.MACOS
        else:
            return SystemType.UNKNOWN
    
    def detect_hardware(self) -> HardwareInfo:
        """Detect hardware capabilities and return hardware info."""
        if self._hardware_info is None:
            # Get system information
            system_info = self.get_system_info()
            
            # Calculate RAM in GB
            ram_gb = system_info.memory_total / (1024**3)
            
            # Determine RAM tier
            if ram_gb < 8:
                ram_tier = RAMTier.LOW
            elif ram_gb < 16:
                ram_tier = RAMTier.MEDIUM
            elif ram_gb < 32:
                ram_tier = RAMTier.HIGH
            else:
                ram_tier = RAMTier.EXTREME
            
            # Determine optimization mode based on hardware
            if ram_gb >= 16 and system_info.cpu_count >= 8:
                optimization_mode = OptimizationMode.PERFORMANCE
            elif ram_gb >= 8 and system_info.cpu_count >= 4:
                optimization_mode = OptimizationMode.BALANCED
            else:
                optimization_mode = OptimizationMode.POWER_SAVING
            
            self._hardware_info = HardwareInfo(
                platform=platform.platform(),
                ram_gb=ram_gb,
                ram_tier=ram_tier,
                optimization_mode=optimization_mode,
                cpu_count=system_info.cpu_count,
                cpu_freq_mhz=system_info.cpu_freq,
                system_type=system_info.system_type.value
            )
        
        return self._hardware_info
    
    def generate_memory_config(self) -> Dict[str, Any]:
        """Generate memory configuration for the system."""
        memory_config = self.get_memory_config()
        hardware_info = self.detect_hardware()
        
        # Calculate optimal memory settings based on available RAM
        ram_gb = hardware_info.ram_gb
        
        if ram_gb >= 32:
            # High-end system
            config = {
                'max_workers': 16,
                'memory_pool_size': int(ram_gb * 0.7 * 1024**3),  # 70% of RAM
                'cache_size': int(ram_gb * 0.2 * 1024**3),  # 20% of RAM
                'batch_size': 1000,
                'queue_size': 10000,
                'optimization_level': 'high'
            }
        elif ram_gb >= 16:
            # Mid-range system
            config = {
                'max_workers': 8,
                'memory_pool_size': int(ram_gb * 0.6 * 1024**3),  # 60% of RAM
                'cache_size': int(ram_gb * 0.15 * 1024**3),  # 15% of RAM
                'batch_size': 500,
                'queue_size': 5000,
                'optimization_level': 'medium'
            }
        elif ram_gb >= 8:
            # Low-end system
            config = {
                'max_workers': 4,
                'memory_pool_size': int(ram_gb * 0.5 * 1024**3),  # 50% of RAM
                'cache_size': int(ram_gb * 0.1 * 1024**3),  # 10% of RAM
                'batch_size': 250,
                'queue_size': 2500,
                'optimization_level': 'low'
            }
        else:
            # Minimal system
            config = {
                'max_workers': 2,
                'memory_pool_size': int(ram_gb * 0.4 * 1024**3),  # 40% of RAM
                'cache_size': int(ram_gb * 0.05 * 1024**3),  # 5% of RAM
                'batch_size': 100,
                'queue_size': 1000,
                'optimization_level': 'minimal'
            }
        
        # Add current memory status
        config.update({
            'total_memory_gb': ram_gb,
            'available_memory_gb': memory_config.available_memory / (1024**3),
            'memory_usage_percent': memory_config.memory_usage_percent,
            'ram_tier': hardware_info.ram_tier.value,
            'optimization_mode': hardware_info.optimization_mode.value
        })
        
        return config
    
    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        if self._system_info is None:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self._system_info = SystemInfo(
                system_type=self.detect_system_type(),
                cpu_count=psutil.cpu_count(),
                cpu_freq=cpu_freq.current if cpu_freq else 0.0,
                memory_total=memory.total,
                memory_available=memory.available,
                disk_total=disk.total,
                disk_free=disk.free,
                python_version=platform.python_version(),
                platform=platform.platform()
            )
        
        return self._system_info
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration information."""
        if self._memory_config is None:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self._memory_config = MemoryConfig(
                total_memory=memory.total,
                available_memory=memory.available,
                memory_usage_percent=memory.percent,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_free=swap.free
            )
        
        return self._memory_config
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def get_disk_usage(self) -> float:
        """Get current disk usage percentage."""
        return psutil.disk_usage('/').percent
    
    def get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        return {
            'pid': process.pid,
            'name': process.name(),
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_info': {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms
            },
            'num_threads': process.num_threads(),
            'status': process.status()
        }
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy based on resource usage."""
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        disk_usage = self.get_disk_usage()
        
        # Consider system healthy if:
        # - CPU usage < 90%
        # - Memory usage < 90%
        # - Disk usage < 95%
        return (cpu_usage < 90 and 
                memory_usage < 90 and 
                disk_usage < 95)
    
    def get_system_load(self) -> Dict[str, float]:
        """Get system load information."""
        return {
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage(),
            'disk_usage': self.get_disk_usage(),
            'is_healthy': self.is_system_healthy()
        }
    
    def get_detailed_system_report(self) -> Dict[str, Any]:
        """Get a detailed system report."""
        system_info = self.get_system_info()
        memory_config = self.get_memory_config()
        system_load = self.get_system_load()
        process_info = self.get_process_info()
        network_io = self.get_network_io()
        
        return {
            'system_info': {
                'system_type': system_info.system_type.value,
                'cpu_count': system_info.cpu_count,
                'cpu_freq_mhz': system_info.cpu_freq,
                'python_version': system_info.python_version,
                'platform': system_info.platform
            },
            'memory_config': {
                'total_memory_gb': memory_config.total_memory / (1024**3),
                'available_memory_gb': memory_config.available_memory / (1024**3),
                'memory_usage_percent': memory_config.memory_usage_percent,
                'swap_total_gb': memory_config.swap_total / (1024**3),
                'swap_used_gb': memory_config.swap_used / (1024**3)
            },
            'system_load': system_load,
            'process_info': process_info,
            'network_io': network_io,
            'timestamp': psutil.boot_time()
        } 