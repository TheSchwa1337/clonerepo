#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 SYSTEM HEALTH MONITOR - SYSTEM RELIABILITY MODULE
===================================================

Provides real-time monitoring and health diagnostics for the Schwabot trading system.

Features:
- Real-time system performance monitoring (CPU, memory, disk, network)
- Trading system and tensor system status
- Health diagnostics and recommendations
- System health reporting
- Cross-platform compatibility
"""

import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

# Optional: Import core components if available
try:
    from advanced_tensor_algebra import AdvancedTensorAlgebra
    from unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
    from core.trading_pipeline_manager import TradingPipelineManager
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False

class SystemHealthMonitor:
    """System Health Monitor for Schwabot trading system."""
    def __init__(self):
        self.tensor_algebra = AdvancedTensorAlgebra() if CORE_COMPONENTS_AVAILABLE else None
        self.profit_system = UnifiedProfitVectorizationSystem() if CORE_COMPONENTS_AVAILABLE else None
        self.trading_manager = TradingPipelineManager() if CORE_COMPONENTS_AVAILABLE else None
        self.start_time = time.time()

    def get_system_info(self) -> Dict[str, str]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "platform": sys.platform,
        }

    def get_cpu_info(self) -> Dict[str, Any]:
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": cpu_freq.current if cpu_freq else 0,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_load": psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0],
            }
        except Exception:
            return {"cpu_count": 0, "cpu_freq": 0, "cpu_percent": 0, "cpu_load": [0, 0, 0]}

    def get_memory_info(self) -> Dict[str, float]:
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total / (1024**3),
                "available": memory.available / (1024**3),
                "used": memory.used / (1024**3),
                "percent": memory.percent,
            }
        except Exception:
            return {"total": 0, "available": 0, "used": 0, "percent": 0}

    def get_disk_info(self) -> Dict[str, float]:
        try:
            disk = psutil.disk_usage("/")
            return {
                "total": disk.total / (1024**3),
                "used": disk.used / (1024**3),
                "free": disk.free / (1024**3),
                "percent": (disk.used / disk.total) * 100,
            }
        except Exception:
            return {"total": 0, "used": 0, "free": 0, "percent": 0}

    def get_network_info(self) -> Dict[str, Any]:
        try:
            network = psutil.net_io_counters()
            return {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        except Exception:
            return {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_health_status(self) -> Dict[str, str]:
        """Return health status for CPU, memory, disk."""
        cpu = self.get_cpu_info()
        memory = self.get_memory_info()
        disk = self.get_disk_info()
        health = {}
        # CPU
        if cpu["cpu_percent"] > 90:
            health["cpu"] = "CRITICAL"
        elif cpu["cpu_percent"] > 70:
            health["cpu"] = "WARNING"
        else:
            health["cpu"] = "HEALTHY"
        # Memory
        if memory["percent"] > 90:
            health["memory"] = "CRITICAL"
        elif memory["percent"] > 80:
            health["memory"] = "WARNING"
        else:
            health["memory"] = "HEALTHY"
        # Disk
        if disk["percent"] > 90:
            health["disk"] = "CRITICAL"
        elif disk["percent"] > 80:
            health["disk"] = "WARNING"
        else:
            health["disk"] = "HEALTHY"
        return health

    def get_overall_health(self) -> str:
        health = self.get_health_status()
        critical = sum(1 for v in health.values() if v == "CRITICAL")
        warning = sum(1 for v in health.values() if v == "WARNING")
        if critical > 0:
            return "CRITICAL"
        elif warning > 0:
            return "WARNING"
        else:
            return "HEALTHY"

    def get_recommendations(self) -> list:
        health = self.get_health_status()
        recs = []
        if health["cpu"] != "HEALTHY":
            recs.append("Consider reducing CPU load or upgrading hardware.")
        if health["memory"] != "HEALTHY":
            recs.append("Consider clearing caches or adding more RAM.")
        if health["disk"] != "HEALTHY":
            recs.append("Consider cleaning up disk space.")
        return recs

    def get_trading_system_status(self) -> Optional[Dict[str, Any]]:
        if self.trading_manager:
            return self.trading_manager.get_system_status()
        return None

    def get_tensor_system_status(self) -> Optional[Dict[str, Any]]:
        if self.tensor_algebra:
            return self.tensor_algebra.get_system_status()
        return None

    def get_profit_system_status(self) -> Optional[Dict[str, Any]]:
        if self.profit_system:
            return self.profit_system.get_system_status()
        return None

    def get_full_report(self) -> Dict[str, Any]:
        """Return a comprehensive system health report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "cpu_info": self.get_cpu_info(),
            "memory_info": self.get_memory_info(),
            "disk_info": self.get_disk_info(),
            "network_info": self.get_network_info(),
            "uptime": self.get_uptime(),
            "health_status": self.get_health_status(),
            "overall_health": self.get_overall_health(),
            "recommendations": self.get_recommendations(),
            "trading_system_status": self.get_trading_system_status(),
            "tensor_system_status": self.get_tensor_system_status(),
            "profit_system_status": self.get_profit_system_status(),
        }

# Global instance
system_health_monitor = SystemHealthMonitor() 