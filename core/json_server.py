#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Server for Schwabot Trading System
=======================================

Secure JSON communication server with hardware-optimized performance
and intelligent packet processing. This system provides real-time
communication capabilities with priority-based packet handling and
advanced encryption for secure data transmission.

Features:
- Hardware-optimized network communication and packet processing
- Priority-based queuing system for different packet types
- Intelligent packet validation and routing
- Real-time message handling and response generation
- Integration with existing mathematical framework
- Secure data transmission with Alpha256 encryption
"""

import asyncio
import json
import logging
import threading
import time
import socket
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
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

class PacketPriority(Enum):
    """Packet processing priority levels."""
    CRITICAL = "critical"      # System commands, emergency signals
    HIGH = "high"             # Trading signals, real-time data
    MEDIUM = "medium"         # Status updates, configuration
    LOW = "low"               # Logging, monitoring data
    BACKGROUND = "background" # Analysis results, reports

@dataclass
class JSONPacket:
    """JSON packet structure."""
    timestamp: float
    packet_type: str
    data: Any
    priority: PacketPriority = PacketPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: str = ""
    encrypted: bool = False
    source: str = ""
    destination: str = ""

class JSONServer:
    """Secure JSON communication server."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, max_connections: int = 100):
        """Initialize JSON server with hardware auto-detection."""
        self.host = host
        self.port = port
        self.max_connections = max_connections
        
        self.hardware_detector = HardwareAutoDetector()
        self.hash_config = HashConfigManager()
        self.alpha256 = Alpha256Encryption()
        
        # Hardware-aware configuration
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # Server state
        self.server = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.packet_queues: Dict[PacketPriority, deque] = {
            priority: deque(maxlen=self._get_queue_size(priority))
            for priority in PacketPriority
        }
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.stats = {
            "packets_received": 0,
            "packets_sent": 0,
            "packets_dropped": 0,
            "connections_active": 0,
            "processing_time_ms": 0.0,
            "encryption_time_ms": 0.0
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
            logger.info("Initializing JSON server with hardware auto-detection...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"Hardware detected: {self.system_info.platform}")
            logger.info(f"RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"CPU: {self.system_info.cpu_count} cores")
            logger.info(f"Optimization: {self.system_info.optimization_mode.value}")
            
            # Load or create configuration
            self._load_configuration()
            
            logger.info("JSON server initialized with hardware optimization")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _get_queue_size(self, priority: PacketPriority) -> int:
        """Get queue size based on priority and hardware capabilities."""
        if not self.memory_config:
            return 1000  # Default fallback
            
        base_sizes = {
            PacketPriority.CRITICAL: 50,
            PacketPriority.HIGH: 200,
            PacketPriority.MEDIUM: 500,
            PacketPriority.LOW: 1000,
            PacketPriority.BACKGROUND: 2000
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
        """Load or create JSON server configuration."""
        config_path = Path("config/json_server_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Loaded existing JSON server configuration")
            else:
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info("✅ Created new JSON server configuration")
                
            # Apply configuration
            self._apply_configuration(config)
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self._apply_configuration(self._create_default_config())
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "version": "1.0.0",
            "system_name": "Schwabot JSON Server",
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "cpu_count": self.system_info.cpu_count if self.system_info else 4,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "server": {
                "host": "localhost",
                "port": 5000,
                "max_connections": 100,
                "max_worker_threads": min(self.system_info.cpu_count, 8) if self.system_info else 4,
                "enable_ssl": False,
                "ssl_cert_path": "",
                "ssl_key_path": "",
                "enable_cors": True,
                "cors_origins": ["*"],
                "enable_rate_limiting": True,
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "enable_request_logging": True,
                "enable_error_logging": True,
                "enable_performance_logging": False,
                "log_level": "INFO",
                "enable_compression": True,
                "enable_encryption": True,
                "max_request_size_mb": 10,
                "request_timeout_seconds": 30,
                "enable_keep_alive": True,
                "keep_alive_timeout": 5,
                "enable_gzip": True,
                "enable_json_validation": True,
                "enable_schema_validation": False,
                "enable_api_documentation": True,
                "docs_path": "/docs",
                "enable_health_check": True,
                "health_check_path": "/health",
                "enable_metrics": True,
                "metrics_path": "/metrics"
            },
            "api": {
                "enable_authentication": False,
                "auth_type": "none",
                "api_key_header": "X-API-Key",
                "enable_rate_limiting": True,
                "enable_request_validation": True,
                "enable_response_validation": True,
                "enable_error_handling": True,
                "enable_logging": True,
                "enable_monitoring": True,
                "enable_caching": True,
                "cache_ttl_seconds": 300,
                "enable_compression": True,
                "enable_encryption": True,
                "max_payload_size_mb": 5,
                "enable_batch_processing": True,
                "batch_size": 100,
                "enable_async_processing": True,
                "async_worker_threads": min(self.system_info.cpu_count, 4) if self.system_info else 2,
                "enable_background_tasks": True,
                "background_task_timeout": 300,
                "enable_task_queue": True,
                "task_queue_size": 1000,
                "enable_priority_queuing": True,
                "enable_task_scheduling": True,
                "enable_task_monitoring": True,
                "enable_task_retry": True,
                "max_retry_attempts": 3,
                "retry_delay_seconds": 5
            },
            "security": {
                "enable_encryption": True,
                "encryption_algorithm": "AES-256",
                "enable_ssl": False,
                "enable_certificate_validation": True,
                "enable_request_signing": False,
                "enable_response_signing": False,
                "enable_audit_logging": True,
                "enable_access_control": False,
                "enable_ip_whitelist": False,
                "allowed_ips": [],
                "enable_request_filtering": True,
                "blocked_patterns": [],
                "enable_dos_protection": True,
                "max_requests_per_ip": 1000,
                "enable_session_management": False,
                "session_timeout_minutes": 30,
                "enable_csrf_protection": False,
                "enable_xss_protection": True,
                "enable_sql_injection_protection": True,
                "enable_path_traversal_protection": True
            },
            "performance": {
                "enable_connection_pooling": True,
                "connection_pool_size": 20,
                "enable_thread_pooling": True,
                "thread_pool_size": min(self.system_info.cpu_count * 2, 16) if self.system_info else 8,
                "enable_memory_pooling": True,
                "memory_pool_size_mb": 100,
                "enable_object_pooling": True,
                "object_pool_size": 1000,
                "enable_caching": True,
                "cache_size_mb": 200,
                "enable_compression": True,
                "compression_level": 6,
                "enable_buffering": True,
                "buffer_size_kb": 64,
                "enable_batching": True,
                "batch_size": 100,
                "enable_async_io": True,
                "enable_non_blocking_io": True,
                "enable_multiplexing": True,
                "enable_load_balancing": False,
                "enable_failover": False,
                "enable_circuit_breaker": False,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout": 60,
                "enable_retry_logic": True,
                "max_retry_attempts": 3,
                "retry_delay_ms": 100,
                "enable_timeout_handling": True,
                "default_timeout_seconds": 30,
                "enable_performance_monitoring": True,
                "performance_check_interval": 60,
                "enable_resource_monitoring": True,
                "resource_check_interval": 30,
                "enable_memory_monitoring": True,
                "memory_threshold_mb": 500,
                "enable_cpu_monitoring": True,
                "cpu_threshold_percent": 80,
                "enable_disk_monitoring": True,
                "disk_threshold_percent": 90,
                "enable_network_monitoring": True,
                "network_threshold_mbps": 100
            },
            "monitoring": {
                "enable_health_monitoring": True,
                "health_check_interval_seconds": 30,
                "enable_performance_monitoring": True,
                "performance_check_interval_seconds": 60,
                "enable_resource_monitoring": True,
                "resource_check_interval_seconds": 30,
                "enable_error_monitoring": True,
                "enable_log_monitoring": True,
                "enable_metric_collection": True,
                "metric_collection_interval_seconds": 60,
                "enable_alerting": False,
                "alert_threshold": 0.9,
                "enable_reporting": True,
                "report_generation_interval_hours": 24,
                "enable_dashboard": True,
                "dashboard_port": 8080,
                "enable_api_metrics": True,
                "enable_system_metrics": True,
                "enable_custom_metrics": True,
                "enable_metric_export": False,
                "metric_export_format": "json",
                "enable_metric_storage": True,
                "metric_storage_path": "data/metrics",
                "enable_metric_retention": True,
                "metric_retention_days": 30,
                "enable_metric_aggregation": True,
                "metric_aggregation_interval_minutes": 5,
                "enable_metric_compression": True,
                "enable_metric_encryption": False,
                "enable_metric_backup": True,
                "metric_backup_interval_hours": 6
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        try:
            # Apply server settings
            server_settings = config.get("server", {})
            self.host = server_settings.get("host", "localhost")
            self.port = server_settings.get("port", 5000)
            self.max_connections = server_settings.get("max_connections", 100)
            self.max_worker_threads = server_settings.get("max_worker_threads", 4)
            
            # Apply API settings
            api_settings = config.get("api", {})
            self.enable_authentication = api_settings.get("enable_authentication", False)
            self.auth_type = api_settings.get("auth_type", "none")
            self.api_key_header = api_settings.get("api_key_header", "X-API-Key")
            self.enable_rate_limiting = api_settings.get("enable_rate_limiting", True)
            self.enable_request_validation = api_settings.get("enable_request_validation", True)
            self.enable_response_validation = api_settings.get("enable_response_validation", True)
            self.enable_error_handling = api_settings.get("enable_error_handling", True)
            self.enable_logging = api_settings.get("enable_logging", True)
            self.enable_monitoring = api_settings.get("enable_monitoring", True)
            self.enable_caching = api_settings.get("enable_caching", True)
            self.cache_ttl_seconds = api_settings.get("cache_ttl_seconds", 300)
            self.enable_compression = api_settings.get("enable_compression", True)
            self.enable_encryption = api_settings.get("enable_encryption", True)
            self.max_payload_size_mb = api_settings.get("max_payload_size_mb", 5)
            self.enable_batch_processing = api_settings.get("enable_batch_processing", True)
            self.batch_size = api_settings.get("batch_size", 100)
            self.enable_async_processing = api_settings.get("enable_async_processing", True)
            self.async_worker_threads = api_settings.get("async_worker_threads", 2)
            self.enable_background_tasks = api_settings.get("enable_background_tasks", True)
            self.background_task_timeout = api_settings.get("background_task_timeout", 300)
            self.enable_task_queue = api_settings.get("enable_task_queue", True)
            self.task_queue_size = api_settings.get("task_queue_size", 1000)
            self.enable_priority_queuing = api_settings.get("enable_priority_queuing", True)
            self.enable_task_scheduling = api_settings.get("enable_task_scheduling", True)
            self.enable_task_monitoring = api_settings.get("enable_task_monitoring", True)
            self.enable_task_retry = api_settings.get("enable_task_retry", True)
            self.max_retry_attempts = api_settings.get("max_retry_attempts", 3)
            self.retry_delay_seconds = api_settings.get("retry_delay_seconds", 5)
            
            # Apply security settings
            security_settings = config.get("security", {})
            self.enable_ssl = security_settings.get("enable_ssl", False)
            self.ssl_cert_path = security_settings.get("ssl_cert_path", "")
            self.ssl_key_path = security_settings.get("ssl_key_path", "")
            self.enable_certificate_validation = security_settings.get("enable_certificate_validation", True)
            self.enable_request_signing = security_settings.get("enable_request_signing", False)
            self.enable_response_signing = security_settings.get("enable_response_signing", False)
            self.enable_audit_logging = security_settings.get("enable_audit_logging", True)
            self.enable_access_control = security_settings.get("enable_access_control", False)
            self.enable_ip_whitelist = security_settings.get("enable_ip_whitelist", False)
            self.allowed_ips = security_settings.get("allowed_ips", [])
            self.enable_request_filtering = security_settings.get("enable_request_filtering", True)
            self.blocked_patterns = security_settings.get("blocked_patterns", [])
            self.enable_dos_protection = security_settings.get("enable_dos_protection", True)
            self.max_requests_per_ip = security_settings.get("max_requests_per_ip", 1000)
            self.enable_session_management = security_settings.get("enable_session_management", False)
            self.session_timeout_minutes = security_settings.get("session_timeout_minutes", 30)
            self.enable_csrf_protection = security_settings.get("enable_csrf_protection", False)
            self.enable_xss_protection = security_settings.get("enable_xss_protection", True)
            self.enable_sql_injection_protection = security_settings.get("enable_sql_injection_protection", True)
            self.enable_path_traversal_protection = security_settings.get("enable_path_traversal_protection", True)
            
            # Apply performance settings
            performance_settings = config.get("performance", {})
            self.enable_connection_pooling = performance_settings.get("enable_connection_pooling", True)
            self.connection_pool_size = performance_settings.get("connection_pool_size", 20)
            self.enable_thread_pooling = performance_settings.get("enable_thread_pooling", True)
            self.thread_pool_size = performance_settings.get("thread_pool_size", 8)
            self.enable_memory_pooling = performance_settings.get("enable_memory_pooling", True)
            self.memory_pool_size_mb = performance_settings.get("memory_pool_size_mb", 100)
            self.enable_object_pooling = performance_settings.get("enable_object_pooling", True)
            self.object_pool_size = performance_settings.get("object_pool_size", 1000)
            self.enable_caching = performance_settings.get("enable_caching", True)
            self.cache_size_mb = performance_settings.get("cache_size_mb", 200)
            self.enable_compression = performance_settings.get("enable_compression", True)
            self.compression_level = performance_settings.get("compression_level", 6)
            self.enable_buffering = performance_settings.get("enable_buffering", True)
            self.buffer_size_kb = performance_settings.get("buffer_size_kb", 64)
            self.enable_batching = performance_settings.get("enable_batching", True)
            self.enable_async_io = performance_settings.get("enable_async_io", True)
            self.enable_non_blocking_io = performance_settings.get("enable_non_blocking_io", True)
            self.enable_multiplexing = performance_settings.get("enable_multiplexing", True)
            self.enable_load_balancing = performance_settings.get("enable_load_balancing", False)
            self.enable_failover = performance_settings.get("enable_failover", False)
            self.enable_circuit_breaker = performance_settings.get("enable_circuit_breaker", False)
            self.circuit_breaker_threshold = performance_settings.get("circuit_breaker_threshold", 5)
            self.circuit_breaker_timeout = performance_settings.get("circuit_breaker_timeout", 60)
            self.enable_retry_logic = performance_settings.get("enable_retry_logic", True)
            self.max_retry_attempts = performance_settings.get("max_retry_attempts", 3)
            self.retry_delay_ms = performance_settings.get("retry_delay_ms", 100)
            self.enable_timeout_handling = performance_settings.get("enable_timeout_handling", True)
            self.default_timeout_seconds = performance_settings.get("default_timeout_seconds", 30)
            self.enable_performance_monitoring = performance_settings.get("enable_performance_monitoring", True)
            self.performance_check_interval = performance_settings.get("performance_check_interval", 60)
            self.enable_resource_monitoring = performance_settings.get("enable_resource_monitoring", True)
            self.resource_check_interval = performance_settings.get("resource_check_interval", 30)
            self.enable_memory_monitoring = performance_settings.get("enable_memory_monitoring", True)
            self.memory_threshold_mb = performance_settings.get("memory_threshold_mb", 500)
            self.enable_cpu_monitoring = performance_settings.get("enable_cpu_monitoring", True)
            self.cpu_threshold_percent = performance_settings.get("cpu_threshold_percent", 80)
            self.enable_disk_monitoring = performance_settings.get("enable_disk_monitoring", True)
            self.disk_threshold_percent = performance_settings.get("disk_threshold_percent", 90)
            self.enable_network_monitoring = performance_settings.get("enable_network_monitoring", True)
            self.network_threshold_mbps = performance_settings.get("network_threshold_mbps", 100)
            
            # Apply monitoring settings
            monitoring_settings = config.get("monitoring", {})
            self.enable_health_monitoring = monitoring_settings.get("enable_health_monitoring", True)
            self.health_check_interval_seconds = monitoring_settings.get("health_check_interval_seconds", 30)
            self.enable_performance_monitoring = monitoring_settings.get("enable_performance_monitoring", True)
            self.performance_check_interval_seconds = monitoring_settings.get("performance_check_interval_seconds", 60)
            self.enable_resource_monitoring = monitoring_settings.get("enable_resource_monitoring", True)
            self.resource_check_interval_seconds = monitoring_settings.get("resource_check_interval_seconds", 30)
            self.enable_error_monitoring = monitoring_settings.get("enable_error_monitoring", True)
            self.enable_log_monitoring = monitoring_settings.get("enable_log_monitoring", True)
            self.enable_metric_collection = monitoring_settings.get("enable_metric_collection", True)
            self.metric_collection_interval_seconds = monitoring_settings.get("metric_collection_interval_seconds", 60)
            self.enable_alerting = monitoring_settings.get("enable_alerting", False)
            self.alert_threshold = monitoring_settings.get("alert_threshold", 0.9)
            self.enable_reporting = monitoring_settings.get("enable_reporting", True)
            self.report_generation_interval_hours = monitoring_settings.get("report_generation_interval_hours", 24)
            self.enable_dashboard = monitoring_settings.get("enable_dashboard", True)
            self.dashboard_port = monitoring_settings.get("dashboard_port", 8080)
            self.enable_api_metrics = monitoring_settings.get("enable_api_metrics", True)
            self.enable_system_metrics = monitoring_settings.get("enable_system_metrics", True)
            self.enable_custom_metrics = monitoring_settings.get("enable_custom_metrics", True)
            self.enable_metric_export = monitoring_settings.get("enable_metric_export", False)
            self.metric_export_format = monitoring_settings.get("metric_export_format", "json")
            self.enable_metric_storage = monitoring_settings.get("enable_metric_storage", True)
            self.metric_storage_path = monitoring_settings.get("metric_storage_path", "data/metrics")
            self.enable_metric_retention = monitoring_settings.get("enable_metric_retention", True)
            self.metric_retention_days = monitoring_settings.get("metric_retention_days", 30)
            self.enable_metric_aggregation = monitoring_settings.get("enable_metric_aggregation", True)
            self.metric_aggregation_interval_minutes = monitoring_settings.get("metric_aggregation_interval_minutes", 5)
            self.enable_metric_compression = monitoring_settings.get("enable_metric_compression", True)
            self.enable_metric_encryption = monitoring_settings.get("enable_metric_encryption", False)
            self.enable_metric_backup = monitoring_settings.get("enable_metric_backup", True)
            self.metric_backup_interval_hours = monitoring_settings.get("metric_backup_interval_hours", 6)
            
            logger.info("Configuration applied successfully")
            
        except Exception as e:
            logger.error(f"Configuration application failed: {e}")
            raise
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            config_path = Path("config/json_server_config.json")
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
        
        logger.info("JSON server initialized with fallback configuration")
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        try:
            self.register_handler("ping", self._handle_ping)
            self.register_handler("status", self._handle_status)
            self.register_handler("stats", self._handle_stats)
            self.register_handler("config", self._handle_config)
            
            logger.info("✅ Registered default message handlers")
            
        except Exception as e:
            logger.error(f"❌ Failed to register default handlers: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    async def start_server(self):
        """Start the JSON server."""
        try:
            if self.running:
                logger.warning("⚠️ JSON server already running")
                return
            
            logger.info(f"🚀 Starting JSON server on {self.host}:{self.port}...")
            
            # Create server
            self.server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port,
                limit=1024 * 1024,  # 1MB limit
                backlog=self.max_connections
            )
            
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info(f"✅ JSON server started on {self.host}:{self.port}")
            
            # Keep server running
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            logger.error(f"❌ Failed to start JSON server: {e}")
    
    def stop_server(self):
        """Stop the JSON server."""
        try:
            logger.info("🛑 Stopping JSON server...")
            self.running = False
            
            if self.server:
                self.server.close()
            
            # Close all client connections
            for client_id, writer in self.clients.items():
                writer.close()
            self.clients.clear()
            
            logger.info("✅ JSON server stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop JSON server: {e}")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection."""
        try:
            client_addr = writer.get_extra_info('peername')
            client_id = f"{client_addr[0]}:{client_addr[1]}"
            
            logger.info(f"🔌 Client connected: {client_id}")
            
            # Add client to active connections
            self.clients[client_id] = writer
            self.stats["connections_active"] = len(self.clients)
            
            try:
                while self.running:
                    # Read packet data
                    packet_data = await reader.read(1024 * 1024)  # 1MB max
                    
                    if not packet_data:
                        break
                    
                    # Process incoming packet
                    await self._process_incoming_packet(packet_data, client_id)
                    
            except asyncio.CancelledError:
                logger.info(f"🔌 Client connection cancelled: {client_id}")
            except Exception as e:
                logger.error(f"❌ Client connection error: {client_id} - {e}")
            finally:
                # Remove client from active connections
                if client_id in self.clients:
                    del self.clients[client_id]
                    self.stats["connections_active"] = len(self.clients)
                
                writer.close()
                await writer.wait_closed()
                logger.info(f"🔌 Client disconnected: {client_id}")
                
        except Exception as e:
            logger.error(f"❌ Client handler error: {e}")
    
    async def _process_incoming_packet(self, packet_data: bytes, client_id: str):
        """Process incoming packet data."""
        try:
            start_time = time.time()
            
            # Decode packet data
            packet_str = packet_data.decode('utf-8')
            
            # Parse JSON packet
            packet_dict = json.loads(packet_str)
            
            # Create packet object
            packet = JSONPacket(
                timestamp=packet_dict.get("timestamp", time.time()),
                packet_type=packet_dict.get("packet_type", "unknown"),
                data=packet_dict.get("data", {}),
                priority=PacketPriority(packet_dict.get("priority", "medium")),
                metadata=packet_dict.get("metadata", {}),
                source=client_id,
                destination=packet_dict.get("destination", "")
            )
            
            # Generate hash ID
            packet.hash_id = self._generate_packet_hash(packet)
            
            # Validate packet
            if not self._validate_packet(packet):
                logger.warning(f"⚠️ Invalid packet from {client_id}")
                self.stats["packets_dropped"] += 1
                return
            
            # Add to appropriate queue
            with self.lock:
                queue = self.packet_queues[packet.priority]
                if len(queue) < queue.maxlen:
                    queue.append(packet)
                    self.stats["packets_received"] += 1
                    
                    # Update processing time
                    processing_time = (time.time() - start_time) * 1000
                    self.stats["processing_time_ms"] = processing_time
                else:
                    logger.warning(f"⚠️ Queue full for priority {packet.priority.value}")
                    self.stats["packets_dropped"] += 1
                    
        except Exception as e:
            logger.error(f"❌ Failed to process incoming packet: {e}")
            self.stats["packets_dropped"] += 1
    
    def _generate_packet_hash(self, packet: JSONPacket) -> str:
        """Generate unique hash for packet."""
        try:
            # Create hash from packet data
            hash_data = f"{packet.timestamp}_{packet.packet_type}_{str(packet.data)}"
            return hashlib.sha256(hash_data.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _validate_packet(self, packet: JSONPacket) -> bool:
        """Validate packet data."""
        try:
            # Check required fields
            if not packet.packet_type or not packet.data:
                return False
            
            # Check packet size
            packet_size_mb = len(str(packet.data)) / (1024 * 1024)
            if packet_size_mb > self.max_packet_size_mb:
                logger.warning(f"⚠️ Packet too large: {packet_size_mb:.2f}MB")
                return False
            
            # Check timestamp (not too old)
            if time.time() - packet.timestamp > 3600:  # 1 hour
                logger.warning("⚠️ Packet timestamp too old")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Packet validation failed: {e}")
            return False
    
    async def send_packet(self, packet: JSONPacket, client_id: Optional[str] = None) -> bool:
        """Send packet to client(s)."""
        try:
            start_time = time.time()
            
            # Prepare packet data
            packet_dict = {
                "timestamp": packet.timestamp,
                "packet_type": packet.packet_type,
                "data": packet.data,
                "priority": packet.priority.value,
                "metadata": packet.metadata,
                "hash_id": packet.hash_id,
                "source": packet.source,
                "destination": packet.destination
            }
            
            # Serialize to JSON
            packet_json = json.dumps(packet_dict)
            
            # Encrypt if enabled
            if self.enable_packet_encryption:
                encrypted_data = self.alpha256.encrypt(packet_json)
                packet.encrypted = True
                packet_data = encrypted_data.encode('utf-8')
            else:
                packet_data = packet_json.encode('utf-8')
            
            # Send to specific client or all clients
            if client_id:
                if client_id in self.clients:
                    writer = self.clients[client_id]
                    writer.write(packet_data)
                    await writer.drain()
                    self.stats["packets_sent"] += 1
                    return True
                else:
                    logger.warning(f"⚠️ Client not found: {client_id}")
                    return False
            else:
                # Send to all clients
                success_count = 0
                for cid, writer in self.clients.items():
                    try:
                        writer.write(packet_data)
                        await writer.drain()
                        success_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to send to client {cid}: {e}")
                
                self.stats["packets_sent"] += success_count
                
                # Update encryption time
                encryption_time = (time.time() - start_time) * 1000
                self.stats["encryption_time_ms"] = encryption_time
                
                return success_count > 0
                
        except Exception as e:
            logger.error(f"❌ Failed to send packet: {e}")
            return False
    
    def _processing_loop(self):
        """Main processing loop for JSON server."""
        try:
            while self.running:
                # Process packet batches
                asyncio.run(self._process_packet_batch())
                
                time.sleep(0.1)  # 100ms interval
                
        except Exception as e:
            logger.error(f"❌ Processing loop error: {e}")
    
    async def _process_packet_batch(self):
        """Process a batch of packets."""
        try:
            # Process packets in priority order
            for priority in [PacketPriority.CRITICAL, PacketPriority.HIGH, PacketPriority.MEDIUM, PacketPriority.LOW, PacketPriority.BACKGROUND]:
                with self.lock:
                    queue = self.packet_queues[priority]
                    
                    while queue and len(queue) < self.batch_size:
                        packet = queue.popleft()
                        await self._handle_packet(packet)
            
        except Exception as e:
            logger.error(f"❌ Packet batch processing failed: {e}")
    
    async def _handle_packet(self, packet: JSONPacket):
        """Handle individual packet."""
        try:
            # Check if handler exists for packet type
            if packet.packet_type in self.message_handlers:
                handler = self.message_handlers[packet.packet_type]
                
                # Call handler
                if asyncio.iscoroutinefunction(handler):
                    response = await handler(packet)
                else:
                    response = handler(packet)
                
                # Send response if provided
                if response:
                    response_packet = JSONPacket(
                        timestamp=time.time(),
                        packet_type=f"{packet.packet_type}_response",
                        data=response,
                        priority=packet.priority,
                        source="server",
                        destination=packet.source
                    )
                    
                    await self.send_packet(response_packet, packet.source)
            else:
                logger.warning(f"⚠️ No handler for packet type: {packet.packet_type}")
                
        except Exception as e:
            logger.error(f"❌ Packet handling failed: {e}")
    
    async def _handle_ping(self, packet: JSONPacket) -> Dict[str, Any]:
        """Handle ping packet."""
        return {"status": "pong", "timestamp": time.time()}
    
    async def _handle_status(self, packet: JSONPacket) -> Dict[str, Any]:
        """Handle status request packet."""
        return {
            "status": "running",
            "timestamp": time.time(),
            "connections": len(self.clients),
            "queues": {
                priority.value: len(queue)
                for priority, queue in self.packet_queues.items()
            }
        }
    
    async def _handle_stats(self, packet: JSONPacket) -> Dict[str, Any]:
        """Handle statistics request packet."""
        return {
            "stats": self.stats,
            "timestamp": time.time()
        }
    
    async def _handle_config(self, packet: JSONPacket) -> Dict[str, Any]:
        """Handle configuration request packet."""
        return {
            "config": {
                "host": self.host,
                "port": self.port,
                "max_connections": self.max_connections,
                "enable_encryption": self.enable_packet_encryption,
                "enable_validation": self.enable_packet_validation
            },
            "timestamp": time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        try:
            return {
                "running": self.running,
                "stats": self.stats,
                "system_info": {
                    "platform": self.system_info.platform if self.system_info else "unknown",
                    "ram_gb": self.system_info.ram_gb if self.system_info else 0.0,
                    "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "unknown"
                },
                "network_info": {
                    "host": self.host,
                    "port": self.port,
                    "connections_active": len(self.clients),
                    "max_connections": self.max_connections
                },
                "queue_info": {
                    priority.value: len(queue)
                    for priority, queue in self.packet_queues.items()
                },
                "configuration": {
                    "enable_encryption": self.enable_packet_encryption,
                    "enable_validation": self.enable_packet_validation,
                    "max_packet_size_mb": self.max_packet_size_mb,
                    "enable_rate_limiting": self.enable_rate_limiting
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics collection failed: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for JSON server testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Testing JSON Server...")
    
    # Create JSON server instance
    json_server = JSONServer(host="localhost", port=8080)
    
    try:
        # Start the server
        await json_server.start_server()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        # Stop the server
        json_server.stop_server()
        
        logger.info("👋 JSON Server test complete")

if __name__ == "__main__":
    asyncio.run(main()) 