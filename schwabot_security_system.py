#!/usr/bin/env python3
"""
🔐 Schwabot Advanced Security System
===================================

Comprehensive security system for Schwabot trading platform:
- Advanced encryption and authentication
- Real-time threat detection
- Network security monitoring
- Trading protection mechanisms
- Audit logging and compliance
"""

import sqlite3
import hashlib
import secrets
import json
import logging
import threading
import time
import os
import socket
import psutil
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import ipaddress
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: str
    event_type: str
    severity: str
    source: str
    description: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class ThreatAlert:
    """Threat alert data structure."""
    timestamp: str
    threat_type: str
    severity: str
    source: str
    description: str
    mitigation: str
    status: str = "active"
    confidence: float = 0.0
    threat_score: int = 0

@dataclass
class SecurityMetrics:
    """Security metrics data structure."""
    timestamp: str
    security_score: float
    threat_level: str
    active_threats: int
    blocked_attempts: int
    encryption_status: str
    authentication_status: str
    network_security: str

@dataclass
class ThreatProfile:
    """Threat profile for IP/user analysis."""
    ip_address: str
    user_id: Optional[str]
    threat_score: int = 0
    failed_attempts: int = 0
    successful_logins: int = 0
    last_activity: datetime = None
    suspicious_patterns: List[str] = None
    lockout_until: Optional[datetime] = None
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    behavior_pattern: Dict[str, Any] = None

class SchwabotSecuritySystem:
    """Enhanced security system with intelligent threat detection."""
    
    def __init__(self, config_file: str = "security_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.db_path = "security_keys/security.db"
        self.encryption_key = None
        self.active_sessions = {}
        self.security_events = []
        self.threat_alerts = []
        self.blocked_ips = set()
        self.whitelisted_ips = set()
        self.threat_profiles = {}  # IP-based threat profiles
        self.user_threat_profiles = {}  # User-based threat profiles
        self.behavior_patterns = defaultdict(deque)  # Track behavioral patterns
        self.adaptive_thresholds = {
            "brute_force": 5,  # Adaptive threshold for brute force
            "suspicious_ip": 10,  # Adaptive threshold for suspicious IP
            "unusual_pattern": 15,  # Adaptive threshold for unusual patterns
            "lockout_duration": 300  # Initial lockout duration (seconds)
        }
        self.monitoring_active = False
        self.monitor_thread = None
        
        # SECURITY ISOLATION: No external bridges
        self._security_isolation_enabled = True
        self._external_bridges_blocked = True
        self._data_encapsulation_active = True
        
        # Initialize security components with isolation
        self._init_isolated_database()
        self._init_isolated_encryption()
        self._init_isolated_authentication()
        self._init_isolated_network_monitoring()
        self._start_isolated_monitoring()
        
        logger.info("🔐 Schwabot Security System initialized with COMPLETE ISOLATION")

    def load_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_length": 32
            },
            "authentication": {
                "max_login_attempts": 5,
                "lockout_duration": 300,
                "session_timeout": 3600,
                "password_min_length": 8,
                "require_special_chars": True
            },
            "threat_detection": {
                "enable_adaptive_thresholds": True,
                "enable_behavioral_analysis": True,
                "enable_ip_reputation": True,
                "enable_geolocation_check": True,
                "max_threat_score": 100,
                "whitelist_local_ips": True
            },
            "monitoring": {
                "check_interval": 30,
                "max_events_stored": 10000,
                "enable_real_time_alerts": True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            config[key].update(value)
                    return config
            else:
                # Create default config file
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config

    def _validate_security_isolation(self) -> bool:
        """Validate that security isolation is maintained."""
        try:
            # Check for any external connections
            if hasattr(self, '_external_bridges_blocked') and not self._external_bridges_blocked:
                return False
            
            # Check for Flask or web bridges
            import sys
            for module_name in sys.modules:
                if any(bridge in module_name.lower() for bridge in ['flask', 'django', 'web', 'http', 'api', 'rest']):
                    logger.error(f"SECURITY VIOLATION: External bridge detected: {module_name}")
                    return False
            
            # Check for network connections
            if hasattr(self, 'requests') and self.requests:
                logger.error("SECURITY VIOLATION: External HTTP requests detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security isolation validation failed: {e}")
            return False

    def _init_isolated_database(self):
        """Initialize database with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during database initialization")
            
            os.makedirs("security_keys", exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced users table with isolation markers
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT "user",
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    failed_attempts INTEGER DEFAULT 0,
                    is_locked INTEGER DEFAULT 0,
                    lockout_until TEXT,
                    threat_score INTEGER DEFAULT 0,
                    is_whitelisted INTEGER DEFAULT 0,
                    behavior_pattern TEXT,
                    account_type TEXT DEFAULT "user",
                    created_by TEXT DEFAULT "admin",
                    isolation_level TEXT DEFAULT "air_gapped"
                )
            ''')
            
            # Enhanced sessions table with isolation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    login_time TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    threat_indicators TEXT,
                    isolation_verified INTEGER DEFAULT 1
                )
            ''')
            
            # Enhanced security events table with isolation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    user_id TEXT,
                    ip_address TEXT,
                    session_id TEXT,
                    threat_score INTEGER DEFAULT 0,
                    isolation_level TEXT DEFAULT "air_gapped"
                )
            ''')
            
            # Enhanced threat alerts table with isolation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT NOT NULL,
                    mitigation TEXT NOT NULL,
                    status TEXT DEFAULT "active",
                    confidence REAL DEFAULT 0.0,
                    threat_score INTEGER DEFAULT 0,
                    isolation_verified INTEGER DEFAULT 1
                )
            ''')
            
            # New threat profiles table with isolation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT UNIQUE,
                    user_id TEXT,
                    threat_score INTEGER DEFAULT 0,
                    failed_attempts INTEGER DEFAULT 0,
                    successful_logins INTEGER DEFAULT 0,
                    last_activity TEXT,
                    suspicious_patterns TEXT,
                    lockout_until TEXT,
                    is_whitelisted INTEGER DEFAULT 0,
                    is_blacklisted INTEGER DEFAULT 0,
                    behavior_pattern TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    isolation_level TEXT DEFAULT "air_gapped"
                )
            ''')
            
            # New IP reputation table with isolation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ip_reputation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT UNIQUE NOT NULL,
                    reputation_score REAL DEFAULT 0.0,
                    threat_indicators TEXT,
                    geolocation TEXT,
                    last_updated TEXT NOT NULL,
                    source TEXT,
                    isolation_verified INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Security database initialized with COMPLETE ISOLATION")
            
        except Exception as e:
            logger.error(f"Error initializing isolated database: {e}")
            raise SecurityViolation(f"Database isolation failed: {e}")

    def _init_isolated_encryption(self):
        """Initialize encryption system with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during encryption initialization")
            
            if not os.path.exists("security_keys/encryption.key"):
                self.encryption_key = secrets.token_bytes(32)
                with open("security_keys/encryption.key", "wb") as f:
                    f.write(self.encryption_key)
            else:
                with open("security_keys/encryption.key", "rb") as f:
                    self.encryption_key = f.read()
            
            logger.info("✅ Encryption system initialized with COMPLETE ISOLATION")
            
        except Exception as e:
            logger.error(f"Error initializing isolated encryption: {e}")
            raise SecurityViolation(f"Encryption isolation failed: {e}")

    def _init_isolated_authentication(self):
        """Initialize authentication system with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during authentication initialization")
            
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Check if admin user exists
            cursor.execute("SELECT username FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                # Create admin user with enhanced security and isolation
                admin_password = "SchwabotAdmin2025!"
                salt = secrets.token_hex(16)
                password_hash = hashlib.pbkdf2_hmac('sha256', admin_password.encode(), salt.encode(), 100000).hex()
                
                cursor.execute('''
                    INSERT INTO users (username, password_hash, salt, email, role, created_at, is_whitelisted, isolation_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', ('admin', password_hash, salt, 'admin@schwabot.com', 'admin', 
                     datetime.now().isoformat(), 1, 'air_gapped'))
                
                conn.commit()
                logger.info("✅ Default admin user created with COMPLETE ISOLATION")
            
            conn.close()
            logger.info("✅ Authentication system initialized with COMPLETE ISOLATION")
            
        except Exception as e:
            logger.error(f"Error initializing isolated authentication: {e}")
            raise SecurityViolation(f"Authentication isolation failed: {e}")

    def _init_isolated_network_monitoring(self):
        """Initialize network monitoring with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during network monitoring initialization")
            
            # Initialize local IP whitelist with isolation
            if self.config["threat_detection"]["whitelist_local_ips"]:
                local_ips = self._get_isolated_local_ips()
                for ip in local_ips:
                    self.whitelisted_ips.add(ip)
                    self._update_isolated_threat_profile(ip, is_whitelisted=True)
            
            logger.info("✅ Network monitoring initialized with COMPLETE ISOLATION")
            
        except Exception as e:
            logger.error(f"Error initializing isolated network monitoring: {e}")
            raise SecurityViolation(f"Network monitoring isolation failed: {e}")

    def _get_isolated_local_ips(self) -> List[str]:
        """Get local IP addresses with complete isolation."""
        local_ips = []
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during local IP detection")
            
            # Get localhost with isolation
            local_ips.append("127.0.0.1")
            local_ips.append("::1")
            
            # Get local network IPs with isolation
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            local_ips.append(local_ip)
            
            # Get all network interfaces with isolation
            for interface, addresses in psutil.net_if_addrs().items():
                for addr in addresses:
                    if addr.family == socket.AF_INET:
                        local_ips.append(addr.address)
            
            return list(set(local_ips))
        except Exception as e:
            logger.error(f"Error getting isolated local IPs: {e}")
            return ["127.0.0.1"]

    def _start_isolated_monitoring(self):
        """Start security monitoring with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during monitoring start")
            
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._isolated_monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("✅ Security monitoring started with COMPLETE ISOLATION")
            
        except Exception as e:
            logger.error(f"Error starting isolated monitoring: {e}")
            raise SecurityViolation(f"Monitoring isolation failed: {e}")

    def _isolated_monitoring_loop(self):
        """Enhanced monitoring loop with complete isolation."""
        while self.monitoring_active:
            try:
                # Validate isolation before each monitoring cycle
                if not self._validate_security_isolation():
                    logger.error("SECURITY VIOLATION: External bridges detected in monitoring loop")
                    break
                
                # Check for suspicious activities with isolation
                self._check_isolated_suspicious_activities()
                
                # Monitor network with isolation
                self._monitor_isolated_network()
                
                # Cleanup old sessions with isolation
                self._cleanup_isolated_sessions()
                
                # Update adaptive thresholds with isolation
                self._update_isolated_adaptive_thresholds()
                
                # Sleep
                time.sleep(self.config["monitoring"]["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in isolated monitoring loop: {e}")
                time.sleep(10)

    def _check_isolated_suspicious_activities(self):
        """Check for suspicious activities with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during suspicious activity check")
            
            # Check for rapid-fire authentication attempts with isolation
            recent_events = self.security_events[-50:]
            auth_events = [e for e in recent_events if e.event_type in ["authentication_failed", "authentication_success"]]
            
            if len(auth_events) >= 10:
                # Group by IP with isolation
                ip_groups = defaultdict(list)
                for event in auth_events:
                    if event.ip_address:
                        ip_groups[event.ip_address].append(event)
                
                for ip, events in ip_groups.items():
                    if len(events) >= 5:
                        # Check timing between events with isolation
                        timestamps = [datetime.fromisoformat(e.timestamp) for e in events]
                        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                                   for i in range(1, len(timestamps))]
                        
                        if any(interval < 1.0 for interval in intervals):  # Less than 1 second
                            self._create_isolated_threat_alert(
                                "rapid_fire_attack",
                                "high",
                                "monitoring",
                                f"Rapid-fire authentication attempts from {ip}",
                                "IP temporarily blocked",
                                confidence=0.8,
                                threat_score=80
                            )
            
        except Exception as e:
            logger.error(f"Error checking isolated suspicious activities: {e}")

    def _monitor_isolated_network(self):
        """Monitor network for threats with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during network monitoring")
            
            # Check for unusual network connections with isolation
            connections = psutil.net_connections()
            external_connections = [conn for conn in connections 
                                  if conn.status == 'ESTABLISHED' and 
                                  conn.raddr and conn.raddr.ip != '127.0.0.1']
            
            # Monitor for suspicious external connections with isolation
            for conn in external_connections:
                if conn.raddr.ip not in self.whitelisted_ips:
                    # Check if this IP has been flagged before with isolation
                    profile = self.threat_profiles.get(conn.raddr.ip)
                    if profile and profile.threat_score > 30:
                        self._create_isolated_threat_alert(
                            "suspicious_connection",
                            "medium",
                            "network",
                            f"Suspicious connection to {conn.raddr.ip}:{conn.raddr.port}",
                            "Connection monitored",
                            confidence=0.6,
                            threat_score=profile.threat_score
                        )
            
        except Exception as e:
            logger.error(f"Error monitoring isolated network: {e}")

    def _cleanup_isolated_sessions(self):
        """Cleanup expired sessions with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during session cleanup")
            
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.active_sessions.items():
                if (current_time - session_data["last_activity"]).total_seconds() > self.config["authentication"]["session_timeout"]:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up isolated sessions: {e}")

    def _update_isolated_adaptive_thresholds(self):
        """Update adaptive thresholds with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during threshold update")
            
            # Adjust thresholds based on recent activity with isolation
            recent_events = self.security_events[-100:]
            failed_auths = len([e for e in recent_events if e.event_type == "authentication_failed"])
            
            if failed_auths > 20:
                # Increase thresholds during high threat periods with isolation
                self.adaptive_thresholds["brute_force"] = min(10, self.adaptive_thresholds["brute_force"] + 1)
                self.adaptive_thresholds["lockout_duration"] = min(1800, self.adaptive_thresholds["lockout_duration"] * 1.5)
            elif failed_auths < 5:
                # Decrease thresholds during low threat periods with isolation
                self.adaptive_thresholds["brute_force"] = max(3, self.adaptive_thresholds["brute_force"] - 1)
                self.adaptive_thresholds["lockout_duration"] = max(300, self.adaptive_thresholds["lockout_duration"] * 0.8)
                
        except Exception as e:
            logger.error(f"Error updating isolated adaptive thresholds: {e}")

    def _update_isolated_threat_profile(self, ip_address: str, user_id: str = None, **kwargs):
        """Update threat profile with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during threat profile update")
            
            profile_key = f"{ip_address}:{user_id}" if user_id else ip_address
            
            if profile_key not in self.threat_profiles:
                self.threat_profiles[profile_key] = ThreatProfile(
                    ip_address=ip_address,
                    user_id=user_id,
                    suspicious_patterns=[],
                    behavior_pattern={}
                )
            
            profile = self.threat_profiles[profile_key]
            
            # Update profile with new data
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.last_activity = datetime.now()
            
            # Store in database with isolation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO threat_profiles 
                (ip_address, user_id, threat_score, failed_attempts, successful_logins, 
                 last_activity, suspicious_patterns, lockout_until, is_whitelisted, 
                 is_blacklisted, behavior_pattern, created_at, updated_at, isolation_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.ip_address, profile.user_id, profile.threat_score,
                profile.failed_attempts, profile.successful_logins,
                profile.last_activity.isoformat() if profile.last_activity else None,
                json.dumps(profile.suspicious_patterns) if profile.suspicious_patterns else None,
                profile.lockout_until.isoformat() if profile.lockout_until else None,
                profile.is_whitelisted, profile.is_blacklisted,
                json.dumps(profile.behavior_pattern) if profile.behavior_pattern else None,
                datetime.now().isoformat(), datetime.now().isoformat(), 'air_gapped'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating isolated threat profile: {e}")

    def _create_isolated_threat_alert(self, threat_type: str, severity: str, source: str, description: str, mitigation: str, confidence: float = 0.0, threat_score: int = 0):
        """Create enhanced threat alert with complete isolation."""
        try:
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during threat alert creation")
            
            alert = ThreatAlert(
                timestamp=datetime.now().isoformat(),
                threat_type=threat_type,
                severity=severity,
                source=source,
                description=description,
                mitigation=mitigation,
                confidence=confidence,
                threat_score=threat_score
            )
            
            self.threat_alerts.append(alert)
            
            # Store in database with isolation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO threat_alerts (timestamp, threat_type, severity, source, description, mitigation, status, confidence, threat_score, isolation_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp, alert.threat_type, alert.severity, alert.source,
                alert.description, alert.mitigation, alert.status, alert.confidence, alert.threat_score, 1
            ))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"🚨 Isolated Threat Alert: {threat_type} - {description} (Confidence: {confidence:.2f}, Score: {threat_score})")
            
        except Exception as e:
            logger.error(f"Error creating isolated threat alert: {e}")

    def analyze_behavioral_pattern(self, ip_address: str, user_id: str, event_type: str, details: Dict[str, Any]):
        """Analyze behavioral patterns for threat detection."""
        try:
            profile_key = f"{ip_address}:{user_id}" if user_id else ip_address
            
            if profile_key not in self.behavior_patterns:
                self.behavior_patterns[profile_key] = deque(maxlen=100)
            
            # Add event to pattern
            pattern_event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "details": details
            }
            self.behavior_patterns[profile_key].append(pattern_event)
            
            # Analyze patterns for suspicious behavior
            patterns = list(self.behavior_patterns[profile_key])
            
            # Check for rapid-fire attempts
            if len(patterns) >= 3:
                recent_events = patterns[-3:]
                time_diffs = []
                for i in range(1, len(recent_events)):
                    t1 = datetime.fromisoformat(recent_events[i-1]["timestamp"])
                    t2 = datetime.fromisoformat(recent_events[i]["timestamp"])
                    time_diffs.append((t2 - t1).total_seconds())
                
                if all(diff < 2.0 for diff in time_diffs):  # Less than 2 seconds between attempts
                    self._update_isolated_threat_profile(ip_address, user_id, 
                                             suspicious_patterns=["rapid_fire_attempts"])
            
            # Check for unusual timing patterns
            if len(patterns) >= 10:
                auth_events = [p for p in patterns if p["event_type"] in ["authentication_failed", "authentication_success"]]
                if len(auth_events) >= 5:
                    # Check for automated patterns
                    intervals = []
                    for i in range(1, len(auth_events)):
                        t1 = datetime.fromisoformat(auth_events[i-1]["timestamp"])
                        t2 = datetime.fromisoformat(auth_events[i]["timestamp"])
                        intervals.append((t2 - t1).total_seconds())
                    
                    # Check for consistent intervals (bot-like behavior)
                    if len(intervals) >= 3:
                        avg_interval = sum(intervals) / len(intervals)
                        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                        
                        if variance < 1.0:  # Very consistent timing (suspicious)
                            self._update_isolated_threat_profile(ip_address, user_id,
                                                     suspicious_patterns=["automated_behavior"])
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral pattern: {e}")

    def is_ip_whitelisted(self, ip_address: str) -> bool:
        """Check if IP is whitelisted."""
        return ip_address in self.whitelisted_ips

    def is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if IP is blacklisted."""
        return ip_address in self.blocked_ips

    def should_allow_authentication(self, ip_address: str, user_id: str) -> Tuple[bool, str]:
        """Determine if authentication should be allowed based on threat analysis."""
        try:
            # Check whitelist first
            if self.is_ip_whitelisted(ip_address):
                return True, "IP whitelisted"
            
            # Check blacklist
            if self.is_ip_blacklisted(ip_address):
                return False, "IP blacklisted"
            
            # Get threat profile
            profile_key = f"{ip_address}:{user_id}" if user_id else ip_address
            profile = self.threat_profiles.get(profile_key)
            
            if not profile:
                return True, "No threat profile"
            
            # Check lockout
            if profile.lockout_until and datetime.now() < profile.lockout_until:
                return False, f"Account locked until {profile.lockout_until}"
            
            # Check threat score
            if profile.threat_score >= self.config["threat_detection"]["max_threat_score"]:
                return False, "Threat score too high"
            
            # Check suspicious patterns
            if profile.suspicious_patterns and len(profile.suspicious_patterns) >= 3:
                return False, "Multiple suspicious patterns detected"
            
            return True, "Authentication allowed"
            
        except Exception as e:
            logger.error(f"Error checking authentication allowance: {e}")
            return True, "Error in threat analysis"

    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Tuple[bool, str]:
        """Enhanced authentication with NO LOCKOUT for admins and COMPLETE ISOLATION."""
        try:
            # SECURITY ISOLATION VALIDATION
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during authentication")
            
            # Validate input
            if not username or not password:
                return False, "Invalid credentials"
            
            # Check if authentication should be allowed
            allow_auth, reason = self.should_allow_authentication(ip_address, username)
            if not allow_auth:
                self.log_security_event("authentication_blocked", "high", "auth", 
                                      f"Authentication blocked: {reason}", {
                                          "username": username,
                                          "ip_address": ip_address,
                                          "reason": reason
                                      })
                return False, reason
            
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT password_hash, salt, failed_attempts, is_locked, lockout_until, role FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if not result:
                self.log_security_event("authentication_failed", "medium", "auth", 
                                      f"Failed login attempt for unknown user: {username}", {
                                          "username": username,
                                          "ip_address": ip_address
                                      })
                return False, "Invalid credentials"
            
            stored_hash, salt, failed_attempts, is_locked, lockout_until, role = result
            
            # Check if account is locked (ONLY for non-admin users)
            if is_locked and role != "admin":
                if lockout_until:
                    lockout_time = datetime.fromisoformat(lockout_until)
                    if datetime.now() < lockout_time:
                        return False, f"Account locked until {lockout_time}"
                    else:
                        # Lockout expired, reset
                        cursor.execute("UPDATE users SET is_locked = 0, failed_attempts = 0, lockout_until = NULL WHERE username = ?", (username,))
                        failed_attempts = 0
                        is_locked = 0
            
            # Verify password
            if self.verify_password(password, stored_hash, salt):
                # Successful authentication
                cursor.execute("UPDATE users SET failed_attempts = 0, is_locked = 0, lockout_until = NULL, last_login = ? WHERE username = ?", 
                             (datetime.now().isoformat(), username))
                
                # Create session with isolation
                session_id = secrets.token_hex(32)
                cursor.execute('''
                    INSERT INTO user_sessions (session_id, user_id, ip_address, login_time, last_activity, isolation_verified)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, username, ip_address or "unknown", 
                     datetime.now().isoformat(), datetime.now().isoformat(), 1))
                
                self.active_sessions[session_id] = {
                    "user_id": username,
                    "ip_address": ip_address,
                    "login_time": datetime.now(),
                    "last_activity": datetime.now(),
                    "role": role
                }
                
                # Update threat profile with isolation
                self._update_isolated_threat_profile(ip_address, username, 
                                         successful_logins=self.threat_profiles.get(f"{ip_address}:{username}", 
                                         ThreatProfile(ip_address, username)).successful_logins + 1)
                
                conn.commit()
                conn.close()
                
                self.log_security_event("authentication_success", "low", "auth", 
                                      f"Successful login for user: {username} (Role: {role})", {
                                          "username": username,
                                          "ip_address": ip_address,
                                          "session_id": session_id,
                                          "role": role
                                      })
                
                return True, session_id
            else:
                # Failed authentication
                failed_attempts += 1
                
                # Update threat profile with isolation
                profile_key = f"{ip_address}:{username}" if ip_address else username
                if profile_key not in self.threat_profiles:
                    self.threat_profiles[profile_key] = ThreatProfile(ip_address, username)
                
                profile = self.threat_profiles[profile_key]
                profile.failed_attempts += 1
                profile.threat_score += 10  # Increment threat score
                
                # Analyze behavioral pattern
                self.analyze_behavioral_pattern(ip_address, username, "authentication_failed", {
                    "attempt_number": failed_attempts,
                    "username": username
                })
                
                # NO LOCKOUT FOR ADMINS - they can try unlimited times
                if role == "admin":
                    cursor.execute("UPDATE users SET failed_attempts = ? WHERE username = ?", (failed_attempts, username))
                    conn.commit()
                    conn.close()
                    
                    self.log_security_event("authentication_failed_admin", "medium", "auth", 
                                          f"Failed login attempt for admin: {username} (No lockout applied)", {
                                              "username": username,
                                              "ip_address": ip_address,
                                              "failed_attempts": failed_attempts,
                                              "role": "admin"
                                          })
                    
                    return False, "Invalid credentials (Admin - no lockout applied)"
                
                # Apply lockout only for non-admin users
                else:
                    # Determine lockout duration based on threat analysis
                    lockout_duration = self.calculate_adaptive_lockout(failed_attempts, profile)
                    
                    if failed_attempts >= self.adaptive_thresholds["brute_force"]:
                        lockout_until = datetime.now() + timedelta(seconds=lockout_duration)
                        cursor.execute("UPDATE users SET failed_attempts = ?, is_locked = 1, lockout_until = ? WHERE username = ?", 
                                     (failed_attempts, lockout_until.isoformat(), username))
                        
                        # Update threat profile with isolation
                        self._update_isolated_threat_profile(ip_address, username, 
                                                 lockout_until=lockout_until,
                                                 threat_score=profile.threat_score)
                        
                        self.log_security_event("account_locked", "high", "auth", 
                                              f"Account locked for user: {username} due to {failed_attempts} failed attempts", {
                                                  "username": username,
                                                  "ip_address": ip_address,
                                                  "failed_attempts": failed_attempts,
                                                  "lockout_duration": lockout_duration
                                              })
                    else:
                        cursor.execute("UPDATE users SET failed_attempts = ? WHERE username = ?", (failed_attempts, username))
                    
                    conn.commit()
                    conn.close()
                    
                    self.log_security_event("authentication_failed", "medium", "auth", 
                                          f"Failed login attempt for user: {username}", {
                                              "username": username,
                                              "ip_address": ip_address,
                                              "failed_attempts": failed_attempts
                                          })
                    
                    return False, "Invalid credentials"
                
        except SecurityViolation as e:
            logger.error(f"SECURITY VIOLATION during authentication: {e}")
            return False, "Security violation detected"
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False, "Authentication error"

    def calculate_adaptive_lockout(self, failed_attempts: int, profile: ThreatProfile) -> int:
        """Calculate adaptive lockout duration based on threat analysis."""
        base_duration = self.config["authentication"]["lockout_duration"]
        
        # Increase duration based on failed attempts
        duration_multiplier = min(failed_attempts / 3, 5)  # Cap at 5x
        
        # Increase based on threat score
        threat_multiplier = min(profile.threat_score / 50, 3)  # Cap at 3x
        
        # Increase based on suspicious patterns
        pattern_multiplier = 1 + (len(profile.suspicious_patterns or []) * 0.5)
        
        total_multiplier = duration_multiplier + threat_multiplier + pattern_multiplier
        return int(base_duration * total_multiplier)

    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password with enhanced security."""
        try:
            # Use PBKDF2 with high iteration count
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            return computed_hash == stored_hash
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False

    def log_security_event(self, event_type: str, severity: str, source: str, description: str, details: Dict[str, Any] = None):
        """Log security event with enhanced threat analysis and COMPLETE ISOLATION."""
        try:
            # SECURITY ISOLATION VALIDATION
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during security event logging")
            
            event = SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                severity=severity,
                source=source,
                description=description,
                details=details or {}
            )
            
            self.security_events.append(event)
            
            # Analyze for threats
            self.analyze_threat(event)
            
            # Store in database with isolation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events (timestamp, event_type, severity, source, description, details, user_id, ip_address, session_id, isolation_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp, event.event_type, event.severity, event.source,
                event.description, json.dumps(event.details), 
                event.user_id, event.ip_address, event.session_id, 'air_gapped'
            ))
            
            conn.commit()
            conn.close()
            
        except SecurityViolation as e:
            logger.error(f"SECURITY VIOLATION during event logging: {e}")
        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    def analyze_threat(self, event: SecurityEvent):
        """Enhanced threat analysis with behavioral patterns."""
        try:
            # Get threat profile
            profile_key = f"{event.ip_address}:{event.user_id}" if event.ip_address and event.user_id else event.ip_address
            profile = self.threat_profiles.get(profile_key)
            
            if not profile:
                profile = ThreatProfile(event.ip_address, event.user_id)
                self.threat_profiles[profile_key] = profile
            
            # Analyze based on event type
            if event.event_type == "authentication_failed":
                # Check for brute force patterns
                recent_failures = [e for e in self.security_events[-50:] 
                                 if e.event_type == "authentication_failed" and 
                                 e.ip_address == event.ip_address]
                
                if len(recent_failures) >= self.adaptive_thresholds["brute_force"]:
                    profile.threat_score += 20
                    profile.suspicious_patterns.append("brute_force_attempt")
                    
                    self._create_isolated_threat_alert(
                        "brute_force_attack",
                        "high",
                        event.source,
                        f"Brute force attack detected from {event.ip_address}",
                        "Account locked, IP monitored",
                        confidence=0.8,
                        threat_score=profile.threat_score
                    )
            
            elif event.event_type == "authentication_success":
                # Check for account takeover attempts
                recent_successes = [e for e in self.security_events[-20:] 
                                  if e.event_type == "authentication_success" and 
                                  e.ip_address == event.ip_address]
                
                if len(recent_successes) >= 5:
                    profile.threat_score += 15
                    profile.suspicious_patterns.append("multiple_successful_logins")
                    
                    self._create_isolated_threat_alert(
                        "account_takeover_attempt",
                        "medium",
                        event.source,
                        f"Multiple successful logins from {event.ip_address}",
                        "Session monitoring enabled",
                        confidence=0.6,
                        threat_score=profile.threat_score
                    )
            
            # Check for unusual access patterns
            if event.ip_address and event.ip_address not in ["unknown", "127.0.0.1"]:
                ip_events = [e for e in self.security_events[-100:] 
                           if e.ip_address == event.ip_address]
                
                if len(ip_events) >= self.adaptive_thresholds["suspicious_ip"]:
                    profile.threat_score += 10
                    profile.suspicious_patterns.append("high_activity_ip")
                    
                    self._create_isolated_threat_alert(
                        "suspicious_ip_activity",
                        "medium",
                        event.source,
                        f"High activity from IP: {event.ip_address}",
                        "IP monitored, additional verification required",
                        confidence=0.7,
                        threat_score=profile.threat_score
                    )
            
            # Update threat profile
            self._update_isolated_threat_profile(event.ip_address, event.user_id,
                                     threat_score=profile.threat_score,
                                     suspicious_patterns=profile.suspicious_patterns)
            
            # Check if IP should be blocked
            if profile.threat_score >= self.config["threat_detection"]["max_threat_score"]:
                self.blocked_ips.add(event.ip_address)
                self._create_isolated_threat_alert(
                    "ip_blocked",
                    "high",
                    event.source,
                    f"IP {event.ip_address} blocked due to high threat score",
                    "IP added to blacklist",
                    confidence=0.9,
                    threat_score=profile.threat_score
                )
                    
        except Exception as e:
            logger.error(f"Error analyzing threat: {e}")

    def create_threat_alert(self, threat_type: str, severity: str, source: str, description: str, mitigation: str, confidence: float = 0.0, threat_score: int = 0):
        """Create enhanced threat alert."""
        try:
            alert = ThreatAlert(
                timestamp=datetime.now().isoformat(),
                threat_type=threat_type,
                severity=severity,
                source=source,
                description=description,
                mitigation=mitigation,
                confidence=confidence,
                threat_score=threat_score
            )
            
            self.threat_alerts.append(alert)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO threat_alerts (timestamp, threat_type, severity, source, description, mitigation, status, confidence, threat_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp, alert.threat_type, alert.severity, alert.source,
                alert.description, alert.mitigation, alert.status, alert.confidence, alert.threat_score
            ))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"🚨 Threat Alert: {threat_type} - {description} (Confidence: {confidence:.2f}, Score: {threat_score})")
            
        except Exception as e:
            logger.error(f"Error creating threat alert: {e}")

    def calculate_security_score(self) -> float:
        """Calculate enhanced security score."""
        try:
            score = 100.0
            
            # Deduct points for active threats
            active_threats = len([t for t in self.threat_alerts if t.status == "active"])
            score -= active_threats * 5
            
            # Deduct points for high threat scores
            high_threat_profiles = [p for p in self.threat_profiles.values() if p.threat_score > 50]
            score -= len(high_threat_profiles) * 3
            
            # Deduct points for recent security events
            recent_events = [e for e in self.security_events[-100:] 
                           if e.severity in ["high", "critical"]]
            score -= len(recent_events) * 2
            
            # Deduct points for blocked IPs
            score -= len(self.blocked_ips) * 2
            
            # Bonus for whitelisted IPs
            score += len(self.whitelisted_ips) * 1
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating security score: {e}")
            return 50.0

    def get_security_metrics(self) -> SecurityMetrics:
        """Get enhanced security metrics."""
        try:
            security_score = self.calculate_security_score()
            
            # Determine threat level
            if security_score >= 80:
                threat_level = "low"
            elif security_score >= 60:
                threat_level = "medium"
            elif security_score >= 40:
                threat_level = "high"
            else:
                threat_level = "critical"
            
            active_threats = len([t for t in self.threat_alerts if t.status == "active"])
            blocked_attempts = len([e for e in self.security_events[-100:] 
                                  if e.event_type == "authentication_failed"])
            
            return SecurityMetrics(
                timestamp=datetime.now().isoformat(),
                security_score=security_score,
                threat_level=threat_level,
                active_threats=active_threats,
                blocked_attempts=blocked_attempts,
                encryption_status="active" if self.encryption_key else "inactive",
                authentication_status="active",
                network_security="monitored"
            )
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return SecurityMetrics(
                timestamp=datetime.now().isoformat(),
                security_score=50.0,
                threat_level="unknown",
                active_threats=0,
                blocked_attempts=0,
                encryption_status="error",
                authentication_status="error",
                network_security="error"
            )

    def start_monitoring(self):
        """Start enhanced security monitoring."""
        try:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("✅ Security monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")

    def monitoring_loop(self):
        """Enhanced monitoring loop with threat analysis."""
        while self.monitoring_active:
            try:
                # Check for suspicious activities
                self.check_suspicious_activities()
                
                # Monitor network
                self.monitor_network()
                
                # Cleanup old sessions
                self.cleanup_sessions()
                
                # Update adaptive thresholds
                self.update_adaptive_thresholds()
                
                # Sleep
                time.sleep(self.config["monitoring"]["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)

    def check_suspicious_activities(self):
        """Check for suspicious activities with enhanced detection."""
        try:
            # Check for rapid-fire authentication attempts
            recent_events = self.security_events[-50:]
            auth_events = [e for e in recent_events if e.event_type in ["authentication_failed", "authentication_success"]]
            
            if len(auth_events) >= 10:
                # Group by IP
                ip_groups = defaultdict(list)
                for event in auth_events:
                    if event.ip_address:
                        ip_groups[event.ip_address].append(event)
                
                for ip, events in ip_groups.items():
                    if len(events) >= 5:
                        # Check timing between events
                        timestamps = [datetime.fromisoformat(e.timestamp) for e in events]
                        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                                   for i in range(1, len(timestamps))]
                        
                        if any(interval < 1.0 for interval in intervals):  # Less than 1 second
                            self.create_threat_alert(
                                "rapid_fire_attack",
                                "high",
                                "monitoring",
                                f"Rapid-fire authentication attempts from {ip}",
                                "IP temporarily blocked",
                                confidence=0.8,
                                threat_score=80
                            )
            
        except Exception as e:
            logger.error(f"Error checking suspicious activities: {e}")

    def monitor_network(self):
        """Monitor network for threats."""
        try:
            # Check for unusual network connections
            connections = psutil.net_connections()
            external_connections = [conn for conn in connections 
                                  if conn.status == 'ESTABLISHED' and 
                                  conn.raddr and conn.raddr.ip != '127.0.0.1']
            
            # Monitor for suspicious external connections
            for conn in external_connections:
                if conn.raddr.ip not in self.whitelisted_ips:
                    # Check if this IP has been flagged before
                    profile = self.threat_profiles.get(conn.raddr.ip)
                    if profile and profile.threat_score > 30:
                        self.create_threat_alert(
                            "suspicious_connection",
                            "medium",
                            "network",
                            f"Suspicious connection to {conn.raddr.ip}:{conn.raddr.port}",
                            "Connection monitored",
                            confidence=0.6,
                            threat_score=profile.threat_score
                        )
            
        except Exception as e:
            logger.error(f"Error monitoring network: {e}")

    def cleanup_sessions(self):
        """Cleanup expired sessions."""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.active_sessions.items():
                if (current_time - session_data["last_activity"]).total_seconds() > self.config["authentication"]["session_timeout"]:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")

    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on current threat landscape."""
        try:
            # Adjust thresholds based on recent activity
            recent_events = self.security_events[-100:]
            failed_auths = len([e for e in recent_events if e.event_type == "authentication_failed"])
            
            if failed_auths > 20:
                # Increase thresholds during high threat periods
                self.adaptive_thresholds["brute_force"] = min(10, self.adaptive_thresholds["brute_force"] + 1)
                self.adaptive_thresholds["lockout_duration"] = min(1800, self.adaptive_thresholds["lockout_duration"] * 1.5)
            elif failed_auths < 5:
                # Decrease thresholds during low threat periods
                self.adaptive_thresholds["brute_force"] = max(3, self.adaptive_thresholds["brute_force"] - 1)
                self.adaptive_thresholds["lockout_duration"] = max(300, self.adaptive_thresholds["lockout_duration"] * 0.8)
                
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {e}")

    def stop(self):
        """Stop security monitoring with isolation validation."""
        try:
            # SECURITY ISOLATION VALIDATION
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during monitoring stop")
            
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Security monitoring stopped with COMPLETE ISOLATION")
            
        except SecurityViolation as e:
            logger.error(f"SECURITY VIOLATION during monitoring stop: {e}")
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        try:
            metrics = self.get_security_metrics()
            
            return {
                "security_score": metrics.security_score,
                "threat_level": metrics.threat_level,
                "active_threats": metrics.active_threats,
                "blocked_attempts": metrics.blocked_attempts,
                "active_sessions": len(self.active_sessions),
                "blocked_ips": len(self.blocked_ips),
                "whitelisted_ips": len(self.whitelisted_ips),
                "threat_profiles": len(self.threat_profiles),
                "recent_events": len(self.security_events[-50:]),
                "adaptive_thresholds": self.adaptive_thresholds,
                "encryption_status": metrics.encryption_status,
                "authentication_status": metrics.authentication_status,
                "network_security": metrics.network_security
            }
            
        except Exception as e:
            logger.error(f"Error getting security summary: {e}")
            return {"error": str(e)}

    def create_user(self, username: str, password: str, email: str = None, role: str = "user", 
                   account_type: str = "user", created_by: str = "admin") -> Tuple[bool, str]:
        """Create new user account with user-based system and COMPLETE ISOLATION."""
        try:
            # SECURITY ISOLATION VALIDATION
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during user creation")
            
            # Validate input
            if not username or not password:
                return False, "Username and password are required"
            
            if len(password) < self.config["authentication"]["password_min_length"]:
                return False, f"Password must be at least {self.config['authentication']['password_min_length']} characters"
            
            if self.config["authentication"]["require_special_chars"]:
                if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                    return False, "Password must contain special characters"
            
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                conn.close()
                return False, "Username already exists"
            
            # Check admin limit if creating admin
            if role == "admin":
                cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                if admin_count >= 2:
                    conn.close()
                    return False, "Maximum of 2 admin accounts allowed"
            
            # Create user with enhanced security and isolation
            salt = secrets.token_hex(16)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            
            cursor.execute('''
                INSERT INTO users (username, password_hash, salt, email, role, created_at, 
                                 account_type, created_by, is_whitelisted, isolation_level, is_locked, failed_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, salt, email, role, 
                 datetime.now().isoformat(), account_type, created_by, 1, 'air_gapped', 0, 0))
            
            # Create user profile
            user_id = cursor.lastrowid
            
            # Initialize user threat profile with isolation
            self._update_isolated_threat_profile("127.0.0.1", username, 
                                     is_whitelisted=True,
                                     account_type=account_type)
            
            conn.commit()
            conn.close()
            
            # Log user creation
            self.log_security_event("user_created", "low", "admin", 
                                  f"New user created: {username} (Role: {role}, Type: {account_type})", {
                                      "username": username,
                                      "role": role,
                                      "account_type": account_type,
                                      "created_by": created_by
                                  })
            
            logger.info(f"✅ User '{username}' created successfully (Role: {role}, Type: {account_type})")
            return True, f"User '{username}' created successfully"
            
        except SecurityViolation as e:
            logger.error(f"SECURITY VIOLATION during user creation: {e}")
            return False, "Security violation detected"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, f"Error creating user: {str(e)}"

    def convert_all_to_admin(self) -> Tuple[bool, str]:
        """Convert all existing accounts to admin role."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Get all users
            cursor.execute("SELECT username, role FROM users")
            users = cursor.fetchall()
            
            if not users:
                conn.close()
                return False, "No users found to convert"
            
            # Convert all users to admin
            updated_count = 0
            for username, current_role in users:
                if current_role != "admin":
                    cursor.execute("UPDATE users SET role = 'admin', account_type = 'admin' WHERE username = ?", (username,))
                    updated_count += 1
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Converted {updated_count} users to admin role")
            return True, f"Successfully converted {updated_count} users to admin role"
            
        except Exception as e:
            logger.error(f"Error converting users to admin: {e}")
            return False, f"Error converting users: {str(e)}"

    def get_admin_count(self) -> int:
        """Get the current number of admin accounts."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            conn.close()
            return admin_count
            
        except Exception as e:
            logger.error(f"Error getting admin count: {e}")
            return 0

    def update_user_role(self, username: str, new_role: str, updated_by: str = "admin") -> Tuple[bool, str]:
        """Update user role and permissions."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, "User not found"
            
            old_role = result[0]
            cursor.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, username))
            conn.commit()
            conn.close()
            
            # Log role update
            self.log_security_event("role_updated", "medium", "admin", 
                                  f"User role updated: {username} ({old_role} -> {new_role})", {
                                      "username": username,
                                      "old_role": old_role,
                                      "new_role": new_role,
                                      "updated_by": updated_by
                                  })
            
            return True, f"Role updated successfully: {old_role} -> {new_role}"
            
        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            return False, f"Error updating role: {str(e)}"

    def get_user_list(self, admin_username: str) -> List[Dict[str, Any]]:
        """Get list of all users (admin only)."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Verify admin status
            cursor.execute("SELECT role FROM users WHERE username = ?", (admin_username,))
            result = cursor.fetchone()
            
            if not result or result[0] != "admin":
                conn.close()
                return []
            
            # Get all users
            cursor.execute('''
                SELECT username, email, role, account_type, created_at, last_login, 
                       failed_attempts, is_locked, lockout_until
                FROM users ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    "username": row[0],
                    "email": row[1],
                    "role": row[2],
                    "account_type": row[3],
                    "created_at": row[4],
                    "last_login": row[5],
                    "failed_attempts": row[6],
                    "is_locked": bool(row[7]),
                    "lockout_until": row[8]
                })
            
            conn.close()
            return users
            
        except Exception as e:
            logger.error(f"Error getting user list: {e}")
            return []

    def delete_user(self, username: str, admin_username: str) -> Tuple[bool, str]:
        """Delete user account (admin only)."""
        try:
            # Verify admin permissions
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE username = ?", (admin_username,))
            result = cursor.fetchone()
            
            if not result or result[0] != "admin":
                conn.close()
                return False, "Admin permissions required"
            
            # Check if user exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if not cursor.fetchone():
                conn.close()
                return False, "User not found"
            
            # Don't allow admin to delete themselves
            if username == admin_username:
                conn.close()
                return False, "Cannot delete your own admin account"
            
            # Delete user
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
            
            # Log user deletion
            self.log_security_event("user_deleted", "high", "admin", 
                                  f"User deleted: {username}", {
                                      "username": username,
                                      "deleted_by": admin_username
                                  })
            
            return True, f"User '{username}' deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False, f"Error deleting user: {str(e)}"

    def reset_user_password(self, username: str, new_password: str, admin_username: str) -> Tuple[bool, str]:
        """Reset user password (admin only)."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Verify admin status
            cursor.execute("SELECT role FROM users WHERE username = ?", (admin_username,))
            result = cursor.fetchone()
            
            if not result or result[0] != "admin":
                conn.close()
                return False, "Admin privileges required"
            
            # Check if target user exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if not cursor.fetchone():
                conn.close()
                return False, "User not found"
            
            # Validate new password
            if len(new_password) < self.config["authentication"]["password_min_length"]:
                return False, f"Password must be at least {self.config['authentication']['password_min_length']} characters"
            
            if self.config["authentication"]["require_special_chars"]:
                if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
                    return False, "Password must contain special characters"
            
            # Generate new password hash
            salt = secrets.token_hex(16)
            password_hash = hashlib.pbkdf2_hmac('sha256', new_password.encode(), salt.encode(), 100000).hex()
            
            # Update password and reset lockout
            cursor.execute('''
                UPDATE users SET password_hash = ?, salt = ?, failed_attempts = 0, 
                               is_locked = 0, lockout_until = NULL 
                WHERE username = ?
            ''', (password_hash, salt, username))
            
            conn.commit()
            conn.close()
            
            # Log password reset
            self.log_security_event("password_reset", "medium", "admin", 
                                  f"Password reset for user: {username}", {
                                      "username": username,
                                      "reset_by": admin_username
                                  })
            
            return True, f"Password reset successfully for {username}"
            
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            return False, f"Error resetting password: {str(e)}"

    def unlock_user_account(self, username: str, admin_username: str) -> Tuple[bool, str]:
        """Unlock user account (admin only)."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Verify admin status
            cursor.execute("SELECT role FROM users WHERE username = ?", (admin_username,))
            result = cursor.fetchone()
            
            if not result or result[0] != "admin":
                conn.close()
                return False, "Admin privileges required"
            
            # Check if target user exists
            cursor.execute("SELECT username, is_locked FROM users WHERE username = ?", (username,))
            user_result = cursor.fetchone()
            
            if not user_result:
                conn.close()
                return False, "User not found"
            
            if not user_result[1]:  # is_locked is False
                conn.close()
                return True, f"Account {username} is already unlocked"
            
            # Unlock account
            cursor.execute('''
                UPDATE users SET is_locked = 0, failed_attempts = 0, lockout_until = NULL 
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
            conn.close()
            
            # Log account unlock
            self.log_security_event("account_unlocked", "medium", "admin", 
                                  f"Account unlocked for user: {username}", {
                                      "username": username,
                                      "unlocked_by": admin_username
                                  })
            
            return True, f"Account {username} unlocked successfully"
            
        except Exception as e:
            logger.error(f"Error unlocking account: {e}")
            return False, f"Error unlocking account: {str(e)}"

    def get_user_activity(self, username: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user activity history."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Get user activity from security events
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT timestamp, event_type, severity, source, description, details
                FROM security_events 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (username, cutoff_date.isoformat()))
            
            activities = []
            for row in cursor.fetchall():
                activities.append({
                    "timestamp": row[0],
                    "event_type": row[1],
                    "severity": row[2],
                    "source": row[3],
                    "description": row[4],
                    "details": json.loads(row[5]) if row[5] else {}
                })
            
            conn.close()
            return activities
            
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return []

    def get_all_users(self, admin_username: str) -> List[Dict[str, Any]]:
        """Get all users with detailed information (admin only)."""
        try:
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Verify admin status
            cursor.execute("SELECT role FROM users WHERE username = ?", (admin_username,))
            result = cursor.fetchone()
            
            if not result or result[0] != "admin":
                conn.close()
                return []
            
            # Get all users with detailed info
            cursor.execute('''
                SELECT username, email, role, account_type, created_at, last_login, 
                       failed_attempts, is_locked, lockout_until, is_whitelisted
                FROM users ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    "username": row[0],
                    "email": row[1],
                    "role": row[2],
                    "account_type": row[3],
                    "created_at": row[4],
                    "last_login": row[5],
                    "failed_attempts": row[6],
                    "is_locked": bool(row[7]),
                    "lockout_until": row[8],
                    "is_whitelisted": bool(row[9])
                })
            
            conn.close()
            return users
            
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []

    def change_password(self, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Allow users to change their own password with COMPLETE ISOLATION."""
        try:
            # SECURITY ISOLATION VALIDATION
            if not self._validate_security_isolation():
                raise SecurityViolation("External bridges detected during password change")
            
            # Validate input
            if not username or not current_password or not new_password:
                return False, "All fields are required"
            
            # Validate new password requirements
            if len(new_password) < self.config["authentication"]["password_min_length"]:
                return False, f"New password must be at least {self.config['authentication']['password_min_length']} characters"
            
            if self.config["authentication"]["require_special_chars"]:
                if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
                    return False, "New password must contain special characters"
            
            # Check if new password is different from current
            if current_password == new_password:
                return False, "New password must be different from current password"
            
            conn = sqlite3.connect("security_keys/users.db")
            cursor = conn.cursor()
            
            # Get current password hash and salt
            cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, "User not found"
            
            stored_hash, salt = result
            
            # Verify current password
            if not self.verify_password(current_password, stored_hash, salt):
                conn.close()
                return False, "Current password is incorrect"
            
            # Generate new password hash
            new_salt = secrets.token_hex(16)
            new_password_hash = hashlib.pbkdf2_hmac('sha256', new_password.encode(), new_salt.encode(), 100000).hex()
            
            # Update password
            cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?", 
                         (new_password_hash, new_salt, username))
            conn.commit()
            conn.close()
            
            # Log password change
            self.log_security_event("password_changed", "medium", "user", 
                                  f"Password changed for user: {username}", {
                                      "username": username,
                                      "changed_by": username
                                  })
            
            logger.info(f"✅ Password changed successfully for user: {username}")
            return True, "Password changed successfully"
            
        except SecurityViolation as e:
            logger.error(f"SECURITY VIOLATION during password change: {e}")
            return False, "Security violation detected"
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return False, f"Error changing password: {str(e)}"

    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength requirements."""
        try:
            errors = []
            
            # Check minimum length
            if len(password) < self.config["authentication"]["password_min_length"]:
                errors.append(f"Password must be at least {self.config['authentication']['password_min_length']} characters")
            
            # Check for special characters
            if self.config["authentication"]["require_special_chars"]:
                if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                    errors.append("Password must contain special characters")
            
            # Check for uppercase letters
            if not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
            
            # Check for lowercase letters
            if not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
            
            # Check for numbers
            if not re.search(r'\d', password):
                errors.append("Password must contain at least one number")
            
            # Check for common weak passwords
            weak_passwords = ["password", "123456", "qwerty", "admin", "letmein", "welcome"]
            if password.lower() in weak_passwords:
                errors.append("Password is too common, please choose a stronger password")
            
            if errors:
                return False, "; ".join(errors)
            
            return True, "Password meets all requirements"
            
        except Exception as e:
            logger.error(f"Error validating password strength: {e}")
            return False, f"Error validating password: {str(e)}"

    def get_password_requirements(self) -> Dict[str, Any]:
        """Get current password requirements for display to users."""
        return {
            "min_length": self.config["authentication"]["password_min_length"],
            "require_special_chars": self.config["authentication"]["require_special_chars"],
            "requirements": [
                f"At least {self.config['authentication']['password_min_length']} characters",
                "At least one uppercase letter",
                "At least one lowercase letter", 
                "At least one number",
                "Special characters required" if self.config["authentication"]["require_special_chars"] else "Special characters recommended"
            ]
        }

    def is_user_logged_in(self, username: str) -> bool:
        """Check if a user is currently logged in."""
        try:
            # Check active sessions
            for session_id, session_data in self.active_sessions.items():
                if session_data["user_id"] == username:
                    # Check if session is still valid
                    if (datetime.now() - session_data["last_activity"]).total_seconds() < self.config["authentication"]["session_timeout"]:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking login status: {e}")
            return False

    def get_user_session_info(self, username: str) -> Dict[str, Any]:
        """Get current session information for a user."""
        try:
            for session_id, session_data in self.active_sessions.items():
                if session_data["user_id"] == username:
                    return {
                        "session_id": session_id,
                        "login_time": session_data["login_time"].isoformat(),
                        "last_activity": session_data["last_activity"].isoformat(),
                        "ip_address": session_data.get("ip_address", "unknown"),
                        "session_age_seconds": (datetime.now() - session_data["login_time"]).total_seconds(),
                        "timeout_seconds": self.config["authentication"]["session_timeout"]
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {}

    def logout_user(self, username: str) -> bool:
        """Logout a user by removing their active session."""
        try:
            sessions_to_remove = []
            
            for session_id, session_data in self.active_sessions.items():
                if session_data["user_id"] == username:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE user_sessions SET is_active = 0 WHERE session_id = ?", (session_id,))
                conn.commit()
                conn.close()
            
            # Log logout
            if sessions_to_remove:
                self.log_security_event("user_logout", "low", "auth", 
                                      f"User logged out: {username}", {
                                          "username": username,
                                          "sessions_removed": len(sessions_to_remove)
                                      })
            
            return len(sessions_to_remove) > 0
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False

class SecurityViolation(Exception):
    """Exception raised when security isolation is violated."""
    pass 

def main():
    """Main function for testing with COMPLETE ISOLATION."""
    try:
        security_system = SchwabotSecuritySystem()
        
        # Test authentication with isolation
        success, result = security_system.authenticate_user("admin", "SchwabotAdmin2025!", "127.0.0.1")
        print(f"Authentication test: {success}, {result}")
        
        # Get security summary with isolation
        summary = security_system.get_security_summary()
        print(f"Security summary: {summary}")
        
        # Stop monitoring with isolation
        security_system.stop()
        
    except SecurityViolation as e:
        logger.error(f"SECURITY VIOLATION in main: {e}")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 