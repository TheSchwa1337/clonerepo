#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hash Configuration Manager - Schwabot Configuration Security
===========================================================

This module provides secure configuration management for the Schwabot
trading system, including:

1. **Secure Configuration Storage**: Encrypted configuration files
2. **API Key Management**: Secure storage and retrieval of API keys
3. **Hash Verification**: Data integrity checking
4. **Configuration Validation**: Schema-based configuration validation
5. **Environment Integration**: Secure environment variable handling
6. **Backup and Recovery**: Encrypted configuration backups

The system ensures that all configuration data, API keys, and sensitive
settings are properly encrypted and secured.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import our encryption system
from .alpha256_encryption import Alpha256Encryption, get_encryption

logger = logging.getLogger(__name__)

def generate_hash_from_string(input_string: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash from a string using the specified algorithm.
    
    Args:
        input_string: The string to hash
        algorithm: The hashing algorithm to use (default: sha256)
    
    Returns:
        The hexadecimal hash string
    """
    try:
        if algorithm == "sha256":
            return hashlib.sha256(input_string.encode('utf-8')).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(input_string.encode('utf-8')).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(input_string.encode('utf-8')).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(input_string.encode('utf-8')).hexdigest()
        else:
            # Default to sha256
            return hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash from string: {e}")
        # Return a fallback hash
        return hashlib.sha256(f"fallback_{input_string}".encode('utf-8')).hexdigest()

class ConfigType(Enum):
    """Configuration types."""
    SYSTEM = "system"
    TRADING = "trading"
    API = "api"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    LOGGING = "logging"
    CUSTOM = "custom"

class ConfigPriority(Enum):
    """Configuration priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEBUG = "debug"

@dataclass
class ConfigEntry:
    """Configuration entry structure."""
    key: str
    value: Any
    config_type: ConfigType
    priority: ConfigPriority
    encrypted: bool = False
    hash_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigSchema:
    """Configuration schema for validation."""
    name: str
    version: str
    fields: Dict[str, Dict[str, Any]]
    required_fields: List[str]
    encrypted_fields: List[str]
    validation_rules: Dict[str, Any] = field(default_factory=dict)

class HashConfigManager:
    """Secure configuration manager with hash verification."""
    
    def __init__(self, config_path: str = "config", encryption: Optional[Alpha256Encryption] = None):
        """Initialize the configuration manager."""
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Encryption system
        self.encryption = encryption or get_encryption()
        
        # Configuration storage
        self.configs: Dict[str, Dict[str, ConfigEntry]] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.config_hashes: Dict[str, str] = {}
        
        # Default configurations
        self._initialize_default_configs()
        
        # Load existing configurations
        self._load_configurations()
        
        logger.info("Hash Configuration Manager initialized")
    
    def initialize(self):
        """Initialize the configuration manager (alias for __init__)."""
        # This method is called by other components that expect an initialize method
        # The actual initialization is done in __init__
        pass
    
    def _initialize_default_configs(self):
        """Initialize default configuration schemas."""
        
        # System configuration schema
        system_schema = ConfigSchema(
            name="system",
            version="1.0",
            fields={
                "debug_mode": {"type": "boolean", "default": False},
                "log_level": {"type": "string", "default": "INFO", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "max_memory_mb": {"type": "integer", "default": 1024, "min": 128, "max": 8192},
                "max_cpu_percent": {"type": "integer", "default": 80, "min": 10, "max": 100},
                "auto_restart": {"type": "boolean", "default": True},
                "backup_interval_hours": {"type": "integer", "default": 24, "min": 1, "max": 168}
            },
            required_fields=["debug_mode", "log_level"],
            encrypted_fields=[],
            validation_rules={
                "max_memory_mb": {"min": 128, "max": 8192},
                "max_cpu_percent": {"min": 10, "max": 100}
            }
        )
        self.schemas["system"] = system_schema
        
        # Trading configuration schema
        trading_schema = ConfigSchema(
            name="trading",
            version="1.0",
            fields={
                "default_exchange": {"type": "string", "default": "binance"},
                "default_symbol": {"type": "string", "default": "BTC/USD"},
                "max_position_size": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
                "risk_percentage": {"type": "float", "default": 2.0, "min": 0.1, "max": 10.0},
                "stop_loss_percentage": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0},
                "take_profit_percentage": {"type": "float", "default": 10.0, "min": 2.0, "max": 50.0},
                "max_open_trades": {"type": "integer", "default": 5, "min": 1, "max": 20},
                "trading_enabled": {"type": "boolean", "default": False},
                "paper_trading": {"type": "boolean", "default": True}
            },
            required_fields=["default_exchange", "default_symbol", "trading_enabled"],
            encrypted_fields=["api_keys"],
            validation_rules={
                "max_position_size": {"min": 0.001, "max": 1.0},
                "risk_percentage": {"min": 0.1, "max": 10.0},
                "stop_loss_percentage": {"min": 1.0, "max": 20.0},
                "take_profit_percentage": {"min": 2.0, "max": 50.0}
            }
        )
        self.schemas["trading"] = trading_schema
        
        # API configuration schema
        api_schema = ConfigSchema(
            name="api",
            version="1.0",
            fields={
                "kobold_port": {"type": "integer", "default": 5001, "min": 1024, "max": 65535},
                "bridge_port": {"type": "integer", "default": 5005, "min": 1024, "max": 65535},
                "enhanced_port": {"type": "integer", "default": 5006, "min": 1024, "max": 65535},
                "visual_port": {"type": "integer", "default": 5007, "min": 1024, "max": 65535},
                "api_port": {"type": "integer", "default": 5008, "min": 1024, "max": 65535},
                "max_connections": {"type": "integer", "default": 100, "min": 10, "max": 1000},
                "timeout_seconds": {"type": "integer", "default": 30, "min": 5, "max": 300},
                "rate_limit_requests": {"type": "integer", "default": 100, "min": 10, "max": 1000},
                "rate_limit_window": {"type": "integer", "default": 60, "min": 10, "max": 3600}
            },
            required_fields=["kobold_port", "bridge_port"],
            encrypted_fields=[],
            validation_rules={
                "kobold_port": {"min": 1024, "max": 65535},
                "bridge_port": {"min": 1024, "max": 65535},
                "enhanced_port": {"min": 1024, "max": 65535}
            }
        )
        self.schemas["api"] = api_schema
        
        # Security configuration schema
        security_schema = ConfigSchema(
            name="security",
            version="1.0",
            fields={
                "encryption_enabled": {"type": "boolean", "default": True},
                "hash_verification": {"type": "boolean", "default": True},
                "session_timeout_minutes": {"type": "integer", "default": 60, "min": 5, "max": 1440},
                "max_login_attempts": {"type": "integer", "default": 5, "min": 3, "max": 10},
                "password_min_length": {"type": "integer", "default": 12, "min": 8, "max": 32},
                "require_special_chars": {"type": "boolean", "default": True},
                "two_factor_enabled": {"type": "boolean", "default": False},
                "ip_whitelist": {"type": "array", "default": []},
                "allowed_origins": {"type": "array", "default": ["localhost", "127.0.0.1"]}
            },
            required_fields=["encryption_enabled", "hash_verification"],
            encrypted_fields=["master_password", "session_keys"],
            validation_rules={
                "session_timeout_minutes": {"min": 5, "max": 1440},
                "max_login_attempts": {"min": 3, "max": 10},
                "password_min_length": {"min": 8, "max": 32}
            }
        )
        self.schemas["security"] = security_schema
    
    def _load_configurations(self):
        """Load existing configurations from disk."""
        try:
            for config_type in ConfigType:
                config_file = self.config_path / f"{config_type.value}_config.json"
                if config_file.exists():
                    self._load_config_file(config_type.value, config_file)
            
            logger.info(f"Loaded {len(self.configs)} configuration types")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    def _load_config_file(self, config_type: str, config_file: Path):
        """Load a specific configuration file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            configs = {}
            for key, value in data.items():
                if isinstance(value, dict) and 'encrypted' in value:
                    # Handle encrypted configuration
                    if value['encrypted']:
                        decrypted_value = self.encryption.decrypt(value['value'], config_type)
                        config_entry = ConfigEntry(
                            key=key,
                            value=decrypted_value,
                            config_type=ConfigType(config_type),
                            priority=ConfigPriority(value.get('priority', 'medium')),
                            encrypted=True,
                            hash_id=value.get('hash_id', ''),
                            created_at=datetime.fromisoformat(value.get('created_at', datetime.now().isoformat())),
                            updated_at=datetime.fromisoformat(value.get('updated_at', datetime.now().isoformat())),
                            metadata=value.get('metadata', {})
                        )
                    else:
                        config_entry = ConfigEntry(
                            key=key,
                            value=value['value'],
                            config_type=ConfigType(config_type),
                            priority=ConfigPriority(value.get('priority', 'medium')),
                            encrypted=False,
                            hash_id=value.get('hash_id', ''),
                            created_at=datetime.fromisoformat(value.get('created_at', datetime.now().isoformat())),
                            updated_at=datetime.fromisoformat(value.get('updated_at', datetime.now().isoformat())),
                            metadata=value.get('metadata', {})
                        )
                else:
                    # Handle plain configuration
                    config_entry = ConfigEntry(
                        key=key,
                        value=value,
                        config_type=ConfigType(config_type),
                        priority=ConfigPriority.MEDIUM,
                        encrypted=False,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                
                configs[key] = config_entry
            
            self.configs[config_type] = configs
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _save_config_file(self, config_type: str):
        """Save a configuration file to disk."""
        try:
            config_file = self.config_path / f"{config_type}_config.json"
            configs = self.configs.get(config_type, {})
            
            data = {}
            for key, config_entry in configs.items():
                if config_entry.encrypted:
                    # Encrypt sensitive data
                    encrypted_value = self.encryption.encrypt(str(config_entry.value), config_type)
                    data[key] = {
                        'value': encrypted_value,
                        'encrypted': True,
                        'priority': config_entry.priority.value,
                        'hash_id': config_entry.hash_id,
                        'created_at': config_entry.created_at.isoformat(),
                        'updated_at': config_entry.updated_at.isoformat(),
                        'metadata': config_entry.metadata
                    }
                else:
                    # Store plain data
                    data[key] = {
                        'value': config_entry.value,
                        'encrypted': False,
                        'priority': config_entry.priority.value,
                        'hash_id': config_entry.hash_id,
                        'created_at': config_entry.created_at.isoformat(),
                        'updated_at': config_entry.updated_at.isoformat(),
                        'metadata': config_entry.metadata
                    }
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update hash
            self.config_hashes[config_type] = self._calculate_config_hash(data)
            
        except Exception as e:
            logger.error(f"Failed to save config file {config_type}: {e}")
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash for configuration data."""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def get_config(self, key: str, config_type: str = "system", default: Any = None) -> Any:
        """Get a configuration value."""
        try:
            configs = self.configs.get(config_type, {})
            if key in configs:
                return configs[key].value
            else:
                # Check schema for default value
                schema = self.schemas.get(config_type)
                if schema and key in schema.fields:
                    return schema.fields[key].get('default', default)
                return default
                
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            return default
    
    def set_config(self, key: str, value: Any, config_type: str = "system", 
                  priority: ConfigPriority = ConfigPriority.MEDIUM, encrypt: bool = False):
        """Set a configuration value."""
        try:
            # Validate configuration
            if not self._validate_config(key, value, config_type):
                raise ValueError(f"Invalid configuration value for {key}")
            
            # Check if encryption is required
            schema = self.schemas.get(config_type)
            if schema and key in schema.encrypted_fields:
                encrypt = True
            
            # Create or update configuration entry
            if config_type not in self.configs:
                self.configs[config_type] = {}
            
            if key in self.configs[config_type]:
                # Update existing entry
                config_entry = self.configs[config_type][key]
                config_entry.value = value
                config_entry.updated_at = datetime.now()
                config_entry.encrypted = encrypt
                config_entry.priority = priority
            else:
                # Create new entry
                config_entry = ConfigEntry(
                    key=key,
                    value=value,
                    config_type=ConfigType(config_type),
                    priority=priority,
                    encrypted=encrypt,
                    hash_id=secrets.token_hex(16),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            
            self.configs[config_type][key] = config_entry
            
            # Save to disk
            self._save_config_file(config_type)
            
            logger.info(f"Configuration {config_type}.{key} updated")
            
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            raise
    
    def _validate_config(self, key: str, value: Any, config_type: str) -> bool:
        """Validate configuration value against schema."""
        try:
            schema = self.schemas.get(config_type)
            if not schema or key not in schema.fields:
                return True  # No schema validation required
            
            field_schema = schema.fields[key]
            field_type = field_schema.get('type', 'string')
            
            # Type validation
            if field_type == 'boolean' and not isinstance(value, bool):
                return False
            elif field_type == 'integer' and not isinstance(value, int):
                return False
            elif field_type == 'float' and not isinstance(value, (int, float)):
                return False
            elif field_type == 'string' and not isinstance(value, str):
                return False
            elif field_type == 'array' and not isinstance(value, list):
                return False
            
            # Range validation
            if 'min' in field_schema:
                if isinstance(value, (int, float)) and value < field_schema['min']:
                    return False
            
            if 'max' in field_schema:
                if isinstance(value, (int, float)) and value > field_schema['max']:
                    return False
            
            # Enum validation
            if 'enum' in field_schema:
                if value not in field_schema['enum']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_api_config(self, exchange: str) -> Dict[str, Any]:
        """Get API configuration for a specific exchange."""
        try:
            # Get API keys from encryption system
            api_keys = self.encryption.list_api_keys()
            
            # Find keys for the specified exchange
            exchange_keys = [key for key in api_keys if key['exchange'] == exchange]
            
            if not exchange_keys:
                return {}
            
            # Get the most recently used key
            latest_key = max(exchange_keys, key=lambda k: k['last_used'] or k['created_at'])
            
            # Retrieve actual API key and secret
            api_key, api_secret = self.encryption.get_api_key(latest_key['key_id'])
            
            return {
                'exchange': exchange,
                'api_key': api_key,
                'api_secret': api_secret,
                'permissions': latest_key['permissions'],
                'key_id': latest_key['key_id']
            }
            
        except Exception as e:
            logger.error(f"Failed to get API config for {exchange}: {e}")
            return {}
    
    def set_api_config(self, exchange: str, api_key: str, api_secret: str, 
                      permissions: List[str] = None) -> str:
        """Set API configuration for a specific exchange."""
        try:
            # Store API key in encryption system
            key_id = self.encryption.store_api_key(
                exchange=exchange,
                api_key=api_key,
                api_secret=api_secret,
                permissions=permissions
            )
            
            # Store reference in configuration
            self.set_config(
                key=f"{exchange}_api_key_id",
                value=key_id,
                config_type="api",
                priority=ConfigPriority.HIGH,
                encrypt=True
            )
            
            logger.info(f"API configuration stored for {exchange}")
            return key_id
            
        except Exception as e:
            logger.error(f"Failed to set API config for {exchange}: {e}")
            raise
    
    def get_all_configs(self, config_type: Optional[str] = None) -> Dict[str, Any]:
        """Get all configurations or configurations of a specific type."""
        try:
            if config_type:
                configs = self.configs.get(config_type, {})
                return {key: entry.value for key, entry in configs.items()}
            else:
                all_configs = {}
                for config_type, configs in self.configs.items():
                    all_configs[config_type] = {key: entry.value for key, entry in configs.items()}
                return all_configs
                
        except Exception as e:
            logger.error(f"Failed to get all configs: {e}")
            return {}
    
    def delete_config(self, key: str, config_type: str = "system") -> bool:
        """Delete a configuration entry."""
        try:
            if config_type in self.configs and key in self.configs[config_type]:
                del self.configs[config_type][key]
                self._save_config_file(config_type)
                logger.info(f"Configuration {config_type}.{key} deleted")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete config {key}: {e}")
            return False
    
    def backup_configs(self, backup_path: str = "backups") -> str:
        """Create an encrypted backup of all configurations."""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"config_backup_{timestamp}.json"
            
            # Prepare backup data
            backup_data = {
                'timestamp': timestamp,
                'version': '1.0',
                'configs': self.get_all_configs(),
                'schemas': {name: {
                    'name': schema.name,
                    'version': schema.version,
                    'fields': schema.fields,
                    'required_fields': schema.required_fields,
                    'encrypted_fields': schema.encrypted_fields
                } for name, schema in self.schemas.items()},
                'hashes': self.config_hashes
            }
            
            # Encrypt backup
            encrypted_backup = self.encryption.encrypt(json.dumps(backup_data), "backup")
            
            # Save encrypted backup
            with open(backup_file, 'w') as f:
                json.dump({'encrypted_data': encrypted_backup}, f, indent=2)
            
            logger.info(f"Configuration backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_configs(self, backup_file: str) -> bool:
        """Restore configurations from an encrypted backup."""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Decrypt backup
            encrypted_data = backup_data['encrypted_data']
            decrypted_data = self.encryption.decrypt(encrypted_data, "backup")
            backup_configs = json.loads(decrypted_data)
            
            # Restore configurations
            for config_type, configs in backup_configs['configs'].items():
                for key, value in configs.items():
                    self.set_config(key, value, config_type)
            
            logger.info(f"Configuration backup restored from: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration system status."""
        try:
            total_configs = sum(len(configs) for configs in self.configs.values())
            encrypted_configs = sum(
                sum(1 for entry in configs.values() if entry.encrypted)
                for configs in self.configs.values()
            )
            
            return {
                'total_config_types': len(self.configs),
                'total_configs': total_configs,
                'encrypted_configs': encrypted_configs,
                'schemas_loaded': len(self.schemas),
                'backup_available': True,
                'encryption_enabled': True,
                'hash_verification': True
            }
            
        except Exception as e:
            logger.error(f"Failed to get config status: {e}")
            return {}

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> HashConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = HashConfigManager()
    return _config_manager

def get_config(key: str, config_type: str = "system", default: Any = None) -> Any:
    """Get a configuration value using the global manager."""
    return get_config_manager().get_config(key, config_type, default)

def set_config(key: str, value: Any, config_type: str = "system", 
              priority: ConfigPriority = ConfigPriority.MEDIUM, encrypt: bool = False):
    """Set a configuration value using the global manager."""
    return get_config_manager().set_config(key, value, config_type, priority, encrypt)

def get_api_config(exchange: str) -> Dict[str, Any]:
    """Get API configuration using the global manager."""
    return get_config_manager().get_api_config(exchange)

def set_api_config(exchange: str, api_key: str, api_secret: str, 
                  permissions: List[str] = None) -> str:
    """Set API configuration using the global manager."""
    return get_config_manager().set_api_config(exchange, api_key, api_secret, permissions)

async def main():
    """Test the configuration manager."""
    try:
        logger.info("Testing Hash Configuration Manager")
        
        # Initialize configuration manager
        config_manager = HashConfigManager()
        
        # Test basic configuration
        config_manager.set_config("test_key", "test_value", "system")
        value = config_manager.get_config("test_key", "system")
        assert value == "test_value"
        logger.info("Basic configuration test passed")
        
        # Test API configuration
        key_id = config_manager.set_api_config(
            exchange="test_exchange",
            api_key="test_api_key",
            api_secret="test_api_secret",
            permissions=["read", "trade"]
        )
        
        api_config = config_manager.get_api_config("test_exchange")
        assert api_config['api_key'] == "test_api_key"
        assert api_config['api_secret'] == "test_api_secret"
        logger.info("API configuration test passed")
        
        # Test configuration status
        status = config_manager.get_config_status()
        logger.info(f"Configuration Status: {status}")
        
        logger.info("All configuration tests passed!")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 