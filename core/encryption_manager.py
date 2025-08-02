#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 ENCRYPTION MANAGER - PRODUCTION SECURITY SYSTEM
==================================================

Real encryption system using AES-256 for Schwabot trading system.
Provides secure storage and transmission of sensitive data.

Features:
- AES-256 encryption/decryption
- Secure key generation and management
- API key encryption
- Configuration file encryption
- Secure credential storage
- Key rotation capabilities
- Hardware-accelerated encryption (when available)
- Audit logging for security events
"""

import base64
import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Production-grade encryption manager for Schwabot system."""
    
    def __init__(self, master_key: Optional[str] = None, key_file: str = "config/encryption.key"):
        self.key_file = key_file
        self.master_key = master_key or self._load_or_generate_master_key()
        
        # Initialize encryption backend
        self.backend = default_backend()
        
        self.fernet = self._create_fernet()
        self.key_rotation_schedule = {}
        self.audit_log = []
        
        logger.info("Encryption manager initialized with AES-256")
    
    def _load_or_generate_master_key(self) -> str:
        """Load existing master key or generate a new one."""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    key_data = f.read()
                    # Decode the stored key
                    return base64.urlsafe_b64decode(key_data).decode('utf-8')
            else:
                # Generate new master key
                master_key = secrets.token_urlsafe(32)
                self._save_master_key(master_key)
                return master_key
        except Exception as e:
            logger.error(f"Error loading/generating master key: {e}")
            # Fallback to environment variable
            return os.getenv('SCHWABOT_MASTER_KEY', secrets.token_urlsafe(32))
    
    def _save_master_key(self, master_key: str):
        """Save master key to file with proper permissions."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            # Encode and save key
            key_data = base64.urlsafe_b64encode(master_key.encode('utf-8'))
            with open(self.key_file, 'wb') as f:
                f.write(key_data)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(self.key_file, 0o600)
            
            logger.info(f"Master key saved to {self.key_file}")
        except Exception as e:
            logger.error(f"Error saving master key: {e}")
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance for encryption."""
        try:
            # Derive key from master key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'schwabot_salt_2024',  # Fixed salt for consistency
                iterations=100000,
                backend=self.backend
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Error creating Fernet instance: {e}")
            raise
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], 
                    key_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Encrypt data using AES-256.
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
            key_id: Optional key identifier for rotation tracking
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        try:
            # Convert data to bytes if needed
            if isinstance(data, dict):
                data_bytes = str(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Generate random IV for AES
            iv = os.urandom(16)
            
            # Create AES cipher
            cipher = Cipher(
                algorithms.AES(self._get_encryption_key()),
                modes.CBC(iv),
                backend=self.backend
            )
            
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(data_bytes)
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Create result
            result = {
                'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                'iv': base64.b64encode(iv).decode('utf-8'),
                'algorithm': 'AES-256-CBC',
                'timestamp': datetime.now().isoformat(),
                'key_id': key_id or 'default'
            }
            
            # Log encryption event
            self._log_security_event('encrypt', key_id, len(data_bytes))
            
            return result
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt data using AES-256.
        
        Args:
            encrypted_package: Dictionary containing encrypted data and metadata
            
        Returns:
            Decrypted data
        """
        try:
            # Extract components
            encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
            iv = base64.b64decode(encrypted_package['iv'])
            algorithm = encrypted_package.get('algorithm', 'AES-256-CBC')
            key_id = encrypted_package.get('key_id', 'default')
            
            # Validate algorithm
            if algorithm != 'AES-256-CBC':
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Create AES cipher
            cipher = Cipher(
                algorithms.AES(self._get_encryption_key()),
                modes.CBC(iv),
                backend=self.backend
            )
            
            decryptor = cipher.decryptor()
            
            # Decrypt data
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            unpadded_data = self._unpad_data(decrypted_data)
            
            # Try to decode as string
            try:
                result = unpadded_data.decode('utf-8')
                # Try to parse as JSON/dict
                try:
                    import json
                    result = json.loads(result)
                except:
                    pass
            except:
                result = unpadded_data
            
            # Log decryption event
            self._log_security_event('decrypt', key_id, len(unpadded_data))
            
            return result
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def _get_encryption_key(self) -> bytes:
        """Get the current encryption key."""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'schwabot_salt_2024',
            iterations=100000,
            backend=self.backend
        )
        
        return kdf.derive(self.master_key.encode())
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to AES block size."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, data: bytes) -> bytes:
        """Remove padding from data."""
        padding_length = data[-1]
        return data[:-padding_length]
    
    def encrypt_api_key(self, api_key: str, exchange: str) -> str:
        """Encrypt API key for specific exchange."""
        try:
            encrypted = self.encrypt_data(api_key, f"api_key_{exchange}")
            return base64.b64encode(str(encrypted).encode()).decode()
        except Exception as e:
            logger.error(f"Error encrypting API key: {e}")
            raise
    
    def decrypt_api_key(self, encrypted_api_key: str) -> str:
        """Decrypt API key."""
        try:
            import json
            encrypted_package = json.loads(base64.b64decode(encrypted_api_key).decode())
            return self.decrypt_data(encrypted_package)
        except Exception as e:
            logger.error(f"Error decrypting API key: {e}")
            raise
    
    def encrypt_config_file(self, config_data: Dict[str, Any], filename: str) -> bool:
        """Encrypt configuration file."""
        try:
            encrypted = self.encrypt_data(config_data, f"config_{filename}")
            
            # Save encrypted config
            encrypted_file = f"{filename}.encrypted"
            with open(encrypted_file, 'w') as f:
                import json
                json.dump(encrypted, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(encrypted_file, 0o600)
            
            logger.info(f"Configuration file encrypted: {encrypted_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting config file: {e}")
            return False
    
    def decrypt_config_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Decrypt configuration file."""
        try:
            encrypted_file = f"{filename}.encrypted"
            
            if not os.path.exists(encrypted_file):
                return None
            
            with open(encrypted_file, 'r') as f:
                import json
                encrypted_package = json.load(f)
            
            return self.decrypt_data(encrypted_package)
            
        except Exception as e:
            logger.error(f"Error decrypting config file: {e}")
            return None
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a secure random password."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password using PBKDF2."""
        try:
            if salt is None:
                salt = secrets.token_hex(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
                backend=self.backend
            )
            
            hash_bytes = kdf.derive(password.encode())
            hash_hex = hash_bytes.hex()
            
            return {
                'hash': hash_hex,
                'salt': salt,
                'algorithm': 'PBKDF2-SHA256',
                'iterations': '100000'
            }
            
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
                backend=self.backend
            )
            
            hash_bytes = kdf.derive(password.encode())
            hash_hex = hash_bytes.hex()
            
            return hash_hex == stored_hash
            
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def rotate_encryption_key(self, new_master_key: Optional[str] = None) -> bool:
        """Rotate encryption key."""
        try:
            old_master_key = self.master_key
            new_master_key = new_master_key or secrets.token_urlsafe(32)
            
            # Update master key
            self.master_key = new_master_key
            self.fernet = self._create_fernet()
            
            # Save new key
            self._save_master_key(new_master_key)
            
            # Log rotation
            self._log_security_event('key_rotation', 'system', 0)
            
            logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating encryption key: {e}")
            return False
    
    def _log_security_event(self, event_type: str, key_id: str, data_size: int):
        """Log security events for audit."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'key_id': key_id,
            'data_size': data_size,
            'user_agent': 'Schwabot-System'
        }
        
        self.audit_log.append(event)
        
        # Keep only last 1000 events
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        logger.info(f"Security event: {event_type} - {key_id} - {data_size} bytes")
    
    def get_audit_log(self, limit: int = 100) -> list:
        """Get security audit log."""
        return self.audit_log[-limit:] if self.audit_log else []
    
    def export_audit_log(self, filename: str) -> bool:
        """Export audit log to file."""
        try:
            with open(filename, 'w') as f:
                import json
                json.dump(self.audit_log, f, indent=2)
            
            logger.info(f"Audit log exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get encryption system security status."""
        return {
            'algorithm': 'AES-256-CBC',
            'key_rotation_enabled': True,
            'audit_logging_enabled': True,
            'total_events': len(self.audit_log),
            'last_event': self.audit_log[-1] if self.audit_log else None,
            'key_file_exists': os.path.exists(self.key_file),
            'key_file_permissions': oct(os.stat(self.key_file).st_mode)[-3:] if os.path.exists(self.key_file) else None
        }

# Global instance
# encryption_manager = EncryptionManager()  # Commented out to prevent initialization errors 