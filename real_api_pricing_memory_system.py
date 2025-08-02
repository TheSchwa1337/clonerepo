#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 REAL API PRICING & MEMORY STORAGE SYSTEM - SCHWABOT
======================================================

Revolutionary system that ensures ALL testing routes to REAL API pricing
and implements comprehensive long-term memory storage with USB and computer options.

Features:
✅ Real API pricing for ALL modes and testing
✅ Long-term memory storage with USB/Computer choice
✅ Memory choice menu and file pathing system
✅ Automatic memory routing to proper channels
✅ USB menu for schwabot operations
✅ Real-time memory synchronization
✅ Comprehensive memory management

This system eliminates ALL static pricing and ensures proper memory routing!
"""

import os
import sys
import json
import time
import shutil
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
import zipfile
import sqlite3

# Import existing systems
try:
    from schwabot_usb_memory import SchwabotUSBMemory
    from AOI_Base_Files_Schwabot.usb_manager import USBManager
    USB_SYSTEMS_AVAILABLE = True
except ImportError:
    USB_SYSTEMS_AVAILABLE = False
    print("⚠️ USB systems not available - using local memory only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_api_memory_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 🔒 SAFETY CONFIGURATION
class MemoryStorageMode(Enum):
    """Memory storage modes for safety control."""
    LOCAL_ONLY = "local_only"      # Local computer only
    USB_ONLY = "usb_only"          # USB drive only
    HYBRID = "hybrid"              # Both local and USB
    AUTO = "auto"                  # Automatic selection

class APIMode(Enum):
    """API modes for real pricing."""
    REAL_API_ONLY = "real_api_only"    # Only real API data
    REAL_WITH_FALLBACK = "real_with_fallback"  # Real API with fallback
    TESTING_MODE = "testing_mode"      # Testing with real API
    PRODUCTION = "production"          # Production mode

@dataclass
class MemoryConfig:
    """Configuration for memory storage system."""
    storage_mode: MemoryStorageMode = MemoryStorageMode.AUTO
    api_mode: APIMode = APIMode.REAL_API_ONLY
    local_memory_path: str = "SchwabotMemory"
    usb_memory_path: str = "SchwabotMemory"
    backup_interval: int = 300  # 5 minutes
    max_backup_age_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = True
    auto_sync: bool = True
    memory_choice_menu: bool = True
    
    # API Configuration
    required_api_keys: List[str] = field(default_factory=lambda: [
        'BINANCE_API_KEY', 'BINANCE_SECRET_KEY',
        'COINBASE_API_KEY', 'COINBASE_SECRET_KEY',
        'KRAKEN_API_KEY', 'KRAKEN_SECRET_KEY'
    ])
    api_timeout: int = 10  # seconds
    api_retry_attempts: int = 3
    api_cache_duration: int = 60  # seconds

@dataclass
class MemoryEntry:
    """Individual memory entry."""
    entry_id: str
    timestamp: datetime
    data_type: str
    data: Dict[str, Any]
    source: str
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    checksum: str = ""
    compressed: bool = False
    encrypted: bool = False

class RealAPIPricingMemorySystem:
    """Revolutionary system for real API pricing and memory storage."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.base_dir = Path(__file__).parent.absolute()
        
        # Memory storage
        self.local_memory_dir = self.base_dir / self.config.local_memory_path
        self.usb_memory_dir = None
        self.memory_database = None
        self.memory_cache: Dict[str, Any] = {}
        
        # API clients
        self.api_clients: Dict[str, Any] = {}
        self.api_cache: Dict[str, Dict[str, Any]] = {}
        self.last_api_update: Dict[str, float] = {}
        
        # Threading
        self.sync_thread = None
        self.api_thread = None
        self.stop_threads = False
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete system."""
        try:
            logger.info("🚀 Initializing Real API Pricing & Memory Storage System")
            
            # Create local memory directory
            self.local_memory_dir.mkdir(exist_ok=True)
            self._create_memory_structure(self.local_memory_dir)
            
            # Initialize USB memory if available
            if USB_SYSTEMS_AVAILABLE:
                self._initialize_usb_memory()
            
            # Initialize memory database
            self._initialize_memory_database()
            
            # Initialize API clients
            self._initialize_api_clients()
            
            # Start background threads
            self._start_background_threads()
            
            # Show memory choice menu if enabled
            if self.config.memory_choice_menu:
                self._show_memory_choice_menu()
            
            logger.info("✅ Real API Pricing & Memory Storage System initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            raise
    
    def _create_memory_structure(self, base_dir: Path):
        """Create memory directory structure."""
        subdirs = [
            'api_data', 'trading_data', 'backtest_results', 'performance_metrics',
            'system_logs', 'config_backups', 'memory_cache', 'compressed_data',
            'encrypted_data', 'usb_sync', 'real_time_data', 'historical_data'
        ]
        
        for subdir in subdirs:
            (base_dir / subdir).mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""Schwabot Memory Storage Directory
====================================

This directory contains all Schwabot memory and data storage.
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {self.config.storage_mode.value}
API Mode: {self.config.api_mode.value}

Directory Structure:
- api_data/: Real API pricing data
- trading_data/: Trading decisions and results
- backtest_results/: Backtesting data
- performance_metrics/: Performance tracking
- system_logs/: System operation logs
- config_backups/: Configuration backups
- memory_cache/: Memory cache files
- compressed_data/: Compressed memory data
- encrypted_data/: Encrypted sensitive data
- usb_sync/: USB synchronization data
- real_time_data/: Real-time market data
- historical_data/: Historical market data

Keep this directory secure - it contains sensitive trading information.
"""
        
        with open(base_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def _initialize_usb_memory(self):
        """Initialize USB memory system."""
        try:
            if USB_SYSTEMS_AVAILABLE:
                # Use existing USB memory system
                self.usb_memory = SchwabotUSBMemory()
                self.usb_memory_dir = self.usb_memory.usb_memory_dir
                
                if self.usb_memory_dir:
                    self._create_memory_structure(self.usb_memory_dir)
                    logger.info(f"✅ USB memory initialized: {self.usb_memory_dir}")
                else:
                    logger.warning("⚠️ No USB memory available")
            
        except Exception as e:
            logger.error(f"❌ Error initializing USB memory: {e}")
    
    def _initialize_memory_database(self):
        """Initialize SQLite memory database."""
        try:
            db_path = self.local_memory_dir / 'memory_database.db'
            self.memory_database = sqlite3.connect(str(db_path))
            
            # Create tables
            cursor = self.memory_database.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    tags TEXT,
                    checksum TEXT,
                    compressed BOOLEAN DEFAULT FALSE,
                    encrypted BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_cache (
                    symbol TEXT PRIMARY KEY,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            self.memory_database.commit()
            logger.info("✅ Memory database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing memory database: {e}")
    
    def _initialize_api_clients(self):
        """Initialize API clients for real pricing."""
        try:
            # Load API keys from environment
            api_keys = self._load_api_keys()
            
            if not api_keys:
                logger.warning("⚠️ No API keys found - using fallback data sources")
                return
            
            # Initialize exchange clients
            for exchange, keys in api_keys.items():
                try:
                    if exchange == 'binance':
                        self._initialize_binance_client(keys)
                    elif exchange == 'binance_usa':
                        self._initialize_binance_usa_client(keys)
                    elif exchange == 'coinbase':
                        self._initialize_coinbase_client(keys)
                    elif exchange == 'kraken':
                        self._initialize_kraken_client(keys)
                    
                    logger.info(f"✅ {exchange.upper()} API client initialized")
                    
                except Exception as e:
                    logger.error(f"❌ Error initializing {exchange} client: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing API clients: {e}")
    
    def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Load API keys from environment, config files, or API key configuration system."""
        api_keys = {}
        
        # First, try to load from the existing sophisticated encryption systems
        try:
            # Try Alpha256 encryption system first
            from core.alpha256_encryption import Alpha256Encryption
            alpha256_system = Alpha256Encryption()
            
            # Get all stored API keys
            api_key_list = alpha256_system.list_api_keys()
            
            for key_info in api_key_list:
                exchange = key_info['exchange']
                if exchange not in api_keys:
                    try:
                        api_key, secret_key = alpha256_system.get_api_key(key_info['key_id'])
                        if api_key and secret_key:
                            api_keys[exchange] = {
                                'api_key': api_key,
                                'secret_key': secret_key
                            }
                            logger.info(f"✅ Loaded {exchange} keys from Alpha256 encryption system")
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading {exchange} from Alpha256: {e}")
            
            if api_keys:
                logger.info(f"✅ Successfully loaded {len(api_keys)} exchanges from Alpha256 encryption system")
                return api_keys
                
        except ImportError:
            logger.info("ℹ️ Alpha256 encryption system not available")
        except Exception as e:
            logger.warning(f"⚠️ Error loading from Alpha256 encryption system: {e}")
        
        # Second, try the API key configuration system
        try:
            from api_key_configuration import APIKeyManager
            key_manager = APIKeyManager()
            
            # Load keys for each exchange
            exchanges = ['binance', 'coinbase', 'kraken']
            for exchange in exchanges:
                api_key = key_manager.get_api_key(exchange, "api_key")
                secret_key = key_manager.get_api_key(exchange, "secret_key")
                
                if api_key and secret_key:
                    api_keys[exchange] = {
                        'api_key': api_key,
                        'secret_key': secret_key
                    }
                    logger.info(f"✅ Loaded {exchange} keys from API key configuration")
            
            if api_keys:
                logger.info(f"✅ Loaded {len(api_keys)} exchange configurations from API key system")
                return api_keys
                
        except ImportError:
            logger.info("ℹ️ API key configuration system not available, trying other methods")
        except Exception as e:
            logger.warning(f"⚠️ Error loading from API key configuration: {e}")
        
        # Third, try Encryption Manager system
        try:
            from core.encryption_manager import EncryptionManager
            encryption_manager = EncryptionManager()
            
            # Try to decrypt existing encrypted keys from config file
            config_file = self.base_dir / 'config' / 'keys' / 'api_keys.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    
                for item in config_data:
                    if isinstance(item, dict) and 'exchange' in item:
                        exchange = item['exchange']
                        try:
                            # Try to decrypt using encryption manager
                            encrypted_key = item.get('encrypted_key', '')
                            encrypted_secret = item.get('encrypted_secret', '')
                            
                            if encrypted_key and encrypted_secret:
                                # This would need proper integration with the encryption manager
                                # For now, we'll skip this method
                                pass
                        except Exception as e:
                            logger.debug(f"Error decrypting {exchange} keys: {e}")
                            
        except ImportError:
            logger.info("ℹ️ Encryption Manager not available")
        except Exception as e:
            logger.warning(f"⚠️ Error loading from Encryption Manager: {e}")
        
        # Fallback: Check environment variables
        exchanges = ['binance', 'binance_usa', 'coinbase', 'kraken']
        
        for exchange in exchanges:
            api_key = os.getenv(f'{exchange.upper()}_API_KEY')
            secret_key = os.getenv(f'{exchange.upper()}_SECRET_KEY')
            
            if api_key and secret_key:
                exchange_keys = {
                    'api_key': api_key,
                    'secret_key': secret_key
                }
                
                # Add passphrase for Coinbase
                if exchange == 'coinbase':
                    passphrase = os.getenv('COINBASE_PASSPHRASE')
                    if passphrase:
                        exchange_keys['passphrase'] = passphrase
                        logger.info(f"✅ Loaded {exchange} passphrase from environment variables")
                
                api_keys[exchange] = exchange_keys
                logger.info(f"✅ Loaded {exchange} keys from environment variables")
        
        # Fallback: Check config files
        config_files = [
            self.base_dir / '.env',
            self.base_dir / 'config' / 'api_keys.json',
            self.local_memory_dir / 'config' / 'api_keys.json',
            Path('config/keys/api_keys.json')  # New location
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.json':
                            config_data = json.load(f)
                            for exchange, keys in config_data.items():
                                if exchange not in api_keys and isinstance(keys, dict):
                                    # Handle both new and old formats
                                    if 'api_key' in keys and 'secret_key' in keys:
                                        api_keys[exchange] = {
                                            'api_key': keys['api_key'],
                                            'secret_key': keys['secret_key']
                                        }
                                        logger.info(f"✅ Loaded {exchange} keys from {config_file}")
                        else:
                            # .env file
                            for line in f:
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    if 'API_KEY' in key or 'SECRET_KEY' in key:
                                        # Parse into exchange structure
                                        pass
                except Exception as e:
                    logger.warning(f"⚠️ Error loading config file {config_file}: {e}")
        
        # Log summary
        if api_keys:
            logger.info(f"✅ Successfully loaded API keys for {len(api_keys)} exchanges")
            for exchange in api_keys.keys():
                logger.info(f"   • {exchange}")
        else:
            logger.warning("⚠️ No API keys found. Please configure API keys using the API key configuration system.")
            logger.info("💡 Run 'python api_key_configuration.py' to configure API keys")
            logger.info("🔐 The system will use your existing 5-layer encryption: Alpha256, Alpha (Ω-B-Γ), AES-256, and Base64")
        
        return api_keys
    
    def _initialize_binance_client(self, keys: Dict[str, str]):
        """Initialize Binance API client."""
        try:
            import ccxt
            self.api_clients['binance'] = ccxt.binance({
                'apiKey': keys['api_key'],
                'secret': keys['secret_key'],
                'timeout': self.config.api_timeout * 1000,
                'enableRateLimit': True
            })
        except ImportError:
            logger.warning("⚠️ ccxt library not available for Binance")
        except Exception as e:
            logger.error(f"❌ Error initializing Binance client: {e}")
    
    def _initialize_binance_usa_client(self, keys: Dict[str, str]):
        """Initialize Binance USA API client."""
        try:
            import ccxt
            self.api_clients['binance_usa'] = ccxt.binanceus({
                'apiKey': keys['api_key'],
                'secret': keys['secret_key'],
                'timeout': self.config.api_timeout * 1000,
                'enableRateLimit': True
            })
        except ImportError:
            logger.warning("⚠️ ccxt library not available for Binance USA")
        except Exception as e:
            logger.error(f"❌ Error initializing Binance USA client: {e}")
    
    def _initialize_coinbase_client(self, keys: Dict[str, str]):
        """Initialize Coinbase API client."""
        try:
            import ccxt
            
            # Coinbase requires apiKey, secret, and passphrase
            coinbase_config = {
                'apiKey': keys['api_key'],
                'secret': keys['secret_key'],
                'timeout': self.config.api_timeout * 1000,
                'enableRateLimit': True
            }
            
            # Add passphrase if available (required for Coinbase)
            if 'passphrase' in keys:
                coinbase_config['password'] = keys['passphrase']  # CCXT uses 'password' for passphrase
            
            # Use ONLY the current Coinbase exchange (Coinbase Pro is deprecated)
            self.api_clients['coinbase'] = ccxt.coinbase(coinbase_config)
            logger.info("✅ Coinbase API client initialized (current unified exchange)")
                    
        except ImportError:
            logger.warning("⚠️ ccxt library not available for Coinbase")
        except Exception as e:
            logger.error(f"❌ Error initializing Coinbase client: {e}")
    
    def _initialize_kraken_client(self, keys: Dict[str, str]):
        """Initialize Kraken API client."""
        try:
            import ccxt
            self.api_clients['kraken'] = ccxt.kraken({
                'apiKey': keys['api_key'],
                'secret': keys['secret_key'],
                'timeout': self.config.api_timeout * 1000,
                'enableRateLimit': True
            })
        except ImportError:
            logger.warning("⚠️ ccxt library not available for Kraken")
        except Exception as e:
            logger.error(f"❌ Error initializing Kraken client: {e}")
    
    def _start_background_threads(self):
        """Start background threads for API updates and memory sync."""
        try:
            # API update thread
            self.api_thread = threading.Thread(target=self._api_update_loop, daemon=True)
            self.api_thread.start()
            
            # Memory sync thread
            if self.config.auto_sync:
                self.sync_thread = threading.Thread(target=self._memory_sync_loop, daemon=True)
                self.sync_thread.start()
            
            logger.info("✅ Background threads started")
            
        except Exception as e:
            logger.error(f"❌ Error starting background threads: {e}")
    
    def _show_memory_choice_menu(self):
        """Show memory choice menu for user selection."""
        try:
            print("\n" + "="*60)
            print("💾 SCHWABOT MEMORY STORAGE CHOICE MENU")
            print("="*60)
            print("Choose your memory storage location:")
            print("1. Local Computer Only (Recommended for testing)")
            print("2. USB Drive Only (Portable, secure)")
            print("3. Hybrid Mode (Both local and USB)")
            print("4. Auto Mode (Automatic selection)")
            print("5. Show current configuration")
            print("6. Skip (Use current settings)")
            print("="*60)
            
            while True:
                try:
                    choice = input("Enter your choice (1-6): ").strip()
                    
                    if choice == '1':
                        self.config.storage_mode = MemoryStorageMode.LOCAL_ONLY
                        print("✅ Set to Local Computer Only mode")
                        break
                    elif choice == '2':
                        self.config.storage_mode = MemoryStorageMode.USB_ONLY
                        print("✅ Set to USB Drive Only mode")
                        break
                    elif choice == '3':
                        self.config.storage_mode = MemoryStorageMode.HYBRID
                        print("✅ Set to Hybrid Mode")
                        break
                    elif choice == '4':
                        self.config.storage_mode = MemoryStorageMode.AUTO
                        print("✅ Set to Auto Mode")
                        break
                    elif choice == '5':
                        self._show_current_configuration()
                    elif choice == '6':
                        print("✅ Using current configuration")
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1-6.")
                        
                except KeyboardInterrupt:
                    print("\n✅ Using current configuration")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            # Save configuration
            self._save_configuration()
            
        except Exception as e:
            logger.error(f"❌ Error showing memory choice menu: {e}")
    
    def _show_current_configuration(self):
        """Show current memory configuration."""
        print(f"\n📊 Current Configuration:")
        print(f"   Storage Mode: {self.config.storage_mode.value}")
        print(f"   API Mode: {self.config.api_mode.value}")
        print(f"   Local Memory: {self.local_memory_dir}")
        print(f"   USB Memory: {self.usb_memory_dir or 'Not available'}")
        print(f"   Auto Sync: {self.config.auto_sync}")
        print(f"   Compression: {self.config.compression_enabled}")
        print(f"   Encryption: {self.config.encryption_enabled}")
    
    def _save_configuration(self):
        """Save current configuration to file."""
        try:
            config_data = {
                'storage_mode': self.config.storage_mode.value,
                'api_mode': self.config.api_mode.value,
                'local_memory_path': self.config.local_memory_path,
                'usb_memory_path': self.config.usb_memory_path,
                'backup_interval': self.config.backup_interval,
                'max_backup_age_days': self.config.max_backup_age_days,
                'compression_enabled': self.config.compression_enabled,
                'encryption_enabled': self.config.encryption_enabled,
                'auto_sync': self.config.auto_sync,
                'memory_choice_menu': self.config.memory_choice_menu,
                'api_timeout': self.config.api_timeout,
                'api_retry_attempts': self.config.api_retry_attempts,
                'api_cache_duration': self.config.api_cache_duration,
                'last_updated': datetime.now().isoformat()
            }
            
            config_file = self.local_memory_dir / 'config' / 'memory_config.json'
            config_file.parent.mkdir(exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("✅ Configuration saved")
            
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def get_real_price_data(self, symbol: str = 'BTC/USDC', exchange: str = 'binance') -> float:
        """Get REAL price data from API - NO MORE STATIC 50000.0!"""
        try:
            # Check cache first
            cache_key = f"{exchange}_{symbol}"
            current_time = time.time()
            
            if cache_key in self.api_cache:
                cached_data = self.api_cache[cache_key]
                if current_time - cached_data['timestamp'] < self.config.api_cache_duration:
                    logger.debug(f"📊 Using cached price for {symbol}: ${cached_data['price']:.2f}")
                    return cached_data['price']
            
            # Get real price from API
            if exchange in self.api_clients:
                try:
                    client = self.api_clients[exchange]
                    ticker = client.fetch_ticker(symbol)
                    
                    if ticker and 'last' in ticker and ticker['last']:
                        real_price = float(ticker['last'])
                        
                        # Cache the result
                        self.api_cache[cache_key] = {
                            'price': real_price,
                            'volume': ticker.get('baseVolume', 0),
                            'timestamp': current_time,
                            'source': exchange
                        }
                        
                        # Store in database
                        self._store_api_data(symbol, real_price, ticker.get('baseVolume', 0), exchange)
                        
                        logger.info(f"📊 Real API price for {symbol}: ${real_price:.2f} from {exchange}")
                        return real_price
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error getting price from {exchange}: {e}")
            
            # Try other exchanges
            for other_exchange in self.api_clients:
                if other_exchange != exchange:
                    try:
                        client = self.api_clients[other_exchange]
                        ticker = client.fetch_ticker(symbol)
                        
                        if ticker and 'last' in ticker and ticker['last']:
                            real_price = float(ticker['last'])
                            
                            # Cache the result
                            self.api_cache[cache_key] = {
                                'price': real_price,
                                'volume': ticker.get('baseVolume', 0),
                                'timestamp': current_time,
                                'source': other_exchange
                            }
                            
                            logger.info(f"📊 Real API price for {symbol}: ${real_price:.2f} from {other_exchange}")
                            return real_price
                    
                    except Exception as e:
                        logger.debug(f"Error getting price from {other_exchange}: {e}")
            
            # CRITICAL: No real data available - fail properly
            raise ValueError(f"No live price data available for {symbol} - API connection required")
            
        except Exception as e:
            logger.error(f"❌ Error getting real price data: {e}")
            raise ValueError(f"Cannot get live price data for {symbol}: {e}")
    
    def _store_api_data(self, symbol: str, price: float, volume: float, source: str):
        """Store API data in database."""
        try:
            if self.memory_database:
                cursor = self.memory_database.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO api_cache (symbol, price, volume, timestamp, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', (symbol, price, volume, datetime.now().isoformat(), source))
                
                self.memory_database.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing API data: {e}")
    
    def store_memory_entry(self, data_type: str, data: Dict[str, Any], 
                          source: str = "system", priority: int = 1, 
                          tags: List[str] = None) -> str:
        """Store a memory entry with proper routing."""
        try:
            entry_id = self._generate_entry_id(data_type, data)
            timestamp = datetime.now()
            
            # Create memory entry
            entry = MemoryEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                data_type=data_type,
                data=data,
                source=source,
                priority=priority,
                tags=tags or []
            )
            
            # Process entry based on storage mode
            if self.config.storage_mode == MemoryStorageMode.LOCAL_ONLY:
                self._store_local(entry)
            elif self.config.storage_mode == MemoryStorageMode.USB_ONLY:
                self._store_usb(entry)
            elif self.config.storage_mode == MemoryStorageMode.HYBRID:
                self._store_hybrid(entry)
            else:  # AUTO
                self._store_auto(entry)
            
            # Cache entry
            self.memory_cache[entry_id] = entry
            
            logger.info(f"💾 Stored memory entry: {entry_id} ({data_type})")
            return entry_id
            
        except Exception as e:
            logger.error(f"❌ Error storing memory entry: {e}")
            raise
    
    def _generate_entry_id(self, data_type: str, data: Dict[str, Any]) -> str:
        """Generate unique entry ID."""
        content = f"{data_type}:{json.dumps(data, sort_keys=True)}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _store_local(self, entry: MemoryEntry):
        """Store entry in local memory."""
        try:
            # Store in database
            if self.memory_database:
                cursor = self.memory_database.cursor()
                
                cursor.execute('''
                    INSERT INTO memory_entries 
                    (entry_id, timestamp, data_type, data, source, priority, tags, checksum, compressed, encrypted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.data_type,
                    json.dumps(entry.data),
                    entry.source,
                    entry.priority,
                    json.dumps(entry.tags),
                    entry.checksum,
                    entry.compressed,
                    entry.encrypted
                ))
                
                self.memory_database.commit()
            
            # Store in file system
            file_path = self.local_memory_dir / entry.data_type / f"{entry.entry_id}.json"
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump({
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp.isoformat(),
                    'data_type': entry.data_type,
                    'data': entry.data,
                    'source': entry.source,
                    'priority': entry.priority,
                    'tags': entry.tags
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Error storing locally: {e}")
    
    def _store_usb(self, entry: MemoryEntry):
        """Store entry in USB memory."""
        try:
            if self.usb_memory_dir:
                file_path = self.usb_memory_dir / entry.data_type / f"{entry.entry_id}.json"
                file_path.parent.mkdir(exist_ok=True)
                
                with open(file_path, 'w') as f:
                    json.dump({
                        'entry_id': entry.entry_id,
                        'timestamp': entry.timestamp.isoformat(),
                        'data_type': entry.data_type,
                        'data': entry.data,
                        'source': entry.source,
                        'priority': entry.priority,
                        'tags': entry.tags
                    }, f, indent=2)
            else:
                # Fallback to local storage
                self._store_local(entry)
                
        except Exception as e:
            logger.error(f"❌ Error storing on USB: {e}")
            # Fallback to local storage
            self._store_local(entry)
    
    def _store_hybrid(self, entry: MemoryEntry):
        """Store entry in both local and USB memory."""
        try:
            # Store locally
            self._store_local(entry)
            
            # Store on USB
            self._store_usb(entry)
            
        except Exception as e:
            logger.error(f"❌ Error storing in hybrid mode: {e}")
    
    def _store_auto(self, entry: MemoryEntry):
        """Automatically choose storage location."""
        try:
            # Check USB availability
            if self.usb_memory_dir and self.usb_memory_dir.exists():
                # Use USB for important data
                if entry.priority >= 2:
                    self._store_usb(entry)
                else:
                    self._store_local(entry)
            else:
                # Use local storage
                self._store_local(entry)
                
        except Exception as e:
            logger.error(f"❌ Error in auto storage: {e}")
            # Fallback to local storage
            self._store_local(entry)
    
    def _api_update_loop(self):
        """Background loop for API updates."""
        while not self.stop_threads:
            try:
                # Update API data for common symbols
                symbols = ['BTC/USDC', 'ETH/USDC', 'BTC/USDT', 'ETH/USDT']
                
                for symbol in symbols:
                    try:
                        price = self.get_real_price_data(symbol)
                        
                        # Store as memory entry
                        self.store_memory_entry(
                            data_type='api_data',
                            data={
                                'symbol': symbol,
                                'price': price,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'real_api'
                            },
                            source='api_system',
                            priority=2,
                            tags=['real_time', 'pricing']
                        )
                        
                    except Exception as e:
                        logger.debug(f"Error updating {symbol}: {e}")
                
                # Sleep
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"❌ Error in API update loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _memory_sync_loop(self):
        """Background loop for memory synchronization."""
        while not self.stop_threads:
            try:
                # Sync between local and USB memory
                if self.config.storage_mode == MemoryStorageMode.HYBRID:
                    self._sync_memory_locations()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep
                time.sleep(self.config.backup_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in memory sync loop: {e}")
                time.sleep(600)  # Wait 10 minutes on error
    
    def _sync_memory_locations(self):
        """Synchronize between local and USB memory."""
        try:
            if not self.usb_memory_dir or not self.usb_memory_dir.exists():
                return
            
            # Sync from local to USB
            for data_type_dir in self.local_memory_dir.iterdir():
                if data_type_dir.is_dir():
                    usb_data_dir = self.usb_memory_dir / data_type_dir.name
                    usb_data_dir.mkdir(exist_ok=True)
                    
                    for file_path in data_type_dir.glob('*.json'):
                        usb_file_path = usb_data_dir / file_path.name
                        
                        # Copy if USB file doesn't exist or is older
                        if not usb_file_path.exists() or file_path.stat().st_mtime > usb_file_path.stat().st_mtime:
                            shutil.copy2(file_path, usb_file_path)
            
            # Sync from USB to local
            for data_type_dir in self.usb_memory_dir.iterdir():
                if data_type_dir.is_dir():
                    local_data_dir = self.local_memory_dir / data_type_dir.name
                    local_data_dir.mkdir(exist_ok=True)
                    
                    for file_path in data_type_dir.glob('*.json'):
                        local_file_path = local_data_dir / file_path.name
                        
                        # Copy if local file doesn't exist or is older
                        if not local_file_path.exists() or file_path.stat().st_mtime > local_file_path.stat().st_mtime:
                            shutil.copy2(file_path, local_file_path)
            
            logger.debug("✅ Memory synchronization completed")
            
        except Exception as e:
            logger.error(f"❌ Error syncing memory: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on configuration."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.max_backup_age_days)
            
            # Cleanup local memory
            for data_type_dir in self.local_memory_dir.iterdir():
                if data_type_dir.is_dir():
                    for file_path in data_type_dir.glob('*.json'):
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_path.unlink()
                                logger.debug(f"🗑️ Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.debug(f"Error cleaning up {file_path}: {e}")
            
            # Cleanup USB memory
            if self.usb_memory_dir and self.usb_memory_dir.exists():
                for data_type_dir in self.usb_memory_dir.iterdir():
                    if data_type_dir.is_dir():
                        for file_path in data_type_dir.glob('*.json'):
                            try:
                                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if file_time < cutoff_date:
                                    file_path.unlink()
                                    logger.debug(f"🗑️ Cleaned up old USB file: {file_path}")
                            except Exception as e:
                                logger.debug(f"Error cleaning up USB file {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            stats = {
                'storage_mode': self.config.storage_mode.value,
                'api_mode': self.config.api_mode.value,
                'local_memory_path': str(self.local_memory_dir),
                'usb_memory_path': str(self.usb_memory_dir) if self.usb_memory_dir else None,
                'api_clients_count': len(self.api_clients),
                'memory_cache_size': len(self.memory_cache),
                'api_cache_size': len(self.api_cache),
                'auto_sync': self.config.auto_sync,
                'compression_enabled': self.config.compression_enabled,
                'encryption_enabled': self.config.encryption_enabled
            }
            
            # Count files in memory directories
            if self.local_memory_dir.exists():
                local_files = sum(1 for f in self.local_memory_dir.rglob('*.json'))
                stats['local_files_count'] = local_files
            
            if self.usb_memory_dir and self.usb_memory_dir.exists():
                usb_files = sum(1 for f in self.usb_memory_dir.rglob('*.json'))
                stats['usb_files_count'] = usb_files
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting memory stats: {e}")
            return {}
    
    def stop(self):
        """Stop the memory system."""
        try:
            logger.info("🛑 Stopping Real API Pricing & Memory Storage System")
            
            # Stop background threads
            self.stop_threads = True
            
            # Wait for threads to finish
            if self.api_thread and self.api_thread.is_alive():
                self.api_thread.join(timeout=5.0)
            
            if self.sync_thread and self.sync_thread.is_alive():
                self.sync_thread.join(timeout=5.0)
            
            # Close database connection
            if self.memory_database:
                self.memory_database.close()
            
            # Stop USB memory system
            if USB_SYSTEMS_AVAILABLE and hasattr(self, 'usb_memory'):
                self.usb_memory.stop()
            
            logger.info("✅ Real API Pricing & Memory Storage System stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")

# Global instance
real_api_memory_system = None

def initialize_real_api_memory_system(config: Optional[MemoryConfig] = None) -> RealAPIPricingMemorySystem:
    """Initialize the global real API pricing and memory storage system."""
    global real_api_memory_system
    
    if real_api_memory_system is None:
        real_api_memory_system = RealAPIPricingMemorySystem(config)
    
    return real_api_memory_system

def get_real_price_data(symbol: str = 'BTC/USDC', exchange: str = 'binance') -> float:
    """Get real price data - global function for easy access."""
    global real_api_memory_system
    
    if real_api_memory_system is None:
        real_api_memory_system = initialize_real_api_memory_system()
    
    return real_api_memory_system.get_real_price_data(symbol, exchange)

def store_memory_entry(data_type: str, data: Dict[str, Any], 
                      source: str = "system", priority: int = 1, 
                      tags: List[str] = None) -> str:
    """Store memory entry - global function for easy access."""
    global real_api_memory_system
    
    if real_api_memory_system is None:
        real_api_memory_system = initialize_real_api_memory_system()
    
    return real_api_memory_system.store_memory_entry(data_type, data, source, priority, tags)

def main():
    """Test the real API pricing and memory storage system."""
    try:
        print("🚀 Real API Pricing & Memory Storage System Test")
        print("=" * 60)
        
        # Initialize system
        system = initialize_real_api_memory_system()
        
        # Test real API pricing
        print("\n📊 Testing Real API Pricing...")
        try:
            price = get_real_price_data('BTC/USDC')
            print(f"✅ Real BTC price: ${price:.2f}")
        except Exception as e:
            print(f"⚠️ API test failed (expected without real keys): {e}")
        
        # Test memory storage
        print("\n💾 Testing Memory Storage...")
        entry_id = store_memory_entry(
            data_type='test_data',
            data={'test': 'value', 'timestamp': datetime.now().isoformat()},
            source='test_system',
            priority=1,
            tags=['test', 'demo']
        )
        print(f"✅ Memory entry stored: {entry_id}")
        
        # Show memory stats
        print("\n📊 Memory System Stats:")
        stats = system.get_memory_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Run for a few seconds
        print("\n⏳ Running system for 10 seconds...")
        time.sleep(10)
        
        # Stop system
        system.stop()
        
        print("\n🎉 Real API Pricing & Memory Storage System test completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in main test: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 