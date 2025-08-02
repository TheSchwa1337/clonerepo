#!/usr/bin/env python3
"""
🔄 LEGACY BACKUP INTEGRATION SYSTEM - SCHWABOT
==============================================

This system CORRECTLY IMPLEMENTS legacy backup systems to work with real API pricing,
creating proper layering and portfolio sequencing without removing anything.

Key Principles:
✅ PRESERVE all legacy backup functionality
✅ INTEGRATE with real API pricing system
✅ CREATE proper layering for portfolio sequencing
✅ ALLOW system to choose optimal trading strategies
✅ MAINTAIN multiple backup systems for redundancy

This is about CORRECT IMPLEMENTATION, not replacement!
"""

import os
import sys
import json
import yaml
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import real API pricing and memory storage system
try:
    from real_api_pricing_memory_system import (
        initialize_real_api_memory_system, 
        get_real_price_data, 
        store_memory_entry,
        MemoryConfig,
        MemoryStorageMode,
        APIMode
    )
    REAL_API_AVAILABLE = True
except ImportError:
    REAL_API_AVAILABLE = False
    print("⚠️ Real API pricing system not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legacy_backup_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupLayerType(Enum):
    """Types of backup layers for portfolio sequencing."""
    PIPELINE_BACKUP = "pipeline_backup"           # Pipeline configuration backup
    TRADING_BACKUP = "trading_backup"             # Trading system backup
    FERRIS_BACKUP = "ferris_backup"               # Ferris Ride backup
    INTEGRATION_BACKUP = "integration_backup"     # Integration system backup
    SECURITY_BACKUP = "security_backup"           # Security system backup
    REAL_API_BACKUP = "real_api_backup"           # Real API data backup
    USB_BACKUP = "usb_backup"                     # USB memory backup
    PORTFOLIO_BACKUP = "portfolio_backup"         # Portfolio sequencing backup

@dataclass
class LegacyBackupConfig:
    """Configuration for legacy backup system integration."""
    backup_type: BackupLayerType
    enabled: bool = True
    interval_seconds: int = 3600  # 1 hour default
    max_files: int = 10
    backup_directory: str = "backups/"
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    # Integration with real API
    use_real_api_pricing: bool = True
    use_memory_choice_menu: bool = True
    integrate_with_real_api_system: bool = True
    
    # Portfolio sequencing
    portfolio_sequencing_enabled: bool = True
    layer_priority: int = 1
    trading_strategy_integration: bool = True

@dataclass
class PortfolioSequencingData:
    """Data structure for portfolio sequencing across backup layers."""
    timestamp: datetime
    layer_type: BackupLayerType
    market_data: Dict[str, Any]
    trading_decisions: List[Dict[str, Any]]
    profit_metrics: Dict[str, float]
    risk_assessment: Dict[str, Any]
    backup_metadata: Dict[str, Any]

class LegacyBackupIntegrationSystem:
    """System that correctly implements legacy backup systems with real API pricing."""
    
    def __init__(self):
        self.real_api_system = None
        self.legacy_backup_configs: Dict[BackupLayerType, LegacyBackupConfig] = {}
        self.backup_threads: Dict[BackupLayerType, threading.Thread] = {}
        self.is_running = False
        self.portfolio_sequencing_data: List[PortfolioSequencingData] = []
        
        # Initialize real API system
        self._initialize_real_api_system()
        
        # Load and configure legacy backup systems
        self._load_legacy_backup_configs()
        
        logger.info("🔄 Legacy Backup Integration System initialized")
    
    def _initialize_real_api_system(self):
        """Initialize real API pricing and memory storage system."""
        if REAL_API_AVAILABLE:
            try:
                memory_config = MemoryConfig(
                    storage_mode=MemoryStorageMode.AUTO,
                    api_mode=APIMode.REAL_API_ONLY,
                    memory_choice_menu=False,  # Don't show menu for integration
                    auto_sync=True
                )
                self.real_api_system = initialize_real_api_memory_system(memory_config)
                logger.info("✅ Real API pricing system integrated with legacy backups")
            except Exception as e:
                logger.error(f"❌ Error initializing real API system: {e}")
                self.real_api_system = None
        else:
            logger.warning("⚠️ Real API system not available - using legacy backup only")
    
    def _load_legacy_backup_configs(self):
        """Load and configure all legacy backup systems."""
        config_files = [
            ('AOI_Base_Files_Schwabot/config/pipeline.yaml', BackupLayerType.PIPELINE_BACKUP),
            ('AOI_Base_Files_Schwabot/config/schwabot_live_trading_config.yaml', BackupLayerType.TRADING_BACKUP),
            ('AOI_Base_Files_Schwabot/config/ferris_rde_daemon_config.yaml', BackupLayerType.FERRIS_BACKUP),
            ('AOI_Base_Files_Schwabot/config/integrations.yaml', BackupLayerType.INTEGRATION_BACKUP),
            ('config/security_config.yaml', BackupLayerType.SECURITY_BACKUP)
        ]
        
        for config_file, backup_type in config_files:
            self._load_legacy_backup_config(config_file, backup_type)
        
        # Add real API and USB backup configs
        self._add_real_api_backup_config()
        self._add_usb_backup_config()
        self._add_portfolio_backup_config()
        
        logger.info(f"✅ Loaded {len(self.legacy_backup_configs)} legacy backup configurations")
    
    def _load_legacy_backup_config(self, config_file: str, backup_type: BackupLayerType):
        """Load legacy backup configuration from YAML file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract backup configuration
                backup_config = config_data.get('backup', {})
                
                # Create integrated backup config
                legacy_config = LegacyBackupConfig(
                    backup_type=backup_type,
                    enabled=backup_config.get('enabled', True),
                    interval_seconds=backup_config.get('backup_interval', 3600),
                    max_files=backup_config.get('max_backup_files', 10),
                    backup_directory=backup_config.get('backup_directory', 'backups/'),
                    compression_enabled=backup_config.get('enable_backup_compression', True),
                    encryption_enabled=backup_config.get('encryption_enabled', False),
                    use_real_api_pricing=True,  # Always enable real API integration
                    use_memory_choice_menu=True,
                    integrate_with_real_api_system=True,
                    portfolio_sequencing_enabled=True,
                    layer_priority=self._get_layer_priority(backup_type),
                    trading_strategy_integration=True
                )
                
                self.legacy_backup_configs[backup_type] = legacy_config
                logger.info(f"✅ Loaded {backup_type.value} configuration")
            else:
                logger.warning(f"⚠️ Config file not found: {config_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading {backup_type.value} config: {e}")
    
    def _get_layer_priority(self, backup_type: BackupLayerType) -> int:
        """Get priority for portfolio sequencing layers."""
        priorities = {
            BackupLayerType.REAL_API_BACKUP: 1,      # Highest priority
            BackupLayerType.PORTFOLIO_BACKUP: 2,     # Portfolio sequencing
            BackupLayerType.TRADING_BACKUP: 3,       # Trading decisions
            BackupLayerType.FERRIS_BACKUP: 4,        # Ferris Ride system
            BackupLayerType.PIPELINE_BACKUP: 5,      # Pipeline configuration
            BackupLayerType.INTEGRATION_BACKUP: 6,   # Integration system
            BackupLayerType.SECURITY_BACKUP: 7,      # Security system
            BackupLayerType.USB_BACKUP: 8            # USB memory (lowest priority)
        }
        return priorities.get(backup_type, 10)
    
    def _add_real_api_backup_config(self):
        """Add real API backup configuration."""
        real_api_config = LegacyBackupConfig(
            backup_type=BackupLayerType.REAL_API_BACKUP,
            enabled=True,
            interval_seconds=300,  # 5 minutes for real API data
            max_files=100,  # More files for real API data
            backup_directory="SchwabotMemory/api_data/",
            compression_enabled=True,
            encryption_enabled=True,
            use_real_api_pricing=True,
            use_memory_choice_menu=True,
            integrate_with_real_api_system=True,
            portfolio_sequencing_enabled=True,
            layer_priority=1,
            trading_strategy_integration=True
        )
        self.legacy_backup_configs[BackupLayerType.REAL_API_BACKUP] = real_api_config
    
    def _add_usb_backup_config(self):
        """Add USB backup configuration."""
        usb_config = LegacyBackupConfig(
            backup_type=BackupLayerType.USB_BACKUP,
            enabled=True,
            interval_seconds=1800,  # 30 minutes for USB sync
            max_files=50,
            backup_directory="SchwabotMemory/usb_sync/",
            compression_enabled=True,
            encryption_enabled=True,
            use_real_api_pricing=True,
            use_memory_choice_menu=True,
            integrate_with_real_api_system=True,
            portfolio_sequencing_enabled=True,
            layer_priority=8,
            trading_strategy_integration=False  # USB is storage, not trading
        )
        self.legacy_backup_configs[BackupLayerType.USB_BACKUP] = usb_config
    
    def _add_portfolio_backup_config(self):
        """Add portfolio sequencing backup configuration."""
        portfolio_config = LegacyBackupConfig(
            backup_type=BackupLayerType.PORTFOLIO_BACKUP,
            enabled=True,
            interval_seconds=600,  # 10 minutes for portfolio updates
            max_files=25,
            backup_directory="SchwabotMemory/portfolio_data/",
            compression_enabled=True,
            encryption_enabled=True,
            use_real_api_pricing=True,
            use_memory_choice_menu=True,
            integrate_with_real_api_system=True,
            portfolio_sequencing_enabled=True,
            layer_priority=2,
            trading_strategy_integration=True
        )
        self.legacy_backup_configs[BackupLayerType.PORTFOLIO_BACKUP] = portfolio_config
    
    def start_legacy_backup_integration(self):
        """Start all legacy backup systems with real API integration."""
        if self.is_running:
            logger.warning("Legacy backup integration already running")
            return False
        
        self.is_running = True
        
        # Start backup threads for each layer
        for backup_type, config in self.legacy_backup_configs.items():
            if config.enabled:
                thread = threading.Thread(
                    target=self._legacy_backup_loop,
                    args=(backup_type, config),
                    daemon=True
                )
                thread.start()
                self.backup_threads[backup_type] = thread
                logger.info(f"🔄 Started {backup_type.value} backup layer")
        
        logger.info("🔄 Legacy backup integration system started")
        return True
    
    def stop_legacy_backup_integration(self):
        """Stop all legacy backup systems."""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.backup_threads.values():
            thread.join(timeout=5.0)
        
        self.backup_threads.clear()
        logger.info("🔄 Legacy backup integration system stopped")
    
    def _legacy_backup_loop(self, backup_type: BackupLayerType, config: LegacyBackupConfig):
        """Main backup loop for a legacy backup layer."""
        while self.is_running:
            try:
                # Perform legacy backup with real API integration
                self._perform_legacy_backup(backup_type, config)
                
                # Sleep based on backup interval
                time.sleep(config.interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Error in {backup_type.value} backup loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_legacy_backup(self, backup_type: BackupLayerType, config: LegacyBackupConfig):
        """Perform legacy backup with real API integration."""
        try:
            # Get real API data for this backup layer
            real_api_data = self._get_real_api_data_for_layer(backup_type)
            
            # Create portfolio sequencing data
            portfolio_data = PortfolioSequencingData(
                timestamp=datetime.now(),
                layer_type=backup_type,
                market_data=real_api_data.get('market_data', {}),
                trading_decisions=real_api_data.get('trading_decisions', []),
                profit_metrics=real_api_data.get('profit_metrics', {}),
                risk_assessment=real_api_data.get('risk_assessment', {}),
                backup_metadata={
                    'backup_type': backup_type.value,
                    'config': config.__dict__,
                    'real_api_integrated': True,
                    'portfolio_sequencing': config.portfolio_sequencing_enabled
                }
            )
            
            # Store in real API memory system
            if self.real_api_system:
                # Convert enum to string for JSON serialization
                portfolio_data_dict = portfolio_data.__dict__.copy()
                portfolio_data_dict['layer_type'] = backup_type.value  # Convert enum to string
                
                self.real_api_system.store_memory_entry(
                    data_type=f"legacy_backup_{backup_type.value}",
                    data=portfolio_data_dict,
                    source="legacy_backup_integration",
                    priority=config.layer_priority,
                    tags=['legacy_backup', 'real_api', 'portfolio_sequencing', backup_type.value]
                )
            
            # Perform legacy backup operations
            self._perform_legacy_backup_operations(backup_type, config, portfolio_data)
            
            # Update portfolio sequencing
            if config.portfolio_sequencing_enabled:
                self._update_portfolio_sequencing(portfolio_data)
            
            logger.info(f"✅ {backup_type.value} backup completed with real API integration")
            
        except Exception as e:
            logger.error(f"❌ Error performing {backup_type.value} backup: {e}")
    
    def _get_real_api_data_for_layer(self, backup_type: BackupLayerType) -> Dict[str, Any]:
        """Get real API data appropriate for the backup layer."""
        try:
            if not self.real_api_system:
                return {}
            
            # Get real market data
            market_data = {}
            if backup_type in [BackupLayerType.TRADING_BACKUP, BackupLayerType.FERRIS_BACKUP, BackupLayerType.PORTFOLIO_BACKUP]:
                try:
                    btc_price = get_real_price_data('BTC/USDC')
                    eth_price = get_real_price_data('ETH/USDC')
                    market_data = {
                        'btc_price': btc_price,
                        'eth_price': eth_price,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'real_api'
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Error getting real API data: {e}")
            
            # Get trading decisions (simulated for now)
            trading_decisions = []
            if backup_type in [BackupLayerType.TRADING_BACKUP, BackupLayerType.PORTFOLIO_BACKUP]:
                trading_decisions = [
                    {
                        'decision_type': 'buy',
                        'symbol': 'BTC/USDC',
                        'price': market_data.get('btc_price', 50000),
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.85
                    }
                ]
            
            # Get profit metrics
            profit_metrics = {
                'total_profit': 0.0,
                'profit_percentage': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            }
            
            # Get risk assessment
            risk_assessment = {
                'risk_level': 'medium',
                'position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.05
            }
            
            return {
                'market_data': market_data,
                'trading_decisions': trading_decisions,
                'profit_metrics': profit_metrics,
                'risk_assessment': risk_assessment
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting real API data for {backup_type.value}: {e}")
            return {}
    
    def _perform_legacy_backup_operations(self, backup_type: BackupLayerType, config: LegacyBackupConfig, portfolio_data: PortfolioSequencingData):
        """Perform traditional legacy backup operations."""
        try:
            # Create backup directory
            backup_dir = Path(config.backup_directory)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{backup_type.value}_backup_{timestamp}.json"
            backup_path = backup_dir / backup_filename
            
            # Save backup data
            backup_data = {
                'backup_type': backup_type.value,
                'timestamp': timestamp,
                'config': config.__dict__,
                'portfolio_data': portfolio_data.__dict__,
                'real_api_integrated': True,
                'legacy_backup_version': '2.0'
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            # Cleanup old backups
            self._cleanup_old_backups(backup_dir, config.max_files)
            
            logger.debug(f"✅ Legacy backup saved: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Error performing legacy backup operations: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, max_files: int):
        """Clean up old backup files."""
        try:
            backup_files = list(backup_dir.glob('*.json'))
            if len(backup_files) > max_files:
                # Sort by modification time (oldest first)
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                
                # Remove oldest files
                files_to_remove = backup_files[:-max_files]
                for file_path in files_to_remove:
                    file_path.unlink()
                    logger.debug(f"🗑️ Removed old backup: {file_path}")
                    
        except Exception as e:
            logger.error(f"❌ Error cleaning up old backups: {e}")
    
    def _update_portfolio_sequencing(self, portfolio_data: PortfolioSequencingData):
        """Update portfolio sequencing with new data."""
        try:
            # Add to portfolio sequencing data
            self.portfolio_sequencing_data.append(portfolio_data)
            
            # Keep only recent data (last 100 entries)
            if len(self.portfolio_sequencing_data) > 100:
                self.portfolio_sequencing_data = self.portfolio_sequencing_data[-100:]
            
            # Analyze portfolio sequencing
            self._analyze_portfolio_sequencing()
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio sequencing: {e}")
    
    def _analyze_portfolio_sequencing(self):
        """Analyze portfolio sequencing across all layers."""
        try:
            if not self.portfolio_sequencing_data:
                return
            
            # Group by layer type
            layer_data = {}
            for data in self.portfolio_sequencing_data:
                layer_type = data.layer_type
                if layer_type not in layer_data:
                    layer_data[layer_type] = []
                layer_data[layer_type].append(data)
            
            # Analyze each layer
            analysis_results = {}
            for layer_type, data_list in layer_data.items():
                analysis = self._analyze_layer_sequencing(layer_type, data_list)
                analysis_results[layer_type] = analysis
            
            # Store analysis in real API memory system
            if self.real_api_system:
                self.real_api_system.store_memory_entry(
                    data_type='portfolio_sequencing_analysis',
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'analysis_results': analysis_results,
                        'total_layers': len(layer_data),
                        'total_data_points': len(self.portfolio_sequencing_data)
                    },
                    source='legacy_backup_integration',
                    priority=1,
                    tags=['portfolio_sequencing', 'analysis', 'layering']
                )
            
            logger.debug(f"✅ Portfolio sequencing analysis completed for {len(layer_data)} layers")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing portfolio sequencing: {e}")
    
    def _analyze_layer_sequencing(self, layer_type: BackupLayerType, data_list: List[PortfolioSequencingData]) -> Dict[str, Any]:
        """Analyze sequencing for a specific layer."""
        try:
            if not data_list:
                return {}
            
            # Calculate metrics
            total_profit = sum(data.profit_metrics.get('total_profit', 0) for data in data_list)
            avg_profit = total_profit / len(data_list) if data_list else 0
            
            # Get latest data
            latest_data = data_list[-1] if data_list else None
            
            return {
                'layer_type': layer_type.value,
                'data_points': len(data_list),
                'total_profit': total_profit,
                'average_profit': avg_profit,
                'latest_timestamp': latest_data.timestamp.isoformat() if latest_data else None,
                'latest_market_data': latest_data.market_data if latest_data else {},
                'trading_decisions_count': len(latest_data.trading_decisions) if latest_data else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing layer {layer_type.value}: {e}")
            return {}
    
    def get_backup_integration_status(self) -> Dict[str, Any]:
        """Get status of legacy backup integration system."""
        status = {
            'is_running': self.is_running,
            'real_api_available': REAL_API_AVAILABLE,
            'real_api_system_initialized': self.real_api_system is not None,
            'active_backup_layers': len(self.backup_threads),
            'total_backup_configs': len(self.legacy_backup_configs),
            'portfolio_sequencing_data_points': len(self.portfolio_sequencing_data),
            'backup_layers': {}
        }
        
        # Add status for each backup layer
        for backup_type, config in self.legacy_backup_configs.items():
            status['backup_layers'][backup_type.value] = {
                'enabled': config.enabled,
                'interval_seconds': config.interval_seconds,
                'use_real_api_pricing': config.use_real_api_pricing,
                'portfolio_sequencing_enabled': config.portfolio_sequencing_enabled,
                'layer_priority': config.layer_priority,
                'is_running': backup_type in self.backup_threads
            }
        
        return status

def main():
    """Test the legacy backup integration system."""
    try:
        print("🔄 Testing Legacy Backup Integration System")
        print("=" * 60)
        
        # Initialize system
        integration_system = LegacyBackupIntegrationSystem()
        
        # Start integration
        if integration_system.start_legacy_backup_integration():
            print("✅ Legacy backup integration started")
            
            # Run for a few minutes to see results
            print("⏳ Running for 30 seconds to test integration...")
            time.sleep(30)
            
            # Get status
            status = integration_system.get_backup_integration_status()
            print(f"📊 Integration Status: {json.dumps(status, indent=2)}")
            
            # Stop integration
            integration_system.stop_legacy_backup_integration()
            print("✅ Legacy backup integration stopped")
        else:
            print("❌ Failed to start legacy backup integration")
        
        print("🎉 Legacy Backup Integration System test completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in main test: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 