#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Setup Script - Schwabot
================================

This script helps users set up the security system for Schwabot, including:

1. **API Key Configuration**: Securely store API keys for trading exchanges
2. **Security Settings**: Configure encryption and security parameters
3. **System Configuration**: Set up trading and system parameters
4. **Backup Configuration**: Configure backup and recovery settings
5. **Validation**: Verify all security settings are properly configured

The script ensures that all API connections, trading data, and sensitive
information are properly encrypted and secured for production use.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import our security components
from core.alpha256_encryption import Alpha256Encryption, get_encryption
from core.hash_config_manager import HashConfigManager, get_config_manager, ConfigPriority

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecuritySetup:
    """Interactive security setup for Schwabot."""
    
    def __init__(self):
        """Initialize the security setup."""
        self.encryption = None
        self.config_manager = None
        self.setup_data = {}
        
        # Supported exchanges
        self.supported_exchanges = [
            'binance', 'coinbase', 'kraken', 'kucoin', 'okx', 'bybit',
            'gate.io', 'huobi', 'bitfinex', 'gemini', 'bitstamp', 'coinbase_pro'
        ]
        
        logger.info("🔐 Security Setup initialized")
    
    def print_banner(self):
        """Print setup banner."""
        print("\n" + "="*80)
        print("🔐 SCHWABOT SECURITY SETUP")
        print("="*80)
        print("This script will help you configure the security system for Schwabot.")
        print("All API keys and sensitive data will be encrypted and secured.")
        print("="*80 + "\n")
    
    def get_user_input(self, prompt: str, default: str = "", required: bool = True, 
                      password: bool = False) -> str:
        """Get user input with validation."""
        while True:
            if password:
                import getpass
                value = getpass.getpass(prompt)
            else:
                value = input(prompt).strip()
            
            if not value and required:
                print("❌ This field is required. Please enter a value.")
                continue
            
            if not value and not required:
                return default
            
            return value
    
    def get_yes_no(self, prompt: str, default: str = "y") -> bool:
        """Get yes/no input from user."""
        while True:
            response = input(f"{prompt} (y/n) [{default}]: ").strip().lower()
            if not response:
                response = default
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("❌ Please enter 'y' or 'n'.")
    
    def get_selection(self, prompt: str, options: List[str], default: int = 0) -> str:
        """Get selection from a list of options."""
        print(f"\n{prompt}")
        for i, option in enumerate(options):
            marker = "→" if i == default else " "
            print(f"  {marker} {i+1}. {option}")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(options)}) [{default+1}]: ").strip()
                if not choice:
                    choice = default + 1
                else:
                    choice = int(choice)
                
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(options)}.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    async def setup_encryption(self):
        """Set up the encryption system."""
        print("\n🔐 ENCRYPTION SETUP")
        print("-" * 40)
        
        try:
            # Initialize encryption
            self.encryption = Alpha256Encryption()
            
            # Check if master key exists
            master_key_file = Path("config/keys/master.key")
            if master_key_file.exists():
                print("✅ Master encryption key found")
                use_password = self.get_yes_no("Do you want to protect the master key with a password?", "y")
                
                if use_password:
                    password = self.get_user_input("Enter master key password: ", password=True)
                    # Reinitialize with password
                    self.encryption = Alpha256Encryption(master_password=password)
                    print("✅ Master key protected with password")
            else:
                print("🔑 Generating new master encryption key...")
                use_password = self.get_yes_no("Do you want to protect the master key with a password?", "y")
                
                if use_password:
                    password = self.get_user_input("Enter master key password: ", password=True)
                    # Reinitialize with password
                    self.encryption = Alpha256Encryption(master_password=password)
                    print("✅ Master key generated and protected with password")
                else:
                    print("✅ Master key generated (stored without password protection)")
            
            # Show encryption status
            status = self.encryption.get_security_status()
            print(f"✅ Encryption Type: {status['encryption_type']}")
            print(f"✅ Hardware Acceleration: {status['hardware_accelerated']}")
            print(f"✅ Cryptography Library: {status['cryptography_available']}")
            
            self.setup_data['encryption'] = status
            return True
            
        except Exception as e:
            logger.error(f"❌ Encryption setup failed: {e}")
            return False
    
    async def setup_configuration(self):
        """Set up the configuration system."""
        print("\n⚙️ CONFIGURATION SETUP")
        print("-" * 40)
        
        try:
            # Initialize configuration manager
            self.config_manager = HashConfigManager()
            
            # System configuration
            print("\n📊 System Configuration:")
            debug_mode = self.get_yes_no("Enable debug mode?", "n")
            log_level = self.get_selection(
                "Select log level:",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                1  # Default to INFO
            )
            max_memory = int(self.get_user_input("Maximum memory usage (MB): ", "2048"))
            
            self.config_manager.set_config("debug_mode", debug_mode, "system")
            self.config_manager.set_config("log_level", log_level, "system")
            self.config_manager.set_config("max_memory_mb", max_memory, "system")
            
            # Trading configuration
            print("\n💰 Trading Configuration:")
            default_exchange = self.get_selection(
                "Select default trading exchange:",
                self.supported_exchanges,
                0  # Default to first exchange
            )
            max_position = float(self.get_user_input("Maximum position size (0.0-1.0): ", "0.1"))
            risk_percentage = float(self.get_user_input("Risk percentage per trade (0.1-10.0): ", "2.0"))
            trading_enabled = self.get_yes_no("Enable live trading?", "n")
            paper_trading = self.get_yes_no("Enable paper trading mode?", "y")
            
            self.config_manager.set_config("default_exchange", default_exchange, "trading")
            self.config_manager.set_config("max_position_size", max_position, "trading")
            self.config_manager.set_config("risk_percentage", risk_percentage, "trading")
            self.config_manager.set_config("trading_enabled", trading_enabled, "trading")
            self.config_manager.set_config("paper_trading", paper_trading, "trading")
            
            # API configuration
            print("\n🌐 API Configuration:")
            kobold_port = int(self.get_user_input("KoboldCPP port: ", "5001"))
            bridge_port = int(self.get_user_input("Bridge port: ", "5005"))
            enhanced_port = int(self.get_user_input("Enhanced interface port: ", "5006"))
            timeout = int(self.get_user_input("API timeout (seconds): ", "30"))
            
            self.config_manager.set_config("kobold_port", kobold_port, "api")
            self.config_manager.set_config("bridge_port", bridge_port, "api")
            self.config_manager.set_config("enhanced_port", enhanced_port, "api")
            self.config_manager.set_config("timeout_seconds", timeout, "api")
            
            # Security configuration
            print("\n🔒 Security Configuration:")
            session_timeout = int(self.get_user_input("Session timeout (minutes): ", "60"))
            max_login_attempts = int(self.get_user_input("Maximum login attempts: ", "5"))
            two_factor = self.get_yes_no("Enable two-factor authentication?", "n")
            
            self.config_manager.set_config("session_timeout_minutes", session_timeout, "security")
            self.config_manager.set_config("max_login_attempts", max_login_attempts, "security")
            self.config_manager.set_config("two_factor_enabled", two_factor, "security")
            
            # Show configuration status
            status = self.config_manager.get_config_status()
            print(f"✅ Total Configurations: {status['total_configs']}")
            print(f"✅ Encrypted Configurations: {status['encrypted_configs']}")
            print(f"✅ Schemas Loaded: {status['schemas_loaded']}")
            
            self.setup_data['configuration'] = status
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration setup failed: {e}")
            return False
    
    async def setup_api_keys(self):
        """Set up API keys for trading exchanges."""
        print("\n🔑 API KEY SETUP")
        print("-" * 40)
        
        try:
            api_keys = {}
            
            # Ask which exchanges to configure
            print("Select exchanges to configure API keys for:")
            selected_exchanges = []
            
            for exchange in self.supported_exchanges:
                if self.get_yes_no(f"Configure {exchange} API keys?", "n"):
                    selected_exchanges.append(exchange)
            
            if not selected_exchanges:
                print("⚠️ No exchanges selected. You can add API keys later.")
                return True
            
            # Configure API keys for selected exchanges
            for exchange in selected_exchanges:
                print(f"\n🔑 {exchange.upper()} Configuration:")
                print("-" * 30)
                
                api_key = self.get_user_input(f"Enter {exchange} API Key: ", password=True)
                api_secret = self.get_user_input(f"Enter {exchange} API Secret: ", password=True)
                
                # Get permissions
                permissions = []
                if self.get_yes_no("Enable read permissions?", "y"):
                    permissions.append("read")
                if self.get_yes_no("Enable trade permissions?", "y"):
                    permissions.append("trade")
                if self.get_yes_no("Enable withdraw permissions?", "n"):
                    permissions.append("withdraw")
                
                # Set expiration (optional)
                expires = None
                if self.get_yes_no("Set API key expiration?", "n"):
                    days = int(self.get_user_input("Expires in how many days?: ", "30"))
                    expires = datetime.now() + timedelta(days=days)
                
                # Store API key
                key_id = self.config_manager.set_api_config(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    permissions=permissions
                )
                
                api_keys[exchange] = {
                    'key_id': key_id,
                    'permissions': permissions,
                    'expires': expires.isoformat() if expires else None
                }
                
                print(f"✅ {exchange} API key stored successfully")
            
            # Test API key retrieval
            print("\n🧪 Testing API key retrieval...")
            for exchange in selected_exchanges:
                try:
                    config = self.config_manager.get_api_config(exchange)
                    print(f"✅ {exchange}: API key retrieved successfully")
                except Exception as e:
                    print(f"❌ {exchange}: Failed to retrieve API key - {e}")
            
            self.setup_data['api_keys'] = api_keys
            return True
            
        except Exception as e:
            logger.error(f"❌ API key setup failed: {e}")
            return False
    
    async def setup_backup(self):
        """Set up backup and recovery configuration."""
        print("\n💾 BACKUP SETUP")
        print("-" * 40)
        
        try:
            # Backup configuration
            enable_backup = self.get_yes_no("Enable automatic configuration backups?", "y")
            
            if enable_backup:
                backup_interval = int(self.get_user_input("Backup interval (hours): ", "24"))
                backup_path = self.get_user_input("Backup directory: ", "backups")
                
                self.config_manager.set_config("backup_enabled", True, "system")
                self.config_manager.set_config("backup_interval_hours", backup_interval, "system")
                self.config_manager.set_config("backup_path", backup_path, "system")
                
                # Create initial backup
                print("Creating initial encrypted backup...")
                backup_file = self.config_manager.backup_configs(backup_path)
                print(f"✅ Initial backup created: {backup_file}")
                
                self.setup_data['backup'] = {
                    'enabled': True,
                    'interval_hours': backup_interval,
                    'path': backup_path,
                    'initial_backup': backup_file
                }
            else:
                self.config_manager.set_config("backup_enabled", False, "system")
                self.setup_data['backup'] = {'enabled': False}
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup setup failed: {e}")
            return False
    
    async def validate_setup(self):
        """Validate the complete setup."""
        print("\n✅ SETUP VALIDATION")
        print("-" * 40)
        
        try:
            validation_results = []
            
            # Test encryption
            test_data = "Schwabot security test"
            encrypted = self.encryption.encrypt(test_data)
            decrypted = self.encryption.decrypt(encrypted)
            
            if decrypted == test_data:
                validation_results.append(("Encryption", "✅ PASS"))
            else:
                validation_results.append(("Encryption", "❌ FAIL"))
            
            # Test configuration
            debug_mode = self.config_manager.get_config("debug_mode", "system")
            if debug_mode is not None:
                validation_results.append(("Configuration", "✅ PASS"))
            else:
                validation_results.append(("Configuration", "❌ FAIL"))
            
            # Test API keys
            api_keys = self.encryption.list_api_keys()
            if api_keys:
                validation_results.append(("API Keys", f"✅ PASS ({len(api_keys)} keys)"))
            else:
                validation_results.append(("API Keys", "⚠️ WARNING (no keys configured)"))
            
            # Test backup
            backup_enabled = self.config_manager.get_config("backup_enabled", "system")
            if backup_enabled:
                validation_results.append(("Backup", "✅ PASS"))
            else:
                validation_results.append(("Backup", "⚠️ WARNING (backup disabled)"))
            
            # Print validation results
            for component, status in validation_results:
                print(f"{component:15} {status}")
            
            # Overall status
            passed = sum(1 for _, status in validation_results if "PASS" in status)
            total = len(validation_results)
            
            if passed == total:
                print(f"\n🎉 All validations passed! ({passed}/{total})")
                return True
            else:
                print(f"\n⚠️ Some validations failed ({passed}/{total})")
                return False
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    def save_setup_summary(self):
        """Save setup summary to file."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'setup_data': self.setup_data,
                'recommendations': [
                    "Keep your master key password secure",
                    "Regularly rotate API keys",
                    "Monitor system logs for security events",
                    "Keep all dependencies updated",
                    "Test your configuration regularly"
                ]
            }
            
            summary_file = "config/setup_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📄 Setup summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save setup summary: {e}")
    
    def print_final_instructions(self):
        """Print final setup instructions."""
        print("\n" + "="*80)
        print("🎉 SCHWABOT SECURITY SETUP COMPLETE")
        print("="*80)
        print("\nYour Schwabot system is now secured and ready for use!")
        print("\n📋 Next Steps:")
        print("1. Start the Schwabot system: python master_integration.py")
        print("2. Access the web interface at: http://localhost:5005")
        print("3. Test your API connections in the trading interface")
        print("4. Monitor system logs for any issues")
        print("\n🔒 Security Features Enabled:")
        print("• Alpha256 encryption for all sensitive data")
        print("• Secure API key storage and management")
        print("• Encrypted configuration files")
        print("• Session-based security")
        print("• Backup and recovery system")
        print("\n⚠️ Important Security Notes:")
        print("• Keep your master key password secure")
        print("• Regularly rotate API keys")
        print("• Monitor system logs for security events")
        print("• Keep all dependencies updated")
        print("• Test your configuration regularly")
        print("\n" + "="*80)
    
    async def run_setup(self):
        """Run the complete security setup."""
        try:
            self.print_banner()
            
            # Run setup steps
            steps = [
                ("Encryption System", self.setup_encryption),
                ("Configuration", self.setup_configuration),
                ("API Keys", self.setup_api_keys),
                ("Backup System", self.setup_backup),
                ("Validation", self.validate_setup)
            ]
            
            for step_name, step_func in steps:
                print(f"\n🔄 Running {step_name} setup...")
                success = await step_func()
                
                if not success:
                    print(f"❌ {step_name} setup failed. Please check the logs and try again.")
                    return False
                
                print(f"✅ {step_name} setup completed")
            
            # Save setup summary
            self.save_setup_summary()
            
            # Print final instructions
            self.print_final_instructions()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Setup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

async def main():
    """Main setup function."""
    try:
        # Check if running as main
        if __name__ != "__main__":
            return
        
        # Create necessary directories
        Path("config").mkdir(exist_ok=True)
        Path("config/keys").mkdir(exist_ok=True)
        Path("backups").mkdir(exist_ok=True)
        
        # Run setup
        setup = SecuritySetup()
        success = await setup.run_setup()
        
        if success:
            print("\n✅ Security setup completed successfully!")
            return 0
        else:
            print("\n❌ Security setup failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 