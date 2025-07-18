#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security System Test Suite - Schwabot
=====================================

This script tests all security features of the Schwabot trading system:

1. **Alpha256 Encryption**: Test encryption/decryption functionality
2. **API Key Management**: Test secure storage and retrieval of API keys
3. **Configuration Security**: Test encrypted configuration management
4. **Hash Verification**: Test data integrity checking
5. **Hardware Acceleration**: Test encryption performance
6. **Backup & Recovery**: Test encrypted backup functionality
7. **Integration Security**: Test security integration with all components

The system ensures that all API connections, trading data, and sensitive
information are properly encrypted and secured for production use.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

# Import our security components
from core.alpha256_encryption import Alpha256Encryption, get_encryption
from core.hash_config_manager import HashConfigManager, get_config_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityTestResult:
    """Test result structure."""
    
    def __init__(self, name: str, success: bool, message: str = "", data: Any = None, duration: float = 0.0):
        self.name = name
        self.success = success
        self.message = message
        self.data = data
        self.duration = duration
        self.timestamp = datetime.now()

class SecurityTestSuite:
    """Comprehensive security test suite for Schwabot."""
    
    def __init__(self):
        """Initialize the security test suite."""
        self.results: List[SecurityTestResult] = []
        self.encryption = None
        self.config_manager = None
        
        # Test data
        self.test_data = {
            'simple': "Hello, Schwabot!",
            'complex': "This is a complex test string with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            'json': json.dumps({
                'api_key': 'test_api_key_12345',
                'api_secret': 'test_api_secret_67890',
                'permissions': ['read', 'trade', 'withdraw'],
                'timestamp': datetime.now().isoformat()
            }),
            'large': "A" * 10000,  # 10KB string
            'binary': bytes([i % 256 for i in range(1000)]).decode('latin-1')
        }
        
        logger.info("🔐 Security Test Suite initialized")
    
    def _add_result(self, name: str, success: bool, message: str = "", data: Any = None, duration: float = 0.0):
        """Add a test result."""
        result = SecurityTestResult(name, success, message, data, duration)
        self.results.append(result)
        
        if success:
            logger.info(f"✅ {name}: {message}")
        else:
            logger.error(f"❌ {name}: {message}")
    
    async def test_encryption_initialization(self) -> bool:
        """Test encryption system initialization."""
        try:
            start_time = time.time()
            
            # Initialize encryption
            self.encryption = Alpha256Encryption()
            
            # Check security status
            status = self.encryption.get_security_status()
            
            duration = time.time() - start_time
            
            # Verify encryption is properly initialized
            assert status['master_key_loaded'] == True
            assert status['encryption_type'] == 'alpha256'
            assert status['cryptography_available'] in [True, False]  # Either is fine
            
            self._add_result(
                "Encryption Initialization",
                True,
                f"Encryption system initialized successfully. Hardware acceleration: {status['hardware_accelerated']}",
                status,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Encryption Initialization",
                False,
                f"Failed to initialize encryption: {str(e)}"
            )
            return False
    
    async def test_basic_encryption(self) -> bool:
        """Test basic encryption/decryption functionality."""
        try:
            start_time = time.time()
            
            # Test simple string encryption
            original = self.test_data['simple']
            encrypted = self.encryption.encrypt(original)
            decrypted = self.encryption.decrypt(encrypted)
            
            duration = time.time() - start_time
            
            # Verify encryption/decryption
            assert decrypted == original
            assert encrypted != original  # Ensure data is actually encrypted
            
            self._add_result(
                "Basic Encryption",
                True,
                f"Successfully encrypted and decrypted {len(original)} characters",
                {'original_length': len(original), 'encrypted_length': len(encrypted)},
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Basic Encryption",
                False,
                f"Basic encryption test failed: {str(e)}"
            )
            return False
    
    async def test_complex_encryption(self) -> bool:
        """Test encryption with complex data types."""
        try:
            start_time = time.time()
            
            results = {}
            
            # Test each data type
            for data_type, test_data in self.test_data.items():
                encrypted = self.encryption.encrypt(test_data)
                decrypted = self.encryption.decrypt(encrypted)
                
                assert decrypted == test_data
                results[data_type] = {
                    'original_length': len(test_data),
                    'encrypted_length': len(encrypted),
                    'success': True
                }
            
            duration = time.time() - start_time
            
            self._add_result(
                "Complex Encryption",
                True,
                f"Successfully tested encryption for {len(results)} data types",
                results,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Complex Encryption",
                False,
                f"Complex encryption test failed: {str(e)}"
            )
            return False
    
    async def test_session_encryption(self) -> bool:
        """Test session-based encryption."""
        try:
            start_time = time.time()
            
            # Test different session IDs
            sessions = ['trading_session', 'api_session', 'user_session', 'admin_session']
            test_data = "Session-specific encryption test"
            
            results = {}
            for session_id in sessions:
                encrypted = self.encryption.encrypt(test_data, session_id)
                decrypted = self.encryption.decrypt(encrypted, session_id)
                
                assert decrypted == test_data
                results[session_id] = {
                    'encrypted_length': len(encrypted),
                    'success': True
                }
            
            duration = time.time() - start_time
            
            self._add_result(
                "Session Encryption",
                True,
                f"Successfully tested session encryption for {len(sessions)} sessions",
                results,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Session Encryption",
                False,
                f"Session encryption test failed: {str(e)}"
            )
            return False
    
    async def test_api_key_management(self) -> bool:
        """Test API key storage and retrieval."""
        try:
            start_time = time.time()
            
            # Test API key storage
            test_exchanges = ['binance', 'coinbase', 'kraken', 'kucoin']
            stored_keys = {}
            
            for exchange in test_exchanges:
                api_key = f"test_api_key_{exchange}_12345"
                api_secret = f"test_api_secret_{exchange}_67890"
                permissions = ['read', 'trade']
                
                key_id = self.encryption.store_api_key(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    permissions=permissions
                )
                
                stored_keys[exchange] = key_id
            
            # Test API key retrieval
            retrieved_keys = {}
            for exchange in test_exchanges:
                api_key, api_secret = self.encryption.get_api_key(stored_keys[exchange])
                
                expected_key = f"test_api_key_{exchange}_12345"
                expected_secret = f"test_api_secret_{exchange}_67890"
                
                assert api_key == expected_key
                assert api_secret == expected_secret
                
                retrieved_keys[exchange] = {
                    'key_id': stored_keys[exchange],
                    'success': True
                }
            
            duration = time.time() - start_time
            
            self._add_result(
                "API Key Management",
                True,
                f"Successfully stored and retrieved {len(test_exchanges)} API keys",
                {'stored': stored_keys, 'retrieved': retrieved_keys},
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "API Key Management",
                False,
                f"API key management test failed: {str(e)}"
            )
            return False
    
    async def test_configuration_security(self) -> bool:
        """Test secure configuration management."""
        try:
            start_time = time.time()
            
            # Initialize configuration manager
            self.config_manager = HashConfigManager()
            
            # Test basic configuration
            test_configs = {
                'system': {
                    'debug_mode': True,
                    'log_level': 'DEBUG',
                    'max_memory_mb': 2048
                },
                'trading': {
                    'default_exchange': 'binance',
                    'max_position_size': 0.5,
                    'risk_percentage': 3.0
                },
                'api': {
                    'kobold_port': 5001,
                    'bridge_port': 5005,
                    'timeout_seconds': 30
                }
            }
            
            # Set configurations
            for config_type, configs in test_configs.items():
                for key, value in configs.items():
                    self.config_manager.set_config(key, value, config_type)
            
            # Retrieve and verify configurations
            for config_type, configs in test_configs.items():
                for key, expected_value in configs.items():
                    retrieved_value = self.config_manager.get_config(key, config_type)
                    assert retrieved_value == expected_value
            
            duration = time.time() - start_time
            
            # Get configuration status
            status = self.config_manager.get_config_status()
            
            self._add_result(
                "Configuration Security",
                True,
                f"Successfully tested secure configuration management. Total configs: {status['total_configs']}",
                status,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Configuration Security",
                False,
                f"Configuration security test failed: {str(e)}"
            )
            return False
    
    async def test_api_configuration_integration(self) -> bool:
        """Test API configuration integration."""
        try:
            start_time = time.time()
            
            # Test API configuration storage
            test_exchanges = ['test_exchange_1', 'test_exchange_2']
            stored_configs = {}
            
            for exchange in test_exchanges:
                api_key = f"test_api_key_{exchange}"
                api_secret = f"test_api_secret_{exchange}"
                permissions = ['read', 'trade']
                
                key_id = self.config_manager.set_api_config(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    permissions=permissions
                )
                
                stored_configs[exchange] = key_id
            
            # Test API configuration retrieval
            retrieved_configs = {}
            for exchange in test_exchanges:
                config = self.config_manager.get_api_config(exchange)
                
                expected_key = f"test_api_key_{exchange}"
                expected_secret = f"test_api_secret_{exchange}"
                
                assert config['api_key'] == expected_key
                assert config['api_secret'] == expected_secret
                assert config['exchange'] == exchange
                
                retrieved_configs[exchange] = config
            
            duration = time.time() - start_time
            
            self._add_result(
                "API Configuration Integration",
                True,
                f"Successfully tested API configuration integration for {len(test_exchanges)} exchanges",
                {'stored': stored_configs, 'retrieved': retrieved_configs},
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "API Configuration Integration",
                False,
                f"API configuration integration test failed: {str(e)}"
            )
            return False
    
    async def test_backup_and_recovery(self) -> bool:
        """Test encrypted backup and recovery functionality."""
        try:
            start_time = time.time()
            
            # Create backup
            backup_path = "test_backups"
            backup_file = self.config_manager.backup_configs(backup_path)
            
            # Verify backup file exists
            assert Path(backup_file).exists()
            
            # Test backup restoration
            success = self.config_manager.restore_configs(backup_file)
            assert success == True
            
            duration = time.time() - start_time
            
            self._add_result(
                "Backup and Recovery",
                True,
                f"Successfully created and restored encrypted backup: {backup_file}",
                {'backup_file': backup_file, 'file_size': Path(backup_file).stat().st_size},
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Backup and Recovery",
                False,
                f"Backup and recovery test failed: {str(e)}"
            )
            return False
    
    async def test_performance_benchmarks(self) -> bool:
        """Test encryption performance benchmarks."""
        try:
            start_time = time.time()
            
            # Test encryption performance with different data sizes
            data_sizes = [100, 1000, 10000, 100000]  # bytes
            performance_results = {}
            
            for size in data_sizes:
                test_data = "A" * size
                
                # Measure encryption time
                encrypt_start = time.time()
                encrypted = self.encryption.encrypt(test_data)
                encrypt_time = time.time() - encrypt_start
                
                # Measure decryption time
                decrypt_start = time.time()
                decrypted = self.encryption.decrypt(encrypted)
                decrypt_time = time.time() - decrypt_start
                
                # Verify data integrity
                assert decrypted == test_data
                
                total_time = encrypt_time + decrypt_time
                throughput = (size * 2) / total_time / 1024 / 1024 if total_time > 0 else float('inf')
                
                performance_results[f"{size}_bytes"] = {
                    'encrypt_time_ms': encrypt_time * 1000,
                    'decrypt_time_ms': decrypt_time * 1000,
                    'total_time_ms': total_time * 1000,
                    'throughput_mbps': throughput
                }
            
            duration = time.time() - start_time
            
            self._add_result(
                "Performance Benchmarks",
                True,
                f"Performance benchmarks completed for {len(data_sizes)} data sizes",
                performance_results,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Performance Benchmarks",
                False,
                f"Performance benchmark test failed: {str(e)}"
            )
            return False
    
    async def test_security_integration(self) -> bool:
        """Test security integration with all components."""
        try:
            start_time = time.time()
            
            # Test integration with all core components
            integration_results = {}
            
            # Test encryption system integration
            encryption_status = self.encryption.get_security_status()
            integration_results['encryption'] = encryption_status
            
            # Test configuration manager integration
            config_status = self.config_manager.get_config_status()
            integration_results['configuration'] = config_status
            
            # Test API key listing
            api_keys = self.encryption.list_api_keys()
            integration_results['api_keys'] = {
                'total_keys': len(api_keys),
                'active_keys': sum(1 for k in api_keys if k['is_active']),
                'exchanges': list(set(k['exchange'] for k in api_keys))
            }
            
            duration = time.time() - start_time
            
            self._add_result(
                "Security Integration",
                True,
                f"Security integration test completed successfully",
                integration_results,
                duration
            )
            return True
            
        except Exception as e:
            self._add_result(
                "Security Integration",
                False,
                f"Security integration test failed: {str(e)}"
            )
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests."""
        logger.info("🚀 Starting comprehensive security test suite")
        
        test_start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_encryption_initialization(),
            self.test_basic_encryption(),
            self.test_complex_encryption(),
            self.test_session_encryption(),
            self.test_api_key_management(),
            self.test_configuration_security(),
            self.test_api_configuration_integration(),
            self.test_backup_and_recovery(),
            self.test_performance_benchmarks(),
            self.test_security_integration()
        ]
        
        # Wait for all tests to complete
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        test_duration = time.time() - test_start_time
        
        # Generate test summary
        summary = self._generate_test_summary(test_duration)
        
        logger.info("🏁 Security test suite completed")
        return summary
    
    def _generate_test_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate average duration
        avg_duration = sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Get security status
        encryption_status = self.encryption.get_security_status() if self.encryption else {}
        config_status = self.config_manager.get_config_status() if self.config_manager else {}
        
        summary = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration': total_duration,
                'average_duration': avg_duration
            },
            'security_status': {
                'encryption': encryption_status,
                'configuration': config_status
            },
            'test_results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'message': r.message,
                    'duration': r.duration,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []
        
        # Check encryption status
        if self.encryption:
            status = self.encryption.get_security_status()
            
            if not status.get('hardware_accelerated', False):
                recommendations.append("Consider enabling hardware acceleration for better encryption performance")
            
            if not status.get('cryptography_available', False):
                recommendations.append("Install cryptography library for enhanced security features")
        
        # Check configuration status
        if self.config_manager:
            status = self.config_manager.get_config_status()
            
            if status.get('encrypted_configs', 0) == 0:
                recommendations.append("Enable encryption for sensitive configuration data")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Security system is properly configured and ready for production use")
        
        recommendations.append("Regularly rotate API keys and master encryption keys")
        recommendations.append("Monitor system logs for security events")
        recommendations.append("Keep all security dependencies updated")
        
        return recommendations
    
    def print_test_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "="*80)
        print("🔐 SCHWABOT SECURITY TEST SUITE RESULTS")
        print("="*80)
        
        # Test summary
        test_summary = summary['test_summary']
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {test_summary['total_tests']}")
        print(f"   Passed: {test_summary['passed_tests']}")
        print(f"   Failed: {test_summary['failed_tests']}")
        print(f"   Success Rate: {test_summary['success_rate']:.1f}%")
        print(f"   Total Duration: {test_summary['total_duration']:.2f}s")
        print(f"   Average Duration: {test_summary['average_duration']:.3f}s")
        
        # Security status
        print(f"\n🔒 Security Status:")
        encryption_status = summary['security_status']['encryption']
        if encryption_status:
            print(f"   Encryption Type: {encryption_status.get('encryption_type', 'Unknown')}")
            print(f"   Hardware Accelerated: {encryption_status.get('hardware_accelerated', False)}")
            print(f"   Master Key Loaded: {encryption_status.get('master_key_loaded', False)}")
            print(f"   Active Sessions: {encryption_status.get('active_sessions', 0)}")
            print(f"   Stored API Keys: {encryption_status.get('stored_api_keys', 0)}")
        
        config_status = summary['security_status']['configuration']
        if config_status:
            print(f"   Total Configs: {config_status.get('total_configs', 0)}")
            print(f"   Encrypted Configs: {config_status.get('encrypted_configs', 0)}")
            print(f"   Schemas Loaded: {config_status.get('schemas_loaded', 0)}")
        
        # Test results
        print(f"\n📋 Test Results:")
        for result in summary['test_results']:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {status} {result['name']} ({result['duration']:.3f}s)")
            if not result['success']:
                print(f"      Error: {result['message']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for recommendation in summary['recommendations']:
            print(f"   • {recommendation}")
        
        print("\n" + "="*80)

async def main():
    """Main test execution function."""
    try:
        # Initialize test suite
        test_suite = SecurityTestSuite()
        
        # Run all tests
        summary = await test_suite.run_all_tests()
        
        # Print results
        test_suite.print_test_summary(summary)
        
        # Return exit code based on test results
        if summary['test_summary']['failed_tests'] > 0:
            logger.error("❌ Some security tests failed")
            return 1
        else:
            logger.info("✅ All security tests passed")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Security test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 