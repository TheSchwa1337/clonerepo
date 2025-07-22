#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Schwabot Deployment Validator
================================

Comprehensive validation script to ensure your Schwabot system is ready for production deployment.
This script checks security, performance, integration, and compliance requirements.

Usage:
    python deployment_validator.py [--full] [--security] [--performance] [--compliance]
"""

import os
import sys
import time
import json
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validator for Schwabot."""
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'checks': {},
            'overall_status': 'UNKNOWN',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete deployment validation."""
        logger.info("🚀 Starting Schwabot Deployment Validation")
        logger.info("=" * 60)
        
        # Run all validation checks
        self._validate_environment()
        self._validate_security()
        self._validate_dependencies()
        self._validate_configuration()
        self._validate_performance()
        self._validate_integration()
        self._validate_compliance()
        
        # Generate final report
        self._generate_report()
        
        return self.results
    
    def _validate_environment(self):
        """Validate environment setup."""
        logger.info("🔍 Validating Environment Setup...")
        
        checks = {}
        
        # Check required environment variables
        required_vars = [
            'SCHWABOT_ENVIRONMENT',
            'SCHWABOT_ENCRYPTION_KEY',
            'SCHWABOT_LOG_LEVEL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        checks['environment_variables'] = {
            'status': 'PASS' if not missing_vars else 'FAIL',
            'missing': missing_vars,
            'message': f"Missing variables: {missing_vars}" if missing_vars else "All required variables set"
        }
        
        # Check Python version
        python_version = sys.version_info
        checks['python_version'] = {
            'status': 'PASS' if python_version >= (3, 8) else 'FAIL',
            'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'message': "Python 3.8+ required" if python_version < (3, 8) else "Python version OK"
        }
        
        # Check file permissions
        env_file = Path('.env')
        if env_file.exists():
            stat = env_file.stat()
            checks['file_permissions'] = {
                'status': 'PASS' if stat.st_mode & 0o777 == 0o600 else 'WARN',
                'permissions': oct(stat.st_mode & 0o777),
                'message': "File permissions should be 600 for .env"
            }
        
        self.results['checks']['environment'] = checks
    
    def _validate_security(self):
        """Validate security configuration."""
        logger.info("🔐 Validating Security Configuration...")
        
        checks = {}
        
        # Check for hardcoded secrets
        try:
            result = subprocess.run([
                'grep', '-r', r'api_key\|secret\|password', '.',
                '--exclude-dir=node_modules', '--exclude-dir=.git', '--exclude-dir=__pycache__'
            ], capture_output=True, text=True)
            
            if result.stdout.strip():
                checks['hardcoded_secrets'] = {
                    'status': 'FAIL',
                    'message': 'Hardcoded secrets found in code',
                    'details': result.stdout.strip()[:200] + '...' if len(result.stdout) > 200 else result.stdout
                }
            else:
                checks['hardcoded_secrets'] = {
                    'status': 'PASS',
                    'message': 'No hardcoded secrets found'
                }
        except Exception as e:
            checks['hardcoded_secrets'] = {
                'status': 'WARN',
                'message': f'Could not check for hardcoded secrets: {e}'
            }
        
        # Check encryption key length
        encryption_key = os.getenv('SCHWABOT_ENCRYPTION_KEY', '')
        checks['encryption_key'] = {
            'status': 'PASS' if len(encryption_key) >= 32 else 'FAIL',
            'length': len(encryption_key),
            'message': 'Encryption key should be at least 32 characters'
        }
        
        # Check SSL configuration
        ssl_enabled = os.getenv('SCHWABOT_API_SSL_ENABLED', 'false').lower() == 'true'
        checks['ssl_configuration'] = {
            'status': 'PASS' if ssl_enabled else 'WARN',
            'enabled': ssl_enabled,
            'message': 'SSL should be enabled for production'
        }
        
        self.results['checks']['security'] = checks
    
    def _validate_dependencies(self):
        """Validate required dependencies."""
        logger.info("📦 Validating Dependencies...")
        
        checks = {}
        
        # Core dependencies to check
        core_deps = [
            'numpy', 'pandas', 'ccxt', 'flask', 'cryptography',
            'requests', 'yaml', 'psutil', 'asyncio'
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        checks['core_dependencies'] = {
            'status': 'PASS' if not missing_deps else 'FAIL',
            'missing': missing_deps,
            'message': f"Missing dependencies: {missing_deps}" if missing_deps else "All core dependencies available"
        }
        
        # Check Schwabot modules
        schwabot_modules = [
            'core.brain_trading_engine',
            'core.clean_unified_math',
            'core.symbolic_profit_router'
        ]
        
        missing_modules = []
        for module in schwabot_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        checks['schwabot_modules'] = {
            'status': 'PASS' if not missing_modules else 'FAIL',
            'missing': missing_modules,
            'message': f"Missing Schwabot modules: {missing_modules}" if missing_modules else "All Schwabot modules available"
        }
        
        self.results['checks']['dependencies'] = checks
    
    def _validate_configuration(self):
        """Validate configuration files."""
        logger.info("⚙️ Validating Configuration...")
        
        checks = {}
        
        # Check configuration files
        config_files = [
            'config/master_integration.yaml',
            'config/production.env.template',
            'config/security_config.yaml'
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not Path(config_file).exists():
                missing_configs.append(config_file)
        
        checks['configuration_files'] = {
            'status': 'PASS' if not missing_configs else 'WARN',
            'missing': missing_configs,
            'message': f"Missing config files: {missing_configs}" if missing_configs else "All config files present"
        }
        
        # Validate YAML configuration
        try:
            import yaml
            with open('config/master_integration.yaml', 'r') as f:
                config = yaml.safe_load(f)
            checks['yaml_validation'] = {
                'status': 'PASS',
                'message': 'YAML configuration is valid'
            }
        except Exception as e:
            checks['yaml_validation'] = {
                'status': 'FAIL',
                'message': f'YAML configuration error: {e}'
            }
        
        self.results['checks']['configuration'] = checks
    
    def _validate_performance(self):
        """Validate performance requirements."""
        logger.info("⚡ Validating Performance...")
        
        checks = {}
        
        # Check system resources
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            checks['cpu_usage'] = {
                'status': 'PASS' if cpu_percent < 80 else 'WARN',
                'usage': cpu_percent,
                'message': f'CPU usage: {cpu_percent:.1f}%'
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            checks['memory_usage'] = {
                'status': 'PASS' if memory.percent < 80 else 'WARN',
                'usage': memory.percent,
                'available_gb': memory.available / (1024**3),
                'message': f'Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)'
            }
            
            # Disk space
            disk = psutil.disk_usage('.')
            checks['disk_space'] = {
                'status': 'PASS' if disk.percent < 90 else 'WARN',
                'usage': disk.percent,
                'free_gb': disk.free / (1024**3),
                'message': f'Disk usage: {disk.percent:.1f}% ({disk.free / (1024**3):.1f}GB free)'
            }
            
        except ImportError:
            checks['system_resources'] = {
                'status': 'WARN',
                'message': 'psutil not available for resource monitoring'
            }
        
        # Performance benchmark
        start_time = time.time()
        try:
            # Simple mathematical operation test
            import numpy as np
            for _ in range(1000):
                np.random.rand(100, 100)
            math_time = time.time() - start_time
            
            checks['mathematical_performance'] = {
                'status': 'PASS' if math_time < 1.0 else 'WARN',
                'time_ms': math_time * 1000,
                'message': f'Mathematical operations: {math_time*1000:.2f}ms for 1000 operations'
            }
        except Exception as e:
            checks['mathematical_performance'] = {
                'status': 'FAIL',
                'message': f'Mathematical performance test failed: {e}'
            }
        
        self.results['checks']['performance'] = checks
    
    def _validate_integration(self):
        """Validate system integration."""
        logger.info("🔗 Validating System Integration...")
        
        checks = {}
        
        # Check if test files exist
        test_files = [
            'tests/integration/test_complete_production_system.py',
            'tests/integration/test_mathematical_integration.py',
            'tests/integration/test_core_integration.py'
        ]
        
        missing_tests = []
        for test_file in test_files:
            if not Path(test_file).exists():
                missing_tests.append(test_file)
        
        checks['test_files'] = {
            'status': 'PASS' if not missing_tests else 'WARN',
            'missing': missing_tests,
            'message': f"Missing test files: {missing_tests}" if missing_tests else "All test files present"
        }
        
        # Check main entry points
        entry_points = [
            'AOI_Base_Files_Schwabot/run_schwabot.py',
            'AOI_Base_Files_Schwabot/launch_unified_interface.py',
            'AOI_Base_Files_Schwabot/launch_unified_mathematical_trading_system.py'
        ]
        
        missing_entries = []
        for entry in entry_points:
            if not Path(entry).exists():
                missing_entries.append(entry)
        
        checks['entry_points'] = {
            'status': 'PASS' if not missing_entries else 'FAIL',
            'missing': missing_entries,
            'message': f"Missing entry points: {missing_entries}" if missing_entries else "All entry points present"
        }
        
        self.results['checks']['integration'] = checks
    
    def _validate_compliance(self):
        """Validate compliance requirements."""
        logger.info("📋 Validating Compliance...")
        
        checks = {}
        
        # Check audit logging configuration
        audit_enabled = os.getenv('SCHWABOT_AUDIT_LOG_ENABLED', 'false').lower() == 'true'
        checks['audit_logging'] = {
            'status': 'PASS' if audit_enabled else 'WARN',
            'enabled': audit_enabled,
            'message': 'Audit logging should be enabled for compliance'
        }
        
        # Check trade logging
        trade_logging = os.getenv('SCHWABOT_ENABLE_TRADE_LOGGING', 'false').lower() == 'true'
        checks['trade_logging'] = {
            'status': 'PASS' if trade_logging else 'WARN',
            'enabled': trade_logging,
            'message': 'Trade logging should be enabled for compliance'
        }
        
        # Check log directory
        log_dir = Path('logs')
        if log_dir.exists():
            checks['log_directory'] = {
                'status': 'PASS',
                'message': 'Log directory exists'
            }
        else:
            checks['log_directory'] = {
                'status': 'WARN',
                'message': 'Log directory should be created'
            }
        
        self.results['checks']['compliance'] = checks
    
    def _generate_report(self):
        """Generate final validation report."""
        logger.info("📊 Generating Validation Report...")
        
        # Count results
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        for category, checks in self.results['checks'].items():
            for check_name, check_result in checks.items():
                total_checks += 1
                status = check_result['status']
                if status == 'PASS':
                    passed_checks += 1
                elif status == 'FAIL':
                    failed_checks += 1
                elif status == 'WARN':
                    warning_checks += 1
        
        # Determine overall status
        if failed_checks > 0:
            self.results['overall_status'] = 'FAIL'
        elif warning_checks > 0:
            self.results['overall_status'] = 'WARN'
        else:
            self.results['overall_status'] = 'PASS'
        
        # Add summary
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'warnings': warning_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
        
        # Print results
        self._print_results()
    
    def _print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("🚀 SCHWABOT DEPLOYMENT VALIDATION RESULTS")
        print("=" * 60)
        
        summary = self.results['summary']
        print(f"📊 Overall Status: {self.results['overall_status']}")
        print(f"✅ Passed: {summary['passed']}/{summary['total_checks']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️ Warnings: {summary['warnings']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for category, checks in self.results['checks'].items():
            print(f"\n🔍 {category.upper()}:")
            for check_name, check_result in checks.items():
                status_icon = {
                    'PASS': '✅',
                    'FAIL': '❌',
                    'WARN': '⚠️'
                }.get(check_result['status'], '❓')
                
                print(f"  {status_icon} {check_name}: {check_result['message']}")
        
        # Recommendations
        if self.results['overall_status'] != 'PASS':
            print(f"\n🚨 CRITICAL ISSUES TO RESOLVE:")
            print("-" * 40)
            
            for category, checks in self.results['checks'].items():
                for check_name, check_result in checks.items():
                    if check_result['status'] == 'FAIL':
                        print(f"  ❌ {category}.{check_name}: {check_result['message']}")
        
        # Final recommendation
        if self.results['overall_status'] == 'PASS':
            print(f"\n🎉 DEPLOYMENT READY!")
            print("Your Schwabot system is ready for production deployment.")
        elif self.results['overall_status'] == 'WARN':
            print(f"\n⚠️ DEPLOYMENT READY WITH WARNINGS")
            print("Your system is ready but consider addressing the warnings.")
        else:
            print(f"\n❌ DEPLOYMENT NOT READY")
            print("Please resolve the critical issues before deployment.")
        
        # Save report
        report_file = f"deployment_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Schwabot Deployment Validator')
    parser.add_argument('--full', action='store_true', help='Run full validation')
    parser.add_argument('--security', action='store_true', help='Run security validation only')
    parser.add_argument('--performance', action='store_true', help='Run performance validation only')
    parser.add_argument('--compliance', action='store_true', help='Run compliance validation only')
    
    args = parser.parse_args()
    
    validator = DeploymentValidator()
    
    if args.security:
        validator._validate_security()
        validator._generate_report()
    elif args.performance:
        validator._validate_performance()
        validator._generate_report()
    elif args.compliance:
        validator._validate_compliance()
        validator._generate_report()
    else:
        # Run full validation by default
        validator.run_full_validation()
    
    # Exit with appropriate code
    if validator.results['overall_status'] == 'FAIL':
        sys.exit(1)
    elif validator.results['overall_status'] == 'WARN':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 