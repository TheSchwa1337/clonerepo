#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Schwabot System Operation Verification

This script verifies that all major components of the Schwabot system
are working correctly after the repository reorganization.

Usage:
    python verify_system_operation.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class SystemVerifier:
    """Comprehensive system verification for Schwabot."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def run_verification(self) -> Dict[str, Any]:
        """Run complete system verification."""
        print("🔍 Schwabot System Operation Verification")
        print("=" * 60)
        
        # Test 1: File Structure
        self._verify_file_structure()
        
        # Test 2: Import Tests
        self._verify_imports()
        
        # Test 3: CLI Interface
        self._verify_cli_interface()
        
        # Test 4: Web Interface
        self._verify_web_interface()
        
        # Test 5: Configuration
        self._verify_configuration()
        
        # Test 6: Dependencies
        self._verify_dependencies()
        
        # Test 7: Documentation
        self._verify_documentation()
        
        # Generate report
        return self._generate_report()
    
    def _verify_file_structure(self):
        """Verify that all required files and directories exist."""
        print("\n📁 Verifying File Structure...")
        
        required_files = [
            "README.md",
            "AOI_Base_Files_Schwabot/main.py",
            "AOI_Base_Files_Schwabot/launch_unified_interface.py",
            "AOI_Base_Files_Schwabot/requirements.txt",
            "docs/guides/getting_started.md",
            "docs/guides/user_guide.md",
            "docs/guides/web_interface.md",
            "docs/api/cli_reference.md",
            "docs/configuration/setup.md",
            "docs/development/architecture.md",
            "docs/development/contributing.md"
        ]
        
        required_dirs = [
            "AOI_Base_Files_Schwabot/core",
            "AOI_Base_Files_Schwabot/gui",
            "AOI_Base_Files_Schwabot/config",
            "AOI_Base_Files_Schwabot/api",
            "tests",
            "monitoring",
            "development",
            "data"
        ]
        
        all_good = True
        
        # Check files
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} - MISSING")
                all_good = False
                self.errors.append(f"Missing file: {file_path}")
        
        # Check directories
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                print(f"  ✅ {dir_path}/")
            else:
                print(f"  ❌ {dir_path}/ - MISSING")
                all_good = False
                self.errors.append(f"Missing directory: {dir_path}")
        
        self.results['file_structure'] = all_good
    
    def _verify_imports(self):
        """Verify that core modules can be imported."""
        print("\n📦 Verifying Module Imports...")
        
        # Add AOI_Base_Files_Schwabot to path
        sys.path.insert(0, 'AOI_Base_Files_Schwabot')
        
        core_modules = [
            "core.hash_config_manager",
            "core.risk_manager",
            "core.tensor_weight_memory",
            "gui.unified_schwabot_interface"
        ]
        
        all_good = True
        
        for module in core_modules:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError as e:
                print(f"  ❌ {module} - {e}")
                all_good = False
                self.errors.append(f"Import error in {module}: {e}")
            except Exception as e:
                print(f"  ⚠️  {module} - {e}")
                self.warnings.append(f"Warning in {module}: {e}")
        
        self.results['imports'] = all_good
    
    def _verify_cli_interface(self):
        """Verify that the CLI interface works."""
        print("\n💻 Verifying CLI Interface...")
        
        try:
            # Test help command
            result = subprocess.run(
                ["python", "AOI_Base_Files_Schwabot/main.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("  ✅ CLI help command works")
                
                # Test system status
                result = subprocess.run(
                    ["python", "AOI_Base_Files_Schwabot/main.py", "--system-status"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("  ✅ CLI system status works")
                    self.results['cli_interface'] = True
                else:
                    print(f"  ❌ CLI system status failed: {result.stderr}")
                    self.results['cli_interface'] = False
                    self.errors.append(f"CLI system status failed: {result.stderr}")
            else:
                print(f"  ❌ CLI help command failed: {result.stderr}")
                self.results['cli_interface'] = False
                self.errors.append(f"CLI help command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("  ❌ CLI command timed out")
            self.results['cli_interface'] = False
            self.errors.append("CLI command timed out")
        except Exception as e:
            print(f"  ❌ CLI verification failed: {e}")
            self.results['cli_interface'] = False
            self.errors.append(f"CLI verification failed: {e}")
    
    def _verify_web_interface(self):
        """Verify that the web interface launcher works."""
        print("\n🌐 Verifying Web Interface...")
        
        try:
            # Test help command
            result = subprocess.run(
                ["python", "AOI_Base_Files_Schwabot/launch_unified_interface.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("  ✅ Web interface launcher works")
                self.results['web_interface'] = True
            else:
                print(f"  ❌ Web interface launcher failed: {result.stderr}")
                self.results['web_interface'] = False
                self.errors.append(f"Web interface launcher failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("  ❌ Web interface command timed out")
            self.results['web_interface'] = False
            self.errors.append("Web interface command timed out")
        except Exception as e:
            print(f"  ❌ Web interface verification failed: {e}")
            self.results['web_interface'] = False
            self.errors.append(f"Web interface verification failed: {e}")
    
    def _verify_configuration(self):
        """Verify that configuration files are accessible."""
        print("\n⚙️  Verifying Configuration...")
        
        config_files = [
            "AOI_Base_Files_Schwabot/config/",
            "AOI_Base_Files_Schwabot/core/hash_config_manager.py"
        ]
        
        all_good = True
        
        for config_path in config_files:
            if os.path.exists(config_path):
                print(f"  ✅ {config_path}")
            else:
                print(f"  ❌ {config_path} - MISSING")
                all_good = False
                self.errors.append(f"Missing configuration: {config_path}")
        
        self.results['configuration'] = all_good
    
    def _verify_dependencies(self):
        """Verify that required dependencies are available."""
        print("\n📋 Verifying Dependencies...")
        
        required_packages = [
            "numpy",
            "pandas",
            "flask",
            "requests",
            "ccxt"
        ]
        
        all_good = True
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - NOT INSTALLED")
                all_good = False
                self.errors.append(f"Missing dependency: {package}")
        
        self.results['dependencies'] = all_good
    
    def _verify_documentation(self):
        """Verify that documentation is complete and accessible."""
        print("\n📚 Verifying Documentation...")
        
        doc_files = [
            "README.md",
            "docs/guides/getting_started.md",
            "docs/guides/user_guide.md",
            "docs/guides/web_interface.md",
            "docs/api/cli_reference.md",
            "docs/configuration/setup.md",
            "docs/development/architecture.md",
            "docs/development/contributing.md"
        ]
        
        all_good = True
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                # Check if file has content
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) > 100:  # At least 100 characters
                            print(f"  ✅ {doc_file}")
                        else:
                            print(f"  ⚠️  {doc_file} - EMPTY OR TOO SHORT")
                            self.warnings.append(f"Documentation file too short: {doc_file}")
                except Exception as e:
                    print(f"  ❌ {doc_file} - CANNOT READ: {e}")
                    all_good = False
                    self.errors.append(f"Cannot read documentation: {doc_file}")
            else:
                print(f"  ❌ {doc_file} - MISSING")
                all_good = False
                self.errors.append(f"Missing documentation: {doc_file}")
        
        self.results['documentation'] = all_good
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        print("\n" + "=" * 60)
        print("📊 VERIFICATION REPORT")
        print("=" * 60)
        
        # Calculate overall status
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Show detailed results
        print("\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Show errors
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        # Show warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Overall assessment
        if failed_tests == 0:
            print("\n🎉 SYSTEM VERIFICATION: PASSED")
            print("✅ All components are operational!")
        elif failed_tests <= 2:
            print("\n⚠️  SYSTEM VERIFICATION: MOSTLY PASSED")
            print("⚠️  Some minor issues detected, but system is functional.")
        else:
            print("\n❌ SYSTEM VERIFICATION: FAILED")
            print("❌ Multiple issues detected. Please fix before use.")
        
        return {
            'overall_status': failed_tests == 0,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings
        }

def main():
    """Main verification function."""
    verifier = SystemVerifier()
    report = verifier.run_verification()
    
    # Save report to file
    with open('system_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: system_verification_report.json")
    
    # Exit with appropriate code
    if report['overall_status']:
        print("\n🚀 System is ready for use!")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the issues before using the system.")
        sys.exit(1)

if __name__ == "__main__":
    main() 