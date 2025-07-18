#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Code Quality Checker for Schwabot Unified System
=============================================================

This script performs comprehensive code quality checks on the entire Schwabot
unified trading system, including:

1. Code formatting (Black, autopep8)
2. Linting (flake8, pylint)
3. Type checking (mypy)
4. Import validation
5. Integration testing
6. System validation

This ensures the complete system is ready for production use.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    check_name: str
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemValidationResult:
    """Result of system validation."""
    component_name: str
    success: bool = False
    import_errors: List[str] = field(default_factory=list)
    runtime_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComprehensiveCodeQualityChecker:
    """Comprehensive code quality checker for Schwabot unified system."""
    
    def __init__(self):
        """Initialize the quality checker."""
        self.project_root = Path(__file__).parent
        self.core_dir = self.project_root / "core"
        self.results: List[QualityCheckResult] = []
        self.system_results: List[SystemValidationResult] = []
        
        # Configuration
        self.max_line_length = 120
        self.ignore_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            "venv",
            "env",
            ".pytest_cache",
            "*.pyc",
            "*.pyo"
        ]
        
        # Files to check
        self.python_files = self._find_python_files()
        
        logger.info(f"🔍 Found {len(self.python_files)} Python files to check")
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        # Only look in core directory and project root
        search_dirs = [self.project_root, self.core_dir]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for root, dirs, files in os.walk(search_dir):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        # Only include files in our project
                        if 'clonerepo' in str(file_path) or 'core' in str(file_path):
                            python_files.append(file_path)
        
        return python_files
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks."""
        logger.info("🚀 Starting comprehensive code quality checks...")
        
        start_time = time.time()
        
        # Run code quality checks
        await self._run_black_formatting()
        await self._run_autopep8_formatting()
        await self._run_flake8_linting()
        await self._run_pylint_analysis()
        await self._run_mypy_type_checking()
        await self._run_import_validation()
        
        # Run system validation
        await self._run_system_validation()
        await self._run_integration_tests()
        
        # Generate report
        total_time = time.time() - start_time
        report = self._generate_report(total_time)
        
        logger.info("✅ All quality checks completed")
        return report
    
    async def _run_black_formatting(self):
        """Run Black code formatting check."""
        logger.info("🎨 Running Black formatting check...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="Black Formatting")
        
        try:
            # Check if Black is available
            black_check = subprocess.run(
                ["black", "--version"],
                capture_output=True,
                text=True
            )
            
            if black_check.returncode != 0:
                result.errors.append("Black not available - install with: pip install black")
                result.success = False
            else:
                # Run Black check on all files
                black_result = subprocess.run(
                    ["black", "--check", "--line-length", str(self.max_line_length)] + 
                    [str(f) for f in self.python_files],
                    capture_output=True,
                    text=True
                )
                
                if black_result.returncode != 0:
                    result.errors.append(f"Black formatting issues found:\n{black_result.stdout}")
                    result.success = False
                else:
                    result.success = True
                    result.metadata["formatted_files"] = len(self.python_files)
        
        except Exception as e:
            result.errors.append(f"Black check failed: {e}")
            result.success = False
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_autopep8_formatting(self):
        """Run autopep8 formatting check."""
        logger.info("🔧 Running autopep8 formatting check...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="autopep8 Formatting")
        
        try:
            # Check if autopep8 is available
            autopep8_check = subprocess.run(
                ["autopep8", "--version"],
                capture_output=True,
                text=True
            )
            
            if autopep8_check.returncode != 0:
                result.warnings.append("autopep8 not available - install with: pip install autopep8")
                result.success = True  # Not critical
            else:
                # Run autopep8 check on core files
                core_files = [f for f in self.python_files if "core" in str(f)]
                
                for file_path in core_files:
                    autopep8_result = subprocess.run(
                        ["autopep8", "--diff", "--max-line-length", str(self.max_line_length), str(file_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if autopep8_result.stdout:
                        result.warnings.append(f"Formatting suggestions for {file_path.name}")
                
                result.success = True
                result.metadata["checked_files"] = len(core_files)
        
        except Exception as e:
            result.warnings.append(f"autopep8 check failed: {e}")
            result.success = True  # Not critical
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_flake8_linting(self):
        """Run flake8 linting check."""
        logger.info("🔍 Running flake8 linting check...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="flake8 Linting")
        
        try:
            # Check if flake8 is available
            flake8_check = subprocess.run(
                ["flake8", "--version"],
                capture_output=True,
                text=True
            )
            
            if flake8_check.returncode != 0:
                result.errors.append("flake8 not available - install with: pip install flake8")
                result.success = False
            else:
                # Run flake8 on all files
                flake8_result = subprocess.run(
                    ["flake8", "--max-line-length", str(self.max_line_length)] + 
                    [str(f) for f in self.python_files],
                    capture_output=True,
                    text=True
                )
                
                if flake8_result.returncode != 0:
                    result.errors.append(f"flake8 issues found:\n{flake8_result.stdout}")
                    result.success = False
                else:
                    result.success = True
                    result.metadata["linted_files"] = len(self.python_files)
        
        except Exception as e:
            result.errors.append(f"flake8 check failed: {e}")
            result.success = False
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_pylint_analysis(self):
        """Run pylint analysis."""
        logger.info("🔍 Running pylint analysis...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="pylint Analysis")
        
        try:
            # Check if pylint is available
            pylint_check = subprocess.run(
                ["pylint", "--version"],
                capture_output=True,
                text=True
            )
            
            if pylint_check.returncode != 0:
                result.warnings.append("pylint not available - install with: pip install pylint")
                result.success = True  # Not critical
            else:
                # Run pylint on core files
                core_files = [f for f in self.python_files if "core" in str(f)]
                
                for file_path in core_files:
                    pylint_result = subprocess.run(
                        ["pylint", "--disable=C0114,C0115,C0116", str(file_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if pylint_result.stdout:
                        lines = pylint_result.stdout.strip().split('\n')
                        for line in lines:
                            if ':' in line and any(level in line for level in ['error', 'warning']):
                                result.warnings.append(f"{file_path.name}: {line}")
                
                result.success = True
                result.metadata["analyzed_files"] = len(core_files)
        
        except Exception as e:
            result.warnings.append(f"pylint analysis failed: {e}")
            result.success = True  # Not critical
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_mypy_type_checking(self):
        """Run mypy type checking."""
        logger.info("🔍 Running mypy type checking...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="mypy Type Checking")
        
        try:
            # Check if mypy is available
            mypy_check = subprocess.run(
                ["mypy", "--version"],
                capture_output=True,
                text=True
            )
            
            if mypy_check.returncode != 0:
                result.warnings.append("mypy not available - install with: pip install mypy")
                result.success = True  # Not critical
            else:
                # Run mypy on core files
                core_files = [f for f in self.python_files if "core" in str(f)]
                
                mypy_result = subprocess.run(
                    ["mypy", "--ignore-missing-imports"] + [str(f) for f in core_files],
                    capture_output=True,
                    text=True
                )
                
                if mypy_result.stdout:
                    result.warnings.append(f"Type checking issues:\n{mypy_result.stdout}")
                
                result.success = True
                result.metadata["type_checked_files"] = len(core_files)
        
        except Exception as e:
            result.warnings.append(f"mypy type checking failed: {e}")
            result.success = True  # Not critical
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_import_validation(self):
        """Validate imports in all files."""
        logger.info("🔍 Running import validation...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="Import Validation")
        
        try:
            import_errors = []
            
            for file_path in self.python_files:
                try:
                    # Try to import the module
                    module_name = str(file_path.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
                    
                    # Skip test files and main scripts
                    if 'test' in module_name or module_name.endswith('_main'):
                        continue
                    
                    # Add project root to path
                    sys.path.insert(0, str(self.project_root))
                    
                    # Try to import
                    __import__(module_name)
                    
                except ImportError as e:
                    import_errors.append(f"{file_path.name}: {e}")
                except Exception as e:
                    import_errors.append(f"{file_path.name}: {e}")
            
            if import_errors:
                result.errors.extend(import_errors)
                result.success = False
            else:
                result.success = True
                result.metadata["validated_files"] = len(self.python_files)
        
        except Exception as e:
            result.errors.append(f"Import validation failed: {e}")
            result.success = False
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_system_validation(self):
        """Validate the unified system components."""
        logger.info("🔍 Running system validation...")
        
        start_time = time.time()
        
        # Test core components
        core_components = [
            "core.schwabot_unified_interface",
            "core.koboldcpp_integration",
            "core.visual_layer_controller",
            "core.tick_loader",
            "core.signal_cache",
            "core.registry_writer",
            "core.json_server",
            "core.utils.windows_cli_compatibility",
            "core.utils.math_utils",
            "core.memory_stack.ai_command_sequencer",
            "core.memory_stack.execution_validator",
            "core.memory_stack.memory_key_allocator"
        ]
        
        for component in core_components:
            result = SystemValidationResult(component_name=component)
            
            try:
                # Try to import
                module = __import__(component, fromlist=['*'])
                result.success = True
                result.metadata["imported"] = True
                
                # Try to create instance if it's a class
                if hasattr(module, component.split('.')[-1]):
                    class_name = component.split('.')[-1]
                    cls = getattr(module, class_name)
                    if hasattr(cls, '__init__'):
                        try:
                            instance = cls()
                            result.metadata["instantiated"] = True
                        except Exception as e:
                            result.runtime_errors.append(f"Instantiation failed: {e}")
                
            except ImportError as e:
                result.import_errors.append(str(e))
                result.success = False
            except Exception as e:
                result.runtime_errors.append(str(e))
                result.success = False
            
            self.system_results.append(result)
        
        logger.info(f"✅ System validation completed for {len(core_components)} components")
    
    async def _run_integration_tests(self):
        """Run integration tests."""
        logger.info("🧪 Running integration tests...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="Integration Tests")
        
        try:
            # Test the unified interface
            from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
            
            # Create interface
            interface = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
            
            # Check initialization
            if interface.initialized:
                result.success = True
                result.metadata["interface_initialized"] = True
                result.metadata["mode"] = interface.mode.value
                
                # Check components
                components_status = {
                    "kobold_integration": interface.kobold_integration is not None,
                    "visual_controller": interface.visual_controller is not None,
                    "tick_loader": interface.tick_loader is not None,
                    "signal_cache": interface.signal_cache is not None,
                    "registry_writer": interface.registry_writer is not None,
                    "json_server": interface.json_server is not None
                }
                
                result.metadata["components_status"] = components_status
                
                # Check if all components are available
                if all(components_status.values()):
                    result.metadata["all_components_available"] = True
                else:
                    missing = [k for k, v in components_status.items() if not v]
                    result.warnings.append(f"Missing components: {missing}")
            else:
                result.errors.append("Interface initialization failed")
                result.success = False
        
        except Exception as e:
            result.errors.append(f"Integration test failed: {e}")
            result.success = False
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "summary": {
                "total_checks": len(self.results),
                "passed_checks": sum(1 for r in self.results if r.success),
                "failed_checks": sum(1 for r in self.results if not r.success),
                "total_errors": sum(len(r.errors) for r in self.results),
                "total_warnings": sum(len(r.warnings) for r in self.results)
            },
            "system_validation": {
                "total_components": len(self.system_results),
                "successful_components": sum(1 for r in self.system_results if r.success),
                "failed_components": sum(1 for r in self.system_results if not r.success)
            },
            "detailed_results": {
                "quality_checks": [
                    {
                        "name": r.check_name,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "errors": r.errors,
                        "warnings": r.warnings,
                        "metadata": r.metadata
                    }
                    for r in self.results
                ],
                "system_components": [
                    {
                        "name": r.component_name,
                        "success": r.success,
                        "import_errors": r.import_errors,
                        "runtime_errors": r.runtime_errors,
                        "metadata": r.metadata
                    }
                    for r in self.system_results
                ]
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print the quality report."""
        print("\n" + "=" * 80)
        print("🔍 SCHWABOT UNIFIED SYSTEM - COMPREHENSIVE QUALITY REPORT")
        print("=" * 80)
        
        # Summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   Passed: {summary['passed_checks']} ✅")
        print(f"   Failed: {summary['failed_checks']} ❌")
        print(f"   Total Errors: {summary['total_errors']}")
        print(f"   Total Warnings: {summary['total_warnings']}")
        
        # System validation
        sys_val = report["system_validation"]
        print(f"\n🔧 SYSTEM VALIDATION:")
        print(f"   Components: {sys_val['total_components']}")
        print(f"   Successful: {sys_val['successful_components']} ✅")
        print(f"   Failed: {sys_val['failed_components']} ❌")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        
        for check in report["detailed_results"]["quality_checks"]:
            status = "✅ PASS" if check["success"] else "❌ FAIL"
            print(f"   {status} {check['name']} ({check['execution_time']:.2f}s)")
            
            if check["errors"]:
                for error in check["errors"][:3]:  # Show first 3 errors
                    print(f"      ❌ {error}")
                if len(check["errors"]) > 3:
                    print(f"      ... and {len(check['errors']) - 3} more errors")
            
            if check["warnings"]:
                for warning in check["warnings"][:3]:  # Show first 3 warnings
                    print(f"      ⚠️  {warning}")
                if len(check["warnings"]) > 3:
                    print(f"      ... and {len(check['warnings']) - 3} more warnings")
        
        # System components
        print(f"\n🔧 SYSTEM COMPONENTS:")
        for component in report["detailed_results"]["system_components"]:
            status = "✅ OK" if component["success"] else "❌ FAIL"
            print(f"   {status} {component['name']}")
            
            if component["import_errors"]:
                for error in component["import_errors"]:
                    print(f"      ❌ Import: {error}")
            
            if component["runtime_errors"]:
                for error in component["runtime_errors"]:
                    print(f"      ❌ Runtime: {error}")
        
        # Overall status
        all_passed = (summary["failed_checks"] == 0 and 
                     sys_val["failed_components"] == 0 and 
                     summary["total_errors"] == 0)
        
        print(f"\n🎯 OVERALL STATUS:")
        if all_passed:
            print("   🎉 ALL CHECKS PASSED - SYSTEM IS READY FOR PRODUCTION! 🎉")
        else:
            print("   ⚠️  SOME ISSUES FOUND - PLEASE REVIEW AND FIX")
        
        print(f"\n⏱️  Total Execution Time: {report['total_execution_time']:.2f} seconds")
        print("=" * 80)
    
    def save_report(self, report: Dict[str, Any], filename: str = "quality_report.json"):
        """Save the quality report to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Quality report saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")

async def main():
    """Main function."""
    print("🚀 Schwabot Unified System - Comprehensive Code Quality Checker")
    print("=" * 80)
    
    # Create checker
    checker = ComprehensiveCodeQualityChecker()
    
    try:
        # Run all checks
        report = await checker.run_all_checks()
        
        # Print report
        checker.print_report(report)
        
        # Save report
        checker.save_report(report)
        
        # Exit with appropriate code
        summary = report["summary"]
        sys_val = report["system_validation"]
        
        if (summary["failed_checks"] == 0 and 
            sys_val["failed_components"] == 0 and 
            summary["total_errors"] == 0):
            print("\n🎉 SUCCESS: All quality checks passed!")
            sys.exit(0)
        else:
            print("\n⚠️  WARNING: Some quality checks failed!")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Quality check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 