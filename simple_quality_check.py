#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Quality Checker for Schwabot Unified System
=================================================

A focused quality checker that validates the core Schwabot components
without the complexity of external tools that may cause encoding issues.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

class SimpleQualityChecker:
    """Simple quality checker for Schwabot unified system."""
    
    def __init__(self):
        """Initialize the quality checker."""
        self.project_root = Path(__file__).parent
        self.core_dir = self.project_root / "core"
        self.results: List[QualityCheckResult] = []
        self.system_results: List[SystemValidationResult] = []
        
        # Core files to check
        self.core_files = [
            "core/schwabot_unified_interface.py",
            "core/koboldcpp_integration.py",
            "core/visual_layer_controller.py",
            "core/tick_loader.py",
            "core/signal_cache.py",
            "core/registry_writer.py",
            "core/json_server.py",
            "core/utils/windows_cli_compatibility.py",
            "core/utils/math_utils.py",
            "core/memory_stack/__init__.py",
            "core/memory_stack/ai_command_sequencer.py",
            "core/memory_stack/execution_validator.py",
            "core/memory_stack/memory_key_allocator.py",
            "start_schwabot_unified.py",
            "test_unified_integration.py"
        ]
        
        logger.info(f"🔍 Will check {len(self.core_files)} core files")
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks."""
        logger.info("🚀 Starting simple quality checks...")
        
        start_time = time.time()
        
        # Run basic checks
        await self._run_syntax_check()
        await self._run_import_validation()
        await self._run_system_validation()
        await self._run_integration_tests()
        
        # Generate report
        total_time = time.time() - start_time
        report = self._generate_report(total_time)
        
        logger.info("✅ All quality checks completed")
        return report
    
    async def _run_syntax_check(self):
        """Run basic syntax check."""
        logger.info("🔍 Running syntax check...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="Syntax Check")
        
        try:
            syntax_errors = []
            
            for file_path in self.core_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    try:
                        # Try to compile the file
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        
                        compile(source, str(full_path), 'exec')
                        
                    except SyntaxError as e:
                        syntax_errors.append(f"{file_path}: {e}")
                    except Exception as e:
                        syntax_errors.append(f"{file_path}: {e}")
                else:
                    syntax_errors.append(f"{file_path}: File not found")
            
            if syntax_errors:
                result.errors.extend(syntax_errors)
                result.success = False
            else:
                result.success = True
                result.metadata["checked_files"] = len(self.core_files)
        
        except Exception as e:
            result.errors.append(f"Syntax check failed: {e}")
            result.success = False
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def _run_import_validation(self):
        """Validate imports in core files."""
        logger.info("🔍 Running import validation...")
        
        start_time = time.time()
        result = QualityCheckResult(check_name="Import Validation")
        
        try:
            import_errors = []
            
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            # Test core imports
            core_modules = [
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
            
            for module_name in core_modules:
                try:
                    __import__(module_name)
                except ImportError as e:
                    import_errors.append(f"{module_name}: {e}")
                except Exception as e:
                    import_errors.append(f"{module_name}: {e}")
            
            if import_errors:
                result.errors.extend(import_errors)
                result.success = False
            else:
                result.success = True
                result.metadata["validated_modules"] = len(core_modules)
        
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
        print("🔍 SCHWABOT UNIFIED SYSTEM - SIMPLE QUALITY REPORT")
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
    
    def save_report(self, report: Dict[str, Any], filename: str = "simple_quality_report.json"):
        """Save the quality report to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Quality report saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")

async def main():
    """Main function."""
    print("🚀 Schwabot Unified System - Simple Quality Checker")
    print("=" * 80)
    
    # Create checker
    checker = SimpleQualityChecker()
    
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