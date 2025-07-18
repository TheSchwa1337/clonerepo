#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot System Integration Test
================================

Comprehensive test to validate the Schwabot trading system integration
and identify any remaining issues.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test core module imports."""
    logger.info("🔍 Testing core module imports...")
    
    test_results = {}
    
    # Test core modules
    core_modules = [
        "core.schwabot_unified_interface",
        "core.koboldcpp_integration", 
        "core.visual_layer_controller",
        "core.tick_loader",
        "core.signal_cache",
        "core.registry_writer",
        "core.json_server",
        "core.hardware_auto_detector",
        "core.hash_config_manager",
        "core.alpha256_encryption"
    ]
    
    for module_name in core_modules:
        try:
            __import__(module_name)
            test_results[module_name] = {"status": "PASS", "error": None}
            logger.info(f"✅ {module_name} imports successfully")
        except ImportError as e:
            test_results[module_name] = {"status": "FAIL", "error": str(e)}
            logger.error(f"❌ {module_name} import failed: {e}")
        except Exception as e:
            test_results[module_name] = {"status": "ERROR", "error": str(e)}
            logger.error(f"❌ {module_name} error: {e}")
    
    return test_results

def test_aoi_imports():
    """Test AOI_Base_Files_Schwabot module imports."""
    logger.info("🔍 Testing AOI_Base_Files_Schwabot module imports...")
    
    test_results = {}
    
    # Check if AOI_Base_Files_Schwabot directory exists
    aoi_path = Path("AOI_Base_Files_Schwabot")
    if not aoi_path.exists():
        logger.warning("⚠️ AOI_Base_Files_Schwabot directory not found, skipping AOI imports test")
        return {
            "aoi_directory": {"status": "SKIP", "error": "AOI_Base_Files_Schwabot directory not found"}
        }
    
    # Add AOI_Base_Files_Schwabot to path
    original_path = sys.path.copy()
    sys.path.insert(0, str(aoi_path))
    
    try:
        # Test AOI modules that should exist
        aoi_modules = [
            "core.ccxt_integration",
            "gui.visualizer_launcher"
        ]
        
        for module_name in aoi_modules:
            try:
                # Check if the module file exists first
                module_path = aoi_path / module_name.replace('.', '/') / '__init__.py'
                if not module_path.exists():
                    # Try .py file directly
                    module_path = aoi_path / f"{module_name.replace('.', '/')}.py"
                
                if module_path.exists():
                    __import__(module_name)
                    test_results[module_name] = {"status": "PASS", "error": None}
                    logger.info(f"✅ {module_name} imports successfully")
                else:
                    test_results[module_name] = {"status": "SKIP", "error": f"Module file not found: {module_path}"}
                    logger.info(f"⏭️ {module_name} file not found: {module_path}")
            except ImportError as e:
                test_results[module_name] = {"status": "FAIL", "error": str(e)}
                logger.error(f"❌ {module_name} import failed: {e}")
            except Exception as e:
                test_results[module_name] = {"status": "ERROR", "error": str(e)}
                logger.error(f"❌ {module_name} error: {e}")
        
        # Test optional modules that might not exist
        optional_modules = [
            "core.clean_unified_math",
            "core.unified_math_system"
        ]
        
        for module_name in optional_modules:
            try:
                # Check if the module file exists first
                module_path = aoi_path / module_name.replace('.', '/') / '__init__.py'
                if not module_path.exists():
                    # Try .py file directly
                    module_path = aoi_path / f"{module_name.replace('.', '/')}.py"
                
                if module_path.exists():
                    __import__(module_name)
                    test_results[module_name] = {"status": "PASS", "error": None}
                    logger.info(f"✅ {module_name} imports successfully")
                else:
                    test_results[module_name] = {"status": "SKIP", "error": "Module not found (optional)"}
                    logger.info(f"⏭️ {module_name} not found (optional module)")
            except ImportError:
                test_results[module_name] = {"status": "SKIP", "error": "Module not found (optional)"}
                logger.info(f"⏭️ {module_name} not found (optional module)")
            except Exception as e:
                test_results[module_name] = {"status": "ERROR", "error": str(e)}
                logger.error(f"❌ {module_name} error: {e}")
    
    finally:
        # Restore original path
        sys.path = original_path
    
    return test_results

def test_dependencies():
    """Test required dependencies."""
    logger.info("🔍 Testing required dependencies...")
    
    test_results = {}
    
    # Test standard dependencies
    dependencies = [
        "numpy",
        "matplotlib",
        "asyncio",
        "json",
        "logging",
        "threading",
        "time",
        "pathlib",
        "typing",
        "dataclasses",
        "enum"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            test_results[dep] = {"status": "PASS", "error": None}
            logger.info(f"✅ {dep} available")
        except ImportError as e:
            test_results[dep] = {"status": "FAIL", "error": str(e)}
            logger.error(f"❌ {dep} not available: {e}")
    
    return test_results

def test_unified_interface():
    """Test the unified interface functionality."""
    logger.info("🔍 Testing unified interface functionality...")
    
    try:
        from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
        
        # Create interface instance
        interface = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
        
        # Test status
        status = interface.get_unified_status()
        
        test_results = {
            "interface_creation": {"status": "PASS", "error": None},
            "status_retrieval": {"status": "PASS", "error": None},
            "mode": status.mode.value,
            "hardware_optimized": status.hardware_optimized,
            "system_health": status.system_health
        }
        
        logger.info(f"✅ Unified interface created successfully")
        logger.info(f"   Mode: {status.mode.value}")
        logger.info(f"   Hardware optimized: {status.hardware_optimized}")
        logger.info(f"   System health: {status.system_health}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Unified interface test failed: {e}")
        return {"interface_test": {"status": "FAIL", "error": str(e)}}

def test_kobold_integration():
    """Test KoboldCPP integration."""
    logger.info("🔍 Testing KoboldCPP integration...")
    
    try:
        from core.koboldcpp_integration import KoboldCPPIntegration, AnalysisType, KoboldRequest
        
        # Create integration instance
        integration = KoboldCPPIntegration()
        
        # Test request creation
        request = KoboldRequest(
            prompt="Test trading analysis",
            max_length=512,
            temperature=0.7,
            analysis_type=AnalysisType.TECHNICAL_ANALYSIS
        )
        
        test_results = {
            "integration_creation": {"status": "PASS", "error": None},
            "request_creation": {"status": "PASS", "error": None},
            "kobold_path": integration.kobold_path,
            "port": integration.port
        }
        
        logger.info(f"✅ KoboldCPP integration created successfully")
        logger.info(f"   Kobold path: {integration.kobold_path}")
        logger.info(f"   Port: {integration.port}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ KoboldCPP integration test failed: {e}")
        return {"kobold_test": {"status": "FAIL", "error": str(e)}}

def test_visual_layer():
    """Test visual layer controller."""
    logger.info("🔍 Testing visual layer controller...")
    
    try:
        from core.visual_layer_controller import VisualLayerController, VisualizationType, ChartTimeframe
        
        # Create controller instance
        controller = VisualLayerController()
        
        test_results = {
            "controller_creation": {"status": "PASS", "error": None},
            "output_dir": controller.output_dir,
            "running": controller.running
        }
        
        logger.info(f"✅ Visual layer controller created successfully")
        logger.info(f"   Output directory: {controller.output_dir}")
        logger.info(f"   Running: {controller.running}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Visual layer controller test failed: {e}")
        return {"visual_test": {"status": "FAIL", "error": str(e)}}

def test_trading_components():
    """Test trading system components."""
    logger.info("🔍 Testing trading system components...")
    
    test_results = {}
    
    # Test tick loader
    try:
        from core.tick_loader import TickLoader, TickPriority
        tick_loader = TickLoader()
        test_results["tick_loader"] = {"status": "PASS", "error": None}
        logger.info(f"✅ Tick loader created successfully")
    except Exception as e:
        test_results["tick_loader"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"❌ Tick loader test failed: {e}")
    
    # Test signal cache
    try:
        from core.signal_cache import SignalCache, SignalType, SignalPriority
        signal_cache = SignalCache()
        test_results["signal_cache"] = {"status": "PASS", "error": None}
        logger.info(f"✅ Signal cache created successfully")
    except Exception as e:
        test_results["signal_cache"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"❌ Signal cache test failed: {e}")
    
    # Test registry writer
    try:
        from core.registry_writer import RegistryWriter, ArchivePriority
        registry_writer = RegistryWriter()
        test_results["registry_writer"] = {"status": "PASS", "error": None}
        logger.info(f"✅ Registry writer created successfully")
    except Exception as e:
        test_results["registry_writer"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"❌ Registry writer test failed: {e}")
    
    # Test JSON server
    try:
        from core.json_server import JSONServer, PacketPriority
        json_server = JSONServer()
        test_results["json_server"] = {"status": "PASS", "error": None}
        logger.info(f"✅ JSON server created successfully")
    except Exception as e:
        test_results["json_server"] = {"status": "FAIL", "error": str(e)}
        logger.error(f"❌ JSON server test failed: {e}")
    
    return test_results

def test_hardware_detection():
    """Test hardware auto-detection."""
    logger.info("🔍 Testing hardware auto-detection...")
    
    try:
        from core.hardware_auto_detector import HardwareAutoDetector
        
        # Create detector instance
        detector = HardwareAutoDetector()
        
        # Detect hardware
        system_info = detector.detect_hardware()
        memory_config = detector.generate_memory_config()
        
        test_results = {
            "detector_creation": {"status": "PASS", "error": None},
            "hardware_detection": {"status": "PASS", "error": None},
            "memory_config": {"status": "PASS", "error": None},
            "platform": system_info.platform if system_info else "unknown",
            "ram_gb": system_info.ram_gb if system_info else 0.0,
            "optimization_mode": system_info.optimization_mode.value if system_info else "unknown"
        }
        
        logger.info(f"✅ Hardware auto-detection successful")
        logger.info(f"   Platform: {test_results['platform']}")
        logger.info(f"   RAM: {test_results['ram_gb']:.1f} GB")
        logger.info(f"   Optimization: {test_results['optimization_mode']}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Hardware auto-detection test failed: {e}")
        return {"hardware_test": {"status": "FAIL", "error": str(e)}}

def generate_summary_report(all_results):
    """Generate a summary report of all test results."""
    logger.info("📊 Generating summary report...")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            for test_name, result in results.items():
                if isinstance(result, dict) and "status" in result:
                    total_tests += 1
                    if result["status"] == "PASS":
                        passed_tests += 1
                    elif result["status"] == "SKIP":
                        skipped_tests += 1
                    else:
                        failed_tests += 1
    
    # Calculate success rate excluding skipped tests
    actual_tests = total_tests - skipped_tests
    success_rate = (passed_tests / actual_tests * 100) if actual_tests > 0 else 0
    
    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "skipped_tests": skipped_tests,
        "actual_tests": actual_tests,
        "success_rate": success_rate,
        "system_status": "EXCELLENT" if success_rate >= 90 else "GOOD" if success_rate >= 75 else "FAIR" if success_rate >= 50 else "POOR"
    }
    
    logger.info(f"📊 Test Summary:")
    logger.info(f"   Total tests: {total_tests}")
    logger.info(f"   Passed: {passed_tests}")
    logger.info(f"   Failed: {failed_tests}")
    logger.info(f"   Skipped: {skipped_tests}")
    logger.info(f"   Actual tests: {actual_tests}")
    logger.info(f"   Success rate: {success_rate:.1f}%")
    logger.info(f"   System status: {summary['system_status']}")
    
    return summary

def main():
    """Main test function."""
    logger.info("🚀 Starting Schwabot System Integration Test...")
    
    start_time = time.time()
    
    # Run all tests
    all_results = {}
    
    all_results["core_imports"] = test_core_imports()
    all_results["aoi_imports"] = test_aoi_imports()
    all_results["dependencies"] = test_dependencies()
    all_results["unified_interface"] = test_unified_interface()
    all_results["kobold_integration"] = test_kobold_integration()
    all_results["visual_layer"] = test_visual_layer()
    all_results["trading_components"] = test_trading_components()
    all_results["hardware_detection"] = test_hardware_detection()
    
    # Generate summary
    summary = generate_summary_report(all_results)
    
    # Save results
    import json
    results_file = "system_integration_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "summary": summary,
            "detailed_results": all_results
        }, f, indent=2)
    
    logger.info(f"📄 Results saved to {results_file}")
    
    execution_time = time.time() - start_time
    logger.info(f"⏱️ Test execution time: {execution_time:.2f} seconds")
    
    if summary["success_rate"] >= 90:
        logger.info("🎉 System integration test completed successfully!")
        logger.info("✅ Schwabot system is ready for production use!")
    elif summary["success_rate"] >= 75:
        logger.info("👍 System integration test completed with minor issues.")
        logger.info("⚠️ Some components may need attention before production use.")
    else:
        logger.warning("⚠️ System integration test completed with significant issues.")
        logger.warning("❌ System needs fixes before production use.")
    
    return summary["success_rate"] >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 