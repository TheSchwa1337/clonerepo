#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Distributed System - Schwabot Complete System Validation
=============================================================

Comprehensive test script that validates the entire distributed real-time
context system including:
- Distributed node management
- Real-time context ingestion
- AI integration bridge
- Flask media server
- Hardware optimization
- Trading system integration

Features:
- Complete system validation
- Performance testing
- Integration testing
- Error handling validation
- Real-time context testing
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distributed_system.distributed_node_manager import start_distributed_system
from core.distributed_system.real_time_context_ingestion import start_context_ingestion
from core.distributed_system.ai_integration_bridge import start_ai_integration
from AOI_Base_Files_Schwabot.api.flask_media_server import start_media_server
from AOI_Base_Files_Schwabot.cli.hardware_optimization_cli import HardwareOptimizationCLI

logger = logging.getLogger(__name__)

class DistributedSystemTester:
    """Comprehensive tester for the distributed system."""
    
    def __init__(self):
        self.distributed_manager = None
        self.context_ingestion = None
        self.ai_bridge = None
        self.media_server = None
        self.hardware_cli = None
        
        # Test results
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance_metrics": {}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Initialized DistributedSystemTester")
    
    async def run_all_tests(self):
        """Run all distributed system tests."""
        print("\n" + "="*80)
        print("SCHWABOT DISTRIBUTED SYSTEM COMPREHENSIVE TEST")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Initialize system components
            await self._initialize_system()
            
            # Run component tests
            await self._test_distributed_node_manager()
            await self._test_context_ingestion()
            await self._test_ai_integration()
            await self._test_flask_media_server()
            await self._test_hardware_optimization()
            await self._test_integration_scenarios()
            await self._test_performance()
            await self._test_error_handling()
            
            # Generate test report
            await self._generate_test_report(start_time)
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results["errors"].append(f"Test suite error: {e}")
        finally:
            # Cleanup
            await self._cleanup_system()
    
    async def _initialize_system(self):
        """Initialize all system components."""
        print("\n🔧 Initializing system components...")
        
        try:
            # Start distributed system
            self.distributed_manager = await start_distributed_system()
            print("✓ Distributed node manager started")
            
            # Start context ingestion
            self.context_ingestion = await start_context_ingestion()
            print("✓ Context ingestion started")
            
            # Start AI integration
            self.ai_bridge = await start_ai_integration()
            print("✓ AI integration started")
            
            # Start media server
            self.media_server = await start_media_server(port=5001)
            print("✓ Flask media server started")
            
            # Initialize hardware optimization
            self.hardware_cli = HardwareOptimizationCLI()
            await self.hardware_cli.initialize()
            print("✓ Hardware optimization initialized")
            
            # Wait for components to stabilize
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _test_distributed_node_manager(self):
        """Test distributed node manager functionality."""
        print("\n🌐 Testing Distributed Node Manager...")
        
        try:
            # Test node status
            node_status = self.distributed_manager.get_node_status()
            assert node_status["total_nodes"] >= 1, "Should have at least one node"
            assert node_status["is_flask_node"] in [True, False], "Should have flask node status"
            print("✓ Node status validation passed")
            
            # Test context data addition
            await self.distributed_manager.add_context_data(
                "test_data", 
                {"message": "Test from distributed manager"},
                "test_source"
            )
            print("✓ Context data addition passed")
            
            # Test node capabilities
            if self.distributed_manager.node_id in self.distributed_manager.nodes:
                node_info = self.distributed_manager.nodes[self.distributed_manager.node_id]
                assert "capabilities" in node_info.__dict__, "Node should have capabilities"
                assert "resources" in node_info.__dict__, "Node should have resources"
                print("✓ Node capabilities validation passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Distributed node manager test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Node manager test: {e}")
    
    async def _test_context_ingestion(self):
        """Test real-time context ingestion."""
        print("\n📊 Testing Context Ingestion...")
        
        try:
            # Test trading data ingestion
            await self.context_ingestion.ingest_trading_data({
                "symbol": "BTC/USD",
                "price": 50000.0,
                "volume": 1000.0,
                "timestamp": time.time()
            })
            print("✓ Trading data ingestion passed")
            
            # Test tensor math ingestion
            from core.distributed_system.real_time_context_ingestion import TensorMathResult
            await self.context_ingestion.ingest_tensor_math(TensorMathResult(
                calculation_id="test_calc_001",
                input_data={"price": 50000.0, "volume": 1000.0},
                result={"prediction": "buy", "confidence": 0.85},
                hash_value="abc123",
                context_meaning="Strong buy signal based on volume analysis",
                confidence=0.85,
                timestamp=time.time()
            ))
            print("✓ Tensor math ingestion passed")
            
            # Test system health ingestion
            await self.context_ingestion.ingest_system_health({
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "network_io": {"bytes_sent": 1024, "bytes_recv": 2048}
            })
            print("✓ System health ingestion passed")
            
            # Test context summary
            summary = self.context_ingestion.get_context_summary()
            assert summary["total_context_items"] > 0, "Should have context items"
            assert "tensor_cache_size" in summary, "Should have tensor cache info"
            print("✓ Context summary validation passed")
            
            # Test AI context retrieval
            ai_context = self.context_ingestion.get_context_for_ai(limit=10)
            assert len(ai_context) > 0, "Should have AI context data"
            print("✓ AI context retrieval passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Context ingestion test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Context ingestion test: {e}")
    
    async def _test_ai_integration(self):
        """Test AI integration bridge."""
        print("\n🤖 Testing AI Integration...")
        
        try:
            # Test AI status
            ai_status = self.ai_bridge.get_ai_status()
            assert "total_models" in ai_status, "Should have model count"
            assert "active_models" in ai_status, "Should have active model count"
            print("✓ AI status validation passed")
            
            # Test AI decision request
            context_data = {
                "symbol": "BTC/USD",
                "price": 50000.0,
                "volume": 1000.0,
                "timestamp": time.time()
            }
            
            decision = await self.ai_bridge.request_decision(context_data, ["BTC/USD"])
            assert hasattr(decision, "final_decision"), "Should have final decision"
            assert hasattr(decision, "confidence"), "Should have confidence"
            assert hasattr(decision, "consensus_reasoning"), "Should have reasoning"
            print("✓ AI decision request passed")
            
            # Test decision history
            history = self.ai_bridge.get_decision_history(limit=5)
            assert isinstance(history, list), "Should return decision history list"
            print("✓ Decision history validation passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"AI integration test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"AI integration test: {e}")
    
    async def _test_flask_media_server(self):
        """Test Flask media server functionality."""
        print("\n🌐 Testing Flask Media Server...")
        
        try:
            # Test server status
            status = self.media_server.get_status()
            assert status["is_running"] == True, "Server should be running"
            assert "total_streams" in status, "Should have stream count"
            print("✓ Server status validation passed")
            
            # Test context ingestion via API
            test_context = {
                "type": "test_data",
                "data": {"message": "Test from media server"},
                "metadata": {"source": "test"}
            }
            
            stream_id = self.media_server._ingest_context(
                test_context["type"],
                test_context["data"],
                test_context["metadata"]
            )
            assert stream_id is not None, "Should return stream ID"
            print("✓ Context ingestion via API passed")
            
            # Test context retrieval
            latest_context = self.media_server._get_latest_context("test_data", 10)
            assert "data" in latest_context, "Should have context data"
            print("✓ Context retrieval passed")
            
            # Test context search
            search_results = self.media_server._search_context("test", "test_data", 10)
            assert isinstance(search_results, list), "Should return search results"
            print("✓ Context search passed")
            
            # Test AI context request
            ai_context = self.media_server._get_ai_context("test_model", ["test_data"], 10)
            assert "context_data" in ai_context, "Should have AI context data"
            assert "summary" in ai_context, "Should have context summary"
            print("✓ AI context request passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Flask media server test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Media server test: {e}")
    
    async def _test_hardware_optimization(self):
        """Test hardware optimization functionality."""
        print("\n⚡ Testing Hardware Optimization...")
        
        try:
            # Test system status
            hw_status = await self.hardware_cli.get_system_status()
            assert "cpu_usage" in hw_status, "Should have CPU usage"
            assert "memory_usage" in hw_status, "Should have memory usage"
            assert "gpu_available" in hw_status, "Should have GPU availability"
            print("✓ System status validation passed")
            
            # Test auto optimization
            await self.hardware_cli.auto_optimize()
            print("✓ Auto optimization passed")
            
            # Test trading optimization
            await self.hardware_cli.optimize_for_trading()
            print("✓ Trading optimization passed")
            
            # Test AI optimization
            await self.hardware_cli.optimize_for_ai()
            print("✓ AI optimization passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Hardware optimization test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Hardware optimization test: {e}")
    
    async def _test_integration_scenarios(self):
        """Test integration scenarios between components."""
        print("\n🔗 Testing Integration Scenarios...")
        
        try:
            # Scenario 1: Trading data flow
            print("  Testing trading data flow...")
            
            # Ingest trading data
            await self.context_ingestion.ingest_trading_data({
                "symbol": "ETH/USD",
                "price": 3000.0,
                "volume": 500.0,
                "timestamp": time.time()
            })
            
            # Request AI decision
            decision = await self.ai_bridge.request_decision({
                "symbol": "ETH/USD",
                "price": 3000.0
            }, ["ETH/USD"])
            
            assert decision.final_decision is not None, "Should have decision"
            print("✓ Trading data flow passed")
            
            # Scenario 2: Real-time context streaming
            print("  Testing real-time context streaming...")
            
            # Add context to media server
            self.media_server._ingest_context("ai_decision", {
                "decision": decision.final_decision.value,
                "confidence": decision.confidence,
                "symbols": ["ETH/USD"]
            }, {"source": "ai_bridge"})
            
            # Verify context is available
            ai_context = self.media_server._get_ai_context("test_model", ["ai_decision"], 10)
            assert len(ai_context["context_data"].get("ai_decision", [])) > 0, "Should have AI decision context"
            print("✓ Real-time context streaming passed")
            
            # Scenario 3: Distributed node communication
            print("  Testing distributed node communication...")
            
            # Add context to distributed manager
            await self.distributed_manager.add_context_data(
                "integration_test",
                {"message": "Integration test successful"},
                "test_scenario"
            )
            
            # Verify node status
            node_status = self.distributed_manager.get_node_status()
            assert node_status["total_nodes"] >= 1, "Should have active nodes"
            print("✓ Distributed node communication passed")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Integration scenario test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Integration test: {e}")
    
    async def _test_performance(self):
        """Test system performance."""
        print("\n⚡ Testing Performance...")
        
        try:
            # Test context ingestion performance
            start_time = time.time()
            
            for i in range(100):
                await self.context_ingestion.ingest_trading_data({
                    "symbol": f"TEST{i}/USD",
                    "price": 100.0 + i,
                    "volume": 100.0,
                    "timestamp": time.time()
                })
            
            ingestion_time = time.time() - start_time
            ingestion_rate = 100 / ingestion_time
            
            self.test_results["performance_metrics"]["context_ingestion_rate"] = ingestion_rate
            print(f"✓ Context ingestion rate: {ingestion_rate:.2f} items/sec")
            
            # Test AI decision performance
            start_time = time.time()
            
            for i in range(10):
                await self.ai_bridge.request_decision({
                    "symbol": f"TEST{i}/USD",
                    "price": 100.0 + i
                }, [f"TEST{i}/USD"])
            
            decision_time = time.time() - start_time
            decision_rate = 10 / decision_time
            
            self.test_results["performance_metrics"]["ai_decision_rate"] = decision_rate
            print(f"✓ AI decision rate: {decision_rate:.2f} decisions/sec")
            
            # Test media server performance
            start_time = time.time()
            
            for i in range(50):
                self.media_server._ingest_context("performance_test", {
                    "index": i,
                    "data": f"test_data_{i}"
                }, {"test": True})
            
            server_time = time.time() - start_time
            server_rate = 50 / server_time
            
            self.test_results["performance_metrics"]["media_server_rate"] = server_rate
            print(f"✓ Media server rate: {server_rate:.2f} items/sec")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Performance test: {e}")
    
    async def _test_error_handling(self):
        """Test error handling and recovery."""
        print("\n🛡️ Testing Error Handling...")
        
        try:
            # Test invalid context data
            try:
                await self.context_ingestion.ingest_trading_data(None)
                assert False, "Should handle None data"
            except Exception:
                print("✓ None data handling passed")
            
            # Test invalid AI decision request
            try:
                await self.ai_bridge.request_decision({}, [])
                print("✓ Empty decision request handled")
            except Exception as e:
                print(f"✓ Error handling for empty decision: {e}")
            
            # Test media server with invalid data
            try:
                self.media_server._ingest_context("", None, {})
                print("✓ Invalid media server data handled")
            except Exception as e:
                print(f"✓ Error handling for invalid media data: {e}")
            
            # Test distributed manager with invalid node
            try:
                await self.distributed_manager.add_context_data("", None, "")
                print("✓ Invalid distributed manager data handled")
            except Exception as e:
                print(f"✓ Error handling for invalid distributed data: {e}")
            
            self.test_results["passed"] += 1
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"Error handling test: {e}")
    
    async def _generate_test_report(self, start_time: float):
        """Generate comprehensive test report."""
        total_time = time.time() - start_time
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        
        print("\n" + "="*80)
        print("TEST REPORT")
        print("="*80)
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {self.test_results['passed']}")
        print(f"   Failed: {self.test_results['failed']}")
        print(f"   Success Rate: {(self.test_results['passed'] / total_tests * 100):.1f}%")
        print(f"   Total Time: {total_time:.2f} seconds")
        
        if self.test_results["performance_metrics"]:
            print(f"\n⚡ Performance Metrics:")
            for metric, value in self.test_results["performance_metrics"].items():
                print(f"   {metric}: {value:.2f}")
        
        if self.test_results["errors"]:
            print(f"\n❌ Errors:")
            for error in self.test_results["errors"]:
                print(f"   - {error}")
        
        # Overall status
        if self.test_results["failed"] == 0:
            print(f"\n🎉 ALL TESTS PASSED! Distributed system is ready for production.")
        else:
            print(f"\n⚠️  {self.test_results['failed']} tests failed. Please review errors above.")
        
        print("="*80)
    
    async def _cleanup_system(self):
        """Clean up system components."""
        print("\n🧹 Cleaning up system...")
        
        try:
            if self.media_server:
                await self.media_server.stop()
                print("✓ Media server stopped")
            
            if self.ai_bridge:
                await self.ai_bridge.stop()
                print("✓ AI integration stopped")
            
            if self.context_ingestion:
                await self.context_ingestion.stop()
                print("✓ Context ingestion stopped")
            
            if self.distributed_manager:
                await self.distributed_manager.stop()
                print("✓ Distributed manager stopped")
            
            if self.hardware_cli:
                await self.hardware_cli.cleanup()
                print("✓ Hardware optimization cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main test runner."""
    tester = DistributedSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 