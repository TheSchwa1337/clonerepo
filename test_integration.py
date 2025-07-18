#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test Script - Schwabot + KoboldCPP
==============================================

This script tests the complete integration between Schwabot's unified
trading system and KoboldCPP's AI interface, verifying:

1. Bridge connectivity and functionality
2. Enhanced interface features
3. Trading command processing
4. Visual layer integration
5. Memory stack operations
6. Real-time data flow
7. Error handling and recovery

Usage:
    python test_integration.py [test_type]

Test Types:
    - all: Run all tests (default)
    - bridge: Test bridge functionality
    - enhanced: Test enhanced interface
    - trading: Test trading commands
    - visual: Test visual layer
    - memory: Test memory stack
    - integration: Test full integration
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests

# Import our components
from core.schwabot_ai_bridge import KoboldCPPBridge
from core.schwabot_ai_enhanced_interface import KoboldCPPEnhancedInterface
from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
from core.visual_layer_controller import VisualLayerController
from core.schwabot_ai_integration import SchwabotAIIntegration
from core.tick_loader import TickLoader
from core.signal_cache import SignalCache
from core.registry_writer import RegistryWriter
from core.json_server import JSONServer

# Import memory stack components
from core.memory_stack.ai_command_sequencer import AICommandSequencer
from core.memory_stack.execution_validator import ExecutionValidator
from core.memory_stack.memory_key_allocator import MemoryKeyAllocator

logger = logging.getLogger(__name__)

class TestResult:
    """Represents a test result."""
    def __init__(self, name: str, success: bool, message: str = "", data: Any = None):
        self.name = name
        self.success = success
        self.message = message
        self.data = data
        self.timestamp = datetime.now()

class IntegrationTester:
    """Integration tester for Schwabot + KoboldCPP."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.results: List[TestResult] = []
        self.components = {}
        
        # Test configuration
        self.test_config = {
            'schwabot_ai_port': 5001,
            'bridge_port': 5005,
            'enhanced_port': 5006,
            'test_symbol': 'BTC/USD',
            'test_amount': 0.001
        }
        
        logger.info("🔧 Integration Tester initialized")
    
    def _add_result(self, name: str, success: bool, message: str = "", data: Any = None):
        """Add a test result."""
        result = TestResult(name, success, message, data)
        self.results.append(result)
        
        if success:
            logger.info(f"✅ {name}: {message}")
        else:
            logger.error(f"❌ {name}: {message}")
        
        return result
    
    async def test_component_initialization(self) -> bool:
        """Test component initialization."""
        try:
            logger.info("🧪 Testing component initialization...")
            
            # Test bridge initialization
            try:
                self.components['bridge'] = KoboldCPPBridge(
                    schwabot_ai_port=self.test_config['schwabot_ai_port'],
                    bridge_port=self.test_config['bridge_port']
                )
                self._add_result("Bridge Initialization", True, "Bridge initialized successfully")
            except Exception as e:
                self._add_result("Bridge Initialization", False, f"Bridge initialization failed: {e}")
                return False
            
            # Test enhanced interface initialization
            try:
                self.components['enhanced'] = KoboldCPPEnhancedInterface(
                    schwabot_ai_port=self.test_config['schwabot_ai_port'],
                    enhanced_port=self.test_config['enhanced_port']
                )
                self._add_result("Enhanced Interface Initialization", True, "Enhanced interface initialized successfully")
            except Exception as e:
                self._add_result("Enhanced Interface Initialization", False, f"Enhanced interface initialization failed: {e}")
                return False
            
            # Test unified interface initialization
            try:
                self.components['unified'] = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
                self._add_result("Unified Interface Initialization", True, "Unified interface initialized successfully")
            except Exception as e:
                self._add_result("Unified Interface Initialization", False, f"Unified interface initialization failed: {e}")
                return False
            
            # Test core components
            try:
                self.components['tick_loader'] = TickLoader()
                self.components['signal_cache'] = SignalCache()
                self.components['registry_writer'] = RegistryWriter()
                self.components['json_server'] = JSONServer()
                self._add_result("Core Components Initialization", True, "Core components initialized successfully")
            except Exception as e:
                self._add_result("Core Components Initialization", False, f"Core components initialization failed: {e}")
                return False
            
            # Test memory stack components
            try:
                self.components['command_sequencer'] = AICommandSequencer()
                self.components['execution_validator'] = ExecutionValidator()
                self.components['memory_allocator'] = MemoryKeyAllocator()
                self._add_result("Memory Stack Initialization", True, "Memory stack components initialized successfully")
            except Exception as e:
                self._add_result("Memory Stack Initialization", False, f"Memory stack initialization failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Component Initialization", False, f"Component initialization test failed: {e}")
            return False
    
    async def test_bridge_functionality(self) -> bool:
        """Test bridge functionality."""
        try:
            logger.info("🧪 Testing bridge functionality...")
            
            bridge = self.components.get('bridge')
            if not bridge:
                self._add_result("Bridge Functionality", False, "Bridge component not available")
                return False
            
            # Test command detection
            test_message = f"analyze {self.test_config['test_symbol']}"
            command = await bridge._detect_command(test_message)
            
            if command and command.command_type.value == 'trading_analysis':
                self._add_result("Bridge Command Detection", True, f"Successfully detected command: {command.command_type.value}")
            else:
                self._add_result("Bridge Command Detection", False, "Failed to detect trading analysis command")
                return False
            
            # Test parameter extraction
            if command.parameters.get('symbol') == self.test_config['test_symbol']:
                self._add_result("Bridge Parameter Extraction", True, f"Successfully extracted symbol: {command.parameters['symbol']}")
            else:
                self._add_result("Bridge Parameter Extraction", False, "Failed to extract symbol parameter")
                return False
            
            # Test message processing
            response = await bridge._process_user_message(test_message)
            if response and "analysis" in response.lower():
                self._add_result("Bridge Message Processing", True, "Successfully processed trading analysis message")
            else:
                self._add_result("Bridge Message Processing", False, "Failed to process trading analysis message")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Bridge Functionality", False, f"Bridge functionality test failed: {e}")
            return False
    
    async def test_enhanced_interface(self) -> bool:
        """Test enhanced interface functionality."""
        try:
            logger.info("🧪 Testing enhanced interface...")
            
            enhanced = self.components.get('enhanced')
            if not enhanced:
                self._add_result("Enhanced Interface", False, "Enhanced interface component not available")
                return False
            
            # Test enhanced pattern detection
            test_message = f"what's the price of {self.test_config['test_symbol']}"
            response = await enhanced._check_enhanced_patterns(test_message, enhanced.conversations.get('test', None))
            
            if response and response.get('source') == 'enhanced_price_check':
                self._add_result("Enhanced Pattern Detection", True, "Successfully detected price check pattern")
            else:
                self._add_result("Enhanced Pattern Detection", False, "Failed to detect price check pattern")
                return False
            
            # Test current price retrieval
            price_data = await enhanced._get_current_price(self.test_config['test_symbol'])
            if price_data and 'price' in price_data:
                self._add_result("Enhanced Price Retrieval", True, f"Successfully retrieved price: ${price_data['price']:.2f}")
            else:
                self._add_result("Enhanced Price Retrieval", False, "Failed to retrieve current price")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Enhanced Interface", False, f"Enhanced interface test failed: {e}")
            return False
    
    async def test_trading_commands(self) -> bool:
        """Test trading command functionality."""
        try:
            logger.info("🧪 Testing trading commands...")
            
            bridge = self.components.get('bridge')
            if not bridge:
                self._add_result("Trading Commands", False, "Bridge component not available")
                return False
            
            # Test portfolio status command
            portfolio_response = await bridge._execute_portfolio_status()
            if portfolio_response and "portfolio" in portfolio_response.lower():
                self._add_result("Portfolio Status Command", True, "Successfully executed portfolio status command")
            else:
                self._add_result("Portfolio Status Command", False, "Failed to execute portfolio status command")
                return False
            
            # Test market insight command
            market_response = await bridge._execute_market_insight()
            if market_response and "market" in market_response.lower():
                self._add_result("Market Insight Command", True, "Successfully executed market insight command")
            else:
                self._add_result("Market Insight Command", False, "Failed to execute market insight command")
                return False
            
            # Test system status command
            system_response = await bridge._execute_system_status()
            if system_response and "system" in system_response.lower():
                self._add_result("System Status Command", True, "Successfully executed system status command")
            else:
                self._add_result("System Status Command", False, "Failed to execute system status command")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Trading Commands", False, f"Trading commands test failed: {e}")
            return False
    
    async def test_visual_layer(self) -> bool:
        """Test visual layer functionality."""
        try:
            logger.info("🧪 Testing visual layer...")
            
            # Test visual layer initialization
            try:
                visual = VisualLayerController()
                await visual.initialize()
                self._add_result("Visual Layer Initialization", True, "Visual layer initialized successfully")
            except Exception as e:
                self._add_result("Visual Layer Initialization", False, f"Visual layer initialization failed: {e}")
                return False
            
            # Test chart generation (mock data)
            mock_tick_data = [
                {'timestamp': datetime.now().timestamp(), 'price': 50000.0, 'volume': 100.0},
                {'timestamp': datetime.now().timestamp(), 'price': 50100.0, 'volume': 150.0},
                {'timestamp': datetime.now().timestamp(), 'price': 50200.0, 'volume': 200.0}
            ]
            
            try:
                chart_result = await visual.generate_price_chart(
                    tick_data=mock_tick_data,
                    symbol=self.test_config['test_symbol'],
                    timeframe='1h'
                )
                
                if chart_result:
                    self._add_result("Chart Generation", True, "Successfully generated price chart")
                else:
                    self._add_result("Chart Generation", False, "Failed to generate price chart")
                    return False
                    
            except Exception as e:
                self._add_result("Chart Generation", False, f"Chart generation failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Visual Layer", False, f"Visual layer test failed: {e}")
            return False
    
    async def test_memory_stack(self) -> bool:
        """Test memory stack functionality."""
        try:
            logger.info("🧪 Testing memory stack...")
            
            # Test command sequencer
            sequencer = self.components.get('command_sequencer')
            if not sequencer:
                self._add_result("Memory Stack", False, "Command sequencer not available")
                return False
            
            # Test command sequencing
            test_command = {
                'type': 'trading_analysis',
                'parameters': {'symbol': self.test_config['test_symbol']},
                'priority': 1
            }
            
            try:
                sequence_result = await sequencer.sequence_command(test_command)
                if sequence_result:
                    self._add_result("Command Sequencing", True, "Successfully sequenced command")
                else:
                    self._add_result("Command Sequencing", False, "Failed to sequence command")
                    return False
            except Exception as e:
                self._add_result("Command Sequencing", False, f"Command sequencing failed: {e}")
                return False
            
            # Test execution validator
            validator = self.components.get('execution_validator')
            if validator:
                try:
                    validation_result = await validator.validate_execution(test_command)
                    if validation_result:
                        self._add_result("Execution Validation", True, "Successfully validated execution")
                    else:
                        self._add_result("Execution Validation", False, "Failed to validate execution")
                        return False
                except Exception as e:
                    self._add_result("Execution Validation", False, f"Execution validation failed: {e}")
                    return False
            
            # Test memory allocator
            allocator = self.components.get('memory_allocator')
            if allocator:
                try:
                    memory_key = await allocator.allocate_memory_key('test_command')
                    if memory_key:
                        self._add_result("Memory Allocation", True, f"Successfully allocated memory key: {memory_key}")
                    else:
                        self._add_result("Memory Allocation", False, "Failed to allocate memory key")
                        return False
                except Exception as e:
                    self._add_result("Memory Allocation", False, f"Memory allocation failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self._add_result("Memory Stack", False, f"Memory stack test failed: {e}")
            return False
    
    async def test_integration_flow(self) -> bool:
        """Test complete integration flow."""
        try:
            logger.info("🧪 Testing complete integration flow...")
            
            bridge = self.components.get('bridge')
            enhanced = self.components.get('enhanced')
            
            if not bridge or not enhanced:
                self._add_result("Integration Flow", False, "Required components not available")
                return False
            
            # Test complete conversation flow
            test_conversation = [
                f"analyze {self.test_config['test_symbol']}",
                "portfolio status",
                "market insight",
                "system status"
            ]
            
            for message in test_conversation:
                try:
                    # Test bridge processing
                    bridge_response = await bridge._process_user_message(message)
                    if not bridge_response:
                        self._add_result(f"Integration Flow - {message}", False, "Bridge processing failed")
                        return False
                    
                    # Test enhanced processing
                    context = enhanced.conversations.get('test', None)
                    if not context:
                        context = enhanced.conversations['test'] = enhanced.ConversationContext(
                            session_id='test',
                            user_id='tester'
                        )
                    
                    enhanced_response = await enhanced._process_enhanced_message(message, context)
                    if not enhanced_response or not enhanced_response.get('text'):
                        self._add_result(f"Integration Flow - {message}", False, "Enhanced processing failed")
                        return False
                    
                    self._add_result(f"Integration Flow - {message}", True, "Successfully processed")
                    
                except Exception as e:
                    self._add_result(f"Integration Flow - {message}", False, f"Processing failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self._add_result("Integration Flow", False, f"Integration flow test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        try:
            logger.info("🧪 Testing error handling...")
            
            bridge = self.components.get('bridge')
            if not bridge:
                self._add_result("Error Handling", False, "Bridge component not available")
                return False
            
            # Test invalid symbol
            try:
                response = await bridge._execute_trading_analysis({'symbol': 'INVALID/SYMBOL'})
                if "couldn't find" in response.lower() or "error" in response.lower():
                    self._add_result("Invalid Symbol Handling", True, "Successfully handled invalid symbol")
                else:
                    self._add_result("Invalid Symbol Handling", False, "Failed to handle invalid symbol")
                    return False
            except Exception as e:
                self._add_result("Invalid Symbol Handling", False, f"Invalid symbol handling failed: {e}")
                return False
            
            # Test invalid command
            try:
                response = await bridge._process_user_message("invalid command that should not work")
                if response and "not sure" in response.lower():
                    self._add_result("Invalid Command Handling", True, "Successfully handled invalid command")
                else:
                    self._add_result("Invalid Command Handling", False, "Failed to handle invalid command")
                    return False
            except Exception as e:
                self._add_result("Invalid Command Handling", False, f"Invalid command handling failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self._add_result("Error Handling", False, f"Error handling test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        try:
            logger.info("🚀 Starting integration tests...")
            
            # Test component initialization
            if not await self.test_component_initialization():
                logger.error("❌ Component initialization failed")
                return self._get_test_summary()
            
            # Test bridge functionality
            if not await self.test_bridge_functionality():
                logger.error("❌ Bridge functionality test failed")
                return self._get_test_summary()
            
            # Test enhanced interface
            if not await self.test_enhanced_interface():
                logger.error("❌ Enhanced interface test failed")
                return self._get_test_summary()
            
            # Test trading commands
            if not await self.test_trading_commands():
                logger.error("❌ Trading commands test failed")
                return self._get_test_summary()
            
            # Test visual layer
            if not await self.test_visual_layer():
                logger.error("❌ Visual layer test failed")
                return self._get_test_summary()
            
            # Test memory stack
            if not await self.test_memory_stack():
                logger.error("❌ Memory stack test failed")
                return self._get_test_summary()
            
            # Test integration flow
            if not await self.test_integration_flow():
                logger.error("❌ Integration flow test failed")
                return self._get_test_summary()
            
            # Test error handling
            if not await self.test_error_handling():
                logger.error("❌ Error handling test failed")
                return self._get_test_summary()
            
            logger.info("✅ All integration tests completed successfully!")
            return self._get_test_summary()
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            return self._get_test_summary()
    
    def _get_test_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'message': r.message,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        return summary

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('integration_test.log')
        ]
    )

async def main():
    """Main test function."""
    try:
        # Setup logging
        setup_logging("INFO")
        
        logger.info("🔧 Starting Integration Tests")
        
        # Create tester
        tester = IntegrationTester()
        
        # Run tests
        summary = await tester.run_all_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print("="*60)
        
        if summary['failed_tests'] > 0:
            print("\nFAILED TESTS:")
            for result in summary['results']:
                if not result['success']:
                    print(f"❌ {result['name']}: {result['message']}")
        
        # Save results to file
        with open('integration_test_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to: integration_test_results.json")
        
        # Exit with appropriate code
        if summary['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Integration tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 