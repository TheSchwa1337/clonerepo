#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test Script for Distributed System Components
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all components can be imported."""
    print("Testing component imports...")
    
    try:
        from core.distributed_system.distributed_node_manager import DistributedNodeManager
        print("✓ DistributedNodeManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DistributedNodeManager: {e}")
    
    try:
        from core.distributed_system.real_time_context_ingestion import RealTimeContextIngestion
        print("✓ RealTimeContextIngestion imported successfully")
    except Exception as e:
        print(f"✗ Failed to import RealTimeContextIngestion: {e}")
    
    try:
        from core.distributed_system.ai_integration_bridge import AIIntegrationBridge
        print("✓ AIIntegrationBridge imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AIIntegrationBridge: {e}")
    
    try:
        from AOI_Base_Files_Schwabot.api.flask_media_server import FlaskMediaServer
        print("✓ FlaskMediaServer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import FlaskMediaServer: {e}")

def test_basic_functionality():
    """Test basic functionality of components."""
    print("\nTesting basic functionality...")
    
    try:
        from core.distributed_system.distributed_node_manager import DistributedNodeManager
        from core.distributed_system.distributed_node_manager import DistributedConfig
        
        config = DistributedConfig()
        manager = DistributedNodeManager(config)
        print("✓ DistributedNodeManager created successfully")
        
        status = manager.get_node_status()
        print(f"✓ Node status retrieved: {status['total_nodes']} nodes")
        
    except Exception as e:
        print(f"✗ Failed to test DistributedNodeManager: {e}")

async def test_async_components():
    """Test async components."""
    print("\nTesting async components...")
    
    try:
        from core.distributed_system.real_time_context_ingestion import RealTimeContextIngestion
        
        ingestion = RealTimeContextIngestion()
        await ingestion.start()
        print("✓ RealTimeContextIngestion started successfully")
        
        summary = ingestion.get_context_summary()
        print(f"✓ Context summary retrieved: {summary['total_context_items']} items")
        
        await ingestion.stop()
        print("✓ RealTimeContextIngestion stopped successfully")
        
    except Exception as e:
        print(f"✗ Failed to test RealTimeContextIngestion: {e}")

def main():
    """Main test function."""
    print("="*60)
    print("SCHWABOT DISTRIBUTED SYSTEM - SIMPLE TEST")
    print("="*60)
    
    # Test imports
    test_imports()
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test async components
    asyncio.run(test_async_components())
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 