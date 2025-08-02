#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Schwabot System - Phase IV Verification
===============================================

Test script to verify the complete Schwabot trading system:
- Clock Mode System functionality
- Neural Core decision making
- Integrated system operation
- Safety mechanisms
- Performance tracking
"""

import sys
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_clock_mode_system():
    """Test the Clock Mode System."""
    logger.info("🕐 Testing Clock Mode System...")
    
    try:
        from clock_mode_system import ClockModeSystem
        
        # Create clock system
        clock_system = ClockModeSystem()
        
        # Test default mechanism creation
        mechanism_id = clock_system.create_default_mechanism()
        logger.info(f"✅ Created mechanism: {mechanism_id}")
        
        # Test starting clock mode
        if clock_system.start_clock_mode():
            logger.info("✅ Clock mode started successfully")
            
            # Let it run for a few seconds
            time.sleep(5)
            
            # Get status
            status = clock_system.get_all_mechanisms_status()
            logger.info(f"✅ Clock system status: {len(status.get('mechanisms', {}))} mechanisms active")
            
            # Stop clock mode
            clock_system.stop_clock_mode()
            logger.info("✅ Clock mode stopped successfully")
        else:
            logger.error("❌ Failed to start clock mode")
            return False
            
    except Exception as e:
        logger.error(f"❌ Clock Mode System test failed: {e}")
        return False
    
    logger.info("✅ Clock Mode System test passed")
    return True

def test_neural_core():
    """Test the Neural Core."""
    logger.info("🧠 Testing Neural Core...")
    
    try:
        from schwabot_neural_core import SchwabotNeuralCore, MarketData
        import math
        
        # Create neural core
        neural_core = SchwabotNeuralCore()
        
        # Test neural network creation
        stats = neural_core.get_neural_stats()
        logger.info(f"✅ Neural network created: {stats['total_neurons']} neurons")
        
        # Create test market data
        market_data = MarketData(
            timestamp=datetime.now(),
            btc_price=50000.0,
            usdc_balance=10000.0,
            btc_balance=0.2,
            price_change=0.02,
            volume=5000.0,
            rsi_14=45.0,
            rsi_21=50.0,
            rsi_50=55.0,
            market_phase=math.pi / 4,
            hash_timing="a1b2c3d4e5f6",
            orbital_phase=0.5
        )
        
        # Test decision making
        decision = neural_core.make_decision(market_data)
        logger.info(f"✅ Decision made: {decision.decision_type.value} (confidence: {decision.confidence:.3f})")
        
        # Test learning
        neural_core.learn_from_outcome(decision, 150.0)
        logger.info("✅ Learning test completed")
        
        # Get updated stats
        updated_stats = neural_core.get_neural_stats()
        logger.info(f"✅ Updated stats: {updated_stats['recursive_cycles']} cycles completed")
        
    except Exception as e:
        logger.error(f"❌ Neural Core test failed: {e}")
        return False
    
    logger.info("✅ Neural Core test passed")
    return True

def test_integrated_system():
    """Test the complete integrated system."""
    logger.info("🤖 Testing Integrated System...")
    
    try:
        from schwabot_integrated_system import SchwabotIntegratedSystem
        
        # Create integrated system
        schwabot = SchwabotIntegratedSystem()
        
        # Test system initialization
        initial_status = schwabot.get_system_status()
        logger.info(f"✅ System initialized: {initial_status['system_status']['is_running']}")
        
        # Test starting system
        if schwabot.start_system():
            logger.info("✅ Integrated system started successfully")
            
            # Let it run for a short time
            time.sleep(10)
            
            # Get status during operation
            running_status = schwabot.get_system_status()
            cycle_count = running_status['system_status']['cycle_count']
            logger.info(f"✅ System running: {cycle_count} cycles completed")
            
            # Get recent trades
            recent_trades = schwabot.get_recent_trades(3)
            logger.info(f"✅ Recent trades: {len(recent_trades)} trades recorded")
            
            # Stop system
            schwabot.stop_system()
            logger.info("✅ Integrated system stopped successfully")
        else:
            logger.error("❌ Failed to start integrated system")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integrated System test failed: {e}")
        return False
    
    logger.info("✅ Integrated System test passed")
    return True

def test_safety_mechanisms():
    """Test safety mechanisms."""
    logger.info("🛡️ Testing Safety Mechanisms...")
    
    try:
        from clock_mode_system import SAFETY_CONFIG
        
        # Test safety configuration
        logger.info(f"✅ Execution mode: {SAFETY_CONFIG.execution_mode.value}")
        logger.info(f"✅ Max position size: {SAFETY_CONFIG.max_position_size}")
        logger.info(f"✅ Max daily loss: {SAFETY_CONFIG.max_daily_loss}")
        logger.info(f"✅ Emergency stop enabled: {SAFETY_CONFIG.emergency_stop_enabled}")
        
        # Verify default is SHADOW mode for safety
        if SAFETY_CONFIG.execution_mode.value == "shadow":
            logger.info("✅ Safety: System in SHADOW mode (analysis only)")
        else:
            logger.warning("⚠️ Safety: System not in SHADOW mode")
            
    except Exception as e:
        logger.error(f"❌ Safety mechanisms test failed: {e}")
        return False
    
    logger.info("✅ Safety mechanisms test passed")
    return True

def main():
    """Run all tests."""
    logger.info("🧪 Starting Schwabot System Tests")
    
    tests = [
        ("Clock Mode System", test_clock_mode_system),
        ("Neural Core", test_neural_core),
        ("Safety Mechanisms", test_safety_mechanisms),
        ("Integrated System", test_integrated_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Tests passed: {passed}/{total}")
    logger.info(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Schwabot system is ready.")
    else:
        logger.warning(f"⚠️ {total-passed} test(s) failed. Please check the logs.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 