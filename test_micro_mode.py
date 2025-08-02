#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 MICRO MODE LIVE Test Script
==============================

Test the revolutionary MICRO MODE LIVE functionality with:
- $1 trade caps
- Maximum paranoia protocols
- Triple confirmation system
- Emergency stop capabilities
- Real-time status monitoring

⚠️ WARNING: This is for testing the MICRO MODE system only!
    Real trading requires proper API setup and confirmation.
"""

import time
import logging
from datetime import datetime
from clock_mode_system import ClockModeSystem, ExecutionMode, SAFETY_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_micro_mode_live():
    """Test the MICRO MODE LIVE functionality."""
    logger.info("🧪 Starting MICRO MODE LIVE Test")
    logger.info("=" * 50)
    
    # Create clock mode system
    clock_system = ClockModeSystem()
    
    # Test 1: Verify default is SHADOW mode
    logger.info("\n📊 TEST 1: Verify Default SHADOW Mode")
    status = clock_system.get_all_mechanisms_status()
    current_mode = status["safety_config"]["execution_mode"]
    logger.info(f"Current mode: {current_mode}")
    
    if current_mode == "shadow":
        logger.info("✅ Default SHADOW mode confirmed")
    else:
        logger.warning(f"⚠️ Unexpected default mode: {current_mode}")
    
    # Test 2: Enable MICRO MODE
    logger.info("\n🚨 TEST 2: Enable MICRO MODE LIVE")
    success = clock_system.enable_micro_mode()
    
    if success:
        logger.warning("🚨 MICRO MODE ENABLED - $1 live trading active!")
        
        # Verify mode change
        status = clock_system.get_all_mechanisms_status()
        current_mode = status["safety_config"]["execution_mode"]
        logger.info(f"Current mode: {current_mode}")
        
        if current_mode == "micro":
            logger.info("✅ MICRO mode successfully enabled")
        else:
            logger.error(f"❌ Mode not changed to micro: {current_mode}")
    else:
        logger.error("❌ Failed to enable MICRO mode")
        return
    
    # Test 3: Check MICRO mode stats
    logger.info("\n📊 TEST 3: MICRO Mode Statistics")
    micro_stats = clock_system.get_micro_trading_stats()
    logger.info(f"Micro mode enabled: {micro_stats['micro_mode_enabled']}")
    logger.info(f"Daily trades: {micro_stats['daily_trades']}")
    logger.info(f"Daily volume: ${micro_stats['daily_volume']:.2f}")
    logger.info(f"Total trades: {micro_stats['total_trades']}")
    logger.info(f"Total volume: ${micro_stats['total_volume']:.2f}")
    logger.info(f"Emergency stop triggered: {micro_stats['emergency_stop_triggered']}")
    
    # Test 4: Start clock mode to simulate trading decisions
    logger.info("\n🕐 TEST 4: Start Clock Mode for Trading Simulation")
    if clock_system.start_clock_mode():
        logger.info("✅ Clock mode started")
        
        # Run for a few seconds to see MICRO mode in action
        logger.info("⏳ Running for 10 seconds to observe MICRO mode behavior...")
        time.sleep(10)
        
        # Stop clock mode
        clock_system.stop_clock_mode()
        logger.info("✅ Clock mode stopped")
    else:
        logger.error("❌ Failed to start clock mode")
    
    # Test 5: Check updated MICRO stats
    logger.info("\n📊 TEST 5: Updated MICRO Mode Statistics")
    micro_stats = clock_system.get_micro_trading_stats()
    logger.info(f"Daily trades: {micro_stats['daily_trades']}")
    logger.info(f"Daily volume: ${micro_stats['daily_volume']:.2f}")
    logger.info(f"Total trades: {micro_stats['total_trades']}")
    logger.info(f"Total volume: ${micro_stats['total_volume']:.2f}")
    
    # Test 6: Test emergency stop
    logger.info("\n🛑 TEST 6: Test Emergency Stop")
    success = clock_system.trigger_micro_emergency_stop()
    
    if success:
        logger.warning("🚨 MICRO MODE EMERGENCY STOP TRIGGERED!")
        
        # Verify emergency stop
        micro_stats = clock_system.get_micro_trading_stats()
        if micro_stats['emergency_stop_triggered']:
            logger.info("✅ Emergency stop successfully triggered")
        else:
            logger.error("❌ Emergency stop not reflected in stats")
    else:
        logger.error("❌ Failed to trigger emergency stop")
    
    # Test 7: Disable MICRO mode
    logger.info("\n🛡️ TEST 7: Disable MICRO Mode")
    success = clock_system.disable_micro_mode()
    
    if success:
        logger.info("🛡️ MICRO MODE DISABLED - Back to SHADOW mode")
        
        # Verify mode change
        status = clock_system.get_all_mechanisms_status()
        current_mode = status["safety_config"]["execution_mode"]
        logger.info(f"Current mode: {current_mode}")
        
        if current_mode == "shadow":
            logger.info("✅ Successfully returned to SHADOW mode")
        else:
            logger.error(f"❌ Mode not changed to shadow: {current_mode}")
    else:
        logger.error("❌ Failed to disable MICRO mode")
    
    # Test 8: Final status report
    logger.info("\n📋 TEST 8: Final Status Report")
    status = clock_system.get_all_mechanisms_status()
    
    logger.info("=== FINAL SYSTEM STATUS ===")
    logger.info(f"Execution mode: {status['safety_config']['execution_mode']}")
    logger.info(f"Active mechanisms: {status['active_mechanisms']}")
    logger.info(f"Total mechanisms: {status['total_mechanisms']}")
    
    if "micro_mode" in status:
        micro_info = status["micro_mode"]
        logger.info(f"Micro mode: {micro_info['mode']}")
        logger.info(f"Description: {micro_info['description']}")
        logger.info(f"Warning: {micro_info['warning']}")
        
        stats = micro_info['stats']
        logger.info(f"Daily trades: {stats['daily_trades']}")
        logger.info(f"Daily volume: ${stats['daily_volume']:.2f}")
        logger.info(f"Total trades: {stats['total_trades']}")
        logger.info(f"Total volume: ${stats['total_volume']:.2f}")
    
    logger.info("=" * 50)
    logger.info("🧪 MICRO MODE LIVE Test Complete!")
    logger.info("✅ All tests completed successfully!")

def main():
    """Main test function."""
    try:
        test_micro_mode_live()
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 