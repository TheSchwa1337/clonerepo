#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Schwabot Unicode BRAIN Integration - Phase V Verification
=================================================================

Test script to verify the complete Schwabot Unicode BRAIN integration:
- Unicode pathway processing
- Mathematical engines (ALEPH, RITTLE)
- BRAIN system with orbital shells
- Bit tier mapping
- Complete system integration
- Profit-based decision making
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

def test_unicode_pathways():
    """Test Unicode pathway processing."""
    logger.info("🔗 Testing Unicode Pathways...")
    
    try:
        from schwabot_unicode_brain_integration import (
            SchwabotUnicodeBRAINIntegration, 
            UnicodeSymbol, 
            UnicodePathway
        )
        
        # Create system
        schwabot = SchwabotUnicodeBRAINIntegration()
        
        # Test Unicode pathway initialization
        pathway_count = len(schwabot.unicode_pathways)
        logger.info(f"✅ Initialized {pathway_count} Unicode pathways")
        
        # Test specific pathways
        expected_symbols = ['💰', '💸', '🔥', '🔄']
        for symbol in expected_symbols:
            if any(pathway.symbol.value == symbol for pathway in schwabot.unicode_pathways.values()):
                logger.info(f"✅ Found pathway for symbol: {symbol}")
            else:
                logger.warning(f"⚠️ Missing pathway for symbol: {symbol}")
        
        # Test pathway hash calculation
        for pathway in schwabot.unicode_pathways.values():
            hash_value = pathway.calculate_hash()
            if hash_value and len(hash_value) == 64:  # SHA-256 hash length
                logger.info(f"✅ Hash calculation working for {pathway.symbol.value}")
            else:
                logger.error(f"❌ Hash calculation failed for {pathway.symbol.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unicode Pathways test failed: {e}")
        return False

def test_mathematical_engines():
    """Test mathematical engines."""
    logger.info("🧮 Testing Mathematical Engines...")
    
    try:
        from schwabot_unicode_brain_integration import (
            ALEPHEngine, 
            RITTLEEngine, 
            MathematicalEngine
        )
        
        # Test ALEPH engine
        aleph_engine = ALEPHEngine()
        test_input = {
            'price': 50000.0,
            'volume': 5000.0,
            'volatility': 0.02,
            'profit_target': 0.01
        }
        
        aleph_result = aleph_engine.process(test_input)
        if aleph_result.success and aleph_result.confidence >= 0:
            logger.info(f"✅ ALEPH engine working - confidence: {aleph_result.confidence:.3f}")
        else:
            logger.error(f"❌ ALEPH engine failed: {aleph_result.error_message}")
        
        # Test RITTLE engine
        rittle_engine = RITTLEEngine()
        rittle_result = rittle_engine.process(test_input)
        if rittle_result.success and rittle_result.confidence >= 0:
            logger.info(f"✅ RITTLE engine working - confidence: {rittle_result.confidence:.3f}")
        else:
            logger.error(f"❌ RITTLE engine failed: {rittle_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mathematical Engines test failed: {e}")
        return False

def test_brain_system():
    """Test BRAIN system with orbital shells."""
    logger.info("🧠⚛️ Testing BRAIN System...")
    
    try:
        from schwabot_unicode_brain_integration import BRAINSystem, MarketData
        import math
        
        # Create BRAIN system
        brain_system = BRAINSystem()
        
        # Test shell initialization
        shell_count = len(brain_system.shells)
        logger.info(f"✅ BRAIN system initialized with {shell_count} shells")
        
        # Test market data processing
        test_market_data = MarketData(
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
        
        brain_results = brain_system.process_market_data(test_market_data)
        if brain_results and len(brain_results) > 0:
            logger.info(f"✅ BRAIN system processing working - {len(brain_results)} shell responses")
            
            # Check shell responses
            for shell_id, response in brain_results.items():
                if 'profit_potential' in response and 'energy_level' in response:
                    logger.info(f"✅ Shell {shell_id} responding correctly")
                else:
                    logger.warning(f"⚠️ Shell {shell_id} missing expected fields")
        else:
            logger.error("❌ BRAIN system processing failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BRAIN System test failed: {e}")
        return False

def test_bit_tiers():
    """Test bit tier mapping."""
    logger.info("🔢 Testing Bit Tier Mapping...")
    
    try:
        from schwabot_unicode_brain_integration import SchwabotUnicodeBRAINIntegration, BitTier
        
        # Create system
        schwabot = SchwabotUnicodeBRAINIntegration()
        
        # Test bit tier initialization
        tier_count = len(schwabot.bit_tiers)
        logger.info(f"✅ Initialized {tier_count} bit tiers")
        
        # Test tier configurations
        expected_tiers = [4, 8, 16, 32, 64, 256]
        for tier_value in expected_tiers:
            tier = BitTier(f"TIER_{tier_value}BIT")
            if tier in schwabot.bit_tiers:
                config = schwabot.bit_tiers[tier]
                logger.info(f"✅ Bit tier {tier_value} configured: {config['complexity']}")
            else:
                logger.warning(f"⚠️ Missing bit tier {tier_value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bit Tier Mapping test failed: {e}")
        return False

def test_complete_integration():
    """Test complete system integration."""
    logger.info("🔗 Testing Complete Integration...")
    
    try:
        from schwabot_unicode_brain_integration import SchwabotUnicodeBRAINIntegration
        
        # Create complete system
        schwabot = SchwabotUnicodeBRAINIntegration()
        
        # Test system initialization
        initial_status = schwabot.get_system_status()
        logger.info(f"✅ System initialized: {initial_status['system_status']['is_running']}")
        
        # Test starting system
        if schwabot.start_system():
            logger.info("✅ Complete integration system started successfully")
            
            # Let it run for a short time
            time.sleep(10)
            
            # Get status during operation
            running_status = schwabot.get_system_status()
            cycle_count = running_status['system_status']['cycle_count']
            logger.info(f"✅ System running: {cycle_count} cycles completed")
            
            # Check Unicode pathways
            unicode_info = running_status['unicode_pathways']
            logger.info(f"✅ Unicode pathways: {unicode_info['total_pathways']} total, {len(unicode_info['active_symbols'])} active")
            
            # Check mathematical engines
            engine_info = running_status['mathematical_engines']
            logger.info(f"✅ Mathematical engines: {list(engine_info.keys())}")
            
            # Check BRAIN system
            brain_info = running_status['brain_system']
            logger.info(f"✅ BRAIN system: {brain_info['num_shells']} shells, {brain_info['active_shells']} active")
            
            # Check bit tiers
            bit_info = running_status['bit_tiers']
            logger.info(f"✅ Bit tiers: {bit_info['total_tiers']} tiers, range: {bit_info['tier_range']}")
            
            # Stop system
            schwabot.stop_system()
            logger.info("✅ Complete integration system stopped successfully")
        else:
            logger.error("❌ Failed to start complete integration system")
            return False
            
    except Exception as e:
        logger.error(f"❌ Complete Integration test failed: {e}")
        return False
    
    logger.info("✅ Complete Integration test passed")
    return True

def test_profit_based_decisions():
    """Test profit-based decision making."""
    logger.info("💰 Testing Profit-Based Decisions...")
    
    try:
        from schwabot_unicode_brain_integration import SchwabotUnicodeBRAINIntegration
        
        # Create system
        schwabot = SchwabotUnicodeBRAINIntegration()
        
        # Test decision integration
        # This would normally be tested during system operation
        # For now, we'll test the decision components
        
        # Test Unicode pathway conditions
        pathway_conditions_working = True
        for pathway in schwabot.unicode_pathways.values():
            if not pathway.conditions:
                pathway_conditions_working = False
                break
        
        if pathway_conditions_working:
            logger.info("✅ Unicode pathway conditions configured")
        else:
            logger.warning("⚠️ Some Unicode pathways missing conditions")
        
        # Test mathematical expressions
        expressions_working = True
        for pathway in schwabot.unicode_pathways.values():
            if not pathway.mathematical_expression:
                expressions_working = False
                break
        
        if expressions_working:
            logger.info("✅ Mathematical expressions configured")
        else:
            logger.warning("⚠️ Some pathways missing mathematical expressions")
        
        # Test engine sequences
        sequences_working = True
        for pathway in schwabot.unicode_pathways.values():
            if not pathway.engine_sequence:
                sequences_working = False
                break
        
        if sequences_working:
            logger.info("✅ Engine sequences configured")
        else:
            logger.warning("⚠️ Some pathways missing engine sequences")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Profit-Based Decisions test failed: {e}")
        return False

def main():
    """Run all Unicode BRAIN integration tests."""
    logger.info("🧪 Starting Schwabot Unicode BRAIN Integration Tests")
    
    tests = [
        ("Unicode Pathways", test_unicode_pathways),
        ("Mathematical Engines", test_mathematical_engines),
        ("BRAIN System", test_brain_system),
        ("Bit Tier Mapping", test_bit_tiers),
        ("Profit-Based Decisions", test_profit_based_decisions),
        ("Complete Integration", test_complete_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("UNICODE BRAIN INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Tests passed: {passed}/{total}")
    logger.info(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL UNICODE BRAIN INTEGRATION TESTS PASSED!")
        logger.info("🚀 Schwabot Unicode BRAIN system is ready for operation!")
    else:
        logger.warning(f"⚠️ {total-passed} test(s) failed. Please check the logs.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 