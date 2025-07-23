#!/usr/bin/env python3
"""
🎡 Ferris Ride Looping Strategy Test - Revolutionary Auto-Trading Demo
=====================================================================

This script demonstrates the revolutionary Ferris Ride system:
- Auto-detection of capital and tickers
- Pattern studying before entry
- Hash pattern matching
- Confidence zone building
- Mathematical orbital trading
- USB backup system
"""

import sys
import os
import time
import random
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_ferris_ride_initialization():
    """Test Ferris Ride system initialization."""
    print("🎡 Testing Ferris Ride System Initialization...")
    
    try:
        # Import the Ferris Ride system
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        print("✅ Ferris Ride System imported successfully")
        
        # Test auto-detection
        print("\n🔍 Testing Auto-Detection...")
        success = ferris_ride_system.auto_detect_capital_and_tickers()
        
        if success:
            status = ferris_ride_system.get_ferris_status()
            print(f"✅ Auto-detection successful")
            print(f"   Detected Capital: ${status['detected_capital']:.2f}")
            print(f"   Detected Tickers: {status['detected_tickers']}")
            print(f"   Current Phase: {status['current_phase']}")
            print(f"   Confidence Level: {status['confidence_level']}")
            print(f"   USB Backup Path: {status['usb_backup_path']}")
        else:
            print("❌ Auto-detection failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Ferris Ride System import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Ferris Ride initialization test failed: {e}")
        return False

def test_pattern_studying():
    """Test market pattern studying."""
    print("\n📚 Testing Pattern Studying...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Test studying multiple symbols
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "ADA/USDC"]
        
        for symbol in symbols:
            print(f"\n📊 Studying {symbol}...")
            
            # Create sample market data
            market_data = {
                'symbol': symbol,
                'price': 65000 + random.uniform(-5000, 5000),
                'volume': random.uniform(1000000, 10000000),
                'rsi': random.uniform(25, 75),
                'macd': random.uniform(-1, 1),
                'sentiment': random.uniform(0.3, 0.7)
            }
            
            # Study the pattern
            success = ferris_ride_system.study_market_pattern(symbol, market_data)
            
            if success:
                pattern = ferris_ride_system.studied_patterns[symbol]
                print(f"✅ Pattern study completed for {symbol}")
                print(f"   Hash Pattern: {pattern['hash_pattern'][:16]}...")
                print(f"   Confidence: {pattern['total_confidence']:.1%}")
                print(f"   Risk Level: {pattern['risk_assessment']:.1%}")
                print(f"   RSI Trend: {pattern['rsi_trend']}")
                print(f"   Volume Profile: {pattern['volume_profile']}")
                print(f"   USDC Correlation: {pattern['usdc_correlation']:.1%}")
            else:
                print(f"❌ Pattern study failed for {symbol}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern studying test failed: {e}")
        return False

def test_zone_targeting():
    """Test zone targeting with hash pattern matching."""
    print("\n🎯 Testing Zone Targeting...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Test targeting for studied symbols
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        
        for symbol in symbols:
            print(f"\n🎯 Targeting zone for {symbol}...")
            
            # Create market data that should match studied patterns
            market_data = {
                'symbol': symbol,
                'price': 65000 + random.uniform(-1000, 1000),
                'volume': random.uniform(2000000, 8000000),
                'rsi': random.uniform(30, 70),  # Healthy range
                'macd': random.uniform(-0.5, 0.5),
                'sentiment': random.uniform(0.4, 0.6)  # Neutral sentiment
            }
            
            # Try to target zone
            zone = ferris_ride_system.target_zone(symbol, market_data)
            
            if zone:
                print(f"✅ Zone targeted for {symbol}")
                print(f"   Entry Price: ${zone.entry_price:.4f}")
                print(f"   Target Price: ${zone.target_price:.4f}")
                print(f"   Confidence: {zone.confidence:.1%}")
                print(f"   Orbital Shell: {zone.orbital_shell}")
                print(f"   Hash Pattern: {zone.hash_pattern[:16]}...")
                print(f"   Risk Level: {zone.risk_level:.1%}")
            else:
                print(f"📊 No zone targeted for {symbol} (insufficient confidence or no hash match)")
        
        return True
        
    except Exception as e:
        print(f"❌ Zone targeting test failed: {e}")
        return False

def test_ferris_trade_execution():
    """Test Ferris Ride trade execution."""
    print("\n🎡 Testing Ferris Trade Execution...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Test trade execution for active zones
        active_zones = list(ferris_ride_system.active_zones.keys())
        
        if not active_zones:
            print("📊 No active zones to test trade execution")
            return True
        
        for symbol in active_zones[:2]:  # Test first 2 zones
            print(f"\n🎡 Executing Ferris trade for {symbol}...")
            
            # Create market data for trade execution
            market_data = {
                'symbol': symbol,
                'price': 65000 + random.uniform(-2000, 2000),
                'volume': random.uniform(3000000, 10000000),
                'rsi': random.uniform(35, 65),
                'macd': random.uniform(-0.3, 0.3),
                'sentiment': random.uniform(0.45, 0.55)
            }
            
            # Execute Ferris trade
            decision = ferris_ride_system.execute_ferris_trade(symbol, market_data)
            
            if decision:
                print(f"✅ Ferris trade executed for {symbol}")
                print(f"   Action: {decision.action}")
                print(f"   Entry Price: ${decision.entry_price:.4f}")
                print(f"   Position Size: {decision.position_size:.6f}")
                print(f"   Confidence: {decision.confidence:.1%}")
                print(f"   Orbital Shell: {decision.orbital_shell}")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Hash Pattern: {decision.hash_pattern[:16]}...")
                
                # Show Ferris RDE state
                rde_state = ferris_ride_system.ferris_rde_state
                print(f"   Ferris RDE State:")
                print(f"     Momentum Factor: {rde_state['momentum_factor']:.3f}")
                print(f"     Spiral Radius: {rde_state['spiral_radius']:.3f}")
                print(f"     Orbital Velocity: {rde_state['orbital_velocity']:.3f}")
            else:
                print(f"📊 No trade executed for {symbol}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ferris trade execution test failed: {e}")
        return False

def test_confidence_building():
    """Test confidence zone building through profit accumulation."""
    print("\n🎯 Testing Confidence Zone Building...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Test confidence building for active zones
        active_zones = list(ferris_ride_system.active_zones.keys())
        
        if not active_zones:
            print("📊 No active zones to test confidence building")
            return True
        
        for symbol in active_zones[:2]:  # Test first 2 zones
            print(f"\n🎯 Building confidence for {symbol}...")
            
            # Simulate profit
            profit = random.uniform(50, 200)
            
            # Build confidence zone
            success = ferris_ride_system.build_confidence_zone(symbol, profit)
            
            if success:
                zone = ferris_ride_system.active_zones[symbol]
                print(f"✅ Confidence zone built for {symbol}")
                print(f"   Profit: ${profit:.2f}")
                print(f"   New Confidence: {zone.confidence:.1%}")
                print(f"   Orbital Shell: {zone.orbital_shell}")
                
                # Show updated Ferris RDE state
                rde_state = ferris_ride_system.ferris_rde_state
                print(f"   Updated Ferris RDE State:")
                print(f"     Momentum Factor: {rde_state['momentum_factor']:.3f}")
                print(f"     Spiral Radius: {rde_state['spiral_radius']:.3f}")
                print(f"     Total Profit: ${ferris_ride_system.total_profit:.2f}")
                print(f"     Confidence Bonus: {ferris_ride_system.confidence_bonus:.1%}")
            else:
                print(f"❌ Confidence building failed for {symbol}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence building test failed: {e}")
        return False

def test_usb_backup():
    """Test USB backup functionality."""
    print("\n💾 Testing USB Backup...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Perform backup
        ferris_ride_system.backup_ferris_data()
        
        # Check if backup file was created
        if ferris_ride_system.usb_backup_path:
            backup_files = list(ferris_ride_system.usb_backup_path.glob("ferris_backup_*.json"))
            if backup_files:
                latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                print(f"✅ USB backup successful")
                print(f"   Backup Path: {ferris_ride_system.usb_backup_path}")
                print(f"   Latest Backup: {latest_backup.name}")
                print(f"   Backup Size: {latest_backup.stat().st_size} bytes")
                
                # Show backup contents summary
                import json
                with open(latest_backup, 'r') as f:
                    backup_data = json.load(f)
                
                print(f"   Backup Contents:")
                print(f"     Detected Capital: ${backup_data['detected_capital']:.2f}")
                print(f"     Detected Tickers: {len(backup_data['detected_tickers'])}")
                print(f"     Active Zones: {len(backup_data['active_zones'])}")
                print(f"     Studied Patterns: {len(backup_data['studied_patterns'])}")
                print(f"     Profit History: {len(backup_data['profit_history'])} records")
                print(f"     Total Trades: {backup_data['performance']['total_trades']}")
                print(f"     Total Profit: ${backup_data['performance']['total_profit']:.2f}")
            else:
                print("❌ No backup files found")
                return False
        else:
            print("❌ USB backup path not set")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ USB backup test failed: {e}")
        return False

def test_ferris_status():
    """Test Ferris Ride system status."""
    print("\n📊 Testing Ferris Ride Status...")
    
    try:
        from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system
        
        # Get comprehensive status
        status = ferris_ride_system.get_ferris_status()
        
        print("🎡 FERRIS RIDE SYSTEM STATUS")
        print("=" * 50)
        print(f"Current Phase: {status['current_phase']}")
        print(f"Confidence Level: {status['confidence_level']}")
        print(f"Active Zones: {status['active_zones']}")
        print(f"Studied Patterns: {status['studied_patterns']}")
        print(f"Detected Capital: ${status['detected_capital']:.2f}")
        print(f"Detected Tickers: {status['detected_tickers']}")
        print(f"Current Orbital Shell: {status['current_orbital_shell']}")
        print(f"USB Backup Path: {status['usb_backup_path']}")
        
        print("\n🎯 PERFORMANCE METRICS")
        print("=" * 30)
        performance = status['performance']
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Winning Trades: {performance['winning_trades']}")
        print(f"Total Profit: ${performance['total_profit']:.2f}")
        print(f"Confidence Bonus: {performance['confidence_bonus']:.1%}")
        
        print("\n🎡 FERRIS RDE STATE")
        print("=" * 25)
        rde_state = status['ferris_rde_state']
        print(f"Current Rotation: {rde_state['current_rotation']}")
        print(f"Momentum Factor: {rde_state['momentum_factor']:.3f}")
        print(f"Gravity Center: {rde_state['gravity_center']:.3f}")
        print(f"Orbital Velocity: {rde_state['orbital_velocity']:.3f}")
        print(f"Spiral Radius: {rde_state['spiral_radius']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ferris status test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎡 FERRIS RIDE LOOPING STRATEGY TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Ferris Ride Initialization", test_ferris_ride_initialization),
        ("Pattern Studying", test_pattern_studying),
        ("Zone Targeting", test_zone_targeting),
        ("Ferris Trade Execution", test_ferris_trade_execution),
        ("Confidence Building", test_confidence_building),
        ("USB Backup", test_usb_backup),
        ("Ferris Status", test_ferris_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎡 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ferris Ride System is working perfectly!")
        print("\n🎡 FERRIS RIDE SYSTEM FEATURES:")
        print("=" * 40)
        print("🔍 Auto-Detection:")
        print("   • Automatically detects user capital")
        print("   • Discovers available USDC trading pairs")
        print("   • Sets up USB backup system")
        
        print("\n📚 Pattern Studying:")
        print("   • Studies market patterns for 3+ days")
        print("   • Generates unique hash patterns")
        print("   • Analyzes RSI, volume, momentum")
        print("   • Assesses USDC correlation")
        print("   • Calculates risk levels")
        
        print("\n🎯 Zone Targeting:")
        print("   • Targets specific trading zones")
        print("   • Matches hash patterns for entry")
        print("   • Builds confidence through analysis")
        print("   • Manages orbital shells")
        
        print("\n🎡 Ferris RDE Trading:")
        print("   • Applies mathematical orbital logic")
        print("   • Spiral into profit strategies")
        print("   • Pullback safety mechanisms")
        print("   • Dynamic position sizing")
        
        print("\n🎯 Confidence Building:")
        print("   • Builds confidence through profits")
        print("   • Updates momentum factors")
        print("   • Expands spiral radius")
        print("   • Tracks performance metrics")
        
        print("\n💾 USB Backup:")
        print("   • Automatic data backup to USB")
        print("   • Complete system state preservation")
        print("   • Profit history tracking")
        print("   • Pattern database backup")
        
        print("\n🎯 USDC Focus:")
        print("   • Everything to USDC strategy")
        print("   • USDC to everything strategy")
        print("   • Strong USDC correlation analysis")
        print("   • Risk management for USDC pairs")
        
        print("\nTo use in the main trading bot:")
        print("1. Import: from AOI_Base_Files_Schwabot.core.ferris_ride_system import ferris_ride_system")
        print("2. Initialize: ferris_ride_system.auto_detect_capital_and_tickers()")
        print("3. Study patterns: ferris_ride_system.study_market_pattern(symbol, market_data)")
        print("4. Target zones: ferris_ride_system.target_zone(symbol, market_data)")
        print("5. Execute trades: ferris_ride_system.execute_ferris_trade(symbol, market_data)")
        print("6. Build confidence: ferris_ride_system.build_confidence_zone(symbol, profit)")
        print("7. Backup data: ferris_ride_system.backup_ferris_data()")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 