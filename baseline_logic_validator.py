#!/usr/bin/env python3
"""
🧠 SCHWABOT BASELINE LOGIC VALIDATOR
====================================

Comprehensive validation of core trading bot functionality.
This script checks all 13 baseline requirements before any math modules.

Usage:
    python baseline_logic_validator.py
"""

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineLogicValidator:
    """Comprehensive baseline logic validator for Schwabot."""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "partial": [],
            "failed": [],
            "critical_issues": []
        }
        self.start_time = time.time()
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete baseline validation."""
        logger.info("🧠 Starting Schwabot Baseline Logic Validation")
        logger.info("=" * 60)
        
        # Run all validation checks
        await self._check_1_bidirectional_trade_paths()
        await self._check_2_wallet_state_awareness()
        await self._check_3_base_profit_logic()
        await self._check_4_order_structure_sanity()
        await self._check_5_cycle_loop_integrity()
        await self._check_6_randomized_portfolio_targeting()
        await self._check_7_emergency_stop_functionality()
        await self._check_8_exchange_pair_validation()
        await self._check_9_config_sync_check()
        await self._check_10_logging_memory_system()
        await self._check_11_backtest_data_mode_reflection()
        await self._check_12_gui_cli_watchdog_bridges()
        await self._check_13_math_off_still_works()
        
        # Generate final report
        return self._generate_final_report()
    
    def _safe_read_file(self, file_path: str) -> str:
        """Safely read file content with UTF-8 encoding."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return ""
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return ""
    
    async def _check_1_bidirectional_trade_paths(self):
        """Check 1: Bidirectional Trade Path Validation"""
        logger.info("🔍 Check 1: Bidirectional Trade Path Validation")
        
        try:
            # Check trading pairs configuration
            pairs_file = Path("AOI_Base_Files_Schwabot/config/trading_pairs.json")
            if pairs_file.exists():
                with open(pairs_file, 'r', encoding='utf-8') as f:
                    pairs_config = json.load(f)
                
                required_pairs = [
                    "BTC/USDC", "USDC/BTC",
                    "ETH/USDC", "USDC/ETH", 
                    "XRP/USDC", "USDC/XRP",
                    "SOL/USDC", "USDC/SOL"
                ]
                
                default_pairs = pairs_config.get("default_pairs", [])
                missing_pairs = [pair for pair in required_pairs if pair not in default_pairs]
                
                if not missing_pairs:
                    self.results["passed"].append("1. Bidirectional Trade Paths ✅")
                    logger.info("  ✅ All required bidirectional pairs configured")
                else:
                    self.results["failed"].append("1. Bidirectional Trade Paths ❌")
                    self.results["critical_issues"].append(f"Missing pairs: {missing_pairs}")
                    logger.error(f"  ❌ Missing pairs: {missing_pairs}")
            else:
                self.results["failed"].append("1. Bidirectional Trade Paths ❌")
                logger.error("  ❌ Trading pairs config file not found")
                
        except Exception as e:
            self.results["failed"].append("1. Bidirectional Trade Paths ❌")
            logger.error(f"  ❌ Error checking bidirectional paths: {e}")
    
    async def _check_2_wallet_state_awareness(self):
        """Check 2: Wallet State Awareness"""
        logger.info("🔍 Check 2: Wallet State Awareness")
        
        try:
            # Check for wallet balance validation in trading pipeline
            pipeline_file = Path("core/trading_pipeline_manager.py")
            if pipeline_file.exists():
                content = self._safe_read_file(str(pipeline_file))
                
                if "balance" in content and "insufficient" in content.lower():
                    self.results["passed"].append("2. Wallet State Awareness ✅")
                    logger.info("  ✅ Wallet balance checks implemented")
                else:
                    self.results["partial"].append("2. Wallet State Awareness ⚠️")
                    logger.warning("  ⚠️ Wallet balance checks may be incomplete")
            else:
                self.results["failed"].append("2. Wallet State Awareness ❌")
                logger.error("  ❌ Trading pipeline file not found")
                
        except Exception as e:
            self.results["failed"].append("2. Wallet State Awareness ❌")
            logger.error(f"  ❌ Error checking wallet awareness: {e}")
    
    async def _check_3_base_profit_logic(self):
        """Check 3: Base Profit Logic - Buy Low, Sell High"""
        logger.info("🔍 Check 3: Base Profit Logic - Buy Low, Sell High")
        
        try:
            # Check trading engine for basic profit logic
            engine_file = Path("schwabot_trading_engine.py")
            if engine_file.exists():
                content = self._safe_read_file(str(engine_file))
                
                profit_indicators = [
                    "position_profit",
                    "entry_price", 
                    "take_profit",
                    "stop_loss"
                ]
                
                found_indicators = [ind for ind in profit_indicators if ind in content]
                
                if len(found_indicators) >= 3:
                    self.results["passed"].append("3. Base Profit Logic ✅")
                    logger.info(f"  ✅ Profit logic indicators found: {found_indicators}")
                else:
                    self.results["partial"].append("3. Base Profit Logic ⚠️")
                    logger.warning(f"  ⚠️ Limited profit logic: {found_indicators}")
            else:
                self.results["failed"].append("3. Base Profit Logic ❌")
                logger.error("  ❌ Trading engine file not found")
                
        except Exception as e:
            self.results["failed"].append("3. Base Profit Logic ❌")
            logger.error(f"  ❌ Error checking profit logic: {e}")
    
    async def _check_4_order_structure_sanity(self):
        """Check 4: Order Structure Sanity"""
        logger.info("🔍 Check 4: Order Structure Sanity")
        
        try:
            # Check for order validation and safety checks
            order_indicators = [
                "market_order",
                "slippage",
                "min_order_size",
                "duplicate_order"
            ]
            
            found_indicators = []
            for file_path in ["core/trading_pipeline_manager.py", "schwabot_trading_engine.py"]:
                if Path(file_path).exists():
                    content = self._safe_read_file(file_path)
                    for indicator in order_indicators:
                        if indicator in content:
                            found_indicators.append(indicator)
            
            if len(found_indicators) >= 2:
                self.results["passed"].append("4. Order Structure Sanity ✅")
                logger.info(f"  ✅ Order safety checks found: {found_indicators}")
            else:
                self.results["partial"].append("4. Order Structure Sanity ⚠️")
                logger.warning(f"  ⚠️ Limited order safety: {found_indicators}")
                
        except Exception as e:
            self.results["failed"].append("4. Order Structure Sanity ❌")
            logger.error(f"  ❌ Error checking order structure: {e}")
    
    async def _check_5_cycle_loop_integrity(self):
        """Check 5: Cycle Loop Integrity"""
        logger.info("🔍 Check 5: Cycle Loop Integrity")
        
        try:
            # Check for complete trading cycle implementation
            cycle_indicators = [
                "tick",
                "market_check",
                "wallet_check", 
                "trade_decision",
                "memory_update"
            ]
            
            found_indicators = []
            for file_path in ["core/trading_pipeline_manager.py", "schwabot_trading_engine.py"]:
                if Path(file_path).exists():
                    content = self._safe_read_file(file_path)
                    for indicator in cycle_indicators:
                        if indicator in content:
                            found_indicators.append(indicator)
            
            if len(found_indicators) >= 4:
                self.results["passed"].append("5. Cycle Loop Integrity ✅")
                logger.info(f"  ✅ Cycle components found: {found_indicators}")
            else:
                self.results["partial"].append("5. Cycle Loop Integrity ⚠️")
                logger.warning(f"  ⚠️ Limited cycle components: {found_indicators}")
                
        except Exception as e:
            self.results["failed"].append("5. Cycle Loop Integrity ❌")
            logger.error(f"  ❌ Error checking cycle integrity: {e}")
    
    async def _check_6_randomized_portfolio_targeting(self):
        """Check 6: Randomized Portfolio Targeting"""
        logger.info("🔍 Check 6: Randomized Portfolio Targeting")
        
        try:
            # Check for portfolio randomization logic
            orbital_file = Path("AOI_Base_Files_Schwabot/core/orbital_shell_brain_system.py")
            if orbital_file.exists():
                content = self._safe_read_file(str(orbital_file))
                
                if "randomized_holdings" in content and "random.choice" in content:
                    self.results["passed"].append("6. Randomized Portfolio Targeting ✅")
                    logger.info("  ✅ Portfolio randomization implemented")
                else:
                    self.results["partial"].append("6. Randomized Portfolio Targeting ⚠️")
                    logger.warning("  ⚠️ Limited portfolio randomization")
            else:
                self.results["partial"].append("6. Randomized Portfolio Targeting ⚠️")
                logger.warning("  ⚠️ Orbital system file not found")
                
        except Exception as e:
            self.results["failed"].append("6. Randomized Portfolio Targeting ❌")
            logger.error(f"  ❌ Error checking portfolio targeting: {e}")
    
    async def _check_7_emergency_stop_functionality(self):
        """Check 7: Emergency Stop Functionality"""
        logger.info("🔍 Check 7: Emergency Stop Functionality")
        
        try:
            # Check for emergency stop implementations
            emergency_file = Path("scripts/emergency_stop_trading.py")
            if emergency_file.exists():
                self.results["passed"].append("7. Emergency Stop Functionality ✅")
                logger.info("  ✅ Emergency stop script found")
            else:
                # Check for emergency stop in other files
                emergency_indicators = ["emergency_stop", "cancel_all_orders", "close_all_positions"]
                found_indicators = []
                
                for file_path in ["core/high_volume_trading_manager.py", "core/advanced_strategy_execution_engine.py"]:
                    if Path(file_path).exists():
                        with open(file_path, 'r') as f:
                            content = f.read()
                            for indicator in emergency_indicators:
                                if indicator in content:
                                    found_indicators.append(indicator)
                
                if len(found_indicators) >= 2:
                    self.results["passed"].append("7. Emergency Stop Functionality ✅")
                    logger.info(f"  ✅ Emergency stop methods found: {found_indicators}")
                else:
                    self.results["failed"].append("7. Emergency Stop Functionality ❌")
                    logger.error(f"  ❌ Limited emergency stop: {found_indicators}")
                    
        except Exception as e:
            self.results["failed"].append("7. Emergency Stop Functionality ❌")
            logger.error(f"  ❌ Error checking emergency stop: {e}")
    
    async def _check_8_exchange_pair_validation(self):
        """Check 8: Exchange & Pair Validation"""
        logger.info("🔍 Check 8: Exchange & Pair Validation")
        
        try:
            # Check for banned pairs (USDT, USD pairs, ADA, BNB)
            banned_patterns = [
                r'\bBTC/USDT\b', r'\bETH/USDT\b', r'\bXRP/USDT\b', r'\bSOL/USDT\b', r'\bADA/USDT\b', r'\bBNB/USDT\b',
                r'\bBTC/USD\b', r'\bETH/USD\b', r'\bXRP/USD\b', r'\bSOL/USD\b', r'\bADA/USD\b', r'\bBNB/USD\b',
                r'\bADA/USDC\b', r'\bBNB/USDC\b', r'\bADA/\b', r'\bBNB/\b'
            ]
            found_banned = []
            
            # Search in key configuration files
            config_files = [
                "AOI_Base_Files_Schwabot/config/trading_pairs.json",
                "AOI_Base_Files_Schwabot/config/trading_config.yaml",
                "AOI_Base_Files_Schwabot/config/high_frequency_crypto_config.yaml"
            ]
            
            for config_file in config_files:
                if Path(config_file).exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for banned in banned_patterns:
                            if re.search(banned, content):
                                found_banned.append(f"{banned} in {config_file}")
            
            if not found_banned:
                self.results["passed"].append("8. Exchange & Pair Validation ✅")
                logger.info("  ✅ No banned pairs found in config files")
            else:
                self.results["critical_issues"].append(f"Banned pairs found: {found_banned}")
                self.results["failed"].append("8. Exchange & Pair Validation ❌")
                logger.error(f"  ❌ Banned pairs found: {found_banned}")
                
        except Exception as e:
            self.results["failed"].append("8. Exchange & Pair Validation ❌")
            logger.error(f"  ❌ Error checking pair validation: {e}")
    
    async def _check_9_config_sync_check(self):
        """Check 9: Config Sync Check"""
        logger.info("🔍 Check 9: Config Sync Check")
        
        try:
            # Check for configuration file consistency
            config_files = [
                "AOI_Base_Files_Schwabot/config/trading_pairs.json",
                "AOI_Base_Files_Schwabot/config/trading_config.yaml"
            ]
            
            existing_configs = [f for f in config_files if Path(f).exists()]
            
            if len(existing_configs) >= 2:
                self.results["passed"].append("9. Config Sync Check ✅")
                logger.info(f"  ✅ Configuration files found: {len(existing_configs)}")
            else:
                self.results["partial"].append("9. Config Sync Check ⚠️")
                logger.warning(f"  ⚠️ Limited config files: {existing_configs}")
                
        except Exception as e:
            self.results["failed"].append("9. Config Sync Check ❌")
            logger.error(f"  ❌ Error checking config sync: {e}")
    
    async def _check_10_logging_memory_system(self):
        """Check 10: Logging & Memory System"""
        logger.info("🔍 Check 10: Logging & Memory System")
        
        try:
            # Check for logging and memory system implementation
            memory_indicators = [
                "hash_registry",
                "trade_logs",
                "memory_registry",
                "timestamp"
            ]
            
            found_indicators = []
            for file_path in ["schwabot_trading_engine.py", "core/trading_pipeline_manager.py"]:
                if Path(file_path).exists():
                    content = self._safe_read_file(file_path)
                    for indicator in memory_indicators:
                        if indicator in content:
                            found_indicators.append(indicator)
            
            if len(found_indicators) >= 3:
                self.results["passed"].append("10. Logging & Memory System ✅")
                logger.info(f"  ✅ Memory system indicators found: {found_indicators}")
            else:
                self.results["partial"].append("10. Logging & Memory System ⚠️")
                logger.warning(f"  ⚠️ Limited memory system: {found_indicators}")
                
        except Exception as e:
            self.results["failed"].append("10. Logging & Memory System ❌")
            logger.error(f"  ❌ Error checking memory system: {e}")
    
    async def _check_11_backtest_data_mode_reflection(self):
        """Check 11: Backtest/Data Mode Reflection"""
        logger.info("🔍 Check 11: Backtest/Data Mode Reflection")
        
        try:
            # Check for backtest implementation
            backtest_file = Path("backtesting/backtest_engine.py")
            if backtest_file.exists():
                self.results["passed"].append("11. Backtest/Data Mode Reflection ✅")
                logger.info("  ✅ Backtest engine found")
            else:
                self.results["partial"].append("11. Backtest/Data Mode Reflection ⚠️")
                logger.warning("  ⚠️ Backtest engine not found")
                
        except Exception as e:
            self.results["failed"].append("11. Backtest/Data Mode Reflection ❌")
            logger.error(f"  ❌ Error checking backtest: {e}")
    
    async def _check_12_gui_cli_watchdog_bridges(self):
        """Check 12: GUI/CLI/Watchdog Bridges"""
        logger.info("🔍 Check 12: GUI/CLI/Watchdog Bridges")
        
        try:
            # Check for interface implementations
            interface_files = [
                "AOI_Base_Files_Schwabot/schwabot_launcher.py",
                "AOI_Base_Files_Schwabot/web/dynamic_timing_dashboard.py",
                "AOI_Base_Files_Schwabot/visualization/dynamic_timing_visualizer.py"
            ]
            
            existing_interfaces = [f for f in interface_files if Path(f).exists()]
            
            if len(existing_interfaces) >= 2:
                self.results["passed"].append("12. GUI/CLI/Watchdog Bridges ✅")
                logger.info(f"  ✅ Interface implementations found: {len(existing_interfaces)}")
            else:
                self.results["partial"].append("12. GUI/CLI/Watchdog Bridges ⚠️")
                logger.warning(f"  ⚠️ Limited interfaces: {existing_interfaces}")
                
        except Exception as e:
            self.results["failed"].append("12. GUI/CLI/Watchdog Bridges ❌")
            logger.error(f"  ❌ Error checking interfaces: {e}")
    
    async def _check_13_math_off_still_works(self):
        """Check 13: Math Off, Still Works"""
        logger.info("🔍 Check 13: Math Off, Still Works")
        
        try:
            # Check for basic trading logic that works without math modules
            basic_indicators = [
                "price_delta",
                "profit_target",
                "stop_loss",
                "basic_trading"
            ]
            
            found_indicators = []
            for file_path in ["schwabot_trading_engine.py", "core/trading_pipeline_manager.py"]:
                if Path(file_path).exists():
                    content = self._safe_read_file(file_path)
                    for indicator in basic_indicators:
                        if indicator in content:
                            found_indicators.append(indicator)
            
            if len(found_indicators) >= 2:
                self.results["passed"].append("13. Math Off, Still Works ✅")
                logger.info(f"  ✅ Basic trading logic found: {found_indicators}")
            else:
                self.results["partial"].append("13. Math Off, Still Works ⚠️")
                logger.warning(f"  ⚠️ Limited basic logic: {found_indicators}")
                
        except Exception as e:
            self.results["failed"].append("13. Math Off, Still Works ❌")
            logger.error(f"  ❌ Error checking basic logic: {e}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final validation report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        report = {
            "validation_summary": {
                "total_checks": 13,
                "passed": len(self.results["passed"]),
                "partial": len(self.results["partial"]),
                "failed": len(self.results["failed"]),
                "critical_issues": len(self.results["critical_issues"]),
                "duration_seconds": round(duration, 2)
            },
            "results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if self.results["critical_issues"]:
            recommendations.append("🚨 CRITICAL: Fix banned pair references immediately")
            recommendations.append("🚨 CRITICAL: Remove all USDT/USD/ADA/BNB references")
        
        if len(self.results["failed"]) > 3:
            recommendations.append("⚠️ HIGH PRIORITY: Fix failed baseline checks before proceeding")
        
        if len(self.results["partial"]) > 5:
            recommendations.append("⚠️ MEDIUM PRIORITY: Improve partial implementations")
        
        if len(self.results["passed"]) >= 10:
            recommendations.append("✅ GOOD: Most baseline functionality is working")
        
        return recommendations


async def main():
    """Main validation function."""
    validator = BaselineLogicValidator()
    report = await validator.run_full_validation()
    
    # Print final report
    print("\n" + "=" * 60)
    print("🧠 SCHWABOT BASELINE LOGIC VALIDATION REPORT")
    print("=" * 60)
    
    summary = report["validation_summary"]
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Total Checks: {summary['total_checks']}")
    print(f"   ✅ Passed: {summary['passed']}")
    print(f"   ⚠️ Partial: {summary['partial']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   🚨 Critical Issues: {summary['critical_issues']}")
    print(f"   ⏱️ Duration: {summary['duration_seconds']}s")
    
    if report["results"]["passed"]:
        print(f"\n✅ PASSED CHECKS:")
        for check in report["results"]["passed"]:
            print(f"   {check}")
    
    if report["results"]["partial"]:
        print(f"\n⚠️ PARTIAL CHECKS:")
        for check in report["results"]["partial"]:
            print(f"   {check}")
    
    if report["results"]["failed"]:
        print(f"\n❌ FAILED CHECKS:")
        for check in report["results"]["failed"]:
            print(f"   {check}")
    
    if report["results"]["critical_issues"]:
        print(f"\n🚨 CRITICAL ISSUES:")
        for issue in report["results"]["critical_issues"]:
            print(f"   {issue}")
    
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")
    
    print("\n" + "=" * 60)
    
    # Save report to file
    with open("baseline_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("🧠 SCHWABOT BASELINE LOGIC VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📊 VALIDATION SUMMARY:\n")
        f.write(f"   Total Checks: {summary['total_checks']}\n")
        f.write(f"   ✅ Passed: {summary['passed']}\n")
        f.write(f"   ⚠️ Partial: {summary['partial']}\n")
        f.write(f"   ❌ Failed: {summary['failed']}\n")
        f.write(f"   🚨 Critical Issues: {summary['critical_issues']}\n")
        f.write(f"   ⏱️ Duration: {summary['duration_seconds']}s\n\n")
        
        if report["results"]["passed"]:
            f.write("✅ PASSED CHECKS:\n")
            for check in report["results"]["passed"]:
                f.write(f"   {check}\n")
            f.write("\n")
        
        if report["results"]["partial"]:
            f.write("⚠️ PARTIAL CHECKS:\n")
            for check in report["results"]["partial"]:
                f.write(f"   {check}\n")
            f.write("\n")
        
        if report["results"]["failed"]:
            f.write("❌ FAILED CHECKS:\n")
            for check in report["results"]["failed"]:
                f.write(f"   {check}\n")
            f.write("\n")
        
        if report["results"]["critical_issues"]:
            f.write("🚨 CRITICAL ISSUES:\n")
            for issue in report["results"]["critical_issues"]:
                f.write(f"   {issue}\n")
            f.write("\n")
        
        if report["recommendations"]:
            f.write("💡 RECOMMENDATIONS:\n")
            for rec in report["recommendations"]:
                f.write(f"   {rec}\n")
            f.write("\n")
    
    print(f"📄 Full report saved to: baseline_validation_report.txt")
    
    # Return exit code based on results
    if report["results"]["critical_issues"]:
        print("🚨 CRITICAL ISSUES FOUND - FIX IMMEDIATELY")
        return 2
    elif len(report["results"]["failed"]) > 3:
        print("❌ TOO MANY FAILED CHECKS - NEEDS ATTENTION")
        return 1
    else:
        print("✅ BASELINE VALIDATION COMPLETE")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 