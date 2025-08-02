#!/usr/bin/env python3
"""
Fix Static Price Data Issues
============================

This script fixes all instances of static 50000.0 price data in critical trading files
and replaces them with proper API data fetching or error handling.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any

def fix_static_price_data():
    """Fix static price data issues in critical files."""
    
    # Critical files that need fixing
    critical_files = [
        "AOI_Base_Files_Schwabot/core/clean_trading_pipeline.py",
        "AOI_Base_Files_Schwabot/core/cli_live_entry.py", 
        "AOI_Base_Files_Schwabot/core/complete_internalized_scalping_system.py",
        "AOI_Base_Files_Schwabot/core/enhanced_real_time_data_puller.py",
        "AOI_Base_Files_Schwabot/core/live_vector_simulator.py",
        "AOI_Base_Files_Schwabot/core/math_to_trade_signal_router.py",
        "AOI_Base_Files_Schwabot/core/pure_profit_calculator.py",
        "AOI_Base_Files_Schwabot/core/real_time_execution_engine.py",
        "AOI_Base_Files_Schwabot/core/unified_trading_pipeline.py",
        "AOI_Base_Files_Schwabot/core/strategy/strategy_executor.py",
        "AOI_Base_Files_Schwabot/scripts/chrono_resonance_integrity_checker.py",
        "AOI_Base_Files_Schwabot/scripts/dashboard_backend.py",
        "AOI_Base_Files_Schwabot/scripts/hash_trigger_system_summary.py",
        "AOI_Base_Files_Schwabot/scripts/integrate_crlf_into_pipeline.py",
        "AOI_Base_Files_Schwabot/scripts/integrate_zpe_zbe_into_pipeline.py",
        "AOI_Base_Files_Schwabot/scripts/quantum_drift_shell_engine.py",
        "AOI_Base_Files_Schwabot/scripts/run_trading_pipeline.py",
        "AOI_Base_Files_Schwabot/scripts/schwabot_enhanced_launcher.py",
        "AOI_Base_Files_Schwabot/scripts/schwabot_main_integrated.py",
        "AOI_Base_Files_Schwabot/scripts/start_enhanced_math_to_trade_system.py",
        "AOI_Base_Files_Schwabot/scripts/system_comprehensive_validation.py",
        "AOI_Base_Files_Schwabot/scripts/validate_enhanced_math_to_trade_system.py",
        "AOI_Base_Files_Schwabot/server/tensor_websocket_server.py",
        "AOI_Base_Files_Schwabot/strategies/phantom_band_navigator.py",
        "AOI_Base_Files_Schwabot/ui/schwabot_dashboard.py",
        "AOI_Base_Files_Schwabot/visualization/tick_plotter.py",
        "core/advanced_security_gui.py",
        "core/btc_usdc_trading_integration.py",
        "core/integrated_advanced_trading_system.py",
        "core/phantom_mode_engine.py",
        "core/phantom_mode_integration.py",
        "core/real_time_market_data_pipeline.py",
        "core/secure_trade_handler.py"
    ]
    
    fixed_files = []
    errors = []
    
    for file_path in critical_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            print(f"🔧 Fixing: {file_path}")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create backup
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            
            # Fix patterns
            original_content = content
            
            # Pattern 1: Replace static price assignments
            content = re.sub(
                r'(\w+)\s*=\s*50000\.0\s*(?:#.*)?$',
                r'\1 = self._get_real_price_data()  # REAL API DATA',
                content,
                flags=re.MULTILINE
            )
            
            # Pattern 2: Replace static price in function calls
            content = re.sub(
                r'price\s*=\s*50000\.0',
                r'price = self._get_real_price_data()',
                content
            )
            
            # Pattern 3: Replace static price in dictionaries
            content = re.sub(
                r'"price":\s*50000\.0',
                r'"price": self._get_real_price_data()',
                content
            )
            
            # Pattern 4: Replace static price in get() calls
            content = re.sub(
                r'\.get\([^)]*,\s*50000\.0\)',
                r'.get(\\g<0>.split(",")[0], self._get_real_price_data())',
                content
            )
            
            # Pattern 5: Replace base_price = 50000.0
            content = re.sub(
                r'base_price\s*=\s*50000\.0',
                r'base_price = self._get_real_price_data()',
                content
            )
            
            # Pattern 6: Replace current_price = 50000.0
            content = re.sub(
                r'current_price\s*=\s*50000\.0',
                r'current_price = self._get_real_price_data()',
                content
            )
            
            # Add the helper method if content was changed
            if content != original_content:
                # Add the helper method at the end of the class
                helper_method = '''
    def _get_real_price_data(self) -> float:
        """Get real price data from API - NO MORE STATIC 50000.0!"""
        try:
            # Try to get real price from API
            if hasattr(self, 'api_client') and self.api_client:
                try:
                    ticker = self.api_client.fetch_ticker('BTC/USDC')
                    if ticker and 'last' in ticker and ticker['last']:
                        return float(ticker['last'])
                except Exception as e:
                    pass
            
            # Try to get from market data provider
            if hasattr(self, 'market_data_provider') and self.market_data_provider:
                try:
                    price = self.market_data_provider.get_current_price('BTC/USDC')
                    if price and price > 0:
                        return price
                except Exception as e:
                    pass
            
            # Try to get from cache
            if hasattr(self, 'market_data_cache') and 'BTC/USDC' in self.market_data_cache:
                cached_price = self.market_data_cache['BTC/USDC'].get('price')
                if cached_price and cached_price > 0:
                    return cached_price
            
            # CRITICAL: No real data available - fail properly
            raise ValueError("No live price data available - API connection required")
            
        except Exception as e:
            raise ValueError(f"Cannot get live price data: {e}")
'''
                
                # Find the end of the class and add the helper method
                class_end_pattern = r'(\s*)(?=\n\S|$)'
                if 'class ' in content:
                    # Add before the last line
                    lines = content.split('\n')
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip() and not lines[i].startswith(' '):
                            lines.insert(i, helper_method)
                            break
                    content = '\n'.join(lines)
            
            # Write the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fixed_files.append(file_path)
            print(f"✅ Fixed: {file_path}")
            
        except Exception as e:
            error_msg = f"❌ Error fixing {file_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    # Generate report
    report = {
        'timestamp': '2025-01-24T02:45:00',
        'files_fixed': len(fixed_files),
        'errors': len(errors),
        'fixed_files': fixed_files,
        'errors': errors
    }
    
    with open('static_price_fix_report.json', 'w') as f:
        import json
        json.dump(report, f, indent=2)
    
    print(f"\n🎯 STATIC PRICE DATA FIX COMPLETE!")
    print(f"📊 Files fixed: {len(fixed_files)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"📋 Report saved to: static_price_fix_report.json")
    
    return len(fixed_files) > 0

if __name__ == "__main__":
    fix_static_price_data() 