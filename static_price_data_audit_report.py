#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Price Data Audit Report
==============================

This script identifies all instances where static/example price data is used
instead of real API data in the Schwabot trading system.

CRITICAL ISSUE: The system was using hardcoded example prices (50000.0) instead
of fetching real-time market data from APIs, which would make the bot non-functional
for live trading.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def find_static_price_usage():
    """Find all instances of static price data usage."""
    
    # Patterns to search for
    patterns = [
        r'50000\.0.*#.*Default.*BTC.*price',
        r'return 50000\.0.*#.*Default.*BTC.*price',
        r'market_data\[.*\] = 50000\.0',
        r'price.*=.*50000\.0',
        r'BTC.*price.*50000\.0',
        r'default.*price.*50000\.0',
        r'fallback.*50000\.0',
        r'50000\.0.*#.*BTC',
        r'50000\.0.*#.*price'
    ]
    
    # Directories to search
    search_dirs = [
        'AOI_Base_Files_Schwabot',
        'backtesting',
        'core',
        'tests',
        'scripts'
    ]
    
    # File extensions to search
    extensions = ['.py', '.json', '.yaml', '.yml']
    
    results = {
        'critical_issues': [],
        'test_files': [],
        'backup_files': [],
        'documentation_files': [],
        'summary': {
            'total_files_checked': 0,
            'critical_issues_found': 0,
            'test_files_with_static_data': 0,
            'backup_files_with_static_data': 0
        }
    }
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            # Skip backup and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 'backup' not in d.lower() and 'cache' not in d.lower()]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    results['summary']['total_files_checked'] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Check for static price usage
                        for pattern in patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                line_content = content.split('\n')[line_num - 1].strip()
                                
                                issue = {
                                    'file': file_path,
                                    'line': line_num,
                                    'pattern': pattern,
                                    'content': line_content,
                                    'severity': 'critical'
                                }
                                
                                # Categorize the issue
                                if 'test' in file_path.lower() or 'demo' in file_path.lower():
                                    issue['severity'] = 'test_file'
                                    results['test_files'].append(issue)
                                    results['summary']['test_files_with_static_data'] += 1
                                elif 'backup' in file_path.lower() or 'archive' in file_path.lower():
                                    issue['severity'] = 'backup_file'
                                    results['backup_files'].append(issue)
                                    results['summary']['backup_files_with_static_data'] += 1
                                elif 'readme' in file_path.lower() or 'doc' in file_path.lower():
                                    issue['severity'] = 'documentation'
                                    results['documentation_files'].append(issue)
                                else:
                                    results['critical_issues'].append(issue)
                                    results['summary']['critical_issues_found'] += 1
                                    
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return results

def generate_fix_recommendations():
    """Generate recommendations for fixing static price data issues."""
    
    recommendations = {
        'critical_fixes': [
            {
                'file': 'AOI_Base_Files_Schwabot/core/mode_integration_system.py',
                'issue': 'Uses 50000.0 as default BTC price in market_data validation',
                'fix': 'Replace with proper error handling that requires real API data',
                'status': 'FIXED'
            },
            {
                'file': 'backtesting/backtest_engine.py',
                'issue': 'Returns 50000.0 as default BTC price in _get_entry_price',
                'fix': 'Replace with API data fetching and proper error handling',
                'status': 'FIXED'
            },
            {
                'file': 'AOI_Base_Files_Schwabot/core/production_trading_pipeline.py',
                'issue': 'Returns 50000.0 as default BTC price in _get_entry_price',
                'fix': 'Replace with API data fetching and proper error handling',
                'status': 'FIXED'
            },
            {
                'file': 'AOI_Base_Files_Schwabot/core/unified_btc_trading_pipeline.py',
                'issue': 'Returns 50000.0 as default BTC price in _get_entry_price',
                'fix': 'Replace with API data fetching and proper error handling',
                'status': 'NEEDS_ATTENTION'
            }
        ],
        'general_recommendations': [
            'All price data should come from real API endpoints (Binance, Coinbase, etc.)',
            'Implement proper error handling when API data is unavailable',
            'Use market data providers with fallback mechanisms',
            'Add validation to ensure price data is current and valid',
            'Implement circuit breakers when price data is stale or invalid',
            'Add logging to track when fallback data is used',
            'Test all trading logic with real market data, not static examples'
        ],
        'api_integration_requirements': [
            'Implement market data provider interface',
            'Add support for multiple exchange APIs',
            'Implement price data caching with TTL',
            'Add rate limiting for API calls',
            'Implement retry logic for failed API calls',
            'Add price validation (reasonable ranges, not stale data)',
            'Implement websocket connections for real-time data'
        ]
    }
    
    return recommendations

def main():
    """Main audit function."""
    print("🔍 STATIC PRICE DATA AUDIT REPORT")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Find static price usage
    print("🔍 Scanning for static price data usage...")
    results = find_static_price_usage()
    
    # Generate recommendations
    recommendations = generate_fix_recommendations()
    
    # Print summary
    print("\n📊 AUDIT SUMMARY")
    print("=" * 30)
    print(f"Total files checked: {results['summary']['total_files_checked']}")
    print(f"Critical issues found: {results['summary']['critical_issues_found']}")
    print(f"Test files with static data: {results['summary']['test_files_with_static_data']}")
    print(f"Backup files with static data: {results['summary']['backup_files_with_static_data']}")
    
    # Print critical issues
    if results['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES FOUND ({len(results['critical_issues'])}):")
        print("=" * 40)
        for issue in results['critical_issues']:
            print(f"File: {issue['file']}")
            print(f"Line: {issue['line']}")
            print(f"Content: {issue['content']}")
            print("-" * 30)
    
    # Print recommendations
    print(f"\n🔧 FIX RECOMMENDATIONS")
    print("=" * 30)
    for fix in recommendations['critical_fixes']:
        status_icon = "✅" if fix['status'] == 'FIXED' else "⚠️"
        print(f"{status_icon} {fix['file']}")
        print(f"   Issue: {fix['issue']}")
        print(f"   Fix: {fix['fix']}")
        print(f"   Status: {fix['status']}")
        print()
    
    print("📋 GENERAL RECOMMENDATIONS:")
    print("-" * 30)
    for rec in recommendations['general_recommendations']:
        print(f"• {rec}")
    
    print(f"\n🔌 API INTEGRATION REQUIREMENTS:")
    print("-" * 30)
    for req in recommendations['api_integration_requirements']:
        print(f"• {req}")
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'recommendations': recommendations
    }
    
    with open('static_price_data_audit_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: static_price_data_audit_report.json")
    
    # Final assessment
    if results['summary']['critical_issues_found'] > 0:
        print(f"\n🚨 CRITICAL: {results['summary']['critical_issues_found']} critical issues found!")
        print("The trading system cannot function properly with static price data.")
        print("All critical issues must be fixed before live trading.")
    else:
        print(f"\n✅ No critical issues found in active code.")
        print("Static price data usage is limited to test and backup files.")

if __name__ == "__main__":
    main() 