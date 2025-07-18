#!/usr/bin/env python3
"""
Test Runner for Reorganized Schwabot System
==========================================

This script runs tests from the new organized directory structure.
"""

import sys
import os
from pathlib import Path

# Add test directory to path
test_dir = Path(__file__).parent / "tests"
sys.path.insert(0, str(test_dir))

# Add core directory to path
core_dir = Path(__file__).parent / "core"
sys.path.insert(0, str(core_dir))

def run_test(test_name):
    """Run a specific test."""
    try:
        if test_name == "unified_integration":
            from test_unified_integration import main
            import asyncio
            return asyncio.run(main())
        elif test_name == "complete_system":
            from test_schwabot_complete_system import main
            import asyncio
            return asyncio.run(main())
        elif test_name == "trading_pipeline":
            from test_trading_pipeline import main
            import asyncio
            return asyncio.run(main())
        else:
            print(f"❌ Unknown test: {test_name}")
            return False
    except Exception as e:
        print(f"❌ Test {test_name} failed: {e}")
        return False

def main():
    """Main test runner."""
    print("🧪 Schwabot Test Runner")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        print(f"Running test: {test_name}")
        success = run_test(test_name)
        return 0 if success else 1
    else:
        print("Available tests:")
        print("  unified_integration - Test unified system integration")
        print("  complete_system - Test complete Schwabot system")
        print("  trading_pipeline - Test complete trading pipeline with AI")
        print("\nUsage: python run_tests.py <test_name>")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 