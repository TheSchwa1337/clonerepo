#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to check imports.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    print("1. Testing core module import...")
    import AOI_Base_Files_Schwabot.core
    print("✅ Core module imported successfully")
except Exception as e:
    print(f"❌ Core module import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing utils module import...")
    import AOI_Base_Files_Schwabot.utils
    print("✅ Utils module imported successfully")
except Exception as e:
    print(f"❌ Utils module import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing backtesting module import...")
    import AOI_Base_Files_Schwabot.backtesting
    print("✅ Backtesting module imported successfully")
except Exception as e:
    print(f"❌ Backtesting module import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed!") 