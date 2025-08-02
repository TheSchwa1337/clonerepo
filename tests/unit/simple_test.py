#!/usr/bin/env python3
"""
Very simple hardware detection test
"""

import psutil
import platform

print("Testing basic system info...")

try:
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"Platform: {platform.platform()}")
    print("✅ Basic system info works")
except Exception as e:
    print(f"❌ Error: {e}")

print("Test completed!") 