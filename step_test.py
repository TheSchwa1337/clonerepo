#!/usr/bin/env python3
"""
Step-by-step hardware detection test
"""

import sys
import os
import psutil
import platform

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Importing hardware detector...")
try:
    from core.hardware_auto_detector import HardwareAutoDetector
    print("✅ HardwareAutoDetector imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print("\nStep 2: Creating detector instance...")
try:
    detector = HardwareAutoDetector()
    print("✅ Detector instance created successfully")
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
    exit(1)

print("\nStep 3: Getting system info...")
try:
    system_info = detector.get_system_info()
    print(f"✅ System info retrieved: {system_info.cpu_count} cores, {system_info.memory_total / (1024**3):.1f}GB RAM")
except Exception as e:
    print(f"❌ System info failed: {e}")
    exit(1)

print("\nStep 4: Detecting hardware...")
try:
    hw_info = detector.detect_hardware()
    print(f"✅ Hardware detected successfully")
    print(f"   Platform: {hw_info.platform}")
    print(f"   CPU Cores: {hw_info.cpu_cores}")
    print(f"   RAM: {hw_info.ram_gb:.1f}GB")
    print(f"   Optimization Mode: {hw_info.optimization_mode.value}")
except Exception as e:
    print(f"❌ Hardware detection failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests completed!") 