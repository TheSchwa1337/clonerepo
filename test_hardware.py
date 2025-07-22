#!/usr/bin/env python3
"""
Simple hardware detection test
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing hardware detection...")
    from core.hardware_auto_detector import hardware_detector
    print("✅ Successfully imported hardware_detector")
    
    hw_info = hardware_detector.detect_hardware()
    print(f"✅ Hardware detected successfully")
    print(f"   Platform: {hw_info.platform}")
    print(f"   CPU Cores: {hw_info.cpu_cores}")
    print(f"   RAM: {hw_info.ram_gb:.1f}GB")
    print(f"   Optimization Mode: {hw_info.optimization_mode.value}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 