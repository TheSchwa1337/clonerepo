#!/usr/bin/env python3
"""
Fix Visual Layer Controller Import
==================================

This script fixes the hardware detector import in the visual layer controller.
"""

def fix_visual_controller():
    """Fix the hardware detector import in visual_layer_controller.py."""
    try:
        with open('core/visual_layer_controller.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic import lines
        old_import = """import sys
import os
sys.path.append('.')
from unified_hardware_detector import unified_hardware_detector as hardware_detector
HardwareAutoDetector = type('HardwareAutoDetector', (), {'detect_hardware': lambda self: hardware_detector.detect_hardware()})()"""
        
        new_import = """import sys
import os
sys.path.append('.')
from unified_hardware_detector import UnifiedHardwareDetector
HardwareAutoDetector = UnifiedHardwareDetector"""
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Fixed visual layer controller import")
            
            with open('core/visual_layer_controller.py', 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print("ℹ️ Import already fixed or not found")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing visual layer controller: {e}")
        return False

if __name__ == "__main__":
    fix_visual_controller() 