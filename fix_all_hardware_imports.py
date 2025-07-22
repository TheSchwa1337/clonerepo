#!/usr/bin/env python3
"""
Fix All Hardware Detector Imports
=================================

This script fixes all hardware detector imports across all Schwabot files.
"""

def fix_file_imports(file_path):
    """Fix hardware detector imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic import patterns
        old_patterns = [
            """import sys
import os
sys.path.append('.')
from unified_hardware_detector import unified_hardware_detector as hardware_detector
HardwareAutoDetector = type('HardwareAutoDetector', (), {'detect_hardware': lambda self: hardware_detector.detect_hardware()})()""",
            
            "from .hardware_auto_detector import HardwareAutoDetector"
        ]
        
        new_import = """import sys
import os
sys.path.append('.')
from unified_hardware_detector import UnifiedHardwareDetector
HardwareAutoDetector = UnifiedHardwareDetector"""
        
        fixed = False
        for old_pattern in old_patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_import)
                fixed = True
                print(f"✅ Fixed import pattern in {file_path}")
        
        if fixed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"ℹ️ No hardware detector import found in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all hardware detector imports."""
    print("🔧 Fixing All Hardware Detector Imports")
    print("=" * 40)
    
    # Files to fix
    files_to_fix = [
        "core/tick_loader.py",
        "core/visual_layer_controller.py",
        "AOI_Base_Files_Schwabot/core/koboldcpp_integration.py"
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file_imports(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} files")
    print("\n🎯 Next steps:")
    print("1. Run: python demo_visual_controls.py")
    print("2. Run: python demo_2025_system.py")
    print("3. Test the Flask dashboard")

if __name__ == "__main__":
    main() 