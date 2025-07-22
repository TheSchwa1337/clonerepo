#!/usr/bin/env python3
"""
Fix Hardware Detector Imports
============================

This script updates all hardware detector imports to use the unified hardware detector.
"""

import os
import re

def fix_file_imports(file_path):
    """Fix hardware detector imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic import
        original_import = "from .hardware_auto_detector import HardwareAutoDetector"
        new_import = "import sys\nimport os\nsys.path.append('.')\nfrom unified_hardware_detector import UnifiedHardwareDetector\nHardwareAutoDetector = UnifiedHardwareDetector"
        
        if original_import in content:
            content = content.replace(original_import, new_import)
            print(f"✅ Fixed imports in {file_path}")
            
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
    print("🔧 Fixing Hardware Detector Imports")
    print("=" * 40)
    
    # Files to fix
    files_to_fix = [
        "AOI_Base_Files_Schwabot/core/koboldcpp_integration.py",
        "core/tick_loader.py",
        "core/visual_layer_controller.py"
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_file_imports(file_path):
                fixed_count += 1
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"\n✅ Fixed {fixed_count} files")
    print("\n🎯 Next steps:")
    print("1. Run: python demo_visual_controls.py")
    print("2. Run: python demo_2025_system.py")
    print("3. Test the Flask dashboard")

if __name__ == "__main__":
    main() 