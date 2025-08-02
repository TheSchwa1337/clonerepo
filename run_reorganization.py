#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-run reorganization script
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the reorganizer
from enhanced_reorganization_with_verification import EnhancedReorganizer

def main():
    print("🚀 Starting automatic reorganization...")
    
    # Create reorganizer and run
    reorganizer = EnhancedReorganizer()
    success = reorganizer.run()
    
    if success:
        print("\n✅ Reorganization completed successfully!")
        print("Your repository is now organized and all functionality preserved.")
    else:
        print("\n❌ Reorganization failed - system has been rolled back.")
    
    return success

if __name__ == "__main__":
    main() 