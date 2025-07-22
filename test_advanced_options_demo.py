#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to show the Advanced Options GUI
"""

import sys
import os

# Add the Schwabot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AOI_Base_Files_Schwabot'))

try:
    from advanced_options_gui import show_advanced_options
    import tkinter as tk
    
    print("🔧 Starting Advanced Options GUI Demo...")
    print("This will show you the intelligent compression interface.")
    print("You can explore all the tabs and features.")
    
    # Create root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show the advanced options
    show_advanced_options(root)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 