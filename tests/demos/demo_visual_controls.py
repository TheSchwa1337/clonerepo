#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Visual Controls Demo
======================

Demo script to showcase the Schwabot Visual Controls System
"""

import sys
import os
import logging

# Add the Schwabot directory to the path
sys.path.append('AOI_Base_Files_Schwabot')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main demo function."""
    print("🎨 SCHWABOT VISUAL CONTROLS DEMO")
    print("=" * 50)
    
    try:
        # Import the visual controls
        from visual_controls_gui import show_visual_controls, VisualControlsGUI
        
        print("✅ Visual Controls imported successfully")
        print("🚀 Launching Visual Controls GUI...")
        
        # Create and show the visual controls GUI
        gui = show_visual_controls()
        
        print("🎨 Visual Controls GUI launched!")
        print("\n📋 Available Features:")
        print("  📊 Chart Controls - Customize chart appearance")
        print("  🔧 Layer Management - Manage visual layers")
        print("  🔍 Pattern Recognition - Detect trading patterns")
        print("  🤖 AI Analysis - AI-powered chart analysis")
        print("  📊 Performance - Monitor system performance")
        print("  ⚙️ Settings - Configure system settings")
        
        print("\n💡 Tips:")
        print("  • Use the '📊 Chart Controls' tab to customize charts")
        print("  • Try different chart types and styles")
        print("  • Adjust layer opacity and visibility")
        print("  • Configure pattern recognition settings")
        print("  • Monitor performance metrics in real-time")
        
        # Keep the GUI running
        if gui.root:
            print("\n🔄 GUI is running... Close the window to exit")
            gui.root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    main() 