#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot AI Trading System Launcher
===================================

Advanced AI-powered trading system with integrated mathematical frameworks,
quantum-inspired algorithms, and real-time market analysis.

Features:
- AI-powered trading decisions
- Advanced mathematical modeling
- Real-time market data integration
- Quantum-inspired optimization
- Multi-dimensional profit analysis
- Secure API integration
- Visual trading interface
"""

import sys
import os
from pathlib import Path

# Add the koboldcpp directory to the path
koboldcpp_dir = Path(__file__).parent / "koboldcpp"
sys.path.insert(0, str(koboldcpp_dir))

# Import and run the main system
if __name__ == "__main__":
    try:
        from koboldcpp import main, argparse
        
        # Create argument parser with Schwabot branding
        parser = argparse.ArgumentParser(
            description="Schwabot AI Trading System - Advanced AI-Powered Trading Platform",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python schwabot_ai_system.py --model model.gguf
  python schwabot_ai_system.py --model model.gguf --port 5001
  python schwabot_ai_system.py --showgui
            """
        )
        
        # Parse arguments and run
        args = parser.parse_args()
        main(args, args)
        
    except ImportError as e:
        print(f"❌ Error importing Schwabot AI Trading System: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Schwabot AI Trading System: {e}")
        sys.exit(1)
