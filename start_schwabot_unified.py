#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Unified System Launcher
================================

Simple launcher script for the complete Schwabot unified trading system.
This script provides easy access to all Schwabot functionality through
the unified interface with KoboldCPP integration.

Usage:
    python start_schwabot_unified.py [mode] [options]

Modes:
    full        - Full integration (default)
    visual      - Visual layer only
    conversation - Conversation interface only
    api         - API only
    dlt         - DLT waveform only

Options:
    --help      - Show this help message
    --config    - Specify config file
    --port      - Override default port
    --model     - Specify KoboldCPP model path
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from core.schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('schwabot_unified.log'),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Schwabot Unified Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_schwabot_unified.py                    # Start full integration
    python start_schwabot_unified.py visual             # Start visual layer only
    python start_schwabot_unified.py conversation       # Start conversation interface
    python start_schwabot_unified.py api                # Start API only
    python start_schwabot_unified.py dlt                # Start DLT waveform only
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        default='full',
        choices=['full', 'visual', 'conversation', 'api', 'dlt'],
        help='Operation mode (default: full)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Override default port'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to KoboldCPP model file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def get_interface_mode(mode_str: str) -> InterfaceMode:
    """Convert mode string to InterfaceMode enum."""
    mode_mapping = {
        'full': InterfaceMode.FULL_INTEGRATION,
        'visual': InterfaceMode.VISUAL_LAYER,
        'conversation': InterfaceMode.CONVERSATION,
        'api': InterfaceMode.API_ONLY,
        'dlt': InterfaceMode.VISUAL_LAYER  # DLT uses visual layer mode
    }
    return mode_mapping.get(mode_str, InterfaceMode.FULL_INTEGRATION)

async def main():
    """Main launcher function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        setup_logging()
    
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    print("=" * 80)
    print("🚀 SCHWABOT UNIFIED TRADING SYSTEM")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {args.config or 'default'}")
    if args.port:
        print(f"Port: {args.port}")
    if args.model:
        print(f"Model: {args.model}")
    print("=" * 80)
    
    try:
        # Get interface mode
        mode = get_interface_mode(args.mode)
        
        # Create unified interface
        unified_interface = SchwabotUnifiedInterface(mode)
        
        # Apply custom configuration if provided
        if args.config:
            logger.info(f"Loading custom configuration: {args.config}")
            # TODO: Implement custom config loading
        
        if args.port:
            logger.info(f"Overriding port: {args.port}")
            # TODO: Implement port override
        
        if args.model:
            logger.info(f"Using model: {args.model}")
            # TODO: Implement model path override
        
        # Start the system
        logger.info(f"Starting Schwabot Unified Interface in {mode.value} mode...")
        await unified_interface.start_unified_system()
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise
    finally:
        # Stop the system
        if 'unified_interface' in locals():
            await unified_interface.stop_unified_system()
            
            # Print final status
            status = unified_interface.get_unified_status()
            logger.info("📊 Final System Status:")
            logger.info(f"   Mode: {status.mode.value}")
            logger.info(f"   Uptime: {status.uptime_seconds:.1f} seconds")
            logger.info(f"   Total Analyses: {status.total_analyses}")
            logger.info(f"   Total Trades: {status.total_trades}")
            logger.info(f"   System Health: {status.system_health}")
            logger.info(f"   KoboldCPP: {'✅' if status.kobold_running else '❌'}")
            logger.info(f"   Visual Layer: {'✅' if status.visual_layer_active else '❌'}")
            logger.info(f"   Trading: {'✅' if status.trading_active else '❌'}")
            logger.info(f"   DLT Waveform: {'✅' if status.dlt_waveform_active else '❌'}")
            logger.info(f"   Conversation: {'✅' if status.conversation_active else '❌'}")
            logger.info(f"   API: {'✅' if status.api_active else '❌'}")
        
        logger.info("👋 Schwabot Unified Interface shutdown complete")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 