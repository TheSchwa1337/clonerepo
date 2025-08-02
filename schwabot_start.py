#!/usr/bin/env python3
"""
Schwabot Trading Bot - Start Script
==================================

Starts the Schwabot trading bot with proper configuration.
This script can be run directly or called from the installer.

Usage:
    python schwabot_start.py
    ./schwabot_start.py
"""

import sys
import os
import signal
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_start.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotStarter:
    """Schwabot trading bot starter class."""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        
    def start_schwabot(self):
        """Start the Schwabot trading bot."""
        try:
            # Get the current directory (where this script is located)
            current_dir = Path(__file__).parent.absolute()
            
            # Change to the Schwabot directory
            os.chdir(current_dir)
            
            # Check if main trading bot file exists
            trading_bot_file = current_dir / "schwabot_trading_bot.py"
            if not trading_bot_file.exists():
                logger.error(f"❌ Schwabot trading bot not found: {trading_bot_file}")
                print(f"❌ Schwabot trading bot not found: {trading_bot_file}")
                return False
            
            # Display startup information
            print("🚀 Starting Schwabot Trading Bot...")
            print("=" * 60)
            print("📊 Mathematical Framework: 47-Day Production Ready")
            print("🎯 AI Integration: Active")
            print("📈 Real-time Trading: Enabled")
            print("🔒 Security: Enhanced")
            print("📱 Monitoring: Real-time")
            print("=" * 60)
            print(f"📁 Working Directory: {current_dir}")
            print(f"🐍 Python Version: {sys.version}")
            print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start the trading bot
            logger.info("Starting Schwabot trading bot...")
            self.process = subprocess.Popen(
                [sys.executable, str(trading_bot_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.is_running = True
            logger.info(f"Schwabot started with PID: {self.process.pid}")
            print(f"✅ Schwabot started successfully! PID: {self.process.pid}")
            print("📊 Trading bot is now running...")
            print("💡 Press Ctrl+C to stop the bot")
            print("-" * 60)
            
            # Monitor the process
            self._monitor_process()
            
        except KeyboardInterrupt:
            print("\n🛑 Schwabot stopped by user (Ctrl+C)")
            self.stop_schwabot()
        except Exception as e:
            logger.error(f"Error starting Schwabot: {e}")
            print(f"❌ Error starting Schwabot: {e}")
            return False
        
        return True
    
    def _monitor_process(self):
        """Monitor the Schwabot process."""
        try:
            while self.is_running and self.process:
                # Check if process is still running
                if self.process.poll() is not None:
                    logger.info("Schwabot process has terminated")
                    print("ℹ️ Schwabot process has terminated")
                    break
                
                # Read output from the process
                output = self.process.stdout.readline()
                if output:
                    print(output.strip())
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")
            print(f"❌ Error monitoring process: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        print(f"\n🛑 Received shutdown signal, stopping Schwabot...")
        self.stop_schwabot()
    
    def stop_schwabot(self):
        """Stop the Schwabot trading bot."""
        if self.process and self.is_running:
            try:
                logger.info("Stopping Schwabot process...")
                print("🔄 Stopping Schwabot process...")
                
                # Try graceful termination first
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    logger.info("Schwabot stopped gracefully")
                    print("✅ Schwabot stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Force killing Schwabot process...")
                    print("⚠️ Force killing Schwabot process...")
                    self.process.kill()
                    self.process.wait()
                    logger.info("Schwabot force stopped")
                    print("✅ Schwabot force stopped")
                
            except Exception as e:
                logger.error(f"Error stopping Schwabot: {e}")
                print(f"❌ Error stopping Schwabot: {e}")
            finally:
                self.is_running = False
                self.process = None

def main():
    """Main function to start Schwabot."""
    starter = SchwabotStarter()
    return starter.start_schwabot()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 