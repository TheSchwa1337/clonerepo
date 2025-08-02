#!/usr/bin/env python3
"""
Schwabot Trading Bot - Stop Script
=================================

Safely stops the Schwabot trading bot.
This script can be run directly or called from the installer.

Usage:
    python schwabot_stop.py
    ./schwabot_stop.py
"""

import os
import signal
import subprocess
import psutil
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_stop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotStopper:
    """Schwabot trading bot stopper class."""
    
    def __init__(self):
        self.stopped_processes = []
        
    def stop_schwabot(self):
        """Stop the Schwabot trading bot."""
        try:
            print("🛑 Stopping Schwabot Trading Bot...")
            print("=" * 50)
            
            # Find Schwabot processes
            schwabot_processes = self._find_schwabot_processes()
            
            if schwabot_processes:
                print(f"📋 Found {len(schwabot_processes)} Schwabot process(es)")
                print("-" * 50)
                
                for proc in schwabot_processes:
                    self._stop_process(proc)
                
                # Wait a moment for processes to terminate
                time.sleep(2)
                
                # Check if any processes are still running
                remaining_processes = self._find_schwabot_processes()
                if remaining_processes:
                    print("⚠️ Some processes are still running, force killing...")
                    for proc in remaining_processes:
                        self._force_kill_process(proc)
                
                print("-" * 50)
                print(f"✅ Successfully stopped {len(self.stopped_processes)} Schwabot process(es)")
                
            else:
                print("ℹ️ No Schwabot processes found")
                print("✅ Schwabot Trading Bot is not running")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Schwabot: {e}")
            print(f"❌ Error stopping Schwabot: {e}")
            return False
    
    def _find_schwabot_processes(self):
        """Find all Schwabot-related processes."""
        schwabot_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
                try:
                    # Check if it's a Schwabot process
                    if self._is_schwabot_process(proc):
                        schwabot_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            logger.error(f"Error finding processes: {e}")
            print(f"❌ Error finding processes: {e}")
        
        return schwabot_processes
    
    def _is_schwabot_process(self, proc):
        """Check if a process is a Schwabot process."""
        try:
            # Check process name
            if proc.info['name'] and 'schwabot' in proc.info['name'].lower():
                return True
            
            # Check command line
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline']).lower()
                if any(keyword in cmdline for keyword in [
                    'schwabot', 'trading_bot', 'schwabot_trading_bot.py',
                    'schwabot_start.py', 'schwabot_stop.py'
                ]):
                    return True
            
            # Check executable path
            if proc.info['exe']:
                exe_path = proc.info['exe'].lower()
                if 'schwabot' in exe_path:
                    return True
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return False
    
    def _stop_process(self, proc):
        """Stop a single process gracefully."""
        try:
            pid = proc.info['pid']
            name = proc.info['name'] or 'Unknown'
            cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'Unknown'
            
            print(f"🔄 Stopping process {pid} ({name})...")
            print(f"   Command: {cmdline}")
            
            # Try graceful termination
            proc.terminate()
            
            # Wait for graceful shutdown
            try:
                proc.wait(timeout=10)
                print(f"✅ Process {pid} stopped gracefully")
                self.stopped_processes.append(pid)
                logger.info(f"Process {pid} stopped gracefully")
                
            except psutil.TimeoutExpired:
                print(f"⚠️ Process {pid} did not stop gracefully, force killing...")
                self._force_kill_process(proc)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"❌ Error stopping process {proc.info.get('pid', 'Unknown')}: {e}")
            logger.error(f"Error stopping process {proc.info.get('pid', 'Unknown')}: {e}")
    
    def _force_kill_process(self, proc):
        """Force kill a process."""
        try:
            pid = proc.info['pid']
            proc.kill()
            proc.wait(timeout=5)
            print(f"✅ Process {pid} force killed")
            self.stopped_processes.append(pid)
            logger.info(f"Process {pid} force killed")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            print(f"❌ Error force killing process {proc.info.get('pid', 'Unknown')}: {e}")
            logger.error(f"Error force killing process {proc.info.get('pid', 'Unknown')}: {e}")
    
    def stop_all_python_processes(self):
        """Stop all Python processes (use with caution)."""
        try:
            print("⚠️ Stopping all Python processes (use with caution)...")
            
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        if proc.info['cmdline'] and any('schwabot' in cmd.lower() for cmd in proc.info['cmdline']):
                            python_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if python_processes:
                print(f"📋 Found {len(python_processes)} Python processes with Schwabot")
                for proc in python_processes:
                    self._stop_process(proc)
            else:
                print("ℹ️ No Python processes with Schwabot found")
                
        except Exception as e:
            logger.error(f"Error stopping Python processes: {e}")
            print(f"❌ Error stopping Python processes: {e}")
    
    def get_system_status(self):
        """Get system status information."""
        try:
            print("📊 System Status:")
            print("-" * 30)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"🖥️ CPU Usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            print(f"💾 Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            print(f"💿 Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
            
            # Network connections
            connections = len(psutil.net_connections())
            print(f"🌐 Network Connections: {connections}")
            
            print("-" * 30)
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            print(f"❌ Error getting system status: {e}")

def main():
    """Main function to stop Schwabot."""
    stopper = SchwabotStopper()
    
    # Get system status
    stopper.get_system_status()
    
    # Stop Schwabot processes
    success = stopper.stop_schwabot()
    
    # Show final status
    print("\n📊 Final Status:")
    print("=" * 30)
    if success:
        print("✅ Schwabot Trading Bot stopped successfully")
        print("🎉 All processes terminated")
    else:
        print("❌ Some issues occurred while stopping Schwabot")
        print("💡 You may need to manually check for remaining processes")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 