#!/usr/bin/env python3
"""
Schwabot Trading Bot - CLI Control
=================================

Command-line interface for controlling the Schwabot trading bot.
Provides start, stop, status, and other management commands.

Usage:
    python schwabot_cli.py start
    python schwabot_cli.py stop
    python schwabot_cli.py status
    python schwabot_cli.py restart
    python schwabot_cli.py logs
"""

import sys
import os
import argparse
import subprocess
import psutil
import time
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotCLI:
    """Schwabot CLI control class."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.start_script = self.base_dir / "schwabot_start.py"
        self.stop_script = self.base_dir / "schwabot_stop.py"
        self.main_bot = self.base_dir / "schwabot_trading_bot.py"
        
    def start(self):
        """Start the Schwabot trading bot."""
        try:
            print("🚀 Starting Schwabot Trading Bot...")
            
            # Check if already running
            if self.is_running():
                print("⚠️ Schwabot is already running!")
                return False
            
            # Check if start script exists
            if not self.start_script.exists():
                print(f"❌ Start script not found: {self.start_script}")
                return False
            
            # Start the bot
            print("📊 Launching Schwabot with 47-Day Mathematical Framework...")
            subprocess.run([sys.executable, str(self.start_script)])
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting Schwabot: {e}")
            print(f"❌ Error starting Schwabot: {e}")
            return False
    
    def stop(self):
        """Stop the Schwabot trading bot."""
        try:
            print("🛑 Stopping Schwabot Trading Bot...")
            
            # Check if running
            if not self.is_running():
                print("ℹ️ Schwabot is not running")
                return True
            
            # Stop the bot
            if self.stop_script.exists():
                subprocess.run([sys.executable, str(self.stop_script)])
            else:
                # Fallback: direct process termination
                self._stop_processes_directly()
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Schwabot: {e}")
            print(f"❌ Error stopping Schwabot: {e}")
            return False
    
    def restart(self):
        """Restart the Schwabot trading bot."""
        try:
            print("🔄 Restarting Schwabot Trading Bot...")
            
            # Stop first
            if self.is_running():
                self.stop()
                time.sleep(2)  # Wait for processes to terminate
            
            # Start again
            time.sleep(1)  # Brief pause
            return self.start()
            
        except Exception as e:
            logger.error(f"Error restarting Schwabot: {e}")
            print(f"❌ Error restarting Schwabot: {e}")
            return False
    
    def status(self):
        """Show status of Schwabot processes."""
        try:
            print("📊 Schwabot Trading Bot Status")
            print("=" * 50)
            
            # Check if running
            if self.is_running():
                print("✅ Status: RUNNING")
                processes = self._get_schwabot_processes()
                print(f"📋 Active Processes: {len(processes)}")
                
                for i, proc in enumerate(processes, 1):
                    try:
                        pid = proc.info['pid']
                        name = proc.info['name'] or 'Unknown'
                        cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'Unknown'
                        cpu_percent = proc.cpu_percent()
                        memory_info = proc.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        
                        print(f"  {i}. PID: {pid} | {name}")
                        print(f"     Command: {cmdline}")
                        print(f"     CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f}MB")
                        print()
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Show uptime
                if processes:
                    try:
                        create_time = processes[0].create_time()
                        uptime = time.time() - create_time
                        uptime_str = self._format_uptime(uptime)
                        print(f"⏰ Uptime: {uptime_str}")
                    except:
                        pass
                
            else:
                print("❌ Status: NOT RUNNING")
                print("💡 Use 'start' command to launch Schwabot")
            
            # Show system info
            self._show_system_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            print(f"❌ Error getting status: {e}")
            return False
    
    def logs(self, lines=50):
        """Show recent logs."""
        try:
            print("📋 Recent Schwabot Logs")
            print("=" * 50)
            
            log_files = [
                "schwabot_trading_bot.log",
                "schwabot_start.log",
                "schwabot_stop.log",
                "schwabot_cli.log",
                "schwabot_monitoring.log"
            ]
            
            for log_file in log_files:
                log_path = self.base_dir / log_file
                if log_path.exists():
                    print(f"\n📄 {log_file}:")
                    print("-" * 30)
                    
                    try:
                        with open(log_path, 'r') as f:
                            lines_content = f.readlines()
                            recent_lines = lines_content[-lines:] if len(lines_content) > lines else lines_content
                            for line in recent_lines:
                                print(line.rstrip())
                    except Exception as e:
                        print(f"❌ Error reading {log_file}: {e}")
                else:
                    print(f"ℹ️ {log_file}: Not found")
            
            return True
            
        except Exception as e:
            logger.error(f"Error showing logs: {e}")
            print(f"❌ Error showing logs: {e}")
            return False
    
    def is_running(self):
        """Check if Schwabot is running."""
        return len(self._get_schwabot_processes()) > 0
    
    def _get_schwabot_processes(self):
        """Get all Schwabot processes."""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if self._is_schwabot_process(proc):
                        processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error getting processes: {e}")
        
        return processes
    
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
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return False
    
    def _stop_processes_directly(self):
        """Stop Schwabot processes directly."""
        processes = self._get_schwabot_processes()
        
        for proc in processes:
            try:
                print(f"🔄 Stopping process {proc.info['pid']}...")
                proc.terminate()
                proc.wait(timeout=10)
                print(f"✅ Process {proc.info['pid']} stopped")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                print(f"⚠️ Error stopping process {proc.info.get('pid', 'Unknown')}: {e}")
    
    def _show_system_info(self):
        """Show system information."""
        try:
            print("\n🖥️ System Information:")
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
            
            # Python version
            print(f"🐍 Python Version: {sys.version.split()[0]}")
            
            # Working directory
            print(f"📁 Working Directory: {self.base_dir}")
            
        except Exception as e:
            logger.error(f"Error showing system info: {e}")
    
    def _format_uptime(self, seconds):
        """Format uptime in human readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Schwabot Trading Bot CLI Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schwabot_cli.py start     # Start Schwabot
  python schwabot_cli.py stop      # Stop Schwabot
  python schwabot_cli.py restart   # Restart Schwabot
  python schwabot_cli.py status    # Show status
  python schwabot_cli.py logs      # Show recent logs
  python schwabot_cli.py logs 100  # Show last 100 log lines
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'restart', 'status', 'logs'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--lines', '-n',
        type=int,
        default=50,
        help='Number of log lines to show (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = SchwabotCLI()
    
    # Execute command
    if args.command == 'start':
        success = cli.start()
    elif args.command == 'stop':
        success = cli.stop()
    elif args.command == 'restart':
        success = cli.restart()
    elif args.command == 'status':
        success = cli.status()
    elif args.command == 'logs':
        success = cli.logs(args.lines)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 