#!/usr/bin/env python3
"""
Schwabot Trading Bot - Modern GUI Interface
==========================================

Modern GUI interface for the Schwabot trading bot with:
- Start/Stop controls
- Real-time monitoring
- Performance metrics
- System status
- Log viewing
- Configuration management
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import subprocess
import psutil
import json
from pathlib import Path
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotGUI:
    """Modern GUI for Schwabot trading bot."""
    
    def __init__(self):
        self.root = None
        self.is_running = False
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # GUI elements
        self.status_label = None
        self.start_button = None
        self.stop_button = None
        self.restart_button = None
        self.log_text = None
        self.metrics_frame = None
        
        # Base directory
        self.base_dir = Path(__file__).parent.absolute()
        self.start_script = self.base_dir / "schwabot_start.py"
        self.stop_script = self.base_dir / "schwabot_stop.py"
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("🚀 Schwabot Trading Bot - Advanced AI Trading System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self.setup_style()
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header(main_container)
        
        # Create control panel
        self.create_control_panel(main_container)
        
        # Create status panel
        self.create_status_panel(main_container)
        
        # Create metrics panel
        self.create_metrics_panel(main_container)
        
        # Create log panel
        self.create_log_panel(main_container)
        
        # Start monitoring
        self.start_monitoring()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_style(self):
        """Setup custom style for the GUI."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='#ffffff')
        style.configure('TButton', background='#007acc', foreground='#ffffff')
        style.configure('Header.TLabel', font=('Arial', 18, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Success.TLabel', foreground='#00ff00')
        style.configure('Warning.TLabel', foreground='#ffff00')
        style.configure('Error.TLabel', foreground='#ff0000')
        style.configure('Info.TLabel', foreground='#00ffff')
    
    def create_header(self, parent):
        """Create the header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="🚀 Schwabot Trading Bot",
            style='Header.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = ttk.Label(
            header_frame,
            text="Advanced AI-Powered Trading with 47-Day Mathematical Framework",
            style='Info.TLabel'
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def create_control_panel(self, parent):
        """Create the control panel with start/stop buttons."""
        control_frame = ttk.LabelFrame(parent, text="🎮 Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="🚀 START SCHWABOT",
            command=self.start_schwabot,
            style='TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(
            button_frame,
            text="🛑 STOP SCHWABOT",
            command=self.stop_schwabot,
            style='TButton',
            state='disabled'
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Restart button
        self.restart_button = ttk.Button(
            button_frame,
            text="🔄 RESTART",
            command=self.restart_schwabot,
            style='TButton'
        )
        self.restart_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            button_frame,
            text="Status: NOT RUNNING",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT)
    
    def create_status_panel(self, parent):
        """Create the status panel."""
        status_frame = ttk.LabelFrame(parent, text="📊 System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(status_grid, text="Bot Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.bot_status_label = ttk.Label(status_grid, text="Stopped", style='Error.TLabel')
        self.bot_status_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(status_grid, text="Processes:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.process_count_label = ttk.Label(status_grid, text="0", style='Info.TLabel')
        self.process_count_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(status_grid, text="Uptime:").grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.uptime_label = ttk.Label(status_grid, text="N/A", style='Info.TLabel')
        self.uptime_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2
        ttk.Label(status_grid, text="CPU Usage:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.cpu_label = ttk.Label(status_grid, text="0%", style='Info.TLabel')
        self.cpu_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(status_grid, text="Memory:").grid(row=1, column=2, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.memory_label = ttk.Label(status_grid, text="0%", style='Info.TLabel')
        self.memory_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(status_grid, text="Last Update:").grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.last_update_label = ttk.Label(status_grid, text="Never", style='Info.TLabel')
        self.last_update_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
    
    def create_metrics_panel(self, parent):
        """Create the metrics panel."""
        self.metrics_frame = ttk.LabelFrame(parent, text="📈 Performance Metrics", padding=10)
        self.metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Metrics grid
        metrics_grid = ttk.Frame(self.metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        # Trading metrics
        ttk.Label(metrics_grid, text="Trades Executed:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.trades_label = ttk.Label(metrics_grid, text="0", style='Info.TLabel')
        self.trades_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(metrics_grid, text="Win Rate:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.winrate_label = ttk.Label(metrics_grid, text="0%", style='Info.TLabel')
        self.winrate_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(metrics_grid, text="Total P&L:").grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.pnl_label = ttk.Label(metrics_grid, text="$0.00", style='Info.TLabel')
        self.pnl_label.grid(row=0, column=5, sticky=tk.W)
        
        # Mathematical metrics
        ttk.Label(metrics_grid, text="Math Signals:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.signals_label = ttk.Label(metrics_grid, text="0", style='Info.TLabel')
        self.signals_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(metrics_grid, text="Strategy:").grid(row=1, column=2, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.strategy_label = ttk.Label(metrics_grid, text="None", style='Info.TLabel')
        self.strategy_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(metrics_grid, text="Confidence:").grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.confidence_label = ttk.Label(metrics_grid, text="0%", style='Info.TLabel')
        self.confidence_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
    
    def create_log_panel(self, parent):
        """Create the log viewing panel."""
        log_frame = ttk.LabelFrame(parent, text="📋 Live Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            log_controls,
            text="🔄 Refresh Logs",
            command=self.refresh_logs
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            log_controls,
            text="🗑️ Clear Logs",
            command=self.clear_logs
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            log_controls,
            text="💾 Save Logs",
            command=self.save_logs
        ).pack(side=tk.LEFT)
    
    def start_schwabot(self):
        """Start the Schwabot trading bot."""
        try:
            if self.is_running:
                messagebox.showwarning("Warning", "Schwabot is already running!")
                return
            
            # Check if start script exists
            if not self.start_script.exists():
                messagebox.showerror("Error", f"Start script not found: {self.start_script}")
                return
            
            # Start in a separate thread
            def start_thread():
                try:
                    subprocess.run([sys.executable, str(self.start_script)], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error starting Schwabot: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to start Schwabot: {e}"))
            
            threading.Thread(target=start_thread, daemon=True).start()
            
            # Update UI
            self.is_running = True
            self.update_ui_state()
            self.add_log_message("🚀 Schwabot started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Schwabot: {e}")
            messagebox.showerror("Error", f"Error starting Schwabot: {e}")
    
    def stop_schwabot(self):
        """Stop the Schwabot trading bot."""
        try:
            if not self.is_running:
                messagebox.showinfo("Info", "Schwabot is not running")
                return
            
            # Stop in a separate thread
            def stop_thread():
                try:
                    if self.stop_script.exists():
                        subprocess.run([sys.executable, str(self.stop_script)], check=True)
                    else:
                        # Fallback: direct process termination
                        self.stop_processes_directly()
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error stopping Schwabot: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to stop Schwabot: {e}"))
            
            threading.Thread(target=stop_thread, daemon=True).start()
            
            # Update UI
            self.is_running = False
            self.update_ui_state()
            self.add_log_message("🛑 Schwabot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Schwabot: {e}")
            messagebox.showerror("Error", f"Error stopping Schwabot: {e}")
    
    def restart_schwabot(self):
        """Restart the Schwabot trading bot."""
        try:
            self.add_log_message("🔄 Restarting Schwabot...")
            
            # Stop first
            if self.is_running:
                self.stop_schwabot()
                time.sleep(2)  # Wait for stop
            
            # Start again
            time.sleep(1)  # Brief pause
            self.start_schwabot()
            
        except Exception as e:
            logger.error(f"Error restarting Schwabot: {e}")
            messagebox.showerror("Error", f"Error restarting Schwabot: {e}")
    
    def update_ui_state(self):
        """Update the UI state based on running status."""
        if self.is_running:
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Status: RUNNING", style='Success.TLabel')
            self.bot_status_label.config(text="Running", style='Success.TLabel')
        else:
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Status: STOPPED", style='Error.TLabel')
            self.bot_status_label.config(text="Stopped", style='Error.TLabel')
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring:
            try:
                # Update status
                self.update_status()
                
                # Sleep
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def update_status(self):
        """Update the status information."""
        try:
            # Check if Schwabot is running
            processes = self.get_schwabot_processes()
            is_running = len(processes) > 0
            
            # Update running status if changed
            if is_running != self.is_running:
                self.is_running = is_running
                self.root.after(0, self.update_ui_state)
            
            # Update process count
            process_count = len(processes)
            self.root.after(0, lambda: self.process_count_label.config(text=str(process_count)))
            
            # Update system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.root.after(0, lambda: self.cpu_label.config(text=f"{cpu_percent:.1f}%"))
            self.root.after(0, lambda: self.memory_label.config(text=f"{memory.percent:.1f}%"))
            
            # Update uptime
            if processes:
                try:
                    create_time = processes[0].create_time()
                    uptime = time.time() - create_time
                    uptime_str = self.format_uptime(uptime)
                    self.root.after(0, lambda: self.uptime_label.config(text=uptime_str))
                except:
                    pass
            
            # Update last update time
            current_time = datetime.now().strftime("%H:%M:%S")
            self.root.after(0, lambda: self.last_update_label.config(text=current_time))
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def get_schwabot_processes(self):
        """Get all Schwabot processes."""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if self.is_schwabot_process(proc):
                        processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error getting processes: {e}")
        
        return processes
    
    def is_schwabot_process(self, proc):
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
    
    def stop_processes_directly(self):
        """Stop Schwabot processes directly."""
        processes = self.get_schwabot_processes()
        
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except:
                    pass
    
    def format_uptime(self, seconds):
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
    
    def add_log_message(self, message):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.log_text.see(tk.END))
    
    def refresh_logs(self):
        """Refresh the log display."""
        try:
            self.log_text.delete(1.0, tk.END)
            self.add_log_message("📋 Logs refreshed")
        except Exception as e:
            logger.error(f"Error refreshing logs: {e}")
    
    def clear_logs(self):
        """Clear the log display."""
        try:
            self.log_text.delete(1.0, tk.END)
            self.add_log_message("🗑️ Logs cleared")
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")
    
    def save_logs(self):
        """Save the current logs to a file."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                
                self.add_log_message(f"💾 Logs saved to {filename}")
                messagebox.showinfo("Success", f"Logs saved to {filename}")
                
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
            messagebox.showerror("Error", f"Error saving logs: {e}")
    
    def on_closing(self):
        """Handle window closing."""
        try:
            self.stop_monitoring = True
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1)
            
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error closing GUI: {e}")
    
    def run(self):
        """Run the GUI."""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error running GUI: {e}")

def main():
    """Main function to run the GUI."""
    try:
        gui = SchwabotGUI()
        gui.run()
    except Exception as e:
        logger.error(f"Error starting GUI: {e}")
        print(f"❌ Error starting GUI: {e}")

if __name__ == "__main__":
    main() 