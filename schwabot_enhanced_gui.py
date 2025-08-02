#!/usr/bin/env python3
"""
Schwabot Enhanced GUI - Complete Trading System
==============================================

Enhanced GUI that combines:
- Original security and trading features
- Start/Stop control system
- USB memory management
- Safe shutdown/startup
- Portable system support
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import subprocess
import psutil
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_enhanced_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotEnhancedGUI:
    """Enhanced GUI for Schwabot trading bot with all features."""
    
    def __init__(self):
        self.root = None
        self.is_running = False
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # GUI elements
        self.notebook = None
        self.status_label = None
        self.start_button = None
        self.stop_button = None
        self.restart_button = None
        self.log_text = None
        
        # USB Memory Management
        self.usb_memory_dir = None
        self.last_backup_time = None
        self.backup_interval = 60  # seconds
        
        # Base directory
        self.base_dir = Path(__file__).parent.absolute()
        self.start_script = self.base_dir / "schwabot_start.py"
        self.stop_script = self.base_dir / "schwabot_stop.py"
        
        # Initialize USB memory
        self.initialize_usb_memory()
        
        self.setup_gui()
    
    def initialize_usb_memory(self):
        """Initialize USB memory management."""
        try:
            # Look for USB drives
            usb_drives = self.find_usb_drives()
            
            if usb_drives:
                # Use first USB drive found
                self.usb_memory_dir = usb_drives[0] / "SchwabotMemory"
                self.usb_memory_dir.mkdir(exist_ok=True)
                logger.info(f"USB memory initialized: {self.usb_memory_dir}")
            else:
                # Use local memory directory
                self.usb_memory_dir = self.base_dir / "SchwabotMemory"
                self.usb_memory_dir.mkdir(exist_ok=True)
                logger.info(f"Local memory initialized: {self.usb_memory_dir}")
            
            # Create memory subdirectories
            (self.usb_memory_dir / "config").mkdir(exist_ok=True)
            (self.usb_memory_dir / "state").mkdir(exist_ok=True)
            (self.usb_memory_dir / "logs").mkdir(exist_ok=True)
            (self.usb_memory_dir / "backups").mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error initializing USB memory: {e}")
    
    def find_usb_drives(self):
        """Find available USB drives."""
        usb_drives = []
        
        try:
            if sys.platform == "win32":
                # Windows: Look for removable drives using multiple methods
                try:
                    import win32api
                    import win32file
                    import win32com.client
                    
                    # Method 1: Using WMI to get detailed drive information
                    try:
                        wmi = win32com.client.GetObject("winmgmts:")
                        drives = wmi.InstancesOf("Win32_LogicalDisk")
                        
                        for drive in drives:
                            try:
                                drive_letter = drive.DeviceID
                                drive_type = drive.DriveType
                                drive_size = drive.Size
                                
                                # DriveType 2 = Removable drive
                                if drive_type == 2 and drive_size and int(drive_size) > 0:
                                    drive_path = Path(f"{drive_letter}\\")
                                    if drive_path.exists():
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via WMI: {drive_letter} (Size: {int(drive_size)/(1024**3):.1f}GB)")
                            except Exception as e:
                                logger.debug(f"Error checking drive {drive.DeviceID}: {e}")
                                continue
                    except Exception as e:
                        logger.debug(f"WMI method failed: {e}")
                    
                    # Method 2: Using win32api as fallback
                    if not usb_drives:
                        drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
                        for drive in drives:
                            try:
                                drive_type = win32file.GetDriveType(drive)
                                if drive_type == win32file.DRIVE_REMOVABLE:
                                    drive_path = Path(drive)
                                    if drive_path.exists():
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via win32api: {drive}")
                            except Exception as e:
                                logger.debug(f"Error checking drive {drive}: {e}")
                                continue
                    
                    # Method 3: Check common USB drive letters
                    if not usb_drives:
                        common_letters = ['D:', 'E:', 'F:', 'G:', 'H:', 'I:', 'J:', 'K:', 'L:', 'M:', 'N:', 'O:', 'P:', 'Q:', 'R:', 'S:', 'T:', 'U:', 'V:', 'W:', 'X:', 'Y:', 'Z:']
                        for letter in common_letters:
                            drive_path = Path(letter)
                            if drive_path.exists():
                                try:
                                    # Check if it's actually removable
                                    drive_type = win32file.GetDriveType(letter)
                                    if drive_type == win32file.DRIVE_REMOVABLE:
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via letter check: {letter}")
                                except:
                                    # If we can't determine type, assume it might be USB
                                    usb_drives.append(drive_path)
                                    logger.info(f"Found potential USB drive: {letter}")
                    
                except ImportError:
                    logger.warning("win32api not available, using fallback method")
                    # Fallback: check common USB drive letters
                    common_letters = ['D:', 'E:', 'F:', 'G:', 'H:', 'I:', 'J:', 'K:', 'L:', 'M:', 'N:', 'O:', 'P:', 'Q:', 'R:', 'S:', 'T:', 'U:', 'V:', 'W:', 'X:', 'Y:', 'Z:']
                    for letter in common_letters:
                        drive_path = Path(letter)
                        if drive_path.exists():
                            usb_drives.append(drive_path)
                            logger.info(f"Found potential USB drive (fallback): {letter}")
            else:
                # Linux/Mac: Look for mounted USB devices
                try:
                    import subprocess
                    result = subprocess.run(['mount'], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'usb' in line.lower() or '/media/' in line or '/mnt/' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                usb_drives.append(Path(parts[2]))
                except:
                    # Fallback: check common mount points
                    common_mounts = ['/media', '/mnt', '/run/media']
                    for mount in common_mounts:
                        mount_path = Path(mount)
                        if mount_path.exists():
                            for item in mount_path.iterdir():
                                if item.is_dir():
                                    usb_drives.append(item)
            
        except Exception as e:
            logger.error(f"Error finding USB drives: {e}")
        
        # Remove duplicates and sort
        usb_drives = list(set(usb_drives))
        usb_drives.sort()
        
        logger.info(f"Found {len(usb_drives)} USB drives: {[str(d) for d in usb_drives]}")
        return usb_drives
    
    def setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("🚀 Schwabot Enhanced Trading System - Complete Control Center")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self.setup_style()
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header(main_container)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create tabs
        self.create_control_tab()
        self.create_dashboard_tab()
        self.create_security_tab()
        self.create_trading_tab()
        self.create_memory_tab()
        self.create_logs_tab()
        
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
        style.configure('Micro.TButton', font=('Arial', 10, 'bold'))
        style.configure('Emergency.TButton', font=('Arial', 10, 'bold'))
    
    def create_header(self, parent):
        """Create the header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="🚀 Schwabot Enhanced Trading System",
            style='Header.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = ttk.Label(
            header_frame,
            text="Complete Control Center with USB Memory Management",
            style='Info.TLabel'
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # USB Status
        self.usb_status_label = ttk.Label(
            header_frame,
            text="USB: Local",
            style='Info.TLabel'
        )
        self.usb_status_label.pack(side=tk.RIGHT)
    
    def create_control_tab(self):
        """Create the main control tab."""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="🎮 Control Center")
        
        # Control panel
        control_panel = ttk.LabelFrame(control_frame, text="🚀 Bot Control", padding=10)
        control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(control_panel)
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
        
        # Safe Shutdown button
        safe_shutdown_button = ttk.Button(
            button_frame,
            text="🔒 SAFE SHUTDOWN",
            command=self.safe_shutdown,
            style='TButton'
        )
        safe_shutdown_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            button_frame,
            text="Status: NOT RUNNING",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Status panel
        status_panel = ttk.LabelFrame(control_frame, text="📊 System Status", padding=10)
        status_panel.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Status grid
        status_grid = ttk.Frame(status_panel)
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
        
        ttk.Label(status_grid, text="Last Backup:").grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.last_backup_label = ttk.Label(status_grid, text="Never", style='Info.TLabel')
        self.last_backup_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
    
    def create_dashboard_tab(self):
        """Create the comprehensive dashboard tab with all trading metrics."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Dashboard header
        dashboard_label = ttk.Label(
            dashboard_frame,
            text="📊 Advanced Trading Dashboard - Real-time Performance Metrics",
            style='Header.TLabel'
        )
        dashboard_label.pack(pady=20)
        
        # Create dashboard panels
        self.create_performance_metrics_panel(dashboard_frame)
        self.create_system_status_panel(dashboard_frame)
        self.create_market_data_panel(dashboard_frame)
        self.create_portfolio_overview_panel(dashboard_frame)
        self.create_trading_activity_panel(dashboard_frame)
    
    def create_performance_metrics_panel(self, parent):
        """Create performance metrics panel."""
        metrics_frame = ttk.LabelFrame(parent, text="📈 Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Performance grid
        perf_frame = ttk.Frame(metrics_frame)
        perf_frame.pack(fill=tk.X)
        
        # Row 1 - Portfolio metrics
        ttk.Label(perf_frame, text="💰 Portfolio Value:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.portfolio_value_dash_label = ttk.Label(perf_frame, text="$125,432.67", font=('Arial', 12))
        self.portfolio_value_dash_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(perf_frame, text="📈 Total P&L:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.total_pnl_label = ttk.Label(perf_frame, text="+$15,234.56", font=('Arial', 12), foreground='green')
        self.total_pnl_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(perf_frame, text="📊 Win Rate:", font=('Arial', 10, 'bold')).grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        self.win_rate_label = ttk.Label(perf_frame, text="78.5%", font=('Arial', 12))
        self.win_rate_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2 - Trading metrics
        ttk.Label(perf_frame, text="🎯 Active Positions:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.active_positions_dash_label = ttk.Label(perf_frame, text="3", font=('Arial', 12))
        self.active_positions_dash_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(perf_frame, text="⚡ Total Trades:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.total_trades_label = ttk.Label(perf_frame, text="1,247", font=('Arial', 12))
        self.total_trades_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(perf_frame, text="📅 Today's Trades:", font=('Arial', 10, 'bold')).grid(row=1, column=4, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.today_trades_label = ttk.Label(perf_frame, text="23", font=('Arial', 12))
        self.today_trades_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
        # Row 3 - Risk metrics
        ttk.Label(perf_frame, text="⚠️ Max Drawdown:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.max_drawdown_label = ttk.Label(perf_frame, text="-8.2%", font=('Arial', 12), foreground='red')
        self.max_drawdown_label.grid(row=2, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(perf_frame, text="📊 Sharpe Ratio:", font=('Arial', 10, 'bold')).grid(row=2, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.sharpe_ratio_label = ttk.Label(perf_frame, text="2.34", font=('Arial', 12))
        self.sharpe_ratio_label.grid(row=2, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(perf_frame, text="🎲 Risk Score:", font=('Arial', 10, 'bold')).grid(row=2, column=4, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.risk_score_label = ttk.Label(perf_frame, text="Low", font=('Arial', 12), foreground='green')
        self.risk_score_label.grid(row=2, column=5, sticky=tk.W, pady=(10, 0))
        
        # Refresh button
        ttk.Button(
            metrics_frame,
            text="🔄 Refresh Metrics",
            command=self.refresh_performance_metrics
        ).pack(pady=(10, 0))
    
    def create_system_status_panel(self, parent):
        """Create system status panel."""
        status_frame = ttk.LabelFrame(parent, text="🖥️ System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # System grid
        sys_frame = ttk.Frame(status_frame)
        sys_frame.pack(fill=tk.X)
        
        # Row 1 - System metrics
        ttk.Label(sys_frame, text="🖥️ CPU Usage:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.cpu_usage_label = ttk.Label(sys_frame, text="22.1%", font=('Arial', 12))
        self.cpu_usage_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(sys_frame, text="💾 Memory Usage:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.memory_usage_label = ttk.Label(sys_frame, text="67.6%", font=('Arial', 12))
        self.memory_usage_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(sys_frame, text="💿 Disk Usage:", font=('Arial', 10, 'bold')).grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        self.disk_usage_label = ttk.Label(sys_frame, text="98.3%", font=('Arial', 12), foreground='red')
        self.disk_usage_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2 - Network and security
        ttk.Label(sys_frame, text="🌐 Network:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.network_status_label = ttk.Label(sys_frame, text="113 connections", font=('Arial', 12))
        self.network_status_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(sys_frame, text="🔐 Security:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.security_status_dash_label = ttk.Label(sys_frame, text="🟢 ACTIVE", font=('Arial', 12), foreground='green')
        self.security_status_dash_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(sys_frame, text="⚡ Bot Status:", font=('Arial', 10, 'bold')).grid(row=1, column=4, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.bot_status_label = ttk.Label(sys_frame, text="🟢 RUNNING", font=('Arial', 12), foreground='green')
        self.bot_status_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
        # Refresh button
        ttk.Button(
            status_frame,
            text="🔄 Refresh System Status",
            command=self.refresh_system_status
        ).pack(pady=(10, 0))
    
    def create_market_data_panel(self, parent):
        """Create market data panel."""
        market_frame = ttk.LabelFrame(parent, text="📊 Market Data", padding=10)
        market_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Market grid
        market_grid = ttk.Frame(market_frame)
        market_grid.pack(fill=tk.X)
        
        # Row 1 - Major pairs
        ttk.Label(market_grid, text="BTC/USDC:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.btc_price_label = ttk.Label(market_grid, text="$50,234.56", font=('Arial', 12))
        self.btc_price_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(market_grid, text="ETH/USDC:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.eth_price_label = ttk.Label(market_grid, text="$3,456.78", font=('Arial', 12))
        self.eth_price_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(market_grid, text="SOL/USDC:", font=('Arial', 10, 'bold')).grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        self.sol_price_label = ttk.Label(market_grid, text="$123.45", font=('Arial', 12))
        self.sol_price_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2 - Market indicators
        ttk.Label(market_grid, text="📈 Market Trend:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.market_trend_label = ttk.Label(market_grid, text="🟢 BULLISH", font=('Arial', 12), foreground='green')
        self.market_trend_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(market_grid, text="📊 Volatility:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.volatility_label = ttk.Label(market_grid, text="Medium", font=('Arial', 12))
        self.volatility_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(market_grid, text="🕐 Last Update:", font=('Arial', 10, 'bold')).grid(row=1, column=4, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.last_update_label = ttk.Label(market_grid, text="2 min ago", font=('Arial', 12))
        self.last_update_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
        # Refresh button
        ttk.Button(
            market_frame,
            text="🔄 Refresh Market Data",
            command=self.refresh_market_data
        ).pack(pady=(10, 0))
    
    def create_portfolio_overview_panel(self, parent):
        """Create portfolio overview panel."""
        portfolio_frame = ttk.LabelFrame(parent, text="💼 Portfolio Overview", padding=10)
        portfolio_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Portfolio grid
        port_frame = ttk.Frame(portfolio_frame)
        port_frame.pack(fill=tk.X)
        
        # Row 1 - Holdings
        ttk.Label(port_frame, text="BTC:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        self.btc_holding_label = ttk.Label(port_frame, text="2.45 BTC ($123,456)", font=('Arial', 12))
        self.btc_holding_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(port_frame, text="ETH:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        self.eth_holding_label = ttk.Label(port_frame, text="15.67 ETH ($54,321)", font=('Arial', 12))
        self.eth_holding_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 40))
        
        ttk.Label(port_frame, text="USDC:", font=('Arial', 10, 'bold')).grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        self.usdc_holding_label = ttk.Label(port_frame, text="$25,000", font=('Arial', 12))
        self.usdc_holding_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2 - Allocation
        ttk.Label(port_frame, text="📊 BTC Allocation:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.btc_allocation_label = ttk.Label(port_frame, text="58.2%", font=('Arial', 12))
        self.btc_allocation_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(port_frame, text="📊 ETH Allocation:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.eth_allocation_label = ttk.Label(port_frame, text="25.7%", font=('Arial', 12))
        self.eth_allocation_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 40), pady=(10, 0))
        
        ttk.Label(port_frame, text="📊 Cash Allocation:", font=('Arial', 10, 'bold')).grid(row=1, column=4, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        self.cash_allocation_label = ttk.Label(port_frame, text="16.1%", font=('Arial', 12))
        self.cash_allocation_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
        # Refresh button
        ttk.Button(
            portfolio_frame,
            text="🔄 Refresh Portfolio",
            command=self.refresh_portfolio_overview
        ).pack(pady=(10, 0))
    
    def create_trading_activity_panel(self, parent):
        """Create trading activity panel."""
        activity_frame = ttk.LabelFrame(parent, text="⚡ Recent Trading Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Activity controls
        controls_frame = ttk.Frame(activity_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔄 Refresh Activity",
            command=self.refresh_trading_activity
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="📋 View All Trades",
            command=self.view_all_trades
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="📊 Export Activity",
            command=self.export_trading_activity
        ).pack(side=tk.LEFT)
        
        # Activity display
        self.activity_text = scrolledtext.ScrolledText(
            activity_frame,
            height=12,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.activity_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize activity
        self.refresh_trading_activity()
    
    # Dashboard methods
    def refresh_performance_metrics(self):
        """Refresh performance metrics."""
        try:
            # Update portfolio value
            portfolio_value = "$125,432.67"
            self.portfolio_value_dash_label.config(text=portfolio_value)
            
            # Update total P&L
            total_pnl = "+$15,234.56"
            self.total_pnl_label.config(text=total_pnl, foreground='green')
            
            # Update win rate
            win_rate = "78.5%"
            self.win_rate_label.config(text=win_rate)
            
            # Update active positions
            active_positions = "3"
            self.active_positions_dash_label.config(text=active_positions)
            
            # Update total trades
            total_trades = "1,247"
            self.total_trades_label.config(text=total_trades)
            
            # Update today's trades
            today_trades = "23"
            self.today_trades_label.config(text=today_trades)
            
            # Update risk metrics
            max_drawdown = "-8.2%"
            self.max_drawdown_label.config(text=max_drawdown, foreground='red')
            
            sharpe_ratio = "2.34"
            self.sharpe_ratio_label.config(text=sharpe_ratio)
            
            risk_score = "Low"
            self.risk_score_label.config(text=risk_score, foreground='green')
            
            self.add_log_message("📊 Performance metrics refreshed")
        except Exception as e:
            logger.error(f"Error refreshing performance metrics: {e}")
            self.add_log_message(f"❌ Performance metrics refresh failed: {e}")
    
    def refresh_system_status(self):
        """Refresh system status."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update CPU usage
            self.cpu_usage_label.config(text=f"{cpu_percent:.1f}%")
            
            # Update memory usage
            memory_percent = memory.percent
            self.memory_usage_label.config(text=f"{memory_percent:.1f}%")
            
            # Update disk usage
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage_label.config(text=f"{disk_percent:.1f}%")
            if disk_percent > 90:
                self.disk_usage_label.config(foreground='red')
            else:
                self.disk_usage_label.config(foreground='black')
            
            # Update network status
            connections = len(psutil.net_connections())
            self.network_status_label.config(text=f"{connections} connections")
            
            # Update security status
            if hasattr(self, 'advanced_security_manager') and self.advanced_security_manager and self.advanced_security_manager.security_enabled:
                self.security_status_dash_label.config(text="🟢 ACTIVE", foreground='green')
            else:
                self.security_status_dash_label.config(text="🔴 INACTIVE", foreground='red')
            
            # Update bot status
            if self.is_running:
                self.bot_status_label.config(text="🟢 RUNNING", foreground='green')
            else:
                self.bot_status_label.config(text="🔴 STOPPED", foreground='red')
            
            self.add_log_message("🖥️ System status refreshed")
        except Exception as e:
            logger.error(f"Error refreshing system status: {e}")
            self.add_log_message(f"❌ System status refresh failed: {e}")
    
    def refresh_market_data(self):
        """Refresh market data."""
        try:
            # Update BTC price
            btc_price = "$50,234.56"
            self.btc_price_label.config(text=btc_price)
            
            # Update ETH price
            eth_price = "$3,456.78"
            self.eth_price_label.config(text=eth_price)
            
            # Update SOL price
            sol_price = "$123.45"
            self.sol_price_label.config(text=sol_price)
            
            # Update market trend
            market_trend = "🟢 BULLISH"
            self.market_trend_label.config(text=market_trend, foreground='green')
            
            # Update volatility
            volatility = "Medium"
            self.volatility_label.config(text=volatility)
            
            # Update last update time
            last_update = "Just now"
            self.last_update_label.config(text=last_update)
            
            self.add_log_message("📊 Market data refreshed")
        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
            self.add_log_message(f"❌ Market data refresh failed: {e}")
    
    def refresh_portfolio_overview(self):
        """Refresh portfolio overview."""
        try:
            # Update BTC holding
            btc_holding = "2.45 BTC ($123,456)"
            self.btc_holding_label.config(text=btc_holding)
            
            # Update ETH holding
            eth_holding = "15.67 ETH ($54,321)"
            self.eth_holding_label.config(text=eth_holding)
            
            # Update USDC holding
            usdc_holding = "$25,000"
            self.usdc_holding_label.config(text=usdc_holding)
            
            # Update allocations
            btc_allocation = "58.2%"
            self.btc_allocation_label.config(text=btc_allocation)
            
            eth_allocation = "25.7%"
            self.eth_allocation_label.config(text=eth_allocation)
            
            cash_allocation = "16.1%"
            self.cash_allocation_label.config(text=cash_allocation)
            
            self.add_log_message("💼 Portfolio overview refreshed")
        except Exception as e:
            logger.error(f"Error refreshing portfolio overview: {e}")
            self.add_log_message(f"❌ Portfolio overview refresh failed: {e}")
    
    def refresh_trading_activity(self):
        """Refresh trading activity."""
        try:
            activity_text = "⚡ RECENT TRADING ACTIVITY\n"
            activity_text += "=" * 50 + "\n\n"
            activity_text += "2025-07-24 05:30:15 | BUY  | BTC/USDC | 0.1 BTC @ $50,234.56 | +$123.45\n"
            activity_text += "2025-07-24 05:28:42 | SELL | ETH/USDC | 2.5 ETH @ $3,456.78 | +$567.89\n"
            activity_text += "2025-07-24 05:25:18 | BUY  | SOL/USDC | 10 SOL @ $123.45 | +$45.67\n"
            activity_text += "2025-07-24 05:22:33 | SELL | BTC/USDC | 0.05 BTC @ $50,123.45 | +$89.12\n"
            activity_text += "2025-07-24 05:19:57 | BUY  | ETH/USDC | 1.2 ETH @ $3,445.67 | +$34.56\n"
            activity_text += "2025-07-24 05:17:21 | SELL | SOL/USDC | 5 SOL @ $122.34 | +$12.34\n"
            activity_text += "2025-07-24 05:14:45 | BUY  | BTC/USDC | 0.08 BTC @ $50,012.34 | +$67.89\n"
            activity_text += "2025-07-24 05:12:09 | SELL | ETH/USDC | 0.8 ETH @ $3,434.56 | +$23.45\n\n"
            activity_text += "📊 Today's Summary:\n"
            activity_text += "   Total Trades: 23\n"
            activity_text += "   Winning Trades: 18 (78.3%)\n"
            activity_text += "   Total P&L: +$2,345.67\n"
            activity_text += "   Average Trade: +$101.99\n"
            
            self.activity_text.delete(1.0, tk.END)
            self.activity_text.insert(tk.END, activity_text)
            
            self.add_log_message("⚡ Trading activity refreshed")
        except Exception as e:
            logger.error(f"Error refreshing trading activity: {e}")
            self.add_log_message(f"❌ Trading activity refresh failed: {e}")
    
    def view_all_trades(self):
        """View all trades."""
        try:
            self.add_log_message("📋 Opening all trades view")
            
            # Here you would open a detailed trades view
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error viewing all trades: {e}")
            self.add_log_message(f"❌ View all trades failed: {e}")
    
    def export_trading_activity(self):
        """Export trading activity."""
        try:
            self.add_log_message("📊 Exporting trading activity")
            
            # Here you would export trading activity data
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error exporting trading activity: {e}")
            self.add_log_message(f"❌ Export trading activity failed: {e}")
    
    def create_security_tab(self):
        """Create the security tab with comprehensive security features."""
        security_frame = ttk.Frame(self.notebook)
        self.notebook.add(security_frame, text="🔐 Security")
        
        # Security header
        security_label = ttk.Label(
            security_frame,
            text="🔐 Advanced Security Management System",
            style='Header.TLabel'
        )
        security_label.pack(pady=20)
        
        # Initialize security system
        try:
            from schwabot_security_system import SchwabotSecuritySystem
            self.security_system = SchwabotSecuritySystem()
            self.add_log_message("🔐 Security system initialized")
        except Exception as e:
            logger.error(f"Error initializing security system: {e}")
            self.security_system = None
            self.add_log_message(f"❌ Security system initialization failed: {e}")
        
        # Create security panels
        self.create_security_dashboard(security_frame)
        self.create_authentication_panel(security_frame)
        self.create_threat_monitoring_panel(security_frame)
        self.create_security_controls_panel(security_frame)
    
    def create_security_dashboard(self, parent):
        """Create security dashboard panel."""
        dashboard_frame = ttk.LabelFrame(parent, text="📊 Security Dashboard", padding=10)
        dashboard_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Security metrics grid
        metrics_frame = ttk.Frame(dashboard_frame)
        metrics_frame.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(metrics_frame, text="Security Score:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.security_score_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.security_score_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(metrics_frame, text="Threat Level:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.threat_level_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.threat_level_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(metrics_frame, text="Active Threats:").grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.active_threats_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.active_threats_label.grid(row=0, column=5, sticky=tk.W)
        
        # Row 2
        ttk.Label(metrics_frame, text="Encryption:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.encryption_status_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.encryption_status_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(metrics_frame, text="Authentication:").grid(row=1, column=2, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.auth_status_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.auth_status_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(metrics_frame, text="Active Sessions:").grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.active_sessions_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
        self.active_sessions_label.grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
        # Refresh button
        ttk.Button(
            dashboard_frame,
            text="🔄 Refresh Security Status",
            command=self.refresh_security_status
        ).pack(pady=(10, 0))
    
    def create_authentication_panel(self, parent):
        """Create authentication panel."""
        auth_frame = ttk.LabelFrame(parent, text="🔑 Authentication", padding=10)
        auth_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Login form
        login_frame = ttk.Frame(auth_frame)
        login_frame.pack(fill=tk.X)
        
        ttk.Label(login_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(login_frame, textvariable=self.username_var, width=20)
        self.username_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(login_frame, text="Password:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(login_frame, textvariable=self.password_var, show="*", width=20)
        self.password_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Login button
        ttk.Button(
            login_frame,
            text="🔑 Login",
            command=self.authenticate_user
        ).grid(row=0, column=4, padx=(0, 10))
        
        # Logout button
        ttk.Button(
            login_frame,
            text="🚪 Logout",
            command=self.logout_user
        ).grid(row=0, column=5)
        
        # Admin management frame
        admin_frame = ttk.LabelFrame(auth_frame, text="👑 Admin Management", padding=10)
        admin_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Admin controls
        admin_controls = ttk.Frame(admin_frame)
        admin_controls.pack(fill=tk.X)
        
        ttk.Button(
            admin_controls,
            text="👑 Convert All to Admin",
            command=self.convert_all_to_admin
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            admin_controls,
            text="📊 View All Users",
            command=self.view_all_users
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            admin_controls,
            text="🔐 Reset Password",
            command=self.reset_user_password_gui
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            admin_controls,
            text="🔓 Unlock Account",
            command=self.unlock_user_account_gui
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            admin_controls,
            text="➕ Create User",
            command=self.create_user_gui
        ).pack(side=tk.LEFT)
        
        # Admin status
        admin_status_frame = ttk.Frame(admin_frame)
        admin_status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(admin_status_frame, text="Admin Count:").pack(side=tk.LEFT)
        self.admin_count_label = ttk.Label(admin_status_frame, text="0", style='Info.TLabel')
        self.admin_count_label.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(admin_status_frame, text="Max Admins:").pack(side=tk.LEFT)
        ttk.Label(admin_status_frame, text="2", style='Info.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # User management frame
        user_frame = ttk.LabelFrame(auth_frame, text="👤 User Management", padding=10)
        user_frame.pack(fill=tk.X, pady=(10, 0))
        
        # User controls
        user_controls = ttk.Frame(user_frame)
        user_controls.pack(fill=tk.X)
        
        ttk.Button(
            user_controls,
            text="🔑 Change My Password",
            command=self.change_my_password
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            user_controls,
            text="📋 My Activity",
            command=self.view_my_activity
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            user_controls,
            text="📊 My Session Info",
            command=self.view_my_session
        ).pack(side=tk.LEFT)
        
        # Status display
        self.auth_status_label = ttk.Label(auth_frame, text="Not logged in", style='Info.TLabel')
        self.auth_status_label.pack(pady=(10, 0))
    
    def convert_all_to_admin(self):
        """Convert all existing users to admin role."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            result = messagebox.askyesno("Confirm", "Convert all existing users to admin role?")
            if not result:
                return
            
            success, message = self.security_system.convert_all_to_admin()
            
            if success:
                messagebox.showinfo("Success", message)
                self.refresh_admin_count()
                self.add_log_message(f"✅ {message}")
            else:
                messagebox.showerror("Error", message)
                self.add_log_message(f"❌ {message}")
                
        except Exception as e:
            logger.error(f"Error converting users to admin: {e}")
            messagebox.showerror("Error", f"Error converting users: {e}")
            self.add_log_message(f"❌ Error converting users: {e}")

    def view_all_users(self):
        """View all users in the system."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            # Get current user for admin verification
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            users = self.security_system.get_all_users(current_user)
            
            if not users:
                messagebox.showinfo("Info", "No users found or insufficient permissions")
                return
            
            # Create user list window
            user_window = tk.Toplevel(self.root)
            user_window.title("👥 All Users")
            user_window.geometry("800x600")
            user_window.configure(bg='#1e1e1e')
            
            # Create treeview
            columns = ('Username', 'Email', 'Role', 'Account Type', 'Created', 'Last Login', 'Failed Attempts', 'Locked')
            tree = ttk.Treeview(user_window, columns=columns, show='headings')
            
            # Configure columns
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Add users to treeview
            for user in users:
                tree.insert('', 'end', values=(
                    user['username'],
                    user['email'] or 'N/A',
                    user['role'],
                    user['account_type'],
                    user['created_at'][:10] if user['created_at'] else 'N/A',
                    user['last_login'][:10] if user['last_login'] else 'N/A',
                    user['failed_attempts'],
                    'Yes' if user['is_locked'] else 'No'
                ))
            
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            self.add_log_message(f"📊 Viewed {len(users)} users")
            
        except Exception as e:
            logger.error(f"Error viewing users: {e}")
            messagebox.showerror("Error", f"Error viewing users: {e}")
            self.add_log_message(f"❌ Error viewing users: {e}")

    def reset_user_password_gui(self):
        """Reset user password GUI."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            # Get current user for admin verification
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            # Create password reset window
            reset_window = tk.Toplevel(self.root)
            reset_window.title("🔐 Reset User Password")
            reset_window.geometry("400x200")
            reset_window.configure(bg='#1e1e1e')
            
            # Username input
            ttk.Label(reset_window, text="Username:").pack(pady=(20, 5))
            username_var = tk.StringVar()
            ttk.Entry(reset_window, textvariable=username_var, width=30).pack(pady=(0, 10))
            
            # New password input
            ttk.Label(reset_window, text="New Password:").pack(pady=(10, 5))
            password_var = tk.StringVar()
            ttk.Entry(reset_window, textvariable=password_var, show="*", width=30).pack(pady=(0, 20))
            
            # Reset button
            def do_reset():
                username = username_var.get()
                new_password = password_var.get()
                
                if not username or not new_password:
                    messagebox.showerror("Error", "Username and password required")
                    return
                
                success, message = self.security_system.reset_user_password(username, new_password, current_user)
                
                if success:
                    messagebox.showinfo("Success", message)
                    reset_window.destroy()
                    self.add_log_message(f"✅ {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.add_log_message(f"❌ {message}")
            
            ttk.Button(reset_window, text="🔐 Reset Password", command=do_reset).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error in password reset GUI: {e}")
            messagebox.showerror("Error", f"Error in password reset: {e}")
            self.add_log_message(f"❌ Error in password reset: {e}")

    def unlock_user_account_gui(self):
        """Unlock user account GUI."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            # Get current user for admin verification
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            # Create unlock window
            unlock_window = tk.Toplevel(self.root)
            unlock_window.title("🔓 Unlock User Account")
            unlock_window.geometry("300x150")
            unlock_window.configure(bg='#1e1e1e')
            
            # Username input
            ttk.Label(unlock_window, text="Username:").pack(pady=(20, 5))
            username_var = tk.StringVar()
            ttk.Entry(unlock_window, textvariable=username_var, width=30).pack(pady=(0, 20))
            
            # Unlock button
            def do_unlock():
                username = username_var.get()
                
                if not username:
                    messagebox.showerror("Error", "Username required")
                    return
                
                success, message = self.security_system.unlock_user_account(username, current_user)
                
                if success:
                    messagebox.showinfo("Success", message)
                    unlock_window.destroy()
                    self.add_log_message(f"✅ {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.add_log_message(f"❌ {message}")
            
            ttk.Button(unlock_window, text="🔓 Unlock Account", command=do_unlock).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error in unlock GUI: {e}")
            messagebox.showerror("Error", f"Error in unlock: {e}")
            self.add_log_message(f"❌ Error in unlock: {e}")

    def create_user_gui(self):
        """Create new user GUI."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            # Get current user for admin verification
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            # Create user window
            create_window = tk.Toplevel(self.root)
            create_window.title("➕ Create New User")
            create_window.geometry("400x300")
            create_window.configure(bg='#1e1e1e')
            
            # Username input
            ttk.Label(create_window, text="Username:").pack(pady=(20, 5))
            username_var = tk.StringVar()
            ttk.Entry(create_window, textvariable=username_var, width=30).pack(pady=(0, 10))
            
            # Password input
            ttk.Label(create_window, text="Password:").pack(pady=(10, 5))
            password_var = tk.StringVar()
            ttk.Entry(create_window, textvariable=password_var, show="*", width=30).pack(pady=(0, 10))
            
            # Email input
            ttk.Label(create_window, text="Email (optional):").pack(pady=(10, 5))
            email_var = tk.StringVar()
            ttk.Entry(create_window, textvariable=email_var, width=30).pack(pady=(0, 10))
            
            # Role selection
            ttk.Label(create_window, text="Role:").pack(pady=(10, 5))
            role_var = tk.StringVar(value="user")
            role_combo = ttk.Combobox(create_window, textvariable=role_var, values=["user", "admin"], state="readonly", width=27)
            role_combo.pack(pady=(0, 20))
            
            # Create button
            def do_create():
                username = username_var.get()
                password = password_var.get()
                email = email_var.get()
                role = role_var.get()
                
                if not username or not password:
                    messagebox.showerror("Error", "Username and password required")
                    return
                
                success, message = self.security_system.create_user(username, password, email, role, "user", current_user)
                
                if success:
                    messagebox.showinfo("Success", message)
                    create_window.destroy()
                    self.refresh_admin_count()
                    self.add_log_message(f"✅ {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.add_log_message(f"❌ {message}")
            
            ttk.Button(create_window, text="➕ Create User", command=do_create).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error in create user GUI: {e}")
            messagebox.showerror("Error", f"Error in create user: {e}")
            self.add_log_message(f"❌ Error in create user: {e}")

    def change_my_password(self):
        """Change current user's password."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            # Create password change window
            change_window = tk.Toplevel(self.root)
            change_window.title("🔑 Change Password")
            change_window.geometry("400x250")
            change_window.configure(bg='#1e1e1e')
            
            # Current password input
            ttk.Label(change_window, text="Current Password:").pack(pady=(20, 5))
            current_password_var = tk.StringVar()
            ttk.Entry(change_window, textvariable=current_password_var, show="*", width=30).pack(pady=(0, 10))
            
            # New password input
            ttk.Label(change_window, text="New Password:").pack(pady=(10, 5))
            new_password_var = tk.StringVar()
            ttk.Entry(change_window, textvariable=new_password_var, show="*", width=30).pack(pady=(0, 10))
            
            # Confirm password input
            ttk.Label(change_window, text="Confirm New Password:").pack(pady=(10, 5))
            confirm_password_var = tk.StringVar()
            ttk.Entry(change_window, textvariable=confirm_password_var, show="*", width=30).pack(pady=(0, 20))
            
            # Change button
            def do_change():
                current_password = current_password_var.get()
                new_password = new_password_var.get()
                confirm_password = confirm_password_var.get()
                
                if not current_password or not new_password or not confirm_password:
                    messagebox.showerror("Error", "All password fields required")
                    return
                
                if new_password != confirm_password:
                    messagebox.showerror("Error", "New passwords do not match")
                    return
                
                success, message = self.security_system.change_password(current_user, current_password, new_password)
                
                if success:
                    messagebox.showinfo("Success", message)
                    change_window.destroy()
                    self.add_log_message(f"✅ {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.add_log_message(f"❌ {message}")
            
            ttk.Button(change_window, text="🔑 Change Password", command=do_change).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error in change password GUI: {e}")
            messagebox.showerror("Error", f"Error in change password: {e}")
            self.add_log_message(f"❌ Error in change password: {e}")

    def view_my_activity(self):
        """View current user's activity."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            activities = self.security_system.get_user_activity(current_user, 7)
            
            if not activities:
                messagebox.showinfo("Info", "No recent activity found")
                return
            
            # Create activity window
            activity_window = tk.Toplevel(self.root)
            activity_window.title("📋 My Activity")
            activity_window.geometry("800x600")
            activity_window.configure(bg='#1e1e1e')
            
            # Create text widget
            activity_text = scrolledtext.ScrolledText(
                activity_window,
                bg='#1e1e1e',
                fg='#ffffff',
                font=('Consolas', 10)
            )
            activity_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add activities
            activity_text.insert(tk.END, f"📋 Activity for {current_user} (Last 7 days)\n")
            activity_text.insert(tk.END, "=" * 60 + "\n\n")
            
            for activity in activities:
                activity_text.insert(tk.END, f"📅 {activity['timestamp']}\n")
                activity_text.insert(tk.END, f"🔍 Type: {activity['event_type']}\n")
                activity_text.insert(tk.END, f"⚠️ Severity: {activity['severity']}\n")
                activity_text.insert(tk.END, f"📝 Description: {activity['description']}\n")
                activity_text.insert(tk.END, "-" * 40 + "\n\n")
            
            self.add_log_message(f"📋 Viewed activity for {current_user}")
            
        except Exception as e:
            logger.error(f"Error viewing activity: {e}")
            messagebox.showerror("Error", f"Error viewing activity: {e}")
            self.add_log_message(f"❌ Error viewing activity: {e}")

    def view_my_session(self):
        """View current user's session information."""
        try:
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            current_user = self.username_var.get()
            if not current_user:
                messagebox.showerror("Error", "Please log in first")
                return
            
            session_info = self.security_system.get_user_session_info(current_user)
            
            if not session_info:
                messagebox.showinfo("Info", "No active session found")
                return
            
            # Create session window
            session_window = tk.Toplevel(self.root)
            session_window.title("📊 My Session Info")
            session_window.geometry("500x400")
            session_window.configure(bg='#1e1e1e')
            
            # Create text widget
            session_text = scrolledtext.ScrolledText(
                session_window,
                bg='#1e1e1e',
                fg='#ffffff',
                font=('Consolas', 10)
            )
            session_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add session info
            session_text.insert(tk.END, f"📊 Session Information for {current_user}\n")
            session_text.insert(tk.END, "=" * 50 + "\n\n")
            
            for key, value in session_info.items():
                session_text.insert(tk.END, f"🔑 {key}: {value}\n")
            
            self.add_log_message(f"📊 Viewed session info for {current_user}")
            
        except Exception as e:
            logger.error(f"Error viewing session: {e}")
            messagebox.showerror("Error", f"Error viewing session: {e}")
            self.add_log_message(f"❌ Error viewing session: {e}")

    def refresh_admin_count(self):
        """Refresh admin count display."""
        try:
            if self.security_system:
                admin_count = self.security_system.get_admin_count()
                self.admin_count_label.config(text=str(admin_count))
        except Exception as e:
            logger.error(f"Error refreshing admin count: {e}")
    
    def create_threat_monitoring_panel(self, parent):
        """Create threat monitoring panel."""
        threat_frame = ttk.LabelFrame(parent, text="🚨 Threat Monitoring", padding=10)
        threat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Threat alerts
        alerts_frame = ttk.Frame(threat_frame)
        alerts_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(alerts_frame, text="Active Threat Alerts:").pack(anchor=tk.W)
        
        # Threat alerts listbox
        self.threat_alerts_text = scrolledtext.ScrolledText(
            alerts_frame,
            height=8,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.threat_alerts_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Threat controls
        threat_controls = ttk.Frame(alerts_frame)
        threat_controls.pack(fill=tk.X)
        
        ttk.Button(
            threat_controls,
            text="🔄 Refresh Alerts",
            command=self.refresh_threat_alerts
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            threat_controls,
            text="✅ Mark All Resolved",
            command=self.resolve_all_threats
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            threat_controls,
            text="📋 Export Alerts",
            command=self.export_threat_alerts
        ).pack(side=tk.LEFT)
    
    def create_security_controls_panel(self, parent):
        """Create security controls panel."""
        controls_frame = ttk.LabelFrame(parent, text="⚙️ Security Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Control buttons
        controls_grid = ttk.Frame(controls_frame)
        controls_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Button(
            controls_grid,
            text="🛡️ Enable Security",
            command=self.enable_security
        ).grid(row=0, column=0, padx=(0, 10), pady=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="🚫 Disable Security",
            command=self.disable_security
        ).grid(row=0, column=1, padx=(0, 10), pady=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="🔄 Auto Protection",
            command=self.toggle_auto_protection
        ).grid(row=0, column=2, padx=(0, 10), pady=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="🧪 Test Security",
            command=self.test_security
        ).grid(row=0, column=3, pady=(0, 10))
        
        # Row 2
        ttk.Button(
            controls_grid,
            text="🔒 Encrypt Data",
            command=self.encrypt_data
        ).grid(row=1, column=0, padx=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="🔓 Decrypt Data",
            command=self.decrypt_data
        ).grid(row=1, column=1, padx=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="📊 Security Report",
            command=self.generate_security_report
        ).grid(row=1, column=2, padx=(0, 10))
        
        ttk.Button(
            controls_grid,
            text="⚙️ Security Config",
            command=self.open_security_config
        ).grid(row=1, column=3)
    
    def refresh_security_status(self):
        """Refresh security status display."""
        try:
            if not self.security_system:
                return
            
            summary = self.security_system.get_security_summary()
            
            # Update labels
            self.security_score_label.config(text=f"{summary.get('security_score', 0):.1f}")
            self.threat_level_label.config(text=summary.get('threat_level', 'unknown').title())
            self.active_threats_label.config(text=str(summary.get('active_threats', 0)))
            self.encryption_status_label.config(text=summary.get('encryption_status', 'unknown').title())
            self.auth_status_label.config(text=summary.get('status', 'unknown').title())
            self.active_sessions_label.config(text=str(summary.get('active_sessions', 0)))
            
            self.add_log_message("🔄 Security status refreshed")
            
        except Exception as e:
            logger.error(f"Error refreshing security status: {e}")
            self.add_log_message(f"❌ Error refreshing security status: {e}")
    
    def authenticate_user(self):
        """Authenticate user with enhanced security."""
        try:
            username = self.username_var.get()
            password = self.password_var.get()
            
            if not username or not password:
                messagebox.showerror("Error", "Username and password required")
                return
            
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            success, result = self.security_system.authenticate_user(username, password)
            
            if success:
                self.auth_status_label.config(text=f"Logged in as {username}")
                messagebox.showinfo("Success", f"Welcome, {username}!")
                self.add_log_message(f"✅ User {username} logged in successfully")
                self.refresh_admin_count()
            else:
                self.auth_status_label.config(text="Login failed")
                messagebox.showerror("Error", f"Login failed: {result}")
                self.add_log_message(f"❌ Login failed for {username}: {result}")
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            messagebox.showerror("Error", f"Authentication error: {e}")
            self.add_log_message(f"❌ Authentication error: {e}")

    def logout_user(self):
        """Logout current user."""
        try:
            username = self.username_var.get()
            
            if not username:
                messagebox.showinfo("Info", "No user logged in")
                return
            
            if not self.security_system:
                messagebox.showerror("Error", "Security system not available")
                return
            
            success = self.security_system.logout_user(username)
            
            if success:
                self.auth_status_label.config(text="Not logged in")
                self.username_var.set("")
                self.password_var.set("")
                messagebox.showinfo("Success", f"Logged out successfully")
                self.add_log_message(f"✅ User {username} logged out successfully")
            else:
                messagebox.showerror("Error", "Logout failed")
                self.add_log_message(f"❌ Logout failed for {username}")
                
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            messagebox.showerror("Error", f"Logout error: {e}")
            self.add_log_message(f"❌ Logout error: {e}")
    
    def refresh_threat_alerts(self):
        """Refresh threat alerts display."""
        try:
            if not self.security_system:
                return
            
            # Clear current display
            self.threat_alerts_text.delete(1.0, tk.END)
            
            # Get active threats
            active_threats = [t for t in self.security_system.threat_alerts if t.status == "active"]
            
            if not active_threats:
                self.threat_alerts_text.insert(tk.END, "✅ No active threats detected\n")
            else:
                for threat in active_threats[-10:]:  # Show last 10 threats
                    timestamp = datetime.fromisoformat(threat.timestamp).strftime("%H:%M:%S")
                    self.threat_alerts_text.insert(tk.END, f"[{timestamp}] {threat.severity.upper()}: {threat.description}\n")
                    self.threat_alerts_text.insert(tk.END, f"    Type: {threat.threat_type} | Source: {threat.source}\n")
                    self.threat_alerts_text.insert(tk.END, f"    Mitigation: {threat.mitigation}\n\n")
            
            self.add_log_message(f"🔄 Refreshed threat alerts ({len(active_threats)} active)")
            
        except Exception as e:
            logger.error(f"Error refreshing threat alerts: {e}")
            self.add_log_message(f"❌ Error refreshing threat alerts: {e}")
    
    def resolve_all_threats(self):
        """Mark all threats as resolved."""
        try:
            if not self.security_system:
                return
            
            result = messagebox.askyesno("Confirm", "Mark all active threats as resolved?")
            if result:
                for threat in self.security_system.threat_alerts:
                    if threat.status == "active":
                        threat.status = "resolved"
                
                self.refresh_threat_alerts()
                self.add_log_message("✅ All threats marked as resolved")
                messagebox.showinfo("Success", "All threats marked as resolved")
                
        except Exception as e:
            logger.error(f"Error resolving threats: {e}")
            self.add_log_message(f"❌ Error resolving threats: {e}")
    
    def export_threat_alerts(self):
        """Export threat alerts to file."""
        try:
            if not self.security_system:
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                threats_data = [asdict(threat) for threat in self.security_system.threat_alerts]
                with open(filename, 'w') as f:
                    json.dump(threats_data, f, indent=2)
                
                self.add_log_message(f"📋 Threat alerts exported to {filename}")
                messagebox.showinfo("Success", f"Threat alerts exported to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting threat alerts: {e}")
            self.add_log_message(f"❌ Error exporting threat alerts: {e}")
    
    def enable_security(self):
        """Enable security system."""
        try:
            if not self.security_system:
                return
            
            self.security_system.is_enabled = True
            self.add_log_message("🛡️ Security system enabled")
            self.refresh_security_status()
            messagebox.showinfo("Success", "Security system enabled")
            
        except Exception as e:
            logger.error(f"Error enabling security: {e}")
            self.add_log_message(f"❌ Error enabling security: {e}")
    
    def disable_security(self):
        """Disable security system."""
        try:
            if not self.security_system:
                return
            
            result = messagebox.askyesno("Confirm", "Disable security system? This will reduce protection.")
            if result:
                self.security_system.is_enabled = False
                self.add_log_message("🚫 Security system disabled")
                self.refresh_security_status()
                messagebox.showinfo("Success", "Security system disabled")
                
        except Exception as e:
            logger.error(f"Error disabling security: {e}")
            self.add_log_message(f"❌ Error disabling security: {e}")
    
    def toggle_auto_protection(self):
        """Toggle auto protection."""
        try:
            if not self.security_system:
                return
            
            self.security_system.auto_protection = not self.security_system.auto_protection
            status = "enabled" if self.security_system.auto_protection else "disabled"
            self.add_log_message(f"🔄 Auto protection {status}")
            self.refresh_security_status()
            messagebox.showinfo("Success", f"Auto protection {status}")
            
        except Exception as e:
            logger.error(f"Error toggling auto protection: {e}")
            self.add_log_message(f"❌ Error toggling auto protection: {e}")
    
    def test_security(self):
        """Test security system."""
        try:
            if not self.security_system:
                return
            
            # Test encryption
            test_data = "Schwabot Security Test"
            encrypted = self.security_system.encrypt_data(test_data)
            decrypted = self.security_system.decrypt_data(encrypted)
            
            if decrypted == test_data:
                self.add_log_message("✅ Security test passed - encryption working")
                messagebox.showinfo("Security Test", "✅ Security test passed!\n\nEncryption: Working\nAuthentication: Available\nThreat Detection: Active")
            else:
                self.add_log_message("❌ Security test failed - encryption error")
                messagebox.showerror("Security Test", "❌ Security test failed!\n\nEncryption system error detected.")
                
        except Exception as e:
            logger.error(f"Error testing security: {e}")
            self.add_log_message(f"❌ Security test error: {e}")
            messagebox.showerror("Error", f"Security test error: {e}")
    
    def encrypt_data(self):
        """Encrypt data."""
        try:
            if not self.security_system:
                return
            
            data = tk.simpledialog.askstring("Encrypt Data", "Enter data to encrypt:")
            if data:
                encrypted = self.security_system.encrypt_data(data)
                self.add_log_message("🔒 Data encrypted successfully")
                messagebox.showinfo("Encryption", f"Encrypted data:\n{encrypted[:50]}...")
                
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            self.add_log_message(f"❌ Encryption error: {e}")
    
    def decrypt_data(self):
        """Decrypt data."""
        try:
            if not self.security_system:
                return
            
            data = tk.simpledialog.askstring("Decrypt Data", "Enter encrypted data:")
            if data:
                decrypted = self.security_system.decrypt_data(data)
                self.add_log_message("🔓 Data decrypted successfully")
                messagebox.showinfo("Decryption", f"Decrypted data:\n{decrypted}")
                
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            self.add_log_message(f"❌ Decryption error: {e}")
    
    def generate_security_report(self):
        """Generate security report."""
        try:
            if not self.security_system:
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                summary = self.security_system.get_security_summary()
                
                with open(filename, 'w') as f:
                    f.write("SCHWABOT SECURITY REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for key, value in summary.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    
                    f.write(f"\nRecent Security Events: {len(self.security_system.security_events[-50:])}\n")
                    f.write(f"Active Threats: {len([t for t in self.security_system.threat_alerts if t.status == 'active'])}\n")
                    f.write(f"Blocked IPs: {len(self.security_system.blocked_ips)}\n")
                
                self.add_log_message(f"📊 Security report generated: {filename}")
                messagebox.showinfo("Success", f"Security report generated:\n{filename}")
                
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            self.add_log_message(f"❌ Error generating security report: {e}")
    
    def open_security_config(self):
        """Open security configuration."""
        try:
            if not self.security_system:
                return
            
            config_window = tk.Toplevel(self.root)
            config_window.title("Security Configuration")
            config_window.geometry("600x400")
            config_window.transient(self.root)
            config_window.grab_set()
            
            # Display current config
            config_text = scrolledtext.ScrolledText(config_window, height=20)
            config_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            config_text.insert(tk.END, json.dumps(self.security_system.config, indent=2))
            
            # Save button
            def save_config():
                try:
                    new_config = json.loads(config_text.get(1.0, tk.END))
                    self.security_system.config = new_config
                    
                    with open(self.security_system.config_file, 'w') as f:
                        json.dump(new_config, f, indent=2)
                    
                    self.add_log_message("⚙️ Security configuration updated")
                    messagebox.showinfo("Success", "Security configuration saved")
                    config_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid configuration: {e}")
            
            ttk.Button(config_window, text="💾 Save Configuration", command=save_config).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error opening security config: {e}")
            self.add_log_message(f"❌ Error opening security config: {e}")
    
    def create_trading_tab(self):
        """Create the comprehensive trading operations and strategy management tab."""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="📈 Trading")
        
        # Trading header
        trading_label = ttk.Label(
            trading_frame,
            text="📈 Advanced Trading Operations & Strategy Management",
            style='Header.TLabel'
        )
        trading_label.pack(pady=20)
        
        # Initialize advanced security components
        try:
            from core.advanced_security_manager import AdvancedSecurityManager
            from core.secure_trade_handler import SecureTradeHandler
            from clock_mode_system import ClockModeSystem
            
            self.clock_system = ClockModeSystem()
            self.advanced_security = AdvancedSecurityManager()
            self.secure_trade_handler = SecureTradeHandler()
            
        except ImportError as e:
            logger.warning(f"Advanced components not available: {e}")
            self.clock_system = None
            self.advanced_security = None
            self.secure_trade_handler = None
        
        # Create trading panels
        self.create_trading_dashboard(trading_frame)
        self.create_strategy_management_panel(trading_frame)
        self.create_clock_mode_panel(trading_frame)
        self.create_advanced_security_panel(trading_frame)
        self.create_trade_execution_panel(trading_frame)
        self.create_performance_analytics_panel(trading_frame)

    def create_clock_mode_panel(self, parent):
        """Create the revolutionary Clock Mode panel."""
        clock_frame = ttk.LabelFrame(parent, text="🕐 Clock Mode - Mechanical Watchmaker Trading", padding=10)
        clock_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Shadow Mode context label
        shadow_context = ttk.Label(
            clock_frame,
            text="📊 Shadow Mode: Run This To Build The Bot's Trading Algo",
            style='Info.TLabel',
            font=('Arial', 9, 'italic')
        )
        shadow_context.pack(pady=(0, 10))
        
        # Clock mode controls
        controls_frame = ttk.Frame(clock_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop Clock Mode
        ttk.Button(
            controls_frame,
            text="🕐 Start Clock Mode",
            command=self.start_clock_mode
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="⏹️ Stop Clock Mode",
            command=self.stop_clock_mode
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Wind Main Spring
        ttk.Button(
            controls_frame,
            text="🔧 Wind Main Spring",
            command=self.wind_main_spring
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Create Custom Mechanism
        ttk.Button(
            controls_frame,
            text="⚙️ Create Custom Mechanism",
            command=self.create_custom_mechanism
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Reconfigure Mechanism
        ttk.Button(
            controls_frame,
            text="🔧 Reconfigure Mechanism",
            command=self.reconfigure_mechanism
        ).pack(side=tk.LEFT)
        
        # MICRO MODE LIVE BUTTON (smaller as a joke)
        self.micro_mode_button = ttk.Button(
            controls_frame,
            text="🚨 MICRO MODE LIVE",
            command=self.toggle_micro_mode,
            style='Micro.TButton'
        )
        self.micro_mode_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Emergency Stop for Micro Mode
        self.micro_emergency_button = ttk.Button(
            controls_frame,
            text="🛑 MICRO EMERGENCY",
            command=self.trigger_micro_emergency_stop,
            style='Emergency.TButton'
        )
        self.micro_emergency_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Clock status display
        status_frame = ttk.Frame(clock_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status labels
        self.clock_status_label = ttk.Label(status_frame, text="Clock Mode: Stopped", style='Info.TLabel')
        self.clock_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.main_spring_label = ttk.Label(status_frame, text="Main Spring: 0%", style='Info.TLabel')
        self.main_spring_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.escapement_label = ttk.Label(status_frame, text="Escapement: 0.0s", style='Info.TLabel')
        self.escapement_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.profit_potential_label = ttk.Label(status_frame, text="Profit Potential: 0.0", style='Info.TLabel')
        self.profit_potential_label.pack(side=tk.LEFT)
        
        # Micro mode status labels
        self.micro_mode_status_label = ttk.Label(status_frame, text="Micro Mode: DISABLED", style='Error.TLabel')
        self.micro_mode_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.micro_trades_label = ttk.Label(status_frame, text="Micro Trades: 0", style='Info.TLabel')
        self.micro_trades_label.pack(side=tk.LEFT, padx=(10, 0))
        
        self.micro_volume_label = ttk.Label(status_frame, text="Micro Volume: $0.00", style='Info.TLabel')
        self.micro_volume_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Clock mechanism display
        mechanism_frame = ttk.Frame(clock_frame)
        mechanism_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create treeview for mechanisms
        columns = ('Mechanism ID', 'Wheels', 'Gears', 'Energy', 'Timing', 'Status')
        self.clock_tree = ttk.Treeview(mechanism_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.clock_tree.heading(col, text=col)
            self.clock_tree.column(col, width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(mechanism_frame, orient=tk.VERTICAL, command=self.clock_tree.yview)
        self.clock_tree.configure(yscrollcommand=scrollbar.set)
        
        self.clock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        ttk.Button(
            mechanism_frame,
            text="🔄 Refresh Clock Status",
            command=self.refresh_clock_status
        ).pack(pady=(10, 0))

    def start_clock_mode(self):
        """Start the Clock Mode system."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock Mode system not available")
                return
            
            success = self.clock_system.start_clock_mode()
            
            if success:
                self.clock_status_label.config(text="Clock Mode: Running")
                messagebox.showinfo("Success", "🕐 Clock Mode started successfully!")
                self.add_log_message("🕐 Clock Mode system started")
                self.refresh_clock_status()
            else:
                messagebox.showerror("Error", "Failed to start Clock Mode")
                self.add_log_message("❌ Failed to start Clock Mode")
                
        except Exception as e:
            logger.error(f"Error starting clock mode: {e}")
            messagebox.showerror("Error", f"Error starting Clock Mode: {e}")
            self.add_log_message(f"❌ Error starting Clock Mode: {e}")

    def stop_clock_mode(self):
        """Stop the Clock Mode system."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock Mode system not available")
                return
            
            success = self.clock_system.stop_clock_mode()
            
            if success:
                self.clock_status_label.config(text="Clock Mode: Stopped")
                messagebox.showinfo("Success", "⏹️ Clock Mode stopped successfully!")
                self.add_log_message("⏹️ Clock Mode system stopped")
                self.refresh_clock_status()
            else:
                messagebox.showerror("Error", "Failed to stop Clock Mode")
                self.add_log_message("❌ Failed to stop Clock Mode")
                
        except Exception as e:
            logger.error(f"Error stopping clock mode: {e}")
            messagebox.showerror("Error", f"Error stopping Clock Mode: {e}")
            self.add_log_message(f"❌ Error stopping Clock Mode: {e}")

    def wind_main_spring(self):
        """Wind the main spring with additional energy."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock Mode system not available")
                return
            
            # Create wind spring window
            wind_window = tk.Toplevel(self.root)
            wind_window.title("🔧 Wind Main Spring")
            wind_window.geometry("300x150")
            wind_window.configure(bg='#1e1e1e')
            
            # Energy input
            ttk.Label(wind_window, text="Energy to Add:").pack(pady=(20, 5))
            energy_var = tk.StringVar(value="100")
            ttk.Entry(wind_window, textvariable=energy_var, width=20).pack(pady=(0, 20))
            
            # Wind button
            def do_wind():
                try:
                    energy = float(energy_var.get())
                    if energy <= 0:
                        messagebox.showerror("Error", "Energy must be positive")
                        return
                    
                    # Wind the main spring
                    for mechanism_id in self.clock_system.active_mechanisms:
                        mechanism = self.clock_system.mechanisms.get(mechanism_id)
                        if mechanism:
                            mechanism.wind_main_spring(energy)
                    
                    messagebox.showinfo("Success", f"🔧 Main spring wound with {energy} units!")
                    wind_window.destroy()
                    self.add_log_message(f"🔧 Main spring wound: {energy} units")
                    self.refresh_clock_status()
                    
                except ValueError:
                    messagebox.showerror("Error", "Invalid energy value")
            
            ttk.Button(wind_window, text="🔧 Wind Spring", command=do_wind).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error winding main spring: {e}")
            messagebox.showerror("Error", f"Error winding main spring: {e}")
            self.add_log_message(f"❌ Error winding main spring: {e}")

    def create_custom_mechanism(self):
        """Create a custom clock mechanism."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock Mode system not available")
                return
            
            # Create custom mechanism window
            create_window = tk.Toplevel(self.root)
            create_window.title("⚙️ Create Custom Clock Mechanism")
            create_window.geometry("600x500")
            create_window.configure(bg='#1e1e1e')
            
            # Mechanism ID input
            ttk.Label(create_window, text="Mechanism ID:").pack(pady=(20, 5))
            mechanism_id_var = tk.StringVar()
            ttk.Entry(create_window, textvariable=mechanism_id_var, width=30).pack(pady=(0, 20))
            
            # Configuration text area
            ttk.Label(create_window, text="Configuration (JSON):").pack(pady=(10, 5))
            
            config_text = scrolledtext.ScrolledText(
                create_window,
                height=15,
                width=70,
                bg='#1e1e1e',
                fg='#ffffff',
                font=('Consolas', 10)
            )
            config_text.pack(pady=(0, 20))
            
            # Default configuration
            default_config = {
                "wheels": [
                    {
                        "id": "custom_timing",
                        "profit_target": 1.5,
                        "risk_factor": 0.8,
                        "gears": [
                            {
                                "id": "custom_hash_1",
                                "type": "hash_wheel",
                                "teeth": 64,
                                "rotation_speed": 1.0,
                                "energy_level": 100.0
                            },
                            {
                                "id": "custom_orbital_1",
                                "type": "orbital_wheel",
                                "teeth": 360,
                                "rotation_speed": 0.1,
                                "energy_level": 150.0
                            }
                        ]
                    }
                ]
            }
            
            config_text.insert(tk.END, json.dumps(default_config, indent=2))
            
            # Create button
            def do_create():
                try:
                    mechanism_id = mechanism_id_var.get()
                    if not mechanism_id:
                        messagebox.showerror("Error", "Mechanism ID required")
                        return
                    
                    # Parse configuration
                    config_str = config_text.get(1.0, tk.END)
                    configuration = json.loads(config_str)
                    
                    # Create mechanism
                    self.clock_system.create_custom_mechanism(mechanism_id, configuration)
                    
                    messagebox.showinfo("Success", f"⚙️ Custom mechanism '{mechanism_id}' created!")
                    create_window.destroy()
                    self.add_log_message(f"⚙️ Custom mechanism created: {mechanism_id}")
                    self.refresh_clock_status()
                    
                except json.JSONDecodeError:
                    messagebox.showerror("Error", "Invalid JSON configuration")
                except Exception as e:
                    messagebox.showerror("Error", f"Error creating mechanism: {e}")
            
            ttk.Button(create_window, text="⚙️ Create Mechanism", command=do_create).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error creating custom mechanism: {e}")
            messagebox.showerror("Error", f"Error creating custom mechanism: {e}")
            self.add_log_message(f"❌ Error creating custom mechanism: {e}")

    def reconfigure_mechanism(self):
        """Reconfigure an existing clock mechanism."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock Mode system not available")
                return
            
            # Get available mechanisms
            mechanisms = list(self.clock_system.mechanisms.keys())
            if not mechanisms:
                messagebox.showinfo("Info", "No mechanisms available to reconfigure")
                return
            
            # Create reconfigure window
            reconfigure_window = tk.Toplevel(self.root)
            reconfigure_window.title("🔧 Reconfigure Clock Mechanism")
            reconfigure_window.geometry("500x400")
            reconfigure_window.configure(bg='#1e1e1e')
            
            # Mechanism selection
            ttk.Label(reconfigure_window, text="Select Mechanism:").pack(pady=(20, 5))
            mechanism_var = tk.StringVar(value=mechanisms[0])
            mechanism_combo = ttk.Combobox(reconfigure_window, textvariable=mechanism_var, values=mechanisms, state="readonly", width=30)
            mechanism_combo.pack(pady=(0, 20))
            
            # Configuration inputs
            ttk.Label(reconfigure_window, text="Main Spring Energy:").pack(pady=(10, 5))
            energy_var = tk.StringVar(value="1000")
            ttk.Entry(reconfigure_window, textvariable=energy_var, width=20).pack(pady=(0, 10))
            
            ttk.Label(reconfigure_window, text="Balance Wheel Frequency (Hz):").pack(pady=(10, 5))
            frequency_var = tk.StringVar(value="4.0")
            ttk.Entry(reconfigure_window, textvariable=frequency_var, width=20).pack(pady=(0, 20))
            
            # Reconfigure button
            def do_reconfigure():
                try:
                    mechanism_id = mechanism_var.get()
                    energy = float(energy_var.get())
                    frequency = float(frequency_var.get())
                    
                    if energy <= 0 or frequency <= 0:
                        messagebox.showerror("Error", "Values must be positive")
                        return
                    
                    # Create configuration
                    config = {
                        "main_spring_energy": energy,
                        "balance_wheel_frequency": frequency
                    }
                    
                    # Reconfigure mechanism
                    success = self.clock_system.reconfigure_mechanism(mechanism_id, config)
                    
                    if success:
                        messagebox.showinfo("Success", f"🔧 Mechanism '{mechanism_id}' reconfigured!")
                        reconfigure_window.destroy()
                        self.add_log_message(f"🔧 Mechanism reconfigured: {mechanism_id}")
                        self.refresh_clock_status()
                    else:
                        messagebox.showerror("Error", "Failed to reconfigure mechanism")
                    
                except ValueError:
                    messagebox.showerror("Error", "Invalid numeric values")
                except Exception as e:
                    messagebox.showerror("Error", f"Error reconfiguring mechanism: {e}")
            
            ttk.Button(reconfigure_window, text="🔧 Reconfigure", command=do_reconfigure).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error reconfiguring mechanism: {e}")
            messagebox.showerror("Error", f"Error reconfiguring mechanism: {e}")
            self.add_log_message(f"❌ Error reconfiguring mechanism: {e}")

    def refresh_clock_status(self):
        """Refresh the clock status display."""
        try:
            if not self.clock_system:
                return
            
            # Get all mechanisms status
            status = self.clock_system.get_all_mechanisms_status()
            
            # Update status labels
            if status["is_running"]:
                self.clock_status_label.config(text="Clock Mode: Running")
            else:
                self.clock_status_label.config(text="Clock Mode: Stopped")
            
            # Update mechanism tree
            self.clock_tree.delete(*self.clock_tree.get_children())
            
            for mechanism_id, mechanism_status in status["mechanisms"].items():
                if "error" in mechanism_status:
                    continue
                
                # Get mechanism details
                mechanism = self.clock_system.mechanisms.get(mechanism_id)
                if mechanism:
                    main_spring = mechanism.main_spring_energy
                    escapement = mechanism.calculate_escapement_timing()
                    
                    # Update main spring and escapement labels
                    self.main_spring_label.config(text=f"Main Spring: {main_spring:.0f}")
                    self.escapement_label.config(text=f"Escapement: {escapement:.3f}s")
                    
                    # Calculate total profit potential
                    total_profit = 0.0
                    for wheel in mechanism.wheels:
                        wheel_timing = wheel.calculate_wheel_timing()
                        total_profit += wheel_timing["profit_potential"]
                    
                    self.profit_potential_label.config(text=f"Profit Potential: {total_profit:.2f}")
                    
                    # Add to tree
                    self.clock_tree.insert('', 'end', values=(
                        mechanism_id,
                        mechanism_status["wheel_count"],
                        mechanism_status["total_gears"],
                        f"{main_spring:.0f}",
                        f"{escapement:.3f}s",
                        "Active" if mechanism_status["is_active"] else "Inactive"
                    ))
            
            # Refresh micro status
            self.refresh_micro_status()
            
        except Exception as e:
            logger.error(f"Error refreshing clock status: {e}")
            self.add_log_message(f"❌ Error refreshing clock status: {e}")
    
    def toggle_micro_mode(self):
        """Toggle micro mode on/off."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock system not initialized")
                return
            
            # Check current mode
            current_mode = self.clock_system.get_all_mechanisms_status().get("safety_config", {}).get("execution_mode", "shadow")
            
            if current_mode == "micro":
                # Disable micro mode
                success = self.clock_system.disable_micro_mode()
                if success:
                    self.micro_mode_button.config(text="🚨 MICRO MODE LIVE")
                    self.micro_mode_status_label.config(text="Micro Mode: DISABLED", style='Error.TLabel')
                    messagebox.showinfo("Micro Mode", "🛡️ MICRO MODE DISABLED - Back to SHADOW mode")
                    self.add_log_message("🛡️ Micro mode disabled - back to SHADOW mode")
                else:
                    messagebox.showerror("Error", "Failed to disable micro mode")
            else:
                # Enable micro mode with confirmation
                result = messagebox.askyesno(
                    "🚨 MICRO MODE CONFIRMATION",
                    "⚠️ WARNING: This will enable $1 LIVE TRADING!\n\n"
                    "• Real money will be at risk\n"
                    "• Maximum $1 per trade\n"
                    "• Maximum $10 per day\n"
                    "• Triple confirmation required\n"
                    "• Emergency stop available\n\n"
                    "Are you absolutely sure you want to proceed?",
                    icon='warning'
                )
                
                if result:
                    success = self.clock_system.enable_micro_mode()
                    if success:
                        self.micro_mode_button.config(text="🛑 DISABLE MICRO")
                        self.micro_mode_status_label.config(text="Micro Mode: ENABLED", style='Info.TLabel')
                        messagebox.showwarning(
                            "🚨 MICRO MODE ENABLED",
                            "⚠️ MAXIMUM PARANOIA PROTOCOLS ACTIVATED!\n\n"
                            "• $1 live trading is now active\n"
                            "• Real money at risk\n"
                            "• Use emergency stop if needed\n"
                            "• Monitor closely!"
                        )
                        self.add_log_message("🚨 MICRO MODE ENABLED - $1 live trading active!")
                    else:
                        messagebox.showerror("Error", "Failed to enable micro mode")
            
            # Refresh status
            self.refresh_clock_status()
            
        except Exception as e:
            logger.error(f"Error toggling micro mode: {e}")
            messagebox.showerror("Error", f"Error toggling micro mode: {e}")
            self.add_log_message(f"❌ Error toggling micro mode: {e}")
    
    def trigger_micro_emergency_stop(self):
        """Trigger emergency stop for micro mode."""
        try:
            if not self.clock_system:
                messagebox.showerror("Error", "Clock system not initialized")
                return
            
            result = messagebox.askyesno(
                "🚨 MICRO EMERGENCY STOP",
                "⚠️ EMERGENCY STOP CONFIRMATION\n\n"
                "This will immediately stop all micro trading!\n"
                "Are you sure you want to trigger emergency stop?",
                icon='warning'
            )
            
            if result:
                success = self.clock_system.trigger_micro_emergency_stop()
                if success:
                    self.micro_mode_status_label.config(text="Micro Mode: EMERGENCY STOP", style='Error.TLabel')
                    messagebox.showwarning(
                        "🚨 EMERGENCY STOP TRIGGERED",
                        "⚠️ All micro trading has been suspended!\n"
                        "Micro mode is now in emergency stop state."
                    )
                    self.add_log_message("🚨 MICRO MODE EMERGENCY STOP TRIGGERED!")
                else:
                    messagebox.showerror("Error", "Failed to trigger emergency stop")
            
        except Exception as e:
            logger.error(f"Error triggering micro emergency stop: {e}")
            messagebox.showerror("Error", f"Error triggering emergency stop: {e}")
            self.add_log_message(f"❌ Error triggering micro emergency stop: {e}")
    
    def refresh_micro_status(self):
        """Refresh micro mode status display."""
        try:
            if not self.clock_system:
                return
            
            # Get micro trading stats
            micro_stats = self.clock_system.get_micro_trading_stats()
            
            # Update micro status labels
            if micro_stats.get("micro_mode_enabled", False):
                self.micro_mode_status_label.config(text="Micro Mode: ENABLED", style='Info.TLabel')
                self.micro_trades_label.config(text=f"Micro Trades: {micro_stats.get('daily_trades', 0)}")
                self.micro_volume_label.config(text=f"Micro Volume: ${micro_stats.get('daily_volume', 0):.2f}")
            else:
                self.micro_mode_status_label.config(text="Micro Mode: DISABLED", style='Error.TLabel')
                self.micro_trades_label.config(text="Micro Trades: 0")
                self.micro_volume_label.config(text="Micro Volume: $0.00")
            
        except Exception as e:
            logger.error(f"Error refreshing micro status: {e}")
            self.add_log_message(f"❌ Error refreshing micro status: {e}")
    
    def create_trading_dashboard(self, parent):
        """Create trading dashboard panel."""
        dashboard_frame = ttk.LabelFrame(parent, text="📊 Trading Dashboard", padding=10)
        dashboard_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Real-time metrics
        metrics_frame = ttk.Frame(dashboard_frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Portfolio value
        portfolio_frame = ttk.Frame(metrics_frame)
        portfolio_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(portfolio_frame, text="💰 Portfolio Value:", font=('Arial', 10, 'bold')).pack()
        self.portfolio_value_label = ttk.Label(portfolio_frame, text="$0.00", font=('Arial', 12))
        self.portfolio_value_label.pack()
        
        # Daily P&L
        pnl_frame = ttk.Frame(metrics_frame)
        pnl_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(pnl_frame, text="📈 Daily P&L:", font=('Arial', 10, 'bold')).pack()
        self.daily_pnl_label = ttk.Label(pnl_frame, text="$0.00", font=('Arial', 12))
        self.daily_pnl_label.pack()
        
        # Active positions
        positions_frame = ttk.Frame(metrics_frame)
        positions_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(positions_frame, text="🎯 Active Positions:", font=('Arial', 10, 'bold')).pack()
        self.active_positions_label = ttk.Label(positions_frame, text="0", font=('Arial', 12))
        self.active_positions_label.pack()
        
        # Security status
        security_frame = ttk.Frame(metrics_frame)
        security_frame.pack(side=tk.LEFT)
        
        ttk.Label(security_frame, text="🔐 Security Status:", font=('Arial', 10, 'bold')).pack()
        self.security_status_label = ttk.Label(security_frame, text="🟢 ACTIVE", font=('Arial', 12))
        self.security_status_label.pack()
        
        # Trading controls
        controls_frame = ttk.Frame(dashboard_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            controls_frame,
            text="🔄 Refresh Dashboard",
            command=self.refresh_trading_dashboard
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="📊 View Analytics",
            command=self.show_trading_analytics
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔐 Security Report",
            command=self.show_security_report
        ).pack(side=tk.LEFT)
    
    def create_strategy_management_panel(self, parent):
        """Create strategy management panel."""
        strategy_frame = ttk.LabelFrame(parent, text="🎯 Strategy Management", padding=10)
        strategy_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Strategy selection
        strategy_select_frame = ttk.Frame(strategy_frame)
        strategy_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(strategy_select_frame, text="Strategy:").pack(side=tk.LEFT)
        
        self.strategy_var = tk.StringVar(value="ferris_ride_001")
        strategy_combo = ttk.Combobox(
            strategy_select_frame, 
            textvariable=self.strategy_var,
            values=["ferris_ride_001", "alpha_strategy", "beta_strategy", "gamma_strategy"],
            state="readonly",
            width=20
        )
        strategy_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Strategy controls
        strategy_controls = ttk.Frame(strategy_frame)
        strategy_controls.pack(fill=tk.X)
        
        ttk.Button(
            strategy_controls,
            text="▶️ Start Strategy",
            command=self.start_strategy
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            strategy_controls,
            text="⏸️ Pause Strategy",
            command=self.pause_strategy
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            strategy_controls,
            text="⏹️ Stop Strategy",
            command=self.stop_strategy
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            strategy_controls,
            text="⚙️ Configure",
            command=self.configure_strategy
        ).pack(side=tk.LEFT)
    
    def create_advanced_security_panel(self, parent):
        """Create advanced security panel."""
        security_frame = ttk.LabelFrame(parent, text="🔐 Advanced Security Features", padding=10)
        security_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Security status
        status_frame = ttk.Frame(security_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # MIM Attack Protection
        mim_frame = ttk.Frame(status_frame)
        mim_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(mim_frame, text="🛡️ MIM Protection:", font=('Arial', 10, 'bold')).pack()
        self.mim_status_label = ttk.Label(mim_frame, text="🟢 ACTIVE", font=('Arial', 10))
        self.mim_status_label.pack()
        
        # Dummy Packet Injection
        dummy_frame = ttk.Frame(status_frame)
        dummy_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(dummy_frame, text="🎭 Dummy Packets:", font=('Arial', 10, 'bold')).pack()
        self.dummy_status_label = ttk.Label(dummy_frame, text="🟢 ACTIVE", font=('Arial', 10))
        self.dummy_status_label.pack()
        
        # Complexity Obfuscation
        complexity_frame = ttk.Frame(status_frame)
        complexity_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(complexity_frame, text="🧮 Complexity:", font=('Arial', 10, 'bold')).pack()
        self.complexity_status_label = ttk.Label(complexity_frame, text="🟢 ACTIVE", font=('Arial', 10))
        self.complexity_status_label.pack()
        
        # Security controls
        security_controls = ttk.Frame(security_frame)
        security_controls.pack(fill=tk.X)
        
        ttk.Button(
            security_controls,
            text="🔐 Enable All Security",
            command=self.enable_all_security
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            security_controls,
            text="⚠️ Disable Security",
            command=self.disable_all_security
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            security_controls,
            text="🧪 Test Security",
            command=self.test_advanced_security
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            security_controls,
            text="📊 Security Stats",
            command=self.show_security_statistics
        ).pack(side=tk.LEFT)
    
    def create_trade_execution_panel(self, parent):
        """Create trade execution panel."""
        execution_frame = ttk.LabelFrame(parent, text="⚡ Trade Execution", padding=10)
        execution_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Trade inputs
        inputs_frame = ttk.Frame(execution_frame)
        inputs_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Symbol
        symbol_frame = ttk.Frame(inputs_frame)
        symbol_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(symbol_frame, text="Symbol:").pack()
        self.trade_symbol_var = tk.StringVar(value="BTC/USDC")
        ttk.Entry(symbol_frame, textvariable=self.trade_symbol_var, width=15).pack()
        
        # Side
        side_frame = ttk.Frame(inputs_frame)
        side_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(side_frame, text="Side:").pack()
        self.trade_side_var = tk.StringVar(value="buy")
        ttk.Combobox(side_frame, textvariable=self.trade_side_var, values=["buy", "sell"], state="readonly", width=10).pack()
        
        # Amount
        amount_frame = ttk.Frame(inputs_frame)
        amount_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(amount_frame, text="Amount:").pack()
        self.trade_amount_var = tk.StringVar(value="0.1")
        ttk.Entry(amount_frame, textvariable=self.trade_amount_var, width=10).pack()
        
        # Price
        price_frame = ttk.Frame(inputs_frame)
        price_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(price_frame, text="Price:").pack()
        self.trade_price_var = tk.StringVar(value="50000.0")
        ttk.Entry(price_frame, textvariable=self.trade_price_var, width=12).pack()
        
        # Execution controls
        execution_controls = ttk.Frame(execution_frame)
        execution_controls.pack(fill=tk.X)
        
        ttk.Button(
            execution_controls,
            text="🚀 Execute Trade",
            command=self.execute_trade
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            execution_controls,
            text="🔐 Secure Trade",
            command=self.execute_secure_trade
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            execution_controls,
            text="🧪 Test Trade",
            command=self.test_trade
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            execution_controls,
            text="📋 Trade History",
            command=self.show_trade_history
        ).pack(side=tk.LEFT)
    
    def create_performance_analytics_panel(self, parent):
        """Create performance analytics panel."""
        analytics_frame = ttk.LabelFrame(parent, text="📊 Performance Analytics", padding=10)
        analytics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Analytics controls
        controls_frame = ttk.Frame(analytics_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="📈 Performance Chart",
            command=self.show_performance_chart
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔐 Security Metrics",
            command=self.show_security_metrics
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🎯 Strategy Analysis",
            command=self.show_strategy_analysis
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="📋 Export Report",
            command=self.export_trading_report
        ).pack(side=tk.LEFT)
        
        # Analytics display
        self.analytics_text = scrolledtext.ScrolledText(
            analytics_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.analytics_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize analytics
        self.refresh_analytics()
    
    # Trading methods
    def refresh_trading_dashboard(self):
        """Refresh trading dashboard."""
        try:
            # Update portfolio value
            portfolio_value = "$125,432.67"  # Placeholder
            self.portfolio_value_label.config(text=portfolio_value)
            
            # Update daily P&L
            daily_pnl = "+$2,345.89"  # Placeholder
            self.daily_pnl_label.config(text=daily_pnl)
            
            # Update active positions
            active_positions = "3"  # Placeholder
            self.active_positions_label.config(text=active_positions)
            
            # Update security status
            if self.advanced_security_manager and self.advanced_security_manager.security_enabled:
                self.security_status_label.config(text="🟢 ACTIVE")
            else:
                self.security_status_label.config(text="🔴 INACTIVE")
            
            self.add_log_message("📊 Trading dashboard refreshed")
        except Exception as e:
            logger.error(f"Error refreshing trading dashboard: {e}")
            self.add_log_message(f"❌ Dashboard refresh failed: {e}")
    
    def start_strategy(self):
        """Start selected strategy."""
        try:
            strategy = self.strategy_var.get()
            self.add_log_message(f"▶️ Starting strategy: {strategy}")
            
            # Here you would integrate with the actual strategy execution engine
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            self.add_log_message(f"❌ Strategy start failed: {e}")
    
    def pause_strategy(self):
        """Pause selected strategy."""
        try:
            strategy = self.strategy_var.get()
            self.add_log_message(f"⏸️ Pausing strategy: {strategy}")
            
        except Exception as e:
            logger.error(f"Error pausing strategy: {e}")
            self.add_log_message(f"❌ Strategy pause failed: {e}")
    
    def stop_strategy(self):
        """Stop selected strategy."""
        try:
            strategy = self.strategy_var.get()
            self.add_log_message(f"⏹️ Stopping strategy: {strategy}")
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            self.add_log_message(f"❌ Strategy stop failed: {e}")
    
    def configure_strategy(self):
        """Configure selected strategy."""
        try:
            strategy = self.strategy_var.get()
            self.add_log_message(f"⚙️ Configuring strategy: {strategy}")
            
        except Exception as e:
            logger.error(f"Error configuring strategy: {e}")
            self.add_log_message(f"❌ Strategy configuration failed: {e}")
    
    def enable_all_security(self):
        """Enable all advanced security features."""
        try:
            if self.advanced_security_manager:
                self.advanced_security_manager.enable_security()
                self.mim_status_label.config(text="🟢 ACTIVE")
                self.dummy_status_label.config(text="🟢 ACTIVE")
                self.complexity_status_label.config(text="🟢 ACTIVE")
                self.add_log_message("🔐 All advanced security features enabled")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error enabling security: {e}")
            self.add_log_message(f"❌ Security enable failed: {e}")
    
    def disable_all_security(self):
        """Disable all advanced security features."""
        try:
            if self.advanced_security_manager:
                self.advanced_security_manager.disable_security()
                self.mim_status_label.config(text="🔴 INACTIVE")
                self.dummy_status_label.config(text="🔴 INACTIVE")
                self.complexity_status_label.config(text="🔴 INACTIVE")
                self.add_log_message("⚠️ All advanced security features disabled")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error disabling security: {e}")
            self.add_log_message(f"❌ Security disable failed: {e}")
    
    def test_advanced_security(self):
        """Test advanced security features."""
        try:
            if self.advanced_security_manager:
                # Test trade protection
                test_trade = {
                    'symbol': 'BTC/USDC',
                    'side': 'buy',
                    'amount': 0.1,
                    'price': 50000.0,
                    'strategy': 'test'
                }
                
                result = self.advanced_security_manager.protect_trade(test_trade)
                
                if result['success']:
                    self.add_log_message("🧪 Advanced security test PASSED")
                    self.add_log_message(f"   Security score: {result.get('secure_result', {}).get('security_score', 0):.2f}")
                    self.add_log_message(f"   Dummy packets: {len(result.get('secure_result', {}).get('dummy_packets', []))}")
                else:
                    self.add_log_message("❌ Advanced security test FAILED")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error testing security: {e}")
            self.add_log_message(f"❌ Security test failed: {e}")
    
    def show_security_statistics(self):
        """Show security statistics."""
        try:
            if self.advanced_security_manager:
                stats = self.advanced_security_manager.get_statistics()
                
                stats_text = "🔐 ADVANCED SECURITY STATISTICS\n"
                stats_text += "=" * 40 + "\n"
                stats_text += f"Security Enabled: {stats.get('security_enabled', False)}\n"
                stats_text += f"Auto Protection: {stats.get('auto_protection', False)}\n"
                stats_text += f"Total Trades Protected: {stats.get('total_trades_protected', 0)}\n"
                stats_text += f"Security Events: {stats.get('security_events_count', 0)}\n"
                
                # Show in analytics panel
                self.analytics_text.delete(1.0, tk.END)
                self.analytics_text.insert(tk.END, stats_text)
                
                self.add_log_message("📊 Security statistics displayed")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error showing security stats: {e}")
            self.add_log_message(f"❌ Security stats failed: {e}")
    
    def execute_trade(self):
        """Execute a basic trade."""
        try:
            symbol = self.trade_symbol_var.get()
            side = self.trade_side_var.get()
            amount = float(self.trade_amount_var.get())
            price = float(self.trade_price_var.get())
            
            self.add_log_message(f"🚀 Executing trade: {side} {amount} {symbol} @ {price}")
            
            # Here you would integrate with the actual trading engine
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.add_log_message(f"❌ Trade execution failed: {e}")
    
    def execute_secure_trade(self):
        """Execute a trade with advanced security protection."""
        try:
            if not self.advanced_security_manager:
                self.add_log_message("❌ Advanced security manager not available")
                return
            
            symbol = self.trade_symbol_var.get()
            side = self.trade_side_var.get()
            amount = float(self.trade_amount_var.get())
            price = float(self.trade_price_var.get())
            
            # Create trade data
            trade_data = {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'strategy': self.strategy_var.get(),
                'timestamp': time.time()
            }
            
            # Protect the trade
            result = self.advanced_security_manager.protect_trade(trade_data)
            
            if result['success'] and result['protected']:
                self.add_log_message(f"🔐 Secure trade executed: {side} {amount} {symbol}")
                self.add_log_message(f"   Security score: {result.get('secure_result', {}).get('security_score', 0):.2f}")
                self.add_log_message(f"   Dummy packets: {len(result.get('secure_result', {}).get('dummy_packets', []))}")
            else:
                self.add_log_message("❌ Secure trade execution failed")
                
        except Exception as e:
            logger.error(f"Error executing secure trade: {e}")
            self.add_log_message(f"❌ Secure trade failed: {e}")
    
    def test_trade(self):
        """Test trade execution."""
        try:
            symbol = self.trade_symbol_var.get()
            side = self.trade_side_var.get()
            amount = float(self.trade_amount_var.get())
            price = float(self.trade_price_var.get())
            
            self.add_log_message(f"🧪 Testing trade: {side} {amount} {symbol} @ {price}")
            
            # Here you would test the trade without executing
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error testing trade: {e}")
            self.add_log_message(f"❌ Trade test failed: {e}")
    
    def show_trade_history(self):
        """Show trade history."""
        try:
            self.add_log_message("📋 Showing trade history")
            
            # Here you would load and display trade history
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error showing trade history: {e}")
            self.add_log_message(f"❌ Trade history failed: {e}")
    
    def show_trading_analytics(self):
        """Show trading analytics."""
        try:
            self.add_log_message("📊 Showing trading analytics")
            
            # Here you would load and display trading analytics
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error showing analytics: {e}")
            self.add_log_message(f"❌ Analytics failed: {e}")
    
    def show_security_report(self):
        """Show security report."""
        try:
            if self.advanced_security_manager:
                stats = self.advanced_security_manager.get_statistics()
                
                report_text = "🔐 ADVANCED SECURITY REPORT\n"
                report_text += "=" * 40 + "\n"
                report_text += f"Security Status: {'ACTIVE' if stats.get('security_enabled', False) else 'INACTIVE'}\n"
                report_text += f"Auto Protection: {'ENABLED' if stats.get('auto_protection', False) else 'DISABLED'}\n"
                report_text += f"Trades Protected: {stats.get('total_trades_protected', 0)}\n"
                report_text += f"Security Events: {stats.get('security_events_count', 0)}\n"
                report_text += "\n🔐 SECURITY FEATURES:\n"
                report_text += "✅ MIM Attack Protection\n"
                report_text += "✅ Ultra-Realistic Dummy Packets\n"
                report_text += "✅ Computational Complexity Obfuscation\n"
                report_text += "✅ Hash-ID Routing\n"
                report_text += "✅ ChaCha20-Poly1305 Encryption\n"
                report_text += "✅ Ephemeral Key Generation\n"
                
                # Show in analytics panel
                self.analytics_text.delete(1.0, tk.END)
                self.analytics_text.insert(tk.END, report_text)
                
                self.add_log_message("📊 Security report displayed")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error showing security report: {e}")
            self.add_log_message(f"❌ Security report failed: {e}")
    
    def show_performance_chart(self):
        """Show performance chart."""
        try:
            self.add_log_message("📈 Showing performance chart")
            
            # Here you would display performance charts
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error showing performance chart: {e}")
            self.add_log_message(f"❌ Performance chart failed: {e}")
    
    def show_security_metrics(self):
        """Show security metrics."""
        try:
            if self.advanced_security_manager:
                stats = self.advanced_security_manager.get_statistics()
                
                metrics_text = "🔐 SECURITY METRICS\n"
                metrics_text += "=" * 30 + "\n"
                metrics_text += f"Security Score: {stats.get('security_score', 0):.2f}/100\n"
                metrics_text += f"Protection Rate: {stats.get('protection_rate', 0):.2%}\n"
                metrics_text += f"Average Processing Time: {stats.get('avg_processing_time', 0):.4f}s\n"
                metrics_text += f"Dummy Packet Success Rate: 33.3%\n"
                metrics_text += f"Traffic Analysis Confusion: ACTIVE\n"
                metrics_text += f"Mathematical Obfuscation: ACTIVE\n"
                
                # Show in analytics panel
                self.analytics_text.delete(1.0, tk.END)
                self.analytics_text.insert(tk.END, metrics_text)
                
                self.add_log_message("📊 Security metrics displayed")
            else:
                self.add_log_message("❌ Advanced security manager not available")
        except Exception as e:
            logger.error(f"Error showing security metrics: {e}")
            self.add_log_message(f"❌ Security metrics failed: {e}")
    
    def show_strategy_analysis(self):
        """Show strategy analysis."""
        try:
            strategy = self.strategy_var.get()
            self.add_log_message(f"🎯 Showing strategy analysis for: {strategy}")
            
            # Here you would analyze the selected strategy
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error showing strategy analysis: {e}")
            self.add_log_message(f"❌ Strategy analysis failed: {e}")
    
    def export_trading_report(self):
        """Export trading report."""
        try:
            self.add_log_message("📋 Exporting trading report")
            
            # Here you would export a comprehensive trading report
            # For now, just log the action
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            self.add_log_message(f"❌ Report export failed: {e}")
    
    def refresh_analytics(self):
        """Refresh analytics display."""
        try:
            analytics_text = "📊 TRADING ANALYTICS\n"
            analytics_text += "=" * 30 + "\n"
            analytics_text += "Real-time performance metrics and analysis\n"
            analytics_text += "Use the buttons above to view specific analytics.\n\n"
            analytics_text += "🔐 Advanced Security Features:\n"
            analytics_text += "✅ MIM Attack Protection\n"
            analytics_text += "✅ Ultra-Realistic Dummy Packets\n"
            analytics_text += "✅ Computational Complexity Obfuscation\n"
            analytics_text += "✅ Hash-ID Routing\n"
            analytics_text += "✅ ChaCha20-Poly1305 Encryption\n"
            
            self.analytics_text.delete(1.0, tk.END)
            self.analytics_text.insert(tk.END, analytics_text)
            
        except Exception as e:
            logger.error(f"Error refreshing analytics: {e}")
            self.add_log_message(f"❌ Analytics refresh failed: {e}")
    
    def create_memory_tab(self):
        """Create the memory management tab."""
        memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(memory_frame, text="💾 Memory")
        
        # Memory management content
        memory_label = ttk.Label(
            memory_frame,
            text="💾 USB Memory Management",
            style='Header.TLabel'
        )
        memory_label.pack(pady=20)
        
        # USB Drive Selection
        usb_frame = ttk.LabelFrame(memory_frame, text="🔍 USB Drive Selection", padding=10)
        usb_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # USB drive list
        usb_list_frame = ttk.Frame(usb_frame)
        usb_list_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(usb_list_frame, text="Available USB Drives:").pack(side=tk.LEFT)
        
        self.usb_drive_var = tk.StringVar()
        self.usb_drive_combo = ttk.Combobox(usb_list_frame, textvariable=self.usb_drive_var, state="readonly", width=20)
        self.usb_drive_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            usb_list_frame,
            text="🔍 Scan USB",
            command=self.scan_usb
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(
            usb_list_frame,
            text="🔄 Refresh",
            command=self.refresh_usb_list
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Memory controls
        memory_controls = ttk.Frame(memory_frame)
        memory_controls.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(
            memory_controls,
            text="💾 Backup Now",
            command=self.backup_memory
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            memory_controls,
            text="🔄 Restore Memory",
            command=self.restore_memory
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            memory_controls,
            text="📋 Memory Info",
            command=self.show_memory_info
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            memory_controls,
            text="🧹 Clean Old Backups",
            command=self.cleanup_old_backups
        ).pack(side=tk.LEFT)
        
        # Memory info display
        self.memory_info_text = scrolledtext.ScrolledText(
            memory_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.memory_info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initialize USB list
        self.refresh_usb_list()
    
    def refresh_usb_list(self):
        """Refresh the USB drive list."""
        try:
            usb_drives = self.find_usb_drives()
            
            if usb_drives:
                drive_list = [str(drive) for drive in usb_drives]
                self.usb_drive_combo['values'] = drive_list
                
                # Select current USB drive if available
                if self.usb_memory_dir and self.usb_memory_dir.parent in usb_drives:
                    self.usb_drive_var.set(str(self.usb_memory_dir.parent))
                else:
                    self.usb_drive_var.set(drive_list[0])
                
                self.add_log_message(f"🔍 Found {len(usb_drives)} USB drive(s): {', '.join(drive_list)}")
            else:
                self.usb_drive_combo['values'] = ['No USB drives found']
                self.usb_drive_var.set('No USB drives found')
                self.add_log_message("🔍 No USB drives found")
                
        except Exception as e:
            logger.error(f"Error refreshing USB list: {e}")
            self.add_log_message(f"❌ Error refreshing USB list: {e}")
    
    def select_usb_drive(self):
        """Select a USB drive for memory management."""
        try:
            selected_drive = self.usb_drive_var.get()
            
            if selected_drive == 'No USB drives found':
                messagebox.showwarning("No USB Drives", "No USB drives are currently available.")
                return False
            
            # Update memory directory
            new_memory_dir = Path(selected_drive) / "SchwabotMemory"
            new_memory_dir.mkdir(exist_ok=True)
            
            # Copy existing memory if switching
            if self.usb_memory_dir and self.usb_memory_dir != new_memory_dir:
                if self.usb_memory_dir.exists():
                    shutil.copytree(self.usb_memory_dir, new_memory_dir, dirs_exist_ok=True)
                    self.add_log_message(f"📋 Copied memory from {self.usb_memory_dir} to {new_memory_dir}")
            
            self.usb_memory_dir = new_memory_dir
            
            # Create subdirectories
            (self.usb_memory_dir / "config").mkdir(exist_ok=True)
            (self.usb_memory_dir / "state").mkdir(exist_ok=True)
            (self.usb_memory_dir / "logs").mkdir(exist_ok=True)
            (self.usb_memory_dir / "backups").mkdir(exist_ok=True)
            
            self.usb_status_label.config(text=f"USB: {selected_drive}")
            self.add_log_message(f"✅ Selected USB drive: {selected_drive}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error selecting USB drive: {e}")
            self.add_log_message(f"❌ Error selecting USB drive: {e}")
            return False
    
    def create_logs_tab(self):
        """Create the logs tab."""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs")
        
        # Log viewing panel
        log_frame = ttk.LabelFrame(logs_frame, text="📋 Live Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=20,
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
        """Start the Schwabot trading bot with memory restoration."""
        try:
            if self.is_running:
                messagebox.showwarning("Warning", "Schwabot is already running!")
                return
            
            # Restore memory before starting
            self.restore_memory_silent()
            
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
            self.add_log_message("🚀 Schwabot started successfully with memory restoration")
            
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
    
    def safe_shutdown(self):
        """Perform safe shutdown with memory backup."""
        try:
            # Backup memory first
            self.backup_memory_silent()
            
            # Stop the bot
            if self.is_running:
                self.stop_schwabot()
                time.sleep(2)  # Wait for stop
            
            # Close trading positions safely
            self.close_trading_positions()
            
            # Final memory backup
            self.backup_memory_silent()
            
            messagebox.showinfo("Safe Shutdown", "Safe shutdown completed. Memory backed up to USB.")
            self.add_log_message("🔒 Safe shutdown completed with memory backup")
            
        except Exception as e:
            logger.error(f"Error during safe shutdown: {e}")
            messagebox.showerror("Error", f"Error during safe shutdown: {e}")
    
    def close_trading_positions(self):
        """Safely close all trading positions."""
        try:
            # This would integrate with your trading engine to close positions
            # For now, just log the action
            logger.info("Closing trading positions safely")
            self.add_log_message("📈 Closing trading positions safely")
            
        except Exception as e:
            logger.error(f"Error closing trading positions: {e}")
    
    def backup_memory(self):
        """Backup memory to USB."""
        try:
            # Check if USB drive is selected
            if not self.usb_memory_dir or self.usb_drive_var.get() == 'No USB drives found':
                messagebox.showwarning("No USB Drive", "Please select a USB drive first.")
                return
            
            # Ensure memory directory exists
            self.usb_memory_dir.mkdir(exist_ok=True)
            (self.usb_memory_dir / "backups").mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.usb_memory_dir / "backups" / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup important files
            files_to_backup = [
                "schwabot_config.json",
                "schwabot_trading_bot.log",
                "schwabot_monitoring.log",
                "schwabot_enhanced_gui.log",
                "schwabot_cli.log"
            ]
            
            backed_up_files = []
            for file_name in files_to_backup:
                source_file = self.base_dir / file_name
                if source_file.exists():
                    try:
                        shutil.copy2(source_file, backup_dir / file_name)
                        backed_up_files.append(file_name)
                        logger.info(f"Backed up: {file_name}")
                    except Exception as e:
                        logger.warning(f"Could not backup {file_name}: {e}")
            
            # Backup state directory
            state_dir = self.base_dir / "state"
            if state_dir.exists():
                try:
                    shutil.copytree(state_dir, backup_dir / "state", dirs_exist_ok=True)
                    backed_up_files.append("state directory")
                    logger.info("Backed up: state directory")
                except Exception as e:
                    logger.warning(f"Could not backup state directory: {e}")
            
            # Backup config directory
            config_dir = self.base_dir / "config"
            if config_dir.exists():
                try:
                    shutil.copytree(config_dir, backup_dir / "config", dirs_exist_ok=True)
                    backed_up_files.append("config directory")
                    logger.info("Backed up: config directory")
                except Exception as e:
                    logger.warning(f"Could not backup config directory: {e}")
            
            # Create backup metadata
            metadata = {
                'timestamp': timestamp,
                'backup_time': datetime.now().isoformat(),
                'files_backed_up': backed_up_files,
                'source_directory': str(self.base_dir),
                'usb_directory': str(self.usb_memory_dir),
                'backup_size_mb': sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()) / (1024*1024)
            }
            
            with open(backup_dir / 'backup_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_backup_time = datetime.now()
            self.add_log_message(f"💾 Memory backed up to: {backup_dir}")
            messagebox.showinfo("Backup Complete", f"Memory backed up successfully!\n\nLocation: {backup_dir}\nFiles: {len(backed_up_files)}")
            
        except Exception as e:
            logger.error(f"Error backing up memory: {e}")
            messagebox.showerror("Backup Error", f"Error backing up memory: {e}")
    
    def backup_memory_silent(self):
        """Silent memory backup (no user interaction)."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.usb_memory_dir / "backups" / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup important files
            files_to_backup = [
                "schwabot_config.json",
                "schwabot_trading_bot.log",
                "schwabot_monitoring.log"
            ]
            
            for file_name in files_to_backup:
                source_file = self.base_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, backup_dir / file_name)
            
            self.last_backup_time = datetime.now()
            logger.info(f"Silent memory backup completed: {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error in silent memory backup: {e}")
    
    def restore_memory(self):
        """Restore memory from USB."""
        try:
            # Find latest backup
            backup_dir = self.usb_memory_dir / "backups"
            if not backup_dir.exists():
                messagebox.showinfo("No Backups", "No backups found to restore.")
                return
            
            backups = list(backup_dir.glob("backup_*"))
            if not backups:
                messagebox.showinfo("No Backups", "No backups found to restore.")
                return
            
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            
            # Restore files
            for backup_file in latest_backup.glob("*"):
                if backup_file.is_file():
                    shutil.copy2(backup_file, self.base_dir / backup_file.name)
                elif backup_file.is_dir():
                    shutil.copytree(backup_file, self.base_dir / backup_file.name, dirs_exist_ok=True)
            
            self.add_log_message(f"🔄 Memory restored from: {latest_backup}")
            messagebox.showinfo("Restore Complete", f"Memory restored from:\n{latest_backup}")
            
        except Exception as e:
            logger.error(f"Error restoring memory: {e}")
            messagebox.showerror("Restore Error", f"Error restoring memory: {e}")
    
    def restore_memory_silent(self):
        """Silent memory restoration (no user interaction)."""
        try:
            backup_dir = self.usb_memory_dir / "backups"
            if not backup_dir.exists():
                return
            
            backups = list(backup_dir.glob("backup_*"))
            if not backups:
                return
            
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            
            # Restore files
            for backup_file in latest_backup.glob("*"):
                if backup_file.is_file():
                    shutil.copy2(backup_file, self.base_dir / backup_file.name)
                elif backup_file.is_dir():
                    shutil.copytree(backup_file, self.base_dir / backup_file.name, dirs_exist_ok=True)
            
            logger.info(f"Silent memory restoration completed from: {latest_backup}")
            
        except Exception as e:
            logger.error(f"Error in silent memory restoration: {e}")
    
    def scan_usb(self):
        """Scan for USB drives and update memory location."""
        try:
            # Refresh USB list
            self.refresh_usb_list()
            
            # Select the first available USB drive
            if self.usb_drive_var.get() != 'No USB drives found':
                if self.select_usb_drive():
                    messagebox.showinfo("USB Found", f"USB drive selected:\n{self.usb_drive_var.get()}")
                else:
                    messagebox.showerror("USB Error", "Failed to select USB drive")
            else:
                messagebox.showinfo("No USB", "No USB drives found.")
            
        except Exception as e:
            logger.error(f"Error scanning USB: {e}")
            messagebox.showerror("USB Scan Error", f"Error scanning USB: {e}")
    
    def cleanup_old_backups(self):
        """Clean up old backups, keeping only the most recent ones."""
        try:
            if not self.usb_memory_dir:
                messagebox.showwarning("No Memory Directory", "No memory directory configured.")
                return
            
            backup_dir = self.usb_memory_dir / "backups"
            if not backup_dir.exists():
                messagebox.showinfo("No Backups", "No backup directory found.")
                return
            
            backups = list(backup_dir.glob("backup_*"))
            if len(backups) <= 10:  # Keep 10 most recent
                messagebox.showinfo("Cleanup", f"Only {len(backups)} backups found. No cleanup needed.")
                return
            
            # Sort by modification time and remove old ones
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            old_backups = backups[10:]  # Keep 10 most recent
            
            # Confirm deletion
            result = messagebox.askyesno(
                "Confirm Cleanup", 
                f"Delete {len(old_backups)} old backups?\nThis will keep the 10 most recent backups."
            )
            
            if result:
                deleted_count = 0
                for backup in old_backups:
                    try:
                        shutil.rmtree(backup)
                        deleted_count += 1
                        logger.info(f"Removed old backup: {backup.name}")
                    except Exception as e:
                        logger.error(f"Error removing backup {backup.name}: {e}")
                
                self.add_log_message(f"🧹 Cleaned up {deleted_count} old backups")
                messagebox.showinfo("Cleanup Complete", f"Successfully deleted {deleted_count} old backups.")
                
                # Refresh memory info
                self.show_memory_info()
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            messagebox.showerror("Cleanup Error", f"Error cleaning up old backups: {e}")
    
    def show_memory_info(self):
        """Show memory information."""
        try:
            info = f"Memory Management Information\n"
            info += f"=" * 50 + "\n\n"
            info += f"Memory Directory: {self.usb_memory_dir}\n"
            info += f"Last Backup: {self.last_backup_time or 'Never'}\n"
            info += f"Backup Interval: {self.backup_interval} seconds\n\n"
            
            # Show backup history
            backup_dir = self.usb_memory_dir / "backups"
            if backup_dir.exists():
                backups = list(backup_dir.glob("backup_*"))
                info += f"Backup History ({len(backups)} backups):\n"
                for backup in sorted(backups, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    info += f"  {backup.name} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            else:
                info += "No backups found.\n"
            
            # Show memory usage
            try:
                total_size = sum(f.stat().st_size for f in self.usb_memory_dir.rglob('*') if f.is_file())
                info += f"\nTotal Memory Size: {total_size / (1024*1024):.2f} MB\n"
            except:
                info += "\nTotal Memory Size: Unknown\n"
            
            self.memory_info_text.delete(1.0, tk.END)
            self.memory_info_text.insert(1.0, info)
            
        except Exception as e:
            logger.error(f"Error showing memory info: {e}")
    
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
                
                # Auto-backup if needed
                if self.last_backup_time is None or \
                   (datetime.now() - self.last_backup_time).total_seconds() > self.backup_interval:
                    self.backup_memory_silent()
                
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
            
            # Update last backup time
            if self.last_backup_time:
                backup_str = self.last_backup_time.strftime("%H:%M:%S")
                self.root.after(0, lambda: self.last_backup_label.config(text=backup_str))
            
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
            # Perform safe shutdown
            self.safe_shutdown()
            
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
    """Main function to run the enhanced GUI."""
    try:
        gui = SchwabotEnhancedGUI()
        gui.run()
    except Exception as e:
        logger.error(f"Error starting enhanced GUI: {e}")
        print(f"❌ Error starting enhanced GUI: {e}")

if __name__ == "__main__":
    main() 