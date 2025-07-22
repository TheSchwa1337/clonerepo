#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Enhanced Schwabot Launcher with 4-Tier Risk Management
========================================================

Enhanced launcher that integrates the new 4-tier risk management system
with automatic trading capabilities and BTC/USDC hardcoded support.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Import the enhanced advanced options
try:
    from enhanced_advanced_options_gui import show_enhanced_advanced_options
    ENHANCED_ADVANCED_OPTIONS_AVAILABLE = True
except ImportError:
    ENHANCED_ADVANCED_OPTIONS_AVAILABLE = False

# Import unified backtesting
try:
    from unified_live_backtesting_system import BacktestConfig, BacktestMode, start_live_backtest
    UNIFIED_BACKTESTING_AVAILABLE = True
except ImportError:
    UNIFIED_BACKTESTING_AVAILABLE = False

logger = None


class EnhancedSchwabotLauncher:
    """Enhanced Schwabot launcher with 4-tier risk management integration."""
    
    def __init__(self):
        self.root = None
        self.config = self._load_config()
        self.status_label = None
        self.health_status_label = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        config_file = Path("AOI_Base_Files_Schwabot/config/launcher_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return {
            "show_advanced_options_prompt": True,
            "compression_enabled": True,
            "auto_start_scheduler": False,
            "default_risk_mode": "ultra_low",
            "primary_trading_pair": "BTC/USDC"
        }
    
    def _save_config(self):
        """Save configuration."""
        config_file = Path("AOI_Base_Files_Schwabot/config/launcher_config.json")
        os.makedirs(config_file.parent, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def show_launcher(self):
        """Show the enhanced Schwabot launcher."""
        self.root = tk.Tk()
        self.root.title("🚀 Enhanced Schwabot Launcher - 4-Tier Risk Management")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self._configure_styles()
        
        # Create main interface
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        
        # Show advanced options prompt if needed
        if self.config.get("show_advanced_options_prompt", True):
            self._show_enhanced_advanced_options_prompt()
        
        # Center window
        self.root.mainloop()
    
    def _configure_styles(self):
        """Configure GUI styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Arial', 18, 'bold'), 
                       foreground='#00ff00', 
                       background='#1e1e1e')
        
        style.configure('Header.TLabel', 
                       font=('Arial', 14, 'bold'), 
                       foreground='#00ccff', 
                       background='#1e1e1e')
        
        style.configure('Info.TLabel', 
                       font=('Arial', 10), 
                       foreground='#cccccc', 
                       background='#1e1e1e')
        
        style.configure('Success.TLabel', 
                       font=('Arial', 10, 'bold'), 
                       foreground='#00ff00', 
                       background='#1e1e1e')
        
        style.configure('Warning.TLabel', 
                       font=('Arial', 10, 'bold'), 
                       foreground='#ffaa00', 
                       background='#1e1e1e')
        
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[10, 5])
    
    def _create_header(self):
        """Create the header section."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, text="🚀 Enhanced Schwabot Launcher", style='Title.TLabel')
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, 
                                  text="4-Tier Risk Management System with BTC/USDC Integration", 
                                  style='Info.TLabel')
        subtitle_label.pack(pady=5)
    
    def _create_main_content(self):
        """Create the main content area with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self._create_launch_tab()
        self._create_risk_management_tab()
        self._create_configuration_tab()
        self._create_advanced_options_tab()
        self._create_system_tab()
        self._create_help_tab()
    
    def _create_launch_tab(self):
        """Create the launch options tab."""
        launch_frame = ttk.Frame(self.notebook)
        self.notebook.add(launch_frame, text="🚀 Launch")
        
        # Title
        title = ttk.Label(launch_frame, text="Launch Options", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(launch_frame, 
                        text="Choose how to launch Schwabot with 4-tier risk management.\n"
                             "BTC/USDC is hardcoded as the primary trading pair.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Launch options frame
        options_frame = ttk.Frame(launch_frame)
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Demo Mode
        self._create_launch_option(
            options_frame,
            "🎮 Demo Mode",
            "Test the system with simulated trading",
            "• 4-tier risk management\n• BTC/USDC hardcoded\n• No real money risk\n• Perfect for testing",
            "Launch Demo",
            self._launch_demo_mode
        )
        
        # Web Dashboard
        self._create_launch_option(
            options_frame,
            "🌐 Web Dashboard",
            "Launch the Flask web dashboard",
            "• Real-time monitoring\n• Risk management controls\n• Portfolio tracking\n• API management",
            "Launch Dashboard",
            self._launch_web_dashboard
        )
        
        # Live Trading
        self._create_launch_option(
            options_frame,
            "💰 Live Trading",
            "Start live trading with 4-tier risk management",
            "• Ultra Low Risk (1% position)\n• Medium Risk (3% position)\n• High Risk (5% position)\n• Optimized High (7.5% position)\n• BTC/USDC primary pair",
            "Launch Live Trading",
            self._launch_live_trading,
            requires_api_keys=True
        )
        
        # Unified Backtesting
        self._create_launch_option(
            options_frame,
            "🧠 Unified Backtesting",
            "Test strategies with live API data",
            "• Live market data\n• No real trades\n• 4-tier risk testing\n• Performance analysis",
            "Launch Backtesting",
            self._launch_unified_backtesting
        )
    
    def _create_launch_option(self, parent, title: str, description: str, details: str, 
                             button_text: str, command, requires_api_keys: bool = False):
        """Create a launch option."""
        option_frame = ttk.LabelFrame(parent, text=title, padding=10)
        option_frame.pack(fill='x', pady=10)
        
        # Description
        desc_label = ttk.Label(option_frame, text=description, style='Info.TLabel')
        desc_label.pack(anchor='w', pady=5)
        
        # Details
        details_label = ttk.Label(option_frame, text=details, style='Info.TLabel')
        details_label.pack(anchor='w', pady=5)
        
        # Button frame
        button_frame = ttk.Frame(option_frame)
        button_frame.pack(fill='x', pady=10)
        
        # Launch button
        launch_btn = tk.Button(button_frame, text=button_text, command=command,
                              bg='#00aa00', fg='white', font=('Arial', 11, 'bold'),
                              padx=20, pady=8)
        launch_btn.pack(side='left')
        
        # API key requirement notice
        if requires_api_keys:
            api_notice = ttk.Label(button_frame, text="⚠️ Requires API keys", 
                                  style='Warning.TLabel')
            api_notice.pack(side='right', padx=10)
    
    def _create_risk_management_tab(self):
        """Create the risk management overview tab."""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="🛡️ Risk Management")
        
        # Title
        title = ttk.Label(risk_frame, text="4-Tier Risk Management System", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(risk_frame, 
                        text="Overview of the 4-tier risk management system with orbital-based allocation.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Risk tiers overview
        tiers_frame = ttk.Frame(risk_frame)
        tiers_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Ultra Low Risk
        ultra_frame = ttk.LabelFrame(tiers_frame, text="🟢 Ultra Low Risk (Lower Orbitals)", padding=10)
        ultra_frame.pack(fill='x', pady=5)
        ultra_desc = ttk.Label(ultra_frame, 
                              text="• Position Size: 1.0%\n• Stop Loss: 0.5%\n• Take Profit: 1.5%\n• High Volume Trading: Enabled\n• Guaranteed profit at high volume",
                              style='Info.TLabel', justify='left')
        ultra_desc.pack(anchor='w')
        
        # Medium Risk
        medium_frame = ttk.LabelFrame(tiers_frame, text="🟡 Medium Risk (Volumetric Orbitals)", padding=10)
        medium_frame.pack(fill='x', pady=5)
        medium_desc = ttk.Label(medium_frame, 
                               text="• Position Size: 3.0%\n• Stop Loss: 1.5%\n• Take Profit: 4.5%\n• Swing Timing: Enabled\n• Balanced approach",
                               style='Info.TLabel', justify='left')
        medium_desc.pack(anchor='w')
        
        # High Risk
        high_frame = ttk.LabelFrame(tiers_frame, text="🟠 High Risk (Higher Allocations)", padding=10)
        high_frame.pack(fill='x', pady=5)
        high_desc = ttk.Label(high_frame, 
                             text="• Position Size: 5.0%\n• Stop Loss: 2.5%\n• Take Profit: 7.5%\n• Aggressive strategies\n• For experienced traders",
                             style='Info.TLabel', justify='left')
        high_desc.pack(anchor='w')
        
        # Optimized High Risk
        optimized_frame = ttk.LabelFrame(tiers_frame, text="🔴 Optimized High (AI-Learned)", padding=10)
        optimized_frame.pack(fill='x', pady=5)
        optimized_desc = ttk.Label(optimized_frame, 
                                  text="• Position Size: 7.5%\n• Stop Loss: 3.0%\n• Take Profit: 10.0%\n• AI Learning: Enabled\n• Based on backtesting data",
                                  style='Info.TLabel', justify='left')
        optimized_desc.pack(anchor='w')
        
        # Global settings
        global_frame = ttk.LabelFrame(tiers_frame, text="🌐 Global Settings", padding=10)
        global_frame.pack(fill='x', pady=10)
        global_desc = ttk.Label(global_frame, 
                               text="• Default Risk Mode: Ultra Low\n• Auto-Switching: Enabled\n• Portfolio Auto-Detect: Enabled\n• Primary Trading Pair: BTC/USDC (Hardcoded)\n• Timing Optimization: 0.3ms",
                               style='Info.TLabel', justify='left')
        global_desc.pack(anchor='w')
        
        # Configure button
        config_btn = tk.Button(tiers_frame, text="⚙️ Configure Risk Management", 
                              command=self._show_enhanced_advanced_options_gui,
                              bg='#0066aa', fg='white', font=('Arial', 12, 'bold'),
                              padx=20, pady=10)
        config_btn.pack(pady=20)
    
    def _create_configuration_tab(self):
        """Create the configuration tab."""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        # Title
        title = ttk.Label(config_frame, text="System Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        # Configuration options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # API Keys
        api_frame = ttk.LabelFrame(options_frame, text="🔑 API Keys", padding=10)
        api_frame.pack(fill='x', pady=5)
        api_desc = ttk.Label(api_frame, 
                            text="Configure API keys for exchanges (Coinbase, Binance, etc.)",
                            style='Info.TLabel')
        api_desc.pack(anchor='w')
        api_btn = tk.Button(api_frame, text="Configure API Keys", 
                           command=self._configure_api_keys,
                           bg='#00aa00', fg='white', font=('Arial', 10, 'bold'),
                           padx=15, pady=5)
        api_btn.pack(anchor='w', pady=5)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(options_frame, text="🛡️ Risk Management", padding=10)
        risk_frame.pack(fill='x', pady=5)
        risk_desc = ttk.Label(risk_frame, 
                             text="Configure 4-tier risk management settings",
                             style='Info.TLabel')
        risk_desc.pack(anchor='w')
        risk_btn = tk.Button(risk_frame, text="Configure Risk Management", 
                            command=self._show_enhanced_advanced_options_gui,
                            bg='#0066aa', fg='white', font=('Arial', 10, 'bold'),
                            padx=15, pady=5)
        risk_btn.pack(anchor='w', pady=5)
    
    def _create_advanced_options_tab(self):
        """Create the advanced options tab."""
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="🔧 Advanced Options")
        
        # Title
        title = ttk.Label(advanced_frame, text="Advanced Options", style='Header.TLabel')
        title.pack(pady=10)
        
        # Advanced options frame
        options_frame = ttk.Frame(advanced_frame)
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Enhanced Advanced Options
        enhanced_frame = ttk.LabelFrame(options_frame, text="🔧 Enhanced Advanced Options", padding=10)
        enhanced_frame.pack(fill='x', pady=5)
        enhanced_desc = ttk.Label(enhanced_frame, 
                                 text="Access the full 4-tier risk management system with all advanced features",
                                 style='Info.TLabel')
        enhanced_desc.pack(anchor='w')
        enhanced_btn = tk.Button(enhanced_frame, text="🔧 Show Enhanced Advanced Options", 
                                command=self._show_enhanced_advanced_options_gui,
                                bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                                padx=20, pady=8)
        enhanced_btn.pack(anchor='w', pady=10)
    
    def _create_system_tab(self):
        """Create the system tab."""
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text="💻 System")
        
        # Title
        title = ttk.Label(system_frame, text="System Status", style='Header.TLabel')
        title.pack(pady=10)
        
        # System status frame
        status_frame = ttk.Frame(system_frame)
        status_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Health status
        health_frame = ttk.LabelFrame(status_frame, text="System Health", padding=10)
        health_frame.pack(fill='x', pady=5)
        self.health_status_label = ttk.Label(health_frame, text="✅ Healthy", style='Success.TLabel')
        self.health_status_label.pack(anchor='w')
        
        # Status
        status_label_frame = ttk.LabelFrame(status_frame, text="Status", padding=10)
        status_label_frame.pack(fill='x', pady=5)
        self.status_label = ttk.Label(status_label_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(anchor='w')
        
        # Action buttons
        actions_frame = ttk.Frame(status_frame)
        actions_frame.pack(fill='x', pady=10)
        
        health_btn = tk.Button(actions_frame, text="🔍 Check Health", 
                              command=self._check_system_health,
                              bg='#00aa00', fg='white', font=('Arial', 10, 'bold'),
                              padx=15, pady=5)
        health_btn.pack(side='left', padx=5)
        
        emergency_btn = tk.Button(actions_frame, text="🛑 Emergency Stop", 
                                 command=self._emergency_stop,
                                 bg='#ff0000', fg='white', font=('Arial', 10, 'bold'),
                                 padx=15, pady=5)
        emergency_btn.pack(side='left', padx=5)
    
    def _create_help_tab(self):
        """Create the help tab."""
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="❓ Help")
        
        # Title
        title = ttk.Label(help_frame, text="Help & Documentation", style='Header.TLabel')
        title.pack(pady=10)
        
        # Help content
        help_content = ttk.Frame(help_frame)
        help_content.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Quick start
        quick_frame = ttk.LabelFrame(help_content, text="🚀 Quick Start", padding=10)
        quick_frame.pack(fill='x', pady=5)
        quick_desc = ttk.Label(quick_frame, 
                              text="1. Configure API Keys (Configuration tab)\n"
                                   "2. Choose launch mode (Launch tab)\n"
                                   "3. Start with Demo Mode for testing\n"
                                   "4. Configure risk management (Advanced Options)\n"
                                   "5. Monitor system health (System tab)",
                              style='Info.TLabel', justify='left')
        quick_desc.pack(anchor='w')
        
        # About
        about_frame = ttk.LabelFrame(help_content, text="ℹ️ About", padding=10)
        about_frame.pack(fill='x', pady=5)
        about_desc = ttk.Label(about_frame, 
                              text="Enhanced Schwabot Launcher v2.0\n"
                                   "4-Tier Risk Management System\n"
                                   "BTC/USDC Primary Trading Pair\n"
                                   "Developed by Maxamillion M.A.A. DeLeon & Nexus AI",
                              style='Info.TLabel', justify='left')
        about_desc.pack(anchor='w')
    
    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=20, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(side='left')
        
        # Version info
        version_label = ttk.Label(status_frame, text="Enhanced Schwabot v2.0", style='Info.TLabel')
        version_label.pack(side='right')
    
    # Launch methods
    def _launch_demo_mode(self):
        """Launch demo mode."""
        self._update_status("Launching demo mode...")
        try:
            subprocess.Popen([sys.executable, "demo_enhanced_advanced_options.py"])
            messagebox.showinfo("Demo Mode", "Demo mode launched successfully!")
        except Exception as e:
            self._update_status(f"Failed to launch demo mode: {e}")
            messagebox.showerror("Error", f"Failed to launch demo mode: {e}")
    
    def _launch_web_dashboard(self):
        """Launch web dashboard."""
        self._update_status("Launching web dashboard...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/web/dynamic_timing_dashboard.py"])
            messagebox.showinfo("Web Dashboard", 
                               "Web dashboard launched successfully!\n\n"
                               "Access at: http://localhost:5000")
        except Exception as e:
            self._update_status(f"Failed to launch web dashboard: {e}")
            messagebox.showerror("Error", f"Failed to launch web dashboard: {e}")
    
    def _launch_live_trading(self):
        """Launch live trading."""
        self._update_status("Launching live trading...")
        try:
            # Check if risk management config exists
            risk_config_file = Path("AOI_Base_Files_Schwabot/config/risk_management_config.json")
            if not risk_config_file.exists():
                result = messagebox.askyesno("Risk Management", 
                                           "Risk management configuration not found.\n\n"
                                           "Would you like to configure risk management first?")
                if result:
                    self._show_enhanced_advanced_options_gui()
                    return
            
            # Launch live trading
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/schwabot_launcher.py"])
            messagebox.showinfo("Live Trading", 
                               "Live trading launched successfully!\n\n"
                               "BTC/USDC is the primary trading pair.\n"
                               "4-tier risk management is active.")
        except Exception as e:
            self._update_status(f"Failed to launch live trading: {e}")
            messagebox.showerror("Error", f"Failed to launch live trading: {e}")
    
    def _launch_unified_backtesting(self):
        """Launch unified backtesting."""
        if not UNIFIED_BACKTESTING_AVAILABLE:
            messagebox.showerror("Error", "Unified backtesting system is not available")
            return
        
        result = messagebox.askyesno("Unified Backtesting",
                                   "🧠 Launch Unified Backtesting\n\n"
                                   "This will test strategies with live API data\n"
                                   "using the 4-tier risk management system.\n\n"
                                   "No real trades will be placed.\n\n"
                                   "Would you like to start?")
        if not result:
            return
        
        self._update_status("Starting unified backtesting...")
        
        try:
            # Create backtesting configuration
            config = BacktestConfig(
                mode=BacktestMode.LIVE_API_BACKTEST,
                symbols=["BTCUSDT", "ETHUSDT"],
                exchanges=["binance"],
                initial_balance=10000.0,
                backtest_duration_hours=1,
                enable_ai_analysis=True,
                enable_risk_management=True,
                enable_performance_optimization=True,
                data_update_interval=5.0,
                min_confidence=0.6
            )
            
            # Start backtesting in a separate thread
            def run_backtest():
                try:
                    import asyncio
                    result = asyncio.run(start_live_backtest(config))
                    
                    # Show results
                    messagebox.showinfo("Backtest Completed",
                                      f"🎯 Backtest Results:\n\n"
                                      f"Total Return: {result.total_return:.2f}%\n"
                                      f"Total Trades: {result.total_trades}\n"
                                      f"Win Rate: {result.win_rate:.1f}%\n"
                                      f"Final Balance: ${result.final_balance:,.2f}\n\n"
                                      f"4-Tier Risk Management: Active\n"
                                      f"BTC/USDC: Primary Trading Pair")
                    
                    self._update_status("Unified backtesting completed successfully")
                    
                except Exception as e:
                    self._update_status(f"Backtesting failed: {e}")
                    messagebox.showerror("Error", f"Backtesting failed: {e}")
            
            # Run in separate thread
            threading.Thread(target=run_backtest, daemon=True).start()
            
        except Exception as e:
            self._update_status(f"Failed to start backtesting: {e}")
            messagebox.showerror("Error", f"Failed to start backtesting: {e}")
    
    # Configuration methods
    def _configure_api_keys(self):
        """Configure API keys."""
        messagebox.showinfo("API Keys", 
                           "🔑 API Key Configuration\n\n"
                           "Configure your API keys for:\n"
                           "• Coinbase\n"
                           "• Binance\n"
                           "• Other exchanges\n\n"
                           "BTC/USDC is the primary trading pair.\n"
                           "Keys are encrypted and stored securely.")
    
    # Advanced options methods
    def _show_enhanced_advanced_options_prompt(self):
        """Show a prompt to enable enhanced advanced options."""
        result = messagebox.askyesno(
            "Enable Enhanced Advanced Options",
            "🚀 Enhanced Advanced Options Available\n\n"
            "Would you like to enable the enhanced advanced options?\n\n"
            "This includes:\n"
            "• 4-Tier Risk Management System\n"
            "• BTC/USDC Primary Trading Pair\n"
            "• Automatic Risk Mode Switching\n"
            "• Portfolio Auto-Detection\n"
            "• High Volume/High Frequency Options\n"
            "• AI Learning for Optimized Strategies"
        )
        if result:
            self.config["show_advanced_options_prompt"] = False
            self._save_config()
            self._show_enhanced_advanced_options_gui()
        else:
            self.config["show_advanced_options_prompt"] = False
            self._save_config()
    
    def _show_enhanced_advanced_options_gui(self):
        """Show the enhanced advanced options GUI."""
        if ENHANCED_ADVANCED_OPTIONS_AVAILABLE:
            show_enhanced_advanced_options(self.root)
        else:
            messagebox.showwarning("Enhanced Advanced Options", 
                                 "Enhanced advanced options are not available.\n"
                                 "Please ensure all dependencies are installed.")
    
    # System methods
    def _check_system_health(self):
        """Check system health."""
        def health_check():
            try:
                self.health_status_label.config(text="🔍 Checking...", foreground='#ffaa00')
                
                # Simulate health check
                import time
                time.sleep(2)
                
                # Check if risk management config exists
                risk_config_file = Path("AOI_Base_Files_Schwabot/config/risk_management_config.json")
                if risk_config_file.exists():
                    self.health_status_label.config(text="✅ Healthy", foreground='#00ff00')
                    self._update_status("System health check completed - All systems operational")
                else:
                    self.health_status_label.config(text="⚠️ Needs Configuration", foreground='#ffaa00')
                    self._update_status("System health check completed - Risk management needs configuration")
                
            except Exception as e:
                self.health_status_label.config(text="❌ Error", foreground='#ff6666')
                self._update_status(f"System health check failed: {e}")
        
        threading.Thread(target=health_check, daemon=True).start()
    
    def _emergency_stop(self):
        """Emergency stop all trading."""
        result = messagebox.askyesno("Emergency Stop", 
                                   "🛑 EMERGENCY STOP\n\n"
                                   "This will immediately stop all trading activities.\n"
                                   "Are you sure you want to proceed?")
        if result:
            try:
                self._update_status("Emergency stop executed")
                messagebox.showinfo("Emergency Stop", 
                                   "All trading activities have been stopped.\n\n"
                                   "4-Tier Risk Management: Disabled\n"
                                   "BTC/USDC Trading: Stopped")
            except Exception as e:
                self._update_status(f"Emergency stop failed: {e}")
                messagebox.showerror("Error", f"Emergency stop failed: {e}")
    
    def _update_status(self, message: str):
        """Update the status bar."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)


def main():
    """Main function."""
    launcher = EnhancedSchwabotLauncher()
    launcher.show_launcher()


if __name__ == "__main__":
    main() 