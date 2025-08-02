#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Cross-Chain Mode GUI - Multi-Strategy Portfolio Interface
============================================================

Advanced GUI interface for the Cross-Chain Mode System featuring:
- Individual strategy toggle buttons with visual feedback
- Cross-chain creation and management
- Real-time status monitoring
- Shadow mode test suite interface
- USB memory status and operations
- Multi-computer synchronization controls
- Performance analytics and visualization

⚠️ SAFETY NOTICE: This system is for analysis and timing only.
    Real trading execution requires additional safety layers.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import cross-chain system
try:
    from cross_chain_mode_system import (
        CrossChainModeSystem, 
        StrategyType, 
        ChainType, 
        CrossChainExecutionMode,
        CROSS_CHAIN_SAFETY_CONFIG
    )
    CROSS_CHAIN_AVAILABLE = True
except ImportError:
    CROSS_CHAIN_AVAILABLE = False
    logging.warning("⚠️ Cross-Chain Mode System not available")

logger = logging.getLogger(__name__)

class CrossChainGUI:
    """Advanced GUI for Cross-Chain Mode System."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔗 Cross-Chain Mode System - Multi-Strategy Portfolio")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#1a1a1a")
        
        # Initialize cross-chain system
        if CROSS_CHAIN_AVAILABLE:
            self.cross_chain_system = CrossChainModeSystem()
        else:
            self.cross_chain_system = None
            messagebox.showerror("Error", "Cross-Chain Mode System not available")
            return
        
        # GUI state
        self.strategy_buttons: Dict[str, tk.Button] = {}
        self.chain_frames: Dict[str, ttk.Frame] = {}
        self.status_labels: Dict[str, tk.Label] = {}
        self.is_running = False
        
        # Setup GUI
        self._setup_gui()
        self._create_strategy_panel()
        self._create_chain_management_panel()
        self._create_status_panel()
        self._create_shadow_mode_panel()
        self._create_usb_memory_panel()
        self._create_performance_panel()
        
        # Start status update thread
        self._start_status_updates()
        
        logger.info("🔗 Cross-Chain Mode GUI initialized")
    
    def _setup_gui(self):
        """Setup the main GUI interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors for dark theme
        style.configure("TFrame", background="#1a1a1a")
        style.configure("TLabel", background="#1a1a1a", foreground="#ffffff")
        style.configure("TButton", background="#2d2d2d", foreground="#ffffff")
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Status.TLabel", font=("Arial", 10))
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        header_label = ttk.Label(
            main_container,
            text="🔗 Cross-Chain Mode System - Multi-Strategy Portfolio",
            style="Header.TLabel"
        )
        header_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
    
    def _create_strategy_panel(self):
        """Create the strategy management panel."""
        strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(strategy_frame, text="🎯 Strategy Management")
        
        # Strategy header
        strategy_header = ttk.Label(
            strategy_frame,
            text="🎯 Individual Strategy Toggle Controls",
            style="Header.TLabel"
        )
        strategy_header.pack(pady=20)
        
        # Strategy grid
        strategy_grid = ttk.Frame(strategy_frame)
        strategy_grid.pack(pady=20, padx=20)
        
        # Create strategy buttons
        strategies = [
            ("clock_mode_001", "🕐 Clock Mode", "Mechanical watchmaker timing"),
            ("ferris_ride_001", "🎡 Ferris Ride", "Ferris Ride looping strategy"),
            ("ghost_mode_001", "👻 Ghost Mode", "Ghost mode trading"),
            ("brain_mode_001", "🧠 Brain Mode", "Neural brain processing"),
            ("unified_backtest_001", "📊 Unified Backtest", "Unified backtesting system")
        ]
        
        for i, (strategy_id, name, description) in enumerate(strategies):
            row = i // 2
            col = i % 2
            
            # Strategy frame
            strategy_frame = ttk.LabelFrame(strategy_grid, text=name, padding=10)
            strategy_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Description
            desc_label = ttk.Label(strategy_frame, text=description, wraplength=200)
            desc_label.pack(pady=(0, 10))
            
            # Toggle button
            toggle_button = tk.Button(
                strategy_frame,
                text="⏹️ DISABLED",
                bg="#d32f2f",
                fg="white",
                font=("Arial", 10, "bold"),
                command=lambda sid=strategy_id: self._toggle_strategy(sid)
            )
            toggle_button.pack(pady=5)
            self.strategy_buttons[strategy_id] = toggle_button
            
            # Status label
            status_label = ttk.Label(
                strategy_frame,
                text="Status: Inactive",
                style="Status.TLabel"
            )
            status_label.pack(pady=5)
            self.status_labels[f"{strategy_id}_status"] = status_label
        
        # Configure grid weights
        strategy_grid.columnconfigure(0, weight=1)
        strategy_grid.columnconfigure(1, weight=1)
    
    def _create_chain_management_panel(self):
        """Create the cross-chain management panel."""
        chain_frame = ttk.Frame(self.notebook)
        self.notebook.add(chain_frame, text="🔗 Cross-Chain Management")
        
        # Chain header
        chain_header = ttk.Label(
            chain_frame,
            text="🔗 Cross-Chain Creation & Management",
            style="Header.TLabel"
        )
        chain_header.pack(pady=20)
        
        # Chain creation section
        creation_frame = ttk.LabelFrame(chain_frame, text="Create New Cross-Chain", padding=15)
        creation_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Chain ID input
        ttk.Label(creation_frame, text="Chain ID:").pack(anchor=tk.W)
        self.chain_id_var = tk.StringVar(value="dual_chain_001")
        chain_id_entry = ttk.Entry(creation_frame, textvariable=self.chain_id_var, width=30)
        chain_id_entry.pack(anchor=tk.W, pady=(0, 10))
        
        # Chain type selection
        ttk.Label(creation_frame, text="Chain Type:").pack(anchor=tk.W)
        self.chain_type_var = tk.StringVar(value="dual")
        chain_type_combo = ttk.Combobox(
            creation_frame,
            textvariable=self.chain_type_var,
            values=["dual", "triple", "quad", "custom"],
            state="readonly",
            width=20
        )
        chain_type_combo.pack(anchor=tk.W, pady=(0, 10))
        
        # Strategy selection
        ttk.Label(creation_frame, text="Select Strategies:").pack(anchor=tk.W)
        strategy_frame = ttk.Frame(creation_frame)
        strategy_frame.pack(anchor=tk.W, pady=(0, 10))
        
        self.strategy_vars = {}
        strategies = ["clock_mode_001", "ferris_ride_001", "ghost_mode_001", "brain_mode_001", "unified_backtest_001"]
        
        for i, strategy in enumerate(strategies):
            var = tk.BooleanVar()
            self.strategy_vars[strategy] = var
            cb = ttk.Checkbutton(strategy_frame, text=strategy, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=(0, 20))
        
        # Create chain button
        create_button = ttk.Button(
            creation_frame,
            text="🔗 Create Cross-Chain",
            command=self._create_cross_chain
        )
        create_button.pack(pady=10)
        
        # Active chains section
        active_frame = ttk.LabelFrame(chain_frame, text="Active Cross-Chains", padding=15)
        active_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Active chains list
        self.active_chains_text = scrolledtext.ScrolledText(
            active_frame,
            height=15,
            bg="#2d2d2d",
            fg="#ffffff",
            font=("Consolas", 10)
        )
        self.active_chains_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_status_panel(self):
        """Create the system status panel."""
        status_frame = ttk.Frame(self.notebook)
        self.notebook.add(status_frame, text="📊 System Status")
        
        # Status header
        status_header = ttk.Label(
            status_frame,
            text="📊 Real-Time System Status",
            style="Header.TLabel"
        )
        status_header.pack(pady=20)
        
        # Status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(pady=20, padx=20)
        
        # System status
        system_frame = ttk.LabelFrame(status_grid, text="System Status", padding=10)
        system_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.system_status_labels = {}
        status_items = [
            ("execution_mode", "Execution Mode:"),
            ("active_strategies", "Active Strategies:"),
            ("active_chains", "Active Chains:"),
            ("shadow_mode", "Shadow Mode:"),
            ("usb_memory", "USB Memory:"),
            ("kraken_connected", "Kraken API:")
        ]
        
        for i, (key, label) in enumerate(status_items):
            ttk.Label(system_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            status_label = ttk.Label(system_frame, text="Unknown", style="Status.TLabel")
            status_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            self.system_status_labels[key] = status_label
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(status_grid, text="Performance Metrics", padding=10)
        perf_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.performance_labels = {}
        perf_items = [
            ("memory_sync_ops", "Memory Sync Operations:"),
            ("cross_chain_trades", "Cross-Chain Trades:"),
            ("usb_writes", "USB Write Operations:"),
            ("sync_operations", "Sync Operations:")
        ]
        
        for i, (key, label) in enumerate(perf_items):
            ttk.Label(perf_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            perf_label = ttk.Label(perf_frame, text="0", style="Status.TLabel")
            perf_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            self.performance_labels[key] = perf_label
        
        # Configure grid weights
        status_grid.columnconfigure(0, weight=1)
        status_grid.columnconfigure(1, weight=1)
    
    def _create_shadow_mode_panel(self):
        """Create the shadow mode test suite panel."""
        shadow_frame = ttk.Frame(self.notebook)
        self.notebook.add(shadow_frame, text="🕵️ Shadow Mode")
        
        # Shadow mode header
        shadow_header = ttk.Label(
            shadow_frame,
            text="🕵️ Shadow Mode Test Suite",
            style="Header.TLabel"
        )
        shadow_header.pack(pady=20)
        
        # Shadow mode controls
        controls_frame = ttk.LabelFrame(shadow_frame, text="Shadow Mode Controls", padding=15)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Enable shadow mode button
        self.shadow_enable_button = ttk.Button(
            controls_frame,
            text="🕵️ Enable Shadow Mode",
            command=self._enable_shadow_mode
        )
        self.shadow_enable_button.pack(pady=10)
        
        # Shadow mode status
        self.shadow_status_label = ttk.Label(
            controls_frame,
            text="Status: Disabled",
            style="Status.TLabel"
        )
        self.shadow_status_label.pack(pady=5)
        
        # Test data display
        test_frame = ttk.LabelFrame(shadow_frame, text="Shadow Test Data", padding=15)
        test_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.shadow_data_text = scrolledtext.ScrolledText(
            test_frame,
            height=20,
            bg="#2d2d2d",
            fg="#ffffff",
            font=("Consolas", 9)
        )
        self.shadow_data_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_usb_memory_panel(self):
        """Create the USB memory management panel."""
        usb_frame = ttk.Frame(self.notebook)
        self.notebook.add(usb_frame, text="💾 USB Memory")
        
        # USB memory header
        usb_header = ttk.Label(
            usb_frame,
            text="💾 USB Memory Management",
            style="Header.TLabel"
        )
        usb_header.pack(pady=20)
        
        # USB controls
        controls_frame = ttk.LabelFrame(usb_frame, text="USB Memory Controls", padding=15)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # USB status
        self.usb_status_label = ttk.Label(
            controls_frame,
            text="Status: Checking USB drives...",
            style="Status.TLabel"
        )
        self.usb_status_label.pack(pady=5)
        
        # USB operations
        operations_frame = ttk.Frame(controls_frame)
        operations_frame.pack(pady=10)
        
        ttk.Button(
            operations_frame,
            text="📝 Force USB Write",
            command=self._force_usb_write
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            operations_frame,
            text="🔄 Refresh USB Status",
            command=self._refresh_usb_status
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            operations_frame,
            text="📊 USB Memory Stats",
            command=self._show_usb_stats
        ).pack(side=tk.LEFT)
        
        # USB memory log
        log_frame = ttk.LabelFrame(usb_frame, text="USB Memory Operations Log", padding=15)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.usb_log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            bg="#2d2d2d",
            fg="#ffffff",
            font=("Consolas", 9)
        )
        self.usb_log_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_performance_panel(self):
        """Create the performance analytics panel."""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="📈 Performance")
        
        # Performance header
        perf_header = ttk.Label(
            perf_frame,
            text="📈 Performance Analytics & Visualization",
            style="Header.TLabel"
        )
        perf_header.pack(pady=20)
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(perf_frame, text="Real-Time Performance Metrics", padding=15)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Performance grid
        perf_grid = ttk.Frame(metrics_frame)
        perf_grid.pack(fill=tk.X)
        
        self.analytics_labels = {}
        analytics_items = [
            ("total_trades", "Total Trades:"),
            ("successful_trades", "Successful Trades:"),
            ("win_rate", "Win Rate:"),
            ("total_profit", "Total Profit:"),
            ("avg_trade_profit", "Avg Trade Profit:"),
            ("max_drawdown", "Max Drawdown:")
        ]
        
        for i, (key, label) in enumerate(analytics_items):
            row = i // 2
            col = i % 2
            
            ttk.Label(perf_grid, text=label).grid(row=row, column=col*2, sticky=tk.W, pady=2, padx=(0, 10))
            analytics_label = ttk.Label(perf_grid, text="0", style="Status.TLabel")
            analytics_label.grid(row=row, column=col*2+1, sticky=tk.W, pady=2)
            self.analytics_labels[key] = analytics_label
        
        # Performance visualization
        viz_frame = ttk.LabelFrame(perf_frame, text="Performance Visualization", padding=15)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.performance_text = scrolledtext.ScrolledText(
            viz_frame,
            height=20,
            bg="#2d2d2d",
            fg="#ffffff",
            font=("Consolas", 9)
        )
        self.performance_text.pack(fill=tk.BOTH, expand=True)
    
    def _toggle_strategy(self, strategy_id: str):
        """Toggle a strategy on/off."""
        try:
            if not self.cross_chain_system:
                return
            
            # Get current state
            strategy = self.cross_chain_system.strategies.get(strategy_id)
            if not strategy:
                return
            
            # Toggle state
            new_state = not strategy.is_active
            success = self.cross_chain_system.toggle_strategy(strategy_id, new_state)
            
            if success:
                # Update button
                button = self.strategy_buttons.get(strategy_id)
                if button:
                    if new_state:
                        button.config(text="✅ ENABLED", bg="#388e3c")
                    else:
                        button.config(text="⏹️ DISABLED", bg="#d32f2f")
                
                # Update status label
                status_label = self.status_labels.get(f"{strategy_id}_status")
                if status_label:
                    status_label.config(text=f"Status: {'Active' if new_state else 'Inactive'}")
                
                logger.info(f"🔗 Strategy {strategy_id} {'enabled' if new_state else 'disabled'}")
            else:
                messagebox.showerror("Error", f"Failed to toggle strategy {strategy_id}")
                
        except Exception as e:
            logger.error(f"❌ Error toggling strategy {strategy_id}: {e}")
            messagebox.showerror("Error", f"Error toggling strategy: {e}")
    
    def _create_cross_chain(self):
        """Create a new cross-chain."""
        try:
            if not self.cross_chain_system:
                return
            
            # Get chain parameters
            chain_id = self.chain_id_var.get()
            chain_type_str = self.chain_type_var.get()
            
            # Convert chain type
            chain_type_map = {
                "dual": ChainType.DUAL,
                "triple": ChainType.TRIPLE,
                "quad": ChainType.QUAD,
                "custom": ChainType.CUSTOM
            }
            chain_type = chain_type_map.get(chain_type_str, ChainType.DUAL)
            
            # Get selected strategies
            selected_strategies = [
                strategy_id for strategy_id, var in self.strategy_vars.items()
                if var.get()
            ]
            
            if not selected_strategies:
                messagebox.showerror("Error", "Please select at least one strategy")
                return
            
            # Create cross-chain
            success = self.cross_chain_system.create_cross_chain(
                chain_id, chain_type, selected_strategies
            )
            
            if success:
                # Activate the chain
                activate_success = self.cross_chain_system.activate_cross_chain(chain_id)
                
                if activate_success:
                    messagebox.showinfo("Success", f"Cross-chain {chain_id} created and activated!")
                    self._update_active_chains()
                else:
                    messagebox.showwarning("Warning", f"Cross-chain {chain_id} created but failed to activate")
            else:
                messagebox.showerror("Error", f"Failed to create cross-chain {chain_id}")
                
        except Exception as e:
            logger.error(f"❌ Error creating cross-chain: {e}")
            messagebox.showerror("Error", f"Error creating cross-chain: {e}")
    
    def _update_active_chains(self):
        """Update the active chains display."""
        try:
            if not self.cross_chain_system:
                return
            
            # Clear current display
            self.active_chains_text.delete(1.0, tk.END)
            
            # Get active chains
            active_chains = [
                chain_id for chain_id in self.cross_chain_system.chains
                if self.cross_chain_system.chains[chain_id].is_active
            ]
            
            if not active_chains:
                self.active_chains_text.insert(tk.END, "No active cross-chains\n")
                return
            
            # Display active chains
            for chain_id in active_chains:
                chain = self.cross_chain_system.chains[chain_id]
                chain_info = f"🔗 {chain_id} ({chain.chain_type.value})\n"
                chain_info += f"   Strategies: {', '.join(chain.strategies)}\n"
                chain_info += f"   Sync Count: {chain.sync_count}\n"
                chain_info += f"   Last Sync: {datetime.fromtimestamp(chain.last_sync).strftime('%H:%M:%S')}\n"
                chain_info += f"   Chain Hash: {chain.chain_hash[:16]}...\n"
                chain_info += "-" * 50 + "\n"
                
                self.active_chains_text.insert(tk.END, chain_info)
                
        except Exception as e:
            logger.error(f"❌ Error updating active chains: {e}")
    
    def _enable_shadow_mode(self):
        """Enable shadow mode test suite."""
        try:
            if not self.cross_chain_system:
                return
            
            success = self.cross_chain_system.enable_shadow_mode()
            
            if success:
                self.shadow_enable_button.config(state="disabled")
                self.shadow_status_label.config(text="Status: Enabled")
                messagebox.showinfo("Success", "Shadow mode test suite enabled!")
            else:
                messagebox.showerror("Error", "Failed to enable shadow mode")
                
        except Exception as e:
            logger.error(f"❌ Error enabling shadow mode: {e}")
            messagebox.showerror("Error", f"Error enabling shadow mode: {e}")
    
    def _force_usb_write(self):
        """Force a USB memory write operation."""
        try:
            if not self.cross_chain_system:
                return
            
            # Add test data to USB queue
            test_data = {
                'type': 'manual_usb_write',
                'timestamp': datetime.now().isoformat(),
                'description': 'Manual USB write operation triggered from GUI'
            }
            
            self.cross_chain_system._queue_usb_write(test_data)
            
            # Update USB log
            self.usb_log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Manual USB write triggered\n")
            self.usb_log_text.see(tk.END)
            
            messagebox.showinfo("Success", "USB write operation queued!")
            
        except Exception as e:
            logger.error(f"❌ Error forcing USB write: {e}")
            messagebox.showerror("Error", f"Error forcing USB write: {e}")
    
    def _refresh_usb_status(self):
        """Refresh USB memory status."""
        try:
            if not self.cross_chain_system:
                return
            
            # Update USB status
            usb_enabled = self.cross_chain_system.usb_memory_enabled
            queue_size = len(self.cross_chain_system.usb_write_queue)
            
            status_text = f"Status: {'Enabled' if usb_enabled else 'Disabled'}"
            status_text += f" | Queue Size: {queue_size}"
            
            self.usb_status_label.config(text=status_text)
            
            # Update USB log
            self.usb_log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] USB status refreshed\n")
            self.usb_log_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"❌ Error refreshing USB status: {e}")
    
    def _show_usb_stats(self):
        """Show USB memory statistics."""
        try:
            if not self.cross_chain_system:
                return
            
            # Get USB stats
            queue_size = len(self.cross_chain_system.usb_write_queue)
            last_write = self.cross_chain_system.usb_last_write
            
            stats_text = f"USB Memory Statistics:\n"
            stats_text += f"Queue Size: {queue_size}\n"
            stats_text += f"Last Write: {datetime.fromtimestamp(last_write).strftime('%H:%M:%S') if last_write > 0 else 'Never'}\n"
            stats_text += f"Memory Enabled: {self.cross_chain_system.usb_memory_enabled}\n"
            
            messagebox.showinfo("USB Memory Stats", stats_text)
            
        except Exception as e:
            logger.error(f"❌ Error showing USB stats: {e}")
            messagebox.showerror("Error", f"Error showing USB stats: {e}")
    
    def _start_status_updates(self):
        """Start the status update thread."""
        def status_update_loop():
            while self.is_running:
                try:
                    self._update_system_status()
                    time.sleep(1.0)  # Update every second
                except Exception as e:
                    logger.error(f"❌ Error in status update loop: {e}")
                    time.sleep(5.0)
        
        self.is_running = True
        self.status_thread = threading.Thread(target=status_update_loop, daemon=True)
        self.status_thread.start()
    
    def _update_system_status(self):
        """Update system status display."""
        try:
            if not self.cross_chain_system:
                return
            
            # Get system status
            status = self.cross_chain_system.get_system_status()
            
            # Update system status labels
            self.system_status_labels["execution_mode"].config(text=status["execution_mode"])
            self.system_status_labels["active_strategies"].config(text=str(status["active_strategies"]))
            self.system_status_labels["active_chains"].config(text=str(status["active_chains"]))
            self.system_status_labels["shadow_mode"].config(text="Enabled" if status["shadow_mode_active"] else "Disabled")
            self.system_status_labels["usb_memory"].config(text="Enabled" if status["usb_memory_enabled"] else "Disabled")
            self.system_status_labels["kraken_connected"].config(text="Connected" if status["kraken_connected"] else "Disconnected")
            
            # Update performance labels
            self.performance_labels["memory_sync_ops"].config(text=str(status["memory_sync_operations"]))
            self.performance_labels["cross_chain_trades"].config(text=str(status["cross_chain_trades"]))
            
            # Update active chains
            self._update_active_chains()
            
        except Exception as e:
            logger.error(f"❌ Error updating system status: {e}")
    
    def run(self):
        """Run the GUI application."""
        try:
            logger.info("🔗 Starting Cross-Chain Mode GUI")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"❌ Error running GUI: {e}")
        finally:
            self.is_running = False
            logger.info("🔗 Cross-Chain Mode GUI stopped")

def main():
    """Main function to run the Cross-Chain Mode GUI."""
    try:
        if not CROSS_CHAIN_AVAILABLE:
            print("❌ Cross-Chain Mode System not available")
            return
        
        gui = CrossChainGUI()
        gui.run()
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")

if __name__ == "__main__":
    main() 