#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 ADVANCED SECURITY GUI - ULTRA-REALISTIC DUMMY PACKET SYSTEM
==============================================================

Developed by Maxamillion M.A.A. DeLeon screen/pen name TheSchwa1337 ("The Schwa") & Nexus AI
– Recursive Systems Architects | Authors of Ω-B-Γ Logic & Alpha Encryption Protocol

This module provides a modern GUI interface for the advanced security manager with
real-time monitoring, statistics, and control capabilities.
"""

import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class AdvancedSecurityGUI:
    """
    🔐 Advanced Security GUI
    
    Modern GUI interface for ultra-realistic dummy packet security system.
    """
    
    def __init__(self, security_manager):
        """Initialize the GUI."""
        self.security_manager = security_manager
        self.root = None
        self.auto_refresh = True
        self.refresh_thread = None
        
        # Data storage for charts
        self.security_scores = []
        self.processing_times = []
        self.timestamps = []
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components."""
        self.root = tk.Tk()
        self.root.title("🔐 Advanced Security Manager - Ultra-Realistic Dummy Packet System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self.setup_style()
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_control_tab()
        self.create_statistics_tab()
        self.create_events_tab()
        self.create_config_tab()
        
        # Start auto-refresh
        self.start_auto_refresh()
        
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
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='#00ff00')
        style.configure('Warning.TLabel', foreground='#ffff00')
        style.configure('Error.TLabel', foreground='#ff0000')
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Header
        header_label = ttk.Label(dashboard_frame, text="🔐 Advanced Security Dashboard", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.Frame(dashboard_frame)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Security status
        self.security_status_label = ttk.Label(status_frame, text="Security: ENABLED", style='Success.TLabel')
        self.security_status_label.pack(side=tk.LEFT, padx=10)
        
        # Auto protection status
        self.auto_protection_label = ttk.Label(status_frame, text="Auto Protection: ON", style='Success.TLabel')
        self.auto_protection_label.pack(side=tk.LEFT, padx=10)
        
        # Logical protection status
        self.logical_protection_label = ttk.Label(status_frame, text="Logical Protection: ON", style='Success.TLabel')
        self.logical_protection_label.pack(side=tk.LEFT, padx=10)
        
        # Statistics frame
        stats_frame = ttk.Frame(dashboard_frame)
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Create statistics widgets
        self.create_statistics_widgets(stats_frame)
        
        # Charts frame
        if MATPLOTLIB_AVAILABLE:
            charts_frame = ttk.Frame(dashboard_frame)
            charts_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            self.create_charts(charts_frame)
        
        # Quick actions frame
        actions_frame = ttk.Frame(dashboard_frame)
        actions_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Quick action buttons
        ttk.Button(actions_frame, text="🔐 Enable Security", command=self.enable_security).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="⚠️ Disable Security", command=self.disable_security).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="🔄 Toggle Auto Protection", command=self.toggle_auto_protection).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="📊 Refresh", command=self.refresh_dashboard).pack(side=tk.LEFT, padx=5)
    
    def create_statistics_widgets(self, parent):
        """Create statistics display widgets."""
        # Create a frame for statistics
        stats_container = ttk.Frame(parent)
        stats_container.pack(fill=tk.X)
        
        # Statistics labels
        self.total_trades_label = ttk.Label(stats_container, text="Total Trades Protected: 0")
        self.total_trades_label.pack(side=tk.LEFT, padx=10)
        
        self.security_events_label = ttk.Label(stats_container, text="Security Events: 0")
        self.security_events_label.pack(side=tk.LEFT, padx=10)
        
        self.avg_security_score_label = ttk.Label(stats_container, text="Avg Security Score: 0.00")
        self.avg_security_score_label.pack(side=tk.LEFT, padx=10)
        
        self.avg_processing_time_label = ttk.Label(stats_container, text="Avg Processing Time: 0.0000s")
        self.avg_processing_time_label.pack(side=tk.LEFT, padx=10)
    
    def create_charts(self, parent):
        """Create real-time charts."""
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 6), facecolor='#2d2d2d')
        
        # Security score chart
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title('Security Score Over Time', color='white')
        self.ax1.set_facecolor('#2d2d2d')
        self.ax1.tick_params(colors='white')
        
        # Processing time chart
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.set_title('Processing Time Over Time', color='white')
        self.ax2.set_facecolor('#2d2d2d')
        self.ax2.tick_params(colors='white')
        
        # Dummy packet count chart
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax3.set_title('Dummy Packets Generated', color='white')
        self.ax3.set_facecolor('#2d2d2d')
        self.ax3.tick_params(colors='white')
        
        # Security events chart
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title('Security Events', color='white')
        self.ax4.set_facecolor('#2d2d2d')
        self.ax4.tick_params(colors='white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_control_tab(self):
        """Create the control tab."""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="🔧 Control")
        
        # Header
        header_label = ttk.Label(control_frame, text="🔧 Security Control Panel", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Control buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(pady=20)
        
        # Security control buttons
        ttk.Button(buttons_frame, text="🔐 Enable Security Protection", 
                  command=self.enable_security, width=30).pack(pady=5)
        ttk.Button(buttons_frame, text="⚠️ Disable Security Protection", 
                  command=self.disable_security, width=30).pack(pady=5)
        ttk.Button(buttons_frame, text="🔄 Toggle Auto Protection", 
                  command=self.toggle_auto_protection, width=30).pack(pady=5)
        
        # Test trade protection frame
        test_frame = ttk.LabelFrame(control_frame, text="🧪 Test Trade Protection")
        test_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Test trade inputs
        input_frame = ttk.Frame(test_frame)
        input_frame.pack(pady=10)
        
        ttk.Label(input_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="BTC/USDC")
        ttk.Entry(input_frame, textvariable=self.symbol_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Side:").grid(row=0, column=2, padx=5, pady=5)
        self.side_var = tk.StringVar(value="buy")
        ttk.Combobox(input_frame, textvariable=self.side_var, values=["buy", "sell"]).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Amount:").grid(row=1, column=0, padx=5, pady=5)
        self.amount_var = tk.StringVar(value="0.1")
        ttk.Entry(input_frame, textvariable=self.amount_var).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Price:").grid(row=1, column=2, padx=5, pady=5)
        self.price_var = tk.StringVar(value="50000.0")
        ttk.Entry(input_frame, textvariable=self.price_var).grid(row=1, column=3, padx=5, pady=5)
        
        # Test button
        ttk.Button(test_frame, text="🧪 Test Trade Protection", 
                  command=self.test_trade_protection).pack(pady=10)
        
        # Test results
        self.test_result_text = tk.Text(test_frame, height=10, width=80, bg='#1e1e1e', fg='#ffffff')
        self.test_result_text.pack(pady=10)
    
    def create_statistics_tab(self):
        """Create the statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📊 Statistics")
        
        # Header
        header_label = ttk.Label(stats_frame, text="📊 Detailed Statistics", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Statistics treeview
        self.stats_tree = ttk.Treeview(stats_frame, columns=('Metric', 'Value'), show='headings')
        self.stats_tree.heading('Metric', text='Metric')
        self.stats_tree.heading('Value', text='Value')
        self.stats_tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Refresh button
        ttk.Button(stats_frame, text="🔄 Refresh Statistics", 
                  command=self.refresh_statistics).pack(pady=10)
    
    def create_events_tab(self):
        """Create the events tab."""
        events_frame = ttk.Frame(self.notebook)
        self.notebook.add(events_frame, text="📋 Events")
        
        # Header
        header_label = ttk.Label(events_frame, text="📋 Security Events Log", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Events treeview
        self.events_tree = ttk.Treeview(events_frame, columns=('Timestamp', 'Event', 'Details'), show='headings')
        self.events_tree.heading('Timestamp', text='Timestamp')
        self.events_tree.heading('Event', text='Event')
        self.events_tree.heading('Details', text='Details')
        self.events_tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Refresh button
        ttk.Button(events_frame, text="🔄 Refresh Events", 
                  command=self.refresh_events).pack(pady=10)
    
    def create_config_tab(self):
        """Create the configuration tab."""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Config")
        
        # Header
        header_label = ttk.Label(config_frame, text="⚙️ Configuration", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Configuration buttons
        buttons_frame = ttk.Frame(config_frame)
        buttons_frame.pack(pady=20)
        
        ttk.Button(buttons_frame, text="📤 Export Configuration", 
                  command=self.export_config).pack(pady=5)
        ttk.Button(buttons_frame, text="📥 Import Configuration", 
                  command=self.import_config).pack(pady=5)
        ttk.Button(buttons_frame, text="🔄 Reset to Defaults", 
                  command=self.reset_config).pack(pady=5)
        
        # Configuration display
        config_display_frame = ttk.LabelFrame(config_frame, text="Current Configuration")
        config_display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.config_text = tk.Text(config_display_frame, bg='#1e1e1e', fg='#ffffff')
        self.config_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Refresh config button
        ttk.Button(config_frame, text="🔄 Refresh Configuration", 
                  command=self.refresh_config).pack(pady=10)
    
    def enable_security(self):
        """Enable security protection."""
        if self.security_manager.enable_security():
            messagebox.showinfo("Success", "🔐 Security Protection ENABLED")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Error", "❌ Failed to enable security protection")
    
    def disable_security(self):
        """Disable security protection."""
        if self.security_manager.disable_security():
            messagebox.showwarning("Warning", "⚠️ Security Protection DISABLED")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Error", "❌ Failed to disable security protection")
    
    def toggle_auto_protection(self):
        """Toggle auto protection."""
        if self.security_manager.toggle_auto_protection():
            status = "ENABLED" if self.security_manager.auto_protection else "DISABLED"
            messagebox.showinfo("Success", f"🔄 Auto Protection {status}")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Error", "❌ Failed to toggle auto protection")
    
    def test_trade_protection(self):
        """Test trade protection."""
        try:
            trade_data = {
                'symbol': self.symbol_var.get(),
                'side': self.side_var.get(),
                'amount': float(self.amount_var.get()),
                'price': float(self.price_var.get()),
                'exchange': 'coinbase',
                'strategy_id': 'ferris_ride_001',
                'user_id': 'schwa_1337',
                'timestamp': time.time()
            }
            
            result = self.security_manager.protect_trade(trade_data)
            
            # Clear previous results
            self.test_result_text.delete(1.0, tk.END)
            
            if result['success'] and result['protected']:
                secure_result = result['secure_result']
                
                # Display results
                self.test_result_text.insert(tk.END, f"✅ Trade Protection Test SUCCESSFUL!\n\n")
                self.test_result_text.insert(tk.END, f"📦 Trade Data:\n")
                self.test_result_text.insert(tk.END, f"   Symbol: {trade_data['symbol']}\n")
                self.test_result_text.insert(tk.END, f"   Side: {trade_data['side']}\n")
                self.test_result_text.insert(tk.END, f"   Amount: {trade_data['amount']}\n")
                self.test_result_text.insert(tk.END, f"   Price: ${trade_data['price']:,.2f}\n\n")
                
                self.test_result_text.insert(tk.END, f"🔐 Security Results:\n")
                self.test_result_text.insert(tk.END, f"   Security Score: {secure_result.security_score:.2f}/100\n")
                self.test_result_text.insert(tk.END, f"   Processing Time: {secure_result.processing_time:.4f}s\n")
                self.test_result_text.insert(tk.END, f"   Dummy Packets: {len(secure_result.dummy_packets)}\n")
                self.test_result_text.insert(tk.END, f"   Key ID: {secure_result.key_id}\n")
                self.test_result_text.insert(tk.END, f"   Hash ID: {secure_result.metadata.get('hash_id', 'N/A')}\n\n")
                
                self.test_result_text.insert(tk.END, f"🎭 Dummy Packet Details:\n")
                for i, dummy in enumerate(secure_result.dummy_packets):
                    self.test_result_text.insert(tk.END, f"   Dummy {i+1}:\n")
                    self.test_result_text.insert(tk.END, f"     Key ID: {dummy['key_id']}\n")
                    self.test_result_text.insert(tk.END, f"     Hash ID: {dummy['hash_id']}\n")
                    self.test_result_text.insert(tk.END, f"     Pseudo Meta Tag: {dummy['pseudo_meta_tag']}\n")
                    self.test_result_text.insert(tk.END, f"     False Run ID: {dummy['false_run_id']}\n\n")
                
                # Update charts
                self.update_charts(secure_result.security_score, secure_result.processing_time)
                
            else:
                self.test_result_text.insert(tk.END, f"❌ Trade Protection Test FAILED!\n")
                self.test_result_text.insert(tk.END, f"Error: {result.get('error', 'Unknown error')}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"❌ Test failed: {str(e)}")
    
    def refresh_dashboard(self):
        """Refresh dashboard data."""
        stats = self.security_manager.get_statistics()
        
        # Update status labels
        security_status = "ENABLED" if stats['security_enabled'] else "DISABLED"
        security_style = 'Success.TLabel' if stats['security_enabled'] else 'Error.TLabel'
        self.security_status_label.config(text=f"Security: {security_status}", style=security_style)
        
        auto_status = "ON" if stats['auto_protection'] else "OFF"
        auto_style = 'Success.TLabel' if stats['auto_protection'] else 'Warning.TLabel'
        self.auto_protection_label.config(text=f"Auto Protection: {auto_status}", style=auto_style)
        
        logical_status = "ON" if stats['logical_protection'] else "OFF"
        logical_style = 'Success.TLabel' if stats['logical_protection'] else 'Warning.TLabel'
        self.logical_protection_label.config(text=f"Logical Protection: {logical_status}", style=logical_style)
        
        # Update statistics
        self.total_trades_label.config(text=f"Total Trades Protected: {stats['total_trades_protected']}")
        self.security_events_label.config(text=f"Security Events: {stats['security_events_count']}")
        
        # Update integration statistics
        integration_stats = stats['integration_status']['statistics']
        self.avg_security_score_label.config(text=f"Avg Security Score: {integration_stats['average_security_score']:.2f}")
        self.avg_processing_time_label.config(text=f"Avg Processing Time: {integration_stats['average_processing_time']:.4f}s")
    
    def refresh_statistics(self):
        """Refresh statistics tab."""
        stats = self.security_manager.get_statistics()
        
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add statistics
        self.stats_tree.insert('', 'end', values=('Security Enabled', str(stats['security_enabled'])))
        self.stats_tree.insert('', 'end', values=('Auto Protection', str(stats['auto_protection'])))
        self.stats_tree.insert('', 'end', values=('Logical Protection', str(stats['logical_protection'])))
        self.stats_tree.insert('', 'end', values=('Total Trades Protected', str(stats['total_trades_protected'])))
        self.stats_tree.insert('', 'end', values=('Security Events', str(stats['security_events_count'])))
        
        # Add integration statistics
        integration_stats = stats['integration_status']['statistics']
        self.stats_tree.insert('', 'end', values=('Total Trades Secured', str(integration_stats['total_trades_secured'])))
        self.stats_tree.insert('', 'end', values=('Success Rate', f"{integration_stats['success_rate']:.2%}"))
        self.stats_tree.insert('', 'end', values=('Average Security Score', f"{integration_stats['average_security_score']:.2f}"))
        self.stats_tree.insert('', 'end', values=('Average Processing Time', f"{integration_stats['average_processing_time']:.4f}s"))
        
        # Add secure handler statistics
        handler_status = stats['secure_handler_status']
        self.stats_tree.insert('', 'end', values=('Key Pool Size', str(handler_status['key_pool_size'])))
        self.stats_tree.insert('', 'end', values=('Cryptography Available', str(handler_status['cryptography_available'])))
    
    def refresh_events(self):
        """Refresh events tab."""
        events = self.security_manager.get_security_events(50)
        
        # Clear existing items
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        
        # Add events
        for event in reversed(events):  # Show newest first
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['timestamp']))
            event_type = event['event_type']
            details = json.dumps(event['details']) if event['details'] else ''
            
            self.events_tree.insert('', 'end', values=(timestamp, event_type, details))
    
    def refresh_config(self):
        """Refresh configuration tab."""
        config = self.security_manager.config
        
        # Clear existing text
        self.config_text.delete(1.0, tk.END)
        
        # Display configuration
        self.config_text.insert(tk.END, json.dumps(config, indent=2))
    
    def export_config(self):
        """Export configuration."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            if self.security_manager.export_config(filename):
                messagebox.showinfo("Success", f"✅ Configuration exported to {filename}")
            else:
                messagebox.showerror("Error", f"❌ Failed to export configuration")
    
    def import_config(self):
        """Import configuration."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            if self.security_manager.import_config(filename):
                messagebox.showinfo("Success", f"✅ Configuration imported from {filename}")
                self.refresh_config()
            else:
                messagebox.showerror("Error", f"❌ Failed to import configuration")
    
    def reset_config(self):
        """Reset configuration to defaults."""
        if messagebox.askyesno("Confirm", "Are you sure you want to reset to default configuration?"):
            self.security_manager.config = self.security_manager._default_config()
            messagebox.showinfo("Success", "✅ Configuration reset to defaults")
            self.refresh_config()
    
    def update_charts(self, security_score: float, processing_time: float):
        """Update real-time charts."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        current_time = time.time()
        
        # Add data points
        self.security_scores.append(security_score)
        self.processing_times.append(processing_time)
        self.timestamps.append(current_time)
        
        # Keep only last 50 points
        if len(self.security_scores) > 50:
            self.security_scores = self.security_scores[-50:]
            self.processing_times = self.processing_times[-50:]
            self.timestamps = self.timestamps[-50:]
        
        # Update charts
        self.ax1.clear()
        self.ax1.plot(self.timestamps, self.security_scores, 'g-')
        self.ax1.set_title('Security Score Over Time', color='white')
        self.ax1.set_facecolor('#2d2d2d')
        self.ax1.tick_params(colors='white')
        
        self.ax2.clear()
        self.ax2.plot(self.timestamps, self.processing_times, 'b-')
        self.ax2.set_title('Processing Time Over Time', color='white')
        self.ax2.set_facecolor('#2d2d2d')
        self.ax2.tick_params(colors='white')
        
        # Update dummy packet count (simulated)
        dummy_counts = [2] * len(self.timestamps)  # Always 2 dummy packets
        self.ax3.clear()
        self.ax3.bar(range(len(dummy_counts)), dummy_counts, color='orange')
        self.ax3.set_title('Dummy Packets Generated', color='white')
        self.ax3.set_facecolor('#2d2d2d')
        self.ax3.tick_params(colors='white')
        
        # Update security events (simulated)
        event_counts = list(range(1, len(self.timestamps) + 1))
        self.ax4.clear()
        self.ax4.plot(self.timestamps, event_counts, 'r-')
        self.ax4.set_title('Security Events', color='white')
        self.ax4.set_facecolor('#2d2d2d')
        self.ax4.tick_params(colors='white')
        
        # Redraw canvas
        self.canvas.draw()
    
    def start_auto_refresh(self):
        """Start auto-refresh thread."""
        def auto_refresh_loop():
            while self.auto_refresh and self.root:
                try:
                    self.root.after(0, self.refresh_dashboard)
                    time.sleep(5)  # Refresh every 5 seconds
                except:
                    break
        
        self.refresh_thread = threading.Thread(target=auto_refresh_loop, daemon=True)
        self.refresh_thread.start()
    
    def on_closing(self):
        """Handle window closing."""
        self.auto_refresh = False
        if self.root:
            self.root.destroy()
    
    def run(self):
        """Run the GUI."""
        if self.root:
            self.root.mainloop() 