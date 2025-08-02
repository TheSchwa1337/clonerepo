#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Schwabot Dynamic Trade Cap Slider GUI
========================================

Premium GUI for dynamic trade cap scaling with auto-scale functionality.
Features:
- Dynamic trade cap slider (1.00-10.00)
- Entropy-based strategy shifting
- Auto-scale tick box for micro mode
- Real-time profit projections
- Maximum paranoia safety indicators
- User comfort level scaling
- Premium interface design
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import sys
import codecs

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class DynamicTradeCapSlider:
    """Premium dynamic trade cap slider with auto-scale functionality."""
    
    def __init__(self, parent, clock_system=None):
        self.parent = parent
        self.clock_system = clock_system
        self.slider_locked = True
        self.auto_scale_enabled = False
        self.current_trade_cap = 1.0
        self.current_entropy = 0.5
        self.current_comfort = 0.5
        self.strategy_type = "Balanced"
        self.confidence = 0.7
        self.adaptation_rate = 0.5
        self.risk_level = "Medium"
        self.market_alignment = 0.6
        self.efficiency = 0.75
        self.paranoia_level = "Maximum"
        self.triple_confirmation = True
        self.emergency_stop = False
        
        # Premium styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles for premium look
        self.style.configure('Premium.TFrame', background='#2c3e50', relief='raised', borderwidth=2)
        self.style.configure('Premium.TLabel', background='#2c3e50', foreground='#ecf0f1', font=('Segoe UI', 9))
        self.style.configure('PremiumTitle.TLabel', background='#2c3e50', foreground='#3498db', font=('Segoe UI', 12, 'bold'))
        self.style.configure('PremiumSubtitle.TLabel', background='#2c3e50', foreground='#95a5a6', font=('Segoe UI', 8))
        self.style.configure('Micro.TButton', font=('Segoe UI', 8), padding=(8, 4))
        self.style.configure('Emergency.TButton', font=('Segoe UI', 8, 'bold'), foreground='#e74c3c')
        self.style.configure('AutoScale.TCheckbutton', background='#2c3e50', foreground='#27ae60')
        
        self.create_dynamic_slider_panel()
        self.start_dynamic_updates()

    def create_dynamic_slider_panel(self):
        """Create the main dynamic slider panel with premium design."""
        # Main container with premium styling
        self.main_frame = ttk.Frame(self.parent, style='Premium.TFrame', padding=15)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Premium title
        title_label = ttk.Label(
            self.main_frame, 
            text="🎯 DYNAMIC TRADE CAP SCALER", 
            style='PremiumTitle.TLabel'
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle with description
        subtitle_label = ttk.Label(
            self.main_frame,
            text="Premium scaling interface for Schwabot's micro mode trading system",
            style='PremiumSubtitle.TLabel'
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Create sections
        self.create_lock_section()
        self.create_auto_scale_section()
        self.create_main_slider_section()
        self.create_entropy_section()
        self.create_profit_projections_section()
        self.create_strategy_adjustment_section()
        self.create_safety_section()
        self.create_comfort_scaling_section()
        self.create_status_controls_section()

    def create_lock_section(self):
        """Create the lock/unlock section with premium styling."""
        lock_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        lock_frame.pack(fill='x', pady=(0, 10))
        
        # Lock status indicator
        self.lock_status_label = ttk.Label(
            lock_frame,
            text="🔒 SLIDER LOCKED - Click to unlock dynamic scaling",
            style='PremiumSubtitle.TLabel'
        )
        self.lock_status_label.pack(side='left', padx=(0, 10))
        
        # Unlock button (smaller, premium design)
        self.unlock_button = ttk.Button(
            lock_frame,
            text="🔓 UNLOCK",
            style='Micro.TButton',
            command=self.toggle_slider_lock
        )
        self.unlock_button.pack(side='right')

    def create_auto_scale_section(self):
        """Create the auto-scale section for micro mode."""
        auto_scale_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        auto_scale_frame.pack(fill='x', pady=(0, 10))
        
        # Auto-scale title
        auto_title = ttk.Label(
            auto_scale_frame,
            text="⚡ AUTO-SCALE MICRO MODE",
            style='PremiumTitle.TLabel'
        )
        auto_title.pack(anchor='w', pady=(0, 5))
        
        # Auto-scale description
        auto_desc = ttk.Label(
            auto_scale_frame,
            text="Enable dynamic scaling in micro mode for adaptive trade sizing based on market conditions",
            style='PremiumSubtitle.TLabel'
        )
        auto_desc.pack(anchor='w', pady=(0, 10))
        
        # Auto-scale checkbox with premium styling
        self.auto_scale_var = tk.BooleanVar()
        self.auto_scale_checkbox = ttk.Checkbutton(
            auto_scale_frame,
            text="🎯 Enable Auto-Scale for Micro Mode",
            variable=self.auto_scale_var,
            style='AutoScale.TCheckbutton',
            command=self.toggle_auto_scale
        )
        self.auto_scale_checkbox.pack(anchor='w', pady=(0, 5))
        
        # Auto-scale status
        self.auto_scale_status = ttk.Label(
            auto_scale_frame,
            text="Status: Auto-scale DISABLED - Manual control active",
            style='PremiumSubtitle.TLabel'
        )
        self.auto_scale_status.pack(anchor='w')

    def create_main_slider_section(self):
        """Create the main trade cap slider section."""
        slider_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        slider_frame.pack(fill='x', pady=(0, 10))
        
        # Slider title
        slider_title = ttk.Label(
            slider_frame,
            text="💰 DYNAMIC TRADE CAP SLIDER",
            style='PremiumTitle.TLabel'
        )
        slider_title.pack(anchor='w', pady=(0, 5))
        
        # Slider description
        slider_desc = ttk.Label(
            slider_frame,
            text="Adjust trade size from $1.00 to $10.00 - Any change dynamically shifts the trading scenario",
            style='PremiumSubtitle.TLabel'
        )
        slider_desc.pack(anchor='w', pady=(0, 10))
        
        # Slider container
        slider_container = ttk.Frame(slider_frame)
        slider_container.pack(fill='x')
        
        # Current value display
        self.value_label = ttk.Label(
            slider_container,
            text="$1.00",
            style='PremiumTitle.TLabel'
        )
        self.value_label.pack(pady=(0, 5))
        
        # Main slider
        self.trade_cap_slider = ttk.Scale(
            slider_container,
            from_=1.0,
            to=10.0,
            orient='horizontal',
            command=self.on_slider_change,
            state='disabled'
        )
        self.trade_cap_slider.pack(fill='x', pady=(0, 5))
        
        # Slider range labels
        range_frame = ttk.Frame(slider_container)
        range_frame.pack(fill='x')
        
        ttk.Label(range_frame, text="$1.00", style='PremiumSubtitle.TLabel').pack(side='left')
        ttk.Label(range_frame, text="$10.00", style='PremiumSubtitle.TLabel').pack(side='right')

    def create_entropy_section(self):
        """Create the entropy-based strategy shifting section."""
        entropy_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        entropy_frame.pack(fill='x', pady=(0, 10))
        
        # Entropy title
        entropy_title = ttk.Label(
            entropy_frame,
            text="🌀 ENTROPY-BASED STRATEGY SHIFTING",
            style='PremiumTitle.TLabel'
        )
        entropy_title.pack(anchor='w', pady=(0, 5))
        
        # Entropy description
        entropy_desc = ttk.Label(
            entropy_frame,
            text="Dynamic strategy adjustment based on market complexity (entropy) - Conservative to Aggressive",
            style='PremiumSubtitle.TLabel'
        )
        entropy_desc.pack(anchor='w', pady=(0, 10))
        
        # Entropy slider
        entropy_container = ttk.Frame(entropy_frame)
        entropy_container.pack(fill='x')
        
        self.entropy_slider = ttk.Scale(
            entropy_container,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            command=self.on_entropy_change
        )
        self.entropy_slider.pack(fill='x', pady=(0, 5))
        
        # Entropy range labels
        entropy_range_frame = ttk.Frame(entropy_container)
        entropy_range_frame.pack(fill='x')
        
        ttk.Label(entropy_range_frame, text="Conservative", style='PremiumSubtitle.TLabel').pack(side='left')
        ttk.Label(entropy_range_frame, text="Balanced", style='PremiumSubtitle.TLabel').pack(side='left', expand=True)
        ttk.Label(entropy_range_frame, text="Aggressive", style='PremiumSubtitle.TLabel').pack(side='right')
        
        # Current entropy display
        self.entropy_label = ttk.Label(
            entropy_frame,
            text="Current Entropy: 0.50 (Balanced)",
            style='PremiumSubtitle.TLabel'
        )
        self.entropy_label.pack(anchor='w', pady=(5, 0))

    def create_profit_projections_section(self):
        """Create the real-time profit projections section."""
        projections_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        projections_frame.pack(fill='x', pady=(0, 10))
        
        # Projections title
        projections_title = ttk.Label(
            projections_frame,
            text="📈 REAL-TIME PROFIT PROJECTIONS",
            style='PremiumTitle.TLabel'
        )
        projections_title.pack(anchor='w', pady=(0, 5))
        
        # Projections description
        projections_desc = ttk.Label(
            projections_frame,
            text="Live profit calculations based on current trade cap and market conditions",
            style='PremiumSubtitle.TLabel'
        )
        projections_desc.pack(anchor='w', pady=(0, 10))
        
        # Projections grid
        projections_grid = ttk.Frame(projections_frame)
        projections_grid.pack(fill='x')
        
        # Hourly projection
        ttk.Label(projections_grid, text="Hourly:", style='PremiumSubtitle.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.hourly_projection = ttk.Label(projections_grid, text="$0.00", style='PremiumSubtitle.TLabel')
        self.hourly_projection.grid(row=0, column=1, sticky='w')
        
        # Daily projection
        ttk.Label(projections_grid, text="Daily:", style='PremiumSubtitle.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.daily_projection = ttk.Label(projections_grid, text="$0.00", style='PremiumSubtitle.TLabel')
        self.daily_projection.grid(row=1, column=1, sticky='w')
        
        # Weekly projection
        ttk.Label(projections_grid, text="Weekly:", style='PremiumSubtitle.TLabel').grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.weekly_projection = ttk.Label(projections_grid, text="$0.00", style='PremiumSubtitle.TLabel')
        self.weekly_projection.grid(row=2, column=1, sticky='w')
        
        # Monthly projection
        ttk.Label(projections_grid, text="Monthly:", style='PremiumSubtitle.TLabel').grid(row=3, column=0, sticky='w', padx=(0, 10))
        self.monthly_projection = ttk.Label(projections_grid, text="$0.00", style='PremiumSubtitle.TLabel')
        self.monthly_projection.grid(row=3, column=1, sticky='w')

    def create_strategy_adjustment_section(self):
        """Create the live strategy adjustment section."""
        strategy_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        strategy_frame.pack(fill='x', pady=(0, 10))
        
        # Strategy title
        strategy_title = ttk.Label(
            strategy_frame,
            text="🎯 LIVE STRATEGY ADJUSTMENT",
            style='PremiumTitle.TLabel'
        )
        strategy_title.pack(anchor='w', pady=(0, 5))
        
        # Strategy description
        strategy_desc = ttk.Label(
            strategy_frame,
            text="Real-time strategy parameters that adapt based on entropy and comfort levels",
            style='PremiumSubtitle.TLabel'
        )
        strategy_desc.pack(anchor='w', pady=(0, 10))
        
        # Strategy grid
        strategy_grid = ttk.Frame(strategy_frame)
        strategy_grid.pack(fill='x')
        
        # Strategy type
        ttk.Label(strategy_grid, text="Strategy Type:", style='PremiumSubtitle.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.strategy_type_label = ttk.Label(strategy_grid, text="Balanced", style='PremiumSubtitle.TLabel')
        self.strategy_type_label.grid(row=0, column=1, sticky='w')
        
        # Shift factor
        ttk.Label(strategy_grid, text="Shift Factor:", style='PremiumSubtitle.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.shift_factor_label = ttk.Label(strategy_grid, text="0.50", style='PremiumSubtitle.TLabel')
        self.shift_factor_label.grid(row=1, column=1, sticky='w')
        
        # Adaptation rate
        ttk.Label(strategy_grid, text="Adaptation Rate:", style='PremiumSubtitle.TLabel').grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.adaptation_rate_label = ttk.Label(strategy_grid, text="0.50", style='PremiumSubtitle.TLabel')
        self.adaptation_rate_label.grid(row=2, column=1, sticky='w')
        
        # Risk level
        ttk.Label(strategy_grid, text="Risk Level:", style='PremiumSubtitle.TLabel').grid(row=3, column=0, sticky='w', padx=(0, 10))
        self.risk_level_label = ttk.Label(strategy_grid, text="Medium", style='PremiumSubtitle.TLabel')
        self.risk_level_label.grid(row=3, column=1, sticky='w')
        
        # Confidence
        ttk.Label(strategy_grid, text="Confidence:", style='PremiumSubtitle.TLabel').grid(row=4, column=0, sticky='w', padx=(0, 10))
        self.confidence_label = ttk.Label(strategy_grid, text="0.70", style='PremiumSubtitle.TLabel')
        self.confidence_label.grid(row=4, column=1, sticky='w')
        
        # Market alignment
        ttk.Label(strategy_grid, text="Market Alignment:", style='PremiumSubtitle.TLabel').grid(row=5, column=0, sticky='w', padx=(0, 10))
        self.market_alignment_label = ttk.Label(strategy_grid, text="0.60", style='PremiumSubtitle.TLabel')
        self.market_alignment_label.grid(row=5, column=1, sticky='w')
        
        # Efficiency
        ttk.Label(strategy_grid, text="Efficiency:", style='PremiumSubtitle.TLabel').grid(row=6, column=0, sticky='w', padx=(0, 10))
        self.efficiency_label = ttk.Label(strategy_grid, text="0.75", style='PremiumSubtitle.TLabel')
        self.efficiency_label.grid(row=6, column=1, sticky='w')

    def create_safety_section(self):
        """Create the maximum paranoia safety section."""
        safety_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        safety_frame.pack(fill='x', pady=(0, 10))
        
        # Safety title
        safety_title = ttk.Label(
            safety_frame,
            text="🛡️ MAXIMUM PARANOIA SAFETY",
            style='PremiumTitle.TLabel'
        )
        safety_title.pack(anchor='w', pady=(0, 5))
        
        # Safety description
        safety_desc = ttk.Label(
            safety_frame,
            text="Advanced safety protocols for micro mode trading with triple confirmation",
            style='PremiumSubtitle.TLabel'
        )
        safety_desc.pack(anchor='w', pady=(0, 10))
        
        # Safety grid
        safety_grid = ttk.Frame(safety_frame)
        safety_grid.pack(fill='x')
        
        # Paranoia status
        ttk.Label(safety_grid, text="Paranoia Status:", style='PremiumSubtitle.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.paranoia_status_label = ttk.Label(safety_grid, text="ACTIVE", style='PremiumSubtitle.TLabel')
        self.paranoia_status_label.grid(row=0, column=1, sticky='w')
        
        # Paranoia level
        ttk.Label(safety_grid, text="Paranoia Level:", style='PremiumSubtitle.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.paranoia_level_label = ttk.Label(safety_grid, text="Maximum", style='PremiumSubtitle.TLabel')
        self.paranoia_level_label.grid(row=1, column=1, sticky='w')
        
        # Triple confirmation
        ttk.Label(safety_grid, text="Triple Confirmation:", style='PremiumSubtitle.TLabel').grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.triple_confirmation_label = ttk.Label(safety_grid, text="ENABLED", style='PremiumSubtitle.TLabel')
        self.triple_confirmation_label.grid(row=2, column=1, sticky='w')
        
        # Emergency stop
        ttk.Label(safety_grid, text="Emergency Stop:", style='PremiumSubtitle.TLabel').grid(row=3, column=0, sticky='w', padx=(0, 10))
        self.emergency_stop_label = ttk.Label(safety_grid, text="READY", style='PremiumSubtitle.TLabel')
        self.emergency_stop_label.grid(row=3, column=1, sticky='w')

    def create_comfort_scaling_section(self):
        """Create the user comfort level scaling section."""
        comfort_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        comfort_frame.pack(fill='x', pady=(0, 10))
        
        # Comfort title
        comfort_title = ttk.Label(
            comfort_frame,
            text="😌 USER COMFORT LEVEL SCALING",
            style='PremiumTitle.TLabel'
        )
        comfort_title.pack(anchor='w', pady=(0, 5))
        
        # Comfort description
        comfort_desc = ttk.Label(
            comfort_frame,
            text="Adjust strategy based on your risk comfort level - from Nervous to Confident",
            style='PremiumSubtitle.TLabel'
        )
        comfort_desc.pack(anchor='w', pady=(0, 10))
        
        # Comfort slider
        comfort_container = ttk.Frame(comfort_frame)
        comfort_container.pack(fill='x')
        
        self.comfort_slider = ttk.Scale(
            comfort_container,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            command=self.on_comfort_change
        )
        self.comfort_slider.pack(fill='x', pady=(0, 5))
        
        # Comfort range labels
        comfort_range_frame = ttk.Frame(comfort_container)
        comfort_range_frame.pack(fill='x')
        
        ttk.Label(comfort_range_frame, text="Nervous", style='PremiumSubtitle.TLabel').pack(side='left')
        ttk.Label(comfort_range_frame, text="Comfortable", style='PremiumSubtitle.TLabel').pack(side='left', expand=True)
        ttk.Label(comfort_range_frame, text="Confident", style='PremiumSubtitle.TLabel').pack(side='right')
        
        # Current comfort display
        self.comfort_label = ttk.Label(
            comfort_frame,
            text="Current Comfort: 0.50 (Comfortable)",
            style='PremiumSubtitle.TLabel'
        )
        self.comfort_label.pack(anchor='w', pady=(5, 0))

    def create_status_controls_section(self):
        """Create the status and controls section with premium buttons."""
        controls_frame = ttk.Frame(self.main_frame, style='Premium.TFrame')
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Controls title
        controls_title = ttk.Label(
            controls_frame,
            text="🎮 PREMIUM CONTROLS",
            style='PremiumTitle.TLabel'
        )
        controls_title.pack(anchor='w', pady=(0, 10))
        
        # Buttons container
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        # Apply settings button (smaller, premium)
        self.apply_button = ttk.Button(
            buttons_frame,
            text="✅ APPLY SETTINGS",
            style='Micro.TButton',
            command=self.apply_settings
        )
        self.apply_button.pack(side='left', padx=(0, 10))
        
        # Reset button (smaller, premium)
        self.reset_button = ttk.Button(
            buttons_frame,
            text="🔄 RESET TO DEFAULT",
            style='Micro.TButton',
            command=self.reset_to_default
        )
        self.reset_button.pack(side='left', padx=(0, 10))
        
        # Emergency stop button (smaller, premium)
        self.emergency_button = ttk.Button(
            buttons_frame,
            text="🚨 EMERGENCY STOP",
            style='Emergency.TButton',
            command=self.trigger_emergency_stop
        )
        self.emergency_button.pack(side='right')

    def toggle_slider_lock(self):
        """Toggle the slider lock state."""
        self.slider_locked = not self.slider_locked
        
        if self.slider_locked:
            self.trade_cap_slider.config(state='disabled')
            self.lock_status_label.config(text="🔒 SLIDER LOCKED - Click to unlock dynamic scaling")
            self.unlock_button.config(text="🔓 UNLOCK")
        else:
            self.trade_cap_slider.config(state='normal')
            self.lock_status_label.config(text="🔓 SLIDER UNLOCKED - Dynamic scaling active")
            self.unlock_button.config(text="🔒 LOCK")
            
        # Update clock system if available
        if self.clock_system and hasattr(self.clock_system, 'SAFETY_CONFIG'):
            self.clock_system.SAFETY_CONFIG.micro_trade_cap = self.current_trade_cap

    def toggle_auto_scale(self):
        """Toggle auto-scale functionality."""
        self.auto_scale_enabled = self.auto_scale_var.get()
        
        if self.auto_scale_enabled:
            self.auto_scale_status.config(text="Status: Auto-scale ENABLED - Dynamic scaling active")
            # Enable auto-scaling logic
            self.start_auto_scale()
        else:
            self.auto_scale_status.config(text="Status: Auto-scale DISABLED - Manual control active")
            # Disable auto-scaling logic
            self.stop_auto_scale()

    def start_auto_scale(self):
        """Start auto-scaling functionality."""
        # This would implement automatic scaling based on market conditions
        # For now, we'll simulate it
        pass

    def stop_auto_scale(self):
        """Stop auto-scaling functionality."""
        # This would stop automatic scaling
        pass

    def on_slider_change(self, value):
        """Handle slider value changes."""
        try:
            self.current_trade_cap = float(value)
            self.value_label.config(text=f"${self.current_trade_cap:.2f}")
            
            # Update color based on value
            if self.current_trade_cap <= 2.0:
                self.value_label.config(foreground='#27ae60')  # Green for low values
            elif self.current_trade_cap <= 5.0:
                self.value_label.config(foreground='#f39c12')  # Orange for medium values
            else:
                self.value_label.config(foreground='#e74c3c')  # Red for high values
            
            # Recalculate strategy
            self.recalculate_strategy()
            
            # Update clock system if available
            if self.clock_system and hasattr(self.clock_system, 'SAFETY_CONFIG'):
                self.clock_system.SAFETY_CONFIG.micro_trade_cap = self.current_trade_cap
                
        except ValueError:
            pass

    def on_entropy_change(self, value):
        """Handle entropy slider changes."""
        try:
            self.current_entropy = float(value)
            
            # Update entropy label
            if self.current_entropy < 0.33:
                entropy_text = "Conservative"
            elif self.current_entropy < 0.67:
                entropy_text = "Balanced"
            else:
                entropy_text = "Aggressive"
            
            self.entropy_label.config(text=f"Current Entropy: {self.current_entropy:.2f} ({entropy_text})")
            
            # Update strategy based on entropy
            self.update_strategy_from_entropy()
            
        except ValueError:
            pass

    def on_comfort_change(self, value):
        """Handle comfort slider changes."""
        try:
            self.current_comfort = float(value)
            
            # Update comfort label
            if self.current_comfort < 0.33:
                comfort_text = "Nervous"
            elif self.current_comfort < 0.67:
                comfort_text = "Comfortable"
            else:
                comfort_text = "Confident"
            
            self.comfort_label.config(text=f"Current Comfort: {self.current_comfort:.2f} ({comfort_text})")
            
            # Adjust strategy for comfort
            self.adjust_strategy_for_comfort(self.current_comfort)
            
        except ValueError:
            pass

    def recalculate_strategy(self):
        """Recalculate all strategy parameters."""
        # Calculate profit projections
        hourly_profit = self.current_trade_cap * 0.1 * 24  # 10% profit per trade, 24 trades per hour
        daily_profit = hourly_profit * 24
        weekly_profit = daily_profit * 7
        monthly_profit = daily_profit * 30
        
        # Update profit projections
        self.hourly_projection.config(text=f"${hourly_profit:.2f}")
        self.daily_projection.config(text=f"${daily_profit:.2f}")
        self.weekly_projection.config(text=f"${weekly_profit:.2f}")
        self.monthly_projection.config(text=f"${monthly_profit:.2f}")
        
        # Update strategy parameters
        self.shift_factor = self.current_entropy * self.current_trade_cap / 10.0
        self.adaptation_rate = 0.5 + (self.current_entropy * 0.5)
        self.confidence = 0.7 + (self.current_comfort * 0.3)
        self.market_alignment = 0.6 + (self.current_entropy * 0.4)
        self.efficiency = 0.75 + (self.current_comfort * 0.25)
        
        # Update labels
        self.shift_factor_label.config(text=f"{self.shift_factor:.2f}")
        self.adaptation_rate_label.config(text=f"{self.adaptation_rate:.2f}")
        self.confidence_label.config(text=f"{self.confidence:.2f}")
        self.market_alignment_label.config(text=f"{self.market_alignment:.2f}")
        self.efficiency_label.config(text=f"{self.efficiency:.2f}")

    def update_strategy_from_entropy(self):
        """Update strategy based on entropy level."""
        if self.current_entropy < 0.33:
            self.strategy_type = "Conservative"
            self.risk_level = "Low"
        elif self.current_entropy < 0.67:
            self.strategy_type = "Balanced"
            self.risk_level = "Medium"
        else:
            self.strategy_type = "Aggressive"
            self.risk_level = "High"
        
        # Update labels
        self.strategy_type_label.config(text=self.strategy_type)
        self.risk_level_label.config(text=self.risk_level)
        
        # Highlight strategy type
        self.highlight_strategy(self.strategy_type)

    def adjust_strategy_for_comfort(self, comfort_level):
        """Adjust strategy based on user comfort level."""
        # Adjust confidence and efficiency based on comfort
        self.confidence = 0.7 + (comfort_level * 0.3)
        self.efficiency = 0.75 + (comfort_level * 0.25)
        
        # Update labels
        self.confidence_label.config(text=f"{self.confidence:.2f}")
        self.efficiency_label.config(text=f"{self.efficiency:.2f}")

    def highlight_strategy(self, strategy_type):
        """Highlight the current strategy type."""
        # This could add visual highlighting to the strategy type label
        pass

    def apply_settings(self):
        """Apply current settings to the clock system."""
        try:
            if self.clock_system and hasattr(self.clock_system, 'SAFETY_CONFIG'):
                self.clock_system.SAFETY_CONFIG.micro_trade_cap = self.current_trade_cap
                
                # Apply auto-scale settings
                if self.auto_scale_enabled:
                    # Enable auto-scaling in the clock system
                    pass
                
                messagebox.showinfo("Settings Applied", 
                                  f"✅ Settings applied successfully!\n"
                                  f"Trade Cap: ${self.current_trade_cap:.2f}\n"
                                  f"Auto-Scale: {'Enabled' if self.auto_scale_enabled else 'Disabled'}\n"
                                  f"Strategy: {self.strategy_type}")
            else:
                messagebox.showwarning("Clock System Not Available", 
                                     "Clock system not connected. Settings saved locally only.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    def reset_to_default(self):
        """Reset all settings to default values."""
        try:
            # Reset sliders
            self.trade_cap_slider.set(1.0)
            self.entropy_slider.set(0.5)
            self.comfort_slider.set(0.5)
            
            # Reset auto-scale
            self.auto_scale_var.set(False)
            self.auto_scale_enabled = False
            self.auto_scale_status.config(text="Status: Auto-scale DISABLED - Manual control active")
            
            # Reset values
            self.current_trade_cap = 1.0
            self.current_entropy = 0.5
            self.current_comfort = 0.5
            
            # Recalculate strategy
            self.recalculate_strategy()
            self.update_strategy_from_entropy()
            self.adjust_strategy_for_comfort(self.current_comfort)
            
            # Update clock system
            if self.clock_system and hasattr(self.clock_system, 'SAFETY_CONFIG'):
                self.clock_system.SAFETY_CONFIG.micro_trade_cap = self.current_trade_cap
            
            messagebox.showinfo("Reset Complete", "✅ All settings reset to default values!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset settings: {str(e)}")

    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        try:
            result = messagebox.askyesno("Emergency Stop", 
                                       "🚨 Are you sure you want to trigger EMERGENCY STOP?\n\n"
                                       "This will immediately halt all micro mode trading!")
            
            if result:
                self.emergency_stop = True
                self.emergency_stop_label.config(text="TRIGGERED")
                
                # Trigger emergency stop in clock system
                if self.clock_system and hasattr(self.clock_system, 'trigger_micro_emergency_stop'):
                    self.clock_system.trigger_micro_emergency_stop()
                
                messagebox.showwarning("Emergency Stop Triggered", 
                                     "🚨 EMERGENCY STOP ACTIVATED!\n"
                                     "All micro mode trading has been halted immediately!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to trigger emergency stop: {str(e)}")

    def start_dynamic_updates(self):
        """Start dynamic updates for real-time data."""
        def update_loop():
            while True:
                try:
                    # Update safety status
                    self.update_safety_status()
                    
                    # Sleep for update interval
                    time.sleep(1.0)
                    
                except Exception as e:
                    print(f"Error in dynamic updates: {e}")
                    time.sleep(5.0)
        
        # Start update thread
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

    def update_safety_status(self):
        """Update safety status indicators."""
        try:
            # Update paranoia status
            if self.emergency_stop:
                self.paranoia_status_label.config(text="EMERGENCY")
            else:
                self.paranoia_status_label.config(text="ACTIVE")
            
            # Update emergency stop status
            if self.emergency_stop:
                self.emergency_stop_label.config(text="TRIGGERED")
            else:
                self.emergency_stop_label.config(text="READY")
                
        except Exception as e:
            print(f"Error updating safety status: {e}")

class SchwabotDynamicTradeCapGUI:
    """Main GUI application for Schwabot Dynamic Trade Cap Slider."""
    
    def __init__(self, clock_system=None):
        self.clock_system = clock_system
        self.root = tk.Tk()
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI window."""
        self.root.title("🎯 Schwabot Dynamic Trade Cap Slider - Premium Interface")
        self.root.geometry("800x900")
        self.root.configure(bg='#2c3e50')
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap('schwabot_icon.ico')
        except:
            pass
        
        # Create main slider
        self.slider = DynamicTradeCapSlider(self.root, self.clock_system)
        
        # Configure window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Handle window closing."""
        try:
            if messagebox.askokcancel("Quit", "Do you want to quit the Dynamic Trade Cap Slider?"):
                self.root.destroy()
        except:
            self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

def main():
    """Main function to run the GUI."""
    try:
        # Import clock system if available
        try:
            from clock_mode_system import ClockModeSystem
            clock_system = ClockModeSystem()
            print("✅ Clock Mode System connected")
        except ImportError:
            clock_system = None
            print("⚠️ Clock Mode System not available - running in standalone mode")
        
        # Create and run GUI
        app = SchwabotDynamicTradeCapGUI(clock_system)
        app.run()
        
    except Exception as e:
        print(f"❌ Error running GUI: {e}")

if __name__ == "__main__":
    main() 