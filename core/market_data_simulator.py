#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Market Data Simulator
==============================

Generates realistic market data for testing the trading pipeline.
Simulates price movements, volume, volatility, and sentiment.
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from trading_pipeline_manager import MarketDataPoint

class MarketDataSimulator:
    """Simulates realistic market data for trading system testing."""
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """Initialize the market data simulator."""
        self.symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD"]
        
        # Market state for each symbol
        self.market_states = {}
        for symbol in self.symbols:
            self.market_states[symbol] = {
                "base_price": self._get_base_price(symbol),
                "current_price": self._get_base_price(symbol),
                "trend": random.choice(["bullish", "bearish", "sideways"]),
                "volatility": random.uniform(0.01, 0.05),
                "volume_base": random.uniform(1000, 5000),
                "sentiment": random.uniform(0.3, 0.7),
                "last_update": time.time()
            }
        
        # Simulation parameters
        self.update_interval = 5  # seconds
        self.running = False
        
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for a symbol."""
        base_prices = {
            "BTC/USD": 45000.0,
            "ETH/USD": 3000.0,
            "SOL/USD": 100.0,
            "XRP/USD": 0.5,
            "ADA/USD": 0.4
        }
        return base_prices.get(symbol, 100.0)
    
    def _generate_price_movement(self, symbol: str) -> Dict[str, float]:
        """Generate realistic price movement for a symbol."""
        state = self.market_states[symbol]
        
        # Time-based factors
        current_time = time.time()
        time_factor = (current_time - state["last_update"]) / 3600  # hours
        
        # Trend influence
        trend_strength = 0.02 if state["trend"] == "bullish" else -0.02 if state["trend"] == "bearish" else 0.0
        
        # Random walk component
        random_component = random.gauss(0, state["volatility"])
        
        # Mean reversion component
        mean_reversion = (state["base_price"] - state["current_price"]) * 0.001
        
        # Calculate price change
        price_change_pct = trend_strength + random_component + mean_reversion
        
        # Apply change
        new_price = state["current_price"] * (1 + price_change_pct)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, state["base_price"] * 0.1)
        
        # Update state
        state["current_price"] = new_price
        state["last_update"] = current_time
        
        # Occasionally change trend
        if random.random() < 0.01:  # 1% chance per update
            state["trend"] = random.choice(["bullish", "bearish", "sideways"])
        
        # Update volatility
        state["volatility"] = max(0.005, min(0.1, state["volatility"] + random.gauss(0, 0.001)))
        
        return {
            "price": new_price,
            "price_change": price_change_pct,
            "volatility": state["volatility"]
        }
    
    def _generate_volume(self, symbol: str, price_change: float) -> float:
        """Generate realistic volume based on price movement."""
        state = self.market_states[symbol]
        
        # Base volume
        base_volume = state["volume_base"]
        
        # Volume increases with price movement
        volume_multiplier = 1 + abs(price_change) * 10
        
        # Add some randomness
        volume_multiplier *= random.uniform(0.8, 1.2)
        
        # Ensure reasonable volume range
        volume = base_volume * volume_multiplier
        volume = max(100, min(10000, volume))
        
        return volume
    
    def _generate_sentiment(self, symbol: str, price_change: float, volume: float) -> float:
        """Generate realistic sentiment based on market conditions."""
        state = self.market_states[symbol]
        
        # Base sentiment
        sentiment = state["sentiment"]
        
        # Price change influence
        sentiment += price_change * 2
        
        # Volume influence
        volume_factor = (volume - state["volume_base"]) / state["volume_base"]
        sentiment += volume_factor * 0.1
        
        # Add some randomness
        sentiment += random.gauss(0, 0.05)
        
        # Clamp to [0, 1]
        sentiment = max(0.0, min(1.0, sentiment))
        
        # Update state
        state["sentiment"] = sentiment
        
        return sentiment
    
    def generate_market_data(self, symbol: str) -> MarketDataPoint:
        """Generate market data point for a symbol."""
        # Generate price movement
        price_data = self._generate_price_movement(symbol)
        
        # Generate volume
        volume = self._generate_volume(symbol, price_data["price_change"])
        
        # Generate sentiment
        sentiment = self._generate_sentiment(symbol, price_data["price_change"], volume)
        
        # Create market data point
        return MarketDataPoint(
            timestamp=time.time(),
            symbol=symbol,
            price=price_data["price"],
            volume=volume,
            price_change=price_data["price_change"],
            volatility=price_data["volatility"],
            sentiment=sentiment,
            metadata={
                "trend": self.market_states[symbol]["trend"],
                "base_price": self.market_states[symbol]["base_price"]
            }
        )
    
    def generate_all_market_data(self) -> List[MarketDataPoint]:
        """Generate market data for all symbols."""
        return [self.generate_market_data(symbol) for symbol in self.symbols]
    
    async def start_simulation(self, callback=None):
        """Start the market data simulation."""
        self.running = True
        
        while self.running:
            try:
                # Generate data for all symbols
                market_data_list = self.generate_all_market_data()
                
                # Call callback if provided
                if callback:
                    for market_data in market_data_list:
                        await callback(market_data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Market data simulation error: {e}")
                await asyncio.sleep(1)
    
    def stop_simulation(self):
        """Stop the market data simulation."""
        self.running = False
    
    def get_market_state(self, symbol: str) -> Dict:
        """Get current market state for a symbol."""
        return self.market_states.get(symbol, {}).copy()
    
    def set_market_state(self, symbol: str, **kwargs):
        """Set market state parameters for a symbol."""
        if symbol in self.market_states:
            self.market_states[symbol].update(kwargs)

# Global instance for easy access
market_simulator = None

def get_market_simulator(symbols: Optional[List[str]] = None) -> MarketDataSimulator:
    """Get the global market simulator instance."""
    global market_simulator
    if market_simulator is None:
        market_simulator = MarketDataSimulator(symbols)
    return market_simulator 