#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data sources for backtesting.
Provides multiple data sources including real APIs and simulated data.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source_type: str  # "binance", "yahoo", "simulated", "csv"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3

class SimulatedDataGenerator:
    """Generate realistic simulated market data for testing."""
    
    def __init__(self, base_price: float = 45000.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
        
    def generate_ohlcv_data(self, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """Generate OHLCV data for the specified period."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate number of intervals
        interval_hours = {
            "1m": 1/60, "5m": 5/60, "15m": 15/60, 
            "1h": 1, "4h": 4, "1d": 24
        }
        
        hours_per_interval = interval_hours.get(interval, 1)
        total_hours = (end_dt - start_dt).total_seconds() / 3600
        num_intervals = int(total_hours / hours_per_interval)
        
        data = []
        current_time = start_dt
        current_price = self.base_price
        
        for i in range(num_intervals):
            # Generate price movement using random walk
            price_change = np.random.normal(0, self.volatility * current_price)
            current_price = max(current_price + price_change, current_price * 0.5)  # Prevent negative prices
            
            # Generate OHLCV
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = current_price * (1 + np.random.normal(0, 0.002))
            close_price = current_price
            volume = np.random.uniform(100, 10000)
            
            data.append({
                "timestamp": current_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })
            
            # Move to next interval
            current_time += timedelta(hours=hours_per_interval)
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        logger.info(f"✅ Generated {len(df)} simulated data points")
        return df

class BinanceDataSource:
    """Binance API data source with fallback options."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.binance.com"
        self.timeout = config.timeout
        self.retry_attempts = config.retry_attempts
        
    async def get_klines(self, symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """Get klines data from Binance API."""
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": 1000
            }
            
            for attempt in range(self.retry_attempts):
                try:
                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    
                    # Convert types
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    
                    df.set_index("timestamp", inplace=True)
                    
                    logger.info(f"✅ Loaded {len(df)} data points from Binance")
                    return df
                    
                except requests.exceptions.RequestException as e:
                    if attempt < self.retry_attempts - 1:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(1)
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"❌ Failed to load Binance data: {e}")
            return pd.DataFrame()

class YahooDataSource:
    """Yahoo Finance data source (alternative to Binance)."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.timeout = config.timeout
        
    async def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from Yahoo Finance."""
        try:
            # Note: This would require yfinance library
            # For now, return empty DataFrame
            logger.warning("⚠️ Yahoo Finance data source not implemented yet")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to load Yahoo data: {e}")
            return pd.DataFrame()

class DataSourceManager:
    """Manages multiple data sources with fallback options."""
    
    def __init__(self, configs: List[DataSourceConfig]):
        self.configs = configs
        self.sources = {}
        self.simulated_generator = SimulatedDataGenerator()
        
        # Initialize data sources
        for config in configs:
            if config.source_type == "binance":
                self.sources["binance"] = BinanceDataSource(config)
            elif config.source_type == "yahoo":
                self.sources["yahoo"] = YahooDataSource(config)
    
    async def get_market_data(self, symbol: str, start_date: str, end_date: str, 
                            interval: str = "1h", prefer_source: str = "binance") -> pd.DataFrame:
        """Get market data from preferred source with fallbacks."""
        
        # Try preferred source first
        if prefer_source in self.sources:
            logger.info(f"📊 Trying {prefer_source} data source...")
            df = await self._get_from_source(prefer_source, symbol, start_date, end_date, interval)
            if not df.empty:
                return df
        
        # Try other sources
        for source_name, source in self.sources.items():
            if source_name != prefer_source:
                logger.info(f"📊 Trying {source_name} data source...")
                df = await self._get_from_source(source_name, symbol, start_date, end_date, interval)
                if not df.empty:
                    return df
        
        # Fallback to simulated data
        logger.warning("⚠️ All real data sources failed, using simulated data")
        return self.simulated_generator.generate_ohlcv_data(start_date, end_date, interval)
    
    async def _get_from_source(self, source_name: str, symbol: str, start_date: str, 
                              end_date: str, interval: str) -> pd.DataFrame:
        """Get data from a specific source."""
        try:
            source = self.sources[source_name]
            
            if source_name == "binance":
                return await source.get_klines(symbol, start_date, end_date, interval)
            elif source_name == "yahoo":
                return await source.get_data(symbol, start_date, end_date)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error getting data from {source_name}: {e}")
            return pd.DataFrame()
    
    def generate_simulated_data(self, symbol: str, start_date: str, end_date: str, 
                               interval: str = "1h", base_price: float = None) -> pd.DataFrame:
        """Generate simulated data for testing."""
        if base_price is None:
            # Use realistic base prices for common symbols
            base_prices = {
                "BTCUSDT": 45000.0,
                "ETHUSDT": 3000.0,
                "XRPUSDT": 0.5,
                "ADAUSDT": 0.4,
                "DOTUSDT": 7.0
            }
            base_price = base_prices.get(symbol, 100.0)
        
        generator = SimulatedDataGenerator(base_price=base_price)
        return generator.generate_ohlcv_data(start_date, end_date, interval)

# Convenience function
async def get_market_data(symbol: str, start_date: str, end_date: str, 
                         interval: str = "1h", source_type: str = "auto") -> pd.DataFrame:
    """Get market data with automatic fallback."""
    
    # Create default configs
    configs = [
        DataSourceConfig(source_type="binance"),
        DataSourceConfig(source_type="yahoo")
    ]
    
    manager = DataSourceManager(configs)
    
    if source_type == "simulated":
        return manager.generate_simulated_data(symbol, start_date, end_date, interval)
    else:
        return await manager.get_market_data(symbol, start_date, end_date, interval) 