#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Market Data Integration for Schwabot Trading System
=======================================================

Real-time integration with Coinbase API, Finance API, and Kraken API
for live trading data, RSI calculations, and time-based phase triggers.

This system provides:
- Live price feeds from multiple exchanges
- Real-time RSI calculations
- Volume analysis and triggers
- Time-based phase detection (midnight/noon patterns)
- Internal hashing system integration
- Cross-exchange arbitrage detection
- Memory key generation and recall
- Strategy tier mapping (2, 6, 8)
- Decimal-based trigger logic
"""

import asyncio
import aiohttp
import json
import logging
import time
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from pathlib import Path

# Data processing imports
import pandas as pd
import numpy as np

# Technical analysis imports with fallbacks
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None
    logging.warning("TA-Lib not available, using fallback calculations")

try:
    import ccxt  # type: ignore
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None
    logging.warning("CCXT not available, using fallback exchange data")

logger = logging.getLogger(__name__)

# =============================================================================
# MARKET DATA ENUMS AND DATA STRUCTURES
# =============================================================================

class ExchangeType(Enum):
    """Supported exchange types."""
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FINANCE_API = "finance_api"
    BINANCE = "binance"
    BINANCE_USA = "binance_usa"
    COINGECKO = "coingecko"

class TimePhase(Enum):
    """Time-based trading phases."""
    MIDNIGHT = "midnight"           # 00:00 - Reset/Re-accumulate
    PRE_DAWN = "pre_dawn"          # 03:00 - Entry zones
    MORNING = "morning"            # 07:00 - Surge start
    HIGH_NOON = "high_noon"        # 12:00 - False peak/dump
    LATE_NOON = "late_noon"        # 16:00 - Lower high
    EVENING = "evening"            # 20:00 - Dip setup
    MIDNIGHT_PLUS = "midnight_plus" # 23:59 - New cycle

class StrategyTier(Enum):
    """Strategy tiers based on decimal mapping."""
    TIER_2 = 2      # Low frequency, conservative
    TIER_6 = 6      # Medium frequency, balanced
    TIER_8 = 8      # High frequency, aggressive

@dataclass
class MarketData:
    """Real-time market data structure."""
    timestamp: float
    symbol: str
    exchange: str
    price: float
    volume: float
    rsi: float
    rsi_14: float
    rsi_1h: float
    rsi_4h: float
    vwap: float
    atr: float
    high: float
    low: float
    open_price: float
    close_price: float
    bid: float
    ask: float
    spread: float
    order_book_depth: Dict[str, float]
    phase: TimePhase
    strategy_tier: StrategyTier
    decimal_key: str
    hash_signature: str
    memory_key: str

@dataclass
class TradingSignal:
    """Trading signal structure."""
    signal_id: str
    timestamp: float
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    price: float
    amount: float
    strategy_tier: StrategyTier
    phase: TimePhase
    rsi_trigger: float
    volume_trigger: float
    hash_match: bool
    memory_recall: bool
    exchange: str
    priority: str  # critical, high, normal, low

@dataclass
class MemoryKey:
    """Memory key for strategy recall."""
    key_id: str
    timestamp: float
    hash_signature: str
    strategy_tier: StrategyTier
    phase: TimePhase
    rsi_value: float
    volume_value: float
    price_value: float
    outcome: str  # win, loss, neutral
    profit_loss: float
    strategy_used: str
    execution_time: float

# =============================================================================
# LIVE MARKET DATA INTEGRATION
# =============================================================================

class LiveMarketDataIntegration:
    """Live market data integration with real-time API feeds."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize live market data integration."""
        self.config = config
        
        # Exchange configurations
        self.coinbase_config = config.get('coinbase', {})
        self.kraken_config = config.get('kraken', {})
        self.binance_config = config.get('binance', {})
        self.binance_usa_config = config.get('binance_usa', {})
        self.finance_api_config = config.get('finance_api', {})
        
        # Finance API key
        self.finance_api_key = self.finance_api_config.get('api_key', '')
        
        # Trading parameters
        self.symbols = config.get('symbols', ['BTC/USD', 'ETH/USD'])
        self.update_interval = config.get('update_interval', 60)
        self.rsi_period = config.get('rsi_period', 14)
        self.volume_threshold = config.get('volume_threshold', 1000000)
        
        # System state
        self.running = False
        self.exchanges = {}
        self.market_data = {}
        self.trading_signals = []
        self.memory_keys = []
        self.data_cache = {}  # Add missing data cache
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_update': 0,
            'uptime_seconds': 0
        }
        
        # Initialize system
        self._initialize_exchanges()
        
        # Data storage
        self.memory_path = Path("memory_keys")
        self.memory_path.mkdir(exist_ok=True)
        
        # Performance tracking
        self.data_fetch_count = 0
        self.error_count = 0
        self.last_fetch_time = time.time()
        
        # Threading
        self.data_thread = None
        self.processing_thread = None
        
        logger.info("Live Market Data Integration initialized")
    
    def _initialize_exchanges(self):
        """Initialize exchange connections."""
        try:
            # Coinbase (updated - Coinbase Pro is deprecated, now unified)
            if self.coinbase_config:
                coinbase_config = {
                    'apiKey': self.coinbase_config.get('api_key'),
                    'secret': self.coinbase_config.get('secret'),
                    'password': self.coinbase_config.get('password')  # CCXT uses 'password' for passphrase
                }
                
                # Handle sandbox mode properly
                if self.coinbase_config.get('sandbox', False):
                    # For sandbox, we'll use testnet configuration
                    coinbase_config['sandbox'] = True
                    # Note: Coinbase may not support sandbox in newer CCXT versions
                    try:
                        # Use ONLY the current Coinbase exchange
                        self.exchanges['coinbase'] = ccxt.coinbase(coinbase_config)
                        logger.info("Coinbase sandbox initialized (current unified exchange)")
                    except Exception as sandbox_error:
                        logger.warning(f"Coinbase sandbox not available: {sandbox_error}")
                        # Fallback to regular Coinbase without sandbox
                        coinbase_config.pop('sandbox', None)
                        self.exchanges['coinbase'] = ccxt.coinbase(coinbase_config)
                        logger.info("Coinbase initialized (sandbox fallback - current unified exchange)")
                else:
                    # Use ONLY the current Coinbase exchange
                    self.exchanges['coinbase'] = ccxt.coinbase(coinbase_config)
                    logger.info("Coinbase initialized (current unified exchange)")
            
            # Kraken
            if self.kraken_config:
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': self.kraken_config.get('api_key'),
                    'secret': self.kraken_config.get('secret'),
                    'sandbox': self.kraken_config.get('sandbox', False)
                })
                logger.info("Kraken initialized")
            
            # Binance
            if self.binance_config:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.binance_config.get('api_key'),
                    'secret': self.binance_config.get('secret'),
                    'sandbox': self.binance_config.get('sandbox', False)
                })
                logger.info("Binance initialized")
            
            # Binance USA
            if self.binance_usa_config:
                self.exchanges['binance_usa'] = ccxt.binanceus({
                    'apiKey': self.binance_usa_config.get('api_key'),
                    'secret': self.binance_usa_config.get('secret'),
                    'sandbox': self.binance_usa_config.get('sandbox', False)
                })
                logger.info("Binance USA initialized")
            
            logger.info(f"Initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
            # Don't raise the exception, just log it and continue
            # This allows the system to work even if some exchanges fail
    
    def start_data_feed(self):
        """Start real-time data feed."""
        try:
            self.running = True
            
            # Start data fetching thread
            self.data_thread = threading.Thread(
                target=self._data_fetching_loop,
                daemon=True,
                name="DataFetcher"
            )
            self.data_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._data_processing_loop,
                daemon=True,
                name="DataProcessor"
            )
            self.processing_thread.start()
            
            logger.info("🚀 Live market data feed started")
            
        except Exception as e:
            logger.error(f"❌ Error starting data feed: {e}")
            raise
    
    def _data_fetching_loop(self):
        """Main data fetching loop."""
        while self.running:
            try:
                # Fetch data from all exchanges
                for exchange_name, exchange in self.exchanges.items():
                    self._fetch_exchange_data(exchange_name, exchange)
                
                # Fetch from Finance API
                if self.finance_api_key:
                    self._fetch_finance_api_data()
                
                # Update fetch metrics
                self.data_fetch_count += 1
                self.last_fetch_time = time.time()
                
                # Sleep between fetches
                time.sleep(1.0)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"❌ Data fetching error: {e}")
                self.error_count += 1
                time.sleep(5.0)  # Longer sleep on error
    
    def _fetch_exchange_data(self, exchange_name: str, exchange):
        """Fetch data from a specific exchange."""
        try:
            # Fetch OHLCV data for major pairs
            symbols = ['BTC/USDC', 'ETH/USDC', 'XRP/USDC', 'SOL/USDC']
            
            for symbol in symbols:
                try:
                    # Fetch OHLCV data
                    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=100)
                    
                    if ohlcv and len(ohlcv) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Calculate technical indicators
                        df = self._calculate_technical_indicators(df)
                        
                        # Get latest data point
                        latest = df.iloc[-1]
                        
                        # Create market data object
                        market_data = self._create_market_data(
                            exchange_name, symbol, latest, df
                        )
                        
                        # Store in cache
                        cache_key = f"{exchange_name}_{symbol}"
                        self.data_cache[cache_key] = market_data
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching {symbol} from {exchange_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error fetching from {exchange_name}: {e}")
    
    def _fetch_finance_api_data(self):
        """Fetch data from Finance API."""
        try:
            # Using Alpha Vantage as example
            base_url = "https://www.alphavantage.co/query"
            
            symbols = ['BTC', 'ETH', 'XRP', 'SOL']
            
            for symbol in symbols:
                try:
                    params = {
                        'function': 'TIME_SERIES_INTRADAY',
                        'symbol': symbol,
                        'interval': '5min',
                        'apikey': self.finance_api_key,
                        'outputsize': 'compact'
                    }
                    
                    # Make API request
                    response = requests.get(base_url, params=params)
                    data = response.json()
                    
                    if 'Time Series (5min)' in data:
                        # Process the data
                        time_series = data['Time Series (5min)']
                        latest_time = max(time_series.keys())
                        latest_data = time_series[latest_time]
                        
                        # Create market data object
                        market_data = self._create_finance_api_market_data(
                            symbol, latest_time, latest_data
                        )
                        
                        # Store in cache
                        cache_key = f"finance_api_{symbol}"
                        self.data_cache[cache_key] = market_data
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching {symbol} from Finance API: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error fetching from Finance API: {e}")
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        try:
            if TALIB_AVAILABLE and talib is not None:
                # Use TA-Lib for calculations
                df['rsi'] = talib.RSI(df['close'], timeperiod=14)
                df['rsi_1h'] = talib.RSI(df['close'], timeperiod=12)  # 1 hour equivalent
                df['rsi_4h'] = talib.RSI(df['close'], timeperiod=48)  # 4 hour equivalent
                
                # VWAP
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                # ATR (Average True Range)
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                
                # Additional indicators
                df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
                df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
                df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            else:
                # Fallback calculations without TA-Lib
                logger.info("📊 Using fallback technical indicator calculations")
                
                # Simple RSI calculation
                def calculate_rsi(prices, period=14):
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gains = pd.Series(gains).rolling(window=period).mean()
                    avg_losses = pd.Series(losses).rolling(window=period).mean()
                    
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                
                df['rsi'] = calculate_rsi(df['close'].values, 14)
                df['rsi_1h'] = calculate_rsi(df['close'].values, 12)
                df['rsi_4h'] = calculate_rsi(df['close'].values, 48)
                
                # VWAP
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                # Simple ATR calculation
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        np.abs(df['high'] - df['close'].shift(1)),
                        np.abs(df['low'] - df['close'].shift(1))
                    )
                )
                df['atr'] = df['tr'].rolling(window=14).mean()
                
                # Simple moving averages
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                
                # Simple MACD
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return df
    
    def _create_market_data(self, exchange: str, symbol: str, latest: pd.Series, df: pd.DataFrame) -> MarketData:
        """Create market data object from exchange data."""
        try:
            # Extract decimal key from price
            price = float(latest['close'])
            decimal_key = self._extract_decimal_key(price)
            
            # Determine time phase
            phase = self._determine_time_phase()
            
            # Determine strategy tier
            strategy_tier = self._determine_strategy_tier(decimal_key)
            
            # Generate hash signature
            hash_signature = self._generate_hash_signature(price, latest['rsi'], latest['volume'])
            
            # Generate memory key
            memory_key = self._generate_memory_key(hash_signature, strategy_tier, phase)
            
            # Get order book depth (simplified)
            order_book_depth = {
                'bids': latest['volume'] * 0.8,
                'asks': latest['volume'] * 1.2
            }
            
            return MarketData(
                timestamp=latest['timestamp'] / 1000,  # Convert to seconds
                symbol=symbol,
                exchange=exchange,
                price=price,
                volume=float(latest['volume']),
                rsi=float(latest['rsi']),
                rsi_14=float(latest['rsi']),
                rsi_1h=float(latest['rsi_1h']),
                rsi_4h=float(latest['rsi_4h']),
                vwap=float(latest['vwap']),
                atr=float(latest['atr']),
                high=float(latest['high']),
                low=float(latest['low']),
                open_price=float(latest['open']),
                close_price=float(latest['close']),
                bid=price * 0.999,  # Approximate bid
                ask=price * 1.001,  # Approximate ask
                spread=price * 0.002,
                order_book_depth=order_book_depth,
                phase=phase,
                strategy_tier=strategy_tier,
                decimal_key=decimal_key,
                hash_signature=hash_signature,
                memory_key=memory_key
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating market data: {e}")
            raise
    
    def _create_finance_api_market_data(self, symbol: str, timestamp: str, data: Dict) -> MarketData:
        """Create market data object from Finance API data."""
        try:
            # Parse timestamp
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            timestamp_float = dt.timestamp()
            
            # Extract data
            price = float(data['4. close'])
            volume = float(data['5. volume'])
            high = float(data['2. high'])
            low = float(data['3. low'])
            open_price = float(data['1. open'])
            
            # Calculate RSI (simplified)
            rsi = 50.0  # Placeholder - would need historical data for proper calculation
            
            # Extract decimal key
            decimal_key = self._extract_decimal_key(price)
            
            # Determine phase and tier
            phase = self._determine_time_phase()
            strategy_tier = self._determine_strategy_tier(decimal_key)
            
            # Generate signatures
            hash_signature = self._generate_hash_signature(price, rsi, volume)
            memory_key = self._generate_memory_key(hash_signature, strategy_tier, phase)
            
            return MarketData(
                timestamp=timestamp_float,
                symbol=f"{symbol}/USD",
                exchange="finance_api",
                price=price,
                volume=volume,
                rsi=rsi,
                rsi_14=rsi,
                rsi_1h=rsi,
                rsi_4h=rsi,
                vwap=price,  # Simplified
                atr=high - low,  # Simplified
                high=high,
                low=low,
                open_price=open_price,
                close_price=price,
                bid=price * 0.999,
                ask=price * 1.001,
                spread=price * 0.002,
                order_book_depth={'bids': volume * 0.8, 'asks': volume * 1.2},
                phase=phase,
                strategy_tier=strategy_tier,
                decimal_key=decimal_key,
                hash_signature=hash_signature,
                memory_key=memory_key
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating Finance API market data: {e}")
            raise
    
    def _extract_decimal_key(self, price: float) -> str:
        """Extract decimal key from price for strategy tier mapping."""
        try:
            price_str = f"{price:.2f}"
            decimal_part = price_str.split('.')[-1]
            return decimal_part[-2:]  # Last 2 digits
        except Exception as e:
            logger.error(f"❌ Error extracting decimal key: {e}")
            return "00"
    
    def _determine_time_phase(self) -> TimePhase:
        """Determine current time phase based on UTC time."""
        try:
            utc_hour = datetime.utcnow().hour
            
            if utc_hour == 0:
                return TimePhase.MIDNIGHT
            elif 1 <= utc_hour <= 2:
                return TimePhase.PRE_DAWN
            elif 3 <= utc_hour <= 11:
                return TimePhase.MORNING
            elif utc_hour == 12:
                return TimePhase.HIGH_NOON
            elif 13 <= utc_hour <= 19:
                return TimePhase.LATE_NOON
            elif 20 <= utc_hour <= 22:
                return TimePhase.EVENING
            else:  # 23
                return TimePhase.MIDNIGHT_PLUS
                
        except Exception as e:
            logger.error(f"❌ Error determining time phase: {e}")
            return TimePhase.MORNING
    
    def _determine_strategy_tier(self, decimal_key: str) -> StrategyTier:
        """Determine strategy tier based on decimal key."""
        try:
            first_digit = int(decimal_key[0])
            
            if first_digit in [2, 3, 4]:
                return StrategyTier.TIER_2
            elif first_digit in [6, 7]:
                return StrategyTier.TIER_6
            elif first_digit in [8, 9]:
                return StrategyTier.TIER_8
            else:
                return StrategyTier.TIER_6  # Default
                
        except Exception as e:
            logger.error(f"❌ Error determining strategy tier: {e}")
            return StrategyTier.TIER_6
    
    def _generate_hash_signature(self, price: float, rsi: float, volume: float) -> str:
        """Generate hash signature for market data."""
        try:
            # Create hash input
            hash_input = f"{price:.2f}_{rsi:.2f}_{volume:.2f}_{int(time.time())}"
            
            # Generate SHA-256 hash
            hash_obj = hashlib.sha256(hash_input.encode())
            return hash_obj.hexdigest()[:16]  # First 16 characters
            
        except Exception as e:
            logger.error(f"❌ Error generating hash signature: {e}")
            return "0000000000000000"
    
    def _generate_memory_key(self, hash_signature: str, strategy_tier: StrategyTier, phase: TimePhase) -> str:
        """Generate memory key for strategy recall."""
        try:
            timestamp = int(time.time())
            memory_key = f"{strategy_tier.value}_{phase.value}_{hash_signature[:8]}_{timestamp}"
            return memory_key
            
        except Exception as e:
            logger.error(f"❌ Error generating memory key: {e}")
            return f"default_{int(time.time())}"
    
    def _data_processing_loop(self):
        """Process market data and generate trading signals."""
        while self.running:
            try:
                # Process cached data
                for cache_key, market_data in self.data_cache.items():
                    # Generate trading signals
                    signal = self._generate_trading_signal(market_data)
                    
                    if signal:
                        # Store signal
                        self.strategy_history.append(signal)
                        
                        # Store memory key
                        self._store_memory_key(market_data, signal)
                        
                        # Log signal
                        logger.info(f"📊 Signal generated: {signal.action} {signal.symbol} "
                                  f"@ {signal.price} (Tier {signal.strategy_tier.value})")
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep between processing
                time.sleep(5.0)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Data processing error: {e}")
                time.sleep(10.0)
    
    def _generate_trading_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate trading signal based on market data."""
        try:
            # Check if we should generate a signal
            if not self._should_generate_signal(market_data):
                return None
            
            # Determine action based on RSI and phase
            action = self._determine_action(market_data)
            
            if action == "hold":
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(market_data)
            
            # Check hash match
            hash_match = self._check_hash_match(market_data)
            
            # Check memory recall
            memory_recall = self._check_memory_recall(market_data)
            
            # Determine priority
            priority = self._determine_priority(market_data, action)
            
            # Calculate amount (simplified)
            amount = self._calculate_trade_amount(market_data, action)
            
            # Generate signal ID
            signal_id = f"signal_{int(time.time() * 1000000)}"
            
            return TradingSignal(
                signal_id=signal_id,
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                price=market_data.price,
                amount=amount,
                strategy_tier=market_data.strategy_tier,
                phase=market_data.phase,
                rsi_trigger=market_data.rsi,
                volume_trigger=market_data.volume,
                hash_match=hash_match,
                memory_recall=memory_recall,
                exchange=market_data.exchange,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signal: {e}")
            return None
    
    def _should_generate_signal(self, market_data: MarketData) -> bool:
        """Determine if we should generate a signal."""
        try:
            # Check RSI conditions
            rsi = market_data.rsi
            
            # RSI extremes
            if rsi < 20 or rsi > 80:
                return True
            
            # Phase-specific conditions
            if market_data.phase == TimePhase.MIDNIGHT and rsi < 40:
                return True
            
            if market_data.phase == TimePhase.HIGH_NOON and rsi > 70:
                return True
            
            if market_data.phase == TimePhase.EVENING and 30 < rsi < 50:
                return True
            
            # Volume spike
            if market_data.volume > self._get_average_volume(market_data.symbol) * 1.5:
                return True
            
            # Strategy tier conditions
            if market_data.strategy_tier == StrategyTier.TIER_8:
                return True  # High frequency
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking signal conditions: {e}")
            return False
    
    def _determine_action(self, market_data: MarketData) -> str:
        """Determine trading action based on market data."""
        try:
            rsi = market_data.rsi
            phase = market_data.phase
            
            # RSI-based actions
            if rsi < 30:
                return "buy"
            elif rsi > 70:
                return "sell"
            
            # Phase-based actions
            if phase == TimePhase.MIDNIGHT and rsi < 40:
                return "buy"
            
            if phase == TimePhase.HIGH_NOON and rsi > 60:
                return "sell"
            
            if phase == TimePhase.EVENING and 30 < rsi < 50:
                return "buy"
            
            # Default
            return "hold"
            
        except Exception as e:
            logger.error(f"❌ Error determining action: {e}")
            return "hold"
    
    def _calculate_confidence(self, market_data: MarketData) -> float:
        """Calculate signal confidence."""
        try:
            confidence = 0.5  # Base confidence
            
            # RSI confidence
            rsi = market_data.rsi
            if rsi < 20 or rsi > 80:
                confidence += 0.3
            elif rsi < 30 or rsi > 70:
                confidence += 0.2
            
            # Volume confidence
            avg_volume = self._get_average_volume(market_data.symbol)
            if market_data.volume > avg_volume * 2:
                confidence += 0.2
            elif market_data.volume > avg_volume * 1.5:
                confidence += 0.1
            
            # Phase confidence
            if market_data.phase in [TimePhase.MIDNIGHT, TimePhase.HIGH_NOON]:
                confidence += 0.1
            
            # Strategy tier confidence
            if market_data.strategy_tier == StrategyTier.TIER_8:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {e}")
            return 0.5
    
    def _check_hash_match(self, market_data: MarketData) -> bool:
        """Check if hash matches previous successful patterns."""
        try:
            # Check memory for similar hash patterns
            memory_file = self.memory_path / f"tier_{market_data.strategy_tier.value}" / f"{market_data.hash_signature[:8]}.json"
            
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                # Check if previous outcome was positive
                if memory_data.get('outcome') == 'win':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking hash match: {e}")
            return False
    
    def _check_memory_recall(self, market_data: MarketData) -> bool:
        """Check memory recall for similar conditions."""
        try:
            # Check for similar conditions in memory
            memory_key = market_data.memory_key
            
            if memory_key in self.memory_keys:
                previous_outcome = self.memory_keys[memory_key].get('outcome')
                if previous_outcome == 'win':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking memory recall: {e}")
            return False
    
    def _determine_priority(self, market_data: MarketData, action: str) -> str:
        """Determine signal priority."""
        try:
            # Critical conditions
            if market_data.rsi < 15 or market_data.rsi > 85:
                return "critical"
            
            if market_data.strategy_tier == StrategyTier.TIER_8:
                return "high"
            
            if action in ["buy", "sell"] and market_data.phase in [TimePhase.MIDNIGHT, TimePhase.HIGH_NOON]:
                return "high"
            
            return "normal"
            
        except Exception as e:
            logger.error(f"❌ Error determining priority: {e}")
            return "normal"
    
    def _calculate_trade_amount(self, market_data: MarketData, action: str) -> float:
        """Calculate trade amount based on strategy tier and confidence."""
        try:
            base_amount = 0.01  # Base amount in BTC
            
            # Adjust by strategy tier
            if market_data.strategy_tier == StrategyTier.TIER_2:
                base_amount *= 0.5
            elif market_data.strategy_tier == StrategyTier.TIER_6:
                base_amount *= 1.0
            elif market_data.strategy_tier == StrategyTier.TIER_8:
                base_amount *= 2.0
            
            # Adjust by volume
            volume_factor = min(market_data.volume / 1000, 2.0)  # Cap at 2x
            base_amount *= volume_factor
            
            return base_amount
            
        except Exception as e:
            logger.error(f"❌ Error calculating trade amount: {e}")
            return 0.01
    
    def _get_average_volume(self, symbol: str) -> float:
        """Get average volume for symbol (simplified)."""
        try:
            # In a real implementation, this would calculate from historical data
            # For now, return a reasonable default
            return 1000.0
            
        except Exception as e:
            logger.error(f"❌ Error getting average volume: {e}")
            return 1000.0
    
    def _store_memory_key(self, market_data: MarketData, signal: TradingSignal):
        """Store memory key for future recall."""
        try:
            # Create memory key object
            memory_key = MemoryKey(
                key_id=market_data.memory_key,
                timestamp=market_data.timestamp,
                hash_signature=market_data.hash_signature,
                strategy_tier=market_data.strategy_tier,
                phase=market_data.phase,
                rsi_value=market_data.rsi,
                volume_value=market_data.volume,
                price_value=market_data.price,
                outcome="pending",  # Will be updated after execution
                profit_loss=0.0,
                strategy_used=signal.action,
                execution_time=time.time()
            )
            
            # Store in memory
            self.memory_keys[market_data.memory_key] = {
                'key_id': memory_key.key_id,
                'timestamp': memory_key.timestamp,
                'hash_signature': memory_key.hash_signature,
                'strategy_tier': memory_key.strategy_tier.value,
                'phase': memory_key.phase.value,
                'rsi_value': memory_key.rsi_value,
                'volume_value': memory_key.volume_value,
                'price_value': memory_key.price_value,
                'outcome': memory_key.outcome,
                'profit_loss': memory_key.profit_loss,
                'strategy_used': memory_key.strategy_used,
                'execution_time': memory_key.execution_time
            }
            
            # Save to file
            self._save_memory_key_to_file(memory_key)
            
        except Exception as e:
            logger.error(f"❌ Error storing memory key: {e}")
    
    def _save_memory_key_to_file(self, memory_key: MemoryKey):
        """Save memory key to file system."""
        try:
            # Create tier directory
            tier_dir = self.memory_path / f"tier_{memory_key.strategy_tier.value}"
            tier_dir.mkdir(exist_ok=True)
            
            # Create file path
            file_path = tier_dir / f"{memory_key.hash_signature[:8]}.json"
            
            # Save data
            with open(file_path, 'w') as f:
                json.dump({
                    'key_id': memory_key.key_id,
                    'timestamp': memory_key.timestamp,
                    'hash_signature': memory_key.hash_signature,
                    'strategy_tier': memory_key.strategy_tier.value,
                    'phase': memory_key.phase.value,
                    'rsi_value': memory_key.rsi_value,
                    'volume_value': memory_key.volume_value,
                    'price_value': memory_key.price_value,
                    'outcome': memory_key.outcome,
                    'profit_loss': memory_key.profit_loss,
                    'strategy_used': memory_key.strategy_used,
                    'execution_time': memory_key.execution_time
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving memory key to file: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks."""
        try:
            current_time = time.time()
            
            # Clean old market data (older than 1 hour)
            old_keys = []
            for key, market_data in self.data_cache.items():
                if current_time - market_data.timestamp > 3600:  # 1 hour
                    old_keys.append(key)
            
            for key in old_keys:
                del self.data_cache[key]
            
            # Clean old signals (older than 24 hours)
            self.strategy_history = [
                signal for signal in self.strategy_history
                if current_time - signal.timestamp < 86400  # 24 hours
            ]
            
            # Clean old memory keys (older than 7 days)
            old_memory_keys = []
            for key, memory_data in self.memory_keys.items():
                if current_time - memory_data['timestamp'] > 604800:  # 7 days
                    old_memory_keys.append(key)
            
            for key in old_memory_keys:
                del self.memory_keys[key]
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
    
    def get_latest_market_data(self, symbol: str, exchange: str = None) -> Optional[MarketData]:
        """Get latest market data for a symbol."""
        try:
            if exchange:
                cache_key = f"{exchange}_{symbol}"
            else:
                # Find any exchange with this symbol
                for key in self.data_cache.keys():
                    if symbol in key:
                        cache_key = key
                        break
                else:
                    return None
            
            return self.data_cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"❌ Error getting latest market data: {e}")
            return None
    
    def get_trading_signals(self, symbol: str = None, limit: int = 100) -> List[TradingSignal]:
        """Get recent trading signals."""
        try:
            signals = self.strategy_history
            
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Sort by timestamp (newest first)
            signals.sort(key=lambda x: x.timestamp, reverse=True)
            
            return signals[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting trading signals: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            return {
                'running': self.running,
                'data_fetch_count': self.data_fetch_count,
                'error_count': self.error_count,
                'last_fetch_time': self.last_fetch_time,
                'cache_size': len(self.data_cache),
                'signal_count': len(self.strategy_history),
                'memory_key_count': len(self.memory_keys),
                'exchanges_connected': len(self.exchanges)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown the market data integration."""
        try:
            logger.info("🛑 Shutting down Live Market Data Integration...")
            
            self.running = False
            
            # Wait for threads to finish
            if self.data_thread:
                self.data_thread.join(timeout=10.0)
            
            if self.processing_thread:
                self.processing_thread.join(timeout=10.0)
            
            logger.info("✅ Live Market Data Integration shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for live market data integration demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'coinbase': {
            'api_key': 'your_coinbase_api_key',
            'secret': 'your_coinbase_secret',
            'password': 'your_coinbase_password',
            'sandbox': True
        },
        'kraken': {
            'api_key': 'your_kraken_api_key',
            'secret': 'your_kraken_secret',
            'sandbox': True
        },
        'finance_api': {
            'api_key': 'your_finance_api_key'
        }
    }
    
    # Initialize market data integration
    market_integration = LiveMarketDataIntegration(config)
    
    try:
        print("🔧 Live Market Data Integration Demo")
        print("=" * 50)
        
        # Start data feed
        market_integration.start_data_feed()
        
        print("📡 Live data feed started")
        print("⏳ Collecting market data...")
        
        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            
            # Print status every 10 seconds
            if (i + 1) % 10 == 0:
                status = market_integration.get_system_status()
                print(f"📊 Status: {status['data_fetch_count']} fetches, "
                      f"{status['signal_count']} signals, "
                      f"{status['error_count']} errors")
        
        # Get latest data
        print("\n📈 Latest Market Data:")
        print("-" * 40)
        
        symbols = ['BTC/USDC', 'ETH/USDC', 'XRP/USDC']
        for symbol in symbols:
            data = market_integration.get_latest_market_data(symbol)
            if data:
                print(f"  {symbol}: ${data.price:.2f} | RSI: {data.rsi:.1f} | "
                      f"Volume: {data.volume:.0f} | Tier: {data.strategy_tier.value}")
        
        # Get recent signals
        print("\n📊 Recent Trading Signals:")
        print("-" * 40)
        
        signals = market_integration.get_trading_signals(limit=10)
        for signal in signals:
            print(f"  {signal.symbol}: {signal.action.upper()} @ ${signal.price:.2f} "
                  f"(Confidence: {signal.confidence:.1%}, Tier: {signal.strategy_tier.value})")
        
        # Final status
        print("\nFinal System Status:")
        print("-" * 40)
        final_status = market_integration.get_system_status()
        for key, value in final_status.items():
            print(f"  {key}: {value}")
        
    finally:
        # Shutdown
        market_integration.shutdown()


if __name__ == "__main__":
    main() 