#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 REAL-TIME MARKET DATA PIPELINE - ADVANCED DATA PROCESSING
===========================================================

Unified Market Data Pipeline for Schwabot Trading System

This pipeline provides advanced market data processing and analysis:
- Multi-source data aggregation and validation
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Sentiment analysis and on-chain metrics integration
- Standardized MarketDataPacket output
- Real-time data cleaning and optimization
- Caching and failover mechanisms
- Registry-compatible data structure
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import core components
from distributed_mathematical_processor import DistributedMathematicalProcessor
from enhanced_error_recovery_system import EnhancedErrorRecoverySystem, error_recovery_decorator
from neural_processing_engine import NeuralProcessingEngine
from unified_profit_vectorization_system import UnifiedProfitVectorizationSystem

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources for market data."""
    COINGECKO = "coingecko"
    GLASSNODE = "glassnode"
    COINMARKETCAP = "coinmarketcap"
    FEAR_GREED = "fear_greed"
    WHALE_ALERT = "whale_alert"
    SIMULATED = "simulated"  # For testing


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # All sources available, fresh data
    GOOD = "good"  # Most sources available, recent data
    ACCEPTABLE = "acceptable"  # Some sources available, older data
    POOR = "poor"  # Limited sources, stale data
    FAILED = "failed"  # No reliable data available


@dataclass
class TechnicalIndicators:
    """Technical indicators calculated from price data."""
    rsi_14: float = 50.0
    rsi_21: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5
    sma_20: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0
    atr_14: float = 0.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0


@dataclass
class OnChainMetrics:
    """On-chain metrics from Glassnode and similar sources."""
    hash_rate: float = 0.0
    difficulty: float = 0.0
    active_addresses: int = 0
    transaction_count: int = 0
    mvrv_ratio: float = 1.0
    nvt_ratio: float = 50.0
    sopr: float = 1.0
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    whale_activity_score: float = 0.0
    network_health_score: float = 50.0
    hodl_waves: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketSentiment:
    """Market sentiment indicators."""
    fear_greed_index: float = 50.0
    fear_greed_classification: str = "Neutral"
    social_volume: float = 0.0
    social_sentiment: float = 0.0
    reddit_sentiment: float = 0.0
    twitter_sentiment: float = 0.0
    news_sentiment: float = 0.0
    trending_score: float = 0.0
    market_dominance: float = 0.0


@dataclass
class MarketDataPacket:
    """Standardized market data packet for trading pipeline input."""
    # Basic price data
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    volume_change_24h: float
    
    # Time and quality
    timestamp: float
    data_quality: DataQuality
    source_count: int
    freshness_score: float
    
    # Technical indicators
    technical_indicators: TechnicalIndicators
    
    # On-chain metrics
    onchain_metrics: OnChainMetrics
    
    # Sentiment data
    market_sentiment: MarketSentiment
    
    # Derived metrics for trading system
    volatility: float = 0.0
    trend_strength: float = 0.0
    entropy_level: float = 4.0
    liquidity_score: float = 0.0
    momentum_score: float = 0.0
    
    # API source tracking
    sources_used: List[str] = field(default_factory=list)
    api_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Registry-compatible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Pipeline performance and health metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    cache_hit_rate: float = 0.0
    data_quality_score: float = 0.0
    uptime_percentage: float = 100.0
    last_update: float = 0.0


class RealTimeMarketDataPipeline:
    """
    📊 Unified Market Data Pipeline that aggregates, cleans, and optimizes
    market data from multiple sources for consistent trading system input.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the unified market data pipeline."""
        self.config = config or self._default_config()
        
        # Initialize core components
        self.distributed_processor = DistributedMathematicalProcessor()
        self.neural_engine = NeuralProcessingEngine()
        self.profit_vectorizer = UnifiedProfitVectorizationSystem()
        self.error_recovery = EnhancedErrorRecoverySystem()
        
        # Data storage
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.data_cache: Dict[str, MarketDataPacket] = {}
        
        # Performance tracking
        self.pipeline_metrics = PipelineMetrics()
        self.last_cache_cleanup = time.time()
        
        # Data sources configuration
        self.enabled_sources = self.config.get("enabled_sources", [DataSource.SIMULATED])
        self.cache_duration = self.config.get("cache_duration", 60)  # seconds
        self.max_history_length = self.config.get("max_history_length", 1000)
        
        logger.info("📊 Real-Time Market Data Pipeline initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the pipeline."""
        return {
            "enabled_sources": [DataSource.SIMULATED],
            "cache_duration": 60,  # seconds
            "max_history_length": 1000,
            "request_timeout": 10.0,
            "retry_attempts": 3,
            "data_quality_threshold": 0.7,
            "technical_indicators": {
                "rsi_periods": [14, 21],
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2,
                "atr_period": 14,
                "stoch_period": 14
            },
            "sentiment_weights": {
                "fear_greed": 0.3,
                "social": 0.2,
                "news": 0.2,
                "onchain": 0.3
            }
        }
    
    @error_recovery_decorator
    async def fetch_market_data(self, symbol: str = "BTC/USDC") -> MarketDataPacket:
        """Fetch and process market data for the given symbol."""
        try:
            start_time = time.time()
            
            # Check cache first
            if self._is_cached(symbol):
                cached_data = self.data_cache[symbol]
                if time.time() - cached_data.timestamp < self.cache_duration:
                    self.pipeline_metrics.cache_hit_rate = (
                        (self.pipeline_metrics.cache_hit_rate * 0.9) + 0.1
                    )
                    return cached_data
            
            # Fetch raw data from enabled sources
            raw_data = await self._fetch_raw_data(symbol)
            
            # Process and validate data
            processed_data = await self._process_market_data(symbol, raw_data)
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(symbol)
            
            # Extract on-chain metrics
            onchain_metrics = self._extract_onchain_metrics(raw_data)
            
            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(raw_data)
            
            # Assess data quality
            data_quality, freshness_score = self._assess_data_quality(raw_data)
            
            # Create market data packet
            market_packet = MarketDataPacket(
                symbol=symbol,
                price=processed_data.get("price", 50000.0),
                volume_24h=processed_data.get("volume_24h", 1000000.0),
                market_cap=processed_data.get("market_cap", 1000000000000.0),
                price_change_24h=processed_data.get("price_change_24h", 0.0),
                volume_change_24h=processed_data.get("volume_change_24h", 0.0),
                timestamp=time.time(),
                data_quality=data_quality,
                source_count=len(raw_data),
                freshness_score=freshness_score,
                technical_indicators=technical_indicators,
                onchain_metrics=onchain_metrics,
                market_sentiment=market_sentiment,
                volatility=processed_data.get("volatility", 0.02),
                trend_strength=processed_data.get("trend_strength", 0.0),
                entropy_level=processed_data.get("entropy_level", 4.0),
                liquidity_score=processed_data.get("liquidity_score", 0.0),
                momentum_score=processed_data.get("momentum_score", 0.0),
                sources_used=list(raw_data.keys()),
                api_latencies=processed_data.get("api_latencies", {}),
                metadata=processed_data.get("metadata", {})
            )
            
            # Cache the data
            self._cache_data(symbol, market_packet)
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(True, latency)
            
            logger.info(f"📊 Market data fetched for {symbol}: {data_quality.value} quality")
            return market_packet
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            self._update_metrics(False, 0.0)
            return self._create_fallback_packet(symbol)
    
    async def _fetch_raw_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw data from enabled sources."""
        try:
            raw_data = {}
            
            for source in self.enabled_sources:
                try:
                    if source == DataSource.SIMULATED:
                        source_data = await self._fetch_simulated_data(symbol)
                    elif source == DataSource.COINGECKO:
                        source_data = await self._fetch_coingecko_data(symbol)
                    elif source == DataSource.GLASSNODE:
                        source_data = await self._fetch_glassnode_data(symbol)
                    else:
                        continue
                    
                    if source_data:
                        raw_data[source.value] = source_data
                        
                except Exception as e:
                    logger.warning(f"Error fetching from {source.value}: {e}")
                    continue
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Error fetching raw data: {e}")
            return {}
    
    async def _fetch_simulated_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch simulated market data for testing."""
        try:
            # Generate realistic simulated data
            base_price = 50000.0
            price_variation = np.random.normal(0, 0.02)  # 2% variation
            current_price = base_price * (1 + price_variation)
            
            volume_variation = np.random.normal(0, 0.1)  # 10% variation
            current_volume = 1000000.0 * (1 + volume_variation)
            
            return {
                "price": current_price,
                "volume_24h": current_volume,
                "market_cap": current_price * 19000000,  # Approx BTC supply
                "price_change_24h": price_variation * 100,
                "volume_change_24h": volume_variation * 100,
                "volatility": abs(price_variation) * 10,
                "trend_strength": np.random.normal(0, 0.5),
                "entropy_level": np.random.uniform(3.5, 4.5),
                "liquidity_score": np.random.uniform(0.7, 1.0),
                "momentum_score": np.random.normal(0, 0.3),
                "timestamp": time.time(),
                "source": "simulated"
            }
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}")
            return {}
    
    async def _fetch_coingecko_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from CoinGecko API."""
        try:
            # This would integrate with actual CoinGecko API
            # For now, return simulated data
            return await self._fetch_simulated_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return {}
    
    async def _fetch_glassnode_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch on-chain data from Glassnode API."""
        try:
            # This would integrate with actual Glassnode API
            # For now, return simulated on-chain data
            return {
                "hash_rate": np.random.uniform(200, 300),  # EH/s
                "difficulty": np.random.uniform(50000000000000, 60000000000000),
                "active_addresses": np.random.randint(800000, 1200000),
                "transaction_count": np.random.randint(200000, 300000),
                "mvrv_ratio": np.random.uniform(0.8, 1.5),
                "nvt_ratio": np.random.uniform(30, 70),
                "sopr": np.random.uniform(0.95, 1.05),
                "exchange_inflow": np.random.uniform(1000, 5000),
                "exchange_outflow": np.random.uniform(1000, 5000),
                "whale_activity_score": np.random.uniform(0.3, 0.8),
                "network_health_score": np.random.uniform(40, 80),
                "hodl_waves": {
                    "1d": np.random.uniform(0.1, 0.3),
                    "1w": np.random.uniform(0.2, 0.4),
                    "1m": np.random.uniform(0.3, 0.5),
                    "3m": np.random.uniform(0.4, 0.6),
                    "6m": np.random.uniform(0.5, 0.7),
                    "1y": np.random.uniform(0.6, 0.8),
                    "2y": np.random.uniform(0.7, 0.9)
                },
                "timestamp": time.time(),
                "source": "glassnode"
            }
            
        except Exception as e:
            logger.error(f"Error fetching Glassnode data: {e}")
            return {}
    
    async def _process_market_data(self, symbol: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate raw market data."""
        try:
            processed_data = {}
            
            # Aggregate data from multiple sources
            prices = []
            volumes = []
            market_caps = []
            price_changes = []
            volume_changes = []
            
            for source, data in raw_data.items():
                if "price" in data:
                    prices.append(data["price"])
                if "volume_24h" in data:
                    volumes.append(data["volume_24h"])
                if "market_cap" in data:
                    market_caps.append(data["market_cap"])
                if "price_change_24h" in data:
                    price_changes.append(data["price_change_24h"])
                if "volume_change_24h" in data:
                    volume_changes.append(data["volume_change_24h"])
            
            # Calculate aggregated values
            if prices:
                processed_data["price"] = np.median(prices)
                self._update_price_history(symbol, processed_data["price"])
            
            if volumes:
                processed_data["volume_24h"] = np.median(volumes)
                self._update_volume_history(symbol, processed_data["volume_24h"])
            
            if market_caps:
                processed_data["market_cap"] = np.median(market_caps)
            
            if price_changes:
                processed_data["price_change_24h"] = np.median(price_changes)
            
            if volume_changes:
                processed_data["volume_change_24h"] = np.median(volume_changes)
            
            # Calculate derived metrics
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                price_array = np.array(self.price_history[symbol])
                processed_data.update(self._calculate_derived_metrics(processed_data, price_array))
            
            # Add metadata
            processed_data["metadata"] = {
                "sources_count": len(raw_data),
                "processing_timestamp": time.time(),
                "data_freshness": self._calculate_freshness_score(raw_data)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {}
    
    def _calculate_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators from price history."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
                return TechnicalIndicators()
            
            prices = np.array(self.price_history[symbol])
            
            # Calculate RSI
            rsi_14 = self._calculate_rsi(prices, 14)
            rsi_21 = self._calculate_rsi(prices, 21)
            
            # Calculate MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_position = self._calculate_bollinger_bands(prices)
            
            # Calculate moving averages
            sma_20 = self._calculate_sma(prices, 20)
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            # Calculate volume indicators
            volume_sma_20 = 0.0
            volume_ratio = 1.0
            if symbol in self.volume_history and len(self.volume_history[symbol]) >= 20:
                volumes = np.array(self.volume_history[symbol])
                volume_sma_20 = np.mean(volumes[-20:])
                volume_ratio = volumes[-1] / volume_sma_20 if volume_sma_20 > 0 else 1.0
            
            # Calculate ATR
            atr_14 = self._calculate_atr(prices, 14)
            
            # Calculate Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(prices, 14)
            
            return TechnicalIndicators(
                rsi_14=rsi_14,
                rsi_21=rsi_21,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_position=bb_position,
                sma_20=sma_20,
                ema_12=ema_12,
                ema_26=ema_26,
                volume_sma_20=volume_sma_20,
                volume_ratio=volume_ratio,
                atr_14=atr_14,
                stoch_k=stoch_k,
                stoch_d=stoch_d
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()
    
    def _extract_onchain_metrics(self, raw_data: Dict[str, Any]) -> OnChainMetrics:
        """Extract on-chain metrics from raw data."""
        try:
            # Look for Glassnode data
            glassnode_data = raw_data.get("glassnode", {})
            
            return OnChainMetrics(
                hash_rate=glassnode_data.get("hash_rate", 0.0),
                difficulty=glassnode_data.get("difficulty", 0.0),
                active_addresses=glassnode_data.get("active_addresses", 0),
                transaction_count=glassnode_data.get("transaction_count", 0),
                mvrv_ratio=glassnode_data.get("mvrv_ratio", 1.0),
                nvt_ratio=glassnode_data.get("nvt_ratio", 50.0),
                sopr=glassnode_data.get("sopr", 1.0),
                exchange_inflow=glassnode_data.get("exchange_inflow", 0.0),
                exchange_outflow=glassnode_data.get("exchange_outflow", 0.0),
                whale_activity_score=glassnode_data.get("whale_activity_score", 0.0),
                network_health_score=glassnode_data.get("network_health_score", 50.0),
                hodl_waves=glassnode_data.get("hodl_waves", {})
            )
            
        except Exception as e:
            logger.error(f"Error extracting on-chain metrics: {e}")
            return OnChainMetrics()
    
    def _calculate_market_sentiment(self, raw_data: Dict[str, Any]) -> MarketSentiment:
        """Calculate market sentiment from raw data."""
        try:
            # Simulate sentiment data (would integrate with real APIs)
            fear_greed_index = np.random.uniform(20, 80)
            
            # Classify fear/greed
            if fear_greed_index < 25:
                classification = "Extreme Fear"
            elif fear_greed_index < 45:
                classification = "Fear"
            elif fear_greed_index < 55:
                classification = "Neutral"
            elif fear_greed_index < 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            return MarketSentiment(
                fear_greed_index=fear_greed_index,
                fear_greed_classification=classification,
                social_volume=np.random.uniform(0, 100),
                social_sentiment=np.random.uniform(-1, 1),
                reddit_sentiment=np.random.uniform(-1, 1),
                twitter_sentiment=np.random.uniform(-1, 1),
                news_sentiment=np.random.uniform(-1, 1),
                trending_score=np.random.uniform(0, 1),
                market_dominance=np.random.uniform(40, 60)
            )
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return MarketSentiment()
    
    def _calculate_derived_metrics(self, data: Dict[str, Any], prices: np.ndarray) -> Dict[str, float]:
        """Calculate derived metrics for trading system."""
        try:
            derived_metrics = {}
            
            # Calculate volatility
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                derived_metrics["volatility"] = np.std(returns) * np.sqrt(252)
            
            # Calculate trend strength
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                current_price = prices[-1]
                derived_metrics["trend_strength"] = (current_price - sma_20) / sma_20
            
            # Calculate entropy level
            if len(prices) > 1:
                price_changes = np.diff(prices)
                entropy = -np.sum(np.abs(price_changes)) / len(price_changes)
                derived_metrics["entropy_level"] = max(0, min(10, entropy + 4))
            
            # Calculate liquidity score
            volume = data.get("volume_24h", 1000000.0)
            market_cap = data.get("market_cap", 1000000000000.0)
            derived_metrics["liquidity_score"] = min(1.0, volume / market_cap * 1000000)
            
            # Calculate momentum score
            if len(prices) >= 10:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                derived_metrics["momentum_score"] = (short_ma - long_ma) / long_ma
            
            return derived_metrics
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100.0
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if len(prices) < 26:
                return 0.0, 0.0, 0.0
            
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd_line = ema_12 - ema_26
            
            # Calculate signal line (EMA of MACD)
            if len(prices) >= 35:  # Need more data for signal line
                macd_values = []
                for i in range(9):
                    if len(prices) - 26 - i >= 0:
                        ema_12_i = self._calculate_ema(prices[:-(26+i)], 12)
                        ema_26_i = self._calculate_ema(prices[:-(26+i)], 26)
                        macd_values.append(ema_12_i - ema_26_i)
                
                if macd_values:
                    macd_signal = np.mean(macd_values)
                else:
                    macd_signal = macd_line
            else:
                macd_signal = macd_line
            
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        try:
            if len(prices) < period:
                return np.mean(prices)
            
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.mean(prices) if len(prices) > 0 else 0.0
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average."""
        try:
            if len(prices) < period:
                return np.mean(prices)
            
            return np.mean(prices[-period:])
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return np.mean(prices) if len(prices) > 0 else 0.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < period:
                return 0.0, np.mean(prices), 0.0, 0.5
            
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            bb_upper = sma + (2 * std)
            bb_middle = sma
            bb_lower = sma - (2 * std)
            
            # Calculate position within bands
            current_price = prices[-1]
            if bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            
            return bb_upper, bb_middle, bb_lower, bb_position
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.0, np.mean(prices), 0.0, 0.5
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            true_ranges = []
            for i in range(1, len(prices)):
                high = prices[i]
                low = prices[i]
                prev_close = prices[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            if len(true_ranges) >= period:
                return np.mean(true_ranges[-period:])
            else:
                return np.mean(true_ranges)
                
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        try:
            if len(prices) < period:
                return 50.0, 50.0
            
            # Calculate %K
            lowest_low = np.min(prices[-period:])
            highest_high = np.max(prices[-period:])
            current_close = prices[-1]
            
            if highest_high != lowest_low:
                stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                stoch_k = 50.0
            
            # Calculate %D (3-period SMA of %K)
            stoch_d = stoch_k  # Simplified for now
            
            return stoch_k, stoch_d
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return 50.0, 50.0
    
    def _assess_data_quality(self, raw_data: Dict[str, Any]) -> Tuple[DataQuality, float]:
        """Assess the quality of raw data."""
        try:
            if not raw_data:
                return DataQuality.FAILED, 0.0
            
            # Count available sources
            source_count = len(raw_data)
            
            # Check data freshness
            current_time = time.time()
            freshness_scores = []
            
            for source, data in raw_data.items():
                timestamp = data.get("timestamp", current_time)
                age = current_time - timestamp
                
                if age < 60:  # Less than 1 minute
                    freshness_scores.append(1.0)
                elif age < 300:  # Less than 5 minutes
                    freshness_scores.append(0.8)
                elif age < 900:  # Less than 15 minutes
                    freshness_scores.append(0.6)
                elif age < 3600:  # Less than 1 hour
                    freshness_scores.append(0.4)
                else:
                    freshness_scores.append(0.2)
            
            avg_freshness = np.mean(freshness_scores) if freshness_scores else 0.0
            
            # Determine quality level
            if source_count >= 3 and avg_freshness > 0.8:
                quality = DataQuality.EXCELLENT
            elif source_count >= 2 and avg_freshness > 0.6:
                quality = DataQuality.GOOD
            elif source_count >= 1 and avg_freshness > 0.4:
                quality = DataQuality.ACCEPTABLE
            elif source_count >= 1 and avg_freshness > 0.2:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.FAILED
            
            return quality, avg_freshness
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return DataQuality.FAILED, 0.0
    
    def _calculate_freshness_score(self, raw_data: Dict[str, Any]) -> float:
        """Calculate data freshness score."""
        try:
            if not raw_data:
                return 0.0
            
            current_time = time.time()
            freshness_scores = []
            
            for source, data in raw_data.items():
                timestamp = data.get("timestamp", current_time)
                age = current_time - timestamp
                
                if age < 60:
                    freshness_scores.append(1.0)
                elif age < 300:
                    freshness_scores.append(0.8)
                elif age < 900:
                    freshness_scores.append(0.6)
                elif age < 3600:
                    freshness_scores.append(0.4)
                else:
                    freshness_scores.append(0.2)
            
            return np.mean(freshness_scores) if freshness_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating freshness score: {e}")
            return 0.0
    
    def _is_cached(self, symbol: str) -> bool:
        """Check if data is cached for the symbol."""
        return symbol in self.data_cache
    
    def _cache_data(self, symbol: str, data: MarketDataPacket) -> None:
        """Cache market data for the symbol."""
        try:
            self.data_cache[symbol] = data
            
            # Clean up old cache entries
            current_time = time.time()
            if current_time - self.last_cache_cleanup > 300:  # Every 5 minutes
                self._cleanup_cache()
                self.last_cache_cleanup = current_time
                
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        try:
            current_time = time.time()
            expired_symbols = []
            
            for symbol, data in self.data_cache.items():
                if current_time - data.timestamp > self.cache_duration:
                    expired_symbols.append(symbol)
            
            for symbol in expired_symbols:
                del self.data_cache[symbol]
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def _update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for the symbol."""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(price)
            
            # Keep only recent history
            if len(self.price_history[symbol]) > self.max_history_length:
                self.price_history[symbol] = self.price_history[symbol][-self.max_history_length:]
                
        except Exception as e:
            logger.error(f"Error updating price history: {e}")
    
    def _update_volume_history(self, symbol: str, volume: float) -> None:
        """Update volume history for the symbol."""
        try:
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            
            self.volume_history[symbol].append(volume)
            
            # Keep only recent history
            if len(self.volume_history[symbol]) > self.max_history_length:
                self.volume_history[symbol] = self.volume_history[symbol][-self.max_history_length:]
                
        except Exception as e:
            logger.error(f"Error updating volume history: {e}")
    
    def _update_metrics(self, success: bool, latency: float) -> None:
        """Update pipeline performance metrics."""
        try:
            self.pipeline_metrics.total_requests += 1
            
            if success:
                self.pipeline_metrics.successful_requests += 1
            else:
                self.pipeline_metrics.failed_requests += 1
            
            # Update average latency
            if self.pipeline_metrics.total_requests > 0:
                self.pipeline_metrics.average_latency = (
                    (self.pipeline_metrics.average_latency * (self.pipeline_metrics.total_requests - 1) + latency) / 
                    self.pipeline_metrics.total_requests
                )
            
            # Update data quality score
            if self.pipeline_metrics.total_requests > 0:
                self.pipeline_metrics.data_quality_score = (
                    self.pipeline_metrics.successful_requests / self.pipeline_metrics.total_requests
                )
            
            self.pipeline_metrics.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _create_fallback_packet(self, symbol: str) -> MarketDataPacket:
        """Create fallback market data packet."""
        try:
            return MarketDataPacket(
                symbol=symbol,
                price=50000.0,
                volume_24h=1000000.0,
                market_cap=1000000000000.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=time.time(),
                data_quality=DataQuality.FAILED,
                source_count=0,
                freshness_score=0.0,
                technical_indicators=TechnicalIndicators(),
                onchain_metrics=OnChainMetrics(),
                market_sentiment=MarketSentiment(),
                volatility=0.02,
                trend_strength=0.0,
                entropy_level=4.0,
                liquidity_score=0.0,
                momentum_score=0.0,
                sources_used=[],
                api_latencies={},
                metadata={"fallback": True}
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback packet: {e}")
            return MarketDataPacket(
                symbol=symbol,
                price=50000.0,
                volume_24h=1000000.0,
                market_cap=1000000000000.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=time.time(),
                data_quality=DataQuality.FAILED,
                source_count=0,
                freshness_score=0.0,
                technical_indicators=TechnicalIndicators(),
                onchain_metrics=OnChainMetrics(),
                market_sentiment=MarketSentiment(),
                sources_used=[],
                api_latencies={},
                metadata={"error": str(e)}
            )
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        try:
            return {
                "pipeline_metrics": self.pipeline_metrics.__dict__,
                "enabled_sources": [source.value for source in self.enabled_sources],
                "cache_size": len(self.data_cache),
                "price_history_size": sum(len(history) for history in self.price_history.values()),
                "volume_history_size": sum(len(history) for history in self.volume_history.values()),
                "config": self.config,
                "last_cache_cleanup": self.last_cache_cleanup
            }
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pipeline."""
        try:
            # Test data fetching
            test_packet = await self.fetch_market_data("BTC/USDC")
            
            health_status = {
                "status": "healthy" if test_packet.data_quality != DataQuality.FAILED else "unhealthy",
                "data_quality": test_packet.data_quality.value,
                "source_count": test_packet.source_count,
                "freshness_score": test_packet.freshness_score,
                "success_rate": self.pipeline_metrics.data_quality_score,
                "average_latency": self.pipeline_metrics.average_latency,
                "uptime_percentage": self.pipeline_metrics.uptime_percentage
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


def create_real_time_market_data_pipeline(config: Optional[Dict[str, Any]] = None) -> RealTimeMarketDataPipeline:
    """Factory function to create a Real-Time Market Data Pipeline."""
    return RealTimeMarketDataPipeline(config)


# Global instance for easy access
real_time_market_data_pipeline = create_real_time_market_data_pipeline() 