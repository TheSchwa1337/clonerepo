#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CROSS-ASSET CORRELATION ENGINE - SCHWABOT (Windows Compatible)
=============================================================

Advanced cross-asset correlation analysis for portfolio optimization.
Analyzes BTC/ETH correlation, crypto vs traditional markets, and cross-exchange arbitrage.

Features:
- Real-time correlation analysis
- Cross-exchange arbitrage detection
- Portfolio optimization signals
- Risk diversification metrics

Windows Compatible: No emoji characters, text-only output
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CorrelationType(Enum):
    """Types of correlation analysis."""
    BTC_ETH = "btc_eth"
    CRYPTO_TRADITIONAL = "crypto_traditional"
    CROSS_EXCHANGE = "cross_exchange"
    SECTOR_ROTATION = "sector_rotation"

@dataclass
class CorrelationSignal:
    """Cross-asset correlation signal."""
    timestamp: float
    correlation_type: CorrelationType
    correlation_value: float
    confidence: float
    arbitrage_opportunity: Optional[float] = None
    portfolio_recommendation: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity."""
    exchange_a: str
    exchange_b: str
    symbol: str
    price_a: float
    price_b: float
    spread_percentage: float
    estimated_profit: float
    confidence: float
    timestamp: float

class CrossAssetCorrelationEngineWindows:
    """Advanced cross-asset correlation analysis engine (Windows Compatible)."""
    
    def __init__(self):
        self.correlation_history: Dict[str, List[float]] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.portfolio_signals: List[CorrelationSignal] = []
        
        # Correlation thresholds
        self.high_correlation_threshold = 0.7
        self.low_correlation_threshold = 0.3
        self.arbitrage_threshold = 0.5  # 0.5% minimum spread
        
        # Performance tracking
        self.total_signals = 0
        self.successful_arbitrages = 0
        
        logger.info("Cross-Asset Correlation Engine initialized (Windows Compatible)")
    
    def analyze_btc_eth_correlation(self, btc_price: float, eth_price: float, 
                                   btc_volume: float, eth_volume: float) -> CorrelationSignal:
        """Analyze BTC/ETH correlation for portfolio optimization."""
        try:
            # Store historical data
            if 'btc_eth_correlation' not in self.correlation_history:
                self.correlation_history['btc_eth_correlation'] = []
            
            # Calculate price ratio
            price_ratio = btc_price / eth_price if eth_price > 0 else 0
            
            # Calculate volume ratio
            volume_ratio = btc_volume / eth_volume if eth_volume > 0 else 0
            
            # Calculate correlation (simplified - in real implementation you'd use historical data)
            correlation = self._calculate_correlation_metric(price_ratio, volume_ratio)
            
            # Determine portfolio recommendation
            if correlation > self.high_correlation_threshold:
                recommendation = "HIGH_CORRELATION - Consider reducing overlap"
            elif correlation < self.low_correlation_threshold:
                recommendation = "LOW_CORRELATION - Good diversification"
            else:
                recommendation = "MODERATE_CORRELATION - Monitor closely"
            
            # Calculate confidence
            confidence = min(1.0, abs(correlation) * 1.2)
            
            signal = CorrelationSignal(
                timestamp=time.time(),
                correlation_type=CorrelationType.BTC_ETH,
                correlation_value=correlation,
                confidence=confidence,
                portfolio_recommendation=recommendation,
                metadata={
                    'price_ratio': price_ratio,
                    'volume_ratio': volume_ratio,
                    'btc_price': btc_price,
                    'eth_price': eth_price
                }
            )
            
            self.correlation_history['btc_eth_correlation'].append(correlation)
            self.portfolio_signals.append(signal)
            self.total_signals += 1
            
            logger.info(f"BTC/ETH Correlation: {correlation:.3f} - {recommendation}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing BTC/ETH correlation: {e}")
            return self._create_fallback_signal(CorrelationType.BTC_ETH)
    
    def detect_cross_exchange_arbitrage(self, exchange_data: Dict[str, Dict[str, float]]) -> List[ArbitrageOpportunity]:
        """Detect cross-exchange arbitrage opportunities."""
        try:
            opportunities = []
            
            # Compare prices across exchanges
            for symbol in ['BTC/USD', 'ETH/USD', 'BTC/USDT', 'ETH/USDT']:
                symbol_prices = {}
                
                # Collect prices for this symbol across exchanges
                for exchange, data in exchange_data.items():
                    if symbol in data:
                        symbol_prices[exchange] = data[symbol]
                
                # Find arbitrage opportunities
                if len(symbol_prices) >= 2:
                    exchanges = list(symbol_prices.keys())
                    prices = list(symbol_prices.values())
                    
                    for i in range(len(exchanges)):
                        for j in range(i + 1, len(exchanges)):
                            price_a = prices[i]
                            price_b = prices[j]
                            
                            if price_a > 0 and price_b > 0:
                                spread = abs(price_a - price_b)
                                spread_percentage = (spread / min(price_a, price_b)) * 100
                                
                                if spread_percentage >= self.arbitrage_threshold:
                                    # Calculate estimated profit (simplified)
                                    estimated_profit = spread * 0.1  # Assume 10% of spread as profit
                                    
                                    # Determine confidence based on spread size
                                    confidence = min(1.0, spread_percentage / 2.0)
                                    
                                    opportunity = ArbitrageOpportunity(
                                        exchange_a=exchanges[i],
                                        exchange_b=exchanges[j],
                                        symbol=symbol,
                                        price_a=price_a,
                                        price_b=price_b,
                                        spread_percentage=spread_percentage,
                                        estimated_profit=estimated_profit,
                                        confidence=confidence,
                                        timestamp=time.time()
                                    )
                                    
                                    opportunities.append(opportunity)
                                    logger.info(f"Arbitrage: {symbol} {exchanges[i]}->{exchanges[j]} {spread_percentage:.2f}%")
            
            self.arbitrage_opportunities.extend(opportunities)
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage: {e}")
            return []
    
    def analyze_crypto_traditional_correlation(self, crypto_data: Dict[str, float], 
                                             traditional_data: Dict[str, float]) -> CorrelationSignal:
        """Analyze correlation between crypto and traditional markets."""
        try:
            # Calculate correlation metrics
            crypto_volatility = self._calculate_volatility(list(crypto_data.values()))
            traditional_volatility = self._calculate_volatility(list(traditional_data.values()))
            
            # Calculate correlation (simplified)
            correlation = self._calculate_cross_market_correlation(crypto_data, traditional_data)
            
            # Determine market regime
            if correlation > self.high_correlation_threshold:
                regime = "HIGH_CORRELATION - Crypto following traditional markets"
            elif correlation < self.low_correlation_threshold:
                regime = "LOW_CORRELATION - Crypto decoupled from traditional markets"
            else:
                regime = "MODERATE_CORRELATION - Mixed signals"
            
            # Calculate confidence
            confidence = min(1.0, abs(correlation) * 1.1)
            
            signal = CorrelationSignal(
                timestamp=time.time(),
                correlation_type=CorrelationType.CRYPTO_TRADITIONAL,
                correlation_value=correlation,
                confidence=confidence,
                portfolio_recommendation=regime,
                metadata={
                    'crypto_volatility': crypto_volatility,
                    'traditional_volatility': traditional_volatility,
                    'regime': regime
                }
            )
            
            self.portfolio_signals.append(signal)
            self.total_signals += 1
            
            logger.info(f"Crypto-Traditional Correlation: {correlation:.3f} - {regime}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing crypto-traditional correlation: {e}")
            return self._create_fallback_signal(CorrelationType.CRYPTO_TRADITIONAL)
    
    def get_portfolio_optimization_signals(self) -> Dict[str, any]:
        """Get portfolio optimization recommendations."""
        try:
            if not self.portfolio_signals:
                return {"message": "No signals available"}
            
            # Analyze recent signals
            recent_signals = self.portfolio_signals[-10:]  # Last 10 signals
            
            # Calculate average correlations
            btc_eth_corr = np.mean([s.correlation_value for s in recent_signals 
                                  if s.correlation_type == CorrelationType.BTC_ETH])
            
            crypto_trad_corr = np.mean([s.correlation_value for s in recent_signals 
                                      if s.correlation_type == CorrelationType.CRYPTO_TRADITIONAL])
            
            # Generate recommendations
            recommendations = []
            
            if btc_eth_corr > self.high_correlation_threshold:
                recommendations.append("Consider reducing BTC/ETH overlap in portfolio")
            elif btc_eth_corr < self.low_correlation_threshold:
                recommendations.append("BTC/ETH provide good diversification")
            
            if crypto_trad_corr > self.high_correlation_threshold:
                recommendations.append("Crypto highly correlated with traditional markets")
            elif crypto_trad_corr < self.low_correlation_threshold:
                recommendations.append("Crypto decoupled - good for portfolio diversification")
            
            # Calculate arbitrage opportunities
            recent_arbitrage = [opp for opp in self.arbitrage_opportunities 
                              if time.time() - opp.timestamp < 300]  # Last 5 minutes
            
            return {
                "btc_eth_correlation": btc_eth_corr,
                "crypto_traditional_correlation": crypto_trad_corr,
                "recommendations": recommendations,
                "recent_arbitrage_opportunities": len(recent_arbitrage),
                "total_signals_analyzed": self.total_signals,
                "confidence_score": min(1.0, self.total_signals / 100.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio signals: {e}")
            return {"error": str(e)}
    
    def _calculate_correlation_metric(self, price_ratio: float, volume_ratio: float) -> float:
        """Calculate correlation metric from price and volume ratios."""
        try:
            # Normalize ratios
            norm_price = min(1.0, price_ratio / 1000.0)  # Normalize BTC/ETH ratio
            norm_volume = min(1.0, volume_ratio / 10.0)   # Normalize volume ratio
            
            # Calculate correlation (simplified)
            correlation = (norm_price + norm_volume) / 2.0
            return max(-1.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Error calculating correlation metric: {e}")
            return 0.0
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate volatility from price list."""
        try:
            if len(prices) < 2:
                return 0.0
            
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_cross_market_correlation(self, crypto_data: Dict[str, float], 
                                          traditional_data: Dict[str, float]) -> float:
        """Calculate correlation between crypto and traditional markets."""
        try:
            # Simplified correlation calculation
            crypto_values = list(crypto_data.values())
            traditional_values = list(traditional_data.values())
            
            if len(crypto_values) != len(traditional_values) or len(crypto_values) < 2:
                return 0.0
            
            # Calculate correlation coefficient
            crypto_mean = np.mean(crypto_values)
            traditional_mean = np.mean(traditional_values)
            
            numerator = sum((c - crypto_mean) * (t - traditional_mean) 
                          for c, t in zip(crypto_values, traditional_values))
            
            crypto_var = sum((c - crypto_mean) ** 2 for c in crypto_values)
            traditional_var = sum((t - traditional_mean) ** 2 for t in traditional_values)
            
            denominator = np.sqrt(crypto_var * traditional_var)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Error calculating cross-market correlation: {e}")
            return 0.0
    
    def _create_fallback_signal(self, correlation_type: CorrelationType) -> CorrelationSignal:
        """Create a fallback correlation signal."""
        return CorrelationSignal(
            timestamp=time.time(),
            correlation_type=correlation_type,
            correlation_value=0.0,
            confidence=0.0,
            portfolio_recommendation="No data available"
        )

# Global instance
correlation_engine_windows = CrossAssetCorrelationEngineWindows()

def get_correlation_engine_windows() -> CrossAssetCorrelationEngineWindows:
    """Get the global correlation engine instance (Windows Compatible)."""
    return correlation_engine_windows 