#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 DYNAMIC POSITION SIZING - SCHWABOT
=====================================

Advanced dynamic position sizing system that adapts position sizes based on:
- Signal confidence levels
- Market volatility factors
- Portfolio heat and risk metrics
- Correlation analysis

Features:
- Adaptive position sizing based on market conditions
- Risk-adjusted position allocation
- Portfolio heat monitoring
- Confidence-based scaling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class PositionSizeType(Enum):
    """Types of position sizing strategies."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class PositionSizeSignal:
    """Dynamic position sizing signal."""
    timestamp: float
    base_position_size: float
    adjusted_position_size: float
    confidence_multiplier: float
    volatility_factor: float
    portfolio_heat_factor: float
    final_multiplier: float
    recommendation: str
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class PortfolioHeat:
    """Portfolio heat metrics."""
    timestamp: float
    total_exposure: float
    max_drawdown: float
    volatility: float
    correlation_risk: float
    heat_score: float
    risk_level: str

class DynamicPositionSizing:
    """Advanced dynamic position sizing system."""
    
    def __init__(self):
        # Base configuration
        self.base_position_size = 0.1  # 10% base position size
        self.max_position_size = 0.25  # 25% maximum position size
        self.min_position_size = 0.01  # 1% minimum position size
        
        # Risk parameters
        self.max_portfolio_heat = 0.8
        self.volatility_threshold_high = 0.05  # 5% volatility
        self.volatility_threshold_low = 0.01   # 1% volatility
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.4
        
        # Portfolio tracking
        self.portfolio_history: List[PortfolioHeat] = []
        self.position_signals: List[PositionSizeSignal] = []
        
        # Performance metrics
        self.total_positions = 0
        self.successful_positions = 0
        
        logger.info("💰 Dynamic Position Sizing initialized")
    
    def calculate_position_size(self, signal_confidence: float, market_volatility: float,
                              portfolio_heat: float, correlation_risk: float = 0.0) -> PositionSizeSignal:
        """Calculate dynamic position size based on multiple factors."""
        try:
            # Base position size
            base_size = self.base_position_size
            
            # Calculate confidence multiplier
            confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
            
            # Calculate volatility factor
            volatility_factor = self._calculate_volatility_factor(market_volatility)
            
            # Calculate portfolio heat factor
            portfolio_heat_factor = self._calculate_portfolio_heat_factor(portfolio_heat)
            
            # Calculate correlation risk factor
            correlation_factor = self._calculate_correlation_factor(correlation_risk)
            
            # Calculate final multiplier
            final_multiplier = (
                confidence_multiplier * 0.4 +
                volatility_factor * 0.3 +
                portfolio_heat_factor * 0.2 +
                correlation_factor * 0.1
            )
            
            # Calculate adjusted position size
            adjusted_size = base_size * final_multiplier
            
            # Apply limits
            adjusted_size = max(self.min_position_size, min(self.max_position_size, adjusted_size))
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                adjusted_size, signal_confidence, market_volatility, portfolio_heat
            )
            
            # Create signal
            signal = PositionSizeSignal(
                timestamp=time.time(),
                base_position_size=base_size,
                adjusted_position_size=adjusted_size,
                confidence_multiplier=confidence_multiplier,
                volatility_factor=volatility_factor,
                portfolio_heat_factor=portfolio_heat_factor,
                final_multiplier=final_multiplier,
                recommendation=recommendation,
                metadata={
                    'correlation_factor': correlation_factor,
                    'market_volatility': market_volatility,
                    'portfolio_heat': portfolio_heat,
                    'signal_confidence': signal_confidence
                }
            )
            
            self.position_signals.append(signal)
            self.total_positions += 1
            
            logger.info(f"💰 Position Size: {adjusted_size:.3f} ({final_multiplier:.2f}x) - {recommendation}")
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self._create_fallback_signal()
    
    def update_portfolio_heat(self, total_exposure: float, max_drawdown: float,
                            volatility: float, correlation_risk: float) -> PortfolioHeat:
        """Update portfolio heat metrics."""
        try:
            # Calculate heat score
            heat_score = self._calculate_heat_score(total_exposure, max_drawdown, volatility, correlation_risk)
            
            # Determine risk level
            if heat_score > 0.7:
                risk_level = "HIGH"
            elif heat_score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Create portfolio heat record
            heat_record = PortfolioHeat(
                timestamp=time.time(),
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                volatility=volatility,
                correlation_risk=correlation_risk,
                heat_score=heat_score,
                risk_level=risk_level
            )
            
            self.portfolio_history.append(heat_record)
            
            # Maintain history size
            if len(self.portfolio_history) > 100:
                self.portfolio_history.pop(0)
            
            logger.info(f"🔥 Portfolio Heat: {heat_score:.3f} ({risk_level})")
            return heat_record
            
        except Exception as e:
            logger.error(f"Error updating portfolio heat: {e}")
            return self._create_fallback_heat()
    
    def get_position_sizing_recommendations(self) -> Dict[str, any]:
        """Get position sizing recommendations and statistics."""
        try:
            if not self.position_signals:
                return {"message": "No position signals available"}
            
            # Analyze recent signals
            recent_signals = self.position_signals[-10:]  # Last 10 signals
            
            # Calculate statistics
            avg_position_size = np.mean([s.adjusted_position_size for s in recent_signals])
            avg_confidence = np.mean([s.confidence_multiplier for s in recent_signals])
            avg_volatility_factor = np.mean([s.volatility_factor for s in recent_signals])
            avg_heat_factor = np.mean([s.portfolio_heat_factor for s in recent_signals])
            
            # Get current portfolio heat
            current_heat = self.portfolio_history[-1] if self.portfolio_history else None
            
            # Generate recommendations
            recommendations = []
            
            if avg_position_size < self.base_position_size * 0.5:
                recommendations.append("Consider increasing position sizes - low risk environment")
            elif avg_position_size > self.base_position_size * 1.5:
                recommendations.append("Consider reducing position sizes - high risk environment")
            
            if avg_confidence < self.low_confidence_threshold:
                recommendations.append("Low signal confidence - reduce position sizes")
            elif avg_confidence > self.high_confidence_threshold:
                recommendations.append("High signal confidence - consider larger positions")
            
            if current_heat and current_heat.heat_score > 0.7:
                recommendations.append("High portfolio heat - reduce exposure")
            elif current_heat and current_heat.heat_score < 0.3:
                recommendations.append("Low portfolio heat - room for increased exposure")
            
            return {
                "average_position_size": avg_position_size,
                "average_confidence": avg_confidence,
                "average_volatility_factor": avg_volatility_factor,
                "average_heat_factor": avg_heat_factor,
                "current_portfolio_heat": current_heat.heat_score if current_heat else 0.0,
                "current_risk_level": current_heat.risk_level if current_heat else "UNKNOWN",
                "recommendations": recommendations,
                "total_positions_analyzed": self.total_positions,
                "success_rate": self.successful_positions / max(1, self.total_positions)
            }
            
        except Exception as e:
            logger.error(f"Error getting position sizing recommendations: {e}")
            return {"error": str(e)}
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position size multiplier based on signal confidence."""
        try:
            if confidence >= self.high_confidence_threshold:
                # High confidence: increase position size
                multiplier = 1.0 + (confidence - self.high_confidence_threshold) * 2.0
            elif confidence <= self.low_confidence_threshold:
                # Low confidence: decrease position size
                multiplier = 0.5 + (confidence / self.low_confidence_threshold) * 0.3
            else:
                # Medium confidence: moderate position size
                multiplier = 0.8 + (confidence - self.low_confidence_threshold) * 0.4
            
            return max(0.1, min(2.0, multiplier))
            
        except Exception as e:
            logger.error(f"Error calculating confidence multiplier: {e}")
            return 1.0
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Calculate position size factor based on market volatility."""
        try:
            if volatility >= self.volatility_threshold_high:
                # High volatility: reduce position size
                factor = 0.5 + (self.volatility_threshold_high / volatility) * 0.3
            elif volatility <= self.volatility_threshold_low:
                # Low volatility: increase position size
                factor = 1.0 + (self.volatility_threshold_low - volatility) * 5.0
            else:
                # Medium volatility: moderate position size
                factor = 0.8 + (self.volatility_threshold_high - volatility) * 0.4
            
            return max(0.3, min(1.5, factor))
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")
            return 1.0
    
    def _calculate_portfolio_heat_factor(self, portfolio_heat: float) -> float:
        """Calculate position size factor based on portfolio heat."""
        try:
            if portfolio_heat >= self.max_portfolio_heat:
                # High heat: significantly reduce position size
                factor = 0.3
            elif portfolio_heat >= 0.6:
                # Medium-high heat: reduce position size
                factor = 0.5 + (self.max_portfolio_heat - portfolio_heat) * 0.5
            elif portfolio_heat >= 0.3:
                # Medium heat: moderate position size
                factor = 0.8 + (0.6 - portfolio_heat) * 0.4
            else:
                # Low heat: increase position size
                factor = 1.0 + (0.3 - portfolio_heat) * 0.5
            
            return max(0.2, min(1.3, factor))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat factor: {e}")
            return 1.0
    
    def _calculate_correlation_factor(self, correlation_risk: float) -> float:
        """Calculate position size factor based on correlation risk."""
        try:
            if correlation_risk >= 0.8:
                # High correlation risk: reduce position size
                factor = 0.6
            elif correlation_risk >= 0.5:
                # Medium correlation risk: moderate position size
                factor = 0.8
            else:
                # Low correlation risk: normal position size
                factor = 1.0
            
            return factor
            
        except Exception as e:
            logger.error(f"Error calculating correlation factor: {e}")
            return 1.0
    
    def _calculate_heat_score(self, total_exposure: float, max_drawdown: float,
                            volatility: float, correlation_risk: float) -> float:
        """Calculate portfolio heat score."""
        try:
            # Normalize factors
            exposure_factor = min(1.0, total_exposure / 2.0)  # Normalize to 200% exposure
            drawdown_factor = min(1.0, max_drawdown / 0.2)    # Normalize to 20% drawdown
            volatility_factor = min(1.0, volatility / 0.1)    # Normalize to 10% volatility
            correlation_factor = correlation_risk
            
            # Calculate weighted heat score
            heat_score = (
                exposure_factor * 0.3 +
                drawdown_factor * 0.3 +
                volatility_factor * 0.2 +
                correlation_factor * 0.2
            )
            
            return max(0.0, min(1.0, heat_score))
            
        except Exception as e:
            logger.error(f"Error calculating heat score: {e}")
            return 0.5
    
    def _generate_recommendation(self, position_size: float, confidence: float,
                               volatility: float, portfolio_heat: float) -> str:
        """Generate position sizing recommendation."""
        try:
            recommendations = []
            
            if position_size < self.base_position_size * 0.5:
                recommendations.append("CONSERVATIVE")
            elif position_size > self.base_position_size * 1.5:
                recommendations.append("AGGRESSIVE")
            else:
                recommendations.append("MODERATE")
            
            if confidence < self.low_confidence_threshold:
                recommendations.append("LOW_CONFIDENCE")
            elif confidence > self.high_confidence_threshold:
                recommendations.append("HIGH_CONFIDENCE")
            
            if volatility > self.volatility_threshold_high:
                recommendations.append("HIGH_VOLATILITY")
            elif volatility < self.volatility_threshold_low:
                recommendations.append("LOW_VOLATILITY")
            
            if portfolio_heat > self.max_portfolio_heat:
                recommendations.append("HIGH_HEAT")
            elif portfolio_heat < 0.3:
                recommendations.append("LOW_HEAT")
            
            return " | ".join(recommendations)
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "UNKNOWN"
    
    def _create_fallback_signal(self) -> PositionSizeSignal:
        """Create a fallback position sizing signal."""
        return PositionSizeSignal(
            timestamp=time.time(),
            base_position_size=self.base_position_size,
            adjusted_position_size=self.base_position_size * 0.5,
            confidence_multiplier=0.5,
            volatility_factor=1.0,
            portfolio_heat_factor=1.0,
            final_multiplier=0.5,
            recommendation="FALLBACK - Use conservative sizing"
        )
    
    def _create_fallback_heat(self) -> PortfolioHeat:
        """Create a fallback portfolio heat record."""
        return PortfolioHeat(
            timestamp=time.time(),
            total_exposure=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            correlation_risk=0.0,
            heat_score=0.5,
            risk_level="UNKNOWN"
        )

# Global instance
position_sizing = DynamicPositionSizing()

def get_position_sizing() -> DynamicPositionSizing:
    """Get the global position sizing instance."""
    return position_sizing 