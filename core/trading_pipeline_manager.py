#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Trading Pipeline Manager
================================

Connects AI integration with trading engine to create a functional trading bot.
This manager handles the complete pipeline from market data to trade execution.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from schwabot_ai_integration import SchwabotAIIntegration, AnalysisType, SchwabotRequest, SchwabotResponse

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Market data point for pipeline processing."""
    timestamp: float
    symbol: str
    price: float
    volume: float
    price_change: float = 0.0
    volatility: float = 0.0
    sentiment: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingDecision:
    """Trading decision with complete analysis."""
    timestamp: float
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    position_size: float
    reasoning: str
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioState:
    """Current portfolio state."""
    balance: float
    positions: Dict[str, float]  # symbol -> quantity
    risk_per_trade: float = 0.02
    total_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingPipelineManager:
    """Manages the complete trading pipeline from data to execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the trading pipeline manager."""
        self.config = config or self._default_config()
        
        # Initialize AI integration
        self.ai_integration = SchwabotAIIntegration()
        
        # Pipeline state
        self.market_data_buffer: List[MarketDataPoint] = []
        self.portfolio_state = PortfolioState(balance=10000.0, positions={})
        self.trading_decisions: List[TradingDecision] = []
        self.executed_trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Pipeline status
        self.running = False
        self.last_analysis_time = 0.0
        
        logger.info("🚀 Trading Pipeline Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the trading pipeline."""
        return {
            "analysis_interval": 60,  # seconds
            "max_buffer_size": 100,
            "min_confidence": 0.7,
            "risk_per_trade": 0.02,
            "stop_loss_percent": 0.02,
            "take_profit_percent": 0.03,
            "enable_ai_analysis": True,
            "enable_pattern_recognition": True,
            "enable_sentiment_analysis": True,
            "enable_technical_analysis": True
        }
    
    async def start_pipeline(self):
        """Start the trading pipeline."""
        try:
            logger.info("🚀 Starting Trading Pipeline...")
            self.running = True
            
            # Start AI integration
            await self.ai_integration.start_schwabot_ai_server()
            
            logger.info("✅ Trading Pipeline started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading pipeline: {e}")
            raise
    
    async def stop_pipeline(self):
        """Stop the trading pipeline."""
        try:
            logger.info("🛑 Stopping Trading Pipeline...")
            self.running = False
            
            # Stop AI integration
            await self.ai_integration.stop_schwabot_ai_server()
            
            logger.info("✅ Trading Pipeline stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop trading pipeline: {e}")
    
    async def process_market_data(self, market_data: MarketDataPoint) -> Optional[TradingDecision]:
        """Process market data through the complete pipeline."""
        try:
            # Add to buffer
            self.market_data_buffer.append(market_data)
            
            # Keep buffer size manageable
            if len(self.market_data_buffer) > self.config["max_buffer_size"]:
                self.market_data_buffer.pop(0)
            
            # Check if it's time for analysis
            if time.time() - self.last_analysis_time >= self.config["analysis_interval"]:
                return await self._perform_complete_analysis(market_data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Market data processing failed: {e}")
            return None
    
    async def _perform_complete_analysis(self, market_data: MarketDataPoint) -> Optional[TradingDecision]:
        """Perform complete AI analysis and generate trading decision."""
        try:
            logger.info(f"🔍 Performing complete analysis for {market_data.symbol}")
            
            # Prepare market context
            market_context = {
                "price": market_data.price,
                "volume": market_data.volume,
                "price_change": market_data.price_change,
                "volatility": market_data.volatility,
                "sentiment": market_data.sentiment,
                "symbol": market_data.symbol,
                "timestamp": market_data.timestamp
            }
            
            # Perform multiple AI analyses
            analyses = {}
            
            if self.config["enable_ai_analysis"]:
                # Market analysis
                market_analysis = await self.ai_integration.analyze_market(
                    market_context, AnalysisType.MARKET_ANALYSIS
                )
                analyses["market_analysis"] = market_analysis
            
            if self.config["enable_pattern_recognition"]:
                # Pattern recognition
                pattern_analysis = await self.ai_integration.analyze_market(
                    market_context, AnalysisType.PATTERN_RECOGNITION
                )
                analyses["pattern_recognition"] = pattern_analysis
            
            if self.config["enable_sentiment_analysis"]:
                # Sentiment analysis
                sentiment_analysis = await self.ai_integration.analyze_market(
                    market_context, AnalysisType.SENTIMENT_ANALYSIS
                )
                analyses["sentiment_analysis"] = sentiment_analysis
            
            if self.config["enable_technical_analysis"]:
                # Technical analysis
                technical_analysis = await self.ai_integration.analyze_market(
                    market_context, AnalysisType.TECHNICAL_ANALYSIS
                )
                analyses["technical_analysis"] = technical_analysis
            
            # Generate trading decision
            trading_decision = await self.ai_integration.make_trading_decision(
                market_context, self.portfolio_state.__dict__
            )
            
            # Parse trading decision
            decision = self._parse_trading_decision(trading_decision, market_data, analyses)
            
            # Update last analysis time
            self.last_analysis_time = time.time()
            
            if decision and decision.confidence >= self.config["min_confidence"]:
                logger.info(f"🎯 Trading decision generated: {decision.action} {decision.symbol}")
                self.trading_decisions.append(decision)
                return decision
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Complete analysis failed: {e}")
            return None
    
    def _parse_trading_decision(self, ai_response: SchwabotResponse, market_data: MarketDataPoint, analyses: Dict[str, SchwabotResponse]) -> Optional[TradingDecision]:
        """Parse AI response into structured trading decision."""
        try:
            # Extract decision from AI response
            response_text = ai_response.response
            
            # Parse action from response
            action = "HOLD"
            if "DECISION: BUY" in response_text:
                action = "BUY"
            elif "DECISION: SELL" in response_text:
                action = "SELL"
            
            # Extract confidence
            confidence = ai_response.confidence
            
            # Calculate position details
            entry_price = market_data.price
            stop_loss = entry_price * (0.98 if action == "BUY" else 1.02)
            target_price = entry_price * (1.03 if action == "BUY" else 0.97)
            
            # Calculate position size
            max_risk_amount = self.portfolio_state.balance * self.config["risk_per_trade"]
            stop_loss_distance = abs(entry_price - stop_loss)
            position_size = max_risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
            
            # Extract reasoning
            reasoning = "AI analysis completed"
            if "Decision Reasoning:" in response_text:
                reasoning_start = response_text.find("Decision Reasoning:")
                reasoning_end = response_text.find("⚠️", reasoning_start)
                if reasoning_end == -1:
                    reasoning_end = response_text.find("🔄", reasoning_start)
                if reasoning_end == -1:
                    reasoning_end = len(response_text)
                reasoning = response_text[reasoning_start:reasoning_end].strip()
            
            # Prepare analysis summary
            ai_analysis = {
                "market_analysis": analyses.get("market_analysis", {}).response if "market_analysis" in analyses else "",
                "pattern_recognition": analyses.get("pattern_recognition", {}).response if "pattern_recognition" in analyses else "",
                "sentiment_analysis": analyses.get("sentiment_analysis", {}).response if "sentiment_analysis" in analyses else "",
                "technical_analysis": analyses.get("technical_analysis", {}).response if "technical_analysis" in analyses else "",
                "trading_decision": ai_response.response
            }
            
            return TradingDecision(
                timestamp=time.time(),
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=position_size,
                reasoning=reasoning,
                ai_analysis=ai_analysis,
                metadata={
                    "market_data": market_data.__dict__,
                    "portfolio_state": self.portfolio_state.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse trading decision: {e}")
            return None
    
    async def execute_trade(self, decision: TradingDecision) -> bool:
        """Execute a trading decision."""
        try:
            logger.info(f"💰 Executing trade: {decision.action} {decision.symbol}")
            
            # Simulate trade execution (replace with actual API call)
            trade_result = {
                "timestamp": time.time(),
                "symbol": decision.symbol,
                "action": decision.action,
                "entry_price": decision.entry_price,
                "quantity": decision.position_size,
                "stop_loss": decision.stop_loss,
                "target_price": decision.target_price,
                "status": "executed",
                "order_id": f"order_{int(time.time() * 1000)}"
            }
            
            # Update portfolio
            if decision.action == "BUY":
                cost = decision.entry_price * decision.position_size
                if cost <= self.portfolio_state.balance:
                    self.portfolio_state.balance -= cost
                    self.portfolio_state.positions[decision.symbol] = decision.position_size
                    logger.info(f"✅ Buy order executed: {decision.position_size} {decision.symbol} at ${decision.entry_price:.4f}")
                else:
                    logger.warning(f"⚠️ Insufficient balance for buy order")
                    return False
            
            elif decision.action == "SELL":
                if decision.symbol in self.portfolio_state.positions:
                    quantity = self.portfolio_state.positions[decision.symbol]
                    revenue = decision.entry_price * quantity
                    self.portfolio_state.balance += revenue
                    del self.portfolio_state.positions[decision.symbol]
                    logger.info(f"✅ Sell order executed: {quantity} {decision.symbol} at ${decision.entry_price:.4f}")
                else:
                    logger.warning(f"⚠️ No position to sell for {decision.symbol}")
                    return False
            
            # Record trade
            self.executed_trades.append(trade_result)
            self.total_trades += 1
            
            logger.info(f"✅ Trade executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "running": self.running,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "portfolio_balance": self.portfolio_state.balance,
            "active_positions": len(self.portfolio_state.positions),
            "buffer_size": len(self.market_data_buffer),
            "last_analysis_time": self.last_analysis_time,
            "ai_integration_status": self.ai_integration.get_status()
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.trading_decisions[-limit:] if self.trading_decisions else []
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent executed trades."""
        return self.executed_trades[-limit:] if self.executed_trades else []

# Global instance for easy access
trading_pipeline = None

def get_trading_pipeline() -> TradingPipelineManager:
    """Get the global trading pipeline instance."""
    global trading_pipeline
    if trading_pipeline is None:
        trading_pipeline = TradingPipelineManager()
    return trading_pipeline 