#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Comprehensive Backtesting Engine
=========================================

Complete backtesting system that can:
- Load real historical data from multiple sources
- Run the full Schwabot trading pipeline
- Test AI analysis and trading decisions
- Generate comprehensive performance reports
- Validate strategies before live trading
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass, field

# Add core directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "core"))

from trading_pipeline_manager import TradingPipelineManager, MarketDataPoint, TradingDecision
from schwabot_ai_integration import SchwabotAIIntegration, AnalysisType

# Import the new data sources
from .data_sources import DataSourceManager, DataSourceConfig, SimulatedDataGenerator

# Add mathematical integration import at the top
from .mathematical_integration_simplified import mathematical_integration, MathematicalSignal

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    symbols: List[str]
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    data_source: str = "auto"       # auto, binance, yahoo, simulated
    timeframe: str = "1h"           # 1m, 5m, 15m, 1h, 4h, 1d
    enable_ai_analysis: bool = True
    enable_risk_management: bool = True
    max_positions: int = 5
    risk_per_trade: float = 0.02
    min_confidence: float = 0.7     # Minimum confidence for trade execution

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    total_commission: float
    total_slippage: float
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class HistoricalDataLoader:
    """Load historical data from various sources."""
    
    def __init__(self):
        self.cache = {}
        # Initialize data source manager with fallback options
        configs = [
            DataSourceConfig(source_type="binance"),
            DataSourceConfig(source_type="yahoo")
        ]
        self.data_manager = DataSourceManager(configs)
        self.simulated_generator = SimulatedDataGenerator()
        
    async def load_binance_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """Load historical data from Binance API with fallback."""
        try:
            logger.info(f"📊 Loading data for {symbol} from {start_date} to {end_date}")
            
            # Try to get data from the data manager (with fallbacks)
            df = await self.data_manager.get_market_data(symbol, start_date, end_date, interval)
            
            if not df.empty:
                logger.info(f"✅ Loaded {len(df)} data points for {symbol}")
                return df
            else:
                logger.warning(f"⚠️ No data loaded from real sources, using simulated data")
                return self.simulated_generator.generate_ohlcv_data(start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"❌ Failed to load data for {symbol}: {e}")
            logger.info("🔄 Falling back to simulated data")
            return self.simulated_generator.generate_ohlcv_data(start_date, end_date, interval)
    
    async def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load historical data from CSV file."""
        try:
            logger.info(f"📊 Loading CSV data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            logger.info(f"✅ Loaded {len(df)} data points from CSV")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV data: {e}")
            return pd.DataFrame()
    
    def generate_simulated_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """Generate simulated data for testing."""
        logger.info(f"🎲 Generating simulated data for {symbol}")
        return self.simulated_generator.generate_ohlcv_data(start_date, end_date, interval)

    def convert_to_market_data_points(self, df: pd.DataFrame, symbol: str) -> List[MarketDataPoint]:
        """Convert DataFrame to MarketDataPoint objects."""
        market_data_points = []
        
        for index, row in df.iterrows():
            # Calculate price change
            if len(market_data_points) > 0:
                prev_price = market_data_points[-1].price
                price_change = (row["close"] - prev_price) / prev_price
            else:
                price_change = 0.0
            
            # Calculate volatility (simplified)
            volatility = abs(price_change)
            
            # Calculate sentiment (simplified)
            sentiment = 0.5 + (price_change * 2)  # 0.5 is neutral
            sentiment = max(0.0, min(1.0, sentiment))
            
            market_data = MarketDataPoint(
                timestamp=index.timestamp(),
                symbol=symbol,
                price=row["close"],
                volume=row["volume"],
                price_change=price_change,
                volatility=volatility,
                sentiment=sentiment,
                metadata={
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"]
                }
            )
            
            market_data_points.append(market_data)
        
        return market_data_points

class BacktestEngine:
    """Comprehensive backtesting engine for Schwabot trading system."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtesting engine."""
        self.config = config
        self.data_loader = HistoricalDataLoader()
        
        # Initialize mathematical integration
        self.mathematical_integration = mathematical_integration
        
        # Initialize trading pipeline
        pipeline_config = {
            "analysis_interval": 60,
            "max_buffer_size": 1000,
            "min_confidence": config.min_confidence,
            "risk_per_trade": config.risk_per_trade,
            "enable_ai_analysis": config.enable_ai_analysis,
            "enable_pattern_recognition": True,
            "enable_sentiment_analysis": True,
            "enable_technical_analysis": True
        }
        
        self.pipeline = TradingPipelineManager(pipeline_config)
        
        # Backtest state
        self.current_balance = config.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_balance = config.initial_balance
        
        # Performance tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Mathematical tracking
        self.mathematical_signals = []
        self.dualistic_consensus_history = []
        self.dlt_waveform_history = []
        self.bit_phase_history = []
        self.ferris_phase_history = []
        
        logger.info(f"🚀 Backtest Engine initialized with ${config.initial_balance:.2f} initial balance")
        logger.info(f"🧮 Mathematical integration enabled with all systems")
    
    async def run_backtest(self) -> BacktestResult:
        """Run complete backtest."""
        try:
            logger.info(f"🎯 Starting backtest from {self.config.start_date} to {self.config.end_date}")
            
            # Load historical data for all symbols
            all_market_data = {}
            
            for symbol in self.config.symbols:
                if self.config.data_source == "simulated":
                    # Use simulated data
                    df = self.data_loader.generate_simulated_data(symbol, self.config.start_date, self.config.end_date, self.config.timeframe)
                elif self.config.data_source == "csv":
                    df = await self.data_loader.load_csv_data(f"data/{symbol}.csv")
                else:
                    # Try real data sources with fallback to simulated
                    df = await self.data_loader.load_binance_data(
                        symbol, self.config.start_date, self.config.end_date, self.config.timeframe
                    )
                
                if not df.empty:
                    market_data_points = self.data_loader.convert_to_market_data_points(df, symbol)
                    all_market_data[symbol] = market_data_points
                    logger.info(f"✅ Loaded {len(market_data_points)} data points for {symbol}")
            
            if not all_market_data:
                logger.error("❌ No market data loaded")
                return self._create_empty_result()
            
            # Start trading pipeline
            await self.pipeline.start_pipeline()
            
            # Run backtest
            await self._run_backtest_loop(all_market_data)
            
            # Stop pipeline
            await self.pipeline.stop_pipeline()
            
            # Calculate results
            result = self._calculate_results()
            
            logger.info(f"✅ Backtest completed: {result.total_return:.2%} return, {result.total_trades} trades")
            return result
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return self._create_empty_result()
    
    async def _run_backtest_loop(self, all_market_data: Dict[str, List[MarketDataPoint]]):
        """Run the main backtest loop."""
        # Combine all market data and sort by timestamp
        all_data_points = []
        for symbol, data_points in all_market_data.items():
            all_data_points.extend(data_points)
        
        all_data_points.sort(key=lambda x: x.timestamp)
        
        logger.info(f"📊 Processing {len(all_data_points)} data points")
        
        for i, market_data in enumerate(all_data_points):
            try:
                # Step 1: Process through mathematical integration
                mathematical_signal = await self._process_mathematical_analysis(market_data, all_data_points[:i+1])
                
                # Step 2: Process through trading pipeline
                decision = await self.pipeline.process_market_data(market_data)
                
                # Step 3: Combine mathematical and AI decisions
                final_decision = self._combine_decisions(mathematical_signal, decision, market_data)
                
                if final_decision and final_decision.confidence >= self.config.min_confidence:
                    # Execute trade
                    success = await self._execute_backtest_trade(final_decision, market_data)
                    
                    if success:
                        logger.debug(f"💰 Trade executed: {final_decision.action} {final_decision.symbol} @ ${final_decision.entry_price:.4f}")
                        logger.debug(f"🧮 Mathematical confidence: {mathematical_signal.confidence:.3f}")
                
                # Update equity curve
                self._update_equity_curve(market_data.timestamp)
                
                # Progress update
                if i % 1000 == 0:
                    logger.info(f"📈 Processed {i}/{len(all_data_points)} data points")
                
            except Exception as e:
                logger.error(f"❌ Error processing data point {i}: {e}")
    
    async def _process_mathematical_analysis(self, market_data: MarketDataPoint, historical_data: List[MarketDataPoint]) -> MathematicalSignal:
        """Process market data through all mathematical systems."""
        try:
            # Prepare market data for mathematical processing
            market_data_dict = {
                'current_price': market_data.price,
                'volume': market_data.volume,
                'price_change': market_data.price_change,
                'volatility': market_data.volatility,
                'sentiment': market_data.sentiment,
                'close_prices': [d.price for d in historical_data[-100:]],  # Last 100 prices
                'entry_price': self._get_entry_price(market_data.symbol),
                'bit_phase': self._get_current_bit_phase(market_data.symbol)
            }
            
            # Process through mathematical integration
            mathematical_signal = await self.mathematical_integration.process_market_data_mathematically(market_data_dict)
            
            # Store mathematical signal for analysis
            self.mathematical_signals.append(mathematical_signal)
            
            # Store specific mathematical components
            if mathematical_signal.dualistic_consensus:
                self.dualistic_consensus_history.append(mathematical_signal.dualistic_consensus)
            
            self.dlt_waveform_history.append(mathematical_signal.dlt_waveform_score)
            self.bit_phase_history.append(mathematical_signal.bit_phase)
            self.ferris_phase_history.append(mathematical_signal.ferris_phase)
            
            return mathematical_signal
            
        except Exception as e:
            logger.error(f"❌ Mathematical analysis failed: {e}")
            return MathematicalSignal()
    
    def _combine_decisions(self, mathematical_signal: MathematicalSignal, ai_decision: TradingDecision, market_data: MarketDataPoint) -> TradingDecision:
        """Combine mathematical and AI decisions."""
        try:
            if not ai_decision:
                # Use mathematical decision only
                return self._create_trading_decision_from_mathematical(mathematical_signal, market_data)
            
            # Weight mathematical vs AI decision
            math_weight = 0.7  # Mathematical systems get higher weight
            ai_weight = 0.3
            
            # Calculate combined confidence
            math_confidence = mathematical_signal.confidence
            ai_confidence = ai_decision.confidence if ai_decision else 0.5
            
            combined_confidence = (math_confidence * math_weight + ai_confidence * ai_weight)
            
            # Determine final decision
            if mathematical_signal.decision == "BUY" and ai_decision.action == "BUY":
                final_action = "BUY"
                final_confidence = combined_confidence
            elif mathematical_signal.decision == "SELL" and ai_decision.action == "SELL":
                final_action = "SELL"
                final_confidence = combined_confidence
            elif mathematical_signal.decision == "BUY" and ai_decision.action == "SELL":
                # Conflict - use mathematical decision with reduced confidence
                final_action = mathematical_signal.decision
                final_confidence = math_confidence * 0.8
            elif mathematical_signal.decision == "SELL" and ai_decision.action == "BUY":
                # Conflict - use mathematical decision with reduced confidence
                final_action = mathematical_signal.decision
                final_confidence = math_confidence * 0.8
            else:
                # Use mathematical decision
                final_action = mathematical_signal.decision
                final_confidence = math_confidence
            
            # Create final trading decision
            final_decision = TradingDecision(
                action=final_action,
                symbol=market_data.symbol,
                entry_price=market_data.price,
                position_size=self._calculate_position_size(final_confidence, market_data.price),
                confidence=final_confidence,
                timestamp=market_data.timestamp,
                metadata={
                    'mathematical_decision': mathematical_signal.decision,
                    'ai_decision': ai_decision.action if ai_decision else 'NONE',
                    'dualistic_consensus': mathematical_signal.dualistic_consensus,
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score
                }
            )
            
            return final_decision
            
        except Exception as e:
            logger.error(f"❌ Decision combination failed: {e}")
            return None
    
    def _create_trading_decision_from_mathematical(self, mathematical_signal: MathematicalSignal, market_data: MarketDataPoint) -> TradingDecision:
        """Create trading decision from mathematical signal only."""
        try:
            position_size = self._calculate_position_size(mathematical_signal.confidence, market_data.price)
            
            return TradingDecision(
                action=mathematical_signal.decision,
                symbol=market_data.symbol,
                entry_price=market_data.price,
                position_size=position_size,
                confidence=mathematical_signal.confidence,
                timestamp=market_data.timestamp,
                metadata={
                    'mathematical_only': True,
                    'dualistic_consensus': mathematical_signal.dualistic_consensus,
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Mathematical decision creation failed: {e}")
            return None
    
    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on confidence and risk management."""
        try:
            # Base position size from risk management
            base_size = self.current_balance * self.config.risk_per_trade / price
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale confidence to 0-1
            
            # Apply maximum position limit
            max_positions = self.config.max_positions
            current_positions = len(self.positions)
            
            if current_positions >= max_positions:
                position_multiplier = 0.5  # Reduce size if at position limit
            else:
                position_multiplier = 1.0
            
            final_size = base_size * confidence_multiplier * position_multiplier
            
            return max(0.0, final_size)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed: {e}")
            return 0.0
    
    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for a symbol."""
        # For simplicity, use current market price
        # In a real implementation, you'd track actual entry prices
        return 50000.0  # Default BTC price
    
    def _get_current_bit_phase(self, symbol: str) -> int:
        """Get current bit phase for a symbol."""
        # For simplicity, return a default bit phase
        # In a real implementation, you'd track actual bit phases
        return 8  # Default 8-bit phase

    async def _execute_backtest_trade(self, decision: TradingDecision, market_data: MarketDataPoint) -> bool:
        """Execute a trade in the backtest."""
        try:
            # Apply slippage
            slippage = decision.entry_price * self.config.slippage_rate
            execution_price = decision.entry_price + slippage if decision.action == "BUY" else decision.entry_price - slippage
            
            # Calculate commission
            trade_value = execution_price * decision.position_size
            commission = trade_value * self.config.commission_rate
            
            # Execute trade
            if decision.action == "BUY":
                if trade_value + commission <= self.current_balance:
                    # Update balance
                    self.current_balance -= (trade_value + commission)
                    
                    # Update positions
                    if decision.symbol in self.positions:
                        self.positions[decision.symbol] += decision.position_size
                    else:
                        self.positions[decision.symbol] = decision.position_size
                    
                    # Record trade
                    trade_record = {
                        "timestamp": market_data.timestamp,
                        "symbol": decision.symbol,
                        "action": "BUY",
                        "quantity": decision.position_size,
                        "price": execution_price,
                        "commission": commission,
                        "slippage": slippage,
                        "balance": self.current_balance
                    }
                    self.trades.append(trade_record)
                    
                    self.total_commission += commission
                    self.total_slippage += slippage
                    
                    return True
                else:
                    logger.warning(f"⚠️ Insufficient balance for buy order: ${trade_value + commission:.2f} > ${self.current_balance:.2f}")
                    return False
            
            elif decision.action == "SELL":
                if decision.symbol in self.positions and self.positions[decision.symbol] >= decision.position_size:
                    # Update balance
                    self.current_balance += (trade_value - commission)
                    
                    # Update positions
                    self.positions[decision.symbol] -= decision.position_size
                    if self.positions[decision.symbol] <= 0:
                        del self.positions[decision.symbol]
                    
                    # Record trade
                    trade_record = {
                        "timestamp": market_data.timestamp,
                        "symbol": decision.symbol,
                        "action": "SELL",
                        "quantity": decision.position_size,
                        "price": execution_price,
                        "commission": commission,
                        "slippage": slippage,
                        "balance": self.current_balance
                    }
                    self.trades.append(trade_record)
                    
                    self.total_commission += commission
                    self.total_slippage += slippage
                    
                    return True
                else:
                    logger.warning(f"⚠️ No position to sell for {decision.symbol}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False
    
    def _update_equity_curve(self, timestamp: float):
        """Update equity curve."""
        # Calculate current portfolio value
        portfolio_value = self.current_balance
        
        for symbol, quantity in self.positions.items():
            # For simplicity, use the last known price
            # In a real implementation, you'd look up the current price
            portfolio_value += quantity * 45000  # Simplified
        
        self.equity_curve.append({
            "timestamp": timestamp,
            "balance": self.current_balance,
            "portfolio_value": portfolio_value,
            "positions": len(self.positions)
        })
        
        # Update peak balance
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        # Calculate basic metrics
        final_balance = self.current_balance
        for symbol, quantity in self.positions.items():
            final_balance += quantity * 45000  # Simplified
        
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance
        total_trades = len(self.trades)
        
        # Calculate win/loss metrics
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                if buy_trade["action"] == "BUY" and sell_trade["action"] == "SELL":
                    buy_cost = buy_trade["quantity"] * buy_trade["price"] + buy_trade["commission"]
                    sell_revenue = sell_trade["quantity"] * sell_trade["price"] - sell_trade["commission"]
                    pnl = sell_revenue - buy_cost
                    
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate drawdown
        max_drawdown = 0.0
        peak = self.config.initial_balance
        
        for point in self.equity_curve:
            if point["portfolio_value"] > peak:
                peak = point["portfolio_value"]
            
            drawdown = (peak - point["portfolio_value"]) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1]["portfolio_value"]
            curr_value = self.equity_curve[i]["portfolio_value"]
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        sharpe_ratio = 0.0
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                if buy_trade["action"] == "BUY" and sell_trade["action"] == "SELL":
                    buy_cost = buy_trade["quantity"] * buy_trade["price"] + buy_trade["commission"]
                    sell_revenue = sell_trade["quantity"] * sell_trade["price"] - sell_trade["commission"]
                    pnl = sell_revenue - buy_cost
                    
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Add mathematical analysis to results
        mathematical_metrics = self._calculate_mathematical_metrics()
        
        return BacktestResult(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=total_pnl / winning_trades if winning_trades > 0 else 0.0,
            avg_loss=total_pnl / losing_trades if losing_trades > 0 else 0.0,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio,  # Simplified
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            trade_history=self.trades.copy(),
            equity_curve=self.equity_curve.copy(),
            performance_metrics={
                "total_pnl": total_pnl,
                "avg_trade_pnl": total_pnl / total_trades if total_trades > 0 else 0.0,
                "profit_factor": abs(total_pnl / max_drawdown) if max_drawdown > 0 else 0.0,
                "mathematical_metrics": mathematical_metrics
            }
        )
    
    def _create_empty_result(self) -> BacktestResult:
        """Create empty result for failed backtests."""
        return BacktestResult(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            final_balance=self.config.initial_balance,
            total_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            total_commission=0.0,
            total_slippage=0.0
        )

    def _calculate_mathematical_metrics(self) -> Dict[str, Any]:
        """Calculate mathematical performance metrics."""
        try:
            if not self.mathematical_signals:
                return {}
            
            # DLT Waveform Analysis
            dlt_scores = [s.dlt_waveform_score for s in self.mathematical_signals if s.dlt_waveform_score > 0]
            avg_dlt_score = np.mean(dlt_scores) if dlt_scores else 0.0
            
            # Dualistic Consensus Analysis
            dualistic_scores = []
            for consensus in self.dualistic_consensus_history:
                if consensus and 'mathematical_score' in consensus:
                    dualistic_scores.append(consensus['mathematical_score'])
            avg_dualistic_score = np.mean(dualistic_scores) if dualistic_scores else 0.0
            
            # Bit Phase Analysis
            bit_phase_distribution = {}
            for phase in self.bit_phase_history:
                bit_phase_distribution[phase] = bit_phase_distribution.get(phase, 0) + 1
            
            # Ferris Phase Analysis
            ferris_phases = [p for p in self.ferris_phase_history if p != 0]
            avg_ferris_phase = np.mean(ferris_phases) if ferris_phases else 0.0
            
            # Decision Distribution
            decisions = [s.decision for s in self.mathematical_signals]
            decision_distribution = {}
            for decision in decisions:
                decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
            
            return {
                "avg_dlt_waveform_score": avg_dlt_score,
                "avg_dualistic_score": avg_dualistic_score,
                "bit_phase_distribution": bit_phase_distribution,
                "avg_ferris_phase": avg_ferris_phase,
                "decision_distribution": decision_distribution,
                "total_mathematical_signals": len(self.mathematical_signals),
                "mathematical_confidence_avg": np.mean([s.confidence for s in self.mathematical_signals]),
                "tensor_score_avg": np.mean([s.tensor_score for s in self.mathematical_signals if s.tensor_score != 0]),
                "entropy_score_avg": np.mean([s.entropy_score for s in self.mathematical_signals if s.entropy_score > 0])
            }
            
        except Exception as e:
            logger.error(f"❌ Mathematical metrics calculation failed: {e}")
            return {}

# Global instance for easy access
backtest_engine = None

def get_backtest_engine(config: BacktestConfig) -> BacktestEngine:
    """Get the global backtest engine instance."""
    global backtest_engine
    if backtest_engine is None:
        backtest_engine = BacktestEngine(config)
    return backtest_engine
