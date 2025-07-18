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
        
        logger.info(f"🚀 Backtest Engine initialized with ${config.initial_balance:.2f} initial balance")
    
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
                # Process through trading pipeline
                decision = await self.pipeline.process_market_data(market_data)
                
                if decision and decision.confidence >= self.config.min_confidence:
                    # Execute trade
                    success = await self._execute_backtest_trade(decision, market_data)
                    
                    if success:
                        logger.debug(f"💰 Trade executed: {decision.action} {decision.symbol} @ ${decision.entry_price:.4f}")
                
                # Update equity curve
                self._update_equity_curve(market_data.timestamp)
                
                # Progress update
                if i % 1000 == 0:
                    logger.info(f"📈 Processed {i}/{len(all_data_points)} data points")
                
            except Exception as e:
                logger.error(f"❌ Error processing data point {i}: {e}")
    
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
                "profit_factor": abs(total_pnl / max_drawdown) if max_drawdown > 0 else 0.0
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

# Global instance for easy access
backtest_engine = None

def get_backtest_engine(config: BacktestConfig) -> BacktestEngine:
    """Get the global backtest engine instance."""
    global backtest_engine
    if backtest_engine is None:
        backtest_engine = BacktestEngine(config)
    return backtest_engine
