# Schwabot Backtesting System

## Overview

The Schwabot Backtesting System is a comprehensive testing framework that allows you to validate trading strategies using historical market data before deploying them in live trading. The system integrates with the full Schwabot trading pipeline, including AI analysis, risk management, and trade execution.

## Features

### ✅ Real Historical Data Integration
- **Binance API**: Direct integration with Binance for real cryptocurrency data
- **Yahoo Finance**: Alternative data source for traditional assets
- **CSV Import**: Support for custom historical data files
- **Simulated Data**: Realistic market data generation for testing

### ✅ Full Trading Pipeline Testing
- **AI Analysis**: Tests the complete AI analysis pipeline
- **Pattern Recognition**: Validates pattern detection algorithms
- **Sentiment Analysis**: Tests market sentiment processing
- **Technical Analysis**: Validates technical indicators
- **Risk Management**: Tests position sizing and risk controls

### ✅ Comprehensive Performance Metrics
- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Trade Statistics**: Detailed trade analysis
- **Equity Curve**: Portfolio value over time

### ✅ Realistic Trading Simulation
- **Commission Modeling**: Realistic trading costs
- **Slippage Simulation**: Market impact modeling
- **Position Management**: Multi-asset portfolio tracking
- **Risk Controls**: Position limits and stop-losses

## Quick Start

### 1. Basic Backtest

```python
from backtesting.backtest_engine import BacktestConfig, BacktestEngine

# Configure backtest
config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-01-31",
    symbols=["BTCUSDT"],
    initial_balance=10000.0,
    data_source="simulated"  # or "binance", "yahoo"
)

# Run backtest
engine = BacktestEngine(config)
result = await engine.run_backtest()

# View results
print(f"Total Return: {result.total_return:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
```

### 2. Using the Backtest Runner

```bash
# Run sample backtest
python backtesting/run_backtest.py

# Choose option 1 for sample backtest or option 2 for custom
```

### 3. Testing the System

```bash
# Test with simulated data
python test_backtest_simple.py

# Run comprehensive tests
python tests/test_backtesting.py
```

## Configuration Options

### BacktestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | str | Required | Start date (YYYY-MM-DD) |
| `end_date` | str | Required | End date (YYYY-MM-DD) |
| `symbols` | List[str] | Required | Trading symbols |
| `initial_balance` | float | 10000.0 | Starting capital |
| `commission_rate` | float | 0.001 | Trading commission (0.1%) |
| `slippage_rate` | float | 0.0005 | Market slippage (0.05%) |
| `data_source` | str | "auto" | Data source type |
| `timeframe` | str | "1h" | Data timeframe |
| `enable_ai_analysis` | bool | True | Enable AI analysis |
| `enable_risk_management` | bool | True | Enable risk controls |
| `max_positions` | int | 5 | Maximum concurrent positions |
| `risk_per_trade` | float | 0.02 | Risk per trade (2%) |
| `min_confidence` | float | 0.7 | Minimum trade confidence |

### Data Sources

| Source | Description | Requirements |
|--------|-------------|--------------|
| `binance` | Real cryptocurrency data | Internet connection |
| `yahoo` | Traditional asset data | yfinance library |
| `simulated` | Generated test data | None |
| `csv` | Custom data files | CSV files in data/ directory |
| `auto` | Automatic fallback | Tries real sources first |

## Performance Metrics

### Key Metrics

- **Total Return**: `(final_balance - initial_balance) / initial_balance`
- **Win Rate**: `winning_trades / total_trades`
- **Sharpe Ratio**: `average_return / standard_deviation * sqrt(252)`
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: `total_profit / total_loss`

### Trade Analysis

- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Consecutive Wins/Losses**: Longest winning/losing streaks
- **Total Commission**: Sum of all trading costs
- **Total Slippage**: Sum of all market impact costs

## Data Sources

### Binance API Integration

```python
# Real cryptocurrency data
config = BacktestConfig(
    data_source="binance",
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframe="1h"
)
```

**Features:**
- Real historical OHLCV data
- Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Automatic rate limiting
- Error handling and retries

### Simulated Data Generation

```python
# Realistic test data
config = BacktestConfig(
    data_source="simulated",
    symbols=["BTCUSDT"]
)
```

**Features:**
- Realistic price movements using random walk
- Configurable volatility and base prices
- No external dependencies
- Consistent for reproducible tests

### CSV Data Import

```python
# Custom data files
config = BacktestConfig(
    data_source="csv",
    symbols=["CUSTOM"]
)
```

**Required CSV Format:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,45000,45100,44900,45050,1000
```

## Advanced Usage

### Custom Data Sources

```python
from backtesting.data_sources import DataSourceConfig, DataSourceManager

# Create custom data source
configs = [
    DataSourceConfig(source_type="binance"),
    DataSourceConfig(source_type="yahoo")
]

manager = DataSourceManager(configs)
df = await manager.get_market_data("BTCUSDT", "2024-01-01", "2024-01-31")
```

### Multiple Symbols

```python
config = BacktestConfig(
    symbols=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    max_positions=3,
    risk_per_trade=0.01
)
```

### Custom Timeframes

```python
config = BacktestConfig(
    timeframe="15m",  # 15-minute intervals
    start_date="2024-01-01",
    end_date="2024-01-07"
)
```

## Results Analysis

### BacktestResult Object

```python
result = await engine.run_backtest()

# Basic metrics
print(f"Final Balance: ${result.final_balance:,.2f}")
print(f"Total Return: {result.total_return:.2%}")
print(f"Total Trades: {result.total_trades}")

# Detailed analysis
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")

# Trade history
for trade in result.trade_history[:5]:
    print(f"{trade['action']} {trade['symbol']} @ ${trade['price']:.2f}")

# Equity curve
for point in result.equity_curve[-5:]:
    print(f"Portfolio Value: ${point['portfolio_value']:,.2f}")
```

### Saving Results

```python
import json

# Save results to file
with open("backtest_results.json", "w") as f:
    json.dump({
        "config": {
            "start_date": result.config.start_date,
            "end_date": result.config.end_date,
            "symbols": result.config.symbols
        },
        "results": {
            "final_balance": result.final_balance,
            "total_return": result.total_return,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio
        }
    }, f, indent=2)
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check internet connection
   - Verify API endpoints are accessible
   - Use simulated data for testing

2. **No Trades Executed**
   - Check minimum confidence threshold
   - Verify AI analysis is enabled
   - Review market data quality

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes project directory
   - Verify module structure

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run backtest with detailed logging
config = BacktestConfig(...)
engine = BacktestEngine(config)
result = await engine.run_backtest()
```

## Integration with Live Trading

The backtesting system uses the same trading pipeline as the live system, ensuring:

- **Consistency**: Same analysis algorithms
- **Validation**: Strategy verification before live deployment
- **Risk Assessment**: Performance evaluation under various conditions
- **Optimization**: Parameter tuning and strategy refinement

## Next Steps

1. **Run Sample Backtest**: Test the system with simulated data
2. **Connect Real Data**: Configure Binance API for real historical data
3. **Optimize Strategy**: Tune parameters based on backtest results
4. **Validate Performance**: Ensure consistent results across time periods
5. **Deploy Live**: Move validated strategies to live trading

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test files for examples
3. Examine the logging output for errors
4. Verify data source connectivity

The backtesting system is designed to be reliable and provide comprehensive validation of your trading strategies before live deployment. 