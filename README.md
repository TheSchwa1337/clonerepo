# Schwabot Trading System

## 🚀 Overview

Schwabot is a comprehensive AI-powered trading system that combines advanced mathematical algorithms, machine learning, and real-time market analysis to execute profitable trading strategies.

## ✨ Key Features

### 🤖 AI-Powered Analysis
- **Market Analysis**: Real-time market sentiment and trend analysis
- **Pattern Recognition**: Advanced pattern detection algorithms
- **Technical Analysis**: Comprehensive technical indicator analysis
- **Risk Management**: Intelligent position sizing and risk controls

### 📊 Comprehensive Backtesting System
- **Real Historical Data**: Integration with Binance API for actual market data
- **Simulated Data**: Realistic market data generation for testing
- **Full Pipeline Testing**: Tests the complete AI analysis and trading pipeline
- **Performance Metrics**: Sharpe ratio, drawdown, win rate, and more
- **Trade Simulation**: Realistic commission and slippage modeling

### 🔄 Live Trading Capabilities
- **Real-time Processing**: Live market data analysis
- **Multi-Exchange Support**: Binance, Coinbase, and more
- **Portfolio Management**: Multi-asset position tracking
- **Performance Monitoring**: Real-time performance tracking

## 🎯 Quick Start

### 1. Test the Backtesting System

```bash
# Run a quick demo
python demo_backtesting.py

# Run comprehensive backtest
python backtesting/run_backtest.py

# Test the system
python tests/test_backtesting.py
```

### 2. Basic Usage

```python
from backtesting.backtest_engine import BacktestConfig, BacktestEngine

# Configure backtest
config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-01-31",
    symbols=["BTCUSDT"],
    data_source="simulated"  # or "binance" for real data
)

# Run backtest
engine = BacktestEngine(config)
result = await engine.run_backtest()

print(f"Total Return: {result.total_return:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
```

### 3. Live Trading

```python
from schwabot_trading_bot import SchwabotTradingBot

# Initialize trading bot
bot = SchwabotTradingBot(config)
await bot.start()
```

## 📁 Project Structure

```
schwabot/
├── backtesting/           # Comprehensive backtesting system
│   ├── backtest_engine.py # Main backtesting engine
│   ├── data_sources.py    # Data source management
│   └── run_backtest.py    # Easy-to-use runner
├── core/                  # Core trading components
│   ├── trading_pipeline_manager.py
│   ├── schwabot_ai_integration.py
│   └── market_data_simulator.py
├── tests/                 # Test suite
├── docs/                  # Documentation
└── schwabot_trading_bot.py # Main trading bot
```

## 🔧 Configuration

### Backtesting Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Backtest start date | Required |
| `end_date` | Backtest end date | Required |
| `symbols` | Trading symbols | Required |
| `data_source` | Data source type | "auto" |
| `initial_balance` | Starting capital | $10,000 |
| `commission_rate` | Trading commission | 0.1% |
| `enable_ai_analysis` | Enable AI analysis | True |

### Data Sources

- **`binance`**: Real cryptocurrency data from Binance API
- **`simulated`**: Realistic generated test data
- **`csv`**: Custom historical data files
- **`auto`**: Automatic fallback to best available source

## 📈 Performance Metrics

The backtesting system provides comprehensive performance analysis:

- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Trade Statistics**: Detailed trade analysis
- **Equity Curve**: Portfolio value over time

## 🛡️ Risk Management

- **Position Sizing**: Intelligent position sizing based on risk
- **Stop Losses**: Automatic stop-loss management
- **Portfolio Limits**: Maximum position and exposure limits
- **Real-time Monitoring**: Continuous risk assessment

## 🔗 Integration

### Supported Exchanges
- **Binance**: Full API integration
- **Coinbase**: API integration ready
- **Other Exchanges**: Extensible architecture

### Data Sources
- **Real-time APIs**: Live market data
- **Historical Data**: Backtesting and analysis
- **Simulated Data**: Testing and development

## 📚 Documentation

- [Backtesting System Guide](docs/BACKTESTING_SYSTEM.md)
- [API Documentation](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test backtesting system
python tests/test_backtesting.py

# Test trading pipeline
python tests/test_trading_pipeline.py
```

## 🚀 Deployment

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo_backtesting.py
```

### Production
```bash
# Configure API keys
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET="your_secret"

# Start trading bot
python schwabot_trading_bot.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always test thoroughly before using with real money.

## 🆘 Support

For questions or issues:
1. Check the documentation
2. Review the troubleshooting guide
3. Open an issue on GitHub

---

**Schwabot Trading System** - AI-Powered Trading with Comprehensive Backtesting 