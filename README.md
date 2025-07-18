# 🌀 Schwabot - AI-Powered Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/your-repo/schwabot)

> **Advanced AI-powered cryptocurrency trading system with real-time market analysis, risk management, and automated execution capabilities.**

## ⚡ Quick Start (5 Minutes)

**🚀 Get up and running immediately with our [Quick Start Guide](QUICK_START_GUIDE.md)!**

```bash
# Clone and install
git clone https://github.com/your-repo/schwabot.git
cd schwabot
pip install -r requirements.txt

# Start web interface (recommended for beginners)
python AOI_Base_Files_Schwabot/launch_unified_interface.py

# Or use command line (advanced users)
python AOI_Base_Files_Schwabot/main.py --system-status
```

**📖 [Complete Quick Start Guide](QUICK_START_GUIDE.md) | 📚 [Full Documentation](docs/) | 🎮 [GPU Support](#-gpu-acceleration)**

---

## 🎯 What is Schwabot?

Schwabot is a sophisticated trading system that combines:
- **🤖 AI-Powered Analysis**: Neural networks and quantum algorithms for market prediction
- **⚡ Real-Time Processing**: Live market data integration with sub-second response times
- **🛡️ Risk Management**: Advanced position sizing and circuit breakers
- **📊 Multiple Interfaces**: Web dashboard and command-line interface
- **🔒 Security**: Encrypted API connections and secure credential management

### Key Features

- **🧠 Neural Memory System**: Tensor-based weight memory for adaptive learning
- **⚛️ Quantum Algorithms**: Advanced mathematical frameworks for market analysis
- **📈 Real-Time Backtesting**: Live performance validation
- **🎛️ Multiple Exchange Support**: Coinbase, Binance, and more
- **📱 Web Dashboard**: Beautiful, responsive interface for monitoring
- **⚙️ CLI Interface**: Full command-line control for advanced users

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with CUDA support (optional, for acceleration)
- Trading account with API access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/schwabot.git
   cd schwabot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API credentials**
   ```bash
   # Copy the example config
   cp AOI_Base_Files_Schwabot/config/config.example.yaml AOI_Base_Files_Schwabot/config/config.yaml
   
   # Edit with your API keys
   nano AOI_Base_Files_Schwabot/config/config.yaml
   ```

### Choose Your Interface

#### 🌐 Web Interface (Recommended for beginners)

```bash
python AOI_Base_Files_Schwabot/launch_unified_interface.py
```

Then open your browser to `http://localhost:8080`

#### 💻 Command Line Interface (Advanced users)

```bash
# Check system status
python AOI_Base_Files_Schwabot/main.py --system-status

# Start live trading
python AOI_Base_Files_Schwabot/main.py --live-trading

# Run backtesting
python AOI_Base_Files_Schwabot/main.py --backtest
```

## 🎮 GPU Acceleration

### Enable GPU Support
```bash
# Check GPU info
python AOI_Base_Files_Schwabot/main.py --gpu-info

# Enable GPU auto-detection
python AOI_Base_Files_Schwabot/main.py --gpu-auto-detect
```

### CPU Fallback
**No GPU? No problem!** The system automatically falls back to CPU processing. It will run slower but still be fully functional.

## 🧪 System Verification

### Verify Everything Works
```bash
# Run comprehensive system verification
python verify_system_operation.py

# Run all tests
python AOI_Base_Files_Schwabot/main.py --run-tests
```

## 📚 Documentation

### For New Users
- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get running in 5 minutes ⚡
- **[Getting Started Guide](docs/guides/getting_started.md)** - Complete setup tutorial
- **[User Guide](docs/guides/user_guide.md)** - Comprehensive usage instructions
- **[Web Interface Guide](docs/guides/web_interface.md)** - Dashboard walkthrough

### For Advanced Users
- **[CLI Reference](docs/api/cli_reference.md)** - Command-line documentation
- **[Configuration Guide](docs/configuration/setup.md)** - Setup and customization
- **[System Architecture](docs/development/architecture.md)** - Technical details

### System Status
- **[Complete System Guide](docs/README_COMPLETE_UNIFIED_SYSTEM.md)** - Full system overview
- **[Integration Status](docs/FINAL_SYSTEM_STATUS_REPORT.md)** - Current system health
- **[Zero Errors Report](docs/ZERO_ERRORS_ACHIEVED.md)** - Quality assurance

## 🏗️ System Architecture

```
Schwabot Trading System
├── 🧠 Neural Processing Engine
│   ├── Tensor Weight Memory
│   ├── Quantum Mathematical Bridge
│   └── Orbital Shell Brain System
├── 📊 Market Data Integration
│   ├── Real-time Data Feeds
│   ├── Historical Data Manager
│   └── Live Market Bridge
├── 🤖 AI Trading Engine
│   ├── Strategy Executor
│   ├── Risk Manager
│   └── Portfolio Tracker
├── 🛡️ Security & Risk Management
│   ├── Encrypted API Manager
│   ├── Position Limits
│   └── Circuit Breakers
└── 🖥️ User Interfaces
    ├── Web Dashboard
    └── Command Line Interface
```

## 🔧 Configuration

### Basic Configuration

The system uses YAML configuration files located in `AOI_Base_Files_Schwabot/config/`:

- **`config.yaml`** - Main configuration
- **`api_keys.yaml`** - Exchange API credentials
- **`risk_limits.yaml`** - Risk management settings
- **`strategies.yaml`** - Trading strategy parameters

### Environment Variables

```bash
# API Configuration
SCHWABOT_API_KEY=your_api_key
SCHWABOT_SECRET_KEY=your_secret_key
SCHWABOT_PASSPHRASE=your_passphrase

# System Configuration
SCHWABOT_ENVIRONMENT=production
SCHWABOT_LOG_LEVEL=INFO
SCHWABOT_DATA_DIR=/path/to/data
```

## 🧪 Testing & Validation

### Run All Tests

```bash
python run_tests.py
```

### Individual Test Categories

```bash
# Integration tests
python -m pytest tests/integration/

# Unit tests
python -m pytest tests/unit/

# Performance tests
python -m pytest tests/performance/

# Security tests
python -m pytest tests/security/
```

### System Validation

```bash
# Complete system validation
python test_complete_production_system.py

# Mathematical integration test
python test_unified_mathematical_trading_integration.py

# API integration test
python test_secure_api_integration.py
```

## 📊 Monitoring & Analytics

### Real-Time Monitoring

The system provides comprehensive monitoring through:

- **Web Dashboard**: Live portfolio tracking and performance metrics
- **Log Files**: Detailed system logs in `logs/` directory
- **Performance Reports**: Generated reports in `reports/` directory
- **Database**: SQLite databases for state and monitoring data

### Key Metrics

- **Portfolio Performance**: P&L, Sharpe ratio, drawdown
- **System Health**: CPU, memory, GPU utilization
- **Trading Metrics**: Win rate, average trade duration
- **Risk Metrics**: VaR, position concentration, exposure

## 🔒 Security & Risk Management

### Security Features

- **🔐 Encrypted Storage**: All sensitive data encrypted at rest
- **🔑 Secure API Management**: Credentials stored securely
- **🛡️ Access Control**: User authentication and authorization
- **📝 Audit Logging**: Complete audit trail of all actions

### Risk Management

- **💰 Position Limits**: Maximum position sizes per asset
- **⚡ Circuit Breakers**: Automatic trading suspension on losses
- **📊 Portfolio Limits**: Maximum portfolio exposure
- **🔄 Dynamic Sizing**: Risk-adjusted position sizing

## 🚨 Important Disclaimers

### Risk Warning

**Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The high degree of leverage can work against you as well as for you.**

- Past performance does not guarantee future results
- You can lose more than your initial investment
- The system is for educational and research purposes
- Always test thoroughly before live trading

### Legal Notice

- This software is provided "as is" without warranty
- Users are responsible for compliance with local regulations
- API usage must comply with exchange terms of service
- Tax implications vary by jurisdiction

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/schwabot.git
cd schwabot

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python run_tests.py

# Check code quality
flake8 AOI_Base_Files_Schwabot/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

1. **📖 Documentation**: Check the [documentation](docs/) first
2. **🐛 Issues**: Report bugs on [GitHub Issues](https://github.com/your-repo/schwabot/issues)
3. **💬 Discussions**: Join our [Discussions](https://github.com/your-repo/schwabot/discussions)
4. **📧 Email**: Contact us at support@schwabot.com

### Common Issues

- **Import Errors**: Ensure all dependencies are installed
- **API Connection**: Verify API credentials and permissions
- **Performance**: Check system resources and GPU availability
- **Configuration**: Validate YAML syntax and required fields

## 🎉 Acknowledgments

- **CUDA Acceleration**: NVIDIA for GPU computing capabilities
- **Mathematical Framework**: Advanced tensor algebra and quantum algorithms
- **Exchange APIs**: Coinbase, Binance, and other exchanges
- **Open Source Community**: All contributors and maintainers

---

**Made with ❤️ by the Schwabot Team**

*Last updated: July 18, 2025* 