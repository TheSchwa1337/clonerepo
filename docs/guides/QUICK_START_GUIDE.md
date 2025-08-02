# 🚀 Schwabot Quick Start Guide

> **Get up and running with Schwabot in under 5 minutes!**

## 🎯 What is Schwabot?

Schwabot is an advanced AI-powered cryptocurrency trading system that combines:
- **🤖 Neural Networks**: Tensor-based weight memory for adaptive learning
- **⚛️ Quantum Algorithms**: Advanced mathematical frameworks
- **⚡ Real-Time Processing**: Live market data with sub-second response
- **🛡️ Risk Management**: Advanced position sizing and circuit breakers
- **📊 Multiple Interfaces**: Web dashboard and command line

## ⚡ Quick Start (5 Minutes)

### 1. Prerequisites
- **Python 3.8+** (3.12 recommended)
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU** (optional, for acceleration)
- **Trading account** with API access

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/schwabot.git
cd schwabot

# Install dependencies
pip install -r requirements.txt
```

### 3. Choose Your Interface

#### 🌐 Web Interface (Recommended for beginners)
```bash
# Start the web dashboard
python AOI_Base_Files_Schwabot/launch_unified_interface.py

# Open your browser to:
# http://localhost:8080
```

#### 💻 Command Line (Advanced users)
```bash
# Check system status
python AOI_Base_Files_Schwabot/main.py --system-status

# Start live trading
python AOI_Base_Files_Schwabot/main.py --live

# Run backtesting
python AOI_Base_Files_Schwabot/main.py --backtest --backtest-days 30
```

## 🎮 GPU Acceleration (Optional)

### Enable GPU Support
```bash
# Check GPU info
python AOI_Base_Files_Schwabot/main.py --gpu-info

# Enable GPU auto-detection
python AOI_Base_Files_Schwabot/main.py --gpu-auto-detect
```

### CPU Fallback
If you don't have a GPU, the system automatically falls back to CPU processing. It will run slower but still be fully functional.

## 🔧 Configuration

### Basic Setup
```bash
# Copy example config
cp AOI_Base_Files_Schwabot/config/config.example.yaml AOI_Base_Files_Schwabot/config/config.yaml

# Edit with your settings
nano AOI_Base_Files_Schwabot/config/config.yaml
```

### Environment Variables
```bash
# API Configuration
export SCHWABOT_API_KEY="your_api_key"
export SCHWABOT_SECRET_KEY="your_secret_key"
export SCHWABOT_PASSPHRASE="your_passphrase"

# System Configuration
export SCHWABOT_ENVIRONMENT="production"
export SCHWABOT_LOG_LEVEL="INFO"
```

## 🧪 Testing the System

### Run System Verification
```bash
# Verify everything is working
python verify_system_operation.py
```

### Run Comprehensive Tests
```bash
# Test all components
python AOI_Base_Files_Schwabot/main.py --run-tests
```

## 📊 What You Can Do

### For Beginners
- **Web Dashboard**: Beautiful, intuitive interface
- **Demo Mode**: Safe testing environment
- **Real-Time Monitoring**: Live portfolio tracking
- **Visual Analytics**: Charts and performance metrics

### For Advanced Users
- **Command Line**: Full control and automation
- **Backtesting**: Historical performance analysis
- **Live Trading**: Real-time automated trading
- **API Integration**: Direct system access

### For Developers
- **Modular Architecture**: Easy to extend and customize
- **Comprehensive Documentation**: Technical guides and references
- **Testing Framework**: Complete test suite
- **Development Tools**: Utilities and debugging tools

## 🔒 Security & Risk Management

### Built-in Safety Features
- **Position Limits**: Maximum position sizes per asset
- **Circuit Breakers**: Automatic trading suspension on losses
- **Portfolio Limits**: Maximum portfolio exposure
- **Encrypted Storage**: All sensitive data encrypted

### Risk Warning
**Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.**

- Past performance does not guarantee future results
- You can lose more than your initial investment
- Always test thoroughly before live trading
- Start with small amounts

## 📚 Next Steps

### Learn More
- **[Getting Started Guide](docs/guides/getting_started.md)** - Complete setup tutorial
- **[User Guide](docs/guides/user_guide.md)** - Comprehensive usage instructions
- **[Web Interface Guide](docs/guides/web_interface.md)** - Dashboard walkthrough
- **[CLI Reference](docs/api/cli_reference.md)** - Command-line documentation

### Configure Your System
- **[Configuration Guide](docs/configuration/setup.md)** - Setup and customization
- **[System Architecture](docs/development/architecture.md)** - Technical details
- **[Contributing Guide](docs/development/contributing.md)** - Developer guidelines

### Get Help
- **Documentation**: [docs/](docs/) directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Email**: support@schwabot.com

## 🎉 You're Ready!

Your Schwabot system is now ready to use! Choose your interface and start exploring:

- **🌐 Web Dashboard**: `python AOI_Base_Files_Schwabot/launch_unified_interface.py`
- **💻 Command Line**: `python AOI_Base_Files_Schwabot/main.py --help`

### Quick Commands Reference

```bash
# System Status
python AOI_Base_Files_Schwabot/main.py --system-status

# Start Web Interface
python AOI_Base_Files_Schwabot/launch_unified_interface.py

# Run Tests
python AOI_Base_Files_Schwabot/main.py --run-tests

# GPU Info
python AOI_Base_Files_Schwabot/main.py --gpu-info

# Backtesting
python AOI_Base_Files_Schwabot/main.py --backtest --backtest-days 30

# Live Trading
python AOI_Base_Files_Schwabot/main.py --live
```

---

**🚀 Welcome to Schwabot - Advanced AI-Powered Trading!**

*Last updated: July 18, 2025* 