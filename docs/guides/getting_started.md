# 🚀 Getting Started with Schwabot

## Welcome to Schwabot!

Schwabot is an advanced AI-powered trading system that combines cutting-edge mathematical frameworks with GPU acceleration to provide automated trading capabilities. This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** installed on your system
- **CUDA-compatible GPU** (optional, but recommended for best performance)
- **Trading API credentials** from your preferred exchange (Coinbase, etc.)
- **Basic understanding** of cryptocurrency trading concepts

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd clonerepo
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python AOI_Base_Files_Schwabot/main.py --system-status
```

You should see a status report indicating the system is ready.

## ⚙️ Configuration

### Setting Up API Credentials

1. **Navigate to the config directory:**
   ```bash
   cd AOI_Base_Files_Schwabot/config
   ```

2. **Edit the coinbase profiles file:**
   ```bash
   # On Windows
   notepad coinbase_profiles.yaml
   
   # On macOS/Linux
   nano coinbase_profiles.yaml
   ```

3. **Add your API credentials:**
   ```yaml
   profiles:
     default:
       api_key: "your_api_key_here"
       api_secret: "your_api_secret_here"
       sandbox: true  # Set to false for live trading
       trading_pairs: ["BTC/USDC", "ETH/USDC"]
       position_limits:
         max_open_positions: 5
         max_position_size: 1000
   ```

### Important Security Notes
- **Never commit your API keys** to version control
- **Start with sandbox mode** to test the system safely
- **Use strong, unique API keys** with limited permissions
- **Enable 2FA** on your exchange account

## 🎮 Choosing Your Interface

Schwabot offers two main interfaces:

### Option A: Web Dashboard (Recommended for Beginners)

The web dashboard provides an intuitive, visual interface for all trading operations.

**Launch the web interface:**
```bash
python AOI_Base_Files_Schwabot/launch_unified_interface.py
```

**Access the dashboard:**
Open your web browser and navigate to: `http://localhost:8080`

**Features:**
- Real-time portfolio tracking
- Visual strategy execution
- Multi-profile management
- Live market data visualization
- AI command interface
- GPU/CPU toggle controls

### Option B: Command Line Interface (Advanced Users)

The CLI provides advanced control and automation capabilities.

**Basic commands:**
```bash
# Check system status
python AOI_Base_Files_Schwabot/main.py --system-status

# Run system tests
python AOI_Base_Files_Schwabot/main.py --run-tests

# Start backtesting
python AOI_Base_Files_Schwabot/main.py --backtest --days 30

# Get GPU information
python AOI_Base_Files_Schwabot/main.py --gpu-info
```

## 🎯 Your First Trading Session

### Starting with Demo Mode

**We strongly recommend starting with demo mode** to learn the system without risking real funds.

1. **Launch the web interface** (if using the dashboard)
2. **Switch to demo mode** in the session controls
3. **Configure a test strategy** with low confidence settings
4. **Execute the strategy** and observe the results
5. **Review the logs** to understand the decision-making process

### Understanding the Dashboard

#### Portfolio Panel
- **Portfolio Value**: Your current total portfolio worth
- **Total Profit**: Cumulative profit/loss from all trades
- **Win Rate**: Percentage of successful trades
- **Active Strategies**: Number of currently executing strategies

#### Strategy Execution Panel
- **Strategy ID**: Unique identifier for your strategy
- **Profile Selection**: Choose which API profile to use
- **Asset Selection**: Select trading pair (BTC/USDC, ETH/USDC, etc.)
- **Confidence**: Set strategy confidence (0.1 - 1.0)
- **Signal Strength**: Set signal strength (0.1 - 1.0)

#### Profile Management
- **Profile Cards**: Display all available trading profiles
- **Status Indicators**: Show active/inactive status
- **Trading Pairs**: List supported trading pairs
- **Position Limits**: Maximum open positions

## 🔒 Security and Risk Management

### Built-in Safety Features
- **Circuit Breakers**: Automatically stop trading if losses exceed thresholds
- **Position Limits**: Prevent overexposure to any single asset
- **Real-time Monitoring**: Continuous oversight of all trading activities
- **Encrypted Communications**: All data is secured with algorithmic encryption

### Best Practices
1. **Start Small**: Begin with small position sizes
2. **Monitor Regularly**: Check your portfolio frequently
3. **Use Stop Losses**: Always set appropriate stop-loss levels
4. **Diversify**: Don't put all your funds in one strategy
5. **Keep Learning**: Study the system's decision-making process

## 🚀 Advanced Features

### GPU Acceleration
If you have a CUDA-compatible GPU, Schwabot will automatically detect and use it for:
- Faster AI model processing
- Accelerated mathematical calculations
- Improved real-time decision making

**Check GPU status:**
```bash
python AOI_Base_Files_Schwabot/main.py --gpu-info
```

### AI Integration
Schwabot supports integration with various AI systems:
- **Claude**: Anthropic's AI assistant
- **GPT-4o**: OpenAI's advanced language model
- **R1**: Real-time AI decision making

### Mathematical Framework
The system uses advanced mathematical concepts:
- **Tensor Algebra**: Multi-dimensional market analysis
- **Quantum Smoothing**: Advanced signal processing
- **Entropy Analysis**: Risk assessment and decision making

## 📊 Monitoring and Analysis

### Real-time Metrics
- **Portfolio Performance**: Track your investment returns
- **Strategy Effectiveness**: Monitor which strategies work best
- **Risk Metrics**: Understand your exposure and risk levels
- **System Health**: Ensure all components are functioning properly

### Logs and Reports
- **Trading Logs**: Detailed records of all trading decisions
- **Error Reports**: System issues and resolutions
- **Performance Analytics**: Comprehensive performance analysis
- **Risk Assessments**: Detailed risk analysis reports

## 🆘 Troubleshooting

### Common Issues

**System won't start:**
```bash
# Check Python version
python --version

# Verify dependencies
pip list | grep flask

# Check configuration files
ls AOI_Base_Files_Schwabot/config/
```

**API connection issues:**
- Verify your API keys are correct
- Check your internet connection
- Ensure your exchange account is active
- Verify API permissions include trading

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help
- **Check the logs**: Look for error messages in the system logs
- **Review documentation**: See the other guides in this documentation
- **Test components**: Use the built-in testing features
- **Start fresh**: Reset to default configurations if needed

## 🎉 Next Steps

Congratulations! You've successfully set up Schwabot. Here's what to do next:

1. **Explore the Dashboard**: Familiarize yourself with all the features
2. **Run Some Tests**: Use the testing features to validate your setup
3. **Try Demo Trading**: Practice with demo mode to learn the system
4. **Read Advanced Guides**: Explore the other documentation
5. **Join the Community**: Connect with other Schwabot users

### Recommended Reading
- [Web Interface Guide](web_interface.md)
- [CLI Reference](../api/cli_reference.md)
- [Configuration Guide](../configuration/setup.md)
- [Security Best Practices](../configuration/security.md)

---

**Ready to start trading?** Remember to always start with demo mode and never invest more than you can afford to lose! 