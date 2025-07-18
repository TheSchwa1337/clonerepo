# 🎮 Web Interface Guide

## Overview

The Schwabot Web Interface provides an intuitive, visual dashboard for all your trading operations. This guide will walk you through every feature and help you master the interface.

## 🚀 Launching the Interface

### Starting the Web Server
```bash
python AOI_Base_Files_Schwabot/launch_unified_interface.py
```

### Accessing the Dashboard
Open your web browser and navigate to: `http://localhost:8080`

### Alternative Launch Options
```bash
# Custom host and port
python AOI_Base_Files_Schwabot/launch_unified_interface.py --host 127.0.0.1 --port 9000

# Debug mode (for developers)
python AOI_Base_Files_Schwabot/launch_unified_interface.py --debug

# Skip startup banner
python AOI_Base_Files_Schwabot/launch_unified_interface.py --no-banner
```

## 📊 Dashboard Overview

The web interface is organized into several key panels, each providing specific functionality:

### 1. Metrics Dashboard (Top Panel)

**Portfolio Metrics:**
- **Portfolio Value**: Real-time total portfolio worth
- **Total Profit**: Cumulative profit/loss from all trades
- **Win Rate**: Percentage of successful trades
- **Active Strategies**: Number of currently executing strategies

**Market Information:**
- **Current Price**: Live price of selected asset
- **Price Change**: 24-hour price change percentage
- **Volume**: Trading volume for the selected asset

**System Status:**
- **API Status**: Connection status to trading platforms
- **GPU Status**: GPU acceleration status
- **System Health**: Overall system operational status

### 2. Strategy Execution Panel

This panel allows you to execute trading strategies with full control over parameters.

**Strategy Configuration:**
- **Strategy ID**: Enter a unique identifier for your strategy
- **Profile Selection**: Choose which API profile to execute on
- **Asset Selection**: Select trading pair (BTC/USDC, ETH/USDC, etc.)
- **Confidence Slider**: Set strategy confidence level (0.1 - 1.0)
- **Signal Strength Slider**: Set signal strength (0.1 - 1.0)

**Execution Controls:**
- **Execute Strategy**: Start strategy execution
- **Stop All Strategies**: Halt all active strategies
- **Strategy Status**: Real-time status of executing strategies

**Strategy Parameters Explained:**
- **Confidence**: How certain the AI is about the trading decision
- **Signal Strength**: The strength of the market signal detected
- **Profile**: Which trading account to use for execution

### 3. Profile Management Panel

Manage multiple trading profiles and their configurations.

**Profile Cards Display:**
- **Profile Name**: Identifier for each profile
- **Status Indicator**: Active/inactive status
- **Trading Pairs**: Supported trading pairs
- **Position Limits**: Maximum open positions
- **Balance**: Current account balance

**Profile Controls:**
- **Activate Profile**: Enable specific profiles for trading
- **Deactivate Profile**: Disable profiles
- **Edit Profile**: Modify profile settings
- **View Details**: See detailed profile information

### 4. Session Control Panel

Control the overall trading session and system settings.

**Trading Mode Selection:**
- **Demo Mode**: Test strategies without real money
- **Live Trading Mode**: Execute real trades
- **Backtest Mode**: Test against historical data

**System Controls:**
- **GPU Toggle**: Enable/disable GPU acceleration
- **Hardware Info**: Display detected hardware capabilities
- **Refresh Status**: Update system status
- **Reset System**: Reset all system components

**Session Information:**
- **Session ID**: Unique session identifier
- **Start Time**: When the session began
- **Uptime**: How long the system has been running

### 5. Visualization Panel

Generate and display trading charts and analysis.

**Chart Types:**
- **Price Chart**: Standard price candlestick chart
- **Volume Analysis**: Trading volume visualization
- **Technical Indicators**: RSI, MACD, moving averages
- **Pattern Recognition**: AI-detected chart patterns
- **Risk Analysis**: Portfolio risk visualization

**Chart Controls:**
- **Generate Chart**: Create new visualization
- **Clear Chart**: Reset visualization area
- **Export Chart**: Save chart as image
- **Timeframe Selection**: Choose chart timeframe

**Symbol Selection:**
- **Asset Picker**: Choose which asset to visualize
- **Multiple Assets**: Compare multiple assets
- **Custom Timeframes**: Set custom time periods

### 6. AI Command Interface

Process commands from various AI systems.

**AI System Selection:**
- **Claude**: Anthropic's AI assistant
- **GPT-4o**: OpenAI's advanced language model
- **R1**: Real-time AI decision making

**Command Processing:**
- **Command Hash Input**: Enter AI-generated command hashes
- **Process Command**: Execute AI command
- **Command Status**: Track command execution status
- **Command History**: View previous commands

**AI Integration Features:**
- **Real-time Processing**: Instant command execution
- **Secure Communication**: Encrypted AI interactions
- **Command Validation**: Verify command integrity
- **Result Feedback**: Get execution results

### 7. System Log Panel

Real-time system events and status updates.

**Log Features:**
- **Real-time Updates**: Live system events
- **Color-coded Entries**: Different colors for info, warning, error
- **Auto-scroll**: Automatically scrolls to latest entries
- **Log Filtering**: Filter by log level or type
- **Export Logs**: Save log data to file

**Log Levels:**
- **Info (Blue)**: General information and status updates
- **Warning (Yellow)**: Potential issues or alerts
- **Error (Red)**: System errors and failures
- **Success (Green)**: Successful operations

## 🎯 Step-by-Step Workflows

### Starting Your First Trading Session

1. **Launch the Interface**
   ```bash
   python AOI_Base_Files_Schwabot/launch_unified_interface.py
   ```

2. **Access the Dashboard**
   - Open browser to `http://localhost:8080`
   - Verify all panels are loading correctly

3. **Check System Status**
   - Review the Metrics Dashboard
   - Ensure API connections are active
   - Verify GPU status (if applicable)

4. **Configure Trading Mode**
   - Start with **Demo Mode** for safety
   - Verify profile configurations
   - Check position limits

5. **Execute Your First Strategy**
   - Enter a unique Strategy ID
   - Select your preferred profile
   - Choose a trading pair (e.g., BTC/USDC)
   - Set confidence to 0.5 (moderate)
   - Set signal strength to 0.5 (moderate)
   - Click "Execute Strategy"

6. **Monitor the Results**
   - Watch the Strategy Execution Panel for status
   - Check the System Log for detailed information
   - Monitor portfolio metrics for changes

### Managing Multiple Profiles

1. **View Available Profiles**
   - Check the Profile Management Panel
   - Review status indicators
   - Note trading pairs and limits

2. **Activate Profiles**
   - Click "Activate" on desired profiles
   - Verify activation in status indicators
   - Check API connection status

3. **Configure Profile Settings**
   - Set appropriate position limits
   - Configure trading pairs
   - Adjust risk parameters

4. **Monitor Profile Performance**
   - Track individual profile metrics
   - Compare performance across profiles
   - Adjust settings as needed

### Using AI Commands

1. **Receive AI Command**
   - Get command hash from your AI system
   - Note the AI source (Claude, GPT-4o, R1)

2. **Process the Command**
   - Select the correct AI source
   - Enter the command hash
   - Click "Process AI Command"

3. **Monitor Execution**
   - Watch command status updates
   - Check system log for details
   - Review execution results

4. **Verify Results**
   - Confirm command was executed correctly
   - Check for any errors or warnings
   - Review impact on portfolio

## 🔧 Advanced Features

### Real-time Data Streaming

The interface uses WebSocket connections for real-time updates:
- **Live Price Feeds**: Instant price updates
- **Portfolio Changes**: Real-time portfolio value updates
- **Strategy Status**: Live strategy execution status
- **System Metrics**: Continuous system health monitoring

### GPU Acceleration Toggle

**Enabling GPU Mode:**
- Click the GPU toggle in Session Control Panel
- Verify GPU detection in hardware info
- Monitor performance improvements

**Disabling GPU Mode:**
- Toggle back to CPU mode
- System automatically falls back to CPU processing
- Useful for troubleshooting or power saving

### Custom Visualizations

**Creating Custom Charts:**
1. Select chart type in Visualization Panel
2. Choose asset and timeframe
3. Click "Generate Chart"
4. Customize display options
5. Export or save as needed

**Advanced Chart Features:**
- **Technical Indicators**: Add RSI, MACD, Bollinger Bands
- **Pattern Recognition**: AI-detected chart patterns
- **Risk Visualization**: Portfolio risk analysis
- **Performance Metrics**: Strategy performance charts

## 🚨 Troubleshooting

### Common Issues

**Interface Won't Load:**
- Check if the server is running
- Verify the correct URL (http://localhost:8080)
- Check browser console for errors
- Restart the server if needed

**No Real-time Updates:**
- Check WebSocket connection status
- Verify network connectivity
- Refresh the page
- Check system logs for connection issues

**Strategy Execution Fails:**
- Verify API credentials are correct
- Check profile activation status
- Review position limits
- Check system logs for error details

**GPU Toggle Not Working:**
- Verify CUDA installation
- Check GPU drivers
- Review hardware detection logs
- Restart the system if needed

### Performance Optimization

**For Better Performance:**
- Use GPU acceleration when available
- Close unnecessary browser tabs
- Ensure stable internet connection
- Monitor system resources

**Reducing Resource Usage:**
- Disable unnecessary visualizations
- Reduce update frequency
- Use CPU mode if GPU is causing issues
- Close unused browser windows

## 📱 Mobile Compatibility

The web interface is designed to work on mobile devices:
- **Responsive Design**: Adapts to different screen sizes
- **Touch Controls**: Optimized for touch interaction
- **Mobile Browsers**: Compatible with iOS Safari and Android Chrome
- **Offline Capabilities**: Basic functionality when offline

## 🔒 Security Features

### Data Protection
- **Encrypted Communications**: All data is encrypted in transit
- **Secure API Integration**: Protected trading platform connections
- **Session Management**: Secure session handling
- **Access Controls**: User authentication and authorization

### Best Practices
- **Use HTTPS**: Always use secure connections
- **Regular Updates**: Keep the system updated
- **Monitor Access**: Check for unauthorized access
- **Secure Credentials**: Never share API keys

## 📊 Analytics and Reporting

### Performance Metrics
- **Portfolio Analytics**: Detailed portfolio performance
- **Strategy Analysis**: Individual strategy effectiveness
- **Risk Assessment**: Comprehensive risk analysis
- **System Health**: Overall system performance

### Export Options
- **Chart Exports**: Save visualizations as images
- **Data Exports**: Export trading data to CSV/JSON
- **Report Generation**: Create comprehensive reports
- **Log Exports**: Save system logs for analysis

---

**Need Help?** Check the system logs, review this guide, or consult the troubleshooting section for common solutions. 