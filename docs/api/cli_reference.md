# 💻 Command Line Interface Reference

## Overview

The Schwabot Command Line Interface (CLI) provides advanced control and automation capabilities for experienced users. This reference covers all available commands, options, and usage patterns.

## 🚀 Basic Usage

### Command Structure
```bash
python AOI_Base_Files_Schwabot/main.py [COMMAND] [OPTIONS]
```

### Getting Help
```bash
# Show all available commands
python AOI_Base_Files_Schwabot/main.py --help

# Show help for specific command
python AOI_Base_Files_Schwabot/main.py [COMMAND] --help
```

## 📋 Available Commands

### System Status Commands

#### `--system-status`
Display comprehensive system status information.

```bash
python AOI_Base_Files_Schwabot/main.py --system-status
```

**Output includes:**
- System operational status
- Component health checks
- API connection status
- GPU detection results
- Memory and resource usage
- Active trading sessions

#### `--gpu-info`
Display detailed GPU information and detection results.

```bash
python AOI_Base_Files_Schwabot/main.py --gpu-info
```

**Output includes:**
- Detected GPU models
- CUDA/OpenCL availability
- Memory capacity and usage
- Performance tier classification
- Fallback chain information
- Optimal configuration details

#### `--gpu-auto-detect`
Enable enhanced GPU auto-detection and optimization.

```bash
python AOI_Base_Files_Schwabot/main.py --gpu-auto-detect
```

**Features:**
- Automatic GPU detection
- Performance optimization
- Fallback chain setup
- Memory management
- Backend selection

### Testing Commands

#### `--run-tests`
Execute comprehensive system tests.

```bash
python AOI_Base_Files_Schwabot/main.py --run-tests
```

**Test Categories:**
- Core component functionality
- Risk management systems
- Trading pipeline validation
- Mathematical framework tests
- Error handling verification
- Performance benchmarks

**Test Results:**
- Individual test pass/fail status
- Performance metrics
- Error reports
- System recommendations

#### `--test-imports`
Verify all module imports are working correctly.

```bash
python AOI_Base_Files_Schwabot/main.py --test-imports
```

**Checks:**
- Core module availability
- Dependency verification
- Import error detection
- Module compatibility

### Trading Commands

#### `--backtest`
Run backtesting with historical data.

```bash
# Basic backtest for 30 days
python AOI_Base_Files_Schwabot/main.py --backtest --days 30

# Custom backtest with specific parameters
python AOI_Base_Files_Schwabot/main.py --backtest --days 60 --symbol BTC/USDC --strategy default
```

**Options:**
- `--days N`: Number of days to backtest
- `--symbol PAIR`: Trading pair to test
- `--strategy NAME`: Strategy to use
- `--config FILE`: Configuration file path

**Output:**
- Performance metrics
- Trade history
- Risk analysis
- Profit/loss summary

#### `--live`
Start live trading mode.

```bash
# Start live trading with default config
python AOI_Base_Files_Schwabot/main.py --live

# Start with custom configuration
python AOI_Base_Files_Schwabot/main.py --live --config my_config.yaml
```

**Options:**
- `--config FILE`: Configuration file path
- `--profile NAME`: Trading profile to use
- `--demo`: Run in demo mode (no real trades)

**Features:**
- Real-time market data
- Live trade execution
- Risk management
- Portfolio tracking

### Monitoring Commands

#### `--error-log`
Display system error logs.

```bash
# Show last 50 error entries
python AOI_Base_Files_Schwabot/main.py --error-log --limit 50

# Show errors from specific time period
python AOI_Base_Files_Schwabot/main.py --error-log --since "2024-01-01"
```

**Options:**
- `--limit N`: Number of entries to show
- `--since DATE`: Show entries since date
- `--level LEVEL`: Minimum log level (INFO, WARNING, ERROR)

#### `--hash-log`
Log hash-based trading decisions.

```bash
# Log decisions for specific symbol
python AOI_Base_Files_Schwabot/main.py --hash-log --symbol BTC/USDC

# Log with custom parameters
python AOI_Base_Files_Schwabot/main.py --hash-log --symbol ETH/USDC --duration 3600
```

**Options:**
- `--symbol PAIR`: Trading pair to monitor
- `--duration SECONDS`: Logging duration
- `--output FILE`: Output file path

#### `--fetch-hash-decision`
Retrieve hash-based trading decisions.

```bash
python AOI_Base_Files_Schwabot/main.py --fetch-hash-decision
```

**Output:**
- Recent trading decisions
- Decision confidence levels
- Market conditions
- Execution results

### Management Commands

#### `--reset-circuit-breakers`
Reset all circuit breakers and safety systems.

```bash
python AOI_Base_Files_Schwabot/main.py --reset-circuit-breakers
```

**Use Cases:**
- System recovery after errors
- Clearing safety locks
- Resetting risk management
- Emergency system restart

#### `--sync-portfolio`
Synchronize portfolio data with exchange.

```bash
python AOI_Base_Files_Schwabot/main.py --sync-portfolio
```

**Actions:**
- Update portfolio balances
- Sync trade history
- Verify positions
- Reconcile discrepancies

## 🔧 Advanced Options

### Configuration Options

#### `--config FILE`
Specify custom configuration file.

```bash
python AOI_Base_Files_Schwabot/main.py --live --config custom_config.yaml
```

#### `--profile NAME`
Select specific trading profile.

```bash
python AOI_Base_Files_Schwabot/main.py --backtest --profile aggressive
```

#### `--log-level LEVEL`
Set logging verbosity.

```bash
python AOI_Base_Files_Schwabot/main.py --run-tests --log-level DEBUG
```

**Levels:**
- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages only

### Performance Options

#### `--gpu-acceleration`
Force GPU acceleration mode.

```bash
python AOI_Base_Files_Schwabot/main.py --backtest --gpu-acceleration
```

#### `--cpu-only`
Force CPU-only mode.

```bash
python AOI_Base_Files_Schwabot/main.py --backtest --cpu-only
```

#### `--memory-limit GB`
Set memory usage limit.

```bash
python AOI_Base_Files_Schwabot/main.py --backtest --memory-limit 8
```

## 📊 Output Formats

### JSON Output
Most commands support JSON output for programmatic use.

```bash
python AOI_Base_Files_Schwabot/main.py --system-status --json
```

### Verbose Output
Get detailed information with verbose mode.

```bash
python AOI_Base_Files_Schwabot/main.py --run-tests --verbose
```

### Quiet Mode
Minimize output for automation.

```bash
python AOI_Base_Files_Schwabot/main.py --system-status --quiet
```

## 🔄 Automation Examples

### System Monitoring Script
```bash
#!/bin/bash
# Monitor system health every 5 minutes

while true; do
    python AOI_Base_Files_Schwabot/main.py --system-status --json > status.json
    python AOI_Base_Files_Schwabot/main.py --error-log --limit 10 > errors.log
    
    # Check for critical errors
    if grep -q "CRITICAL" errors.log; then
        echo "Critical error detected!"
        python AOI_Base_Files_Schwabot/main.py --reset-circuit-breakers
    fi
    
    sleep 300
done
```

### Automated Backtesting
```bash
#!/bin/bash
# Run daily backtests

DATE=$(date +%Y-%m-%d)
python AOI_Base_Files_Schwabot/main.py --backtest --days 30 --json > "backtest_${DATE}.json"
python AOI_Base_Files_Schwabot/main.py --backtest --days 7 --json > "backtest_weekly_${DATE}.json"
```

### Portfolio Sync
```bash
#!/bin/bash
# Sync portfolio every hour

while true; do
    python AOI_Base_Files_Schwabot/main.py --sync-portfolio
    sleep 3600
done
```

## 🚨 Error Handling

### Common Error Codes
- `1`: General error
- `2`: Configuration error
- `3`: API connection error
- `4`: GPU detection error
- `5`: Trading error

### Error Recovery
```bash
# Reset system after error
python AOI_Base_Files_Schwabot/main.py --reset-circuit-breakers

# Check error logs
python AOI_Base_Files_Schwabot/main.py --error-log --limit 20

# Verify system status
python AOI_Base_Files_Schwabot/main.py --system-status
```

## 📈 Performance Tuning

### GPU Optimization
```bash
# Check GPU capabilities
python AOI_Base_Files_Schwabot/main.py --gpu-info

# Enable GPU acceleration
python AOI_Base_Files_Schwabot/main.py --gpu-auto-detect

# Monitor GPU performance
python AOI_Base_Files_Schwabot/main.py --system-status --gpu-metrics
```

### Memory Management
```bash
# Set memory limits
python AOI_Base_Files_Schwabot/main.py --backtest --memory-limit 4

# Monitor memory usage
python AOI_Base_Files_Schwabot/main.py --system-status --memory-info
```

## 🔒 Security Commands

### API Key Management
```bash
# Verify API connections
python AOI_Base_Files_Schwabot/main.py --verify-api-keys

# Test API permissions
python AOI_Base_Files_Schwabot/main.py --test-api-permissions
```

### Encryption Status
```bash
# Check encryption status
python AOI_Base_Files_Schwabot/main.py --encryption-status

# Verify secure communications
python AOI_Base_Files_Schwabot/main.py --verify-security
```

## 📝 Logging and Debugging

### Log Management
```bash
# View recent logs
python AOI_Base_Files_Schwabot/main.py --view-logs --limit 100

# Export logs to file
python AOI_Base_Files_Schwabot/main.py --export-logs --output logs.txt

# Clear old logs
python AOI_Base_Files_Schwabot/main.py --clear-logs --older-than 7
```

### Debug Mode
```bash
# Enable debug logging
python AOI_Base_Files_Schwabot/main.py --run-tests --debug

# Debug specific component
python AOI_Base_Files_Schwabot/main.py --debug-component trading_engine
```

## 🎯 Best Practices

### Command Organization
1. **Always check system status first**
   ```bash
   python AOI_Base_Files_Schwabot/main.py --system-status
   ```

2. **Run tests before live trading**
   ```bash
   python AOI_Base_Files_Schwabot/main.py --run-tests
   ```

3. **Monitor error logs regularly**
   ```bash
   python AOI_Base_Files_Schwabot/main.py --error-log --limit 50
   ```

4. **Use appropriate log levels**
   - `INFO` for normal operation
   - `DEBUG` for troubleshooting
   - `ERROR` for automation

### Automation Guidelines
- Always include error handling
- Use JSON output for parsing
- Implement proper logging
- Set appropriate timeouts
- Monitor system resources

### Security Considerations
- Never log API keys
- Use secure configuration files
- Monitor access patterns
- Regular security audits
- Keep system updated

---

**Need More Help?** Check the system logs, review the troubleshooting guide, or consult the main documentation for additional information. 