# ⚙️ Configuration Setup Guide

## Overview

Proper configuration is essential for Schwabot to function correctly and securely. This guide covers all configuration files, settings, and best practices for setting up your trading environment.

## 📁 Configuration Directory Structure

```
AOI_Base_Files_Schwabot/
├── config/
│   ├── coinbase_profiles.yaml      # API credentials and profiles
│   ├── unified_settings.yaml       # Core system settings
│   ├── trading_config.yaml         # Trading parameters
│   ├── risk_config.yaml           # Risk management settings
│   └── gpu_config.yaml            # GPU acceleration settings
```

## 🔑 API Configuration

### Setting Up Coinbase API

The `config/coinbase_profiles.yaml` file manages your trading API credentials and profiles.

#### Basic Configuration
```yaml
# config/coinbase_profiles.yaml
profiles:
  default:
    api_key: "your_api_key_here"
    api_secret: "your_api_secret_here"
    sandbox: true  # Set to false for live trading
    trading_pairs: ["BTC/USDC", "ETH/USDC"]
    position_limits:
      max_open_positions: 5
      max_position_size: 1000
    risk_settings:
      max_daily_loss: 100
      stop_loss_percentage: 0.05
      take_profit_percentage: 0.10
```

#### Multiple Profile Configuration
```yaml
profiles:
  conservative:
    api_key: "conservative_api_key"
    api_secret: "conservative_api_secret"
    sandbox: true
    trading_pairs: ["BTC/USDC"]
    position_limits:
      max_open_positions: 2
      max_position_size: 500
    risk_settings:
      max_daily_loss: 50
      stop_loss_percentage: 0.03
      take_profit_percentage: 0.08

  aggressive:
    api_key: "aggressive_api_key"
    api_secret: "aggressive_api_secret"
    sandbox: false  # Live trading
    trading_pairs: ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    position_limits:
      max_open_positions: 10
      max_position_size: 2000
    risk_settings:
      max_daily_loss: 200
      stop_loss_percentage: 0.07
      take_profit_percentage: 0.15
```

#### Profile Settings Explained

**Basic Settings:**
- `api_key`: Your Coinbase API key
- `api_secret`: Your Coinbase API secret
- `sandbox`: `true` for testing, `false` for live trading

**Trading Configuration:**
- `trading_pairs`: List of trading pairs to monitor
- `max_open_positions`: Maximum number of simultaneous trades
- `max_position_size`: Maximum size of any single position

**Risk Management:**
- `max_daily_loss`: Maximum daily loss limit in USD
- `stop_loss_percentage`: Automatic stop-loss percentage
- `take_profit_percentage`: Automatic take-profit percentage

### Security Best Practices

1. **Never commit API keys to version control**
   ```bash
   # Add to .gitignore
   echo "config/coinbase_profiles.yaml" >> .gitignore
   ```

2. **Use environment variables for sensitive data**
   ```bash
   # Set environment variables
   export COINBASE_API_KEY="your_key"
   export COINBASE_API_SECRET="your_secret"
   ```

3. **Create a template file**
   ```yaml
   # config/coinbase_profiles.template.yaml
   profiles:
     default:
       api_key: "${COINBASE_API_KEY}"
       api_secret: "${COINBASE_API_SECRET}"
       sandbox: true
       # ... other settings
   ```

## ⚙️ Core System Configuration

### Unified Settings Configuration

The `config/unified_settings.yaml` file controls core system behavior.

#### Basic Configuration
```yaml
# config/unified_settings.yaml
system:
  name: "Schwabot Trading System"
  version: "0.5.0"
  environment: "development"  # development, staging, production
  debug_mode: false
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR

trading:
  default_mode: "demo"  # demo, live, backtest
  auto_start: false
  session_timeout: 3600  # seconds
  max_concurrent_sessions: 5

mathematical_framework:
  tensor_algebra_enabled: true
  quantum_smoothing_enabled: true
  entropy_analysis_enabled: true
  confidence_threshold: 0.7
  signal_strength_threshold: 0.6

performance:
  gpu_acceleration: true
  memory_limit_gb: 8
  cpu_threads: 4
  cache_enabled: true
  cache_size_mb: 512

monitoring:
  health_check_interval: 300  # seconds
  performance_metrics_enabled: true
  alerting_enabled: true
  log_retention_days: 30
```

#### Advanced Configuration
```yaml
system:
  # ... basic settings ...
  
  security:
    encryption_enabled: true
    encryption_algorithm: "AES-256"
    session_encryption: true
    api_key_encryption: true
    
  networking:
    max_connections: 100
    connection_timeout: 30
    retry_attempts: 3
    rate_limiting_enabled: true

trading:
  # ... basic settings ...
  
  execution:
    order_timeout: 60  # seconds
    max_slippage: 0.002  # 0.2%
    retry_failed_orders: true
    partial_fills_enabled: true
    
  risk_management:
    circuit_breaker_enabled: true
    max_drawdown: 0.15  # 15%
    correlation_limit: 0.8
    volatility_threshold: 0.05

mathematical_framework:
  # ... basic settings ...
  
  tensor_operations:
    precision: "float32"  # float16, float32, float64
    parallel_processing: true
    memory_optimization: true
    
  quantum_parameters:
    smoothing_factor: 0.1
    collapse_threshold: 0.5
    entanglement_limit: 1000
    
  entropy_settings:
    calculation_method: "shannon"  # shannon, renyi, tsallis
    window_size: 100
    update_frequency: 60  # seconds
```

## 🎯 Trading Configuration

### Trading Parameters

The `config/trading_config.yaml` file defines trading-specific parameters.

```yaml
# config/trading_config.yaml
strategies:
  default:
    name: "Default Strategy"
    description: "Balanced risk-reward strategy"
    enabled: true
    parameters:
      confidence_threshold: 0.7
      signal_strength_threshold: 0.6
      position_size_percentage: 0.1
      max_positions: 5
      
  conservative:
    name: "Conservative Strategy"
    description: "Low-risk, steady returns"
    enabled: true
    parameters:
      confidence_threshold: 0.8
      signal_strength_threshold: 0.7
      position_size_percentage: 0.05
      max_positions: 3
      
  aggressive:
    name: "Aggressive Strategy"
    description: "High-risk, high-reward"
    enabled: false  # Disabled by default
    parameters:
      confidence_threshold: 0.6
      signal_strength_threshold: 0.5
      position_size_percentage: 0.2
      max_positions: 10

execution:
  order_types:
    - "market"
    - "limit"
    - "stop_loss"
    - "take_profit"
    
  default_order_type: "limit"
  limit_order_timeout: 300  # seconds
  market_order_slippage: 0.001  # 0.1%
  
  timing:
    execution_delay: 1  # seconds
    retry_interval: 5  # seconds
    max_retries: 3

analysis:
  technical_indicators:
    - "RSI"
    - "MACD"
    - "Bollinger_Bands"
    - "Moving_Averages"
    
  timeframes:
    - "1m"
    - "5m"
    - "15m"
    - "1h"
    - "4h"
    - "1d"
    
  default_timeframe: "15m"
```

## 🛡️ Risk Management Configuration

### Risk Settings

The `config/risk_config.yaml` file manages risk management parameters.

```yaml
# config/risk_config.yaml
circuit_breakers:
  enabled: true
  triggers:
    max_daily_loss:
      enabled: true
      threshold: 100  # USD
      action: "stop_trading"
      
    max_drawdown:
      enabled: true
      threshold: 0.15  # 15%
      action: "reduce_position_sizes"
      
    consecutive_losses:
      enabled: true
      threshold: 5
      action: "pause_trading"
      
    volatility_spike:
      enabled: true
      threshold: 0.05  # 5%
      action: "increase_stop_loss"

position_management:
  max_position_size: 0.1  # 10% of portfolio
  max_total_exposure: 0.5  # 50% of portfolio
  correlation_limit: 0.8
  
  stop_loss:
    enabled: true
    default_percentage: 0.05  # 5%
    trailing_enabled: true
    trailing_distance: 0.02  # 2%
    
  take_profit:
    enabled: true
    default_percentage: 0.10  # 10%
    trailing_enabled: true
    trailing_distance: 0.03  # 3%

portfolio_limits:
  max_open_positions: 10
  max_positions_per_asset: 3
  min_position_size: 10  # USD
  max_position_size: 1000  # USD
  
  diversification:
    max_allocation_per_asset: 0.3  # 30%
    min_assets: 2
    target_assets: 5
```

## 🎮 GPU Configuration

### GPU Acceleration Settings

The `config/gpu_config.yaml` file controls GPU acceleration parameters.

```yaml
# config/gpu_config.yaml
gpu:
  enabled: true
  auto_detect: true
  preferred_backend: "cuda"  # cuda, opencl, cpu
  
  cuda:
    enabled: true
    memory_fraction: 0.8  # Use 80% of GPU memory
    allow_growth: true
    device_id: 0  # Use first GPU
    
  opencl:
    enabled: true
    platform_id: 0
    device_id: 0
    
  fallback:
    enabled: true
    fallback_to_cpu: true
    memory_limit_gb: 4
    
performance:
  batch_size: 32
  precision: "float32"  # float16, float32, float64
  parallel_processing: true
  
  optimization:
    memory_optimization: true
    kernel_optimization: true
    cache_enabled: true
    
monitoring:
  gpu_metrics_enabled: true
  memory_monitoring: true
  temperature_monitoring: true
  performance_logging: true
```

## 🔧 Environment-Specific Configurations

### Development Environment
```yaml
# config/development.yaml
system:
  environment: "development"
  debug_mode: true
  log_level: "DEBUG"
  
trading:
  default_mode: "demo"
  auto_start: false
  
performance:
  gpu_acceleration: false  # Disable for development
  memory_limit_gb: 2
```

### Production Environment
```yaml
# config/production.yaml
system:
  environment: "production"
  debug_mode: false
  log_level: "WARNING"
  
trading:
  default_mode: "live"
  auto_start: true
  
performance:
  gpu_acceleration: true
  memory_limit_gb: 16
  
security:
  encryption_enabled: true
  session_encryption: true
```

## 📊 Configuration Validation

### Validation Script
Create a configuration validation script to check your settings:

```python
# scripts/validate_config.py
import yaml
import os
from pathlib import Path

def validate_config():
    config_dir = Path("AOI_Base_Files_Schwabot/config")
    
    # Check required files
    required_files = [
        "coinbase_profiles.yaml",
        "unified_settings.yaml",
        "trading_config.yaml",
        "risk_config.yaml"
    ]
    
    for file in required_files:
        file_path = config_dir / file
        if not file_path.exists():
            print(f"❌ Missing required config file: {file}")
            return False
    
    # Validate API configuration
    try:
        with open(config_dir / "coinbase_profiles.yaml", 'r') as f:
            profiles = yaml.safe_load(f)
            
        if not profiles or 'profiles' not in profiles:
            print("❌ Invalid coinbase_profiles.yaml structure")
            return False
            
        for profile_name, profile_data in profiles['profiles'].items():
            required_keys = ['api_key', 'api_secret', 'sandbox']
            for key in required_keys:
                if key not in profile_data:
                    print(f"❌ Missing {key} in profile {profile_name}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error validating API config: {e}")
        return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    validate_config()
```

### Running Validation
```bash
python scripts/validate_config.py
```

## 🔄 Configuration Management

### Environment Variables
Use environment variables for sensitive configuration:

```bash
# .env file
SCHWABOT_ENVIRONMENT=development
SCHWABOT_DEBUG=true
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
GPU_MEMORY_LIMIT=8
```

### Configuration Loading
```python
import os
from pathlib import Path

def load_config():
    env = os.getenv('SCHWABOT_ENVIRONMENT', 'development')
    config_file = f"config/{env}.yaml"
    
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Load default config
        with open("config/unified_settings.yaml", 'r') as f:
            return yaml.safe_load(f)
```

## 🚨 Security Considerations

### API Key Security
1. **Never store API keys in plain text**
2. **Use environment variables**
3. **Encrypt sensitive configuration**
4. **Regular key rotation**
5. **Monitor API usage**

### Configuration Security
1. **Restrict file permissions**
   ```bash
   chmod 600 config/coinbase_profiles.yaml
   ```

2. **Use secure configuration storage**
3. **Regular security audits**
4. **Monitor configuration changes**
5. **Backup configurations securely**

## 📝 Configuration Best Practices

### General Guidelines
1. **Start with demo mode** for testing
2. **Use conservative settings** initially
3. **Monitor and adjust** based on performance
4. **Document changes** to configuration
5. **Test configurations** before production use

### Performance Optimization
1. **Adjust memory limits** based on system capabilities
2. **Enable GPU acceleration** when available
3. **Optimize batch sizes** for your hardware
4. **Monitor resource usage** regularly
5. **Scale configurations** with system growth

### Risk Management
1. **Set appropriate position limits**
2. **Configure circuit breakers**
3. **Use stop-loss and take-profit**
4. **Monitor drawdown limits**
5. **Regular risk assessments**

---

**Need Help?** Check the troubleshooting guide or consult the main documentation for additional configuration options. 