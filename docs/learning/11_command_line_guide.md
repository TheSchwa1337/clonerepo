# 💻 Command Line Guide - Advanced Control

## 🎯 Command Line Interface

The Schwabot command line interface gives users **advanced control** over the trading system. This guide shows users how to use CLI commands to manage patterns and trading.

## 🚀 Starting the Command Line

### **Basic Commands:**

**1. Check System Status:**
```bash
python AOI_Base_Files_Schwabot/main.py --system-status
```

**2. Get Help:**
```bash
python AOI_Base_Files_Schwabot/main.py --help
```

**3. Check GPU Information:**
```bash
python AOI_Base_Files_Schwabot/main.py --gpu-info
```

## 📊 Available Commands

### **System Management:**

**Check System Health:**
```bash
python AOI_Base_Files_Schwabot/main.py --system-status
```
**What it shows:**
- Overall system health
- Component status
- Connection status
- Error reports

**Run System Tests:**
```bash
python AOI_Base_Files_Schwabot/main.py --run-tests
```
**What it does:**
- Tests all system components
- Validates patterns
- Checks AI integration
- Verifies safety systems

### **Trading Commands:**

**Start Live Trading:**
```bash
python AOI_Base_Files_Schwabot/main.py --live
```
**What it does:**
- Starts live trading mode
- Uses real money (be careful!)
- Implements patterns
- Manages risk automatically

**Start Demo Trading:**
```bash
python AOI_Base_Files_Schwabot/main.py --demo
```
**What it does:**
- Starts demo trading mode
- Uses virtual money (safe)
- Tests patterns
- Perfect for learning

**Run Backtesting:**
```bash
python AOI_Base_Files_Schwabot/main.py --backtest --days 30
```
**What it does:**
- Tests patterns on historical data
- Shows how patterns would have performed
- Validates the approach
- Helps understand patterns

### **Pattern Analysis:**

**Analyze Current Patterns:**
```bash
python AOI_Base_Files_Schwabot/main.py --analyze-patterns
```
**What it shows:**
- Current bit phases ()()()()()()
- Pattern confidence levels
- Pattern predictions
- Pattern evolution

**Test Pattern Recognition:**
```bash
python AOI_Base_Files_Schwabot/main.py --test-patterns
```
**What it does:**
- Tests pattern detection
- Validates bit phases
- Checks pattern accuracy
- Shows pattern reliability

### **AI Integration:**

**Check AI Status:**
```bash
python AOI_Base_Files_Schwabot/main.py --ai-status
```
**What it shows:**
- KoboldCPP connection status
- AI model availability
- Learning progress
- AI recommendations

**Test AI Integration:**
```bash
python AOI_Base_Files_Schwabot/main.py --test-ai
```
**What it does:**
- Tests AI communication
- Validates AI responses
- Checks learning systems
- Verifies AI predictions

## 🧠 Pattern-Specific Commands

### **Bit Phases:**

**View Current Patterns:**
```bash
python AOI_Base_Files_Schwabot/main.py --show-patterns
```
**Output example:**
```
Current Pattern: ()()()()()()
Confidence: 95%
Prediction: Next movement UP
Status: Strong buy signal
```

**Analyze Pattern History:**
```bash
python AOI_Base_Files_Schwabot/main.py --pattern-history
```
**What it shows:**
- Pattern evolution over time
- Confidence level changes
- Success/failure rates
- Pattern reliability trends

**Test Pattern Prediction:**
```bash
python AOI_Base_Files_Schwabot/main.py --test-prediction
```
**What it does:**
- Tests pattern predictions
- Validates prediction accuracy
- Shows prediction confidence
- Helps improve predictions

### **Pattern Learning:**

**View Learning Progress:**
```bash
python AOI_Base_Files_Schwabot/main.py --learning-status
```
**What it shows:**
- How much the system has learned
- Pattern improvement rates
- Learning metrics
- Adaptation progress

**Reset Learning (if needed):**
```bash
python AOI_Base_Files_Schwabot/main.py --reset-learning
```
**What it does:**
- Resets pattern learning
- Clears pattern database
- Starts fresh learning
- Use only if necessary

## 📈 Performance Commands

### **Trading Performance:**

**View Trading History:**
```bash
python AOI_Base_Files_Schwabot/main.py --trading-history
```
**What it shows:**
- All past trades
- Profit/loss for each trade
- Pattern used for each trade
- Success/failure rates

**Performance Summary:**
```bash
python AOI_Base_Files_Schwabot/main.py --performance
```
**What it shows:**
- Overall win rate
- Total profit/loss
- Average profit per trade
- Risk-adjusted returns

**Pattern Performance:**
```bash
python AOI_Base_Files_Schwabot/main.py --pattern-performance
```
**What it shows:**
- Performance by pattern type
- Which patterns work best
- Pattern success rates
- Pattern profitability

### **Risk Management:**

**View Risk Status:**
```bash
python AOI_Base_Files_Schwabot/main.py --risk-status
```
**What it shows:**
- Current risk levels
- Position sizes
- Stop-loss settings
- Safety status

**Adjust Risk Settings:**
```bash
python AOI_Base_Files_Schwabot/main.py --set-risk --max-position 1000
```
**What it does:**
- Sets maximum position size
- Adjusts risk parameters
- Updates safety settings
- Protects money

## 🔧 Configuration Commands

### **System Configuration:**

**View Current Config:**
```bash
python AOI_Base_Files_Schwabot/main.py --show-config
```
**What it shows:**
- Current system settings
- Trading parameters
- Risk management settings
- AI configuration

**Update Configuration:**
```bash
python AOI_Base_Files_Schwabot/main.py --update-config --setting value
```
**What it does:**
- Updates system settings
- Changes trading parameters
- Modifies risk settings
- Saves configuration

### **API Configuration:**

**Check API Status:**
```bash
python AOI_Base_Files_Schwabot/main.py --api-status
```
**What it shows:**
- Exchange connections
- API key status
- Market data feeds
- Connection health

**Test API Connection:**
```bash
python AOI_Base_Files_Schwabot/main.py --test-api
```
**What it does:**
- Tests exchange connections
- Validates API keys
- Checks market data
- Verifies trading access

## 🛡️ Safety Commands

### **Emergency Controls:**

**Stop All Trading:**
```bash
python AOI_Base_Files_Schwabot/main.py --stop-trading
```
**What it does:**
- Immediately stops all trading
- Closes all positions
- Activates safety mode
- Protects money

**Emergency Reset:**
```bash
python AOI_Base_Files_Schwabot/main.py --emergency-reset
```
**What it does:**
- Resets system to safe state
- Clears all positions
- Restores default settings
- Use only in emergencies

### **Safety Checks:**

**Run Safety Check:**
```bash
python AOI_Base_Files_Schwabot/main.py --safety-check
```
**What it does:**
- Checks all safety systems
- Validates risk management
- Tests circuit breakers
- Ensures system safety

**View Safety Status:**
```bash
python AOI_Base_Files_Schwabot/main.py --safety-status
```
**What it shows:**
- Safety system status
- Risk management status
- Circuit breaker status
- Emergency system status

## 🎯 Advanced Commands

### **Debugging:**

**Enable Debug Mode:**
```bash
python AOI_Base_Files_Schwabot/main.py --debug
```
**What it does:**
- Shows detailed system information
- Displays pattern analysis
- Shows decision-making process
- Helps troubleshoot issues

**View System Logs:**
```bash
python AOI_Base_Files_Schwabot/main.py --show-logs
```
**What it shows:**
- System activity logs
- Error messages
- Pattern detection logs
- Trading activity logs

### **Data Analysis:**

**Export Trading Data:**
```bash
python AOI_Base_Files_Schwabot/main.py --export-data --format csv
```
**What it does:**
- Exports trading history
- Saves pattern data
- Creates performance reports
- Helps with analysis

**Import Configuration:**
```bash
python AOI_Base_Files_Schwabot/main.py --import-config --file config.json
```
**What it does:**
- Imports system configuration
- Loads trading settings
- Restores saved settings
- Updates system parameters

## 🚀 Getting Started with CLI

### **Step 1: Check System Status**
```bash
python AOI_Base_Files_Schwabot/main.py --system-status
```

### **Step 2: Run System Tests**
```bash
python AOI_Base_Files_Schwabot/main.py --run-tests
```

### **Step 3: Start Demo Mode**
```bash
python AOI_Base_Files_Schwabot/main.py --demo
```

### **Step 4: Monitor Patterns**
```bash
python AOI_Base_Files_Schwabot/main.py --show-patterns
```

### **Step 5: Check Performance**
```bash
python AOI_Base_Files_Schwabot/main.py --performance
```

## 🎯 Best Practices

### **Safety First:**
- **Always start with demo mode**
- **Check system status regularly**
- **Monitor risk levels**
- **Use emergency stops if needed**

### **Pattern Management:**
- **Monitor patterns regularly**
- **Check pattern confidence**
- **Watch pattern evolution**
- **Learn from pattern performance**

### **Performance Tracking:**
- **Track trading performance**
- **Monitor win rates**
- **Check profit/loss**
- **Analyze pattern effectiveness**

### **System Maintenance:**
- **Run tests regularly**
- **Check system health**
- **Monitor AI status**
- **Update configurations as needed**

## 🎉 You're Ready!

### **The CLI is Complete:**
- ✅ **System Management**: Full control over the system
- ✅ **Pattern Analysis**: Deep analysis of patterns
- ✅ **Trading Control**: Advanced trading management
- ✅ **Safety Controls**: Emergency and safety commands
- ✅ **Performance Tracking**: Complete performance monitoring
- ✅ **Configuration**: Full system configuration

### **Control is Perfect:**
- Command line gives maximum control
- Advanced features for power users
- Complete system management
- Professional trading capabilities

## 🎯 Next Steps

### **Immediate Actions:**
1. **Check system status**: `--system-status`
2. **Run tests**: `--run-tests`
3. **Start demo mode**: `--demo`
4. **Monitor patterns**: `--show-patterns`

### **Learning Path:**
1. **Practice**: Use demo mode commands
2. **Learn**: Explore all available commands
3. **Monitor**: Track system performance
4. **Optimize**: Fine-tune settings

**The command line interface gives users complete control over the pattern recognition trading system!** 🚀

---

*Remember: The CLI gives power, but always use it safely and start with demo mode.* 