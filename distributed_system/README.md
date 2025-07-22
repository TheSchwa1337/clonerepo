# 🧠 Schwabot Distributed Trading System

A **grab-and-go** distributed trading system that automatically detects hardware, installs dependencies, and sets up a complete multi-node trading infrastructure with Discord integration.

## 🚀 **ONE-COMMAND INSTALLATION**

### Quick Start (Any Linux Machine)

```bash
# Master node (GPU with 8GB+ or 8+ CPU cores)
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash

# Worker node (any machine)
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash worker

# Specific node type
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash master
```

### What Happens Automatically

1. **Hardware Detection**: GPU/CPU, memory, platform detection
2. **Dependency Installation**: Python, CUDA, trading libraries
3. **Configuration Setup**: Master/worker configs, environment variables
4. **Service Installation**: Systemd services for auto-start
5. **Discord Integration**: Bot setup and configuration
6. **Dashboard Creation**: Web-based monitoring interface

## 🏗️ **SYSTEM ARCHITECTURE**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MASTER NODE   │    │  WORKER NODE 1  │    │  WORKER NODE N  │
│                 │    │                 │    │                 │
│ • Flask API     │◄──►│ • Task Executor │    │ • Task Executor │
│ • Discord Bot   │    │ • GPU/CPU Tasks │    │ • GPU/CPU Tasks │
│ • Task Router   │    │ • Heartbeat     │    │ • Heartbeat     │
│ • Monitoring    │    │ • Auto-register │    │ • Auto-register │
│ • Trading Logic │    │ • Performance   │    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  SHARED STORAGE │
                    │                 │
                    │ • Task Results  │
                    │ • Trading Data  │
                    │ • Models        │
                    │ • Logs          │
                    └─────────────────┘
```

## 🖥️ **HARDWARE REQUIREMENTS**

### Master Node
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, 3080, 3090, etc.)
- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space

### Worker Nodes
- **GPU**: Any NVIDIA GPU (GTX 1060, RTX 2060, etc.)
- **CPU**: 4+ cores
- **RAM**: 8GB+ recommended
- **Storage**: 20GB+ free space

### Raspberry Pi Support
- **Raspberry Pi 4**: 4GB+ RAM
- **Raspberry Pi 5**: 8GB RAM
- **Storage**: 32GB+ SD card

## 📦 **AUTO-INSTALLATION FEATURES**

### Automatic Detection
- ✅ **Operating System**: Ubuntu, Debian, CentOS, Arch, macOS
- ✅ **Hardware**: CPU cores, GPU model, memory capacity
- ✅ **Capabilities**: CUDA support, PyTorch compatibility
- ✅ **Network**: IP address, port availability

### Smart Configuration
- ✅ **Node Type**: Auto-determines master vs worker
- ✅ **Port Assignment**: Automatic port selection
- ✅ **Service Setup**: Systemd services for auto-start
- ✅ **Environment**: Virtual environment with all dependencies

### Discord Integration
- ✅ **Bot Setup**: Automatic Discord bot configuration
- ✅ **Commands**: `/status`, `/nodes`, `/trading`, `/ai`
- ✅ **Real-time Updates**: Live system status and alerts
- ✅ **AI Consultation**: Direct AI queries with system context

## 🎯 **USE CASES**

### Single Machine Setup
```bash
# One powerful machine as master
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash
```

### Multi-Machine Network
```bash
# Master machine (your best GPU rig)
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash master

# Worker machines (your other computers)
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash worker
```

### Raspberry Pi Cluster
```bash
# Each Pi as a worker node
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash worker
```

## 🔧 **MANUAL INSTALLATION**

If you prefer manual installation:

```bash
# Clone repository
git clone https://github.com/schwabot/Schwabot.git ~/Schwabot
cd ~/Schwabot

# Run installer
chmod +x distributed_system/auto_install.sh
./distributed_system/auto_install.sh [master|worker]
```

## 🚀 **STARTING THE SYSTEM**

### Master Node
```bash
cd ~/Schwabot
./start_master.sh

# Or use systemd service
sudo systemctl enable schwabot-master
sudo systemctl start schwabot-master
```

### Worker Node
```bash
cd ~/Schwabot
# Edit config/worker_config.yaml to set master IP
./start_worker.sh

# Or use systemd service
sudo systemctl enable schwabot-worker
sudo systemctl start schwabot-worker
```

## 📊 **MONITORING & CONTROL**

### Web Dashboard
- **URL**: http://localhost:5000
- **Features**: Real-time system status, node monitoring, trading metrics

### Discord Commands
```
/status          - Get system status
/nodes           - List connected nodes
/trading         - Get trading status
/ai <query>      - Consult AI with system context
```

### Log Monitoring
```bash
# Master node logs
tail -f ~/Schwabot/master_node.log

# Worker node logs
tail -f ~/Schwabot/worker_node.log

# Installation log
tail -f ~/Schwabot/install.log
```

## 🔄 **TASK DISTRIBUTION**

### Automatic Task Routing
- **GPU Tasks**: Fractal analysis, quantum calculations, model training
- **CPU Tasks**: Data processing, backtesting, statistical analysis
- **Memory Tasks**: Large dataset processing, model inference

### Task Types
- `fractal_analysis` - Market pattern analysis
- `quantum_calculation` - Quantum-inspired calculations
- `waveform_processing` - Signal processing
- `entropy_analysis` - Information theory analysis
- `temporal_analysis` - Time series analysis
- `backtest` - Trading strategy backtesting

## 🔐 **SECURITY & CONFIGURATION**

### Environment Variables
```bash
# .env file
NODE_TYPE=master|worker
MASTER_HOST=localhost
MASTER_PORT=5000
WORKER_PORT=5001
DISCORD_TOKEN=your_discord_token
DISCORD_CHANNEL_ID=your_channel_id
OBSERVATION_MODE=true
AUTO_TRADE=false
```

### Network Security
- **Firewall**: Configure ports 5000-5010
- **Authentication**: Optional API key authentication
- **SSL**: HTTPS support for production

## 📈 **TRADING FEATURES**

### Observation Mode
- **Duration**: 72 hours minimum
- **Performance**: 1.5% minimum profit requirement
- **Safety**: No real trading until requirements met

### Live Trading
- **Multi-Profile**: Support for multiple trading accounts
- **Risk Management**: Automatic stop-loss and position sizing
- **Performance Tracking**: Real-time profit/loss monitoring

### Strategy Integration
- **Kaprekar Analysis**: Advanced mathematical analysis
- **Quantum Smoothing**: Quantum-inspired signal processing
- **Cross-Chain**: Multi-blockchain analysis
- **AI Integration**: Machine learning model support

## 🛠️ **TROUBLESHOOTING**

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep :5000

# Kill the process or change port in config
```

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Install drivers if needed
sudo apt install nvidia-driver-470
```

#### Connection Refused
```bash
# Check firewall
sudo ufw status

# Allow ports
sudo ufw allow 5000:5010/tcp
```

#### Discord Bot Not Responding
```bash
# Check token and permissions
# Verify bot is in the correct channel
# Check bot permissions
```

### Log Analysis
```bash
# Check for errors
grep -i error ~/Schwabot/*.log

# Check for warnings
grep -i warning ~/Schwabot/*.log

# Monitor real-time
tail -f ~/Schwabot/*.log
```

## 🔄 **UPDATES & MAINTENANCE**

### System Updates
```bash
cd ~/Schwabot
git pull origin main
./distributed_system/auto_install.sh
```

### Backup
```bash
# Backup configuration
cp -r ~/Schwabot/config ~/schwabot_backup_config

# Backup data
cp -r ~/Schwabot/shared_data ~/schwabot_backup_data
```

### Performance Monitoring
```bash
# System resources
htop

# GPU usage
nvidia-smi

# Network usage
iotop
```

## 📚 **ADVANCED CONFIGURATION**

### Custom Task Types
```python
# Add custom task in worker_node.py
def _run_custom_task(self, task_data):
    # Your custom logic here
    return {'result': 'custom_result'}
```

### Custom Discord Commands
```python
# Add custom command in master_node.py
@self.discord_bot.command(name='custom')
async def custom_command(ctx, *, query):
    # Your custom logic here
    await ctx.send("Custom response")
```

### Load Balancing
```yaml
# config/master_config.yaml
nodes:
  load_balancing:
    enabled: true
    algorithm: round_robin  # or least_loaded
    health_check_interval: 30
```

## 🤝 **CONTRIBUTING**

### Development Setup
```bash
# Clone repository
git clone https://github.com/schwabot/Schwabot.git
cd Schwabot

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 **LICENSE**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **SUPPORT**

### Documentation
- [Quick Start Guide](QUICK_START.md)
- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)

### Community
- [Discord Server](https://discord.gg/schwabot)
- [GitHub Issues](https://github.com/schwabot/Schwabot/issues)
- [Wiki](https://github.com/schwabot/Schwabot/wiki)

### Emergency Contacts
- **Critical Issues**: Create GitHub issue with "URGENT" label
- **Security Issues**: Email security@schwabot.com
- **Discord**: Join our Discord server for real-time support

---

## 🎉 **GETTING STARTED**

Ready to start? Run this command on any Linux machine:

```bash
curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash
```

Your distributed trading system will be ready in minutes! 🚀

---

**Made with ❤️ by the Schwabot Team** 