#!/bin/bash
# -*- coding: utf-8 -*-
"""
🚀 SCHWABOT DISTRIBUTED TRADING SYSTEM - AUTO INSTALLER
=======================================================

One-command installer for the Schwabot distributed trading system.
This script will automatically detect hardware, install dependencies,
and configure the system for either master or worker node operation.

Usage:
    curl -sSL https://raw.githubusercontent.com/schwabot/Schwabot/main/distributed_system/auto_install.sh | bash
    OR
    ./auto_install.sh [master|worker]

Features:
- Automatic hardware detection (GPU/CPU)
- Dependency installation
- Configuration setup
- Service installation
- Discord integration setup
- Multi-profile trading support
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCHWABOT_DIR="$HOME/Schwabot"
REPO_URL="https://github.com/schwabot/Schwabot.git"
NODE_TYPE="${1:-auto}"  # master, worker, or auto
PYTHON_VERSION="3.8"
PIP_VERSION="21.0"

# Logging
LOG_FILE="$SCHWABOT_DIR/install.log"
mkdir -p "$SCHWABOT_DIR"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Banner
print_banner() {
    clear
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🧠 SCHWABOT DISTRIBUTED                   ║"
    echo "║                    🚀 TRADING SYSTEM                         ║"
    echo "║                    📦 AUTO INSTALLER                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "This installer will set up the Schwabot distributed trading system"
    echo "with automatic hardware detection and configuration."
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user."
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
        elif [[ -f /etc/redhat-release ]]; then
            OS="redhat"
        elif [[ -f /etc/arch-release ]]; then
            OS="arch"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Unsupported operating system: $OSTYPE"
    fi
    
    log "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    case $OS in
        "debian"|"ubuntu")
            sudo apt update
            sudo apt install -y \
                python3 python3-pip python3-venv \
                git curl wget \
                build-essential \
                libssl-dev libffi-dev \
                nvidia-cuda-toolkit \
                nvidia-driver-470 \
                htop iotop \
                screen tmux
            ;;
        "redhat"|"centos"|"fedora")
            sudo yum update -y
            sudo yum install -y \
                python3 python3-pip \
                git curl wget \
                gcc gcc-c++ make \
                openssl-devel libffi-devel \
                cuda-toolkit \
                nvidia-driver \
                htop iotop \
                screen tmux
            ;;
        "arch")
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip \
                git curl wget \
                base-devel \
                openssl libffi \
                cuda nvidia-utils \
                htop iotop \
                screen tmux
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python3 git curl wget htop
            ;;
    esac
    
    success "System dependencies installed"
}

# Check Python version
check_python() {
    log "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.8 or higher."
    fi
    
    PYTHON_VERSION_ACTUAL=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION_ACTUAL"
    
    if [[ $(echo "$PYTHON_VERSION_ACTUAL >= 3.8" | bc -l) -eq 0 ]]; then
        error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION_ACTUAL"
    fi
}

# Detect hardware capabilities
detect_hardware() {
    log "Detecting hardware capabilities..."
    
    # CPU info
    CPU_CORES=$(nproc)
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    log "CPU: $CPU_MODEL ($CPU_CORES cores)"
    
    # Memory info
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    log "Memory: ${MEMORY_GB}GB"
    
    # GPU detection
    GPU_AVAILABLE=false
    GPU_NAME="None"
    GPU_MEMORY_GB=0
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print $1/1024}')
        log "GPU: $GPU_NAME (${GPU_MEMORY_GB}GB)"
    else
        log "No NVIDIA GPU detected"
    fi
    
    # Determine node type
    if [[ "$NODE_TYPE" == "auto" ]]; then
        if [[ "$GPU_AVAILABLE" == "true" ]]; then
            if (( $(echo "$GPU_MEMORY_GB >= 8" | bc -l) )); then
                NODE_TYPE="master"
            else
                NODE_TYPE="worker"
            fi
        elif [[ "$CPU_CORES" -ge 8 ]]; then
            NODE_TYPE="master"
        else
            NODE_TYPE="worker"
        fi
    fi
    
    log "Node type: $NODE_TYPE"
}

# Clone repository
clone_repository() {
    log "Cloning Schwabot repository..."
    
    if [[ -d "$SCHWABOT_DIR" ]]; then
        warn "Directory $SCHWABOT_DIR already exists. Backing up..."
        mv "$SCHWABOT_DIR" "${SCHWABOT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    git clone "$REPO_URL" "$SCHWABOT_DIR"
    cd "$SCHWABOT_DIR"
    
    success "Repository cloned successfully"
}

# Create virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    cd "$SCHWABOT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    success "Virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    cd "$SCHWABOT_DIR"
    source venv/bin/activate
    
    # Install base requirements
    pip install -r requirements.txt
    
    # Install additional dependencies for distributed system
    pip install \
        flask flask-cors flask-socketio \
        psutil requests pyyaml python-dotenv \
        discord.py \
        numpy pandas scipy scikit-learn \
        matplotlib seaborn plotly \
        ccxt websocket-client \
        schedule
    
    # Install GPU dependencies if available
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log "Installing GPU dependencies..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install cupy-cuda11x
    fi
    
    success "Python dependencies installed"
}

# Create configuration files
setup_config() {
    log "Setting up configuration files..."
    
    cd "$SCHWABOT_DIR"
    
    # Create config directory
    mkdir -p config
    
    # Master node configuration
    if [[ "$NODE_TYPE" == "master" ]]; then
        cat > config/master_config.yaml << EOF
master:
  host: '0.0.0.0'
  port: 5000
  debug: false

discord:
  enabled: true
  token: ''  # Set your Discord bot token
  channel_id: ''  # Set your Discord channel ID

trading:
  observation_mode: true
  auto_trade: false
  backtest_duration: 72  # hours
  min_performance: 1.5  # percent

nodes:
  heartbeat_interval: 30
  timeout: 120
EOF
    fi
    
    # Worker node configuration
    if [[ "$NODE_TYPE" == "worker" ]]; then
        cat > config/worker_config.yaml << EOF
worker_port: 5001
master_host: 'localhost'  # Change to master node IP
master_port: 5000
heartbeat_interval: 30
task_poll_interval: 10
max_concurrent_tasks: 4
task_timeout: 300
auto_register: true
EOF
    fi
    
    # Environment file
    cat > .env << EOF
# Schwabot Distributed Trading System
NODE_TYPE=$NODE_TYPE
MASTER_HOST=localhost
MASTER_PORT=5000
WORKER_PORT=5001

# Discord Configuration (optional)
DISCORD_TOKEN=
DISCORD_CHANNEL_ID=

# Trading Configuration
OBSERVATION_MODE=true
AUTO_TRADE=false
BACKTEST_DURATION=72
MIN_PERFORMANCE=1.5

# Hardware Configuration
GPU_AVAILABLE=$GPU_AVAILABLE
CPU_CORES=$CPU_CORES
MEMORY_GB=$MEMORY_GB
EOF
    
    success "Configuration files created"
}

# Create startup scripts
create_startup_scripts() {
    log "Creating startup scripts..."
    
    cd "$SCHWABOT_DIR"
    
    # Master node startup script
    if [[ "$NODE_TYPE" == "master" ]]; then
        cat > start_master.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python distributed_system/master_node.py
EOF
        chmod +x start_master.sh
    fi
    
    # Worker node startup script
    if [[ "$NODE_TYPE" == "worker" ]]; then
        cat > start_worker.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python distributed_system/worker_node.py
EOF
        chmod +x start_worker.sh
    fi
    
    # Systemd service files
    if [[ "$OS" != "macos" ]]; then
        # Master service
        if [[ "$NODE_TYPE" == "master" ]]; then
            sudo tee /etc/systemd/system/schwabot-master.service > /dev/null << EOF
[Unit]
Description=Schwabot Master Node
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCHWABOT_DIR
Environment=PATH=$SCHWABOT_DIR/venv/bin
ExecStart=$SCHWABOT_DIR/venv/bin/python $SCHWABOT_DIR/distributed_system/master_node.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        fi
        
        # Worker service
        if [[ "$NODE_TYPE" == "worker" ]]; then
            sudo tee /etc/systemd/system/schwabot-worker.service > /dev/null << EOF
[Unit]
Description=Schwabot Worker Node
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCHWABOT_DIR
Environment=PATH=$SCHWABOT_DIR/venv/bin
ExecStart=$SCHWABOT_DIR/venv/bin/python $SCHWABOT_DIR/distributed_system/worker_node.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        fi
        
        # Reload systemd
        sudo systemctl daemon-reload
    fi
    
    success "Startup scripts created"
}

# Setup Discord integration
setup_discord() {
    log "Setting up Discord integration..."
    
    echo -e "${CYAN}Discord Bot Setup${NC}"
    echo "To enable Discord integration, you need to:"
    echo "1. Create a Discord application at https://discord.com/developers/applications"
    echo "2. Create a bot and get the token"
    echo "3. Invite the bot to your server"
    echo "4. Get the channel ID where you want the bot to send messages"
    echo ""
    
    read -p "Do you want to set up Discord integration now? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter Discord bot token: " DISCORD_TOKEN
        read -p "Enter Discord channel ID: " DISCORD_CHANNEL_ID
        
        # Update configuration
        if [[ "$NODE_TYPE" == "master" ]]; then
            sed -i "s/token: ''/token: '$DISCORD_TOKEN'/" config/master_config.yaml
            sed -i "s/channel_id: ''/channel_id: '$DISCORD_CHANNEL_ID'/" config/master_config.yaml
        fi
        
        # Update .env file
        sed -i "s/DISCORD_TOKEN=/DISCORD_TOKEN=$DISCORD_TOKEN/" .env
        sed -i "s/DISCORD_CHANNEL_ID=/DISCORD_CHANNEL_ID=$DISCORD_CHANNEL_ID/" .env
        
        success "Discord integration configured"
    else
        warn "Discord integration skipped. You can configure it later."
    fi
}

# Create shared data directory
setup_shared_data() {
    log "Setting up shared data directory..."
    
    cd "$SCHWABOT_DIR"
    mkdir -p shared_data/{tasks,results,logs,models}
    
    # Create initial data files
    cat > shared_data/system_info.json << EOF
{
    "node_type": "$NODE_TYPE",
    "hardware": {
        "cpu_cores": $CPU_CORES,
        "cpu_model": "$CPU_MODEL",
        "memory_gb": $MEMORY_GB,
        "gpu_available": $GPU_AVAILABLE,
        "gpu_name": "$GPU_NAME",
        "gpu_memory_gb": $GPU_MEMORY_GB
    },
    "installation_date": "$(date -Iseconds)",
    "version": "1.0.0"
}
EOF
    
    success "Shared data directory created"
}

# Create monitoring dashboard
setup_dashboard() {
    log "Setting up monitoring dashboard..."
    
    cd "$SCHWABOT_DIR"
    mkdir -p templates static/{css,js,img}
    
    # Create simple dashboard template
    cat > templates/dashboard.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schwabot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #333; color: white; padding: 20px; border-radius: 5px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .status-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; color: #007bff; }
        .label { color: #666; margin-bottom: 10px; }
        .online { color: #28a745; }
        .offline { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Schwabot Distributed Trading System</h1>
            <p>Real-time monitoring and control dashboard</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <div class="label">Master Status</div>
                <div class="metric" id="master-status">{{ state.master_status }}</div>
            </div>
            <div class="status-card">
                <div class="label">Connected Nodes</div>
                <div class="metric" id="connected-nodes">{{ state.connected_nodes }}</div>
            </div>
            <div class="status-card">
                <div class="label">Active Trades</div>
                <div class="metric" id="active-trades">{{ state.active_trades }}</div>
            </div>
            <div class="status-card">
                <div class="label">Total Profit</div>
                <div class="metric" id="total-profit">${{ "%.2f"|format(state.total_profit) }}</div>
            </div>
            <div class="status-card">
                <div class="label">System Health</div>
                <div class="metric" id="system-health">{{ "%.1f"|format(state.system_health) }}%</div>
            </div>
            <div class="status-card">
                <div class="label">Memory Usage</div>
                <div class="metric" id="memory-usage">{{ "%.1f"|format(state.memory_usage) }}%</div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 5 seconds
        setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('master-status').textContent = data.master_status;
                    document.getElementById('connected-nodes').textContent = data.connected_nodes;
                    document.getElementById('active-trades').textContent = data.active_trades;
                    document.getElementById('total-profit').textContent = '$' + data.total_profit.toFixed(2);
                    document.getElementById('system-health').textContent = data.system_health.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = data.memory_usage.toFixed(1) + '%';
                });
        }, 5000);
    </script>
</body>
</html>
EOF
    
    success "Dashboard created"
}

# Create quick start guide
create_quick_start() {
    log "Creating quick start guide..."
    
    cd "$SCHWABOT_DIR"
    
    cat > QUICK_START.md << EOF
# 🚀 Schwabot Distributed Trading System - Quick Start Guide

## System Overview
- **Node Type**: $NODE_TYPE
- **Hardware**: $CPU_CORES cores, ${MEMORY_GB}GB RAM
- **GPU**: $GPU_NAME ($GPU_MEMORY_GB GB)
- **Installation Date**: $(date)

## Quick Start

### 1. Start the System
\`\`\`bash
# For master node:
./start_master.sh

# For worker node:
./start_worker.sh
\`\`\`

### 2. Access Dashboard
Open your browser and go to: http://localhost:5000

### 3. Monitor Logs
\`\`\`bash
# Master node logs
tail -f master_node.log

# Worker node logs
tail -f worker_node.log
\`\`\`

### 4. System Service (Linux)
\`\`\`bash
# Enable and start service
sudo systemctl enable schwabot-$NODE_TYPE
sudo systemctl start schwabot-$NODE_TYPE

# Check status
sudo systemctl status schwabot-$NODE_TYPE
\`\`\`

## Configuration

### Master Node Setup
1. Edit \`config/master_config.yaml\` to configure:
   - Discord integration
   - Trading parameters
   - Node management

2. Set environment variables in \`.env\`:
   - DISCORD_TOKEN
   - DISCORD_CHANNEL_ID

### Worker Node Setup
1. Edit \`config/worker_config.yaml\` to set:
   - Master node IP address
   - Worker port
   - Task limits

2. Update \`.env\` with master node details:
   - MASTER_HOST
   - MASTER_PORT

## Adding More Nodes

### Master Node
1. Run this installer on the master machine
2. Configure Discord integration
3. Start the master node

### Worker Nodes
1. Run this installer on worker machines
2. Set MASTER_HOST to master node IP
3. Start worker nodes

## Discord Commands
- \`/status\` - Get system status
- \`/nodes\` - List connected nodes
- \`/trading\` - Get trading status
- \`/ai <query>\` - Consult AI with system context

## Troubleshooting

### Common Issues
1. **Port already in use**: Change port in config file
2. **Connection refused**: Check firewall settings
3. **GPU not detected**: Install NVIDIA drivers
4. **Discord bot not responding**: Check token and permissions

### Logs Location
- Master node: \`master_node.log\`
- Worker node: \`worker_node.log\`
- Install log: \`install.log\`

### Support
- Check logs for detailed error messages
- Verify network connectivity between nodes
- Ensure all dependencies are installed

## Next Steps
1. Configure trading parameters
2. Set up multiple worker nodes
3. Configure Discord integration
4. Start observation mode
5. Monitor performance
6. Enable live trading when ready

Happy trading! 🚀
EOF
    
    success "Quick start guide created"
}

# Final setup and instructions
final_setup() {
    log "Finalizing setup..."
    
    cd "$SCHWABOT_DIR"
    
    # Set proper permissions
    chmod -R 755 .
    chmod +x *.sh
    
    # Create symbolic links for easy access
    ln -sf "$SCHWABOT_DIR" ~/schwabot
    
    success "Setup completed successfully!"
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉 INSTALLATION COMPLETE!                 ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}System Information:${NC}"
    echo "  Node Type: $NODE_TYPE"
    echo "  Installation: $SCHWABOT_DIR"
    echo "  Hardware: $CPU_CORES cores, ${MEMORY_GB}GB RAM"
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        echo "  GPU: $GPU_NAME (${GPU_MEMORY_GB}GB)"
    fi
    echo ""
    
    echo -e "${CYAN}Quick Start Commands:${NC}"
    if [[ "$NODE_TYPE" == "master" ]]; then
        echo "  cd $SCHWABOT_DIR"
        echo "  ./start_master.sh"
        echo "  # Or use systemd service:"
        echo "  sudo systemctl enable schwabot-master"
        echo "  sudo systemctl start schwabot-master"
    else
        echo "  cd $SCHWABOT_DIR"
        echo "  # Edit config/worker_config.yaml to set master IP"
        echo "  ./start_worker.sh"
        echo "  # Or use systemd service:"
        echo "  sudo systemctl enable schwabot-worker"
        echo "  sudo systemctl start schwabot-worker"
    fi
    echo ""
    
    echo -e "${CYAN}Access Dashboard:${NC}"
    echo "  http://localhost:5000"
    echo ""
    
    echo -e "${CYAN}Documentation:${NC}"
    echo "  Quick Start: $SCHWABOT_DIR/QUICK_START.md"
    echo "  Install Log: $LOG_FILE"
    echo ""
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Configure Discord integration (optional)"
    echo "  2. Set up additional worker nodes"
    echo "  3. Configure trading parameters"
    echo "  4. Start observation mode"
    echo "  5. Monitor system performance"
    echo ""
    
    echo -e "${GREEN}🚀 Your Schwabot distributed trading system is ready!${NC}"
}

# Main installation function
main() {
    print_banner
    
    # Check prerequisites
    check_root
    detect_os
    
    # Installation steps
    install_system_deps
    check_python
    detect_hardware
    clone_repository
    setup_venv
    install_python_deps
    setup_config
    create_startup_scripts
    setup_discord
    setup_shared_data
    setup_dashboard
    create_quick_start
    final_setup
}

# Run main function
main "$@" 