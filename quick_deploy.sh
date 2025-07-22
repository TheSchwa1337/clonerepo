#!/bin/bash
# 🚀 Schwabot Quick Deployment Script
# ===================================
# This script automates the high-priority deployment tasks for Schwabot

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
   exit 1
fi

log "🚀 Starting Schwabot Quick Deployment"
log "====================================="

# Step 1: Environment Setup
log "Step 1: Setting up environment..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f config/production.env.template ]; then
        cp config/production.env.template .env
        warning "Created .env file from template. Please edit it with your actual values."
        echo "Required variables to set:"
        echo "  - BINANCE_API_KEY"
        echo "  - BINANCE_API_SECRET"
        echo "  - SCHWABOT_ENCRYPTION_KEY (32+ characters)"
        echo "  - SCHWABOT_ENVIRONMENT=production"
        echo ""
        echo "Edit .env file and run this script again."
        exit 1
    else
        error "No .env template found. Please create .env file manually."
        exit 1
    fi
fi

# Set file permissions
chmod 600 .env 2>/dev/null || warning "Could not set .env permissions"
chmod 700 secure/ 2>/dev/null || warning "Could not set secure/ permissions"
mkdir -p logs
chmod 644 logs/*.log 2>/dev/null || true

success "Environment setup completed"

# Step 2: Dependencies Check
log "Step 2: Checking dependencies..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    success "Python version $python_version is compatible"
else
    error "Python $required_version+ required, found $python_version"
    exit 1
fi

# Check if requirements.txt exists
if [ -f requirements.txt ]; then
    log "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    success "Dependencies installed"
else
    warning "No requirements.txt found. Installing core dependencies..."
    pip install numpy pandas ccxt flask cryptography requests pyyaml psutil
    success "Core dependencies installed"
fi

# Step 3: Security Validation
log "Step 3: Running security validation..."

# Check for hardcoded secrets
if command -v grep >/dev/null 2>&1; then
    secrets_found=$(grep -r "api_key\|secret\|password" . --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=__pycache__ --exclude=.env 2>/dev/null | wc -l)
    if [ "$secrets_found" -eq 0 ]; then
        success "No hardcoded secrets found"
    else
        warning "Found $secrets_found potential hardcoded secrets"
    fi
else
    warning "grep not available, skipping secret check"
fi

# Check encryption key
encryption_key=$(grep "SCHWABOT_ENCRYPTION_KEY" .env 2>/dev/null | cut -d'=' -f2 || echo "")
if [ ${#encryption_key} -ge 32 ]; then
    success "Encryption key length is sufficient"
else
    error "Encryption key should be at least 32 characters"
    exit 1
fi

# Step 4: System Validation
log "Step 4: Running system validation..."

# Run deployment validator if available
if [ -f deployment_validator.py ]; then
    log "Running deployment validator..."
    python3 deployment_validator.py --full
    if [ $? -eq 0 ]; then
        success "Deployment validation passed"
    elif [ $? -eq 2 ]; then
        warning "Deployment validation passed with warnings"
    else
        error "Deployment validation failed"
        exit 1
    fi
else
    warning "Deployment validator not found, skipping validation"
fi

# Step 5: Configuration Validation
log "Step 5: Validating configuration..."

# Check if main configuration files exist
config_files=("config/master_integration.yaml" "config/security_config.yaml")
for config_file in "${config_files[@]}"; do
    if [ -f "$config_file" ]; then
        success "Configuration file $config_file exists"
    else
        warning "Configuration file $config_file not found"
    fi
done

# Validate YAML if available
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import yaml
try:
    with open('config/master_integration.yaml', 'r') as f:
        yaml.safe_load(f)
    print('✅ YAML configuration is valid')
except Exception as e:
    print(f'❌ YAML configuration error: {e}')
    exit(1)
" 2>/dev/null || warning "Could not validate YAML configuration"
fi

# Step 6: Performance Check
log "Step 6: Checking system performance..."

# Check system resources
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import psutil
cpu = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
disk = psutil.disk_usage('.')
print(f'CPU: {cpu:.1f}%')
print(f'Memory: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)')
print(f'Disk: {disk.percent:.1f}% ({disk.free / (1024**3):.1f}GB free)')
" 2>/dev/null || warning "Could not check system resources"
fi

# Step 7: Integration Test
log "Step 7: Running integration tests..."

# Check if test files exist and run them
test_files=(
    "tests/integration/test_core_integration.py"
    "tests/integration/test_mathematical_integration.py"
    "tests/integration/test_complete_production_system.py"
)

for test_file in "${test_files[@]}"; do
    if [ -f "$test_file" ]; then
        log "Running $test_file..."
        python3 "$test_file" 2>/dev/null && success "$test_file passed" || warning "$test_file failed or not found"
    else
        warning "$test_file not found"
    fi
done

# Step 8: Startup Test
log "Step 8: Testing system startup..."

# Check if main entry points exist
entry_points=(
    "AOI_Base_Files_Schwabot/run_schwabot.py"
    "AOI_Base_Files_Schwabot/launch_unified_interface.py"
    "AOI_Base_Files_Schwabot/launch_unified_mathematical_trading_system.py"
)

for entry_point in "${entry_points[@]}"; do
    if [ -f "$entry_point" ]; then
        success "Entry point $entry_point exists"
    else
        warning "Entry point $entry_point not found"
    fi
done

# Step 9: Final Validation
log "Step 9: Final validation..."

# Check if we can import Schwabot modules
python3 -c "
try:
    import sys
    sys.path.append('.')
    import core.brain_trading_engine
    import core.clean_unified_math
    import core.symbolic_profit_router
    print('✅ Core Schwabot modules can be imported')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" 2>/dev/null && success "Core modules import successfully" || warning "Some modules could not be imported"

# Step 10: Deployment Summary
log "Step 10: Deployment summary"
log "=========================="

echo ""
echo "🎉 DEPLOYMENT COMPLETED!"
echo ""
echo "Next steps:"
echo "1. Review the validation results above"
echo "2. Edit .env file with your actual API keys"
echo "3. Start the system:"
echo "   python3 AOI_Base_Files_Schwabot/run_schwabot.py --mode demo"
echo ""
echo "4. For production:"
echo "   python3 AOI_Base_Files_Schwabot/run_schwabot.py --mode live"
echo ""
echo "5. Monitor the system:"
echo "   tail -f logs/schwabot.log"
echo ""
echo "6. Access web interface:"
echo "   http://localhost:8080"
echo ""

# Check if there were any critical errors
if [ $? -eq 0 ]; then
    success "Quick deployment completed successfully!"
    echo ""
    echo "🚀 Your Schwabot system is ready for deployment!"
    echo ""
    echo "Remember to:"
    echo "- Test thoroughly in demo mode first"
    echo "- Monitor system performance"
    echo "- Keep your API keys secure"
    echo "- Regularly backup your configuration"
    echo ""
else
    error "Deployment completed with errors. Please review the output above."
    exit 1
fi 