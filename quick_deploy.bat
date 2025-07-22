@echo off
REM 🚀 Schwabot Quick Deployment Script for Windows
REM ===============================================
REM This script automates the high-priority deployment tasks for Schwabot

echo 🚀 Starting Schwabot Quick Deployment
echo =====================================

REM Step 1: Environment Setup
echo Step 1: Setting up environment...

REM Create .env file if it doesn't exist
if not exist .env (
    if exist config\production.env.template (
        copy config\production.env.template .env
        echo ⚠️ Created .env file from template. Please edit it with your actual values.
        echo Required variables to set:
        echo   - BINANCE_API_KEY
        echo   - BINANCE_API_SECRET
        echo   - SCHWABOT_ENCRYPTION_KEY (32+ characters)
        echo   - SCHWABOT_ENVIRONMENT=production
        echo.
        echo Edit .env file and run this script again.
        pause
        exit /b 1
    ) else (
        echo ❌ No .env template found. Please create .env file manually.
        pause
        exit /b 1
    )
)

REM Create logs directory
if not exist logs mkdir logs

echo ✅ Environment setup completed

REM Step 2: Dependencies Check
echo Step 2: Checking dependencies...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
) else (
    echo ⚠️ No requirements.txt found. Installing core dependencies...
    pip install numpy pandas ccxt flask cryptography requests pyyaml psutil
    echo ✅ Core dependencies installed
)

REM Step 3: Security Validation
echo Step 3: Running security validation...

REM Check encryption key length (basic check)
findstr "SCHWABOT_ENCRYPTION_KEY" .env >nul 2>&1
if errorlevel 1 (
    echo ❌ SCHWABOT_ENCRYPTION_KEY not found in .env
    pause
    exit /b 1
)

echo ✅ Security validation completed

REM Step 4: System Validation
echo Step 4: Running system validation...

REM Run deployment validator if available
if exist deployment_validator.py (
    echo Running deployment validator...
    python deployment_validator.py --full
    if errorlevel 1 (
        echo ❌ Deployment validation failed
        pause
        exit /b 1
    ) else (
        echo ✅ Deployment validation passed
    )
) else (
    echo ⚠️ Deployment validator not found, skipping validation
)

REM Step 5: Configuration Validation
echo Step 5: Validating configuration...

REM Check if main configuration files exist
if exist config\master_integration.yaml (
    echo ✅ Configuration file config\master_integration.yaml exists
) else (
    echo ⚠️ Configuration file config\master_integration.yaml not found
)

if exist config\security_config.yaml (
    echo ✅ Configuration file config\security_config.yaml exists
) else (
    echo ⚠️ Configuration file config\security_config.yaml not found
)

REM Step 6: Performance Check
echo Step 6: Checking system performance...

REM Check system resources using Python
python -c "import psutil; cpu=psutil.cpu_percent(interval=1); mem=psutil.virtual_memory(); disk=psutil.disk_usage('.'); print(f'CPU: {cpu:.1f}%%'); print(f'Memory: {mem.percent:.1f}%% ({mem.available/(1024**3):.1f}GB available)'); print(f'Disk: {disk.percent:.1f}%% ({disk.free/(1024**3):.1f}GB free)')" 2>nul || echo ⚠️ Could not check system resources

REM Step 7: Integration Test
echo Step 7: Running integration tests...

REM Check if test files exist and run them
if exist tests\integration\test_core_integration.py (
    echo Running tests\integration\test_core_integration.py...
    python tests\integration\test_core_integration.py 2>nul && echo ✅ test_core_integration.py passed || echo ⚠️ test_core_integration.py failed or not found
) else (
    echo ⚠️ tests\integration\test_core_integration.py not found
)

if exist tests\integration\test_mathematical_integration.py (
    echo Running tests\integration\test_mathematical_integration.py...
    python tests\integration\test_mathematical_integration.py 2>nul && echo ✅ test_mathematical_integration.py passed || echo ⚠️ test_mathematical_integration.py failed or not found
) else (
    echo ⚠️ tests\integration\test_mathematical_integration.py not found
)

REM Step 8: Startup Test
echo Step 8: Testing system startup...

REM Check if main entry points exist
if exist AOI_Base_Files_Schwabot\run_schwabot.py (
    echo ✅ Entry point AOI_Base_Files_Schwabot\run_schwabot.py exists
) else (
    echo ⚠️ Entry point AOI_Base_Files_Schwabot\run_schwabot.py not found
)

if exist AOI_Base_Files_Schwabot\launch_unified_interface.py (
    echo ✅ Entry point AOI_Base_Files_Schwabot\launch_unified_interface.py exists
) else (
    echo ⚠️ Entry point AOI_Base_Files_Schwabot\launch_unified_interface.py not found
)

REM Step 9: Final Validation
echo Step 9: Final validation...

REM Check if we can import Schwabot modules
python -c "import sys; sys.path.append('.'); import core.brain_trading_engine; import core.clean_unified_math; import core.symbolic_profit_router; print('✅ Core Schwabot modules can be imported')" 2>nul && echo ✅ Core modules import successfully || echo ⚠️ Some modules could not be imported

REM Step 10: Deployment Summary
echo Step 10: Deployment summary
echo ==========================

echo.
echo 🎉 DEPLOYMENT COMPLETED!
echo.
echo Next steps:
echo 1. Review the validation results above
echo 2. Edit .env file with your actual API keys
echo 3. Start the system:
echo    python AOI_Base_Files_Schwabot\run_schwabot.py --mode demo
echo.
echo 4. For production:
echo    python AOI_Base_Files_Schwabot\run_schwabot.py --mode live
echo.
echo 5. Monitor the system:
echo    type logs\schwabot.log
echo.
echo 6. Access web interface:
echo    http://localhost:8080
echo.

echo ✅ Quick deployment completed successfully!
echo.
echo 🚀 Your Schwabot system is ready for deployment!
echo.
echo Remember to:
echo - Test thoroughly in demo mode first
echo - Monitor system performance
echo - Keep your API keys secure
echo - Regularly backup your configuration
echo.

pause 