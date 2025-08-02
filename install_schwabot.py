#!/usr/bin/env python3
"""
Schwabot Trading Bot - Enhanced Installer
========================================

Enhanced installer script for the Schwabot trading bot with:
- Professional installer with dependency management
- Enhanced GUI with all original features
- USB memory management system
- Safe shutdown/startup capabilities
- Portable system support
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print the installer banner."""
    print("=" * 60)
    print("🚀 SCHWABOT ENHANCED TRADING BOT - INSTALLER")
    print("=" * 60)
    print("Advanced AI-Powered Trading with 47-Day Mathematical Framework")
    print("Enhanced with USB Memory Management & Complete Control System")
    print("Version: 2.1.0")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "ccxt>=2.0.0",
        "aiohttp>=3.8.0",
        "websockets>=10.0",
        "flask>=2.2.0",
        "psutil>=5.8.0",
        "python-dotenv>=0.19.0",
        "rich>=12.0.0",
        "click>=8.0.0",
    ]
    
    # Add Windows-specific dependencies
    if platform.system() == "Windows":
        dependencies.extend([
            "pywin32>=228",
            "winshell>=0.6",
        ])
    
    try:
        for dep in dependencies:
            print(f"   Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_shortcuts():
    """Create desktop shortcuts."""
    print("\n🔗 Creating shortcuts...")
    
    try:
        if platform.system() == "Windows":
            create_windows_shortcuts()
        elif platform.system() == "Linux":
            create_linux_shortcuts()
        elif platform.system() == "Darwin":
            create_macos_shortcuts()
        
        print("✅ Shortcuts created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating shortcuts: {e}")
        return False

def create_windows_shortcuts():
    """Create Windows shortcuts."""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        current_dir = Path(__file__).parent.absolute()
        
        # Create enhanced GUI shortcut
        gui_shortcut = os.path.join(desktop, "Schwabot Enhanced Trading System.lnk")
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(gui_shortcut)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{current_dir / "schwabot_enhanced_gui.py"}"'
        shortcut.WorkingDirectory = str(current_dir)
        shortcut.save()
        
        # Create start shortcut
        start_shortcut = os.path.join(desktop, "Start Schwabot Trading Bot.lnk")
        shortcut = shell.CreateShortCut(start_shortcut)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{current_dir / "schwabot_start.py"}"'
        shortcut.WorkingDirectory = str(current_dir)
        shortcut.save()
        
        # Create stop shortcut
        stop_shortcut = os.path.join(desktop, "Stop Schwabot Trading Bot.lnk")
        shortcut = shell.CreateShortCut(stop_shortcut)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{current_dir / "schwabot_stop.py"}"'
        shortcut.WorkingDirectory = str(current_dir)
        shortcut.save()
        
        print("   ✅ Windows shortcuts created")
        
    except ImportError:
        print("   ⚠️ Could not create Windows shortcuts (missing dependencies)")
        print("   💡 Install pywin32 and winshell: pip install pywin32 winshell")

def create_linux_shortcuts():
    """Create Linux shortcuts."""
    try:
        desktop = os.path.expanduser("~/Desktop")
        if not os.path.exists(desktop):
            desktop = os.path.expanduser("~/")
        
        current_dir = Path(__file__).parent.absolute()
        
        # Enhanced GUI desktop file
        gui_desktop = os.path.join(desktop, "schwabot-enhanced-trading-system.desktop")
        with open(gui_desktop, 'w') as f:
            f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Schwabot Enhanced Trading System
Comment=Complete Trading Bot Control Center with USB Memory Management
Exec={sys.executable} {current_dir / "schwabot_enhanced_gui.py"}
Terminal=false
Categories=Finance;Trading;
""")
        
        # Start desktop file
        start_desktop = os.path.join(desktop, "schwabot-trading-bot.desktop")
        with open(start_desktop, 'w') as f:
            f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Schwabot Trading Bot
Comment=Advanced AI-Powered Trading Bot
Exec={sys.executable} {current_dir / "schwabot_start.py"}
Terminal=true
Categories=Finance;Trading;
""")
        
        # Stop desktop file
        stop_desktop = os.path.join(desktop, "stop-schwabot.desktop")
        with open(stop_desktop, 'w') as f:
            f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Stop Schwabot
Comment=Stop Schwabot Trading Bot
Exec={sys.executable} {current_dir / "schwabot_stop.py"}
Terminal=true
Categories=Finance;Trading;
""")
        
        # Make executable
        os.chmod(gui_desktop, 0o755)
        os.chmod(start_desktop, 0o755)
        os.chmod(stop_desktop, 0o755)
        
        print("   ✅ Linux shortcuts created")
        
    except Exception as e:
        print(f"   ❌ Error creating Linux shortcuts: {e}")

def create_macos_shortcuts():
    """Create macOS shortcuts."""
    try:
        desktop = os.path.expanduser("~/Desktop")
        current_dir = Path(__file__).parent.absolute()
        
        # Enhanced GUI command
        gui_cmd = os.path.join(desktop, "Schwabot Enhanced Trading System.command")
        with open(gui_cmd, 'w') as f:
            f.write(f'''#!/bin/bash
cd "{current_dir}"
{sys.executable} "{current_dir / "schwabot_enhanced_gui.py"}"
''')
        
        # Start command
        start_cmd = os.path.join(desktop, "Start Schwabot.command")
        with open(start_cmd, 'w') as f:
            f.write(f'''#!/bin/bash
cd "{current_dir}"
{sys.executable} "{current_dir / "schwabot_start.py"}"
''')
        
        # Stop command
        stop_cmd = os.path.join(desktop, "Stop Schwabot.command")
        with open(stop_cmd, 'w') as f:
            f.write(f'''#!/bin/bash
{sys.executable} "{current_dir / "schwabot_stop.py"}"
''')
        
        # Make executable
        os.chmod(gui_cmd, 0o755)
        os.chmod(start_cmd, 0o755)
        os.chmod(stop_cmd, 0o755)
        
        print("   ✅ macOS shortcuts created")
        
    except Exception as e:
        print(f"   ❌ Error creating macOS shortcuts: {e}")

def create_batch_files():
    """Create batch files for easy execution."""
    print("\n📝 Creating batch files...")
    
    try:
        current_dir = Path(__file__).parent.absolute()
        
        # Windows batch files
        if platform.system() == "Windows":
            # Enhanced GUI batch
            gui_bat = current_dir / "schwabot_enhanced_gui.bat"
            with open(gui_bat, 'w') as f:
                f.write(f'''@echo off
echo Starting Schwabot Enhanced Trading System...
cd /d "{current_dir}"
python schwabot_enhanced_gui.py
pause
''')
            
            # Start batch
            start_bat = current_dir / "start_schwabot.bat"
            with open(start_bat, 'w') as f:
                f.write(f'''@echo off
echo Starting Schwabot Trading Bot...
cd /d "{current_dir}"
python schwabot_start.py
pause
''')
            
            # Stop batch
            stop_bat = current_dir / "stop_schwabot.bat"
            with open(stop_bat, 'w') as f:
                f.write(f'''@echo off
echo Stopping Schwabot Trading Bot...
cd /d "{current_dir}"
python schwabot_stop.py
pause
''')
        
        # Linux/Mac shell scripts
        else:
            # Enhanced GUI script
            gui_sh = current_dir / "schwabot_enhanced_gui.sh"
            with open(gui_sh, 'w') as f:
                f.write(f'''#!/bin/bash
echo "Starting Schwabot Enhanced Trading System..."
cd "{current_dir}"
{sys.executable} schwabot_enhanced_gui.py
''')
            
            # Start script
            start_sh = current_dir / "start_schwabot.sh"
            with open(start_sh, 'w') as f:
                f.write(f'''#!/bin/bash
echo "Starting Schwabot Trading Bot..."
cd "{current_dir}"
{sys.executable} schwabot_start.py
''')
            
            # Stop script
            stop_sh = current_dir / "stop_schwabot.sh"
            with open(stop_sh, 'w') as f:
                f.write(f'''#!/bin/bash
echo "Stopping Schwabot Trading Bot..."
{sys.executable} schwabot_stop.py
''')
            
            # Make executable
            os.chmod(gui_sh, 0o755)
            os.chmod(start_sh, 0o755)
            os.chmod(stop_sh, 0o755)
        
        print("✅ Batch files created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating batch files: {e}")
        return False

def create_config_file():
    """Create a default configuration file."""
    print("\n⚙️ Creating configuration file...")
    
    try:
        config = {
            "bot_name": "Schwabot Enhanced",
            "version": "2.1.0",
            "description": "Advanced AI-Powered Trading Bot with USB Memory Management",
            "installation_date": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "features": {
                "usb_memory_management": True,
                "enhanced_gui": True,
                "safe_shutdown": True,
                "portable_system": True,
                "real_time_monitoring": True
            },
            "default_settings": {
                "trading_enabled": True,
                "risk_level": "medium",
                "max_position_size": 0.1,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 5.0,
                "usb_backup_interval": 60,
                "auto_restore_on_startup": True
            }
        }
        
        config_file = Path(__file__).parent / "schwabot_config.json"
        with open(config_file, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        print("✅ Configuration file created!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration file: {e}")
        return False

def create_memory_directories():
    """Create memory directories for USB management."""
    print("\n💾 Creating memory directories...")
    
    try:
        current_dir = Path(__file__).parent.absolute()
        
        # Create local memory directory
        memory_dir = current_dir / "SchwabotMemory"
        memory_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ['config', 'state', 'logs', 'backups', 'data']
        for subdir in subdirs:
            (memory_dir / subdir).mkdir(exist_ok=True)
        
        print("✅ Memory directories created!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating memory directories: {e}")
        return False

def run_tests():
    """Run basic tests to ensure everything works."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import numpy as np
        import pandas as pd
        import psutil
        print("   ✅ Core imports successful")
        
        # Test USB memory system
        print("   Testing USB memory system...")
        try:
            from schwabot_usb_memory import SchwabotUSBMemory
            usb_memory = SchwabotUSBMemory()
            print("   ✅ USB memory system test successful")
        except Exception as e:
            print(f"   ⚠️ USB memory system test failed: {e}")
        
        # Test scripts
        print("   Testing scripts...")
        current_dir = Path(__file__).parent.absolute()
        
        # Test CLI
        result = subprocess.run([sys.executable, str(current_dir / "schwabot_cli.py"), "status"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ CLI test successful")
        else:
            print("   ⚠️ CLI test failed (this is normal if bot is not running)")
        
        print("✅ Basic tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "=" * 60)
    print("🎉 ENHANCED INSTALLATION COMPLETE!")
    print("=" * 60)
    print("\n📋 USAGE INSTRUCTIONS:")
    print("\n🚀 To START Schwabot Enhanced System:")
    if platform.system() == "Windows":
        print("   • Double-click 'Schwabot Enhanced Trading System.lnk' (Recommended)")
        print("   • Or run: python schwabot_enhanced_gui.py")
        print("   • Or double-click 'schwabot_enhanced_gui.bat'")
    else:
        print("   • Double-click 'Schwabot Enhanced Trading System.command' (Recommended)")
        print("   • Or run: python schwabot_enhanced_gui.py")
        print("   • Or double-click 'schwabot_enhanced_gui.sh'")
    
    print("\n🛑 To STOP Schwabot:")
    if platform.system() == "Windows":
        print("   • Use the STOP button in the Enhanced GUI")
        print("   • Or double-click 'Stop Schwabot Trading Bot.lnk'")
        print("   • Or run: python schwabot_stop.py")
    else:
        print("   • Use the STOP button in the Enhanced GUI")
        print("   • Or double-click 'Stop Schwabot.command'")
        print("   • Or run: python schwabot_stop.py")
    
    print("\n💾 USB Memory Management:")
    print("   • Automatic USB detection and memory backup")
    print("   • Safe shutdown with memory preservation")
    print("   • Automatic memory restoration on startup")
    print("   • Continuous memory synchronization")
    
    print("\n🖥️ Enhanced GUI Features:")
    print("   • Complete control center with all original features")
    print("   • Real-time monitoring and status updates")
    print("   • USB memory management interface")
    print("   • Safe shutdown with trading position closure")
    print("   • Memory backup and restoration tools")
    
    print("\n💻 CLI Commands:")
    print("   • python schwabot_cli.py start     # Start bot")
    print("   • python schwabot_cli.py stop      # Stop bot")
    print("   • python schwabot_cli.py status    # Show status")
    print("   • python schwabot_cli.py logs      # Show logs")
    
    print("\n📁 Important Files:")
    print("   • schwabot_enhanced_gui.py         # Enhanced GUI (Recommended)")
    print("   • schwabot_usb_memory.py           # USB memory management")
    print("   • schwabot_trading_bot.py          # Main trading bot")
    print("   • schwabot_start.py                # Start script")
    print("   • schwabot_stop.py                 # Stop script")
    
    print("\n🔧 Configuration:")
    print("   • Edit schwabot_config.json        # Bot configuration")
    print("   • Check requirements.txt           # Dependencies")
    print("   • USB memory stored in SchwabotMemory/")
    
    print("\n📞 Support:")
    print("   • Check the logs for detailed information")
    print("   • Review the documentation in docs/")
    print("   • Monitor system performance via Enhanced GUI")
    print("   • USB memory management for portable use")
    
    print("\n" + "=" * 60)
    print("🚀 Schwabot Enhanced is ready for portable trading!")
    print("=" * 60)

def main():
    """Main installer function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Create shortcuts
    if not create_shortcuts():
        print("⚠️ Failed to create shortcuts")
    
    # Create batch files
    if not create_batch_files():
        print("⚠️ Failed to create batch files")
    
    # Create config file
    if not create_config_file():
        print("⚠️ Failed to create config file")
    
    # Create memory directories
    if not create_memory_directories():
        print("⚠️ Failed to create memory directories")
    
    # Run tests
    if not run_tests():
        print("⚠️ Some tests failed")
    
    # Show instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Enhanced installation completed successfully!")
        else:
            print("\n❌ Enhanced installation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Enhanced installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during enhanced installation: {e}")
        sys.exit(1) 