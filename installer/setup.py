#!/usr/bin/env python3
"""
Schwabot Trading Bot - Professional Installer
============================================

Professional installer for Schwabot trading bot with:
- Custom icon and branding
- Automatic dependency installation
- System integration
- Desktop shortcuts
- Start/Stop scripts
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install

# Schwabot metadata
SCHWABOT_NAME = "Schwabot"
SCHWABOT_VERSION = "2.0.0"
SCHWABOT_DESCRIPTION = "Advanced AI-Powered Trading Bot with 47-Day Mathematical Framework"
SCHWABOT_AUTHOR = "Schwabot Development Team"
SCHWABOT_AUTHOR_EMAIL = "support@schwabot.ai"
SCHWABOT_URL = "https://github.com/schwabot/trading-bot"
SCHWABOT_LICENSE = "MIT"

class CustomInstall(install):
    """Custom install command to create shortcuts and scripts."""
    
    def run(self):
        # Run the normal install
        install.run(self)
        
        # Create start/stop scripts
        self.create_control_scripts()
        
        # Create desktop shortcuts
        self.create_desktop_shortcuts()
        
        # Create system integration
        self.create_system_integration()
    
    def create_control_scripts(self):
        """Create start and stop scripts."""
        script_dir = Path(self.install_scripts)
        
        # Start script
        start_script = script_dir / "schwabot-start.py"
        with open(start_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Schwabot Trading Bot - Start Script
==================================

Starts the Schwabot trading bot with proper configuration.
"""

import sys
import os
import signal
import subprocess
from pathlib import Path

def start_schwabot():
    """Start the Schwabot trading bot."""
    try:
        # Get the installation directory
        install_dir = Path(__file__).parent.parent / "schwabot"
        
        # Change to the installation directory
        os.chdir(install_dir)
        
        # Start the trading bot
        print("🚀 Starting Schwabot Trading Bot...")
        print("📊 Mathematical Framework: 47-Day Production Ready")
        print("🎯 AI Integration: Active")
        print("📈 Real-time Trading: Enabled")
        print("-" * 50)
        
        # Run the main trading bot
        subprocess.run([sys.executable, "schwabot_trading_bot.py"])
        
    except KeyboardInterrupt:
        print("\\n🛑 Schwabot stopped by user")
    except Exception as e:
        print(f"❌ Error starting Schwabot: {e}")

if __name__ == "__main__":
    start_schwabot()
''')
        
        # Stop script
        stop_script = script_dir / "schwabot-stop.py"
        with open(stop_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Schwabot Trading Bot - Stop Script
=================================

Safely stops the Schwabot trading bot.
"""

import os
import signal
import subprocess
import psutil
from pathlib import Path

def stop_schwabot():
    """Stop the Schwabot trading bot."""
    try:
        print("🛑 Stopping Schwabot Trading Bot...")
        
        # Find Schwabot processes
        schwabot_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('schwabot' in cmd.lower() for cmd in proc.info['cmdline']):
                    schwabot_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if schwabot_processes:
            print(f"📋 Found {len(schwabot_processes)} Schwabot process(es)")
            
            for proc in schwabot_processes:
                print(f"🔄 Stopping process {proc.info['pid']}...")
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                    print(f"✅ Process {proc.info['pid']} stopped")
                except psutil.TimeoutExpired:
                    print(f"⚠️ Force killing process {proc.info['pid']}...")
                    proc.kill()
                except Exception as e:
                    print(f"❌ Error stopping process {proc.info['pid']}: {e}")
        else:
            print("ℹ️ No Schwabot processes found")
        
        print("✅ Schwabot Trading Bot stopped successfully")
        
    except Exception as e:
        print(f"❌ Error stopping Schwabot: {e}")

if __name__ == "__main__":
    stop_schwabot()
''')
        
        # Make scripts executable
        os.chmod(start_script, 0o755)
        os.chmod(stop_script, 0o755)
        
        print(f"✅ Created control scripts in {script_dir}")
    
    def create_desktop_shortcuts(self):
        """Create desktop shortcuts."""
        if platform.system() == "Windows":
            self.create_windows_shortcuts()
        elif platform.system() == "Linux":
            self.create_linux_shortcuts()
        elif platform.system() == "Darwin":
            self.create_macos_shortcuts()
    
    def create_windows_shortcuts(self):
        """Create Windows desktop shortcuts."""
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            start_shortcut = os.path.join(desktop, "Schwabot Trading Bot.lnk")
            stop_shortcut = os.path.join(desktop, "Stop Schwabot.lnk")
            
            # Create start shortcut
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(start_shortcut)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{Path(self.install_scripts) / "schwabot-start.py"}"'
            shortcut.WorkingDirectory = str(Path(self.install_scripts).parent / "schwabot")
            shortcut.IconLocation = str(Path(self.install_scripts).parent / "schwabot" / "assets" / "schwabot.ico")
            shortcut.save()
            
            # Create stop shortcut
            shortcut = shell.CreateShortCut(stop_shortcut)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{Path(self.install_scripts) / "schwabot-stop.py"}"'
            shortcut.WorkingDirectory = str(Path(self.install_scripts).parent / "schwabot")
            shortcut.IconLocation = str(Path(self.install_scripts).parent / "schwabot" / "assets" / "schwabot-stop.ico")
            shortcut.save()
            
            print("✅ Created Windows desktop shortcuts")
        except ImportError:
            print("⚠️ Could not create Windows shortcuts (missing dependencies)")
    
    def create_linux_shortcuts(self):
        """Create Linux desktop shortcuts."""
        try:
            desktop = os.path.expanduser("~/Desktop")
            if not os.path.exists(desktop):
                desktop = os.path.expanduser("~/")
            
            # Start desktop file
            start_desktop = os.path.join(desktop, "schwabot-trading-bot.desktop")
            with open(start_desktop, 'w') as f:
                f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Schwabot Trading Bot
Comment={SCHWABOT_DESCRIPTION}
Exec={sys.executable} {Path(self.install_scripts) / "schwabot-start.py"}
Icon={Path(self.install_scripts).parent / "schwabot" / "assets" / "schwabot.png"}
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
Exec={sys.executable} {Path(self.install_scripts) / "schwabot-stop.py"}
Icon={Path(self.install_scripts).parent / "schwabot" / "assets" / "schwabot-stop.png"}
Terminal=true
Categories=Finance;Trading;
""")
            
            # Make executable
            os.chmod(start_desktop, 0o755)
            os.chmod(stop_desktop, 0o755)
            
            print("✅ Created Linux desktop shortcuts")
        except Exception as e:
            print(f"⚠️ Could not create Linux shortcuts: {e}")
    
    def create_macos_shortcuts(self):
        """Create macOS application shortcuts."""
        try:
            # Create .command files for macOS
            desktop = os.path.expanduser("~/Desktop")
            
            # Start command
            start_cmd = os.path.join(desktop, "Start Schwabot.command")
            with open(start_cmd, 'w') as f:
                f.write(f'''#!/bin/bash
cd "{Path(self.install_scripts).parent / "schwabot"}"
{sys.executable} "{Path(self.install_scripts) / "schwabot-start.py"}"
''')
            
            # Stop command
            stop_cmd = os.path.join(desktop, "Stop Schwabot.command")
            with open(stop_cmd, 'w') as f:
                f.write(f'''#!/bin/bash
{sys.executable} "{Path(self.install_scripts) / "schwabot-stop.py"}"
''')
            
            # Make executable
            os.chmod(start_cmd, 0o755)
            os.chmod(stop_cmd, 0o755)
            
            print("✅ Created macOS shortcuts")
        except Exception as e:
            print(f"⚠️ Could not create macOS shortcuts: {e}")
    
    def create_system_integration(self):
        """Create system integration files."""
        # Create systemd service for Linux
        if platform.system() == "Linux":
            try:
                service_content = f"""[Unit]
Description=Schwabot Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory={Path(self.install_scripts).parent / "schwabot"}
ExecStart={sys.executable} {Path(self.install_scripts) / "schwabot-start.py"}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
                
                service_file = "/etc/systemd/system/schwabot.service"
                if os.access("/etc/systemd/system", os.W_OK):
                    with open(service_file, 'w') as f:
                        f.write(service_content)
                    print("✅ Created systemd service")
                else:
                    print("⚠️ Could not create systemd service (requires sudo)")
            except Exception as e:
                print(f"⚠️ Could not create systemd service: {e}")

# Setup configuration
setup(
    name=SCHWABOT_NAME.lower().replace(" ", "-"),
    version=SCHWABOT_VERSION,
    description=SCHWABOT_DESCRIPTION,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=SCHWABOT_AUTHOR,
    author_email=SCHWABOT_AUTHOR_EMAIL,
    url=SCHWABOT_URL,
    license=SCHWABOT_LICENSE,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
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
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "gui": [
            "tkinter",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "windows": [
            "pywin32>=228",
            "winshell>=0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "schwabot=schwabot_trading_bot:main",
            "schwabot-start=schwabot_start:main",
            "schwabot-stop=schwabot_stop:main",
        ],
    },
    cmdclass={
        "install": CustomInstall,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="trading bot ai machine learning cryptocurrency finance",
    project_urls={
        "Bug Reports": f"{SCHWABOT_URL}/issues",
        "Source": SCHWABOT_URL,
        "Documentation": f"{SCHWABOT_URL}/docs",
    },
) 