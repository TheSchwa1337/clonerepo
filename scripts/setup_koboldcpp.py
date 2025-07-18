#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Setup Script for Schwabot
===================================

This script helps configure KoboldCPP properly for the Schwabot trading system.
It will:
1. Check if KoboldCPP is installed
2. Download KoboldCPP if needed
3. Configure the system for optimal performance
4. Set up model paths and settings
"""

import os
import sys
import json
import logging
import subprocess
import platform
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KoboldCPPSetup:
    """KoboldCPP setup and configuration manager."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.project_root = Path(__file__).parent
        self.kobold_dir = self.project_root / "koboldcpp"
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        
        # Create directories
        self.kobold_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # KoboldCPP download URLs
        self.download_urls = {
            "windows": {
                "x86_64": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-windows-x64.exe",
                "x86": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-windows-x86.exe"
            },
            "linux": {
                "x86_64": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-linux-x64",
                "aarch64": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-linux-aarch64"
            },
            "darwin": {
                "x86_64": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-macos-x64",
                "arm64": "https://github.com/LostRuins/koboldcpp/releases/download/v1.92.1/koboldcpp-macos-arm64"
            }
        }
    
    def check_koboldcpp_installation(self) -> bool:
        """Check if KoboldCPP is properly installed."""
        try:
            # Check for executable
            if self.system == "windows":
                kobold_exe = self.kobold_dir / "koboldcpp.exe"
            else:
                kobold_exe = self.kobold_dir / "koboldcpp"
            
            if not kobold_exe.exists():
                logger.warning("⚠️ KoboldCPP executable not found")
                return False
            
            # Check if executable
            if not os.access(kobold_exe, os.X_OK):
                logger.warning("⚠️ KoboldCPP executable not executable")
                return False
            
            # Test version
            try:
                result = subprocess.run([str(kobold_exe), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ KoboldCPP found: {result.stdout.strip()}")
                    return True
                else:
                    logger.warning("⚠️ KoboldCPP version check failed")
                    return False
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ KoboldCPP version check timed out")
                return False
            except Exception as e:
                logger.warning(f"⚠️ KoboldCPP version check error: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking KoboldCPP installation: {e}")
            return False
    
    def download_koboldcpp(self) -> bool:
        """Download KoboldCPP for the current platform."""
        try:
            logger.info("📥 Downloading KoboldCPP...")
            
            # Determine download URL
            if self.system not in self.download_urls:
                logger.error(f"❌ Unsupported system: {self.system}")
                return False
            
            if self.architecture not in self.download_urls[self.system]:
                logger.error(f"❌ Unsupported architecture: {self.architecture}")
                return False
            
            url = self.download_urls[self.system][self.architecture]
            filename = url.split("/")[-1]
            
            # Download file
            logger.info(f"📥 Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save file
            if self.system == "windows":
                output_path = self.kobold_dir / "koboldcpp.exe"
            else:
                output_path = self.kobold_dir / "koboldcpp"
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Make executable on Unix systems
            if self.system != "windows":
                os.chmod(output_path, 0o755)
            
            logger.info(f"✅ KoboldCPP downloaded to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download KoboldCPP: {e}")
            return False
    
    def create_kobold_config(self) -> Dict[str, Any]:
        """Create KoboldCPP configuration for Schwabot."""
        config = {
            "kobold_integration": {
                "enabled": True,
                "kobold_path": str(self.kobold_dir / ("koboldcpp.exe" if self.system == "windows" else "koboldcpp")),
                "model_path": "",
                "port": 5001,
                "host": "localhost",
                "auto_start": True,
                "auto_load_model": False,
                "enable_visual_takeover": True,
                "max_connections": 10,
                "threads": self._get_optimal_threads(),
                "context_size": 2048,
                "batch_size": 512,
                "gpu_layers": 0,
                "enable_vision": False,
                "enable_embeddings": False
            },
            "model_settings": {
                "recommended_models": [
                    "llama-2-7b-chat.gguf",
                    "mistral-7b-instruct-v0.2.gguf",
                    "phi-2.gguf",
                    "qwen2-7b-instruct.gguf"
                ],
                "model_download_urls": {
                    "llama-2-7b-chat.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
                    "mistral-7b-instruct-v0.2.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    "phi-2.gguf": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
                    "qwen2-7b-instruct.gguf": "https://huggingface.co/TheBloke/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct.Q4_K_M.gguf"
                }
            },
            "system_optimization": {
                "platform": self.system,
                "architecture": self.architecture,
                "ram_gb": self._get_system_ram(),
                "cpu_cores": os.cpu_count(),
                "optimization_mode": self._get_optimization_mode()
            }
        }
        
        return config
    
    def _get_optimal_threads(self) -> int:
        """Get optimal number of threads for KoboldCPP."""
        cpu_count = os.cpu_count() or 4
        if cpu_count <= 4:
            return max(1, cpu_count - 1)
        else:
            return max(2, cpu_count - 2)
    
    def _get_system_ram(self) -> float:
        """Get system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Default assumption
    
    def _get_optimization_mode(self) -> str:
        """Get optimization mode based on system capabilities."""
        ram_gb = self._get_system_ram()
        cpu_count = os.cpu_count() or 4
        
        if ram_gb >= 16 and cpu_count >= 8:
            return "performance"
        elif ram_gb >= 8 and cpu_count >= 4:
            return "balanced"
        else:
            return "conservative"
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            config_path = self.config_dir / "koboldcpp_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def download_model(self, model_name: str) -> bool:
        """Download a specific model."""
        try:
            config = self.create_kobold_config()
            model_urls = config["model_settings"]["model_download_urls"]
            
            if model_name not in model_urls:
                logger.error(f"❌ Model not found: {model_name}")
                return False
            
            url = model_urls[model_name]
            output_path = self.models_dir / model_name
            
            if output_path.exists():
                logger.info(f"✅ Model already exists: {output_path}")
                return True
            
            logger.info(f"📥 Downloading model: {model_name}")
            logger.info(f"📥 URL: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Download progress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            logger.info(f"✅ Model downloaded: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_name}: {e}")
            return False
    
    def test_koboldcpp(self) -> bool:
        """Test KoboldCPP installation."""
        try:
            logger.info("🧪 Testing KoboldCPP installation...")
            
            # Get kobold path
            if self.system == "windows":
                kobold_exe = self.kobold_dir / "koboldcpp.exe"
            else:
                kobold_exe = self.kobold_dir / "koboldcpp"
            
            if not kobold_exe.exists():
                logger.error("❌ KoboldCPP executable not found")
                return False
            
            # Test basic functionality
            try:
                result = subprocess.run([str(kobold_exe), "--help"], 
                                      capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    logger.info("✅ KoboldCPP test successful")
                    return True
                else:
                    logger.error(f"❌ KoboldCPP test failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                logger.error("❌ KoboldCPP test timed out")
                return False
            except Exception as e:
                logger.error(f"❌ KoboldCPP test error: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing KoboldCPP: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Perform complete KoboldCPP setup."""
        try:
            logger.info("🚀 Starting KoboldCPP setup...")
            
            # Check if already installed
            if self.check_koboldcpp_installation():
                logger.info("✅ KoboldCPP already installed")
            else:
                # Download KoboldCPP
                if not self.download_koboldcpp():
                    logger.error("❌ Failed to download KoboldCPP")
                    return False
            
            # Create configuration
            config = self.create_kobold_config()
            if not self.save_config(config):
                logger.error("❌ Failed to save configuration")
                return False
            
            # Test installation
            if not self.test_koboldcpp():
                logger.error("❌ KoboldCPP test failed")
                return False
            
            logger.info("✅ KoboldCPP setup completed successfully!")
            logger.info("📋 Next steps:")
            logger.info("   1. Download a model using: python setup_koboldcpp.py --download-model <model_name>")
            logger.info("   2. Start Schwabot with: python -m core.schwabot_unified_interface")
            logger.info("   3. Access KoboldCPP at: http://localhost:5001")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KoboldCPP Setup for Schwabot")
    parser.add_argument("--download-model", type=str, help="Download a specific model")
    parser.add_argument("--test", action="store_true", help="Test KoboldCPP installation")
    parser.add_argument("--setup", action="store_true", help="Perform complete setup")
    
    args = parser.parse_args()
    
    setup = KoboldCPPSetup()
    
    if args.download_model:
        if setup.download_model(args.download_model):
            logger.info(f"✅ Model {args.download_model} downloaded successfully")
        else:
            logger.error(f"❌ Failed to download model {args.download_model}")
            sys.exit(1)
    
    elif args.test:
        if setup.test_koboldcpp():
            logger.info("✅ KoboldCPP test passed")
        else:
            logger.error("❌ KoboldCPP test failed")
            sys.exit(1)
    
    elif args.setup:
        if setup.setup_complete():
            logger.info("✅ Setup completed successfully")
        else:
            logger.error("❌ Setup failed")
            sys.exit(1)
    
    else:
        # Default: perform complete setup
        if setup.setup_complete():
            logger.info("✅ Setup completed successfully")
        else:
            logger.error("❌ Setup failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 