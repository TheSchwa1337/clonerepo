#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR CONFIG LOADER - CONFIGURATION MANAGEMENT
===================================================

Loads and manages configuration for the Kaprekar system integration.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class KaprekarConfigLoader:
    """Configuration loader for Kaprekar system."""
    
    def __init__(self, config_path: str = "config/kaprekar_config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
        
    def load_config(self) -> bool:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Kaprekar config loaded from {self.config_path}")
                return True
            else:
                logger.warning(f"Kaprekar config file not found: {self.config_path}")
                self.config = self.get_default_config()
                self.save_config()
                return True
        except Exception as e:
            logger.error(f"Error loading Kaprekar config: {e}")
            self.config = self.get_default_config()
            return False
    
    def save_config(self) -> bool:
        """Save configuration to YAML file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Kaprekar config saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Kaprekar config: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "kaprekar_system": {
                "enabled": True,
                "version": "1.0.0",
                "description": "Kaprekar's Constant entropy classification system",
                "kaprekar_constant": 6174,
                "max_steps": 7,
                "reject_threshold": 99,
                "entropy_classes": {
                    1: "ULTRA_STABLE",
                    2: "STABLE",
                    3: "MODERATE",
                    4: "ACTIVE",
                    5: "VOLATILE",
                    6: "HIGH_VOLATILITY",
                    7: "EXTREME_VOLATILITY"
                },
                "strategy_mapping": {
                    1: "conservative_hold",
                    2: "moderate_buy",
                    3: "aggressive_buy",
                    4: "volatility_play",
                    5: "momentum_follow",
                    6: "breakout_trade",
                    7: "swing_trading"
                }
            },
            "integration": {
                "strategy_mapper": {
                    "enabled": True,
                    "entropy_weight": 0.3,
                    "confidence_threshold": 0.7,
                    "strategy_boost": True
                },
                "hash_config": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                    "entropy_weight": 0.3,
                    "strategy_boost": True
                },
                "lantern_core": {
                    "enabled": True,
                    "echo_enhancement": True,
                    "confidence_boost": True,
                    "ghost_memory_integration": True
                }
            },
            "trg_analyzer": {
                "enabled": True,
                "rsi_phases": {
                    "OVERSOLD": [0, 30],
                    "LOW": [30, 45],
                    "NEUTRAL": [45, 55],
                    "HIGH": [55, 70],
                    "OVERBOUGHT": [70, 100]
                },
                "signal_rules": {
                    "btc_long_entry": {
                        "kcs_range": [1, 3],
                        "rsi_range": [20, 35],
                        "phantom_delta": "positive",
                        "confidence_threshold": 0.7
                    },
                    "usdc_exit_trigger": {
                        "kcs_range": [5, 7],
                        "rsi_range": [70, 100],
                        "phantom_delta": "any",
                        "confidence_threshold": 0.6
                    },
                    "phantom_band_swing": {
                        "kcs_range": [3, 5],
                        "rsi_range": [35, 65],
                        "phantom_delta": "any",
                        "confidence_threshold": 0.5
                    },
                    "conservative_hold": {
                        "kcs_range": [1, 2],
                        "rsi_range": [40, 60],
                        "phantom_delta": "any",
                        "confidence_threshold": 0.8
                    }
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "track_convergence_rate": True,
                    "track_chaotic_rate": True,
                    "track_signal_generation_rate": True,
                    "track_average_confidence": True
                },
                "logging": {
                    "level": "INFO",
                    "log_kaprekar_analysis": True,
                    "log_trg_analysis": True,
                    "log_enhanced_signals": True
                },
                "health": {
                    "check_interval": 60,
                    "alert_threshold": 0.8,
                    "performance_threshold": 0.7
                }
            },
            "advanced": {
                "caching": {
                    "enabled": True,
                    "cache_size": 1000,
                    "cache_ttl": 3600
                },
                "batch_processing": {
                    "enabled": True,
                    "batch_size": 10,
                    "max_concurrent": 4
                },
                "ai_validation": {
                    "enabled": True,
                    "validation_endpoint": "http://localhost:5000/ai/validate",
                    "timeout": 5.0,
                    "retry_attempts": 3
                }
            },
            "development": {
                "demo_mode": False,
                "test_scenarios": [
                    {
                        "name": "BTC Long Entry",
                        "hash_fragment": "a1b2c3d4",
                        "rsi": 29.7,
                        "expected_result": "btc_long_entry"
                    },
                    {
                        "name": "USDC Exit",
                        "hash_fragment": "f4e3d2c1",
                        "rsi": 75.2,
                        "expected_result": "usdc_exit_trigger"
                    },
                    {
                        "name": "Phantom Swing",
                        "hash_fragment": "12345678",
                        "rsi": 52.1,
                        "expected_result": "phantom_band_swing"
                    }
                ]
            }
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'kaprekar_system.enabled')."""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value by key path."""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Error setting config {key}: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if Kaprekar system is enabled."""
        return self.get_config('kaprekar_system.enabled', True)
    
    def get_integration_config(self, component: str) -> Dict[str, Any]:
        """Get integration configuration for a specific component."""
        return self.get_config(f'integration.{component}', {})
    
    def get_trg_config(self) -> Dict[str, Any]:
        """Get TRG analyzer configuration."""
        return self.get_config('trg_analyzer', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get_config('monitoring', {})
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration."""
        return self.get_config('advanced', {})
    
    def validate_config(self) -> bool:
        """Validate configuration structure."""
        required_keys = [
            'kaprekar_system.enabled',
            'integration.strategy_mapper.enabled',
            'integration.hash_config.enabled',
            'integration.lantern_core.enabled'
        ]
        
        for key in required_keys:
            if self.get_config(key) is None:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True

# Global instance
kaprekar_config = KaprekarConfigLoader() 