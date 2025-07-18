#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Backtesting System
===========================

This module provides backtesting functionality for the Schwabot trading system.
"""

# Import existing modules
try:
    from .backtest_engine import BacktestConfig, BacktestEngine, BacktestResult, HistoricalDataLoader
except ImportError:
    # Fallback if backtest engine is not available
    BacktestConfig = None
    BacktestEngine = None
    BacktestResult = None
    HistoricalDataLoader = None

try:
    from .data_sources import DataSourceManager, DataSourceConfig, SimulatedDataGenerator
except ImportError:
    DataSourceManager = None
    DataSourceConfig = None
    SimulatedDataGenerator = None

# Use simplified mathematical integration to avoid import issues
try:
    from .mathematical_integration_simplified import mathematical_integration, MathematicalSignal
except ImportError:
    # Fallback if simplified version is not available
    mathematical_integration = None
    MathematicalSignal = None

__all__ = [
    'mathematical_integration',
    'MathematicalSignal',
    'BacktestConfig',
    'BacktestEngine',
    'BacktestResult',
    'HistoricalDataLoader',
    'DataSourceManager',
    'DataSourceConfig',
    'SimulatedDataGenerator'
] 