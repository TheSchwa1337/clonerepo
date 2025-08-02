#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Utilities Module
============================

Provides mathematical utility functions for the Schwabot trading system.
This module includes statistical analysis, signal processing, and mathematical
operations used throughout the system.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MathematicalResult:
    """Result of mathematical operations."""
    value: float
    confidence: float
    metadata: Dict[str, Any]

def calculate_entropy(data: List[float]) -> float:
    """
    Calculate Shannon entropy of a data series.
    
    Args:
        data: List of numerical values
        
    Returns:
        Entropy value
    """
    try:
        if not data:
            return 0.0
        
        # Convert to numpy array
        arr = np.array(data)
        
        # Calculate histogram
        hist, _ = np.histogram(arr, bins=min(50, len(arr)//10))
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate probabilities
        probs = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Error calculating entropy: {e}")
        return 0.0

def calculate_correlation(series1: List[float], series2: List[float]) -> float:
    """
    Calculate correlation coefficient between two series.
    
    Args:
        series1: First data series
        series2: Second data series
        
    Returns:
        Correlation coefficient (-1 to 1)
    """
    try:
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        # Convert to numpy arrays
        arr1 = np.array(series1)
        arr2 = np.array(series2)
        
        # Calculate correlation
        correlation = np.corrcoef(arr1, arr2)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return 0.0

def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average of a data series.
    
    Args:
        data: Input data series
        window: Window size for moving average
        
    Returns:
        List of moving average values
    """
    try:
        if not data or window <= 0 or window > len(data):
            return data
        
        # Convert to numpy array
        arr = np.array(data)
        
        # Calculate moving average
        ma = np.convolve(arr, np.ones(window)/window, mode='valid')
        
        # Pad the beginning with the first value
        padding = [ma[0]] * (window - 1)
        
        return list(padding) + list(ma)
        
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        return data

def exponential_smoothing(data: List[float], alpha: float = 0.1) -> List[float]:
    """
    Calculate exponential smoothing of a data series.
    
    Args:
        data: Input data series
        alpha: Smoothing factor (0 to 1)
        
    Returns:
        List of smoothed values
    """
    try:
        if not data:
            return []
        
        # Convert to numpy array
        arr = np.array(data)
        
        # Calculate exponential smoothing
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        
        for i in range(1, len(arr)):
            smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
        
        return list(smoothed)
        
    except Exception as e:
        logger.error(f"Error calculating exponential smoothing: {e}")
        return data

def calculate_volatility(data: List[float], window: int = 20) -> List[float]:
    """
    Calculate rolling volatility of a data series.
    
    Args:
        data: Input data series
        window: Window size for volatility calculation
        
    Returns:
        List of volatility values
    """
    try:
        if not data or window <= 1 or window > len(data):
            return [0.0] * len(data)
        
        # Convert to numpy array
        arr = np.array(data)
        
        # Calculate returns
        returns = np.diff(arr) / arr[:-1]
        
        # Calculate rolling volatility
        volatility = []
        for i in range(len(arr)):
            if i < window - 1:
                volatility.append(0.0)
            else:
                window_returns = returns[i-window+1:i]
                vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                volatility.append(vol)
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return [0.0] * len(data)

def calculate_rsi(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Price data series
        period: RSI period
        
    Returns:
        List of RSI values
    """
    try:
        if not data or period <= 1 or period >= len(data):
            return [50.0] * len(data)
        
        # Convert to numpy array
        arr = np.array(data)
        
        # Calculate price changes
        deltas = np.diff(arr)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate RSI
        rsi_values = []
        for i in range(len(arr)):
            if i < period:
                rsi_values.append(50.0)
            else:
                avg_gain = np.mean(gains[i-period:i])
                avg_loss = np.mean(losses[i-period:i])
                
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                rsi_values.append(rsi)
        
        return rsi_values
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return [50.0] * len(data)

def calculate_bollinger_bands(data: List[float], window: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price data series
        window: Window size for moving average
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    try:
        if not data or window <= 1 or window > len(data):
            return data, data, data
        
        # Calculate moving average
        ma = moving_average(data, window)
        
        # Calculate standard deviation
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            if i < window - 1:
                upper_band.append(data[i])
                lower_band.append(data[i])
            else:
                window_data = data[i-window+1:i+1]
                std = np.std(window_data)
                upper_band.append(ma[i] + std_dev * std)
                lower_band.append(ma[i] - std_dev * std)
        
        return upper_band, ma, lower_band
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return data, data, data

def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data series
        method: Normalization method ('minmax', 'zscore', 'decimal')
        
    Returns:
        List of normalized values
    """
    try:
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val == min_val:
                return [0.5] * len(data)
            normalized = (arr - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            if std_val == 0:
                return [0.0] * len(data)
            normalized = (arr - mean_val) / std_val
            
        elif method == 'decimal':
            # Decimal scaling
            max_abs = np.max(np.abs(arr))
            if max_abs == 0:
                return [0.0] * len(data)
            normalized = arr / max_abs
            
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data
        
        return list(normalized)
        
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return data

def calculate_momentum(data: List[float], period: int = 10) -> List[float]:
    """
    Calculate momentum indicator.
    
    Args:
        data: Price data series
        period: Momentum period
        
    Returns:
        List of momentum values
    """
    try:
        if not data or period <= 0 or period >= len(data):
            return [0.0] * len(data)
        
        momentum = []
        for i in range(len(data)):
            if i < period:
                momentum.append(0.0)
            else:
                momentum_val = data[i] - data[i - period]
                momentum.append(momentum_val)
        
        return momentum
        
    except Exception as e:
        logger.error(f"Error calculating momentum: {e}")
        return [0.0] * len(data)

def calculate_stochastic(data: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: Price data series
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K values, %D values)
    """
    try:
        if not data or k_period <= 1 or k_period >= len(data):
            return [50.0] * len(data), [50.0] * len(data)
        
        # Calculate %K
        k_values = []
        for i in range(len(data)):
            if i < k_period - 1:
                k_values.append(50.0)
            else:
                window_data = data[i-k_period+1:i+1]
                high = max(window_data)
                low = min(window_data)
                close = data[i]
                
                if high == low:
                    k_val = 50.0
                else:
                    k_val = ((close - low) / (high - low)) * 100
                
                k_values.append(k_val)
        
        # Calculate %D (SMA of %K)
        d_values = moving_average(k_values, d_period)
        
        return k_values, d_values
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        return [50.0] * len(data), [50.0] * len(data)

def calculate_macd(data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price data series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    try:
        if not data or fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            return [0.0] * len(data), [0.0] * len(data), [0.0] * len(data)
        
        # Calculate EMAs
        fast_ema = exponential_smoothing(data, alpha=2.0/(fast_period+1))
        slow_ema = exponential_smoothing(data, alpha=2.0/(slow_period+1))
        
        # Calculate MACD line
        macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
        
        # Calculate signal line
        signal_line = exponential_smoothing(macd_line, alpha=2.0/(signal_period+1))
        
        # Calculate histogram
        histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return [0.0] * len(data), [0.0] * len(data), [0.0] * len(data)

def calculate_fibonacci_retracements(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: High price
        low: Low price
        
    Returns:
        Dictionary of retracement levels
    """
    try:
        diff = high - low
        
        levels = {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
        
        return levels
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci retracements: {e}")
        return {}

def calculate_support_resistance(data: List[float], window: int = 20) -> Tuple[List[float], List[float]]:
    """
    Calculate support and resistance levels.
    
    Args:
        data: Price data series
        window: Window size for analysis
        
    Returns:
        Tuple of (support levels, resistance levels)
    """
    try:
        if not data or window <= 1 or window > len(data):
            return [], []
        
        support_levels = []
        resistance_levels = []
        
        for i in range(len(data)):
            if i < window - 1:
                support_levels.append(data[i])
                resistance_levels.append(data[i])
            else:
                window_data = data[i-window+1:i+1]
                support = min(window_data)
                resistance = max(window_data)
                support_levels.append(support)
                resistance_levels.append(resistance)
        
        return support_levels, resistance_levels
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return [], []

# Convenience functions for common operations
def safe_math_operation(operation: str, *args, **kwargs) -> MathematicalResult:
    """
    Safely perform mathematical operations with error handling.
    
    Args:
        operation: Operation to perform
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        MathematicalResult with value, confidence, and metadata
    """
    try:
        if operation == 'entropy':
            value = calculate_entropy(*args)
        elif operation == 'correlation':
            value = calculate_correlation(*args)
        elif operation == 'moving_average':
            value = moving_average(*args)
        elif operation == 'exponential_smoothing':
            value = exponential_smoothing(*args)
        elif operation == 'volatility':
            value = calculate_volatility(*args)
        elif operation == 'rsi':
            value = calculate_rsi(*args)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return MathematicalResult(
            value=value if isinstance(value, (int, float)) else 0.0,
            confidence=1.0,
            metadata={'operation': operation, 'success': True}
        )
        
    except Exception as e:
        logger.error(f"Mathematical operation failed: {e}")
        return MathematicalResult(
            value=0.0,
            confidence=0.0,
            metadata={'operation': operation, 'success': False, 'error': str(e)}
        ) 