#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Utilities Module
============================

Provides comprehensive mathematical functions for the Schwabot trading system.
This module includes all mathematical operations needed for trading algorithms,
entropy calculations, pattern recognition, and signal processing.
"""

import math
import hashlib
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Try to import optional dependencies
try:
    import scipy.signal as signal
    import scipy.fft as fft
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SignalType(Enum):
    """Signal types for mathematical operations."""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    OSCILLATOR = "oscillator"
    COMPOSITE = "composite"

@dataclass
class MathematicalResult:
    """Container for mathematical operation results."""
    value: float
    confidence: float
    signal_type: SignalType
    metadata: Dict[str, Any]

def calculate_entropy(data: List[float], base: float = 2.0) -> float:
    """
    Calculate Shannon entropy of a data series.
    
    Args:
        data: List of numerical values
        base: Logarithm base (default: 2.0 for bits)
        
    Returns:
        Entropy value
    """
    if not data or len(data) < 2:
        return 0.0
    
    try:
        if SCIPY_AVAILABLE:
            return entropy(np.histogram(data, bins='auto')[0], base=base)
        else:
            # Manual entropy calculation
            hist, _ = np.histogram(data, bins='auto')
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) == 0:
                return 0.0
            prob = hist / hist.sum()
            return -np.sum(prob * np.log(prob) / np.log(base))
    except Exception:
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
    if len(series1) != len(series2) or len(series1) < 2:
        return 0.0
    
    try:
        return np.corrcoef(series1, series2)[0, 1]
    except Exception:
        return 0.0

def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate simple moving average.
    
    Args:
        data: Input data series
        window: Window size
        
    Returns:
        List of moving average values
    """
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(np.mean(data[:i+1]))
        else:
            result.append(np.mean(data[i-window+1:i+1]))
    
    return result

def exponential_smoothing(data: List[float], alpha: float = 0.1) -> List[float]:
    """
    Calculate exponential smoothing.
    
    Args:
        data: Input data series
        alpha: Smoothing factor (0 to 1)
        
    Returns:
        List of smoothed values
    """
    if not data:
        return []
    
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[i-1])
    
    return result

def calculate_rsi(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price data
        period: RSI period
        
    Returns:
        List of RSI values
    """
    if len(data) < period + 1:
        return [50.0] * len(data)
    
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = []
    avg_losses = []
    
    # Initial averages
    avg_gains.append(np.mean(gains[:period]))
    avg_losses.append(np.mean(losses[:period]))
    
    # Subsequent averages using exponential smoothing
    for i in range(period, len(gains)):
        avg_gains.append((avg_gains[-1] * (period - 1) + gains[i]) / period)
        avg_losses.append((avg_losses[-1] * (period - 1) + losses[i]) / period)
    
    rsi_values = []
    for i in range(len(avg_gains)):
        if avg_losses[i] == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    
    # Pad with 50.0 for initial values
    return [50.0] * period + rsi_values

def calculate_stochastic(data: List[float], period: int = 14) -> Tuple[List[float], List[float]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: Price data
        period: Stochastic period
        
    Returns:
        Tuple of (%K, %D) values
    """
    if len(data) < period:
        return [50.0] * len(data), [50.0] * len(data)
    
    k_values = []
    for i in range(period - 1, len(data)):
        high = max(data[i-period+1:i+1])
        low = min(data[i-period+1:i+1])
        close = data[i]
        
        if high == low:
            k_values.append(50.0)
        else:
            k_values.append(((close - low) / (high - low)) * 100.0)
    
    # Pad with 50.0 for initial values
    k_values = [50.0] * (period - 1) + k_values
    
    # Calculate %D (3-period SMA of %K)
    d_values = moving_average(k_values, 3)
    
    return k_values, d_values

def calculate_hash_distance(hash1: str, hash2: str) -> float:
    """
    Calculate Hamming distance between two hash strings.
    
    Args:
        hash1: First hash string
        hash2: Second hash string
        
    Returns:
        Hamming distance
    """
    if len(hash1) != len(hash2):
        return float('inf')
    
    distance = 0
    for c1, c2 in zip(hash1, hash2):
        if c1 != c2:
            distance += 1
    
    return distance

def calculate_weighted_confidence(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted confidence score.
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted confidence score
    """
    if len(values) != len(weights) or not values:
        return 0.0
    
    try:
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    except Exception:
        return 0.0

def waveform_pattern_match(signal: List[float], pattern: List[float]) -> float:
    """
    Calculate pattern matching score for waveform analysis.
    
    Args:
        signal: Input signal
        pattern: Pattern to match
        
    Returns:
        Pattern matching score (0 to 1)
    """
    if len(signal) < len(pattern) or not pattern:
        return 0.0
    
    try:
        # Use cross-correlation for pattern matching
        if SCIPY_AVAILABLE:
            correlation = signal.correlate(signal, pattern, mode='valid')
            max_corr = np.max(np.abs(correlation))
            return max_corr / (np.linalg.norm(signal) * np.linalg.norm(pattern))
        else:
            # Simple correlation calculation
            min_len = min(len(signal), len(pattern))
            signal_norm = signal[:min_len]
            pattern_norm = pattern[:min_len]
            
            correlation = np.corrcoef(signal_norm, pattern_norm)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0

def wavelet_decompose(data: List[float], levels: int = 3) -> List[List[float]]:
    """
    Perform wavelet decomposition (simplified version).
    
    Args:
        data: Input data
        levels: Number of decomposition levels
        
    Returns:
        List of wavelet coefficients
    """
    if not data or levels < 1:
        return []
    
    try:
        if SCIPY_AVAILABLE:
            # Use scipy wavelet decomposition
            coeffs = signal.wavedec(data, 'db1', level=levels)
            return [coeff.tolist() for coeff in coeffs]
        else:
            # Simple Haar wavelet implementation
            result = []
            current_data = data.copy()
            
            for level in range(levels):
                if len(current_data) < 2:
                    break
                
                # Haar wavelet transform
                coeffs = []
                for i in range(0, len(current_data) - 1, 2):
                    avg = (current_data[i] + current_data[i + 1]) / 2
                    diff = (current_data[i] - current_data[i + 1]) / 2
                    coeffs.extend([avg, diff])
                
                result.append(coeffs)
                current_data = coeffs[::2]  # Keep approximation coefficients
            
            return result
    except Exception:
        return []

def calculate_fractal_dimension(data: List[float]) -> float:
    """
    Calculate fractal dimension using box-counting method.
    
    Args:
        data: Input data series
        
    Returns:
        Fractal dimension estimate
    """
    if len(data) < 4:
        return 1.0
    
    try:
        # Simplified box-counting method
        data_array = np.array(data)
        data_range = data_array.max() - data_array.min()
        
        if data_range == 0:
            return 1.0
        
        # Normalize data
        normalized = (data_array - data_array.min()) / data_range
        
        # Count boxes at different scales
        scales = [0.1, 0.05, 0.025, 0.0125]
        counts = []
        
        for scale in scales:
            if scale > 0:
                count = len(np.unique(np.floor(normalized / scale)))
                counts.append(count)
        
        if len(counts) < 2:
            return 1.0
        
        # Calculate fractal dimension
        log_scales = np.log(scales[:len(counts)])
        log_counts = np.log(counts)
        
        # Linear regression
        slope = np.polyfit(log_scales, log_counts, 1)[0]
        return -slope
    except Exception:
        return 1.0

def calculate_volatility(data: List[float], window: int = 20) -> List[float]:
    """
    Calculate rolling volatility.
    
    Args:
        data: Price data
        window: Rolling window size
        
    Returns:
        List of volatility values
    """
    if len(data) < window:
        return [0.0] * len(data)
    
    volatility = []
    for i in range(len(data)):
        if i < window - 1:
            volatility.append(0.0)
        else:
            window_data = data[i-window+1:i+1]
            returns = np.diff(np.log(window_data))
            volatility.append(np.std(returns) * np.sqrt(252))  # Annualized
    
    return volatility

def calculate_momentum(data: List[float], period: int = 10) -> List[float]:
    """
    Calculate momentum indicator.
    
    Args:
        data: Price data
        period: Momentum period
        
    Returns:
        List of momentum values
    """
    if len(data) < period:
        return [0.0] * len(data)
    
    momentum = []
    for i in range(len(data)):
        if i < period - 1:
            momentum.append(0.0)
        else:
            momentum.append(data[i] - data[i - period])
    
    return momentum

def calculate_support_resistance(data: List[float], window: int = 20) -> Tuple[List[float], List[float]]:
    """
    Calculate support and resistance levels.
    
    Args:
        data: Price data
        window: Rolling window size
        
    Returns:
        Tuple of (support, resistance) levels
    """
    if len(data) < window:
        return [0.0] * len(data), [0.0] * len(data)
    
    support = []
    resistance = []
    
    for i in range(len(data)):
        if i < window - 1:
            support.append(data[i])
            resistance.append(data[i])
        else:
            window_data = data[i-window+1:i+1]
            support.append(min(window_data))
            resistance.append(max(window_data))
    
    return support, resistance

def calculate_trend_strength(data: List[float], period: int = 20) -> List[float]:
    """
    Calculate trend strength using linear regression.
    
    Args:
        data: Price data
        period: Analysis period
        
    Returns:
        List of trend strength values
    """
    if len(data) < period:
        return [0.0] * len(data)
    
    trend_strength = []
    for i in range(len(data)):
        if i < period - 1:
            trend_strength.append(0.0)
        else:
            window_data = data[i-period+1:i+1]
            x = np.arange(len(window_data))
            
            # Linear regression
            slope, intercept = np.polyfit(x, window_data, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((window_data - y_pred) ** 2)
            ss_tot = np.sum((window_data - np.mean(window_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            trend_strength.append(r_squared)
    
    return trend_strength

def generate_hash_vector(data: List[float], length: int = 64) -> str:
    """
    Generate hash vector from numerical data.
    
    Args:
        data: Input data
        length: Hash length
        
    Returns:
        Hash string
    """
    if not data:
        return "0" * length
    
    try:
        # Convert data to bytes
        data_str = ",".join(map(str, data))
        data_bytes = data_str.encode('utf-8')
        
        # Generate hash
        hash_obj = hashlib.sha256(data_bytes)
        hash_hex = hash_obj.hexdigest()
        
        # Truncate or pad to desired length
        if len(hash_hex) > length:
            return hash_hex[:length]
        else:
            return hash_hex.ljust(length, '0')
    except Exception:
        return "0" * length

def calculate_entropy_ratio(data: List[float], window1: int = 10, window2: int = 50) -> List[float]:
    """
    Calculate entropy ratio between two time windows.
    
    Args:
        data: Input data
        window1: Short window
        window2: Long window
        
    Returns:
        List of entropy ratios
    """
    if len(data) < max(window1, window2):
        return [1.0] * len(data)
    
    ratios = []
    for i in range(len(data)):
        if i < max(window1, window2) - 1:
            ratios.append(1.0)
        else:
            short_data = data[i-window1+1:i+1]
            long_data = data[i-window2+1:i+1]
            
            short_entropy = calculate_entropy(short_data)
            long_entropy = calculate_entropy(long_data)
            
            ratio = short_entropy / long_entropy if long_entropy > 0 else 1.0
            ratios.append(ratio)
    
    return ratios

def calculate_signal_quality(signal: List[float], noise_threshold: float = 0.1) -> float:
    """
    Calculate signal quality based on signal-to-noise ratio.
    
    Args:
        signal: Input signal
        noise_threshold: Noise threshold
        
    Returns:
        Signal quality score (0 to 1)
    """
    if not signal:
        return 0.0
    
    try:
        signal_array = np.array(signal)
        signal_power = np.mean(signal_array ** 2)
        noise_power = noise_threshold ** 2
        
        snr = signal_power / noise_power if noise_power > 0 else 0
        quality = 1.0 / (1.0 + np.exp(-snr))  # Sigmoid function
        
        return min(quality, 1.0)
    except Exception:
        return 0.0

# Convenience functions for common operations
def normalize_data(data: List[float]) -> List[float]:
    """Normalize data to [0, 1] range."""
    if not data:
        return []
    
    data_array = np.array(data)
    min_val = data_array.min()
    max_val = data_array.max()
    
    if max_val == min_val:
        return [0.5] * len(data)
    
    return ((data_array - min_val) / (max_val - min_val)).tolist()

def standardize_data(data: List[float]) -> List[float]:
    """Standardize data to zero mean and unit variance."""
    if not data:
        return []
    
    data_array = np.array(data)
    mean_val = data_array.mean()
    std_val = data_array.std()
    
    if std_val == 0:
        return [0.0] * len(data)
    
    return ((data_array - mean_val) / std_val).tolist()

def smooth_data(data: List[float], method: str = "sma", **kwargs) -> List[float]:
    """
    Smooth data using various methods.
    
    Args:
        data: Input data
        method: Smoothing method ("sma", "ema", "savgol")
        **kwargs: Method-specific parameters
        
    Returns:
        Smoothed data
    """
    if method == "sma":
        window = kwargs.get("window", 5)
        return moving_average(data, window)
    elif method == "ema":
        alpha = kwargs.get("alpha", 0.1)
        return exponential_smoothing(data, alpha)
    elif method == "savgol" and SCIPY_AVAILABLE:
        window = kwargs.get("window", 5)
        polyorder = kwargs.get("polyorder", 2)
        return signal.savgol_filter(data, window, polyorder).tolist()
    else:
        return data

# Export all functions
__all__ = [
    'calculate_entropy', 'calculate_correlation', 'moving_average',
    'exponential_smoothing', 'calculate_rsi', 'calculate_stochastic',
    'calculate_hash_distance', 'calculate_weighted_confidence',
    'waveform_pattern_match', 'wavelet_decompose', 'calculate_fractal_dimension',
    'calculate_volatility', 'calculate_momentum', 'calculate_support_resistance',
    'calculate_trend_strength', 'generate_hash_vector', 'calculate_entropy_ratio',
    'calculate_signal_quality', 'normalize_data', 'standardize_data', 'smooth_data',
    'MathematicalResult', 'SignalType'
] 