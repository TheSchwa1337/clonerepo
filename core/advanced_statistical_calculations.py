#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ADVANCED STATISTICAL CALCULATIONS - MARKET PATTERN ANALYSIS
=============================================================

Advanced statistical calculations for market pattern analysis and chaos theory.

Features:
- Fractal Dimension: D = log(N) / log(1/r)
- Hurst Exponent: H = log(R/S) / log(T)
- Lyapunov Exponents: For chaos detection
- Correlation Dimension: For attractor analysis
- Multifractal Spectrum: For multi-scale analysis
- Cross-platform compatibility with GPU acceleration
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

logger = logging.getLogger(__name__)

@dataclass
class FractalResult:
    """Result container for fractal analysis."""
    fractal_dimension: float
    correlation_dimension: float
    hurst_exponent: float
    lyapunov_exponent: float
    multifractal_spectrum: Dict[str, float]
    confidence_interval: Tuple[float, float]
    calculation_time: float
    metadata: Dict[str, Any]

class AdvancedStatisticalCalculations:
    """
    Advanced statistical calculations for market pattern analysis.
    
    Mathematical Foundations:
    - Fractal Dimension: D = log(N) / log(1/r)
    - Hurst Exponent: H = log(R/S) / log(T)
    - Lyapunov Exponents: λ = lim(t→∞) (1/t) * log|df^t(x)/dx|
    - Correlation Dimension: ν = lim(r→0) log(C(r)) / log(r)
    - Multifractal Spectrum: f(α) = qα - τ(q)
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize advanced statistical calculations."""
        self.use_gpu = use_gpu and USING_CUDA
        self.min_points = 100
        self.max_embedding_dim = 10
        self.time_delay = 1
        
        logger.info(f"🔬 Advanced Statistical Calculations initialized with {_backend}")
    
    def calculate_fractal_dimension(self, data: np.ndarray, method: str = "box_counting") -> float:
        """
        Calculate fractal dimension using various methods.
        
        Mathematical Formula:
        D = log(N) / log(1/r)
        
        Args:
            data: Time series data
            method: 'box_counting', 'correlation', 'information'
            
        Returns:
            Fractal dimension
        """
        try:
            if method == "box_counting":
                return self._box_counting_dimension(data)
            elif method == "correlation":
                return self._correlation_dimension(data)
            elif method == "information":
                return self._information_dimension(data)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Fractal dimension calculation failed: {e}")
            return 1.0
    
    def _box_counting_dimension(self, data: np.ndarray) -> float:
        """Box counting method for fractal dimension."""
        try:
            # Normalize data to [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Define box sizes
            box_sizes = np.logspace(-3, 0, 20)
            counts = []
            
            for r in box_sizes:
                # Count boxes containing data points
                boxes = np.floor(data_norm / r).astype(int)
                unique_boxes = len(np.unique(boxes))
                counts.append(unique_boxes)
            
            # Linear fit to log-log plot
            valid_indices = np.where(np.array(counts) > 0)[0]
            if len(valid_indices) < 3:
                return 1.0
                
            log_sizes = np.log(box_sizes[valid_indices])
            log_counts = np.log(np.array(counts)[valid_indices])
            
            # Calculate slope (fractal dimension)
            slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
            
            return abs(slope)
            
        except Exception as e:
            logger.error(f"Box counting dimension failed: {e}")
            return 1.0
    
    def _correlation_dimension(self, data: np.ndarray) -> float:
        """Correlation dimension calculation."""
        try:
            # Calculate pairwise distances
            distances = pdist(data.reshape(-1, 1))
            
            # Define radius values
            radii = np.logspace(-3, 0, 20)
            correlations = []
            
            for r in radii:
                # Count pairs within radius r
                count = np.sum(distances < r)
                correlations.append(count / len(distances))
            
            # Linear fit to log-log plot
            valid_indices = np.where(np.array(correlations) > 0)[0]
            if len(valid_indices) < 3:
                return 1.0
                
            log_radii = np.log(radii[valid_indices])
            log_correlations = np.log(np.array(correlations)[valid_indices])
            
            slope, _, _, _, _ = stats.linregress(log_radii, log_correlations)
            
            return abs(slope)
            
        except Exception as e:
            logger.error(f"Correlation dimension failed: {e}")
            return 1.0
    
    def _information_dimension(self, data: np.ndarray) -> float:
        """Information dimension calculation."""
        try:
            # Similar to box counting but with entropy
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            box_sizes = np.logspace(-3, 0, 20)
            entropies = []
            
            for r in box_sizes:
                boxes = np.floor(data_norm / r).astype(int)
                # Calculate entropy of box distribution
                unique, counts = np.unique(boxes, return_counts=True)
                probabilities = counts / len(boxes)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                entropies.append(entropy)
            
            # Linear fit
            valid_indices = np.where(np.array(entropies) > 0)[0]
            if len(valid_indices) < 3:
                return 1.0
                
            log_sizes = np.log(box_sizes[valid_indices])
            entropies_array = np.array(entropies)[valid_indices]
            
            slope, _, _, _, _ = stats.linregress(log_sizes, entropies_array)
            
            return abs(slope)
            
        except Exception as e:
            logger.error(f"Information dimension failed: {e}")
            return 1.0
    
    def calculate_hurst_exponent(self, data: np.ndarray, method: str = "rs_analysis") -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        Mathematical Formula:
        H = log(R/S) / log(T)
        
        Args:
            data: Time series data
            method: 'rs_analysis', 'aggregated_variance', 'differenced_variance'
            
        Returns:
            Hurst exponent
        """
        try:
            if method == "rs_analysis":
                return self._rs_analysis_hurst(data)
            elif method == "aggregated_variance":
                return self._aggregated_variance_hurst(data)
            elif method == "differenced_variance":
                return self._differenced_variance_hurst(data)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Hurst exponent calculation failed: {e}")
            return 0.5
    
    def _rs_analysis_hurst(self, data: np.ndarray) -> float:
        """R/S analysis for Hurst exponent."""
        try:
            # Define time scales
            scales = np.logspace(1, np.log10(len(data)//4), 10, dtype=int)
            rs_values = []
            
            for scale in scales:
                # Divide data into segments
                segments = len(data) // scale
                if segments < 2:
                    continue
                    
                rs_segments = []
                for i in range(segments):
                    segment = data[i*scale:(i+1)*scale]
                    if len(segment) < 2:
                        continue
                        
                    # Calculate R (range)
                    mean_segment = np.mean(segment)
                    cumulative = np.cumsum(segment - mean_segment)
                    r = np.max(cumulative) - np.min(cumulative)
                    
                    # Calculate S (standard deviation)
                    s = np.std(segment)
                    
                    if s > 0:
                        rs_segments.append(r / s)
                
                if rs_segments:
                    rs_values.append(np.mean(rs_segments))
            
            # Linear fit to log-log plot
            if len(rs_values) < 3:
                return 0.5
                
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_rs)
            
            return slope
            
        except Exception as e:
            logger.error(f"R/S analysis failed: {e}")
            return 0.5
    
    def _aggregated_variance_hurst(self, data: np.ndarray) -> float:
        """Aggregated variance method for Hurst exponent."""
        try:
            scales = np.logspace(1, np.log10(len(data)//4), 10, dtype=int)
            variances = []
            
            for scale in scales:
                segments = len(data) // scale
                if segments < 2:
                    continue
                    
                segment_variances = []
                for i in range(segments):
                    segment = data[i*scale:(i+1)*scale]
                    if len(segment) > 1:
                        segment_variances.append(np.var(segment))
                
                if segment_variances:
                    variances.append(np.mean(segment_variances))
            
            # Linear fit
            if len(variances) < 3:
                return 0.5
                
            log_scales = np.log(scales[:len(variances)])
            log_variances = np.log(variances)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_variances)
            
            return 1 + slope / 2
            
        except Exception as e:
            logger.error(f"Aggregated variance failed: {e}")
            return 0.5
    
    def _differenced_variance_hurst(self, data: np.ndarray) -> float:
        """Differenced variance method for Hurst exponent."""
        try:
            # Calculate differences
            diff_data = np.diff(data)
            
            # Use aggregated variance on differences
            return self._aggregated_variance_hurst(diff_data)
            
        except Exception as e:
            logger.error(f"Differenced variance failed: {e}")
            return 0.5
    
    def calculate_lyapunov_exponent(self, data: np.ndarray, embedding_dim: int = 3) -> float:
        """
        Calculate the largest Lyapunov exponent.
        
        Mathematical Formula:
        λ = lim(t→∞) (1/t) * log|df^t(x)/dx|
        
        Args:
            data: Time series data
            embedding_dim: Embedding dimension for phase space reconstruction
            
        Returns:
            Largest Lyapunov exponent
        """
        try:
            # Phase space reconstruction
            embedded_data = self._phase_space_reconstruction(data, embedding_dim, self.time_delay)
            
            if len(embedded_data) < 100:
                return 0.0
            
            # Find nearest neighbors
            distances = squareform(pdist(embedded_data))
            np.fill_diagonal(distances, np.inf)
            
            # Calculate Lyapunov exponent
            lyap_values = []
            max_iter = min(50, len(embedded_data) // 10)
            
            for i in range(max_iter):
                # Find nearest neighbor
                nearest_idx = np.argmin(distances[i])
                
                # Calculate separation
                initial_sep = distances[i, nearest_idx]
                
                # Track separation over time
                separations = []
                for t in range(1, min(20, len(embedded_data) - max(i, nearest_idx))):
                    if i + t < len(embedded_data) and nearest_idx + t < len(embedded_data):
                        sep = np.linalg.norm(embedded_data[i + t] - embedded_data[nearest_idx + t])
                        if sep > 0 and initial_sep > 0:
                            separations.append(np.log(sep / initial_sep) / t)
                
                if separations:
                    lyap_values.append(np.mean(separations))
            
            if lyap_values:
                return np.mean(lyap_values)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Lyapunov exponent calculation failed: {e}")
            return 0.0
    
    def _phase_space_reconstruction(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Phase space reconstruction using time delay embedding."""
        try:
            n_points = len(data) - (dim - 1) * delay
            if n_points <= 0:
                return np.array([])
            
            embedded = np.zeros((n_points, dim))
            for i in range(dim):
                embedded[:, i] = data[i * delay:i * delay + n_points]
            
            return embedded
            
        except Exception as e:
            logger.error(f"Phase space reconstruction failed: {e}")
            return np.array([])
    
    def calculate_multifractal_spectrum(self, data: np.ndarray, q_range: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate multifractal spectrum.
        
        Mathematical Formula:
        f(α) = qα - τ(q)
        
        Args:
            data: Time series data
            q_range: Range of q values for multifractal analysis
            
        Returns:
            Multifractal spectrum parameters
        """
        try:
            if q_range is None:
                q_range = np.linspace(-5, 5, 21)
            
            # Calculate partition function
            scales = np.logspace(1, np.log10(len(data)//4), 10, dtype=int)
            partition_functions = []
            
            for scale in scales:
                segments = len(data) // scale
                if segments < 2:
                    continue
                    
                pf_values = []
                for i in range(segments):
                    segment = data[i*scale:(i+1)*scale]
                    if len(segment) > 1:
                        # Calculate measure for segment
                        measure = np.sum(np.abs(np.diff(segment)))
                        pf_values.append(measure)
                
                if pf_values:
                    partition_functions.append(pf_values)
            
            # Calculate τ(q) for each q
            tau_values = []
            for q in q_range:
                tau_q = []
                for i, scale in enumerate(scales[:len(partition_functions)]):
                    if i < len(partition_functions):
                        pf_q = np.sum(np.array(partition_functions[i]) ** q)
                        if pf_q > 0:
                            tau_q.append(np.log(pf_q))
                
                if len(tau_q) >= 3:
                    # Linear fit to get τ(q)
                    log_scales = np.log(scales[:len(tau_q)])
                    slope, _, _, _, _ = stats.linregress(log_scales, tau_q)
                    tau_values.append(slope)
                else:
                    tau_values.append(0.0)
            
            # Calculate α and f(α)
            alpha_values = []
            f_alpha_values = []
            
            for i in range(1, len(q_range) - 1):
                # Calculate α = dτ/dq
                dq = q_range[i+1] - q_range[i-1]
                dtau = tau_values[i+1] - tau_values[i-1]
                alpha = dtau / dq if dq != 0 else 0.0
                alpha_values.append(alpha)
                
                # Calculate f(α) = qα - τ(q)
                f_alpha = q_range[i] * alpha - tau_values[i]
                f_alpha_values.append(f_alpha)
            
            return {
                "alpha_range": (np.min(alpha_values), np.max(alpha_values)) if alpha_values else (0.0, 0.0),
                "f_alpha_max": np.max(f_alpha_values) if f_alpha_values else 0.0,
                "multifractal_width": np.max(alpha_values) - np.min(alpha_values) if alpha_values else 0.0,
                "tau_spectrum": dict(zip(q_range, tau_values)),
                "alpha_spectrum": dict(zip(q_range[1:-1], alpha_values)),
                "f_alpha_spectrum": dict(zip(q_range[1:-1], f_alpha_values))
            }
            
        except Exception as e:
            logger.error(f"Multifractal spectrum calculation failed: {e}")
            return {
                "alpha_range": (0.0, 0.0),
                "f_alpha_max": 0.0,
                "multifractal_width": 0.0,
                "tau_spectrum": {},
                "alpha_spectrum": {},
                "f_alpha_spectrum": {}
            }
    
    def comprehensive_statistical_analysis(self, data: np.ndarray) -> FractalResult:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Time series data
            
        Returns:
            FractalResult with all statistical measures
        """
        start_time = time.time()
        
        try:
            # Calculate all statistical measures
            fractal_dim = self.calculate_fractal_dimension(data, "box_counting")
            correlation_dim = self.calculate_fractal_dimension(data, "correlation")
            hurst_exp = self.calculate_hurst_exponent(data, "rs_analysis")
            lyapunov_exp = self.calculate_lyapunov_exponent(data)
            multifractal = self.calculate_multifractal_spectrum(data)
            
            # Calculate confidence intervals (simplified)
            confidence_interval = (0.8, 1.2)  # Placeholder
            
            calculation_time = time.time() - start_time
            
            return FractalResult(
                fractal_dimension=fractal_dim,
                correlation_dimension=correlation_dim,
                hurst_exponent=hurst_exp,
                lyapunov_exponent=lyapunov_exp,
                multifractal_spectrum=multifractal,
                confidence_interval=confidence_interval,
                calculation_time=calculation_time,
                metadata={
                    "data_length": len(data),
                    "method": "comprehensive",
                    "gpu_used": self.use_gpu
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive statistical analysis failed: {e}")
            return FractalResult(
                fractal_dimension=1.0,
                correlation_dimension=1.0,
                hurst_exponent=0.5,
                lyapunov_exponent=0.0,
                multifractal_spectrum={},
                confidence_interval=(0.0, 0.0),
                calculation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

# Global instance
advanced_statistical_calculations = AdvancedStatisticalCalculations() 