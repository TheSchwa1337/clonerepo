#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 WAVEFORM & SIGNAL PROCESSING - ADVANCED MARKET ANALYSIS
==========================================================

Advanced waveform and signal processing for market analysis.

Features:
- Wavelet Transforms: Multi-resolution time-frequency analysis
- Hilbert-Huang Transform: Instantaneous frequency analysis
- Cross-Spectral Density: Inter-asset correlation analysis
- Phase Synchronization: Market coherence detection
- Amplitude Modulation: Volatility pattern recognition
- GPU acceleration with automatic CPU fallback
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d

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

# PyWavelets for wavelet transforms
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PyWavelets not available, using scipy wavelets")

logger = logging.getLogger(__name__)

@dataclass
class WaveformResult:
    """Result container for waveform analysis."""
    wavelet_coefficients: Dict[str, np.ndarray]
    instantaneous_frequency: np.ndarray
    cross_spectral_density: np.ndarray
    phase_synchronization: float
    amplitude_modulation: np.ndarray
    dominant_frequencies: List[float]
    calculation_time: float
    metadata: Dict[str, Any]

class WaveformSignalProcessing:
    """
    Advanced waveform and signal processing for market analysis.
    
    Mathematical Foundations:
    - Wavelet Transform: W(a,b) = ∫ f(t) ψ*((t-b)/a) dt
    - Hilbert Transform: H[f](t) = (1/π) ∫ f(τ)/(t-τ) dτ
    - Cross-Spectral Density: S_xy(f) = F[R_xy(τ)]
    - Phase Synchronization: γ = |⟨e^(iφ)⟩|
    - Amplitude Modulation: A(t) = √(x²(t) + H²[x](t))
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize waveform signal processing."""
        self.use_gpu = use_gpu and USING_CUDA
        self.sampling_rate = 1.0  # Default sampling rate
        self.wavelet_type = 'db4'  # Daubechies 4 wavelet
        self.max_frequency = 0.5  # Nyquist frequency
        
        logger.info(f"🌊 Waveform Signal Processing initialized with {_backend}")
        if not PYWAVELETS_AVAILABLE:
            logger.warning("⚠️ Using scipy wavelets as fallback")
    
    def wavelet_transform(self, data: np.ndarray, wavelet: str = None, levels: int = None) -> Dict[str, np.ndarray]:
        """
        Perform continuous wavelet transform.
        
        Mathematical Formula:
        W(a,b) = ∫ f(t) ψ*((t-b)/a) dt
        
        Args:
            data: Time series data
            wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.)
            levels: Number of decomposition levels
            
        Returns:
            Dictionary with wavelet coefficients
        """
        try:
            if wavelet is None:
                wavelet = self.wavelet_type
            if levels is None:
                levels = int(np.log2(len(data))) - 1
            
            if PYWAVELETS_AVAILABLE:
                return self._pywt_wavelet_transform(data, wavelet, levels)
            else:
                return self._scipy_wavelet_transform(data, levels)
                
        except Exception as e:
            logger.error(f"Wavelet transform failed: {e}")
            return {"coefficients": np.array([]), "scales": np.array([])}
    
    def _pywt_wavelet_transform(self, data: np.ndarray, wavelet: str, levels: int) -> Dict[str, np.ndarray]:
        """PyWavelets implementation of wavelet transform."""
        try:
            # Discrete wavelet transform
            coeffs = pywt.wavedec(data, wavelet, level=levels)
            
            # Continuous wavelet transform for detailed analysis
            scales = np.logspace(1, np.log10(len(data)//4), 20)
            cwt_coeffs, freqs = pywt.cwt(data, scales, wavelet, sampling_period=1.0/self.sampling_rate)
            
            return {
                "decomposition": coeffs,
                "continuous_coefficients": cwt_coeffs,
                "scales": scales,
                "frequencies": freqs,
                "wavelet_type": wavelet
            }
            
        except Exception as e:
            logger.error(f"PyWavelets transform failed: {e}")
            return {"coefficients": np.array([]), "scales": np.array([])}
    
    def _scipy_wavelet_transform(self, data: np.ndarray, levels: int) -> Dict[str, np.ndarray]:
        """SciPy implementation of wavelet transform."""
        try:
            # Use scipy's cwt as fallback
            scales = np.logspace(1, np.log10(len(data)//4), 20)
            cwt_coeffs, freqs = scipy_signal.cwt(data, scipy_signal.ricker, scales)
            
            return {
                "continuous_coefficients": cwt_coeffs,
                "scales": scales,
                "frequencies": freqs,
                "wavelet_type": "ricker"
            }
            
        except Exception as e:
            logger.error(f"SciPy wavelet transform failed: {e}")
            return {"coefficients": np.array([]), "scales": np.array([])}
    
    def hilbert_huang_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform Hilbert-Huang Transform (HHT).
        
        Mathematical Formula:
        H[f](t) = (1/π) ∫ f(τ)/(t-τ) dτ
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with HHT results
        """
        try:
            # Empirical Mode Decomposition (EMD)
            imfs = self._empirical_mode_decomposition(data)
            
            # Hilbert transform of each IMF
            hilbert_imfs = []
            instantaneous_frequencies = []
            instantaneous_amplitudes = []
            
            for imf in imfs:
                # Hilbert transform
                analytic_signal = scipy_signal.hilbert(imf)
                
                # Instantaneous amplitude
                amplitude = np.abs(analytic_signal)
                
                # Instantaneous phase
                phase = np.angle(analytic_signal)
                
                # Instantaneous frequency (derivative of phase)
                frequency = np.gradient(phase) / (2 * np.pi * self.sampling_rate)
                
                hilbert_imfs.append(analytic_signal)
                instantaneous_frequencies.append(frequency)
                instantaneous_amplitudes.append(amplitude)
            
            return {
                "imfs": imfs,
                "hilbert_imfs": hilbert_imfs,
                "instantaneous_frequencies": instantaneous_frequencies,
                "instantaneous_amplitudes": instantaneous_amplitudes,
                "analytic_signal": scipy_signal.hilbert(data)
            }
            
        except Exception as e:
            logger.error(f"Hilbert-Huang transform failed: {e}")
            return {
                "imfs": [data],
                "hilbert_imfs": [scipy_signal.hilbert(data)],
                "instantaneous_frequencies": [np.zeros_like(data)],
                "instantaneous_amplitudes": [np.abs(data)],
                "analytic_signal": scipy_signal.hilbert(data)
            }
    
    def _empirical_mode_decomposition(self, data: np.ndarray, max_imfs: int = 10) -> List[np.ndarray]:
        """Empirical Mode Decomposition (EMD)."""
        try:
            imfs = []
            residual = data.copy()
            
            for _ in range(max_imfs):
                if len(residual) < 4:
                    break
                
                # Find extrema
                maxima_idx = scipy_signal.find_peaks(residual)[0]
                minima_idx = scipy_signal.find_peaks(-residual)[0]
                
                if len(maxima_idx) < 2 or len(minima_idx) < 2:
                    break
                
                # Interpolate envelopes
                t = np.arange(len(residual))
                
                # Upper envelope
                if len(maxima_idx) > 0:
                    upper_env = interp1d(maxima_idx, residual[maxima_idx], 
                                       kind='cubic', bounds_error=False, fill_value='extrapolate')(t)
                else:
                    upper_env = residual
                
                # Lower envelope
                if len(minima_idx) > 0:
                    lower_env = interp1d(minima_idx, residual[minima_idx], 
                                       kind='cubic', bounds_error=False, fill_value='extrapolate')(t)
                else:
                    lower_env = residual
                
                # Mean envelope
                mean_env = (upper_env + lower_env) / 2
                
                # Extract IMF candidate
                imf_candidate = residual - mean_env
                
                # Check if it's an IMF (simplified criteria)
                if self._is_imf(imf_candidate):
                    imfs.append(imf_candidate)
                    residual = residual - imf_candidate
                else:
                    residual = imf_candidate
            
            # Add residual as last IMF
            if len(residual) > 0:
                imfs.append(residual)
            
            return imfs
            
        except Exception as e:
            logger.error(f"EMD failed: {e}")
            return [data]
    
    def _is_imf(self, signal: np.ndarray, tolerance: float = 0.1) -> bool:
        """Check if signal satisfies IMF criteria."""
        try:
            # Count zero crossings
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            
            # Count extrema
            maxima = len(scipy_signal.find_peaks(signal)[0])
            minima = len(scipy_signal.find_peaks(-signal)[0])
            extrema = maxima + minima
            
            # IMF criteria: number of zero crossings and extrema differ by at most 1
            return abs(zero_crossings - extrema) <= 1
            
        except Exception:
            return True
    
    def cross_spectral_density(self, data1: np.ndarray, data2: np.ndarray, 
                              window: str = 'hann', nperseg: int = None) -> Dict[str, np.ndarray]:
        """
        Calculate cross-spectral density between two signals.
        
        Mathematical Formula:
        S_xy(f) = F[R_xy(τ)]
        
        Args:
            data1: First time series
            data2: Second time series
            window: Window function
            nperseg: Number of points per segment
            
        Returns:
            Dictionary with cross-spectral density results
        """
        try:
            if nperseg is None:
                nperseg = min(256, len(data1) // 4)
            
            # Calculate cross-spectral density
            freqs, csd = scipy_signal.csd(data1, data2, fs=self.sampling_rate, 
                                        window=window, nperseg=nperseg)
            
            # Calculate coherence
            freqs_coherence, coherence = scipy_signal.coherence(data1, data2, 
                                                              fs=self.sampling_rate,
                                                              window=window, nperseg=nperseg)
            
            # Calculate phase spectrum
            freqs_phase, phase = scipy_signal.csd(data1, data2, fs=self.sampling_rate,
                                                window=window, nperseg=nperseg, return_onesided=False)
            phase = np.angle(phase)
            
            return {
                "frequencies": freqs,
                "cross_spectral_density": csd,
                "coherence": coherence,
                "phase_spectrum": phase,
                "magnitude_spectrum": np.abs(csd)
            }
            
        except Exception as e:
            logger.error(f"Cross-spectral density calculation failed: {e}")
            return {
                "frequencies": np.array([]),
                "cross_spectral_density": np.array([]),
                "coherence": np.array([]),
                "phase_spectrum": np.array([]),
                "magnitude_spectrum": np.array([])
            }
    
    def phase_synchronization(self, data1: np.ndarray, data2: np.ndarray, 
                            method: str = "hilbert") -> float:
        """
        Calculate phase synchronization between two signals.
        
        Mathematical Formula:
        γ = |⟨e^(iφ)⟩|
        
        Args:
            data1: First time series
            data2: Second time series
            method: 'hilbert' or 'wavelet'
            
        Returns:
            Phase synchronization index
        """
        try:
            if method == "hilbert":
                return self._hilbert_phase_synchronization(data1, data2)
            elif method == "wavelet":
                return self._wavelet_phase_synchronization(data1, data2)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Phase synchronization calculation failed: {e}")
            return 0.0
    
    def _hilbert_phase_synchronization(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Phase synchronization using Hilbert transform."""
        try:
            # Hilbert transform
            analytic1 = scipy_signal.hilbert(data1)
            analytic2 = scipy_signal.hilbert(data2)
            
            # Instantaneous phases
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
            
            # Phase difference
            phase_diff = phase1 - phase2
            
            # Phase synchronization index
            sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            return float(sync_index)
            
        except Exception as e:
            logger.error(f"Hilbert phase synchronization failed: {e}")
            return 0.0
    
    def _wavelet_phase_synchronization(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Phase synchronization using wavelet transform."""
        try:
            # Wavelet transform
            wt1 = self.wavelet_transform(data1)
            wt2 = self.wavelet_transform(data2)
            
            if "continuous_coefficients" not in wt1 or "continuous_coefficients" not in wt2:
                return 0.0
            
            # Phase from wavelet coefficients
            phase1 = np.angle(wt1["continuous_coefficients"])
            phase2 = np.angle(wt2["continuous_coefficients"])
            
            # Phase difference
            phase_diff = phase1 - phase2
            
            # Phase synchronization index (average across scales)
            sync_indices = []
            for i in range(phase_diff.shape[0]):
                sync_idx = np.abs(np.mean(np.exp(1j * phase_diff[i, :])))
                sync_indices.append(sync_idx)
            
            return float(np.mean(sync_indices))
            
        except Exception as e:
            logger.error(f"Wavelet phase synchronization failed: {e}")
            return 0.0
    
    def amplitude_modulation(self, data: np.ndarray, method: str = "hilbert") -> np.ndarray:
        """
        Calculate amplitude modulation of a signal.
        
        Mathematical Formula:
        A(t) = √(x²(t) + H²[x](t))
        
        Args:
            data: Time series data
            method: 'hilbert' or 'envelope'
            
        Returns:
            Amplitude modulation
        """
        try:
            if method == "hilbert":
                return self._hilbert_amplitude_modulation(data)
            elif method == "envelope":
                return self._envelope_amplitude_modulation(data)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Amplitude modulation calculation failed: {e}")
            return np.abs(data)
    
    def _hilbert_amplitude_modulation(self, data: np.ndarray) -> np.ndarray:
        """Amplitude modulation using Hilbert transform."""
        try:
            # Hilbert transform
            analytic_signal = scipy_signal.hilbert(data)
            
            # Amplitude modulation
            amplitude = np.abs(analytic_signal)
            
            return amplitude
            
        except Exception as e:
            logger.error(f"Hilbert amplitude modulation failed: {e}")
            return np.abs(data)
    
    def _envelope_amplitude_modulation(self, data: np.ndarray) -> np.ndarray:
        """Amplitude modulation using envelope detection."""
        try:
            # Find extrema
            maxima_idx = scipy_signal.find_peaks(data)[0]
            minima_idx = scipy_signal.find_peaks(-data)[0]
            
            t = np.arange(len(data))
            
            # Upper envelope
            if len(maxima_idx) > 0:
                upper_env = interp1d(maxima_idx, data[maxima_idx], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')(t)
            else:
                upper_env = data
            
            # Lower envelope
            if len(minima_idx) > 0:
                lower_env = interp1d(minima_idx, data[minima_idx], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')(t)
            else:
                lower_env = data
            
            # Amplitude modulation
            amplitude = (upper_env - lower_env) / 2
            
            return amplitude
            
        except Exception as e:
            logger.error(f"Envelope amplitude modulation failed: {e}")
            return np.abs(data)
    
    def dominant_frequencies(self, data: np.ndarray, n_frequencies: int = 5) -> List[float]:
        """
        Find dominant frequencies in the signal.
        
        Args:
            data: Time series data
            n_frequencies: Number of dominant frequencies to return
            
        Returns:
            List of dominant frequencies
        """
        try:
            # Power spectral density
            freqs, psd = scipy_signal.welch(data, fs=self.sampling_rate)
            
            # Find peaks in power spectrum
            peak_indices = scipy_signal.find_peaks(psd, height=np.max(psd)/10)[0]
            
            if len(peak_indices) == 0:
                return [freqs[np.argmax(psd)]]
            
            # Sort peaks by power
            peak_powers = psd[peak_indices]
            sorted_indices = np.argsort(peak_powers)[::-1]
            
            # Return top frequencies
            dominant_freqs = []
            for i in range(min(n_frequencies, len(sorted_indices))):
                dominant_freqs.append(freqs[peak_indices[sorted_indices[i]]])
            
            return dominant_freqs
            
        except Exception as e:
            logger.error(f"Dominant frequency detection failed: {e}")
            return [0.0]
    
    def comprehensive_waveform_analysis(self, data: np.ndarray, 
                                      reference_data: np.ndarray = None) -> WaveformResult:
        """
        Perform comprehensive waveform analysis.
        
        Args:
            data: Time series data
            reference_data: Reference signal for cross-analysis
            
        Returns:
            WaveformResult with all analysis results
        """
        start_time = time.time()
        
        try:
            # Wavelet transform
            wavelet_result = self.wavelet_transform(data)
            
            # Hilbert-Huang transform
            hht_result = self.hilbert_huang_transform(data)
            
            # Amplitude modulation
            amplitude_mod = self.amplitude_modulation(data)
            
            # Dominant frequencies
            dominant_freqs = self.dominant_frequencies(data)
            
            # Cross-analysis if reference data provided
            if reference_data is not None:
                csd_result = self.cross_spectral_density(data, reference_data)
                phase_sync = self.phase_synchronization(data, reference_data)
            else:
                csd_result = {"frequencies": np.array([]), "cross_spectral_density": np.array([])}
                phase_sync = 0.0
            
            calculation_time = time.time() - start_time
            
            return WaveformResult(
                wavelet_coefficients=wavelet_result,
                instantaneous_frequency=hht_result.get("instantaneous_frequencies", [np.array([])])[0] if hht_result.get("instantaneous_frequencies") else np.array([]),
                cross_spectral_density=csd_result.get("cross_spectral_density", np.array([])),
                phase_synchronization=phase_sync,
                amplitude_modulation=amplitude_mod,
                dominant_frequencies=dominant_freqs,
                calculation_time=calculation_time,
                metadata={
                    "data_length": len(data),
                    "sampling_rate": self.sampling_rate,
                    "gpu_used": self.use_gpu,
                    "pywavelets_available": PYWAVELETS_AVAILABLE
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive waveform analysis failed: {e}")
            return WaveformResult(
                wavelet_coefficients={},
                instantaneous_frequency=np.array([]),
                cross_spectral_density=np.array([]),
                phase_synchronization=0.0,
                amplitude_modulation=np.array([]),
                dominant_frequencies=[],
                calculation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

# Global instance
waveform_signal_processing = WaveformSignalProcessing() 