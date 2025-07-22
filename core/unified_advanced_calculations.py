#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 UNIFIED ADVANCED CALCULATIONS - COMPREHENSIVE MATHEMATICAL ANALYSIS
=====================================================================

Unified interface for all advanced mathematical calculations.

Features:
- Integration of all advanced calculation modules
- Comprehensive mathematical analysis pipeline
- Cross-module data sharing and optimization
- GPU acceleration with automatic CPU fallback
- Real-time calculation orchestration
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

# Import all advanced calculation modules
try:
    from .advanced_statistical_calculations import AdvancedStatisticalCalculations, FractalResult
    from .waveform_signal_processing import WaveformSignalProcessing, WaveformResult
    from .cross_chain_mathematical_bridges import CrossChainMathematicalBridges, CrossChainResult, ChainData
    from .advanced_entropy_calculations import AdvancedEntropyCalculations, EntropyResult
    from .temporal_causal_analysis import TemporalCausalAnalysis, CausalResult
    ALL_MODULES_AVAILABLE = True
except ImportError as e:
    ALL_MODULES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Some advanced calculation modules not available: {e}")

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
class UnifiedAnalysisResult:
    """Comprehensive result container for unified analysis."""
    # Statistical Analysis
    fractal_analysis: Optional[FractalResult]
    # Waveform Analysis
    waveform_analysis: Optional[WaveformResult]
    # Cross-Chain Analysis
    cross_chain_analysis: Optional[CrossChainResult]
    # Entropy Analysis
    entropy_analysis: Optional[EntropyResult]
    # Temporal Analysis
    temporal_analysis: Optional[CausalResult]
    # Unified Features
    unified_features: Dict[str, float]
    # Performance Metrics
    total_calculation_time: float
    module_calculation_times: Dict[str, float]
    # Metadata
    metadata: Dict[str, Any]

class UnifiedAdvancedCalculations:
    """
    Unified interface for all advanced mathematical calculations.
    
    This class orchestrates all the advanced calculation modules and provides
    a comprehensive analysis pipeline for complex mathematical operations.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize unified advanced calculations."""
        self.use_gpu = use_gpu and USING_CUDA
        
        # Initialize all calculation modules
        if ALL_MODULES_AVAILABLE:
            self.statistical_calc = AdvancedStatisticalCalculations(use_gpu)
            self.waveform_calc = WaveformSignalProcessing(use_gpu)
            self.cross_chain_calc = CrossChainMathematicalBridges(use_gpu)
            self.entropy_calc = AdvancedEntropyCalculations(use_gpu)
            self.temporal_calc = TemporalCausalAnalysis(use_gpu)
        else:
            self.statistical_calc = None
            self.waveform_calc = None
            self.cross_chain_calc = None
            self.entropy_calc = None
            self.temporal_calc = None
        
        # Performance tracking
        self.analysis_count = 0
        self.total_time = 0.0
        
        logger.info(f"🧮 Unified Advanced Calculations initialized with {_backend}")
        if not ALL_MODULES_AVAILABLE:
            logger.warning("⚠️ Running with limited functionality due to missing modules")
    
    def comprehensive_analysis(self, data: np.ndarray, 
                             reference_data: np.ndarray = None,
                             chain_data: List[ChainData] = None,
                             analysis_types: List[str] = None) -> UnifiedAnalysisResult:
        """
        Perform comprehensive analysis using all available modules.
        
        Args:
            data: Primary data array
            reference_data: Reference data for cross-analysis
            chain_data: Chain data for cross-chain analysis
            analysis_types: List of analysis types to perform
            
        Returns:
            UnifiedAnalysisResult with all analysis results
        """
        start_time = time.time()
        module_times = {}
        
        try:
            # Default analysis types if none specified
            if analysis_types is None:
                analysis_types = ["statistical", "waveform", "entropy", "temporal"]
                if chain_data is not None:
                    analysis_types.append("cross_chain")
            
            # Initialize result containers
            fractal_analysis = None
            waveform_analysis = None
            cross_chain_analysis = None
            entropy_analysis = None
            temporal_analysis = None
            
            # Statistical Analysis
            if "statistical" in analysis_types and self.statistical_calc:
                module_start = time.time()
                fractal_analysis = self.statistical_calc.comprehensive_statistical_analysis(data)
                module_times["statistical"] = time.time() - module_start
            
            # Waveform Analysis
            if "waveform" in analysis_types and self.waveform_calc:
                module_start = time.time()
                waveform_analysis = self.waveform_calc.comprehensive_waveform_analysis(data, reference_data)
                module_times["waveform"] = time.time() - module_start
            
            # Cross-Chain Analysis
            if "cross_chain" in analysis_types and self.cross_chain_calc and chain_data:
                module_start = time.time()
                # Register chain data
                for chain in chain_data:
                    self.cross_chain_calc.register_chain(chain)
                
                chain_ids = [chain.chain_id for chain in chain_data]
                cross_chain_analysis = self.cross_chain_calc.comprehensive_cross_chain_analysis(chain_ids)
                module_times["cross_chain"] = time.time() - module_start
            
            # Entropy Analysis
            if "entropy" in analysis_types and self.entropy_calc:
                module_start = time.time()
                entropy_analysis = self.entropy_calc.comprehensive_entropy_analysis(data, reference_data)
                module_times["entropy"] = time.time() - module_start
            
            # Temporal Analysis
            if "temporal" in analysis_types and self.temporal_calc:
                module_start = time.time()
                temporal_analysis = self.temporal_calc.comprehensive_temporal_analysis(data, reference_data)
                module_times["temporal"] = time.time() - module_start
            
            # Generate unified features
            unified_features = self._generate_unified_features(
                data, reference_data, fractal_analysis, waveform_analysis,
                cross_chain_analysis, entropy_analysis, temporal_analysis
            )
            
            total_calculation_time = time.time() - start_time
            self.analysis_count += 1
            self.total_time += total_calculation_time
            
            return UnifiedAnalysisResult(
                fractal_analysis=fractal_analysis,
                waveform_analysis=waveform_analysis,
                cross_chain_analysis=cross_chain_analysis,
                entropy_analysis=entropy_analysis,
                temporal_analysis=temporal_analysis,
                unified_features=unified_features,
                total_calculation_time=total_calculation_time,
                module_calculation_times=module_times,
                metadata={
                    "analysis_count": self.analysis_count,
                    "gpu_used": self.use_gpu,
                    "modules_available": ALL_MODULES_AVAILABLE,
                    "analysis_types": analysis_types,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return UnifiedAnalysisResult(
                fractal_analysis=None,
                waveform_analysis=None,
                cross_chain_analysis=None,
                entropy_analysis=None,
                temporal_analysis=None,
                unified_features={},
                total_calculation_time=time.time() - start_time,
                module_calculation_times=module_times,
                metadata={"error": str(e)}
            )
    
    def _generate_unified_features(self, data: np.ndarray, reference_data: np.ndarray,
                                 fractal_analysis: FractalResult,
                                 waveform_analysis: WaveformResult,
                                 cross_chain_analysis: CrossChainResult,
                                 entropy_analysis: EntropyResult,
                                 temporal_analysis: CausalResult) -> Dict[str, float]:
        """Generate unified features from all analysis results."""
        try:
            features = {}
            
            # Basic data features
            if len(data) > 0:
                features["data_length"] = float(len(data))
                features["data_mean"] = float(np.mean(data))
                features["data_std"] = float(np.std(data))
                features["data_range"] = float(np.max(data) - np.min(data))
            
            # Fractal features
            if fractal_analysis:
                features["fractal_dimension"] = fractal_analysis.fractal_dimension
                features["hurst_exponent"] = fractal_analysis.hurst_exponent
                features["lyapunov_exponent"] = fractal_analysis.lyapunov_exponent
                features["correlation_dimension"] = fractal_analysis.correlation_dimension
            
            # Waveform features
            if waveform_analysis:
                features["phase_synchronization"] = waveform_analysis.phase_synchronization
                features["dominant_frequency_count"] = float(len(waveform_analysis.dominant_frequencies))
                if waveform_analysis.dominant_frequencies:
                    features["dominant_frequency_max"] = max(waveform_analysis.dominant_frequencies)
            
            # Cross-chain features
            if cross_chain_analysis:
                features["consensus_score"] = cross_chain_analysis.consensus_score
                features["network_connectivity"] = cross_chain_analysis.network_connectivity
                features["arbitrage_opportunity_count"] = float(len(cross_chain_analysis.arbitrage_opportunities))
            
            # Entropy features
            if entropy_analysis:
                features["shannon_entropy"] = entropy_analysis.shannon_entropy
                features["fisher_information"] = entropy_analysis.fisher_information
                features["mutual_information"] = entropy_analysis.mutual_information
                features["transfer_entropy"] = entropy_analysis.transfer_entropy
            
            # Temporal features
            if temporal_analysis:
                features["granger_causality"] = temporal_analysis.granger_causality.get("y_to_x", 0.0)
                features["dtw_distance"] = temporal_analysis.dtw_distance
                features["regime_count"] = float(len(set(temporal_analysis.regime_states)))
                features["hidden_state_count"] = float(len(set(temporal_analysis.hidden_states)))
            
            # Composite features
            features["complexity_score"] = self._calculate_complexity_score(features)
            features["stability_score"] = self._calculate_stability_score(features)
            features["predictability_score"] = self._calculate_predictability_score(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Unified feature generation failed: {e}")
            return {}
    
    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate complexity score from features."""
        try:
            complexity_factors = [
                features.get("fractal_dimension", 1.0),
                features.get("hurst_exponent", 0.5),
                features.get("shannon_entropy", 0.0),
                features.get("dominant_frequency_count", 1.0),
                features.get("regime_count", 1.0)
            ]
            
            # Normalize and combine
            complexity_score = np.mean(complexity_factors)
            return float(complexity_score)
            
        except Exception:
            return 0.5
    
    def _calculate_stability_score(self, features: Dict[str, float]) -> float:
        """Calculate stability score from features."""
        try:
            stability_factors = [
                1.0 - abs(features.get("lyapunov_exponent", 0.0)),
                features.get("consensus_score", 0.0),
                features.get("network_connectivity", 0.0),
                1.0 - features.get("volatility_clustering", 0.0) if "volatility_clustering" in features else 0.5
            ]
            
            # Normalize and combine
            stability_score = np.mean(stability_factors)
            return float(stability_score)
            
        except Exception:
            return 0.5
    
    def _calculate_predictability_score(self, features: Dict[str, float]) -> float:
        """Calculate predictability score from features."""
        try:
            predictability_factors = [
                1.0 - features.get("shannon_entropy", 0.0) / 10.0,  # Normalize entropy
                features.get("granger_causality", 0.0),
                1.0 - features.get("dtw_distance", 0.0) / 100.0,  # Normalize DTW
                features.get("fisher_information", 0.0) / 100.0  # Normalize Fisher info
            ]
            
            # Normalize and combine
            predictability_score = np.mean(predictability_factors)
            return float(max(0.0, min(1.0, predictability_score)))
            
        except Exception:
            return 0.5
    
    def real_time_analysis(self, data_stream: List[np.ndarray], 
                          window_size: int = 100) -> List[UnifiedAnalysisResult]:
        """
        Perform real-time analysis on streaming data.
        
        Args:
            data_stream: List of data arrays
            window_size: Size of analysis window
            
        Returns:
            List of analysis results
        """
        try:
            results = []
            
            for i in range(len(data_stream)):
                # Create analysis window
                start_idx = max(0, i - window_size + 1)
                window_data = np.concatenate(data_stream[start_idx:i+1])
                
                # Perform analysis on window
                result = self.comprehensive_analysis(window_data)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Real-time analysis failed: {e}")
            return []
    
    def batch_analysis(self, data_batch: List[np.ndarray], 
                      batch_size: int = 10) -> List[UnifiedAnalysisResult]:
        """
        Perform batch analysis on multiple datasets.
        
        Args:
            data_batch: List of data arrays
            batch_size: Size of processing batches
            
        Returns:
            List of analysis results
        """
        try:
            results = []
            
            for i in range(0, len(data_batch), batch_size):
                batch = data_batch[i:i+batch_size]
                
                for data in batch:
                    result = self.comprehensive_analysis(data)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the unified analysis system."""
        try:
            avg_time = self.total_time / self.analysis_count if self.analysis_count > 0 else 0.0
            
            return {
                "total_analyses": self.analysis_count,
                "total_time": self.total_time,
                "average_time": avg_time,
                "gpu_used": self.use_gpu,
                "modules_available": ALL_MODULES_AVAILABLE,
                "modules_loaded": {
                    "statistical": self.statistical_calc is not None,
                    "waveform": self.waveform_calc is not None,
                    "cross_chain": self.cross_chain_calc is not None,
                    "entropy": self.entropy_calc is not None,
                    "temporal": self.temporal_calc is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def export_analysis_report(self, result: UnifiedAnalysisResult, 
                             format: str = "json") -> str:
        """
        Export analysis result to various formats.
        
        Args:
            result: Analysis result to export
            format: Export format ('json', 'csv', 'summary')
            
        Returns:
            Formatted report string
        """
        try:
            if format == "json":
                import json
                return json.dumps({
                    "unified_features": result.unified_features,
                    "total_calculation_time": result.total_calculation_time,
                    "module_calculation_times": result.module_calculation_times,
                    "metadata": result.metadata
                }, indent=2)
            
            elif format == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(["Feature", "Value"])
                
                # Write features
                for key, value in result.unified_features.items():
                    writer.writerow([key, value])
                
                return output.getvalue()
            
            elif format == "summary":
                summary = f"""
🧮 UNIFIED ADVANCED CALCULATIONS REPORT
=======================================

📊 Analysis Summary:
- Total Calculation Time: {result.total_calculation_time:.3f}s
- Analysis Count: {result.metadata.get('analysis_count', 0)}
- GPU Used: {result.metadata.get('gpu_used', False)}

🔬 Key Features:
- Complexity Score: {result.unified_features.get('complexity_score', 0.0):.3f}
- Stability Score: {result.unified_features.get('stability_score', 0.0):.3f}
- Predictability Score: {result.unified_features.get('predictability_score', 0.0):.3f}

⏱️ Module Performance:
"""
                for module, time_taken in result.module_calculation_times.items():
                    summary += f"- {module}: {time_taken:.3f}s\n"
                
                return summary
            
            else:
                raise ValueError(f"Unknown export format: {format}")
                
        except Exception as e:
            logger.error(f"Export analysis report failed: {e}")
            return f"Export failed: {str(e)}"

# Global instance
unified_advanced_calculations = UnifiedAdvancedCalculations() 