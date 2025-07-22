#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ADVANCED ENTROPY CALCULATIONS - INFORMATION THEORY FOUNDATION
================================================================

Advanced entropy calculations for enhanced information theory analysis.

Features:
- Rényi Entropy: H_α = (1/(1-α)) * log(Σ p_i^α)
- Tsallis Entropy: S_q = (1/(q-1)) * (1 - Σ p_i^q)
- Fisher Information: I(θ) = E[(∂/∂θ log f(X|θ))²]
- Mutual Information: I(X;Y) = Σ P(x,y) * log(P(x,y)/(P(x)*P(y)))
- Transfer Entropy: T_Y→X = Σ p(x_{t+1}, x_t, y_t) * log(p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t))
- GPU acceleration with automatic CPU fallback
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy

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
class EntropyResult:
    """Result container for entropy calculations."""
    shannon_entropy: float
    renyi_entropy: Dict[float, float]
    tsallis_entropy: Dict[float, float]
    fisher_information: float
    mutual_information: float
    transfer_entropy: float
    calculation_time: float
    metadata: Dict[str, Any]

class AdvancedEntropyCalculations:
    """
    Advanced entropy calculations for information theory analysis.
    
    Mathematical Foundations:
    - Shannon Entropy: H = -Σ p_i * log2(p_i)
    - Rényi Entropy: H_α = (1/(1-α)) * log(Σ p_i^α)
    - Tsallis Entropy: S_q = (1/(q-1)) * (1 - Σ p_i^q)
    - Fisher Information: I(θ) = E[(∂/∂θ log f(X|θ))²]
    - Mutual Information: I(X;Y) = Σ P(x,y) * log(P(x,y)/(P(x)*P(y)))
    - Transfer Entropy: T_Y→X = Σ p(x_{t+1}, x_t, y_t) * log(p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t))
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize advanced entropy calculations."""
        self.use_gpu = use_gpu and USING_CUDA
        self.epsilon = 1e-12  # Small value to avoid log(0)
        self.default_alpha_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        self.default_q_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        
        logger.info(f"📊 Advanced Entropy Calculations initialized with {_backend}")
    
    def shannon_entropy(self, probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy.
        
        Mathematical Formula:
        H = -Σ p_i * log2(p_i)
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Shannon entropy value
        """
        try:
            if not probabilities:
                return 0.0
            
            # Convert to numpy array and ensure probabilities sum to 1
            prob_array = xp.array(probabilities)
            prob_array = xp.clip(prob_array, self.epsilon, 1.0)
            prob_array = prob_array / xp.sum(prob_array)
            
            # Calculate Shannon entropy
            entropy = -xp.sum(prob_array * xp.log2(prob_array + self.epsilon))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Shannon entropy calculation failed: {e}")
            return 0.0
    
    def renyi_entropy(self, probabilities: List[float], alpha: float = 2.0) -> float:
        """
        Calculate Rényi entropy.
        
        Mathematical Formula:
        H_α = (1/(1-α)) * log(Σ p_i^α)
        
        Args:
            probabilities: List of probability values
            alpha: Rényi parameter (α ≠ 1)
            
        Returns:
            Rényi entropy value
        """
        try:
            if alpha == 1.0:
                # Rényi entropy converges to Shannon entropy for α = 1
                return self.shannon_entropy(probabilities)
            
            if not probabilities:
                return 0.0
            
            # Convert to numpy array and normalize
            prob_array = xp.array(probabilities)
            prob_array = xp.clip(prob_array, self.epsilon, 1.0)
            prob_array = prob_array / xp.sum(prob_array)
            
            # Calculate Rényi entropy
            sum_p_alpha = xp.sum(prob_array ** alpha)
            if sum_p_alpha > 0:
                entropy = (1 / (1 - alpha)) * xp.log(sum_p_alpha)
            else:
                entropy = 0.0
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Rényi entropy calculation failed: {e}")
            return 0.0
    
    def tsallis_entropy(self, probabilities: List[float], q: float = 2.0) -> float:
        """
        Calculate Tsallis entropy.
        
        Mathematical Formula:
        S_q = (1/(q-1)) * (1 - Σ p_i^q)
        
        Args:
            probabilities: List of probability values
            q: Tsallis parameter (q ≠ 1)
            
        Returns:
            Tsallis entropy value
        """
        try:
            if q == 1.0:
                # Tsallis entropy converges to Shannon entropy for q = 1
                return self.shannon_entropy(probabilities)
            
            if not probabilities:
                return 0.0
            
            # Convert to numpy array and normalize
            prob_array = xp.array(probabilities)
            prob_array = xp.clip(prob_array, self.epsilon, 1.0)
            prob_array = prob_array / xp.sum(prob_array)
            
            # Calculate Tsallis entropy
            sum_p_q = xp.sum(prob_array ** q)
            entropy = (1 / (q - 1)) * (1 - sum_p_q)
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Tsallis entropy calculation failed: {e}")
            return 0.0
    
    def fisher_information(self, data: np.ndarray, parameter: str = "mean") -> float:
        """
        Calculate Fisher information.
        
        Mathematical Formula:
        I(θ) = E[(∂/∂θ log f(X|θ))²]
        
        Args:
            data: Data array
            parameter: Parameter type ('mean', 'variance')
            
        Returns:
            Fisher information value
        """
        try:
            if len(data) < 2:
                return 0.0
            
            data_array = xp.array(data)
            
            if parameter == "mean":
                # For normal distribution with known variance
                variance = xp.var(data_array)
                if variance > 0:
                    fisher_info = len(data_array) / variance
                else:
                    fisher_info = 0.0
                    
            elif parameter == "variance":
                # For normal distribution with known mean
                mean = xp.mean(data_array)
                variance = xp.var(data_array)
                if variance > 0:
                    fisher_info = len(data_array) / (2 * variance ** 2)
                else:
                    fisher_info = 0.0
                    
            else:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            return float(fisher_info)
            
        except Exception as e:
            logger.error(f"Fisher information calculation failed: {e}")
            return 0.0
    
    def mutual_information(self, data_x: np.ndarray, data_y: np.ndarray, 
                          bins: int = 10) -> float:
        """
        Calculate mutual information between two variables.
        
        Mathematical Formula:
        I(X;Y) = Σ P(x,y) * log(P(x,y)/(P(x)*P(y)))
        
        Args:
            data_x: First variable data
            data_y: Second variable data
            bins: Number of bins for histogram
            
        Returns:
            Mutual information value
        """
        try:
            if len(data_x) != len(data_y) or len(data_x) < 2:
                return 0.0
            
            # Create 2D histogram
            hist_2d, x_edges, y_edges = xp.histogram2d(data_x, data_y, bins=bins)
            
            # Normalize to get joint probability
            joint_prob = hist_2d / xp.sum(hist_2d)
            
            # Calculate marginal probabilities
            marginal_x = xp.sum(joint_prob, axis=1)
            marginal_y = xp.sum(joint_prob, axis=0)
            
            # Calculate mutual information
            mutual_info = 0.0
            
            for i in range(joint_prob.shape[0]):
                for j in range(joint_prob.shape[1]):
                    if (joint_prob[i, j] > 0 and 
                        marginal_x[i] > 0 and 
                        marginal_y[j] > 0):
                        
                        mutual_info += joint_prob[i, j] * xp.log2(
                            joint_prob[i, j] / (marginal_x[i] * marginal_y[j])
                        )
            
            return float(mutual_info)
            
        except Exception as e:
            logger.error(f"Mutual information calculation failed: {e}")
            return 0.0
    
    def transfer_entropy(self, data_x: np.ndarray, data_y: np.ndarray, 
                        lag: int = 1, bins: int = 5) -> float:
        """
        Calculate transfer entropy from Y to X.
        
        Mathematical Formula:
        T_Y→X = Σ p(x_{t+1}, x_t, y_t) * log(p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t))
        
        Args:
            data_x: Target variable data
            data_y: Source variable data
            lag: Time lag for transfer entropy
            bins: Number of bins for discretization
            
        Returns:
            Transfer entropy value
        """
        try:
            if len(data_x) != len(data_y) or len(data_x) < lag + 2:
                return 0.0
            
            # Discretize data
            x_discrete = xp.digitize(data_x, xp.linspace(xp.min(data_x), xp.max(data_x), bins))
            y_discrete = xp.digitize(data_y, xp.linspace(xp.min(data_y), xp.max(data_y), bins))
            
            # Create time series with lag
            x_t = x_discrete[lag:-1]
            x_t1 = x_discrete[lag+1:]
            y_t = y_discrete[:-lag-1]
            
            # Calculate conditional probabilities
            transfer_entropy = 0.0
            
            # Count occurrences
            for i in range(1, bins + 1):
                for j in range(1, bins + 1):
                    for k in range(1, bins + 1):
                        # Count p(x_{t+1}, x_t, y_t)
                        count_3d = xp.sum((x_t1 == i) & (x_t == j) & (y_t == k))
                        
                        # Count p(x_t, y_t)
                        count_2d = xp.sum((x_t == j) & (y_t == k))
                        
                        # Count p(x_t)
                        count_1d = xp.sum(x_t == j)
                        
                        # Count p(x_{t+1}, x_t)
                        count_2d_x = xp.sum((x_t1 == i) & (x_t == j))
                        
                        total = len(x_t)
                        
                        if (count_3d > 0 and count_2d > 0 and 
                            count_1d > 0 and count_2d_x > 0):
                            
                            # Calculate conditional probabilities
                            p_3d = count_3d / total
                            p_2d = count_2d / total
                            p_1d = count_1d / total
                            p_2d_x = count_2d_x / total
                            
                            # Calculate transfer entropy term
                            term = p_3d * xp.log2((p_3d * p_1d) / (p_2d * p_2d_x))
                            transfer_entropy += term
            
            return float(transfer_entropy)
            
        except Exception as e:
            logger.error(f"Transfer entropy calculation failed: {e}")
            return 0.0
    
    def conditional_entropy(self, data_x: np.ndarray, data_y: np.ndarray, 
                           bins: int = 10) -> float:
        """
        Calculate conditional entropy H(X|Y).
        
        Mathematical Formula:
        H(X|Y) = -Σ P(x,y) * log(P(x|y))
        
        Args:
            data_x: Target variable data
            data_y: Condition variable data
            bins: Number of bins for histogram
            
        Returns:
            Conditional entropy value
        """
        try:
            if len(data_x) != len(data_y) or len(data_x) < 2:
                return 0.0
            
            # Create 2D histogram
            hist_2d, x_edges, y_edges = xp.histogram2d(data_x, data_y, bins=bins)
            
            # Normalize to get joint probability
            joint_prob = hist_2d / xp.sum(hist_2d)
            
            # Calculate marginal probability of Y
            marginal_y = xp.sum(joint_prob, axis=0)
            
            # Calculate conditional entropy
            conditional_entropy = 0.0
            
            for i in range(joint_prob.shape[0]):
                for j in range(joint_prob.shape[1]):
                    if joint_prob[i, j] > 0 and marginal_y[j] > 0:
                        conditional_prob = joint_prob[i, j] / marginal_y[j]
                        conditional_entropy -= joint_prob[i, j] * xp.log2(conditional_prob)
            
            return float(conditional_entropy)
            
        except Exception as e:
            logger.error(f"Conditional entropy calculation failed: {e}")
            return 0.0
    
    def relative_entropy(self, p: List[float], q: List[float]) -> float:
        """
        Calculate relative entropy (KL divergence) D(P||Q).
        
        Mathematical Formula:
        D(P||Q) = Σ p_i * log(p_i/q_i)
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            Relative entropy value
        """
        try:
            if len(p) != len(q) or len(p) == 0:
                return 0.0
            
            # Convert to numpy arrays and normalize
            p_array = xp.array(p)
            q_array = xp.array(q)
            
            p_array = xp.clip(p_array, self.epsilon, 1.0)
            q_array = xp.clip(q_array, self.epsilon, 1.0)
            
            p_array = p_array / xp.sum(p_array)
            q_array = q_array / xp.sum(q_array)
            
            # Calculate relative entropy
            relative_entropy = xp.sum(p_array * xp.log2(p_array / q_array))
            
            return float(relative_entropy)
            
        except Exception as e:
            logger.error(f"Relative entropy calculation failed: {e}")
            return 0.0
    
    def comprehensive_entropy_analysis(self, data: np.ndarray, 
                                     reference_data: np.ndarray = None) -> EntropyResult:
        """
        Perform comprehensive entropy analysis.
        
        Args:
            data: Primary data array
            reference_data: Reference data for mutual information
            
        Returns:
            EntropyResult with all entropy measures
        """
        start_time = time.time()
        
        try:
            # Convert data to probabilities
            if len(data) > 0:
                hist, _ = xp.histogram(data, bins=min(20, len(data)//10))
                probabilities = hist / xp.sum(hist)
            else:
                probabilities = [1.0]
            
            # Shannon entropy
            shannon_ent = self.shannon_entropy(probabilities)
            
            # Rényi entropy for different α values
            renyi_ent = {}
            for alpha in self.default_alpha_values:
                renyi_ent[alpha] = self.renyi_entropy(probabilities, alpha)
            
            # Tsallis entropy for different q values
            tsallis_ent = {}
            for q in self.default_q_values:
                tsallis_ent[q] = self.tsallis_entropy(probabilities, q)
            
            # Fisher information
            fisher_info = self.fisher_information(data)
            
            # Mutual information (if reference data provided)
            if reference_data is not None:
                mutual_info = self.mutual_information(data, reference_data)
                transfer_ent = self.transfer_entropy(data, reference_data)
            else:
                mutual_info = 0.0
                transfer_ent = 0.0
            
            calculation_time = time.time() - start_time
            
            return EntropyResult(
                shannon_entropy=shannon_ent,
                renyi_entropy=renyi_ent,
                tsallis_entropy=tsallis_ent,
                fisher_information=fisher_info,
                mutual_information=mutual_info,
                transfer_entropy=transfer_ent,
                calculation_time=calculation_time,
                metadata={
                    "data_length": len(data),
                    "gpu_used": self.use_gpu,
                    "alpha_values": self.default_alpha_values,
                    "q_values": self.default_q_values
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive entropy analysis failed: {e}")
            return EntropyResult(
                shannon_entropy=0.0,
                renyi_entropy={},
                tsallis_entropy={},
                fisher_information=0.0,
                mutual_information=0.0,
                transfer_entropy=0.0,
                calculation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def entropy_based_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract entropy-based features from data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with entropy-based features
        """
        try:
            features = {}
            
            # Basic entropy measures
            if len(data) > 0:
                hist, _ = xp.histogram(data, bins=min(20, len(data)//10))
                probabilities = hist / xp.sum(hist)
                
                features["shannon_entropy"] = self.shannon_entropy(probabilities)
                features["renyi_entropy_alpha_2"] = self.renyi_entropy(probabilities, 2.0)
                features["tsallis_entropy_q_2"] = self.tsallis_entropy(probabilities, 2.0)
                features["fisher_information"] = self.fisher_information(data)
                
                # Entropy ratios
                if features["shannon_entropy"] > 0:
                    features["renyi_shannon_ratio"] = features["renyi_entropy_alpha_2"] / features["shannon_entropy"]
                    features["tsallis_shannon_ratio"] = features["tsallis_entropy_q_2"] / features["shannon_entropy"]
                else:
                    features["renyi_shannon_ratio"] = 1.0
                    features["tsallis_shannon_ratio"] = 1.0
                
                # Entropy stability
                features["entropy_stability"] = 1.0 - abs(features["renyi_entropy_alpha_2"] - features["shannon_entropy"])
            
            return features
            
        except Exception as e:
            logger.error(f"Entropy-based feature extraction failed: {e}")
            return {
                "shannon_entropy": 0.0,
                "renyi_entropy_alpha_2": 0.0,
                "tsallis_entropy_q_2": 0.0,
                "fisher_information": 0.0,
                "renyi_shannon_ratio": 1.0,
                "tsallis_shannon_ratio": 1.0,
                "entropy_stability": 0.0
            }

# Global instance
advanced_entropy_calculations = AdvancedEntropyCalculations() 