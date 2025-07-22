#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 CROSS-CHAIN MATHEMATICAL BRIDGES - MULTI-CHAIN ANALYSIS
==========================================================

Advanced mathematical bridges for cross-chain analysis and coordination.

Features:
- Consensus Algorithms: Byzantine fault tolerance calculations
- Cross-Chain Correlation Matrices: Inter-chain dependencies
- Liquidity Flow Calculations: Across different chains
- Arbitrage Opportunity Detection: Multi-chain price differences
- Risk Propagation Models: How failures spread across chains
- GPU acceleration with automatic CPU fallback
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

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
class ChainData:
    """Data structure for chain information."""
    chain_id: str
    price: float
    volume: float
    liquidity: float
    volatility: float
    timestamp: float
    status: str  # 'active', 'inactive', 'faulty'

@dataclass
class CrossChainResult:
    """Result container for cross-chain analysis."""
    consensus_score: float
    correlation_matrix: np.ndarray
    liquidity_flow: Dict[str, float]
    arbitrage_opportunities: List[Dict[str, Any]]
    risk_propagation: Dict[str, float]
    network_connectivity: float
    calculation_time: float
    metadata: Dict[str, Any]

class CrossChainMathematicalBridges:
    """
    Advanced mathematical bridges for cross-chain analysis.
    
    Mathematical Foundations:
    - Byzantine Fault Tolerance: f < n/3 for consensus
    - Cross-Chain Correlation: ρ_ij = Cov(X_i, X_j) / (σ_i * σ_j)
    - Liquidity Flow: F_ij = L_i * (P_i - P_j) / D_ij
    - Arbitrage Detection: ΔP = |P_i - P_j| - (F_i + F_j)
    - Risk Propagation: R_i = Σ_j w_ij * R_j * exp(-λ * t)
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize cross-chain mathematical bridges."""
        self.use_gpu = use_gpu and USING_CUDA
        self.chains = {}  # Chain registry
        self.consensus_threshold = 0.67  # 2/3 majority
        self.arbitrage_threshold = 0.001  # 0.1% minimum spread
        self.risk_decay_rate = 0.1  # Risk propagation decay
        
        logger.info(f"🔗 Cross-Chain Mathematical Bridges initialized with {_backend}")
    
    def register_chain(self, chain_data: ChainData):
        """Register a chain in the system."""
        self.chains[chain_data.chain_id] = chain_data
    
    def byzantine_fault_tolerance(self, chain_ids: List[str], 
                                fault_tolerance_type: str = "pbft") -> Dict[str, float]:
        """
        Calculate Byzantine fault tolerance metrics.
        
        Mathematical Formula:
        - PBFT: f < n/3 for consensus
        - PoS: stake_weight > 2/3 for finality
        - DPoS: delegate_count > 2/3 for consensus
        
        Args:
            chain_ids: List of chain IDs to analyze
            fault_tolerance_type: 'pbft', 'pos', 'dpos'
            
        Returns:
            Dictionary with BFT metrics
        """
        try:
            n_chains = len(chain_ids)
            if n_chains < 3:
                return {"consensus_possible": False, "fault_tolerance": 0.0}
            
            # Calculate consensus thresholds
            if fault_tolerance_type == "pbft":
                max_faulty = (n_chains - 1) // 3
                consensus_threshold = (2 * n_chains + 1) // 3
            elif fault_tolerance_type == "pos":
                max_faulty = n_chains // 3
                consensus_threshold = (2 * n_chains) // 3
            elif fault_tolerance_type == "dpos":
                max_faulty = n_chains // 3
                consensus_threshold = (2 * n_chains) // 3
            else:
                raise ValueError(f"Unknown fault tolerance type: {fault_tolerance_type}")
            
            # Calculate actual consensus score
            active_chains = [cid for cid in chain_ids if cid in self.chains and 
                           self.chains[cid].status == 'active']
            active_count = len(active_chains)
            
            consensus_score = active_count / n_chains
            fault_tolerance = max_faulty / n_chains
            
            return {
                "consensus_possible": active_count >= consensus_threshold,
                "consensus_score": consensus_score,
                "fault_tolerance": fault_tolerance,
                "active_chains": active_count,
                "total_chains": n_chains,
                "max_faulty": max_faulty,
                "consensus_threshold": consensus_threshold
            }
            
        except Exception as e:
            logger.error(f"Byzantine fault tolerance calculation failed: {e}")
            return {"consensus_possible": False, "fault_tolerance": 0.0}
    
    def cross_chain_correlation_matrix(self, chain_ids: List[str], 
                                     metric: str = "price") -> np.ndarray:
        """
        Calculate cross-chain correlation matrix.
        
        Mathematical Formula:
        ρ_ij = Cov(X_i, X_j) / (σ_i * σ_j)
        
        Args:
            chain_ids: List of chain IDs
            metric: 'price', 'volume', 'volatility'
            
        Returns:
            Correlation matrix
        """
        try:
            n_chains = len(chain_ids)
            if n_chains < 2:
                return np.array([[1.0]])
            
            # Extract metric values for each chain
            metric_values = []
            valid_chains = []
            
            for chain_id in chain_ids:
                if chain_id in self.chains:
                    chain = self.chains[chain_id]
                    if metric == "price":
                        value = chain.price
                    elif metric == "volume":
                        value = chain.volume
                    elif metric == "volatility":
                        value = chain.volatility
                    else:
                        value = chain.price  # Default to price
                    
                    metric_values.append(value)
                    valid_chains.append(chain_id)
            
            if len(metric_values) < 2:
                return np.array([[1.0]])
            
            # Calculate correlation matrix
            metric_array = np.array(metric_values).reshape(-1, 1)
            
            # For single values, create a time series simulation
            if len(metric_array) == 1:
                # Simulate time series with noise
                time_series = np.random.normal(metric_array[0, 0], 
                                             metric_array[0, 0] * 0.1, 100)
                metric_array = time_series.reshape(-1, 1)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(metric_array.T)
            
            # Ensure positive definiteness
            correlation_matrix = self._ensure_positive_definite(correlation_matrix)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Cross-chain correlation calculation failed: {e}")
            return np.eye(len(chain_ids))
    
    def _ensure_positive_definite(self, matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive definite."""
        try:
            # Add small diagonal elements if needed
            eigenvalues = np.linalg.eigvals(matrix)
            min_eigenvalue = np.min(eigenvalues)
            
            if min_eigenvalue < epsilon:
                matrix = matrix + (epsilon - min_eigenvalue) * np.eye(matrix.shape[0])
            
            return matrix
            
        except Exception:
            return matrix
    
    def liquidity_flow_calculation(self, chain_ids: List[str]) -> Dict[str, float]:
        """
        Calculate liquidity flow between chains.
        
        Mathematical Formula:
        F_ij = L_i * (P_i - P_j) / D_ij
        
        Args:
            chain_ids: List of chain IDs
            
        Returns:
            Dictionary with liquidity flow values
        """
        try:
            liquidity_flows = {}
            
            for i, chain_id_i in enumerate(chain_ids):
                if chain_id_i not in self.chains:
                    continue
                
                chain_i = self.chains[chain_id_i]
                
                for j, chain_id_j in enumerate(chain_ids):
                    if i == j or chain_id_j not in self.chains:
                        continue
                    
                    chain_j = self.chains[chain_id_j]
                    
                    # Calculate price difference
                    price_diff = chain_i.price - chain_j.price
                    
                    # Calculate distance (simplified as 1 for now)
                    distance = 1.0
                    
                    # Calculate liquidity flow
                    liquidity_flow = chain_i.liquidity * price_diff / distance
                    
                    flow_key = f"{chain_id_i}_to_{chain_id_j}"
                    liquidity_flows[flow_key] = liquidity_flow
            
            return liquidity_flows
            
        except Exception as e:
            logger.error(f"Liquidity flow calculation failed: {e}")
            return {}
    
    def arbitrage_opportunity_detection(self, chain_ids: List[str], 
                                      min_spread: float = None) -> List[Dict[str, Any]]:
        """
        Detect arbitrage opportunities across chains.
        
        Mathematical Formula:
        ΔP = |P_i - P_j| - (F_i + F_j)
        
        Args:
            chain_ids: List of chain IDs
            min_spread: Minimum spread for arbitrage
            
        Returns:
            List of arbitrage opportunities
        """
        try:
            if min_spread is None:
                min_spread = self.arbitrage_threshold
            
            arbitrage_opportunities = []
            
            for i, chain_id_i in enumerate(chain_ids):
                if chain_id_i not in self.chains:
                    continue
                
                chain_i = self.chains[chain_id_i]
                
                for j, chain_id_j in enumerate(chain_ids):
                    if i >= j or chain_id_j not in self.chains:
                        continue
                    
                    chain_j = self.chains[chain_id_j]
                    
                    # Calculate price spread
                    price_spread = abs(chain_i.price - chain_j.price)
                    relative_spread = price_spread / min(chain_i.price, chain_j.price)
                    
                    # Calculate transaction fees (simplified)
                    fee_i = chain_i.price * 0.001  # 0.1% fee
                    fee_j = chain_j.price * 0.001  # 0.1% fee
                    total_fees = fee_i + fee_j
                    
                    # Calculate net profit
                    net_profit = price_spread - total_fees
                    profit_margin = net_profit / min(chain_i.price, chain_j.price)
                    
                    # Check if arbitrage is profitable
                    if relative_spread > min_spread and net_profit > 0:
                        opportunity = {
                            "chain_a": chain_id_i,
                            "chain_b": chain_id_j,
                            "price_a": chain_i.price,
                            "price_b": chain_j.price,
                            "price_spread": price_spread,
                            "relative_spread": relative_spread,
                            "total_fees": total_fees,
                            "net_profit": net_profit,
                            "profit_margin": profit_margin,
                            "direction": "buy_low_sell_high" if chain_i.price < chain_j.price else "buy_high_sell_low",
                            "timestamp": time.time()
                        }
                        arbitrage_opportunities.append(opportunity)
            
            # Sort by profit margin
            arbitrage_opportunities.sort(key=lambda x: x["profit_margin"], reverse=True)
            
            return arbitrage_opportunities
            
        except Exception as e:
            logger.error(f"Arbitrage opportunity detection failed: {e}")
            return []
    
    def risk_propagation_model(self, chain_ids: List[str], 
                             initial_risk: Dict[str, float] = None) -> Dict[str, float]:
        """
        Model risk propagation across chains.
        
        Mathematical Formula:
        R_i = Σ_j w_ij * R_j * exp(-λ * t)
        
        Args:
            chain_ids: List of chain IDs
            initial_risk: Initial risk values for chains
            
        Returns:
            Dictionary with propagated risk values
        """
        try:
            n_chains = len(chain_ids)
            if n_chains == 0:
                return {}
            
            # Initialize risk values
            if initial_risk is None:
                risk_values = {chain_id: 0.1 for chain_id in chain_ids}
            else:
                risk_values = initial_risk.copy()
                # Fill missing chains with default risk
                for chain_id in chain_ids:
                    if chain_id not in risk_values:
                        risk_values[chain_id] = 0.1
            
            # Create adjacency matrix based on correlations
            adjacency_matrix = np.zeros((n_chains, n_chains))
            
            for i, chain_id_i in enumerate(chain_ids):
                if chain_id_i not in self.chains:
                    continue
                
                for j, chain_id_j in enumerate(chain_ids):
                    if i == j or chain_id_j not in self.chains:
                        continue
                    
                    # Calculate correlation-based weight
                    chain_i = self.chains[chain_id_i]
                    chain_j = self.chains[chain_id_j]
                    
                    # Simple correlation based on price similarity
                    price_correlation = 1.0 - abs(chain_i.price - chain_j.price) / max(chain_i.price, chain_j.price)
                    adjacency_matrix[i, j] = max(0, price_correlation)
            
            # Normalize adjacency matrix
            row_sums = adjacency_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            adjacency_matrix = adjacency_matrix / row_sums[:, np.newaxis]
            
            # Risk propagation simulation
            max_iterations = 10
            current_risk = np.array([risk_values.get(chain_id, 0.1) for chain_id in chain_ids])
            
            for iteration in range(max_iterations):
                # Propagate risk
                new_risk = np.zeros(n_chains)
                
                for i in range(n_chains):
                    # Self-risk decay
                    self_decay = current_risk[i] * np.exp(-self.risk_decay_rate)
                    
                    # Risk from other chains
                    external_risk = 0.0
                    for j in range(n_chains):
                        if i != j:
                            external_risk += adjacency_matrix[i, j] * current_risk[j] * np.exp(-self.risk_decay_rate)
                    
                    new_risk[i] = self_decay + external_risk
                
                current_risk = new_risk
            
            # Return final risk values
            final_risk = {}
            for i, chain_id in enumerate(chain_ids):
                final_risk[chain_id] = float(current_risk[i])
            
            return final_risk
            
        except Exception as e:
            logger.error(f"Risk propagation model failed: {e}")
            return {chain_id: 0.1 for chain_id in chain_ids}
    
    def network_connectivity_analysis(self, chain_ids: List[str]) -> Dict[str, float]:
        """
        Analyze network connectivity between chains.
        
        Args:
            chain_ids: List of chain IDs
            
        Returns:
            Dictionary with connectivity metrics
        """
        try:
            n_chains = len(chain_ids)
            if n_chains < 2:
                return {"connectivity": 1.0, "clustering_coefficient": 1.0}
            
            # Create connectivity matrix
            connectivity_matrix = np.zeros((n_chains, n_chains))
            
            for i, chain_id_i in enumerate(chain_ids):
                if chain_id_i not in self.chains:
                    continue
                
                for j, chain_id_j in enumerate(chain_ids):
                    if i == j or chain_id_j not in self.chains:
                        continue
                    
                    # Check if chains are connected (simplified)
                    chain_i = self.chains[chain_id_i]
                    chain_j = self.chains[chain_id_j]
                    
                    # Connection based on price correlation
                    price_correlation = 1.0 - abs(chain_i.price - chain_j.price) / max(chain_i.price, chain_j.price)
                    connectivity_matrix[i, j] = 1.0 if price_correlation > 0.5 else 0.0
            
            # Calculate connectivity metrics
            total_possible_connections = n_chains * (n_chains - 1) / 2
            actual_connections = np.sum(connectivity_matrix) / 2
            connectivity = actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
            
            # Calculate clustering coefficient
            clustering_coefficient = self._calculate_clustering_coefficient(connectivity_matrix)
            
            return {
                "connectivity": connectivity,
                "clustering_coefficient": clustering_coefficient,
                "total_connections": int(actual_connections),
                "possible_connections": int(total_possible_connections)
            }
            
        except Exception as e:
            logger.error(f"Network connectivity analysis failed: {e}")
            return {"connectivity": 0.0, "clustering_coefficient": 0.0}
    
    def _calculate_clustering_coefficient(self, adjacency_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient of the network."""
        try:
            n_nodes = adjacency_matrix.shape[0]
            if n_nodes < 3:
                return 0.0
            
            # Calculate number of triangles
            triangles = 0
            triplets = 0
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    
                    for k in range(n_nodes):
                        if k == i or k == j:
                            continue
                        
                        # Check if i-j-k forms a triplet
                        if adjacency_matrix[i, j] > 0 and adjacency_matrix[j, k] > 0:
                            triplets += 1
                            
                            # Check if it forms a triangle
                            if adjacency_matrix[i, k] > 0:
                                triangles += 1
            
            # Clustering coefficient
            if triplets == 0:
                return 0.0
            
            return triangles / triplets
            
        except Exception:
            return 0.0
    
    def comprehensive_cross_chain_analysis(self, chain_ids: List[str]) -> CrossChainResult:
        """
        Perform comprehensive cross-chain analysis.
        
        Args:
            chain_ids: List of chain IDs to analyze
            
        Returns:
            CrossChainResult with all analysis results
        """
        start_time = time.time()
        
        try:
            # Byzantine fault tolerance
            bft_result = self.byzantine_fault_tolerance(chain_ids)
            consensus_score = bft_result.get("consensus_score", 0.0)
            
            # Cross-chain correlation
            correlation_matrix = self.cross_chain_correlation_matrix(chain_ids)
            
            # Liquidity flow
            liquidity_flow = self.liquidity_flow_calculation(chain_ids)
            
            # Arbitrage opportunities
            arbitrage_opportunities = self.arbitrage_opportunity_detection(chain_ids)
            
            # Risk propagation
            risk_propagation = self.risk_propagation_model(chain_ids)
            
            # Network connectivity
            connectivity_result = self.network_connectivity_analysis(chain_ids)
            network_connectivity = connectivity_result.get("connectivity", 0.0)
            
            calculation_time = time.time() - start_time
            
            return CrossChainResult(
                consensus_score=consensus_score,
                correlation_matrix=correlation_matrix,
                liquidity_flow=liquidity_flow,
                arbitrage_opportunities=arbitrage_opportunities,
                risk_propagation=risk_propagation,
                network_connectivity=network_connectivity,
                calculation_time=calculation_time,
                metadata={
                    "chain_count": len(chain_ids),
                    "active_chains": len([cid for cid in chain_ids if cid in self.chains]),
                    "gpu_used": self.use_gpu,
                    "bft_result": bft_result,
                    "connectivity_result": connectivity_result
                }
            )
            
        except Exception as e:
            logger.error(f"Comprehensive cross-chain analysis failed: {e}")
            return CrossChainResult(
                consensus_score=0.0,
                correlation_matrix=np.array([]),
                liquidity_flow={},
                arbitrage_opportunities=[],
                risk_propagation={},
                network_connectivity=0.0,
                calculation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

# Global instance
cross_chain_mathematical_bridges = CrossChainMathematicalBridges() 