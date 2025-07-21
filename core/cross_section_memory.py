#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CROSS-SECTION MEMORY - TICK VARIATION INTELLIGENCE
=====================================================

Cross-section memory system that integrates Kaprekar analysis with existing
soulprint registry for tick variation memory and profit mismatch detection.

Features:
- Tick variation memory mapping
- Profit mismatch detection and analysis
- Cross-session memory persistence
- Integration with soulprint registry
- Volatility-to-profit correlation tracking
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Import existing systems
try:
    from .soulprint_registry import SoulprintRegistry
    from .ghost_kaprekar_hash import generate_kaprekar_strategy_hash, generate_enhanced_kaprekar_hash
    from .ferris_tick_logic import process_tick, process_tick_with_metadata
    from .profit_cycle_allocator import allocate_profit_zone
    EXISTING_SYSTEMS_AVAILABLE = True
except ImportError:
    EXISTING_SYSTEMS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Some existing systems not available for cross-section memory")

logger = logging.getLogger(__name__)

@dataclass
class TickVariationMemory:
    """Memory entry for tick variation analysis."""
    timestamp: float
    price_tick: float
    kaprekar_index: int
    volatility_classification: str
    routing_signal: str
    profit_zone: str
    strategy_triggers: List[str]
    kaprekar_hash: str
    soulprint_hash: Optional[str] = None
    profit_result: Optional[float] = None
    memory_weight: float = 1.0
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProfitMismatchAnalysis:
    """Analysis of profit mismatches in tick variations."""
    timestamp: float
    tick_variation_id: str
    expected_profit: float
    actual_profit: float
    profit_delta: float
    mismatch_factor: float
    volatility_context: str
    strategy_performance: Dict[str, float]
    correlation_score: float
    recommendations: List[str]

class CrossSectionMemory:
    """Cross-section memory system for tick variation intelligence."""
    
    def __init__(self, memory_file: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize cross-section memory system."""
        self.memory_file = memory_file
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Memory storage
        self.tick_variations: List[TickVariationMemory] = []
        self.profit_mismatches: List[ProfitMismatchAnalysis] = []
        self.volatility_patterns: Dict[str, List[float]] = {}
        self.profit_correlations: Dict[str, float] = {}
        
        # Performance tracking
        self.memory_stats = {
            'total_variations': 0,
            'total_mismatches': 0,
            'avg_correlation_score': 0.0,
            'last_analysis': None
        }
        
        # Initialize existing systems
        if EXISTING_SYSTEMS_AVAILABLE:
            self.soulprint_registry = SoulprintRegistry()
        else:
            self.soulprint_registry = None
        
        logger.info(f"🧠 Cross-section memory initialized for session {self.session_id}")
    
    def record_tick_variation(self, price_tick: float, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a tick variation in memory.
        
        Args:
            price_tick: Float price value
            metadata: Optional metadata dictionary
            
        Returns:
            Memory entry ID
        """
        try:
            # Import functions directly to avoid circular imports
            from .tick_kaprekar_bridge import price_to_kaprekar_index, analyze_price_volatility
            from .ferris_tick_logic import process_tick
            from .profit_cycle_allocator import allocate_profit_zone
            from .ghost_kaprekar_hash import generate_kaprekar_strategy_hash
            
            # Process tick through Kaprekar analysis
            volatility_analysis = analyze_price_volatility(price_tick)
            routing_signal = process_tick(price_tick)
            
            # Allocate profit zone
            profit_allocation = allocate_profit_zone(price_tick)
            
            # Generate Kaprekar hash
            kaprekar_hash = generate_kaprekar_strategy_hash(price_tick)
            
            # Create memory entry
            memory_entry = TickVariationMemory(
                timestamp=time.time(),
                price_tick=price_tick,
                kaprekar_index=volatility_analysis['kaprekar_index'],
                volatility_classification=volatility_analysis['volatility_classification'],
                routing_signal=routing_signal,
                profit_zone=profit_allocation['profit_zone'],
                strategy_triggers=profit_allocation['strategy_triggers'],
                kaprekar_hash=kaprekar_hash,
                session_id=self.session_id,
                metadata=metadata or {}
            )
            
            # Register with soulprint registry if available
            if self.soulprint_registry:
                try:
                    soulprint_vector = {
                        'price_tick': price_tick,
                        'kaprekar_index': memory_entry.kaprekar_index,
                        'volatility_classification': memory_entry.volatility_classification,
                        'routing_signal': memory_entry.routing_signal,
                        'profit_zone': memory_entry.profit_zone
                    }
                    
                    soulprint_hash = self.soulprint_registry.register_soulprint(
                        vector=soulprint_vector,
                        strategy_id=f"kaprekar_{memory_entry.kaprekar_index}",
                        confidence=profit_allocation['allocation_confidence']
                    )
                    memory_entry.soulprint_hash = soulprint_hash
                    
                except Exception as e:
                    logger.warning(f"Failed to register with soulprint registry: {e}")
            
            # Store in memory
            self.tick_variations.append(memory_entry)
            self.memory_stats['total_variations'] += 1
            
            # Update volatility patterns
            vol_class = memory_entry.volatility_classification
            if vol_class not in self.volatility_patterns:
                self.volatility_patterns[vol_class] = []
            self.volatility_patterns[vol_class].append(price_tick)
            
            logger.debug(f"📝 Recorded tick variation: {price_tick} → {memory_entry.routing_signal}")
            
            return kaprekar_hash
            
        except Exception as e:
            logger.error(f"Error recording tick variation for {price_tick}: {e}")
            return ""
    
    def analyze_profit_mismatch(self, tick_variation_id: str, actual_profit: float, 
                               expected_profit: float) -> Optional[ProfitMismatchAnalysis]:
        """
        Analyze profit mismatch for a tick variation.
        
        Args:
            tick_variation_id: ID of the tick variation
            actual_profit: Actual profit achieved
            expected_profit: Expected profit based on analysis
            
        Returns:
            Profit mismatch analysis or None if not found
        """
        try:
            # Find the tick variation
            tick_variation = None
            for variation in self.tick_variations:
                if variation.kaprekar_hash == tick_variation_id:
                    tick_variation = variation
                    break
            
            if not tick_variation:
                logger.warning(f"Tick variation {tick_variation_id} not found")
                return None
            
            # Calculate mismatch metrics
            profit_delta = actual_profit - expected_profit
            mismatch_factor = abs(profit_delta) / max(abs(expected_profit), 0.001)
            
            # Analyze strategy performance
            strategy_performance = {}
            for strategy in tick_variation.strategy_triggers:
                # Calculate performance based on profit delta
                if profit_delta > 0:
                    strategy_performance[strategy] = 1.0  # Outperformed
                elif profit_delta < 0:
                    strategy_performance[strategy] = -1.0  # Underperformed
                else:
                    strategy_performance[strategy] = 0.0  # Met expectations
            
            # Calculate correlation score
            correlation_score = 1.0 - min(mismatch_factor, 1.0)
            
            # Generate recommendations
            recommendations = []
            if mismatch_factor > 0.5:
                recommendations.append("High profit mismatch detected - review strategy allocation")
            if tick_variation.volatility_classification == "non_convergent":
                recommendations.append("Non-convergent volatility - consider defensive positioning")
            if correlation_score < 0.3:
                recommendations.append("Low correlation - adjust volatility thresholds")
            
            # Create analysis
            analysis = ProfitMismatchAnalysis(
                timestamp=time.time(),
                tick_variation_id=tick_variation_id,
                expected_profit=expected_profit,
                actual_profit=actual_profit,
                profit_delta=profit_delta,
                mismatch_factor=mismatch_factor,
                volatility_context=tick_variation.volatility_classification,
                strategy_performance=strategy_performance,
                correlation_score=correlation_score,
                recommendations=recommendations
            )
            
            # Store analysis
            self.profit_mismatches.append(analysis)
            self.memory_stats['total_mismatches'] += 1
            
            # Update profit correlations
            self.profit_correlations[tick_variation_id] = correlation_score
            
            # Update tick variation with profit result
            tick_variation.profit_result = actual_profit
            
            logger.info(f"💰 Analyzed profit mismatch: {profit_delta:.4f} (factor: {mismatch_factor:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing profit mismatch: {e}")
            return None
    
    def get_volatility_profit_correlation(self, volatility_class: str) -> float:
        """
        Get profit correlation for a specific volatility classification.
        
        Args:
            volatility_class: Volatility classification to analyze
            
        Returns:
            Correlation score (0.0 to 1.0)
        """
        try:
            # Find all variations with this volatility class
            relevant_variations = [
                v for v in self.tick_variations 
                if v.volatility_classification == volatility_class and v.profit_result is not None
            ]
            
            if not relevant_variations:
                return 0.0
            
            # Calculate average correlation
            correlations = []
            for variation in relevant_variations:
                if variation.kaprekar_hash in self.profit_correlations:
                    correlations.append(self.profit_correlations[variation.kaprekar_hash])
            
            return sum(correlations) / len(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility profit correlation: {e}")
            return 0.0
    
    def analyze_tick_patterns(self, window_size: int = 50) -> Dict[str, Any]:
        """
        Analyze patterns in recent tick variations.
        
        Args:
            window_size: Number of recent variations to analyze
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            if len(self.tick_variations) < window_size:
                return {'error': 'Insufficient data for pattern analysis'}
            
            # Get recent variations
            recent_variations = self.tick_variations[-window_size:]
            
            # Analyze volatility distribution
            volatility_counts = {}
            signal_counts = {}
            profit_zone_counts = {}
            
            for variation in recent_variations:
                # Count volatility classifications
                vol_class = variation.volatility_classification
                volatility_counts[vol_class] = volatility_counts.get(vol_class, 0) + 1
                
                # Count routing signals
                signal = variation.routing_signal
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                # Count profit zones
                profit_zone = variation.profit_zone
                profit_zone_counts[profit_zone] = profit_zone_counts.get(profit_zone, 0) + 1
            
            # Calculate average correlation scores
            correlation_scores = []
            for variation in recent_variations:
                if variation.kaprekar_hash in self.profit_correlations:
                    correlation_scores.append(self.profit_correlations[variation.kaprekar_hash])
            
            avg_correlation = sum(correlation_scores) / len(correlation_scores) if correlation_scores else 0.0
            
            # Find dominant patterns
            dominant_volatility = max(volatility_counts.items(), key=lambda x: x[1])[0] if volatility_counts else 'unknown'
            dominant_signal = max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else 'unknown'
            dominant_zone = max(profit_zone_counts.items(), key=lambda x: x[1])[0] if profit_zone_counts else 'unknown'
            
            return {
                'total_variations': len(recent_variations),
                'volatility_distribution': volatility_counts,
                'signal_distribution': signal_counts,
                'profit_zone_distribution': profit_zone_counts,
                'dominant_volatility': dominant_volatility,
                'dominant_signal': dominant_signal,
                'dominant_profit_zone': dominant_zone,
                'average_correlation': avg_correlation,
                'analysis_window': window_size
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tick patterns: {e}")
            return {'error': str(e)}
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory contents and statistics.
        
        Returns:
            Dictionary with memory summary
        """
        try:
            # Calculate correlation statistics
            all_correlations = list(self.profit_correlations.values())
            avg_correlation = sum(all_correlations) / len(all_correlations) if all_correlations else 0.0
            
            # Update memory stats
            self.memory_stats['avg_correlation_score'] = avg_correlation
            self.memory_stats['last_analysis'] = time.time()
            
            return {
                'session_id': self.session_id,
                'memory_stats': self.memory_stats,
                'total_tick_variations': len(self.tick_variations),
                'total_profit_mismatches': len(self.profit_mismatches),
                'volatility_patterns_count': len(self.volatility_patterns),
                'profit_correlations_count': len(self.profit_correlations),
                'average_correlation_score': avg_correlation,
                'memory_file': self.memory_file
            }
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {'error': str(e)}
    
    def save_memory(self) -> bool:
        """
        Save memory to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.memory_file:
                return False
            
            # Prepare data for saving
            memory_data = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'tick_variations': [asdict(v) for v in self.tick_variations],
                'profit_mismatches': [asdict(a) for a in self.profit_mismatches],
                'volatility_patterns': self.volatility_patterns,
                'profit_correlations': self.profit_correlations,
                'memory_stats': self.memory_stats
            }
            
            # Save to file
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
            
            logger.info(f"💾 Memory saved to {self.memory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def load_memory(self) -> bool:
        """
        Load memory from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.memory_file:
                return False
            
            # Load from file
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            # Restore data
            self.session_id = memory_data.get('session_id', self.session_id)
            self.volatility_patterns = memory_data.get('volatility_patterns', {})
            self.profit_correlations = memory_data.get('profit_correlations', {})
            self.memory_stats = memory_data.get('memory_stats', self.memory_stats)
            
            # Restore tick variations
            self.tick_variations = []
            for var_data in memory_data.get('tick_variations', []):
                variation = TickVariationMemory(**var_data)
                self.tick_variations.append(variation)
            
            # Restore profit mismatches
            self.profit_mismatches = []
            for mismatch_data in memory_data.get('profit_mismatches', []):
                mismatch = ProfitMismatchAnalysis(**mismatch_data)
                self.profit_mismatches.append(mismatch)
            
            logger.info(f"📂 Memory loaded from {self.memory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return False


# Test function for validation
def test_cross_section_memory():
    """Test the cross-section memory system."""
    print("🧠 Testing Cross-Section Memory...")
    
    # Initialize memory system
    memory = CrossSectionMemory(session_id="test_session")
    
    # Test tick variation recording
    test_prices = [2045.29, 123.456, 9999.99, 1111.11]
    for price in test_prices:
        variation_id = memory.record_tick_variation(price)
        print(f"Recorded variation: {price} → {variation_id[:16]}...")
    
    # Test profit mismatch analysis
    if memory.tick_variations:
        first_variation = memory.tick_variations[0]
        analysis = memory.analyze_profit_mismatch(
            first_variation.kaprekar_hash, 
            actual_profit=0.05, 
            expected_profit=0.03
        )
        if analysis:
            print(f"Profit mismatch analyzed: {analysis.profit_delta:.4f}")
    
    # Test pattern analysis
    patterns = memory.analyze_tick_patterns(window_size=10)
    print(f"Pattern analysis: {patterns.get('dominant_volatility', 'unknown')}")
    
    # Test memory summary
    summary = memory.get_memory_summary()
    print(f"Memory summary: {summary['total_tick_variations']} variations recorded")
    
    print("✅ Cross-Section Memory test completed")


if __name__ == "__main__":
    test_cross_section_memory() 