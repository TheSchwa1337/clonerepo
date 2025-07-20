#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR ANALYZER - ENTROPY CLASSIFICATION MODULE
===================================================

Implements Kaprekar's Constant (6174) logic for entropy classification
and signal stability analysis in Schwabot trading system.

Features:
- 4-digit number convergence analysis
- Entropy scoring (1-7 steps to 6174)
- Hash fragment analysis
- Signal stability classification
- Cross-system integration with existing Schwabot components
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class KaprekarResult:
    """Result of Kaprekar analysis."""
    input_number: int
    steps_to_converge: int
    convergence_path: List[int]
    entropy_class: str
    stability_score: float
    is_convergent: bool
    timestamp: datetime
    hash_fragment: str

class KaprekarAnalyzer:
    """Kaprekar's Constant analyzer for entropy classification."""
    
    def __init__(self):
        self.KAPREKAR_CONSTANT = 6174
        self.MAX_STEPS = 7
        self.REJECT_THRESHOLD = 99
        
        # Entropy classification thresholds
        self.ENTROPY_CLASSES = {
            1: "ULTRA_STABLE",
            2: "STABLE", 
            3: "MODERATE",
            4: "ACTIVE",
            5: "VOLATILE",
            6: "HIGH_VOLATILITY",
            7: "EXTREME_VOLATILITY"
        }
        
        # Strategy mapping based on Kaprekar steps
        self.STRATEGY_MAPPING = {
            1: "conservative_hold",
            2: "moderate_buy", 
            3: "aggressive_buy",
            4: "volatility_play",
            5: "momentum_follow",
            6: "breakout_trade",
            7: "swing_trading"
        }
        
        # Performance tracking
        self.analysis_count = 0
        self.convergent_count = 0
        self.chaotic_count = 0
        
        logger.info("Kaprekar Analyzer initialized")

    def kaprekar_steps(self, n: int, target: int = 6174) -> int:
        """
        Calculate steps to reach Kaprekar's constant from a 4-digit number.
        
        Args:
            n: 4-digit number to analyze
            target: Target constant (default 6174)
            
        Returns:
            Number of steps to converge, or 99 if chaotic
        """
        if n < 1000 or n > 9999:
            return self.REJECT_THRESHOLD
            
        def to_digits(x: int) -> List[str]:
            return sorted(f"{x:04d}")
            
        steps = 0
        seen = set()
        current = n
        
        while current != target:
            digits = to_digits(current)
            small = int("".join(digits))
            large = int("".join(digits[::-1]))
            current = large - small
            
            if current in seen or current == 0 or steps > self.MAX_STEPS:
                return self.REJECT_THRESHOLD
                
            seen.add(current)
            steps += 1
            
        return steps

    def analyze_kaprekar(self, n: int, hash_fragment: str = "") -> KaprekarResult:
        """
        Complete Kaprekar analysis of a 4-digit number.
        
        Args:
            n: 4-digit number to analyze
            hash_fragment: Original hash fragment for reference
            
        Returns:
            KaprekarResult with full analysis
        """
        self.analysis_count += 1
        steps = self.kaprekar_steps(n)
        
        # Determine entropy class
        if steps <= 7:
            entropy_class = self.ENTROPY_CLASSES.get(steps, "UNKNOWN")
            is_convergent = True
            stability_score = max(0.1, 1.0 - (steps / 7.0))
            self.convergent_count += 1
        else:
            entropy_class = "CHAOTIC"
            is_convergent = False
            stability_score = 0.0
            self.chaotic_count += 1
            
        # Generate convergence path
        convergence_path = self._generate_convergence_path(n)
        
        return KaprekarResult(
            input_number=n,
            steps_to_converge=steps,
            convergence_path=convergence_path,
            entropy_class=entropy_class,
            stability_score=stability_score,
            is_convergent=is_convergent,
            timestamp=datetime.now(),
            hash_fragment=hash_fragment
        )

    def _generate_convergence_path(self, n: int) -> List[int]:
        """Generate the full convergence path to 6174."""
        path = [n]
        current = n
        
        while current != 6174 and len(path) <= 7:
            digits = sorted(f"{current:04d}")
            small = int("".join(digits))
            large = int("".join(digits[::-1]))
            current = large - small
            path.append(current)
            
        return path

    def analyze_hash_fragment(self, hash_fragment: str) -> KaprekarResult:
        """
        Analyze a hash fragment for Kaprekar convergence.
        
        Args:
            hash_fragment: Hex string fragment
            
        Returns:
            KaprekarResult for the 4-digit segment
        """
        try:
            # Extract 4-digit number from hash
            hex_val = hash_fragment[:4]
            number = int(hex_val, 16) % 10000
            return self.analyze_kaprekar(number, hash_fragment)
        except (ValueError, IndexError):
            return KaprekarResult(
                input_number=0,
                steps_to_converge=self.REJECT_THRESHOLD,
                convergence_path=[],
                entropy_class="INVALID",
                stability_score=0.0,
                is_convergent=False,
                timestamp=datetime.now(),
                hash_fragment=hash_fragment
            )

    def get_strategy_recommendation(self, kaprekar_result: KaprekarResult) -> str:
        """
        Get strategy recommendation based on Kaprekar analysis.
        
        Args:
            kaprekar_result: Result from Kaprekar analysis
            
        Returns:
            Recommended strategy name
        """
        if not kaprekar_result.is_convergent:
            return "standby_or_retry"
            
        return self.STRATEGY_MAPPING.get(
            kaprekar_result.steps_to_converge, 
            "neutral_wait"
        )

    def get_confidence_boost(self, kaprekar_result: KaprekarResult) -> float:
        """
        Calculate confidence boost from Kaprekar analysis.
        
        Args:
            kaprekar_result: Result from Kaprekar analysis
            
        Returns:
            Confidence boost value (0.0 to 1.0)
        """
        if not kaprekar_result.is_convergent:
            return -0.2  # Penalty for chaotic signals
            
        # Fast convergence = higher confidence
        if kaprekar_result.steps_to_converge <= 2:
            return 0.3
        elif kaprekar_result.steps_to_converge <= 4:
            return 0.2
        else:
            return 0.1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the Kaprekar analyzer."""
        total = self.analysis_count
        if total == 0:
            return {
                "total_analyses": 0,
                "convergence_rate": 0.0,
                "chaotic_rate": 0.0,
                "average_stability": 0.0
            }
            
        return {
            "total_analyses": total,
            "convergence_rate": self.convergent_count / total,
            "chaotic_rate": self.chaotic_count / total,
            "average_stability": (self.convergent_count * 0.7) / total
        }

# Global instance
kaprekar_analyzer = KaprekarAnalyzer() 