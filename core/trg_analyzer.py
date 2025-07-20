#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 TRG ANALYZER - TECHNICAL RESONANCE GRID
=========================================

Integrates Kaprekar entropy analysis with real-time technical indicators
for comprehensive trading signal classification.

Features:
- RSI + Kaprekar integration
- Price pole mapping
- Phantom band analysis
- Technical resonance scoring
- Real-time signal classification
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

# Import Kaprekar analyzer
try:
    from .mathlib.kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult
    KAPREKAR_AVAILABLE = True
except ImportError:
    KAPREKAR_AVAILABLE = False
    logging.warning("Kaprekar analyzer not available")

logger = logging.getLogger(__name__)

@dataclass
class TRGSnapshot:
    """Technical Resonance Grid snapshot."""
    kcs: int  # Kaprekar Collapse Score
    rsi: float
    price: float
    pole_range: Tuple[float, float]
    phantom_delta: float
    asset: str
    timestamp: float
    hash_fragment: str = ""

@dataclass
class TRGResult:
    """TRG analysis result."""
    signal_class: str
    confidence: float
    risk_level: str
    recommended_action: str
    technical_context: Dict[str, Any]
    kaprekar_result: Optional[KaprekarResult] = None

class TRGAnalyzer:
    """Technical Resonance Grid analyzer."""
    
    def __init__(self):
        # Initialize Kaprekar analyzer
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_AVAILABLE else None
        
        # RSI phase bands
        self.RSI_PHASES = {
            "OVERSOLD": (0, 30),
            "LOW": (30, 45),
            "NEUTRAL": (45, 55),
            "HIGH": (55, 70),
            "OVERBOUGHT": (70, 100)
        }
        
        # Signal classification rules
        self.SIGNAL_RULES = {
            "btc_long_entry": {
                "kcs_range": (1, 3),
                "rsi_range": (20, 35),
                "phantom_delta": "positive",
                "confidence_threshold": 0.7
            },
            "usdc_exit_trigger": {
                "kcs_range": (5, 7),
                "rsi_range": (70, 100),
                "phantom_delta": "any",
                "confidence_threshold": 0.6
            },
            "phantom_band_swing": {
                "kcs_range": (3, 5),
                "rsi_range": (35, 65),
                "phantom_delta": "any",
                "confidence_threshold": 0.5
            },
            "conservative_hold": {
                "kcs_range": (1, 2),
                "rsi_range": (40, 60),
                "phantom_delta": "any",
                "confidence_threshold": 0.8
            }
        }
        
        # Performance tracking
        self.analysis_count = 0
        self.signal_generated_count = 0
        
        logger.info("TRG Analyzer initialized")

    def interpret_trg(self, snapshot: TRGSnapshot) -> TRGResult:
        """
        Interpret Technical Resonance Grid snapshot.
        
        Args:
            snapshot: TRG snapshot with all technical data
            
        Returns:
            TRGResult with signal classification
        """
        self.analysis_count += 1
        
        # Perform Kaprekar analysis if hash fragment provided
        kaprekar_result = None
        if self.kaprekar_analyzer and snapshot.hash_fragment:
            kaprekar_result = self.kaprekar_analyzer.analyze_hash_fragment(snapshot.hash_fragment)
            # Use Kaprekar result if available, otherwise use provided KCS
            if kaprekar_result.is_convergent:
                snapshot.kcs = kaprekar_result.steps_to_converge
        
        # Calculate base confidence from KCS
        kcs_confidence = max(0.1, 1.0 - (snapshot.kcs / 7.0))
        
        # RSI phase analysis
        rsi_phase = self._get_rsi_phase(snapshot.rsi)
        rsi_confidence = self._calculate_rsi_confidence(snapshot.rsi)
        
        # Pole proximity analysis
        pole_confidence = self._calculate_pole_confidence(
            snapshot.price, snapshot.pole_range
        )
        
        # Phantom delta analysis
        phantom_confidence = self._calculate_phantom_confidence(snapshot.phantom_delta)
        
        # Determine signal class
        signal_class = self._determine_signal_class(snapshot)
        
        # Calculate overall confidence
        overall_confidence = (
            kcs_confidence * 0.4 +
            rsi_confidence * 0.3 +
            pole_confidence * 0.2 +
            phantom_confidence * 0.1
        )
        
        # Apply Kaprekar confidence boost if available
        if kaprekar_result and self.kaprekar_analyzer:
            confidence_boost = self.kaprekar_analyzer.get_confidence_boost(kaprekar_result)
            overall_confidence = min(1.0, overall_confidence + confidence_boost)
        
        # Determine risk level
        risk_level = self._determine_risk_level(snapshot.kcs, snapshot.rsi)
        
        # Get recommended action
        recommended_action = self._get_recommended_action(signal_class, overall_confidence)
        
        # Track signal generation
        if recommended_action != "defer_or_hold":
            self.signal_generated_count += 1
        
        return TRGResult(
            signal_class=signal_class,
            confidence=overall_confidence,
            risk_level=risk_level,
            recommended_action=recommended_action,
            technical_context={
                "kcs": snapshot.kcs,
                "rsi_phase": rsi_phase,
                "pole_proximity": pole_confidence,
                "phantom_delta": snapshot.phantom_delta,
                "kaprekar_entropy_class": kaprekar_result.entropy_class if kaprekar_result else "UNKNOWN"
            },
            kaprekar_result=kaprekar_result
        )

    def _get_rsi_phase(self, rsi: float) -> str:
        """Get RSI phase classification."""
        for phase, (low, high) in self.RSI_PHASES.items():
            if low <= rsi <= high:
                return phase
        return "UNKNOWN"

    def _calculate_rsi_confidence(self, rsi: float) -> float:
        """Calculate confidence based on RSI position."""
        if rsi <= 30 or rsi >= 70:
            return 0.9  # High confidence at extremes
        elif 40 <= rsi <= 60:
            return 0.5  # Lower confidence in neutral zone
        else:
            return 0.7  # Moderate confidence

    def _calculate_pole_confidence(self, price: float, pole_range: Tuple[float, float]) -> float:
        """Calculate confidence based on proximity to price poles."""
        low, high = pole_range
        if low <= price <= high:
            return 0.8  # Price within pole range
        else:
            distance = min(abs(price - low), abs(price - high))
            return max(0.1, 1.0 - (distance / price))

    def _calculate_phantom_confidence(self, phantom_delta: float) -> float:
        """Calculate confidence based on phantom delta."""
        if abs(phantom_delta) < 0.001:
            return 0.3  # Low confidence for minimal movement
        elif abs(phantom_delta) < 0.01:
            return 0.7  # Good confidence for moderate movement
        else:
            return 0.9  # High confidence for significant movement

    def _determine_signal_class(self, snapshot: TRGSnapshot) -> str:
        """Determine signal class based on TRG analysis."""
        # Check for BTC long entry conditions
        if (1 <= snapshot.kcs <= 3 and 
            20 <= snapshot.rsi <= 35 and 
            snapshot.phantom_delta > 0):
            return "btc_long_entry"
            
        # Check for USDC exit conditions
        elif (5 <= snapshot.kcs <= 7 and 
              70 <= snapshot.rsi <= 100):
            return "usdc_exit_trigger"
            
        # Check for phantom band swing
        elif (3 <= snapshot.kcs <= 5 and 
              35 <= snapshot.rsi <= 65):
            return "phantom_band_swing"
            
        # Check for conservative hold
        elif (1 <= snapshot.kcs <= 2 and 
              40 <= snapshot.rsi <= 60):
            return "conservative_hold"
            
        else:
            return "standby_or_retry"

    def _determine_risk_level(self, kcs: int, rsi: float) -> str:
        """Determine risk level based on KCS and RSI."""
        if kcs <= 2 and 25 <= rsi <= 35:
            return "LOW"
        elif kcs <= 4 and 40 <= rsi <= 60:
            return "MODERATE"
        elif kcs >= 5 or rsi >= 70:
            return "HIGH"
        else:
            return "MEDIUM"

    def _get_recommended_action(self, signal_class: str, confidence: float) -> str:
        """Get recommended action based on signal class and confidence."""
        if confidence < 0.5:
            return "defer_or_hold"
        elif signal_class == "btc_long_entry":
            return "execute_btc_reentry"
        elif signal_class == "usdc_exit_trigger":
            return "execute_usdc_exit"
        elif signal_class == "phantom_band_swing":
            return "execute_phantom_swing"
        elif signal_class == "conservative_hold":
            return "maintain_position"
        else:
            return "standby_or_retry"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the TRG analyzer."""
        if self.analysis_count == 0:
            return {
                "total_analyses": 0,
                "signal_generation_rate": 0.0,
                "average_confidence": 0.0
            }
            
        return {
            "total_analyses": self.analysis_count,
            "signal_generation_rate": self.signal_generated_count / self.analysis_count,
            "average_confidence": 0.7  # Placeholder - would need to track actual values
        }

# Global instance
trg_analyzer = TRGAnalyzer() 