#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕯️ LANTERN CORE ENHANCED - KAPREKAR-INTEGRATED ECHO ENGINE
==========================================================

Enhanced Lantern Core with Kaprekar entropy analysis integration
for improved ghost memory and reentry signal processing.

Features:
- Kaprekar-based echo strength calculation
- TRG-integrated signal validation
- Cross-AI validation support
- Enhanced soulprint analysis
- Real-time entropy classification
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Import Kaprekar components
try:
    from .mathlib.kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult
    from .trg_analyzer import TRGAnalyzer, TRGSnapshot, TRGResult
    KAPREKAR_COMPONENTS_AVAILABLE = True
except ImportError:
    KAPREKAR_COMPONENTS_AVAILABLE = False
    logging.warning("Kaprekar components not available")

logger = logging.getLogger(__name__)

# Define EchoType enum locally if not available
class EchoType(Enum):
    """Types of echo signals for different reentry scenarios."""
    GHOST_REENTRY = "ghost_reentry"
    TRIPLET_MATCH = "triplet_match"
    SILENT_ZONE_ESCAPE = "silent_zone_escape"
    VOLUME_SPIKE = "volume_spike"
    TIME_SEEDED = "time_seeded"
    RESONANCE_TRIGGER = "resonance_trigger"

@dataclass
class EchoSignal:
    """Echo signal result from Lantern Core."""
    echo_type: EchoType
    symbol: str
    strength: float
    hash_value: str
    timestamp: datetime
    metadata: Dict[str, Any]
    confidence: float = 0.0

@dataclass
class LanternKaprekarSignal:
    """Enhanced Lantern signal with Kaprekar analysis."""
    echo_signal: EchoSignal
    kaprekar_result: Optional[KaprekarResult]
    trg_result: Optional[TRGResult]
    ai_validation: str
    confidence_boost: float
    final_confidence: float

class LanternCoreEnhanced:
    """Enhanced Lantern Core with Kaprekar integration."""
    
    def __init__(self, *args, **kwargs):
        # Initialize Kaprekar components
        self.kaprekar_analyzer = KaprekarAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        self.trg_analyzer = TRGAnalyzer() if KAPREKAR_COMPONENTS_AVAILABLE else None
        
        # Enhanced metrics
        self.kaprekar_metrics = {
            'total_kaprekar_analyses': 0,
            'convergent_signals': 0,
            'chaotic_signals': 0,
            'trg_validations': 0,
            'ai_validations': 0,
            'enhanced_signals_generated': 0
        }
        
        logger.info("Lantern Core Enhanced initialized with Kaprekar integration")

    def process_enhanced_echo(
        self,
        symbol: str,
        current_price: float,
        rsi: float,
        pole_range: tuple,
        phantom_delta: float,
        hash_fragment: str,
        ai_validation: Optional[str] = None
    ) -> Optional[LanternKaprekarSignal]:
        """
        Process echo signal with full Kaprekar and TRG analysis.
        
        Args:
            symbol: Trading symbol
            current_price: Current asset price
            rsi: Current RSI value
            pole_range: Price pole range (support/resistance)
            phantom_delta: Phantom band delta
            hash_fragment: Hash fragment for Kaprekar analysis
            ai_validation: Optional AI validation result
            
        Returns:
            Enhanced Lantern signal if valid, None otherwise
        """
        try:
            # Generate base echo signal
            echo_signal = self._generate_base_echo_signal(symbol, current_price)
            
            # Perform Kaprekar analysis
            kaprekar_result = None
            if self.kaprekar_analyzer:
                kaprekar_result = self.kaprekar_analyzer.analyze_hash_fragment(hash_fragment)
                self.kaprekar_metrics['total_kaprekar_analyses'] += 1
                
                if kaprekar_result.is_convergent:
                    self.kaprekar_metrics['convergent_signals'] += 1
                else:
                    self.kaprekar_metrics['chaotic_signals'] += 1
            
            # Perform TRG analysis
            trg_result = None
            if self.trg_analyzer:
                trg_snapshot = TRGSnapshot(
                    kcs=kaprekar_result.steps_to_converge if kaprekar_result else 7,
                    rsi=rsi,
                    price=current_price,
                    pole_range=pole_range,
                    phantom_delta=phantom_delta,
                    asset=symbol,
                    timestamp=time.time(),
                    hash_fragment=hash_fragment
                )
                
                trg_result = self.trg_analyzer.interpret_trg(trg_snapshot)
                self.kaprekar_metrics['trg_validations'] += 1
            
            # Apply AI validation if available
            if ai_validation:
                self.kaprekar_metrics['ai_validations'] += 1
            else:
                ai_validation = "pending"
            
            # Calculate confidence boost from Kaprekar
            confidence_boost = 0.0
            if kaprekar_result and self.kaprekar_analyzer:
                confidence_boost = self.kaprekar_analyzer.get_confidence_boost(kaprekar_result)
            
            # Calculate final confidence
            base_confidence = echo_signal.confidence if echo_signal else 0.5
            final_confidence = min(1.0, base_confidence + confidence_boost)
            
            # Create enhanced signal
            enhanced_signal = LanternKaprekarSignal(
                echo_signal=echo_signal,
                kaprekar_result=kaprekar_result,
                trg_result=trg_result,
                ai_validation=ai_validation,
                confidence_boost=confidence_boost,
                final_confidence=final_confidence
            )
            
            # Validate signal
            if self._validate_enhanced_signal(enhanced_signal):
                self.kaprekar_metrics['enhanced_signals_generated'] += 1
                return enhanced_signal
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in enhanced echo processing: {e}")
            return None

    def _generate_base_echo_signal(self, symbol: str, current_price: float) -> EchoSignal:
        """Generate base echo signal."""
        return EchoSignal(
            echo_type=EchoType.GHOST_REENTRY,
            symbol=symbol,
            strength=0.7,
            hash_value=f"enhanced_{symbol}_{int(time.time())}",
            timestamp=datetime.now(),
            metadata={"price": current_price},
            confidence=0.7
        )

    def _validate_enhanced_signal(self, signal: LanternKaprekarSignal) -> bool:
        """Validate enhanced signal based on multiple criteria."""
        # Check TRG confidence if available
        if signal.trg_result and signal.trg_result.confidence < 0.5:
            return False
            
        # Check Kaprekar convergence if available
        if signal.kaprekar_result and not signal.kaprekar_result.is_convergent:
            return False
            
        # Check AI validation if available
        if signal.ai_validation == "reject":
            return False
            
        # Check echo signal strength
        if signal.echo_signal.strength < 0.3:
            return False
            
        # Check final confidence
        if signal.final_confidence < 0.4:
            return False
            
        return True

    def get_kaprekar_metrics(self) -> Dict[str, Any]:
        """Get Kaprekar integration metrics."""
        return {
            **self.kaprekar_metrics,
            "convergence_rate": (
                self.kaprekar_metrics['convergent_signals'] / 
                max(1, self.kaprekar_metrics['total_kaprekar_analyses'])
            ),
            "trg_validation_rate": (
                self.kaprekar_metrics['trg_validations'] / 
                max(1, self.kaprekar_metrics['total_kaprekar_analyses'])
            ),
            "enhanced_signal_rate": (
                self.kaprekar_metrics['enhanced_signals_generated'] / 
                max(1, self.kaprekar_metrics['total_kaprekar_analyses'])
            )
        }

    def get_strategy_recommendation(self, enhanced_signal: LanternKaprekarSignal) -> str:
        """Get strategy recommendation from enhanced signal."""
        if enhanced_signal.kaprekar_result and self.kaprekar_analyzer:
            return self.kaprekar_analyzer.get_strategy_recommendation(enhanced_signal.kaprekar_result)
        elif enhanced_signal.trg_result:
            return enhanced_signal.trg_result.recommended_action
        else:
            return "standby_or_retry"

    def process_batch_signals(
        self,
        signals: List[Dict[str, Any]]
    ) -> List[LanternKaprekarSignal]:
        """Process a batch of signals with Kaprekar analysis."""
        results = []
        
        for signal_data in signals:
            enhanced_signal = self.process_enhanced_echo(
                symbol=signal_data.get('symbol', ''),
                current_price=signal_data.get('price', 0.0),
                rsi=signal_data.get('rsi', 50.0),
                pole_range=signal_data.get('pole_range', (0.0, 0.0)),
                phantom_delta=signal_data.get('phantom_delta', 0.0),
                hash_fragment=signal_data.get('hash_fragment', ''),
                ai_validation=signal_data.get('ai_validation')
            )
            
            if enhanced_signal:
                results.append(enhanced_signal)
        
        return results

# Global instance
lantern_core_enhanced = LanternCoreEnhanced() 