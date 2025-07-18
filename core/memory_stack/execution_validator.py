#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Validator
==================

Validates trading executions and detects drift in the Schwabot trading system.
This module provides execution validation, drift detection, and cost simulation
for trading operations.
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status for executions."""
    PENDING = "pending"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"

class DriftType(Enum):
    """Types of drift that can be detected."""
    PRICE_DRIFT = "price_drift"
    VOLUME_DRIFT = "volume_drift"
    VOLATILITY_DRIFT = "volatility_drift"
    TREND_DRIFT = "trend_drift"
    PATTERN_DRIFT = "pattern_drift"

@dataclass
class ExecutionValidation:
    """Result of execution validation."""
    execution_id: str
    status: ValidationStatus
    confidence: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)

@dataclass
class DriftDetection:
    """Result of drift detection."""
    drift_type: DriftType
    severity: float  # 0.0 to 1.0
    confidence: float
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionCost:
    """Simulated execution cost."""
    base_cost: float
    slippage_cost: float
    market_impact: float
    total_cost: float
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExecutionValidator:
    """Validates trading executions and detects drift."""
    
    def __init__(self):
        """Initialize the execution validator."""
        self.validation_history: List[ExecutionValidation] = []
        self.drift_history: List[DriftDetection] = []
        self.cost_history: List[ExecutionCost] = []
        
        # Configuration
        self.max_slippage_threshold = 0.02  # 2%
        self.max_volume_threshold = 0.1  # 10%
        self.max_volatility_threshold = 0.05  # 5%
        self.drift_detection_window = 100  # Data points
        
        # Statistics
        self.stats = {
            "total_validations": 0,
            "approved_validations": 0,
            "rejected_validations": 0,
            "drift_detections": 0,
            "average_validation_time": 0.0
        }
    
    async def validate_execution(self, execution_data: Dict[str, Any]) -> ExecutionValidation:
        """
        Validate a trading execution.
        
        Args:
            execution_data: Execution data to validate
            
        Returns:
            ExecutionValidation result
        """
        start_time = time.time()
        
        try:
            execution_id = execution_data.get("execution_id", str(time.time()))
            validation = ExecutionValidation(execution_id=execution_id)
            
            # Perform validation checks
            checks = await self._perform_validation_checks(execution_data)
            
            # Determine overall status
            if checks["errors"]:
                validation.status = ValidationStatus.REJECTED
                validation.confidence = 0.0
            elif checks["warnings"]:
                validation.status = ValidationStatus.WARNING
                validation.confidence = 0.7
            else:
                validation.status = ValidationStatus.APPROVED
                validation.confidence = 0.95
            
            validation.warnings = checks["warnings"]
            validation.errors = checks["errors"]
            validation.metadata = checks["metadata"]
            
            # Update statistics
            self.stats["total_validations"] += 1
            if validation.status == ValidationStatus.APPROVED:
                self.stats["approved_validations"] += 1
            elif validation.status == ValidationStatus.REJECTED:
                self.stats["rejected_validations"] += 1
            
            # Add to history
            self.validation_history.append(validation)
            
            validation_time = time.time() - start_time
            self.stats["average_validation_time"] = (
                (self.stats["average_validation_time"] * (self.stats["total_validations"] - 1) + validation_time) /
                self.stats["total_validations"]
            )
            
            logger.info(f"✅ Execution {execution_id} validation: {validation.status.value}")
            return validation
            
        except Exception as e:
            logger.error(f"❌ Execution validation failed: {e}")
            return ExecutionValidation(
                execution_id=execution_data.get("execution_id", "unknown"),
                status=ValidationStatus.REJECTED,
                confidence=0.0,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def validate_drift(self, historical_data: List[float], current_data: List[float]) -> List[DriftDetection]:
        """
        Detect drift in trading data.
        
        Args:
            historical_data: Historical data series
            current_data: Current data series
            
        Returns:
            List of drift detections
        """
        try:
            drift_detections = []
            
            # Price drift detection
            price_drift = await self._detect_price_drift(historical_data, current_data)
            if price_drift:
                drift_detections.append(price_drift)
            
            # Volume drift detection
            volume_drift = await self._detect_volume_drift(historical_data, current_data)
            if volume_drift:
                drift_detections.append(volume_drift)
            
            # Volatility drift detection
            volatility_drift = await self._detect_volatility_drift(historical_data, current_data)
            if volatility_drift:
                drift_detections.append(volatility_drift)
            
            # Trend drift detection
            trend_drift = await self._detect_trend_drift(historical_data, current_data)
            if trend_drift:
                drift_detections.append(trend_drift)
            
            # Update statistics
            self.stats["drift_detections"] += len(drift_detections)
            
            # Add to history
            self.drift_history.extend(drift_detections)
            
            if drift_detections:
                logger.warning(f"⚠️ Detected {len(drift_detections)} drift(s)")
            
            return drift_detections
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            return []
    
    async def simulate_execution_cost(self, order_data: Dict[str, Any]) -> ExecutionCost:
        """
        Simulate execution cost for an order.
        
        Args:
            order_data: Order data for cost simulation
            
        Returns:
            Simulated execution cost
        """
        try:
            # Extract order parameters
            symbol = order_data.get("symbol", "")
            side = order_data.get("side", "buy")
            quantity = order_data.get("quantity", 0.0)
            price = order_data.get("price", 0.0)
            order_type = order_data.get("order_type", "market")
            
            # Base cost calculation
            base_cost = quantity * price
            
            # Slippage estimation
            slippage_cost = await self._estimate_slippage(order_data)
            
            # Market impact estimation
            market_impact = await self._estimate_market_impact(order_data)
            
            # Total cost
            total_cost = base_cost + slippage_cost + market_impact
            
            cost = ExecutionCost(
                base_cost=base_cost,
                slippage_cost=slippage_cost,
                market_impact=market_impact,
                total_cost=total_cost,
                currency=order_data.get("currency", "USD"),
                metadata={
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type
                }
            )
            
            # Add to history
            self.cost_history.append(cost)
            
            logger.info(f"💰 Simulated cost for {symbol}: ${total_cost:.2f}")
            return cost
            
        except Exception as e:
            logger.error(f"❌ Cost simulation failed: {e}")
            return ExecutionCost(
                base_cost=0.0,
                slippage_cost=0.0,
                market_impact=0.0,
                total_cost=0.0,
                metadata={"error": str(e)}
            )
    
    async def _perform_validation_checks(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform all validation checks."""
        warnings = []
        errors = []
        metadata = {}
        
        # Price validation
        price_check = await self._validate_price(execution_data)
        if price_check["error"]:
            errors.append(price_check["error"])
        if price_check["warning"]:
            warnings.append(price_check["warning"])
        metadata["price_check"] = price_check["metadata"]
        
        # Volume validation
        volume_check = await self._validate_volume(execution_data)
        if volume_check["error"]:
            errors.append(volume_check["error"])
        if volume_check["warning"]:
            warnings.append(volume_check["warning"])
        metadata["volume_check"] = volume_check["metadata"]
        
        # Timing validation
        timing_check = await self._validate_timing(execution_data)
        if timing_check["error"]:
            errors.append(timing_check["error"])
        if timing_check["warning"]:
            warnings.append(timing_check["warning"])
        metadata["timing_check"] = timing_check["metadata"]
        
        # Risk validation
        risk_check = await self._validate_risk(execution_data)
        if risk_check["error"]:
            errors.append(risk_check["error"])
        if risk_check["warning"]:
            warnings.append(risk_check["warning"])
        metadata["risk_check"] = risk_check["metadata"]
        
        return {
            "warnings": warnings,
            "errors": errors,
            "metadata": metadata
        }
    
    async def _validate_price(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution price."""
        expected_price = execution_data.get("expected_price", 0.0)
        actual_price = execution_data.get("actual_price", 0.0)
        
        if expected_price <= 0 or actual_price <= 0:
            return {
                "error": "Invalid price data",
                "warning": None,
                "metadata": {"expected": expected_price, "actual": actual_price}
            }
        
        price_diff = abs(actual_price - expected_price) / expected_price
        
        if price_diff > self.max_slippage_threshold:
            return {
                "error": f"Price slippage too high: {price_diff:.2%}",
                "warning": None,
                "metadata": {"slippage": price_diff, "threshold": self.max_slippage_threshold}
            }
        elif price_diff > self.max_slippage_threshold * 0.5:
            return {
                "error": None,
                "warning": f"High price slippage: {price_diff:.2%}",
                "metadata": {"slippage": price_diff}
            }
        
        return {
            "error": None,
            "warning": None,
            "metadata": {"slippage": price_diff}
        }
    
    async def _validate_volume(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution volume."""
        expected_volume = execution_data.get("expected_volume", 0.0)
        actual_volume = execution_data.get("actual_volume", 0.0)
        
        if expected_volume <= 0 or actual_volume <= 0:
            return {
                "error": "Invalid volume data",
                "warning": None,
                "metadata": {"expected": expected_volume, "actual": actual_volume}
            }
        
        volume_diff = abs(actual_volume - expected_volume) / expected_volume
        
        if volume_diff > self.max_volume_threshold:
            return {
                "error": f"Volume difference too high: {volume_diff:.2%}",
                "warning": None,
                "metadata": {"volume_diff": volume_diff, "threshold": self.max_volume_threshold}
            }
        
        return {
            "error": None,
            "warning": None,
            "metadata": {"volume_diff": volume_diff}
        }
    
    async def _validate_timing(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution timing."""
        expected_time = execution_data.get("expected_time")
        actual_time = execution_data.get("actual_time")
        
        if not expected_time or not actual_time:
            return {
                "error": "Missing timing data",
                "warning": None,
                "metadata": {}
            }
        
        # Convert to datetime if needed
        if isinstance(expected_time, str):
            expected_time = datetime.fromisoformat(expected_time)
        if isinstance(actual_time, str):
            actual_time = datetime.fromisoformat(actual_time)
        
        time_diff = abs((actual_time - expected_time).total_seconds())
        
        if time_diff > 60:  # More than 1 minute
            return {
                "error": f"Execution delay too high: {time_diff:.1f}s",
                "warning": None,
                "metadata": {"delay_seconds": time_diff}
            }
        elif time_diff > 30:  # More than 30 seconds
            return {
                "error": None,
                "warning": f"High execution delay: {time_diff:.1f}s",
                "metadata": {"delay_seconds": time_diff}
            }
        
        return {
            "error": None,
            "warning": None,
            "metadata": {"delay_seconds": time_diff}
        }
    
    async def _validate_risk(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution risk parameters."""
        # This is a placeholder for risk validation
        # In a real system, this would check against risk limits
        
        return {
            "error": None,
            "warning": None,
            "metadata": {"risk_check": "passed"}
        }
    
    async def _detect_price_drift(self, historical: List[float], current: List[float]) -> Optional[DriftDetection]:
        """Detect price drift."""
        if len(historical) < 10 or len(current) < 10:
            return None
        
        try:
            hist_mean = np.mean(historical)
            curr_mean = np.mean(current)
            
            drift_magnitude = abs(curr_mean - hist_mean) / hist_mean if hist_mean > 0 else 0
            
            if drift_magnitude > 0.05:  # 5% drift threshold
                return DriftDetection(
                    drift_type=DriftType.PRICE_DRIFT,
                    severity=min(drift_magnitude, 1.0),
                    confidence=0.8,
                    description=f"Price drift detected: {drift_magnitude:.2%}",
                    metadata={"historical_mean": hist_mean, "current_mean": curr_mean}
                )
        
        except Exception as e:
            logger.error(f"Price drift detection error: {e}")
        
        return None
    
    async def _detect_volume_drift(self, historical: List[float], current: List[float]) -> Optional[DriftDetection]:
        """Detect volume drift."""
        if len(historical) < 10 or len(current) < 10:
            return None
        
        try:
            hist_std = np.std(historical)
            curr_std = np.std(current)
            
            if hist_std > 0:
                drift_magnitude = abs(curr_std - hist_std) / hist_std
                
                if drift_magnitude > 0.2:  # 20% drift threshold
                    return DriftDetection(
                        drift_type=DriftType.VOLUME_DRIFT,
                        severity=min(drift_magnitude, 1.0),
                        confidence=0.7,
                        description=f"Volume drift detected: {drift_magnitude:.2%}",
                        metadata={"historical_std": hist_std, "current_std": curr_std}
                    )
        
        except Exception as e:
            logger.error(f"Volume drift detection error: {e}")
        
        return None
    
    async def _detect_volatility_drift(self, historical: List[float], current: List[float]) -> Optional[DriftDetection]:
        """Detect volatility drift."""
        if len(historical) < 20 or len(current) < 20:
            return None
        
        try:
            # Calculate rolling volatility
            hist_vol = np.std(np.diff(np.log(historical)))
            curr_vol = np.std(np.diff(np.log(current)))
            
            if hist_vol > 0:
                drift_magnitude = abs(curr_vol - hist_vol) / hist_vol
                
                if drift_magnitude > 0.3:  # 30% drift threshold
                    return DriftDetection(
                        drift_type=DriftType.VOLATILITY_DRIFT,
                        severity=min(drift_magnitude, 1.0),
                        confidence=0.75,
                        description=f"Volatility drift detected: {drift_magnitude:.2%}",
                        metadata={"historical_vol": hist_vol, "current_vol": curr_vol}
                    )
        
        except Exception as e:
            logger.error(f"Volatility drift detection error: {e}")
        
        return None
    
    async def _detect_trend_drift(self, historical: List[float], current: List[float]) -> Optional[DriftDetection]:
        """Detect trend drift."""
        if len(historical) < 20 or len(current) < 20:
            return None
        
        try:
            # Calculate trend using linear regression
            hist_trend = np.polyfit(range(len(historical)), historical, 1)[0]
            curr_trend = np.polyfit(range(len(current)), current, 1)[0]
            
            trend_change = abs(curr_trend - hist_trend)
            
            if trend_change > 0.01:  # Trend change threshold
                return DriftDetection(
                    drift_type=DriftType.TREND_DRIFT,
                    severity=min(trend_change * 100, 1.0),
                    confidence=0.6,
                    description=f"Trend drift detected: {trend_change:.4f}",
                    metadata={"historical_trend": hist_trend, "current_trend": curr_trend}
                )
        
        except Exception as e:
            logger.error(f"Trend drift detection error: {e}")
        
        return None
    
    async def _estimate_slippage(self, order_data: Dict[str, Any]) -> float:
        """Estimate slippage cost."""
        try:
            quantity = order_data.get("quantity", 0.0)
            price = order_data.get("price", 0.0)
            order_type = order_data.get("order_type", "market")
            
            # Simple slippage estimation
            if order_type == "market":
                # Market orders have higher slippage
                slippage_rate = 0.001  # 0.1%
            else:
                # Limit orders have lower slippage
                slippage_rate = 0.0001  # 0.01%
            
            return quantity * price * slippage_rate
            
        except Exception:
            return 0.0
    
    async def _estimate_market_impact(self, order_data: Dict[str, Any]) -> float:
        """Estimate market impact cost."""
        try:
            quantity = order_data.get("quantity", 0.0)
            price = order_data.get("price", 0.0)
            
            # Simple market impact estimation
            # Larger orders have higher impact
            impact_rate = min(quantity / 1000.0, 0.01)  # Max 1% impact
            
            return quantity * price * impact_rate
            
        except Exception:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.stats,
            "validation_history_size": len(self.validation_history),
            "drift_history_size": len(self.drift_history),
            "cost_history_size": len(self.cost_history)
        }

# Global instance
_validator = ExecutionValidator()

async def validate_execution(execution_data: Dict[str, Any]) -> ExecutionValidation:
    """Validate a trading execution."""
    return await _validator.validate_execution(execution_data)

async def validate_drift(historical_data: List[float], current_data: List[float]) -> List[DriftDetection]:
    """Detect drift in trading data."""
    return await _validator.validate_drift(historical_data, current_data)

async def simulate_execution_cost(order_data: Dict[str, Any]) -> ExecutionCost:
    """Simulate execution cost for an order."""
    return await _validator.simulate_execution_cost(order_data) 