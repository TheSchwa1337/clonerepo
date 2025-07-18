#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils module for Schwabot trading system.

This module provides utility functions and helper classes for the trading system.
"""

import logging
from typing import Any, Dict, Optional, List
import numpy as np
import hashlib
import time

logger = logging.getLogger(__name__)

def utils_status() -> str:
    """Return status of utils module."""
    return "utils module import OK"

def calculate_hash(data: str) -> str:
    """Calculate SHA256 hash of data."""
    return hashlib.sha256(data.encode()).hexdigest()

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize numpy array to [0, 1] range."""
    if np.max(arr) == np.min(arr):
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def calculate_entropy(probabilities: List[float]) -> float:
    """Calculate Shannon entropy from probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

__all__ = ["utils_status", "calculate_hash", "normalize_array", "calculate_entropy"] 