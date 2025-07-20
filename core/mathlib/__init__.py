#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 MATHLIB - MATHEMATICAL LIBRARY MODULE
========================================

Mathematical components for the Schwabot trading system.

Modules:
- kaprekar_analyzer: Kaprekar's Constant entropy analysis
"""

# Import main components
try:
    from .kaprekar_analyzer import KaprekarAnalyzer, KaprekarResult, kaprekar_analyzer
    __all__ = ['KaprekarAnalyzer', 'KaprekarResult', 'kaprekar_analyzer']
except ImportError:
    __all__ = []

# Version info
__version__ = "1.0.0"
__author__ = "Schwabot Development Team" 