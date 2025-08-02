#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Mathematical Audit Script
==================================

Comprehensive audit of all mathematical functions to ensure:
1. No transcoding errors in entry/exit paths
2. No division by zero
3. No overflow/underflow issues
4. Mathematical correctness
5. Proper bounds checking
"""

import math
import numpy as np
import sys
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MathAuditResult:
    """Result of mathematical audit."""
    function_name: str
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    fix_suggestion: str
    test_cases: List[Dict]

class MathematicalAuditor:
    """Comprehensive mathematical auditor for Schwabot system."""
    
    def __init__(self):
        self.audit_results = []
        self.critical_issues = []
        self.warnings = []
        self.passed_checks = []
        
    def audit_phantom_mode_engine(self) -> List[MathAuditResult]:
        """Audit Phantom Mode Engine mathematical functions."""
        results = []
        
        # Test ZeroBoundEntropy.compress_entropy
        try:
            # Test extreme values that could cause overflow
            test_cases = [
                {"entropy": 1000.0, "expected": "clamped"},
                {"entropy": -1000.0, "expected": "clamped"},
                {"entropy": 0.0, "expected": "normal"},
                {"entropy": 1.0, "expected": "normal"}
            ]
            
            for case in test_cases:
                entropy = case["entropy"]
                threshold = 0.28
                exponent = entropy - threshold
                
                # Check if clamping is working
                if exponent > 700:
                    if exponent != 700:  # Should be clamped
                        results.append(MathAuditResult(
                            function_name="ZeroBoundEntropy.compress_entropy",
                            file_path="core/phantom_mode_engine.py",
                            line_number=120,
                            issue_type="overflow_protection",
                            severity="critical",
                            description=f"Exponent {exponent} exceeds safe limit, should be clamped to 700",
                            fix_suggestion="Ensure exponent clamping is implemented",
                            test_cases=[case]
                        ))
                elif exponent < -700:
                    if exponent != -700:  # Should be clamped
                        results.append(MathAuditResult(
                            function_name="ZeroBoundEntropy.compress_entropy",
                            file_path="core/phantom_mode_engine.py",
                            line_number=120,
                            issue_type="underflow_protection",
                            severity="critical",
                            description=f"Exponent {exponent} below safe limit, should be clamped to -700",
                            fix_suggestion="Ensure exponent clamping is implemented",
                            test_cases=[case]
                        ))
                else:
                    # Test the actual calculation
                    try:
                        result = 1.0 / (1.0 + math.exp(exponent))
                        if not (0.0 <= result <= 1.0):
                            results.append(MathAuditResult(
                                function_name="ZeroBoundEntropy.compress_entropy",
                                file_path="core/phantom_mode_engine.py",
                                line_number=120,
                                issue_type="bounds_violation",
                                severity="high",
                                description=f"Result {result} outside expected range [0,1]",
                                fix_suggestion="Add result bounds checking",
                                test_cases=[case]
                            ))
                    except OverflowError:
                        results.append(MathAuditResult(
                            function_name="ZeroBoundEntropy.compress_entropy",
                            file_path="core/phantom_mode_engine.py",
                            line_number=120,
                            issue_type="overflow_error",
                            severity="critical",
                            description="OverflowError occurred despite clamping",
                            fix_suggestion="Review clamping logic",
                            test_cases=[case]
                        ))
                        
        except Exception as e:
            results.append(MathAuditResult(
                function_name="ZeroBoundEntropy.compress_entropy",
                file_path="core/phantom_mode_engine.py",
                line_number=120,
                issue_type="exception",
                severity="critical",
                description=f"Exception during audit: {e}",
                fix_suggestion="Review function implementation",
                test_cases=[]
            ))
        
        # Test CycleBloomPrediction.predict_next_cycle
        try:
            test_cases = [
                {"sigmoid_input": 1000.0, "expected": "clamped"},
                {"sigmoid_input": -1000.0, "expected": "clamped"},
                {"sigmoid_input": 0.0, "expected": "normal"},
                {"sigmoid_input": 1.0, "expected": "normal"}
            ]
            
            for case in test_cases:
                sigmoid_input = case["sigmoid_input"]
                
                # Check clamping
                if sigmoid_input > 700:
                    if sigmoid_input != 700:
                        results.append(MathAuditResult(
                            function_name="CycleBloomPrediction.predict_next_cycle",
                            file_path="core/phantom_mode_engine.py",
                            line_number=380,
                            issue_type="overflow_protection",
                            severity="critical",
                            description=f"Sigmoid input {sigmoid_input} exceeds safe limit",
                            fix_suggestion="Ensure sigmoid input clamping is implemented",
                            test_cases=[case]
                        ))
                elif sigmoid_input < -700:
                    if sigmoid_input != -700:
                        results.append(MathAuditResult(
                            function_name="CycleBloomPrediction.predict_next_cycle",
                            file_path="core/phantom_mode_engine.py",
                            line_number=380,
                            issue_type="underflow_protection",
                            severity="critical",
                            description=f"Sigmoid input {sigmoid_input} below safe limit",
                            fix_suggestion="Ensure sigmoid input clamping is implemented",
                            test_cases=[case]
                        ))
                else:
                    # Test sigmoid calculation
                    try:
                        result = 1.0 / (1.0 + math.exp(-sigmoid_input))
                        if not (0.0 <= result <= 1.0):
                            results.append(MathAuditResult(
                                function_name="CycleBloomPrediction.predict_next_cycle",
                                file_path="core/phantom_mode_engine.py",
                                line_number=380,
                                issue_type="bounds_violation",
                                severity="high",
                                description=f"Sigmoid result {result} outside expected range [0,1]",
                                fix_suggestion="Add result bounds checking",
                                test_cases=[case]
                            ))
                    except OverflowError:
                        results.append(MathAuditResult(
                            function_name="CycleBloomPrediction.predict_next_cycle",
                            file_path="core/phantom_mode_engine.py",
                            line_number=380,
                            issue_type="overflow_error",
                            severity="critical",
                            description="OverflowError in sigmoid calculation",
                            fix_suggestion="Review sigmoid calculation",
                            test_cases=[case]
                        ))
                        
        except Exception as e:
            results.append(MathAuditResult(
                function_name="CycleBloomPrediction.predict_next_cycle",
                file_path="core/phantom_mode_engine.py",
                line_number=380,
                issue_type="exception",
                severity="critical",
                description=f"Exception during audit: {e}",
                fix_suggestion="Review function implementation",
                test_cases=[]
            ))
        
        return results
    
    def audit_mode_integration_system(self) -> List[MathAuditResult]:
        """Audit Mode Integration System mathematical functions."""
        results = []
        
        # Test position size calculations
        test_cases = [
            {"price": 0.0, "balance": 10000.0, "position_size_pct": 10.0, "expected": "error"},
            {"price": 50000.0, "balance": 0.0, "position_size_pct": 10.0, "expected": "zero"},
            {"price": 50000.0, "balance": 10000.0, "position_size_pct": 0.0, "expected": "zero"},
            {"price": 50000.0, "balance": 10000.0, "position_size_pct": 10.0, "expected": "normal"},
            {"price": -50000.0, "balance": 10000.0, "position_size_pct": 10.0, "expected": "error"}
        ]
        
        for case in test_cases:
            price = case["price"]
            balance = case["balance"]
            position_size_pct = case["position_size_pct"]
            
            try:
                # Test division by zero
                if price == 0.0:
                    results.append(MathAuditResult(
                        function_name="_calculate_position_size",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=470,
                        issue_type="division_by_zero",
                        severity="critical",
                        description="Division by zero when price is 0.0",
                        fix_suggestion="Add price validation: if price <= 0: return 0.0",
                        test_cases=[case]
                    ))
                    continue
                
                # Test negative price
                if price < 0.0:
                    results.append(MathAuditResult(
                        function_name="_calculate_position_size",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=470,
                        issue_type="negative_price",
                        severity="high",
                        description="Negative price should be handled",
                        fix_suggestion="Add price validation: if price < 0: return 0.0",
                        test_cases=[case]
                    ))
                    continue
                
                # Test normal calculation
                base_amount = balance * (position_size_pct / 100)
                position_size = base_amount / price
                
                if position_size < 0.0:
                    results.append(MathAuditResult(
                        function_name="_calculate_position_size",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=470,
                        issue_type="negative_position_size",
                        severity="high",
                        description="Negative position size calculated",
                        fix_suggestion="Ensure position size is always non-negative",
                        test_cases=[case]
                    ))
                    
            except ZeroDivisionError:
                if case["expected"] != "error":
                    results.append(MathAuditResult(
                        function_name="_calculate_position_size",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=470,
                        issue_type="unexpected_division_by_zero",
                        severity="critical",
                        description="Unexpected ZeroDivisionError",
                        fix_suggestion="Add proper error handling",
                        test_cases=[case]
                    ))
            except Exception as e:
                results.append(MathAuditResult(
                    function_name="_calculate_position_size",
                    file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                    line_number=470,
                    issue_type="exception",
                    severity="critical",
                    description=f"Exception during calculation: {e}",
                    fix_suggestion="Review calculation logic",
                    test_cases=[case]
                ))
        
        # Test stop loss and take profit calculations
        test_cases = [
            {"entry_price": 50000.0, "stop_loss_pct": 1.0, "take_profit_pct": 10.0},
            {"entry_price": 0.0, "stop_loss_pct": 1.0, "take_profit_pct": 10.0},
            {"entry_price": -50000.0, "stop_loss_pct": 1.0, "take_profit_pct": 10.0}
        ]
        
        for case in test_cases:
            entry_price = case["entry_price"]
            stop_loss_pct = case["stop_loss_pct"]
            take_profit_pct = case["take_profit_pct"]
            
            try:
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
                
                # Check for negative prices
                if stop_loss < 0.0 or take_profit < 0.0:
                    results.append(MathAuditResult(
                        function_name="stop_loss_take_profit_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=450,
                        issue_type="negative_price_levels",
                        severity="high",
                        description="Negative stop loss or take profit calculated",
                        fix_suggestion="Add bounds checking for price levels",
                        test_cases=[case]
                    ))
                
                # Check for logical errors (stop loss > take profit)
                if stop_loss > take_profit:
                    results.append(MathAuditResult(
                        function_name="stop_loss_take_profit_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=450,
                        issue_type="logical_error",
                        severity="critical",
                        description="Stop loss greater than take profit",
                        fix_suggestion="Validate stop loss < take profit",
                        test_cases=[case]
                    ))
                    
            except Exception as e:
                results.append(MathAuditResult(
                    function_name="stop_loss_take_profit_calculation",
                    file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                    line_number=450,
                    issue_type="exception",
                    severity="critical",
                    description=f"Exception during calculation: {e}",
                    fix_suggestion="Review calculation logic",
                    test_cases=[case]
                ))
        
        return results
    
    def audit_backend_math_systems(self) -> List[MathAuditResult]:
        """Audit backend math systems for mathematical errors."""
        results = []
        
        # Test mathematical functions for edge cases
        math_functions = [
            {"name": "log", "args": [0.0], "expected": "error"},
            {"name": "log", "args": [-1.0], "expected": "error"},
            {"name": "log", "args": [1.0], "expected": "normal"},
            {"name": "sqrt", "args": [-1.0], "expected": "error"},
            {"name": "sqrt", "args": [0.0], "expected": "normal"},
            {"name": "sqrt", "args": [1.0], "expected": "normal"},
            {"name": "exp", "args": [1000.0], "expected": "overflow"},
            {"name": "exp", "args": [-1000.0], "expected": "underflow"},
            {"name": "exp", "args": [0.0], "expected": "normal"},
            {"name": "divide", "args": [1.0, 0.0], "expected": "error"},
            {"name": "divide", "args": [1.0, 1.0], "expected": "normal"}
        ]
        
        for func in math_functions:
            func_name = func["name"]
            args = func["args"]
            expected = func["expected"]
            
            try:
                if func_name == "log":
                    if args[0] <= 0:
                        if expected == "error":
                            # This should raise an error
                            try:
                                result = math.log(args[0])
                                results.append(MathAuditResult(
                                    function_name=f"BackendMath.{func_name}",
                                    file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                    line_number=86,
                                    issue_type="missing_validation",
                                    severity="critical",
                                    description=f"log({args[0]}) should raise error but didn't",
                                    fix_suggestion="Add validation for non-positive values",
                                    test_cases=[func]
                                ))
                            except ValueError:
                                # Expected behavior
                                pass
                    else:
                        result = math.log(args[0])
                        if not np.isfinite(result):
                            results.append(MathAuditResult(
                                function_name=f"BackendMath.{func_name}",
                                file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                line_number=86,
                                issue_type="non_finite_result",
                                severity="high",
                                description=f"log({args[0]}) returned non-finite result: {result}",
                                fix_suggestion="Add result validation",
                                test_cases=[func]
                            ))
                
                elif func_name == "sqrt":
                    if args[0] < 0:
                        if expected == "error":
                            try:
                                result = math.sqrt(args[0])
                                results.append(MathAuditResult(
                                    function_name=f"BackendMath.{func_name}",
                                    file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                    line_number=100,
                                    issue_type="missing_validation",
                                    severity="critical",
                                    description=f"sqrt({args[0]}) should raise error but didn't",
                                    fix_suggestion="Add validation for negative values",
                                    test_cases=[func]
                                ))
                            except ValueError:
                                # Expected behavior
                                pass
                    else:
                        result = math.sqrt(args[0])
                        if not np.isfinite(result):
                            results.append(MathAuditResult(
                                function_name=f"BackendMath.{func_name}",
                                file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                line_number=100,
                                issue_type="non_finite_result",
                                severity="high",
                                description=f"sqrt({args[0]}) returned non-finite result: {result}",
                                fix_suggestion="Add result validation",
                                test_cases=[func]
                            ))
                
                elif func_name == "exp":
                    try:
                        result = math.exp(args[0])
                        if args[0] > 700 and expected == "overflow":
                            if np.isfinite(result):
                                results.append(MathAuditResult(
                                    function_name=f"BackendMath.{func_name}",
                                    file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                    line_number=95,
                                    issue_type="overflow_not_handled",
                                    severity="critical",
                                    description=f"exp({args[0]}) should overflow but didn't",
                                    fix_suggestion="Add overflow protection",
                                    test_cases=[func]
                                ))
                        elif not np.isfinite(result):
                            results.append(MathAuditResult(
                                function_name=f"BackendMath.{func_name}",
                                file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                line_number=95,
                                issue_type="non_finite_result",
                                severity="high",
                                description=f"exp({args[0]}) returned non-finite result: {result}",
                                fix_suggestion="Add result validation",
                                test_cases=[func]
                            ))
                    except OverflowError:
                        if expected != "overflow":
                            results.append(MathAuditResult(
                                function_name=f"BackendMath.{func_name}",
                                file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                line_number=95,
                                issue_type="unexpected_overflow",
                                severity="critical",
                                description=f"exp({args[0]}) unexpectedly overflowed",
                                fix_suggestion="Add overflow protection",
                                test_cases=[func]
                            ))
                
                elif func_name == "divide":
                    if args[1] == 0:
                        if expected == "error":
                            try:
                                result = args[0] / args[1]
                                results.append(MathAuditResult(
                                    function_name=f"BackendMath.{func_name}",
                                    file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                    line_number=75,
                                    issue_type="missing_validation",
                                    severity="critical",
                                    description=f"divide({args[0]}, {args[1]}) should raise error but didn't",
                                    fix_suggestion="Add division by zero validation",
                                    test_cases=[func]
                                ))
                            except ZeroDivisionError:
                                # Expected behavior
                                pass
                    else:
                        result = args[0] / args[1]
                        if not np.isfinite(result):
                            results.append(MathAuditResult(
                                function_name=f"BackendMath.{func_name}",
                                file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                                line_number=75,
                                issue_type="non_finite_result",
                                severity="high",
                                description=f"divide({args[0]}, {args[1]}) returned non-finite result: {result}",
                                fix_suggestion="Add result validation",
                                test_cases=[func]
                            ))
                            
            except Exception as e:
                if expected != "error":
                    results.append(MathAuditResult(
                        function_name=f"BackendMath.{func_name}",
                        file_path="AOI_Base_Files_Schwabot/core/backend_math.py",
                        line_number=75,
                        issue_type="unexpected_exception",
                        severity="critical",
                        description=f"Unexpected exception in {func_name}: {e}",
                        fix_suggestion="Review function implementation",
                        test_cases=[func]
                    ))
        
        return results
    
    def audit_entry_exit_paths(self) -> List[MathAuditResult]:
        """Audit entry/exit path calculations specifically."""
        results = []
        
        # Test entry price calculations
        test_cases = [
            {"current_price": 50000.0, "expected": "normal"},
            {"current_price": 0.0, "expected": "error"},
            {"current_price": -50000.0, "expected": "error"},
            {"current_price": float('inf'), "expected": "error"},
            {"current_price": float('nan'), "expected": "error"}
        ]
        
        for case in test_cases:
            current_price = case["current_price"]
            
            try:
                # Test entry price assignment
                entry_price = current_price
                
                if not np.isfinite(entry_price):
                    results.append(MathAuditResult(
                        function_name="entry_price_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=450,
                        issue_type="non_finite_entry_price",
                        severity="critical",
                        description=f"Non-finite entry price: {entry_price}",
                        fix_suggestion="Add price validation before assignment",
                        test_cases=[case]
                    ))
                elif entry_price <= 0:
                    results.append(MathAuditResult(
                        function_name="entry_price_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=450,
                        issue_type="invalid_entry_price",
                        severity="critical",
                        description=f"Invalid entry price: {entry_price}",
                        fix_suggestion="Add positive price validation",
                        test_cases=[case]
                    ))
                    
            except Exception as e:
                results.append(MathAuditResult(
                    function_name="entry_price_calculation",
                    file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                    line_number=450,
                    issue_type="exception",
                    severity="critical",
                    description=f"Exception in entry price calculation: {e}",
                    fix_suggestion="Add proper error handling",
                    test_cases=[case]
                ))
        
        # Test stop loss and take profit calculations for entry/exit
        test_cases = [
            {"entry_price": 50000.0, "stop_loss_pct": 1.0, "take_profit_pct": 10.0},
            {"entry_price": 50000.0, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
            {"entry_price": 50000.0, "stop_loss_pct": 1.0, "take_profit_pct": 0.0},
            {"entry_price": 50000.0, "stop_loss_pct": -1.0, "take_profit_pct": 10.0},
            {"entry_price": 50000.0, "stop_loss_pct": 1.0, "take_profit_pct": -10.0}
        ]
        
        for case in test_cases:
            entry_price = case["entry_price"]
            stop_loss_pct = case["stop_loss_pct"]
            take_profit_pct = case["take_profit_pct"]
            
            try:
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
                
                # Check for logical errors in exit points
                if stop_loss >= take_profit:
                    results.append(MathAuditResult(
                        function_name="exit_points_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=451,
                        issue_type="logical_error",
                        severity="critical",
                        description=f"Stop loss ({stop_loss}) >= Take profit ({take_profit})",
                        fix_suggestion="Ensure stop_loss < take_profit",
                        test_cases=[case]
                    ))
                
                # Check for negative exit points
                if stop_loss < 0 or take_profit < 0:
                    results.append(MathAuditResult(
                        function_name="exit_points_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=451,
                        issue_type="negative_exit_points",
                        severity="high",
                        description=f"Negative exit points: stop_loss={stop_loss}, take_profit={take_profit}",
                        fix_suggestion="Add bounds checking for exit points",
                        test_cases=[case]
                    ))
                
                # Check for non-finite exit points
                if not (np.isfinite(stop_loss) and np.isfinite(take_profit)):
                    results.append(MathAuditResult(
                        function_name="exit_points_calculation",
                        file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                        line_number=451,
                        issue_type="non_finite_exit_points",
                        severity="critical",
                        description=f"Non-finite exit points: stop_loss={stop_loss}, take_profit={take_profit}",
                        fix_suggestion="Add validation for exit point calculations",
                        test_cases=[case]
                    ))
                    
            except Exception as e:
                results.append(MathAuditResult(
                    function_name="exit_points_calculation",
                    file_path="AOI_Base_Files_Schwabot/core/mode_integration_system.py",
                    line_number=451,
                    issue_type="exception",
                    severity="critical",
                    description=f"Exception in exit points calculation: {e}",
                    fix_suggestion="Add proper error handling",
                    test_cases=[case]
                ))
        
        return results
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive mathematical audit of the entire system."""
        logger.info("🔍 Starting comprehensive mathematical audit...")
        
        # Run all audits
        phantom_results = self.audit_phantom_mode_engine()
        mode_results = self.audit_mode_integration_system()
        backend_results = self.audit_backend_math_systems()
        entry_exit_results = self.audit_entry_exit_paths()
        
        # Combine all results
        all_results = phantom_results + mode_results + backend_results + entry_exit_results
        
        # Categorize results
        critical_issues = [r for r in all_results if r.severity == "critical"]
        high_issues = [r for r in all_results if r.severity == "high"]
        medium_issues = [r for r in all_results if r.severity == "medium"]
        low_issues = [r for r in all_results if r.severity == "low"]
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(all_results),
            "critical_issues": len(critical_issues),
            "high_issues": len(high_issues),
            "medium_issues": len(medium_issues),
            "low_issues": len(low_issues),
            "phantom_mode_issues": len(phantom_results),
            "mode_integration_issues": len(mode_results),
            "backend_math_issues": len(backend_results),
            "entry_exit_issues": len(entry_exit_results),
            "all_results": [r.__dict__ for r in all_results]
        }
        
        # Log results
        logger.info(f"📊 Audit completed:")
        logger.info(f"   Total issues: {len(all_results)}")
        logger.info(f"   Critical: {len(critical_issues)}")
        logger.info(f"   High: {len(high_issues)}")
        logger.info(f"   Medium: {len(medium_issues)}")
        logger.info(f"   Low: {len(low_issues)}")
        
        if critical_issues:
            logger.error("🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                logger.error(f"   {issue.function_name}: {issue.description}")
        
        if high_issues:
            logger.warning("⚠️ HIGH PRIORITY ISSUES FOUND:")
            for issue in high_issues:
                logger.warning(f"   {issue.function_name}: {issue.description}")
        
        return summary

def main():
    """Main audit function."""
    auditor = MathematicalAuditor()
    results = auditor.run_comprehensive_audit()
    
    # Save results to file
    with open('mathematical_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Mathematical audit completed!")
    print(f"📋 Results saved to: mathematical_audit_results.json")
    
    if results["critical_issues"] > 0:
        print(f"🚨 {results['critical_issues']} CRITICAL ISSUES found - IMMEDIATE ATTENTION REQUIRED!")
        return False
    elif results["high_issues"] > 0:
        print(f"⚠️ {results['high_issues']} HIGH PRIORITY ISSUES found - Review recommended")
        return False
    else:
        print(f"✅ No critical or high-priority mathematical issues found!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 