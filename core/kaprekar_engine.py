#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR ENGINE - VOLATILITY CONVERGENCE INDEX
==================================================

Core Kaprekar engine for calculating iterations to reach Kaprekar's constant (6174).
This forms the foundation of our volatility convergence index for tick analysis.

Features:
- Kaprekar iteration calculation for 4-digit numbers
- Volatility convergence detection
- Cycle detection and non-convergence handling
- Integration with tick processing pipeline
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def kaprekar_iterations(n: int, max_iter: int = 10) -> int:
    """
    Calculate the number of iterations to reach Kaprekar's constant (6174)
    for a 4-digit number n. This forms our volatility convergence index.
    
    Args:
        n: 4-digit number to process
        max_iter: Maximum iterations before giving up
        
    Returns:
        Number of iterations to reach 6174, or -1 if non-convergent
    """
    try:
        # Ensure 4-digit format
        if n < 1000:
            n = n * 1000  # Pad with zeros
        elif n > 9999:
            n = n % 10000  # Take last 4 digits
            
        seen = set()
        iter_count = 0
        current = n

        while current != 6174 and iter_count < max_iter:
            # Convert to 4-digit string
            digits = f"{current:04d}"
            
            # Sort ascending and descending
            asc = int("".join(sorted(digits)))
            desc = int("".join(sorted(digits, reverse=True)))
            
            # Calculate difference
            current = desc - asc
            
            # Check for cycles (non-convergence)
            if current in seen:
                logger.debug(f"Cycle detected at iteration {iter_count} for input {n}")
                break
                
            seen.add(current)
            iter_count += 1

        # Return iterations if converged, -1 otherwise
        return iter_count if current == 6174 else -1
        
    except Exception as e:
        logger.error(f"Error in kaprekar_iterations for {n}: {e}")
        return -1


def analyze_kaprekar_convergence(numbers: list) -> dict:
    """
    Analyze Kaprekar convergence patterns across a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with convergence statistics
    """
    try:
        results = []
        convergent_count = 0
        total_iterations = 0
        
        for num in numbers:
            iterations = kaprekar_iterations(num)
            results.append({
                'number': num,
                'iterations': iterations,
                'convergent': iterations != -1
            })
            
            if iterations != -1:
                convergent_count += 1
                total_iterations += iterations
        
        convergence_rate = convergent_count / len(numbers) if numbers else 0
        avg_iterations = total_iterations / convergent_count if convergent_count > 0 else 0
        
        return {
            'total_numbers': len(numbers),
            'convergent_count': convergent_count,
            'convergence_rate': convergence_rate,
            'average_iterations': avg_iterations,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_kaprekar_convergence: {e}")
        return {}


def get_volatility_classification(iterations: int) -> str:
    """
    Classify volatility based on Kaprekar iterations.
    
    Args:
        iterations: Number of iterations to reach 6174
        
    Returns:
        Volatility classification string
    """
    if iterations == -1:
        return "non_convergent"
    elif iterations <= 2:
        return "low_volatility"
    elif iterations <= 4:
        return "medium_volatility"
    elif iterations <= 6:
        return "high_volatility"
    else:
        return "extreme_volatility"


def validate_kaprekar_input(n: int) -> bool:
    """
    Validate input for Kaprekar processing.
    
    Args:
        n: Number to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it's a positive integer
        if not isinstance(n, int) or n <= 0:
            return False
            
        # Check if it can be represented as 4 digits
        if n > 9999:
            return False
            
        return True
        
    except Exception:
        return False


def debug_kaprekar_iterations(n: int, max_iter: int = 10) -> None:
    """
    Debug function to trace through Kaprekar iterations.
    
    Args:
        n: 4-digit number to process
        max_iter: Maximum iterations before giving up
    """
    try:
        # Ensure 4-digit format
        if n < 1000:
            n = n * 1000  # Pad with zeros
        elif n > 9999:
            n = n % 10000  # Take last 4 digits
            
        seen = set()
        iter_count = 0
        current = n
        
        print(f"Debug: Starting with {current}")
        
        while current != 6174 and iter_count < max_iter:
            # Convert to 4-digit string
            digits = f"{current:04d}"
            
            # Sort ascending and descending
            asc = int("".join(sorted(digits)))
            desc = int("".join(sorted(digits, reverse=True)))
            
            # Calculate difference
            current = desc - asc
            
            print(f"Debug: {digits} -> desc={desc}, asc={asc} -> {current}")
            
            # Check for cycles (non-convergence)
            if current in seen:
                print(f"Debug: Cycle detected at iteration {iter_count}")
                break
                
            seen.add(current)
            iter_count += 1
        
        if current == 6174:
            print(f"Debug: Converged to 6174 in {iter_count} iterations")
        else:
            print(f"Debug: Did not converge, ended at {current}")
        
    except Exception as e:
        print(f"Debug error: {e}")


# Test function for validation
def test_kaprekar_engine():
    """Test the Kaprekar engine with known values."""
    test_cases = [
        (1234, 3),   # Known convergent case
        (1000, 5),   # Another convergent case
        (1111, -1),  # Non-convergent case (all same digits)
        (9999, -1),  # Non-convergent case (leads to cycle of zeros)
    ]
    
    print("🧮 Testing Kaprekar Engine...")
    for input_num, expected in test_cases:
        result = kaprekar_iterations(input_num)
        status = "✅" if result == expected else "❌"
        print(f"{status} Input: {input_num}, Expected: {expected}, Got: {result}")
    
    print("✅ Kaprekar Engine test completed")


if __name__ == "__main__":
    test_kaprekar_engine() 