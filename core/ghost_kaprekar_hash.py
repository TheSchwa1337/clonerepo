#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👻 GHOST KAPREKAR HASH - STRATEGY TRACKING SYSTEM
=================================================

Ghost Kaprekar hash module that generates SHA256 hashes for strategy tracking
based on Kaprekar analysis and price tick data.

Features:
- Kaprekar-based SHA256 hash generation
- Strategy tracking and identification
- Hash collision detection and resolution
- Integration with existing hash systems
"""

import hashlib
import logging
import time
from typing import Dict, Any, Optional, List
from .tick_kaprekar_bridge import price_to_kaprekar_index

logger = logging.getLogger(__name__)

def generate_kaprekar_strategy_hash(price_tick: float) -> str:
    """
    Encode a tick into a Kaprekar-based SHA256 hash for strategy tracking.
    
    Args:
        price_tick: Float price value
        
    Returns:
        SHA256 hash string for strategy tracking
    """
    try:
        k_index = price_to_kaprekar_index(price_tick)
        payload = f"kaprekar-{k_index}-{price_tick}"
        return hashlib.sha256(payload.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Error in generate_kaprekar_strategy_hash for {price_tick}: {e}")
        # Fallback hash
        fallback_payload = f"fallback-{price_tick}-{time.time()}"
        return hashlib.sha256(fallback_payload.encode()).hexdigest()


def generate_enhanced_kaprekar_hash(price_tick: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate enhanced Kaprekar hash with additional metadata.
    
    Args:
        price_tick: Float price value
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary with hash and metadata
    """
    try:
        k_index = price_to_kaprekar_index(price_tick)
        timestamp = time.time()
        
        # Create enhanced payload
        base_payload = f"kaprekar-{k_index}-{price_tick}-{timestamp}"
        
        # Add metadata if provided
        if metadata:
            metadata_str = str(sorted(metadata.items()))
            base_payload += f"-{metadata_str}"
        
        # Generate hash
        hash_result = hashlib.sha256(base_payload.encode()).hexdigest()
        
        return {
            'hash': hash_result,
            'kaprekar_index': k_index,
            'price_tick': price_tick,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'payload_length': len(base_payload),
            'hash_type': 'enhanced_kaprekar'
        }
        
    except Exception as e:
        logger.error(f"Error in generate_enhanced_kaprekar_hash for {price_tick}: {e}")
        return {
            'hash': hashlib.sha256(str(time.time()).encode()).hexdigest(),
            'error': str(e),
            'hash_type': 'error_fallback'
        }


def batch_generate_kaprekar_hashes(price_ticks: List[float]) -> List[Dict[str, Any]]:
    """
    Generate Kaprekar hashes for multiple price ticks.
    
    Args:
        price_ticks: List of float price values
        
    Returns:
        List of hash results
    """
    try:
        results = []
        for price_tick in price_ticks:
            hash_result = generate_enhanced_kaprekar_hash(price_tick)
            results.append(hash_result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch_generate_kaprekar_hashes: {e}")
        return []


def analyze_hash_patterns(hashes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in generated Kaprekar hashes.
    
    Args:
        hashes: List of hash results
        
    Returns:
        Dictionary with pattern analysis
    """
    try:
        if not hashes:
            return {'error': 'No hashes to analyze'}
        
        # Count hash types
        hash_type_counts = {}
        kaprekar_index_distribution = {}
        payload_lengths = []
        
        for hash_result in hashes:
            # Count hash types
            hash_type = hash_result.get('hash_type', 'unknown')
            hash_type_counts[hash_type] = hash_type_counts.get(hash_type, 0) + 1
            
            # Count Kaprekar indices
            k_index = hash_result.get('kaprekar_index', -1)
            kaprekar_index_distribution[k_index] = kaprekar_index_distribution.get(k_index, 0) + 1
            
            # Collect payload lengths
            payload_length = hash_result.get('payload_length', 0)
            if payload_length > 0:
                payload_lengths.append(payload_length)
        
        # Calculate statistics
        avg_payload_length = sum(payload_lengths) / len(payload_lengths) if payload_lengths else 0
        
        return {
            'total_hashes': len(hashes),
            'hash_type_distribution': hash_type_counts,
            'kaprekar_index_distribution': kaprekar_index_distribution,
            'average_payload_length': avg_payload_length,
            'hash_results': hashes
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_hash_patterns: {e}")
        return {'error': str(e)}


def detect_hash_collisions(hashes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect potential hash collisions in generated hashes.
    
    Args:
        hashes: List of hash results
        
    Returns:
        List of detected collisions
    """
    try:
        hash_values = {}
        collisions = []
        
        for hash_result in hashes:
            hash_value = hash_result.get('hash', '')
            if hash_value in hash_values:
                # Collision detected
                collision = {
                    'hash_value': hash_value,
                    'first_occurrence': hash_values[hash_value],
                    'second_occurrence': hash_result,
                    'collision_type': 'exact_match'
                }
                collisions.append(collision)
            else:
                hash_values[hash_value] = hash_result
        
        return collisions
        
    except Exception as e:
        logger.error(f"Error in detect_hash_collisions: {e}")
        return []


def generate_strategy_signature(price_tick: float, strategy_name: str, confidence: float) -> Dict[str, Any]:
    """
    Generate a strategy signature using Kaprekar hash.
    
    Args:
        price_tick: Float price value
        strategy_name: Name of the strategy
        confidence: Confidence level (0.0 to 1.0)
        
    Returns:
        Dictionary with strategy signature
    """
    try:
        k_index = price_to_kaprekar_index(price_tick)
        timestamp = time.time()
        
        # Create strategy-specific payload
        payload = f"strategy-{strategy_name}-kaprekar-{k_index}-{price_tick}-{confidence}-{timestamp}"
        hash_result = hashlib.sha256(payload.encode()).hexdigest()
        
        return {
            'signature': hash_result,
            'strategy_name': strategy_name,
            'kaprekar_index': k_index,
            'price_tick': price_tick,
            'confidence': confidence,
            'timestamp': timestamp,
            'payload': payload,
            'signature_type': 'strategy_kaprekar'
        }
        
    except Exception as e:
        logger.error(f"Error in generate_strategy_signature: {e}")
        return {
            'signature': hashlib.sha256(str(time.time()).encode()).hexdigest(),
            'error': str(e),
            'signature_type': 'error_fallback'
        }


def validate_hash_integrity(hash_result: Dict[str, Any]) -> bool:
    """
    Validate the integrity of a generated hash.
    
    Args:
        hash_result: Hash result dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ['hash', 'hash_type']
        for field in required_fields:
            if field not in hash_result:
                return False
        
        # Check hash format (should be 64 character hex string)
        hash_value = hash_result.get('hash', '')
        if len(hash_value) != 64 or not all(c in '0123456789abcdef' for c in hash_value):
            return False
        
        # Check hash type
        hash_type = hash_result.get('hash_type', '')
        valid_types = ['enhanced_kaprekar', 'error_fallback']
        if hash_type not in valid_types:
            return False
        
        return True
        
    except Exception:
        return False


def generate_hash_chain(price_ticks: List[float], chain_length: int = 10) -> List[Dict[str, Any]]:
    """
    Generate a chain of Kaprekar hashes for sequential price ticks.
    
    Args:
        price_ticks: List of float price values
        chain_length: Length of hash chain to generate
        
    Returns:
        List of chained hash results
    """
    try:
        if len(price_ticks) < chain_length:
            return []
        
        chain = []
        previous_hash = None
        
        for i in range(chain_length):
            price_tick = price_ticks[i]
            
            # Include previous hash in payload for chaining
            if previous_hash:
                payload = f"chain-{i}-{price_tick}-{previous_hash}"
            else:
                payload = f"chain-{i}-{price_tick}-start"
            
            hash_result = hashlib.sha256(payload.encode()).hexdigest()
            
            chain_entry = {
                'chain_position': i,
                'price_tick': price_tick,
                'hash': hash_result,
                'previous_hash': previous_hash,
                'payload': payload
            }
            
            chain.append(chain_entry)
            previous_hash = hash_result
        
        return chain
        
    except Exception as e:
        logger.error(f"Error in generate_hash_chain: {e}")
        return []


# Test function for validation
def test_ghost_kaprekar_hash():
    """Test the ghost Kaprekar hash system with sample prices."""
    test_prices = [
        2045.29,    # Should generate stable hash
        123.456,    # Should generate midrange hash
        9999.99,    # Should generate high vol hash
        1111.11,    # Should generate non-convergent hash
    ]
    
    print("👻 Testing Ghost Kaprekar Hash...")
    
    # Test basic hash generation
    for price in test_prices:
        basic_hash = generate_kaprekar_strategy_hash(price)
        enhanced_hash = generate_enhanced_kaprekar_hash(price)
        print(f"Price: {price} → Basic Hash: {basic_hash[:16]}... → Enhanced: {enhanced_hash['hash'][:16]}...")
    
    # Test strategy signature
    strategy_sig = generate_strategy_signature(2045.29, "BTC_MICROHOLD_REBUY", 0.85)
    print(f"Strategy Signature: {strategy_sig['signature'][:16]}...")
    
    print("✅ Ghost Kaprekar Hash test completed")


if __name__ == "__main__":
    test_ghost_kaprekar_hash() 