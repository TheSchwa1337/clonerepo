#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Key Allocator
===================

Manages memory keys and links for the Schwabot trading system.
This module provides memory key allocation, similarity matching, and
memory link management for efficient data storage and retrieval.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

class KeyType(Enum):
    """Types of memory keys."""
    PRICE = "price"
    VOLUME = "volume"
    PATTERN = "pattern"
    SIGNAL = "signal"
    ANALYSIS = "analysis"
    TRADE = "trade"
    COMPOSITE = "composite"

@dataclass
class MemoryKey:
    """Represents a memory key."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key_type: KeyType = KeyType.COMPOSITE
    hash_value: str = ""
    data_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    similarity_score: float = 0.0

@dataclass
class MemoryLink:
    """Represents a link between memory keys."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_key_id: str = ""
    target_key_id: str = ""
    link_type: str = ""
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

class MemoryKeyAllocator:
    """Manages memory keys and links."""
    
    def __init__(self, max_keys: int = 10000, max_links: int = 50000):
        """Initialize the memory key allocator."""
        self.max_keys = max_keys
        self.max_links = max_links
        
        # Storage
        self.memory_keys: Dict[str, MemoryKey] = {}
        self.memory_links: Dict[str, MemoryLink] = {}
        self.key_type_index: Dict[KeyType, Set[str]] = {ktype: set() for ktype in KeyType}
        
        # Similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.cache_size_limit = 10000
        
        # Statistics
        self.stats = {
            "total_keys_created": 0,
            "total_links_created": 0,
            "total_allocations": 0,
            "total_similarity_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def allocate_memory_key(self, data: Any, key_type: KeyType = KeyType.COMPOSITE, 
                                metadata: Optional[Dict[str, Any]] = None) -> MemoryKey:
        """
        Allocate a new memory key for data.
        
        Args:
            data: Data to create key for
            key_type: Type of memory key
            metadata: Additional metadata
            
        Returns:
            Allocated memory key
        """
        try:
            # Generate data hash
            data_hash = self._generate_data_hash(data)
            
            # Check if key already exists
            existing_key = await self._find_existing_key(data_hash, key_type)
            if existing_key:
                # Update access statistics
                existing_key.last_accessed = datetime.now()
                existing_key.access_count += 1
                self.stats["total_allocations"] += 1
                logger.debug(f"🔑 Reused existing memory key: {existing_key.id}")
                return existing_key
            
            # Create new key
            key = MemoryKey(
                key_type=key_type,
                hash_value=self._generate_key_hash(data_hash, key_type),
                data_hash=data_hash,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1
            )
            
            # Check capacity
            if len(self.memory_keys) >= self.max_keys:
                await self._cleanup_old_keys()
            
            # Store key
            self.memory_keys[key.id] = key
            self.key_type_index[key_type].add(key.id)
            
            # Update statistics
            self.stats["total_keys_created"] += 1
            self.stats["total_allocations"] += 1
            
            logger.info(f"🔑 Allocated new memory key: {key.id} ({key_type.value})")
            return key
            
        except Exception as e:
            logger.error(f"❌ Memory key allocation failed: {e}")
            raise
    
    async def create_memory_link(self, source_key_id: str, target_key_id: str, 
                               link_type: str = "similarity", strength: float = 1.0,
                               metadata: Optional[Dict[str, Any]] = None) -> MemoryLink:
        """
        Create a link between memory keys.
        
        Args:
            source_key_id: Source key ID
            target_key_id: Target key ID
            link_type: Type of link
            strength: Link strength (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Created memory link
        """
        try:
            # Validate keys exist
            if source_key_id not in self.memory_keys:
                raise ValueError(f"Source key not found: {source_key_id}")
            if target_key_id not in self.memory_keys:
                raise ValueError(f"Target key not found: {target_key_id}")
            
            # Check if link already exists
            existing_link = await self._find_existing_link(source_key_id, target_key_id, link_type)
            if existing_link:
                # Update existing link
                existing_link.strength = max(existing_link.strength, strength)
                existing_link.last_accessed = datetime.now()
                if metadata:
                    existing_link.metadata.update(metadata)
                logger.debug(f"🔗 Updated existing memory link: {existing_link.id}")
                return existing_link
            
            # Check capacity
            if len(self.memory_links) >= self.max_links:
                await self._cleanup_old_links()
            
            # Create new link
            link = MemoryLink(
                source_key_id=source_key_id,
                target_key_id=target_key_id,
                link_type=link_type,
                strength=strength,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            
            # Store link
            self.memory_links[link.id] = link
            
            # Update statistics
            self.stats["total_links_created"] += 1
            
            logger.info(f"🔗 Created memory link: {link.id} ({link_type})")
            return link
            
        except Exception as e:
            logger.error(f"❌ Memory link creation failed: {e}")
            raise
    
    async def find_similar_memory_keys(self, query_key_id: str, key_type: Optional[KeyType] = None,
                                     max_results: int = 10, similarity_threshold: float = 0.5) -> List[Tuple[MemoryKey, float]]:
        """
        Find memory keys similar to a query key.
        
        Args:
            query_key_id: Query key ID
            key_type: Filter by key type
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (key, similarity_score) tuples
        """
        try:
            if query_key_id not in self.memory_keys:
                raise ValueError(f"Query key not found: {query_key_id}")
            
            query_key = self.memory_keys[query_key_id]
            similar_keys = []
            
            # Get candidate keys
            candidate_keys = self._get_candidate_keys(key_type)
            
            # Calculate similarities
            for key_id in candidate_keys:
                if key_id == query_key_id:
                    continue
                
                candidate_key = self.memory_keys[key_id]
                similarity = await self._calculate_similarity(query_key, candidate_key)
                
                if similarity >= similarity_threshold:
                    similar_keys.append((candidate_key, similarity))
            
            # Sort by similarity and limit results
            similar_keys.sort(key=lambda x: x[1], reverse=True)
            similar_keys = similar_keys[:max_results]
            
            # Update statistics
            self.stats["total_similarity_searches"] += 1
            
            logger.debug(f"🔍 Found {len(similar_keys)} similar keys for {query_key_id}")
            return similar_keys
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    async def get_memory_key(self, key_id: str) -> Optional[MemoryKey]:
        """Get a memory key by ID."""
        key = self.memory_keys.get(key_id)
        if key:
            key.last_accessed = datetime.now()
            key.access_count += 1
        return key
    
    async def get_memory_links(self, key_id: str, link_type: Optional[str] = None) -> List[MemoryLink]:
        """Get all links for a key."""
        links = []
        for link in self.memory_links.values():
            if (link.source_key_id == key_id or link.target_key_id == key_id) and \
               (link_type is None or link.link_type == link_type):
                links.append(link)
        return links
    
    async def delete_memory_key(self, key_id: str) -> bool:
        """Delete a memory key and its links."""
        try:
            if key_id not in self.memory_keys:
                return False
            
            key = self.memory_keys[key_id]
            
            # Remove from type index
            self.key_type_index[key.key_type].discard(key_id)
            
            # Remove key
            del self.memory_keys[key_id]
            
            # Remove associated links
            links_to_remove = []
            for link_id, link in self.memory_links.items():
                if link.source_key_id == key_id or link.target_key_id == key_id:
                    links_to_remove.append(link_id)
            
            for link_id in links_to_remove:
                del self.memory_links[link_id]
            
            # Clear similarity cache entries
            self._clear_similarity_cache_entries(key_id)
            
            logger.info(f"🗑️ Deleted memory key: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory key deletion failed: {e}")
            return False
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old memory keys and links."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up old keys
            keys_to_remove = []
            for key_id, key in self.memory_keys.items():
                if key.last_accessed < cutoff_time and key.access_count < 5:
                    keys_to_remove.append(key_id)
            
            for key_id in keys_to_remove:
                await self.delete_memory_key(key_id)
            
            # Clean up old links
            links_to_remove = []
            for link_id, link in self.memory_links.items():
                if link.last_accessed < cutoff_time and link.strength < 0.5:
                    links_to_remove.append(link_id)
            
            for link_id in links_to_remove:
                del self.memory_links[link_id]
            
            logger.info(f"🧹 Cleaned up {len(keys_to_remove)} keys and {len(links_to_remove)} links")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory allocator statistics."""
        return {
            **self.stats,
            "current_keys": len(self.memory_keys),
            "current_links": len(self.memory_links),
            "keys_by_type": {ktype.value: len(keys) for ktype, keys in self.key_type_index.items()},
            "similarity_cache_size": len(self.similarity_cache)
        }
    
    def _generate_data_hash(self, data: Any) -> str:
        """Generate hash for data."""
        try:
            if isinstance(data, (str, bytes)):
                data_str = str(data)
            else:
                data_str = str(data)
            
            return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        except Exception:
            return hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest()
    
    def _generate_key_hash(self, data_hash: str, key_type: KeyType) -> str:
        """Generate hash for memory key."""
        key_str = f"{data_hash}:{key_type.value}:{time.time()}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    async def _find_existing_key(self, data_hash: str, key_type: KeyType) -> Optional[MemoryKey]:
        """Find existing key with same data hash and type."""
        for key in self.memory_keys.values():
            if key.data_hash == data_hash and key.key_type == key_type:
                return key
        return None
    
    async def _find_existing_link(self, source_id: str, target_id: str, link_type: str) -> Optional[MemoryLink]:
        """Find existing link between keys."""
        for link in self.memory_links.values():
            if (link.source_key_id == source_id and link.target_key_id == target_id and 
                link.link_type == link_type):
                return link
        return None
    
    def _get_candidate_keys(self, key_type: Optional[KeyType]) -> List[str]:
        """Get candidate keys for similarity search."""
        if key_type:
            return list(self.key_type_index[key_type])
        else:
            return list(self.memory_keys.keys())
    
    async def _calculate_similarity(self, key1: MemoryKey, key2: MemoryKey) -> float:
        """Calculate similarity between two memory keys."""
        cache_key = tuple(sorted([key1.id, key2.id]))
        
        # Check cache
        if cache_key in self.similarity_cache:
            self.stats["cache_hits"] += 1
            return self.similarity_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        try:
            # Calculate similarity based on key type
            if key1.key_type == key2.key_type:
                # Same type - use hash similarity
                similarity = self._calculate_hash_similarity(key1.hash_value, key2.hash_value)
            else:
                # Different types - use metadata similarity
                similarity = self._calculate_metadata_similarity(key1.metadata, key2.metadata)
            
            # Cache result
            if len(self.similarity_cache) < self.cache_size_limit:
                self.similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between hash values."""
        if len(hash1) != len(hash2):
            return 0.0
        
        # Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        max_distance = len(hash1)
        
        return 1.0 - (distance / max_distance)
    
    def _calculate_metadata_similarity(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> float:
        """Calculate similarity between metadata."""
        if not metadata1 or not metadata2:
            return 0.0
        
        # Simple key overlap similarity
        keys1 = set(metadata1.keys())
        keys2 = set(metadata2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    async def _cleanup_old_keys(self):
        """Clean up old keys when at capacity."""
        # Remove least recently used keys
        sorted_keys = sorted(
            self.memory_keys.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        
        # Remove 10% of keys
        remove_count = max(1, len(sorted_keys) // 10)
        for key_id, _ in sorted_keys[:remove_count]:
            await self.delete_memory_key(key_id)
    
    async def _cleanup_old_links(self):
        """Clean up old links when at capacity."""
        # Remove weakest links
        sorted_links = sorted(
            self.memory_links.items(),
            key=lambda x: (x[1].strength, x[1].last_accessed)
        )
        
        # Remove 10% of links
        remove_count = max(1, len(sorted_links) // 10)
        for link_id, _ in sorted_links[:remove_count]:
            del self.memory_links[link_id]
    
    def _clear_similarity_cache_entries(self, key_id: str):
        """Clear similarity cache entries for a key."""
        keys_to_remove = []
        for cache_key in self.similarity_cache.keys():
            if key_id in cache_key:
                keys_to_remove.append(cache_key)
        
        for cache_key in keys_to_remove:
            del self.similarity_cache[cache_key]

# Global instance
_allocator = MemoryKeyAllocator()

async def allocate_memory_key(data: Any, key_type: KeyType = KeyType.COMPOSITE, 
                            metadata: Optional[Dict[str, Any]] = None) -> MemoryKey:
    """Allocate a new memory key."""
    return await _allocator.allocate_memory_key(data, key_type, metadata)

async def create_memory_link(source_key_id: str, target_key_id: str, 
                           link_type: str = "similarity", strength: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> MemoryLink:
    """Create a memory link."""
    return await _allocator.create_memory_link(source_key_id, target_key_id, link_type, strength, metadata)

async def find_similar_memory_keys(query_key_id: str, key_type: Optional[KeyType] = None,
                                 max_results: int = 10, similarity_threshold: float = 0.5) -> List[Tuple[MemoryKey, float]]:
    """Find similar memory keys."""
    return await _allocator.find_similar_memory_keys(query_key_id, key_type, max_results, similarity_threshold) 