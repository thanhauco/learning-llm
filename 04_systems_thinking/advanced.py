"""
Systems Thinking - Advanced Level
Production-grade distributed systems patterns
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json


@dataclass
class DistributedCacheConfig:
    """Configuration for distributed cache"""
    nodes: List[str]
    replication_factor: int = 2
    consistency_level: str = "quorum"  # one, quorum, all


class ConsistentHashing:
    """
    Consistent hashing for distributed caching
    
    Real-world: Used by Memcached, Redis Cluster, Cassandra
    
    Benefits:
    - Minimal key redistribution when nodes added/removed
    - Even load distribution
    - Fault tolerance
    """
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """Hash function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        """Add node to ring"""
        self.nodes.add(node)
        
        # Add virtual nodes
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
    
    def remove_node(self, node: str):
        """Remove node from ring"""
        self.nodes.discard(node)
        
        # Remove virtual nodes
        keys_to_remove = []
        for hash_value, n in self.ring.items():
            if n == node:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
    
    def get_node(self, key: str) -> str:
        """Get node for key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find first node clockwise
        sorted_hashes = sorted(self.ring.keys())
        
        for h in sorted_hashes:
            if h >= hash_value:
                return self.ring[h]
        
        # Wrap around
        return self.ring[sorted_hashes[0]]
    
    def get_nodes(self, key: str, count: int = 1) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring:
            return []
        
        hash_value = self._hash(key)
        sorted_hashes = sorted(self.ring.keys())
        
        # Find starting position
        start_idx = 0
        for i, h in enumerate(sorted_hashes):
            if h >= hash_value:
                start_idx = i
                break
        
        # Get unique nodes
        nodes = []
        seen = set()
        idx = start_idx
        
        while len(nodes) < count and len(seen) < len(self.nodes):
            node = self.ring[sorted_hashes[idx % len(sorted_hashes)]]
            if node not in seen:
                nodes.append(node)
                seen.add(node)
            idx += 1
        
        return nodes


class RateLimiterDistributed:
    """
    Distributed rate limiter using sliding window
    
    Real-world: Redis-based rate limiting
    
    Algorithms:
    - Token bucket
    - Leaky bucket
    - Sliding window
    - Fixed window
    """
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def allow_request(self, user_id: str) -> tuple[bool, Dict]:
        """
        Check if request is allowed (sliding window)
        
        Returns: (allowed, metadata)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Get user's requests
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests = [t for t in user_requests if t > window_start]
        self.requests[user_id] = user_requests
        
        # Check limit
        if len(user_requests) < self.max_requests:
            user_requests.append(now)
            return True, {
                "remaining": self.max_requests - len(user_requests),
                "reset_at": window_start + self.window_seconds,
                "limit": self.max_requests
            }
        else:
            # Calculate retry after
            oldest_request = min(user_requests)
            retry_after = oldest_request + self.window_seconds - now
            
            return False, {
                "remaining": 0,
                "reset_at": oldest_request + self.window_seconds,
                "retry_after": retry_after,
                "limit": self.max_requests
            }


class RequestBatcher:
    """
    Batch requests for efficiency
    
    Real-world: Batch database queries, API calls
    
    Benefits:
    - Reduce network overhead
    - Better throughput
    - Lower latency (amortized)
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 100,
        processor: Optional[Callable] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        self.processor = processor or self._default_processor
        
        self.pending_requests: List[Dict] = []
        self.batch_count = 0
    
    async def add_request(self, request: Dict) -> any:
        """
        Add request to batch
        
        Returns when batch is processed
        """
        self.pending_requests.append(request)
        
        # Process if batch is full
        if len(self.pending_requests) >= self.max_batch_size:
            return await self._process_batch()
        
        # Wait for more requests or timeout
        await asyncio.sleep(self.max_wait_ms)
        
        if self.pending_requests:
            return await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch"""
        if not self.pending_requests:
            return None
        
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        self.batch_count += 1
        
        # Process batch
        results = await self.processor(batch)
        
        return results
    
    async def _default_processor(self, batch: List[Dict]) -> List:
        """Default batch processor"""
        # Simulate processing
        await asyncio.sleep(0.01)
        return [{"result": f"processed_{i}"} for i in range(len(batch))]


class AdaptiveTimeout:
    """
    Adaptive timeout based on historical latency
    
    Real-world: Prevent slow requests from blocking
    
    Formula: timeout = P99 latency * safety_factor
    """
    
    def __init__(self, initial_timeout: float = 5.0, safety_factor: float = 1.5):
        self.initial_timeout = initial_timeout
        self.safety_factor = safety_factor
        self.latencies: List[float] = []
        self.max_samples = 1000
    
    def record_latency(self, latency: float):
        """Record request latency"""
        self.latencies.append(latency)
        
        # Keep only recent samples
        if len(self.latencies) > self.max_samples:
            self.latencies = self.latencies[-self.max_samples:]
    
    def get_timeout(self) -> float:
        """
        Calculate adaptive timeout
        
        Uses P99 latency as baseline
        """
        if len(self.latencies) < 10:
            return self.initial_timeout
        
        import numpy as np
        p99 = np.percentile(self.latencies, 99)
        
        return p99 * self.safety_factor
    
    def get_stats(self) -> Dict:
        """Get timeout statistics"""
        if not self.latencies:
            return {}
        
        import numpy as np
        
        return {
            "current_timeout": self.get_timeout(),
            "p50_latency": np.percentile(self.latencies, 50),
            "p95_latency": np.percentile(self.latencies, 95),
            "p99_latency": np.percentile(self.latencies, 99),
            "samples": len(self.latencies)
        }


if __name__ == "__main__":
    print("=== Consistent Hashing ===\n")
    
    # Create ring with 3 nodes
    nodes = ["node1", "node2", "node3"]
    ring = ConsistentHashing(nodes)
    
    # Distribute keys
    keys = [f"key{i}" for i in range(20)]
    distribution = {node: 0 for node in nodes}
    
    print("Initial distribution:")
    for key in keys:
        node = ring.get_node(key)
        distribution[node] += 1
    
    for node, count in distribution.items():
        print(f"  {node}: {count} keys ({count/len(keys)*100:.1f}%)")
    
    # Add new node
    print("\nAdding node4...")
    ring.add_node("node4")
    
    new_distribution = {node: 0 for node in ["node1", "node2", "node3", "node4"]}
    moved_keys = 0
    
    for key in keys:
        new_node = ring.get_node(key)
        new_distribution[new_node] += 1
        
        # Check if key moved
        old_node = None
        for node in nodes:
            if distribution.get(node, 0) > 0:
                old_node = node
                break
        
        if new_node not in nodes:
            moved_keys += 1
    
    print("\nNew distribution:")
    for node, count in new_distribution.items():
        print(f"  {node}: {count} keys ({count/len(keys)*100:.1f}%)")
    
    print(f"\nKeys moved: {moved_keys}/{len(keys)} ({moved_keys/len(keys)*100:.1f}%)")
    print("(Consistent hashing minimizes key movement)")
    
    print("\n=== Distributed Rate Limiter ===\n")
    
    limiter = RateLimiterDistributed(max_requests=5, window_seconds=10)
    
    print("Testing rate limiter (5 requests per 10 seconds):")
    
    for i in range(8):
        allowed, metadata = limiter.allow_request("user1")
        
        if allowed:
            print(f"  Request {i+1}: ✓ Allowed (remaining: {metadata['remaining']})")
        else:
            print(f"  Request {i+1}: ✗ Denied (retry after: {metadata['retry_after']:.1f}s)")
        
        time.sleep(0.5)
    
    print("\n=== Adaptive Timeout ===\n")
    
    timeout_manager = AdaptiveTimeout(initial_timeout=5.0)
    
    # Simulate requests with varying latency
    print("Simulating requests:")
    latencies = [0.1, 0.15, 0.2, 0.12, 0.18, 0.25, 0.3, 0.5, 1.0, 0.15]
    
    for i, latency in enumerate(latencies):
        timeout_manager.record_latency(latency)
        current_timeout = timeout_manager.get_timeout()
        print(f"  Request {i+1}: latency={latency:.2f}s, timeout={current_timeout:.2f}s")
    
    stats = timeout_manager.get_stats()
    print(f"\nTimeout Statistics:")
    print(f"  Current timeout: {stats['current_timeout']:.2f}s")
    print(f"  P50 latency: {stats['p50_latency']:.2f}s")
    print(f"  P95 latency: {stats['p95_latency']:.2f}s")
    print(f"  P99 latency: {stats['p99_latency']:.2f}s")
    
    print("\n=== Production Patterns Summary ===\n")
    print("1. Consistent Hashing:")
    print("   - Minimal key redistribution")
    print("   - Even load distribution")
    print("   - Used by: Redis Cluster, Cassandra, DynamoDB")
    
    print("\n2. Distributed Rate Limiting:")
    print("   - Sliding window algorithm")
    print("   - Per-user limits")
    print("   - Used by: API gateways, CDNs")
    
    print("\n3. Request Batching:")
    print("   - Reduce network overhead")
    print("   - Better throughput")
    print("   - Used by: Database drivers, ML inference")
    
    print("\n4. Adaptive Timeout:")
    print("   - Based on P99 latency")
    print("   - Prevents slow requests")
    print("   - Used by: Microservices, RPC frameworks")
    
    print("\n5. Circuit Breaker:")
    print("   - Prevent cascading failures")
    print("   - Fast failure detection")
    print("   - Used by: Netflix Hystrix, Resilience4j")
