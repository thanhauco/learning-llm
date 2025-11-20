"""
Systems Thinking - Intermediate Level
Advanced caching, load balancing, and request routing
"""

import time
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: any
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None


class MultiLevelCache:
    """
    Multi-level caching strategy
    
    L1: In-memory (fast, small)
    L2: Redis (medium, larger)
    L3: Database (slow, unlimited)
    
    Real-world: Used by CDNs, web apps
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l1_ttl: float = 60,
        l2_ttl: float = 3600
    ):
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        
        # Metrics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[any]:
        """
        Get value with multi-level lookup
        
        1. Check L1 (in-memory)
        2. Check L2 (Redis)
        3. Return None (cache miss)
        """
        now = time.time()
        
        # L1 lookup
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if entry.ttl is None or (now - entry.created_at) < entry.ttl:
                entry.access_count += 1
                entry.last_accessed = now
                self.l1_hits += 1
                return entry.value
            else:
                del self.l1_cache[key]
        
        # L2 lookup
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if entry.ttl is None or (now - entry.created_at) < entry.ttl:
                entry.access_count += 1
                entry.last_accessed = now
                self.l2_hits += 1
                
                # Promote to L1
                self._set_l1(key, entry.value, self.l1_ttl)
                return entry.value
            else:
                del self.l2_cache[key]
        
        # Cache miss
        self.misses += 1
        return None
    
    def set(self, key: str, value: any):
        """Set value in both L1 and L2"""
        self._set_l1(key, value, self.l1_ttl)
        self._set_l2(key, value, self.l2_ttl)
    
    def _set_l1(self, key: str, value: any, ttl: float):
        """Set in L1 cache"""
        if len(self.l1_cache) >= self.l1_size:
            # Evict LRU
            lru_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k].last_accessed
            )
            del self.l1_cache[lru_key]
        
        self.l1_cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
    
    def _set_l2(self, key: str, value: any, ttl: float):
        """Set in L2 cache"""
        if len(self.l2_cache) >= self.l2_size:
            # Evict LRU
            lru_key = min(
                self.l2_cache.keys(),
                key=lambda k: self.l2_cache[k].last_accessed
            )
            del self.l2_cache[lru_key]
        
        self.l2_cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "l1_hit_rate": self.l1_hits / total_requests if total_requests > 0 else 0,
            "l2_hit_rate": self.l2_hits / total_requests if total_requests > 0 else 0,
            "total_hit_rate": (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache)
        }


class LoadBalancer:
    """
    Load balancer with multiple strategies
    
    Real-world: Distribute requests across multiple servers
    """
    
    def __init__(self, servers: List[str], strategy: str = "round_robin"):
        self.servers = servers
        self.strategy = strategy
        self.current_index = 0
        self.server_loads: Dict[str, int] = {s: 0 for s in servers}
        self.server_latencies: Dict[str, List[float]] = {s: [] for s in servers}
    
    def get_server(self) -> str:
        """
        Get next server based on strategy
        
        Strategies:
        - round_robin: Rotate through servers
        - least_connections: Server with fewest active connections
        - least_latency: Server with lowest average latency
        - weighted: Based on server capacity
        """
        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "least_connections":
            return self._least_connections()
        elif self.strategy == "least_latency":
            return self._least_latency()
        else:
            return self._round_robin()
    
    def _round_robin(self) -> str:
        """Simple round-robin"""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def _least_connections(self) -> str:
        """Route to server with fewest connections"""
        return min(self.server_loads.keys(), key=lambda s: self.server_loads[s])
    
    def _least_latency(self) -> str:
        """Route to server with lowest latency"""
        avg_latencies = {}
        for server, latencies in self.server_latencies.items():
            if latencies:
                avg_latencies[server] = np.mean(latencies[-10:])  # Last 10 requests
            else:
                avg_latencies[server] = 0
        
        return min(avg_latencies.keys(), key=lambda s: avg_latencies[s])
    
    def record_request(self, server: str, latency: float):
        """Record request metrics"""
        self.server_loads[server] += 1
        self.server_latencies[server].append(latency)
    
    def complete_request(self, server: str):
        """Mark request as complete"""
        self.server_loads[server] = max(0, self.server_loads[server] - 1)


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests
    - HALF_OPEN: Testing if service recovered
    
    Real-world: Prevent cascading failures
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker
        
        Returns: (success, result)
        """
        if self.state == "OPEN":
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                return False, "Circuit breaker OPEN"
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return True, result
        except Exception as e:
            self._on_failure()
            return False, str(e)
    
    def _on_success(self):
        """Handle successful request"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"


class RequestRouter:
    """
    Intelligent request routing
    
    Routes requests based on:
    - Model type
    - User tier
    - Geographic location
    - Load
    """
    
    def __init__(self):
        self.routes: Dict[str, List[str]] = {}
        self.user_tiers: Dict[str, str] = {}
    
    def add_route(self, pattern: str, servers: List[str]):
        """Add routing rule"""
        self.routes[pattern] = servers
    
    def route(self, request: Dict) -> str:
        """
        Route request to appropriate server
        
        Priority:
        1. User tier (premium users get better servers)
        2. Model type (different models on different servers)
        3. Load balancing
        """
        user_id = request.get("user_id")
        model = request.get("model", "default")
        
        # Check user tier
        tier = self.user_tiers.get(user_id, "free")
        
        if tier == "premium":
            route_key = f"premium_{model}"
        else:
            route_key = model
        
        # Get servers for route
        servers = self.routes.get(route_key, self.routes.get("default", []))
        
        if not servers:
            raise ValueError("No servers available")
        
        # Simple round-robin for now
        return servers[hash(user_id) % len(servers)]


if __name__ == "__main__":
    print("=== Multi-Level Cache ===\n")
    
    cache = MultiLevelCache(l1_size=3, l2_size=10)
    
    # Simulate access pattern
    keys = ["key1", "key2", "key3", "key4", "key5"]
    
    # First access - all misses
    print("First access (cold cache):")
    for key in keys:
        result = cache.get(key)
        if result is None:
            cache.set(key, f"value_{key}")
            print(f"  {key}: MISS")
    
    # Second access - L1 hits for recent keys
    print("\nSecond access (warm cache):")
    for key in ["key3", "key4", "key5", "key1", "key2"]:
        result = cache.get(key)
        print(f"  {key}: {'HIT' if result else 'MISS'}")
    
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  L1 hit rate: {stats['l1_hit_rate']:.1%}")
    print(f"  L2 hit rate: {stats['l2_hit_rate']:.1%}")
    print(f"  Total hit rate: {stats['total_hit_rate']:.1%}")
    
    print("\n=== Load Balancer ===\n")
    
    servers = ["server1", "server2", "server3"]
    
    # Test different strategies
    strategies = ["round_robin", "least_connections", "least_latency"]
    
    for strategy in strategies:
        print(f"{strategy.upper()}:")
        lb = LoadBalancer(servers, strategy=strategy)
        
        # Simulate 10 requests
        for i in range(10):
            server = lb.get_server()
            latency = np.random.uniform(0.1, 0.5)
            lb.record_request(server, latency)
            print(f"  Request {i+1} → {server}")
            
            # Simulate request completion
            if i % 3 == 0:
                lb.complete_request(server)
        print()
    
    print("=== Circuit Breaker ===\n")
    
    breaker = CircuitBreaker(failure_threshold=3, timeout=2)
    
    # Simulate flaky service
    call_count = 0
    def flaky_service():
        global call_count
        call_count += 1
        if call_count <= 5:
            raise Exception("Service unavailable")
        return "Success"
    
    print("Calling flaky service:")
    for i in range(10):
        success, result = breaker.call(flaky_service)
        print(f"  Call {i+1}: State={breaker.state}, Success={success}, Result={result}")
        time.sleep(0.1)
    
    print("\n=== Request Router ===\n")
    
    router = RequestRouter()
    
    # Configure routes
    router.add_route("default", ["server1", "server2"])
    router.add_route("gpt-4", ["gpu-server1", "gpu-server2"])
    router.add_route("premium_gpt-4", ["premium-gpu1", "premium-gpu2"])
    
    # Set user tiers
    router.user_tiers["user1"] = "free"
    router.user_tiers["user2"] = "premium"
    
    # Route requests
    requests = [
        {"user_id": "user1", "model": "gpt-4"},
        {"user_id": "user2", "model": "gpt-4"},
        {"user_id": "user1", "model": "default"},
    ]
    
    for req in requests:
        server = router.route(req)
        print(f"User: {req['user_id']}, Model: {req['model']} → {server}")
    
    print("\n=== Key Takeaways ===\n")
    print("1. Multi-level caching: 80%+ hit rates with proper TTL")
    print("2. Load balancing: Distribute load evenly or intelligently")
    print("3. Circuit breaker: Prevent cascading failures")
    print("4. Request routing: Optimize for user tier and model type")
    print("5. Combine all for production-grade systems")
