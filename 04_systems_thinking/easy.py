"""
Systems Thinking - Easy Level
Basic latency measurement and simple caching
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from collections import OrderedDict
import hashlib


class LatencyTracker:
    """
    Track API latency
    
    Real-world: Monitor system performance
    """
    
    def __init__(self):
        self.latencies = []
    
    def measure(self, func: Callable) -> Callable:
        """Decorator to measure function latency"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            latency = time.time() - start
            
            self.latencies.append(latency)
            return result
        
        return wrapper
    
    def get_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        import numpy as np
        
        return {
            "count": len(self.latencies),
            "mean": np.mean(self.latencies),
            "median": np.median(self.latencies),
            "p95": np.percentile(self.latencies, 95),
            "p99": np.percentile(self.latencies, 99),
            "min": np.min(self.latencies),
            "max": np.max(self.latencies)
        }


class SimpleCache:
    """
    Simple LRU cache
    
    Real-world: Cache API responses to reduce costs
    """
    
    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def cached(self, func: Callable) -> Callable:
        """Decorator to cache function results"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(*args, **kwargs)
            
            # Check cache
            result = self.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        
        return wrapper
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "max_size": self.max_size
        }


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm
    
    Real-world: Stay within API rate limits
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
    
    def _refill_tokens(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on time elapsed
        tokens_to_add = elapsed * (self.requests_per_minute / 60)
        self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
        self.last_update = now
    
    def allow_request(self) -> bool:
        """Check if request is allowed"""
        self._refill_tokens()
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        return False
    
    def wait_time(self) -> float:
        """Calculate wait time until next request is allowed"""
        self._refill_tokens()
        
        if self.tokens >= 1:
            return 0.0
        
        tokens_needed = 1 - self.tokens
        return tokens_needed * (60 / self.requests_per_minute)


if __name__ == "__main__":
    print("=== Latency Tracking ===\n")
    
    tracker = LatencyTracker()
    
    @tracker.measure
    def slow_api_call(delay: float):
        """Simulate API call"""
        time.sleep(delay)
        return "result"
    
    # Simulate multiple API calls
    for delay in [0.1, 0.15, 0.2, 0.12, 0.18, 0.25, 0.11, 0.19, 0.22, 0.14]:
        slow_api_call(delay)
    
    stats = tracker.get_stats()
    print("Latency Statistics:")
    print(f"  Mean: {stats['mean']*1000:.1f}ms")
    print(f"  Median: {stats['median']*1000:.1f}ms")
    print(f"  P95: {stats['p95']*1000:.1f}ms")
    print(f"  P99: {stats['p99']*1000:.1f}ms")
    print(f"  Min: {stats['min']*1000:.1f}ms")
    print(f"  Max: {stats['max']*1000:.1f}ms")
    
    print("\n=== Simple Caching ===\n")
    
    cache = SimpleCache(max_size=3)
    
    @cache.cached
    def expensive_computation(x: int) -> int:
        """Simulate expensive computation"""
        time.sleep(0.1)
        return x * x
    
    # First calls - cache misses
    print("First calls (cache misses):")
    start = time.time()
    for i in range(3):
        result = expensive_computation(i)
        print(f"  expensive_computation({i}) = {result}")
    first_time = time.time() - start
    
    # Second calls - cache hits
    print("\nSecond calls (cache hits):")
    start = time.time()
    for i in range(3):
        result = expensive_computation(i)
        print(f"  expensive_computation({i}) = {result}")
    second_time = time.time() - start
    
    print(f"\nFirst run: {first_time:.3f}s")
    print(f"Second run: {second_time:.3f}s")
    print(f"Speedup: {first_time / second_time:.1f}x")
    
    cache_stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    
    print("\n=== Rate Limiting ===\n")
    
    limiter = RateLimiter(requests_per_minute=10)
    
    print("Attempting 15 requests (limit: 10/min):")
    allowed = 0
    denied = 0
    
    for i in range(15):
        if limiter.allow_request():
            allowed += 1
            print(f"  Request {i+1}: ✓ Allowed")
        else:
            denied += 1
            wait = limiter.wait_time()
            print(f"  Request {i+1}: ✗ Denied (wait {wait:.1f}s)")
    
    print(f"\nAllowed: {allowed}")
    print(f"Denied: {denied}")
    
    print("\n=== Real-World Example ===")
    print("OpenAI API limits:")
    print("  GPT-4: 10,000 tokens/min")
    print("  GPT-3.5: 90,000 tokens/min")
    print("\nWithout rate limiting:")
    print("  → 429 errors")
    print("  → Failed requests")
    print("  → Wasted API calls")
    print("\nWith rate limiting:")
    print("  → Smooth request flow")
    print("  → No failed requests")
    print("  → Optimal throughput")
