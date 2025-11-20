"""
Inference Optimization - Easy Level
Understanding inference optimization and vLLM basics
"""

import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class InferenceRequest:
    """Single inference request"""
    request_id: str
    prompt: str
    max_tokens: int
    arrival_time: float


class NaiveInferenceEngine:
    """
    Naive inference: Process one request at a time
    
    Problems:
    - Low GPU utilization
    - High latency for queued requests
    - Wasted compute during generation
    """
    
    def __init__(self, model_name: str = "llama-7b"):
        self.model_name = model_name
        self.requests_processed = 0
    
    def generate(self, prompt: str, max_tokens: int = 100) -> Dict:
        """
        Generate tokens one at a time
        
        Time per token: ~50ms
        For 100 tokens: 5 seconds
        """
        start_time = time.time()
        
        # Simulate token generation
        time_per_token = 0.05  # 50ms
        total_time = max_tokens * time_per_token
        
        # Simulate generation
        time.sleep(min(total_time, 0.1))  # Cap for demo
        
        self.requests_processed += 1
        
        return {
            "text": f"Generated {max_tokens} tokens",
            "tokens": max_tokens,
            "latency": time.time() - start_time,
            "throughput": max_tokens / (time.time() - start_time)
        }


class BatchedInferenceEngine:
    """
    Static batching: Process multiple requests together
    
    Improvements:
    - Better GPU utilization
    - Higher throughput
    
    Problems:
    - All requests must finish together
    - Padding waste for different lengths
    - Head-of-line blocking
    """
    
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.queue: deque = deque()
    
    def add_request(self, request: InferenceRequest):
        """Add request to queue"""
        self.queue.append(request)
    
    def process_batch(self) -> List[Dict]:
        """
        Process batch of requests
        
        All requests finish at the same time
        """
        if len(self.queue) < self.batch_size:
            return []
        
        # Get batch
        batch = [self.queue.popleft() for _ in range(self.batch_size)]
        
        # Find max tokens in batch
        max_tokens = max(req.max_tokens for req in batch)
        
        # All requests take time of longest request
        time_per_token = 0.05
        total_time = max_tokens * time_per_token
        
        results = []
        for req in batch:
            # Wasted compute for shorter requests
            wasted_tokens = max_tokens - req.max_tokens
            
            results.append({
                "request_id": req.request_id,
                "tokens": req.max_tokens,
                "latency": total_time,
                "wasted_tokens": wasted_tokens
            })
        
        return results


class ContinuousBatchingEngine:
    """
    Continuous batching: Add/remove requests dynamically
    
    Used by: vLLM, TGI, TensorRT-LLM
    
    Improvements:
    - No head-of-line blocking
    - Better GPU utilization
    - Lower latency
    - Higher throughput
    """
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.active_requests: List[InferenceRequest] = []
        self.queue: deque = deque()
    
    def add_request(self, request: InferenceRequest):
        """Add request to queue"""
        self.queue.append(request)
    
    def step(self) -> List[Dict]:
        """
        Single generation step
        
        1. Add new requests to batch (up to max_batch_size)
        2. Generate one token for all active requests
        3. Remove finished requests
        """
        # Add new requests
        while len(self.active_requests) < self.max_batch_size and self.queue:
            self.active_requests.append(self.queue.popleft())
        
        if not self.active_requests:
            return []
        
        # Generate one token for all active requests
        time_per_token = 0.05
        
        finished = []
        remaining = []
        
        for req in self.active_requests:
            req.max_tokens -= 1
            
            if req.max_tokens <= 0:
                finished.append({
                    "request_id": req.request_id,
                    "status": "completed"
                })
            else:
                remaining.append(req)
        
        self.active_requests = remaining
        
        return finished


def compare_inference_strategies():
    """
    Compare different inference strategies
    
    Metrics:
    - Throughput (tokens/second)
    - Latency (time to first token, total time)
    - GPU utilization
    """
    print("=== Inference Strategy Comparison ===\n")
    
    # Scenario: 100 requests, varying lengths
    num_requests = 100
    avg_tokens = 100
    
    print("Scenario: 100 requests, avg 100 tokens each\n")
    
    # 1. Naive (sequential)
    print("1. Naive Sequential Processing:")
    time_per_token = 0.05
    total_time_naive = num_requests * avg_tokens * time_per_token
    throughput_naive = (num_requests * avg_tokens) / total_time_naive
    
    print(f"   Total time: {total_time_naive:.1f}s")
    print(f"   Throughput: {throughput_naive:.1f} tokens/s")
    print(f"   Avg latency: {total_time_naive / num_requests:.1f}s")
    print(f"   GPU utilization: ~20%")
    
    # 2. Static batching (batch_size=8)
    print("\n2. Static Batching (batch_size=8):")
    batch_size = 8
    num_batches = num_requests // batch_size
    total_time_batched = num_batches * avg_tokens * time_per_token
    throughput_batched = (num_requests * avg_tokens) / total_time_batched
    
    print(f"   Total time: {total_time_batched:.1f}s")
    print(f"   Throughput: {throughput_batched:.1f} tokens/s")
    print(f"   Speedup: {total_time_naive / total_time_batched:.1f}x")
    print(f"   GPU utilization: ~60%")
    print(f"   Problem: Head-of-line blocking")
    
    # 3. Continuous batching
    print("\n3. Continuous Batching (vLLM):")
    # Assumes optimal batching with no waste
    total_time_continuous = (num_requests * avg_tokens) / 32 * time_per_token
    throughput_continuous = (num_requests * avg_tokens) / total_time_continuous
    
    print(f"   Total time: {total_time_continuous:.1f}s")
    print(f"   Throughput: {throughput_continuous:.1f} tokens/s")
    print(f"   Speedup: {total_time_naive / total_time_continuous:.1f}x")
    print(f"   GPU utilization: ~90%")
    print(f"   No head-of-line blocking!")


def calculate_kv_cache_memory(
    batch_size: int,
    sequence_length: int,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    precision: str = "fp16"
) -> Dict:
    """
    Calculate KV cache memory requirements
    
    KV cache is the bottleneck for inference
    vLLM's PagedAttention reduces waste by 50%+
    """
    bytes_per_element = {"fp16": 2, "fp32": 4, "int8": 1}[precision]
    
    # KV cache: 2 (K and V) * batch * layers * seq_len * heads * head_dim
    kv_cache_elements = 2 * batch_size * num_layers * sequence_length * num_heads * head_dim
    kv_cache_bytes = kv_cache_elements * bytes_per_element
    
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "kv_cache_gb": kv_cache_bytes / (1024**3),
        "per_request_mb": (kv_cache_bytes / batch_size) / (1024**2)
    }


if __name__ == "__main__":
    print("=== Inference Optimization Basics ===\n")
    
    # Compare strategies
    compare_inference_strategies()
    
    print("\n=== KV Cache Memory Analysis ===\n")
    
    # LLaMA 7B configuration
    configs = [
        {"batch_size": 1, "sequence_length": 2048},
        {"batch_size": 8, "sequence_length": 2048},
        {"batch_size": 32, "sequence_length": 2048},
        {"batch_size": 32, "sequence_length": 4096},
    ]
    
    for config in configs:
        mem = calculate_kv_cache_memory(**config)
        print(f"Batch={mem['batch_size']}, SeqLen={mem['sequence_length']}:")
        print(f"  KV cache: {mem['kv_cache_gb']:.2f} GB")
        print(f"  Per request: {mem['per_request_mb']:.1f} MB")
        print()
    
    print("=== vLLM PagedAttention Benefits ===\n")
    print("Traditional KV cache:")
    print("  - Pre-allocate max sequence length")
    print("  - Wasted memory for shorter sequences")
    print("  - Fragmentation issues")
    print("  - Typical utilization: 20-40%")
    
    print("\nvLLM PagedAttention:")
    print("  - Allocate memory in pages (like OS virtual memory)")
    print("  - No waste for shorter sequences")
    print("  - No fragmentation")
    print("  - Typical utilization: 80-90%")
    print("  - Result: 2-4x higher throughput!")
    
    print("\n=== Real-World Performance ===\n")
    print("LLaMA 7B on A100 80GB:")
    print("\nNaive Implementation:")
    print("  - Throughput: 50 tokens/s")
    print("  - Batch size: 1-4")
    print("  - GPU utilization: 20%")
    
    print("\nvLLM:")
    print("  - Throughput: 2000+ tokens/s")
    print("  - Batch size: 100+")
    print("  - GPU utilization: 90%")
    print("  - Speedup: 40x!")
    
    print("\n=== Key Takeaways ===\n")
    print("1. Continuous batching > Static batching > Sequential")
    print("2. KV cache is the memory bottleneck")
    print("3. PagedAttention reduces memory waste by 50%+")
    print("4. vLLM achieves 10-40x speedup over naive inference")
    print("5. Use vLLM/TGI for production serving")
