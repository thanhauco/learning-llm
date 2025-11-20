"""
Core Brainpower - Advanced Level
Production-ready tokenization, embeddings, and attention optimization
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import tiktoken
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import pickle
import time


@dataclass
class TokenStats:
    """Token usage statistics for cost tracking"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    model: str


class ProductionTokenizer:
    """
    Production-grade tokenizer with cost tracking and optimization
    
    Real-world scenario: Track token usage across multiple API calls
    to optimize costs and stay within budget
    """
    
    PRICING = {
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
    }
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.total_cost = 0.0
        self.call_history: List[TokenStats] = []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt: str, completion: str = "") -> TokenStats:
        """
        Estimate cost for a prompt/completion pair
        
        Real-world use: Budget planning and cost optimization
        """
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(completion) if completion else 0
        total_tokens = prompt_tokens + completion_tokens
        
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4"])
        cost = (prompt_tokens * pricing["input"] + 
                completion_tokens * pricing["output"])
        
        stats = TokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            model=self.model
        )
        
        self.call_history.append(stats)
        self.total_cost += cost
        
        return stats
    
    def optimize_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate prompt to fit within token limit
        
        Real-world scenario: Handling long documents that exceed context window
        """
        tokens = self.encoding.encode(prompt)
        
        if len(tokens) <= max_tokens:
            return prompt
        
        # Truncate from the middle, keep beginning and end
        keep_start = max_tokens // 2
        keep_end = max_tokens - keep_start
        
        truncated_tokens = tokens[:keep_start] + tokens[-keep_end:]
        return self.encoding.decode(truncated_tokens)
    
    def get_report(self) -> Dict:
        """Generate cost report"""
        return {
            "total_calls": len(self.call_history),
            "total_tokens": sum(s.total_tokens for s in self.call_history),
            "total_cost": self.total_cost,
            "avg_tokens_per_call": np.mean([s.total_tokens for s in self.call_history]) if self.call_history else 0
        }


class EmbeddingPipeline:
    """
    Production embedding pipeline with caching and batch processing
    
    Real-world scenario: Process millions of documents efficiently
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = ".cache/embeddings",
        device: str = "cpu"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = cache_dir
        self.cache: OrderedDict = OrderedDict()
        self.max_cache_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text with caching
        
        Real-world benefit: 10-100x speedup for repeated queries
        """
        if not use_cache:
            return self.model.encode(text)
        
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        self.cache_misses += 1
        embedding = self.model.encode(text)
        
        # Add to cache
        self.cache[cache_key] = embedding
        
        # Evict oldest if cache is full
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Batch encoding with optimal batch size
        
        Real-world scenario: Process large document collections
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance metrics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "estimated_speedup": f"{1 / (1 - hit_rate):.2f}x" if hit_rate < 1 else "∞"
        }


if __name__ == "__main__":
    print("=== Production Tokenizer with Cost Tracking ===")
    tokenizer = ProductionTokenizer(model="gpt-4")
    
    # Simulate multiple API calls
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a Python function to sort a list",
        "What are the benefits of exercise?" * 100
    ]
    
    for prompt in prompts:
        stats = tokenizer.estimate_cost(prompt, completion="A" * 500)
        print(f"Tokens: {stats.total_tokens}, Cost: ${stats.estimated_cost:.4f}")
    
    report = tokenizer.get_report()
    print(f"\nTotal cost: ${report['total_cost']:.4f}")
    print(f"Total tokens: {report['total_tokens']}")
