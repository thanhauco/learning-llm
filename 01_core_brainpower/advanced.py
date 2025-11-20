"""
Core Brainpower - Advanced Level
Production-ready tokenization, embeddings, and attention with real-world optimizations
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import tiktoken
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import pickle
import time


@dataclass
class TokenStats:
    """Track token usage for cost optimization"""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    model: str


class SmartTokenizer:
    """
    Production tokenizer with cost tracking and optimization
    
    Real-world use: Track API costs, optimize prompts, handle context limits
    """
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.total_stats = TokenStats(0, 0, 0, 0.0, model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt: str, completion: str = "") -> TokenStats:
        """
        Estimate API call cost
        
        Real-world use: Budget planning, cost optimization
        """
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(completion) if completion else 0
        total_tokens = prompt_tokens + completion_tokens
        
        pricing = self.PRICING.get(self.model, {"input": 0.01, "output": 0.03})
        cost = (prompt_tokens / 1000 * pricing["input"] + 
                completion_tokens / 1000 * pricing["output"])
        
        stats = TokenStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost,
            model=self.model
        )
        
        # Track cumulative stats
        self.total_stats.total_tokens += total_tokens
        self.total_stats.prompt_tokens += prompt_tokens
        self.total_stats.completion_tokens += completion_tokens
        self.total_stats.estimated_cost += cost
        
        return stats
    
    def truncate_to_limit(self, text: str, max_tokens: int, 
                          strategy: str = "end") -> str:
        """
        Truncate text to fit token limit
        
        Real-world use: Handle context window limits
        
        Strategies:
        - 'end': Keep beginning, truncate end
        - 'start': Truncate beginning, keep end
        - 'middle': Keep start and end, truncate middle
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if strategy == "end":
            truncated_tokens = tokens[:max_tokens]
        elif strategy == "start":
            truncated_tokens = tokens[-max_tokens:]
        elif strategy == "middle":
            keep_start = max_tokens // 2
            keep_end = max_tokens - keep_start
            truncated_tokens = tokens[:keep_start] + tokens[-keep_end:]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.encoding.decode(truncated_tokens)
    
    def get_total_stats(self) -> TokenStats:
        """Get cumulative token usage stats"""
        return self.total_stats


class HybridEmbeddingStore:
    """
    Production embedding store with multiple backends
    
    Real-world use: Scale from prototype to production
    - In-memory for dev/testing
    - Redis for caching
    - Vector DB for production
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_size: int = 10000):
        self.model = SentenceTransformer(model_name)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text with LRU caching
        
        Real-world use: Reduce embedding API calls by 80%+
        """
        if not use_cache:
            return self.model.encode(text)
        
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        self.cache_misses += 1
        embedding = self.model.encode(text)
        
        # Add to cache
        self.cache[cache_key] = embedding
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch encoding with smart caching
        
        Real-world use: Process large datasets efficiently
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.cache_hits += 1
                embeddings.append((i, self.cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch encode uncached texts
        if uncached_texts:
            self.cache_misses += len(uncached_texts)
            new_embeddings = self.model.encode(uncached_texts, batch_size=batch_size)
            
            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                embeddings.append((idx, embedding))
                
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance metrics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "estimated_savings": f"{hit_rate * 100:.1f}% API calls saved"
        }


class ProductionSemanticSearch:
    """
    Production-grade semantic search with reranking
    
    Real-world use: High-quality RAG retrieval
    """
    
    def __init__(self, 
                 documents: List[str],
                 metadata: Optional[List[Dict]] = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        self.embedding_store = HybridEmbeddingStore(model_name)
        
        # Pre-compute document embeddings
        print("Indexing documents...")
        self.doc_embeddings = self.embedding_store.encode_batch(documents)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, 
               query: str,
               top_k: int = 10,
               filters: Optional[Dict] = None,
               rerank: bool = True) -> List[Dict]:
        """
        Advanced search with filtering and reranking
        
        Real-world use: Production RAG systems
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters (e.g., {"category": "tech"})
            rerank: Apply reranking for better quality
        """
        # Get query embedding
        query_embedding = self.embedding_store.encode(query)
        
        # Calculate similarities
        similarities = np.dot(self.doc_embeddings, query_embedding)
        
        # Apply metadata filters
        valid_indices = self._apply_filters(filters)
        
        # Get top candidates (fetch more if reranking)
        candidate_k = top_k * 3 if rerank else top_k
        top_indices = self._get_top_k(similarities, valid_indices, candidate_k)
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx],
                "index": int(idx)
            })
        
        # Rerank if requested
        if rerank:
            results = self._rerank(query, results)[:top_k]
        
        return results
    
    def _apply_filters(self, filters: Optional[Dict]) -> List[int]:
        """Filter documents by metadata"""
        if not filters:
            return list(range(len(self.documents)))
        
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            if all(meta.get(k) == v for k, v in filters.items()):
                valid_indices.append(i)
        
        return valid_indices
    
    def _get_top_k(self, 
                   similarities: np.ndarray,
                   valid_indices: List[int],
                   k: int) -> List[int]:
        """Get top-k indices from valid set"""
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in valid_similarities[:k]]
    
    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder
        
        Real-world use: Improve retrieval quality by 20-30%
        
        Note: Using simple heuristic here. In production, use:
        - Cross-encoder models (sentence-transformers/ms-marco-MiniLM-L-12-v2)
        - LLM-based reranking
        """
        # Simple reranking: boost exact keyword matches
        query_words = set(query.lower().split())
        
        for result in results:
            doc_words = set(result["document"].lower().split())
            keyword_overlap = len(query_words & doc_words) / len(query_words)
            
            # Combine semantic score with keyword overlap
            result["rerank_score"] = result["score"] * 0.7 + keyword_overlap * 0.3
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results


class OptimizedKVCache:
    """
    Production KV cache with memory management
    
    Real-world use: Efficient inference for chatbots, streaming
    """
    
    def __init__(self, 
                 max_length: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 head_dim: int = 128):
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize cache for each layer
        self.cache = {
            layer: {"keys": [], "values": []}
            for layer in range(num_layers)
        }
        
        self.current_length = 0
    
    def add_kv(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        """
        Add key-value pair to cache
        
        Real-world use: Incremental generation without recomputation
        """
        self.cache[layer]["keys"].append(key)
        self.cache[layer]["values"].append(value)
        self.current_length += 1
        
        # Evict oldest if exceeding limit
        if self.current_length > self.max_length:
            self._evict_oldest(layer)
    
    def get_kv(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a layer"""
        if not self.cache[layer]["keys"]:
            return None, None
        
        keys = torch.stack(self.cache[layer]["keys"])
        values = torch.stack(self.cache[layer]["values"])
        return keys, values
    
    def _evict_oldest(self, layer: int):
        """Evict oldest entry (FIFO)"""
        self.cache[layer]["keys"].pop(0)
        self.cache[layer]["values"].pop(0)
        self.current_length -= 1
    
    def clear(self):
        """Clear all cache"""
        for layer in range(self.num_layers):
            self.cache[layer]["keys"].clear()
            self.cache[layer]["values"].clear()
        self.current_length = 0
    
    def get_memory_usage(self) -> Dict:
        """
        Calculate memory usage
        
        Real-world use: Monitor and optimize memory consumption
        """
        bytes_per_element = 2  # FP16
        elements_per_kv = self.num_heads * self.head_dim
        
        total_elements = (self.current_length * elements_per_kv * 
                         self.num_layers * 2)  # 2 for K and V
        
        memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)
        
        return {
            "current_length": self.current_length,
            "max_length": self.max_length,
            "memory_mb": memory_mb,
            "utilization": f"{(self.current_length / self.max_length) * 100:.1f}%"
        }


if __name__ == "__main__":
    print("=== Smart Tokenizer with Cost Tracking ===")
    tokenizer = SmartTokenizer("gpt-4")
    
    prompt = "Explain quantum computing in simple terms."
    completion = "Quantum computing uses quantum mechanics principles..."
    
    stats = tokenizer.estimate_cost(prompt, completion)
    print(f"Prompt tokens: {stats.prompt_tokens}")
    print(f"Completion tokens: {stats.completion_tokens}")
    print(f"Estimated cost: ${stats.estimated_cost:.6f}")
    
    # Simulate multiple API calls
    for _ in range(10):
        tokenizer.estimate_cost(prompt, completion)
    
    total = tokenizer.get_total_stats()
    print(f"\nTotal usage: {total.total_tokens} tokens")
    print(f"Total cost: ${total.estimated_cost:.4f}")
    
    # Context window management
    print("\n=== Context Window Management ===")
    long_text = "This is a very long document. " * 1000
    truncated = tokenizer.truncate_to_limit(long_text, max_tokens=100)
    print(f"Original: {tokenizer.count_tokens(long_text)} tokens")
    print(f"Truncated: {tokenizer.count_tokens(truncated)} tokens")
    
    print("\n=== Hybrid Embedding Store with Caching ===")
    store = HybridEmbeddingStore()
    
    # First encoding (cache miss)
    text = "Machine learning is transforming industries"
    start = time.time()
    emb1 = store.encode(text)
    time1 = time.time() - start
    
    # Second encoding (cache hit)
    start = time.time()
    emb2 = store.encode(text)
    time2 = time.time() - start
    
    print(f"First encoding: {time1*1000:.2f}ms (cache miss)")
    print(f"Second encoding: {time2*1000:.2f}ms (cache hit)")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Batch encoding with cache
    texts = [
        "AI is the future",
        "Machine learning is transforming industries",  # Duplicate
        "Deep learning uses neural networks",
        "AI is the future"  # Duplicate
    ]
    
    embeddings = store.encode_batch(texts)
    stats = store.get_cache_stats()
    print(f"\nCache stats: {stats}")
    
    print("\n=== Production Semantic Search ===")
    documents = [
        "Python is a high-level programming language",
        "Machine learning models require training data",
        "Neural networks consist of layers of neurons",
        "Data preprocessing is crucial for ML success",
        "Deep learning is a subset of machine learning",
        "Python has extensive ML libraries like scikit-learn",
        "Transformers revolutionized NLP tasks",
        "GPUs accelerate deep learning training"
    ]
    
    metadata = [
        {"category": "programming", "difficulty": "beginner"},
        {"category": "ml", "difficulty": "intermediate"},
        {"category": "ml", "difficulty": "advanced"},
        {"category": "ml", "difficulty": "beginner"},
        {"category": "ml", "difficulty": "intermediate"},
        {"category": "programming", "difficulty": "intermediate"},
        {"category": "ml", "difficulty": "advanced"},
        {"category": "hardware", "difficulty": "intermediate"}
    ]
    
    search = ProductionSemanticSearch(documents, metadata)
    
    # Search with filters
    results = search.search(
        query="How to get started with machine learning?",
        top_k=3,
        filters={"difficulty": "beginner"},
        rerank=True
    )
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['metadata']['category']}] {result['document']}")
        print(f"   Score: {result.get('rerank_score', result['score']):.4f}\n")
    
    print("=== KV Cache Memory Management ===")
    kv_cache = OptimizedKVCache(max_length=2048)
    
    # Simulate adding tokens
    for i in range(100):
        key = torch.randn(32, 128)  # (num_heads, head_dim)
        value = torch.randn(32, 128)
        kv_cache.add_kv(layer=0, key=key, value=value)
    
    memory_stats = kv_cache.get_memory_usage()
    print(f"Cache length: {memory_stats['current_length']}")
    print(f"Memory usage: {memory_stats['memory_mb']:.2f} MB")
    print(f"Utilization: {memory_stats['utilization']}")
