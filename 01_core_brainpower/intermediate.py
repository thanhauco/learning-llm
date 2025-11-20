"""
Core Brainpower - Intermediate Level
Advanced embeddings, batch processing, and attention basics
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import time


class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
    
    def encode(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text not in self.cache:
            self.cache[text] = self.model.encode(text)
        return self.cache[text]
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encoding for efficiency"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)


class SemanticSearch:
    """Efficient semantic search with multiple distance metrics"""
    
    def __init__(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(documents, show_progress_bar=False)
    
    def search(self, query: str, top_k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        """
        Search for most similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            metric: 'cosine', 'dot', or 'euclidean'
        """
        query_embedding = self.model.encode(query)
        
        if metric == "cosine":
            scores = self._cosine_similarity(query_embedding, self.doc_embeddings)
        elif metric == "dot":
            scores = self._dot_product(query_embedding, self.doc_embeddings)
        elif metric == "euclidean":
            scores = -self._euclidean_distance(query_embedding, self.doc_embeddings)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.documents[idx], scores[idx])
            for idx in top_indices
        ]
        
        return results
    
    @staticmethod
    def _cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity"""
        query_norm = query / np.linalg.norm(query)
        docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
        return np.dot(docs_norm, query_norm)
    
    @staticmethod
    def _dot_product(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """Dot product similarity (faster, assumes normalized vectors)"""
        return np.dot(docs, query)
    
    @staticmethod
    def _euclidean_distance(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """L2 distance (lower is better)"""
        return np.linalg.norm(docs - query, axis=1)


def simple_attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Simplified attention mechanism
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    This is the core of transformer models
    """
    d_k = query.shape[-1]
    
    # Calculate attention scores
    scores = np.dot(query, keys.T) / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Weighted sum of values
    output = np.dot(attention_weights, values)
    
    return output, attention_weights


def multi_head_attention_simulation(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    num_heads: int = 4
) -> np.ndarray:
    """
    Simulate multi-head attention
    
    Multiple attention heads learn different aspects of relationships
    """
    d_model = query.shape[-1]
    d_k = d_model // num_heads
    
    outputs = []
    
    for head in range(num_heads):
        # Split into heads
        start_idx = head * d_k
        end_idx = start_idx + d_k
        
        q_head = query[start_idx:end_idx]
        k_head = keys[:, start_idx:end_idx]
        v_head = values[:, start_idx:end_idx]
        
        # Apply attention
        output, _ = simple_attention(q_head, k_head, v_head)
        outputs.append(output)
    
    # Concatenate heads
    return np.concatenate(outputs)


class KVCacheSimulator:
    """
    Simulate KV cache for efficient inference
    
    In autoregressive generation, we cache key/value pairs
    to avoid recomputing attention for previous tokens
    """
    
    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
        self.keys_cache = []
        self.values_cache = []
    
    def add(self, key: np.ndarray, value: np.ndarray):
        """Add new key-value pair to cache"""
        self.keys_cache.append(key)
        self.values_cache.append(value)
        
        # Evict oldest if exceeding max length
        if len(self.keys_cache) > self.max_length:
            self.keys_cache.pop(0)
            self.values_cache.pop(0)
    
    def get_cached_kv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all cached keys and values"""
        if not self.keys_cache:
            return np.array([]), np.array([])
        
        keys = np.stack(self.keys_cache)
        values = np.stack(self.values_cache)
        return keys, values
    
    def clear(self):
        """Clear cache"""
        self.keys_cache = []
        self.values_cache = []


if __name__ == "__main__":
    # Example 1: Semantic search with different metrics
    print("=== Semantic Search ===")
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language for data science",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images"
    ]
    
    search = SemanticSearch(documents)
    query = "How do computers understand language?"
    
    results = search.search(query, top_k=3)
    print(f"Query: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | {doc}")
    
    # Example 2: Attention mechanism
    print("\n=== Attention Mechanism ===")
    # Simulate 3 tokens with 8-dimensional embeddings
    query = np.random.randn(8)
    keys = np.random.randn(3, 8)
    values = np.random.randn(3, 8)
    
    output, weights = simple_attention(query, keys, values)
    print(f"Attention weights: {weights}")
    print(f"Output shape: {output.shape}")
    
    # Example 3: KV Cache efficiency
    print("\n=== KV Cache Efficiency ===")
    cache = KVCacheSimulator(max_length=10)
    
    # Simulate token generation
    print("Without cache: recompute all tokens each time")
    start = time.time()
    for i in range(20):
        # Simulate computing attention for all previous tokens
        _ = np.random.randn(i + 1, 64)
    no_cache_time = time.time() - start
    
    print("With cache: only compute new token")
    start = time.time()
    for i in range(20):
        # Only compute new token, use cached previous tokens
        new_key = np.random.randn(64)
        new_value = np.random.randn(64)
        cache.add(new_key, new_value)
        _ = cache.get_cached_kv()
    cache_time = time.time() - start
    
    print(f"Without cache: {no_cache_time:.4f}s")
    print(f"With cache: {cache_time:.4f}s")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x")
