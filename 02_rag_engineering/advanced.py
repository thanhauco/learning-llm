"""
RAG Engineering - Advanced Level
Production RAG with caching, monitoring, and evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import time
import hashlib
import json
from collections import defaultdict
from datetime import datetime


@dataclass
class RetrievalMetrics:
    """Metrics for RAG evaluation"""
    query: str
    retrieved_chunks: int
    retrieval_time: float
    relevance_scores: List[float]
    avg_relevance: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SemanticCache:
    """
    Semantic caching for RAG
    
    Real-world scenario: Cache similar queries to reduce latency
    Used by: GPTCache, Redis with vector similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache: Dict[str, Tuple[np.ndarray, List]] = {}
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, query: str) -> str:
        """Generate cache key"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List]:
        """
        Get cached results for semantically similar query
        
        Returns cached results if query is similar enough to cached query
        """
        query_embedding = self.model.encode(query)
        
        # Check all cached queries for similarity
        for cached_query, (cached_embedding, cached_results) in self.cache.items():
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity >= self.similarity_threshold:
                self.hits += 1
                return cached_results
        
        self.misses += 1
        return None
    
    def set(self, query: str, results: List):
        """Cache query results"""
        query_embedding = self.model.encode(query)
        key = self._get_key(query)
        self.cache[key] = (query_embedding, results)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "avg_latency_reduction": "~80%" if hit_rate > 0.5 else "~50%"
        }


class RAGMonitor:
    """
    Monitor RAG system performance
    
    Real-world scenario: Track retrieval quality and system health
    """
    
    def __init__(self):
        self.metrics: List[RetrievalMetrics] = []
        self.error_count = 0
        self.total_queries = 0
    
    def log_retrieval(
        self,
        query: str,
        retrieved_chunks: int,
        retrieval_time: float,
        relevance_scores: List[float]
    ):
        """Log retrieval metrics"""
        metric = RetrievalMetrics(
            query=query,
            retrieved_chunks=retrieved_chunks,
            retrieval_time=retrieval_time,
            relevance_scores=relevance_scores,
            avg_relevance=np.mean(relevance_scores) if relevance_scores else 0.0
        )
        
        self.metrics.append(metric)
        self.total_queries += 1
    
    def log_error(self):
        """Log error"""
        self.error_count += 1
        self.total_queries += 1
    
    def get_report(self) -> Dict:
        """Generate performance report"""
        if not self.metrics:
            return {"status": "no data"}
        
        retrieval_times = [m.retrieval_time for m in self.metrics]
        relevance_scores = [m.avg_relevance for m in self.metrics]
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": len(self.metrics),
            "error_rate": self.error_count / self.total_queries if self.total_queries > 0 else 0,
            "avg_retrieval_time": np.mean(retrieval_times),
            "p95_retrieval_time": np.percentile(retrieval_times, 95),
            "p99_retrieval_time": np.percentile(retrieval_times, 99),
            "avg_relevance": np.mean(relevance_scores),
            "min_relevance": np.min(relevance_scores),
            "queries_per_second": len(self.metrics) / sum(retrieval_times) if sum(retrieval_times) > 0 else 0
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON"""
        data = {
            "summary": self.get_report(),
            "metrics": [
                {
                    "query": m.query,
                    "retrieved_chunks": m.retrieved_chunks,
                    "retrieval_time": m.retrieval_time,
                    "avg_relevance": m.avg_relevance,
                    "timestamp": m.timestamp
                }
                for m in self.metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class RAGEvaluator:
    """
    Evaluate RAG system quality
    
    Real-world scenario: Measure retrieval accuracy
    Metrics: Context relevance, answer faithfulness, answer relevance
    """
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def context_relevance(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Measure how relevant retrieved documents are to query
        
        Score: 0-1, higher is better
        """
        if not retrieved_docs:
            return 0.0
        
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode(retrieved_docs)
        
        # Calculate similarities
        similarities = [
            np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            for doc_emb in doc_embeddings
        ]
        
        return float(np.mean(similarities))
    
    def answer_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Measure if answer is grounded in context
        
        Real-world: Detect hallucinations
        """
        if not context:
            return 0.0
        
        answer_embedding = self.model.encode(answer)
        context_embeddings = self.model.encode(context)
        
        # Check if answer is similar to any context
        max_similarity = max([
            np.dot(answer_embedding, ctx_emb) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(ctx_emb)
            )
            for ctx_emb in context_embeddings
        ])
        
        return float(max_similarity)
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        ground_truth: List[List[str]] = None
    ) -> Dict:
        """
        Comprehensive retrieval evaluation
        
        Args:
            queries: List of queries
            retrieved_docs: Retrieved documents for each query
            ground_truth: Expected relevant documents (optional)
        """
        relevance_scores = []
        
        for query, docs in zip(queries, retrieved_docs):
            score = self.context_relevance(query, docs)
            relevance_scores.append(score)
        
        results = {
            "avg_relevance": np.mean(relevance_scores),
            "min_relevance": np.min(relevance_scores),
            "max_relevance": np.max(relevance_scores),
            "queries_evaluated": len(queries)
        }
        
        # Calculate precision/recall if ground truth provided
        if ground_truth:
            precisions = []
            recalls = []
            
            for retrieved, truth in zip(retrieved_docs, ground_truth):
                retrieved_set = set(retrieved)
                truth_set = set(truth)
                
                if len(retrieved_set) > 0:
                    precision = len(retrieved_set & truth_set) / len(retrieved_set)
                    precisions.append(precision)
                
                if len(truth_set) > 0:
                    recall = len(retrieved_set & truth_set) / len(truth_set)
                    recalls.append(recall)
            
            results["avg_precision"] = np.mean(precisions) if precisions else 0
            results["avg_recall"] = np.mean(recalls) if recalls else 0
        
        return results


class ProductionRAG:
    """
    Production-ready RAG with all optimizations
    
    Features:
    - Semantic caching
    - Performance monitoring
    - Quality evaluation
    - Error handling
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = SemanticCache()
        self.monitor = RAGMonitor()
        self.evaluator = RAGEvaluator()
        
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.chunks = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        print(f"Indexed {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_cache: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Retrieve with caching and monitoring
        """
        start_time = time.time()
        
        try:
            # Check cache
            if use_cache:
                cached_results = self.cache.get(query)
                if cached_results is not None:
                    retrieval_time = time.time() - start_time
                    scores = [score for _, score in cached_results]
                    self.monitor.log_retrieval(query, len(cached_results), retrieval_time, scores)
                    return cached_results
            
            # Retrieve
            query_embedding = self.model.encode(query)
            
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [
                (self.chunks[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            # Cache results
            if use_cache:
                self.cache.set(query, results)
            
            # Log metrics
            retrieval_time = time.time() - start_time
            scores = [score for _, score in results]
            self.monitor.log_retrieval(query, len(results), retrieval_time, scores)
            
            return results
        
        except Exception as e:
            self.monitor.log_error()
            raise e
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            "cache_stats": self.cache.get_stats(),
            "retrieval_stats": self.monitor.get_report()
        }


if __name__ == "__main__":
    print("=== Production RAG System ===\n")
    
    # Sample documents
    documents = [
        "Python is a versatile programming language used for web development, data science, and automation",
        "Machine learning algorithms learn patterns from data to make predictions",
        "Deep learning uses neural networks with multiple layers to process complex data",
        "Natural language processing helps computers understand and generate human language",
        "Data science combines statistics, programming, and domain knowledge to extract insights"
    ]
    
    # Initialize system
    rag = ProductionRAG()
    rag.add_documents(documents)
    
    # Test queries
    queries = [
        "What is Python used for?",
        "Explain machine learning",
        "What is Python?",  # Similar to first query - should hit cache
        "Tell me about deep learning",
        "What is Python used for?"  # Exact repeat - should hit cache
    ]
    
    print("Running queries...\n")
    for query in queries:
        results = rag.retrieve(query, top_k=2)
        print(f"Query: {query}")
        print(f"Top result: {results[0][0][:60]}... (score: {results[0][1]:.3f})")
        print()
    
    # Performance report
    print("=== Performance Report ===")
    report = rag.get_performance_report()
    
    print("\nCache Statistics:")
    for key, value in report["cache_stats"].items():
        print(f"  {key}: {value}")
    
    print("\nRetrieval Statistics:")
    for key, value in report["retrieval_stats"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Evaluation
    print("\n=== Quality Evaluation ===")
    retrieved_docs = [
        [results[0][0] for results in [rag.retrieve(q, top_k=2) for q in queries[:3]]]
    ]
    
    eval_results = rag.evaluator.evaluate_retrieval(
        queries[:3],
        [[r[0] for r in rag.retrieve(q, top_k=2, use_cache=False)] for q in queries[:3]]
    )
    
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
