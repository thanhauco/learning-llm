"""
RAG Engineering - Intermediate Level
Advanced chunking, re-ranking, and query optimization
"""

import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Structured chunk with metadata"""
    text: str
    doc_id: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict = None


class SemanticChunker:
    """
    Semantic chunking - split by meaning, not just size
    
    Real-world benefit: Better retrieval accuracy
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        """
        Split text into semantically coherent chunks
        
        Algorithm:
        1. Split into sentences
        2. Group sentences with high similarity
        3. Create chunks from groups
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(text, doc_id, 0, 0, len(text))]
        
        # Embed sentences
        embeddings = self.model.encode(sentences)
        
        # Group by similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_start = 0
        chunk_id = 0
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
            )
            
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text)
                ))
                
                current_chunk = [sentences[i]]
                current_start += len(chunk_text) + 1
                chunk_id += 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(chunk_text)
            ))
        
        return chunks
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class ReRanker:
    """
    Re-rank retrieved documents using cross-encoder
    
    Real-world scenario: Improve top-k accuracy by 20-40%
    Used by: Cohere, Pinecone, Weaviate
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using cross-encoder
        
        Cross-encoder is slower but more accurate than bi-encoder
        Use after initial retrieval to refine top results
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            (documents[idx], float(scores[idx]))
            for idx in ranked_indices
        ]


class QueryExpander:
    """
    Expand queries for better retrieval
    
    Real-world scenario: Handle ambiguous or short queries
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Generate query variations
        
        In production, use LLM to generate better expansions
        """
        # Simple expansion (in production, use GPT/Claude)
        expansions = [query]
        
        # Add common variations
        if "how" in query.lower():
            expansions.append(query.replace("how", "what is the way"))
        
        if "what" in query.lower():
            expansions.append(query.replace("what", "explain"))
        
        return expansions
    
    def expand_with_context(self, query: str, conversation_history: List[str]) -> str:
        """
        Add context from conversation history
        
        Real-world scenario: Multi-turn conversations
        """
        if not conversation_history:
            return query
        
        # Take last 2 turns for context
        context = " ".join(conversation_history[-2:])
        expanded = f"{context} {query}"
        
        return expanded


class AdvancedRAG:
    """
    Advanced RAG with re-ranking and query optimization
    
    Real-world scenario: Production-grade retrieval system
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = ReRanker(rerank_model)
        self.chunker = SemanticChunker()
        self.expander = QueryExpander()
        
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None):
        """Add documents with semantic chunking"""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Chunk all documents
        for doc, doc_id in zip(documents, doc_ids):
            chunks = self.chunker.chunk(doc, doc_id)
            self.chunks.extend(chunks)
        
        # Embed chunks
        chunk_texts = [c.text for c in self.chunks]
        self.embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True)
        
        print(f"Added {len(documents)} documents -> {len(self.chunks)} chunks")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 3,
        use_reranking: bool = True
    ) -> List[Tuple[Chunk, float]]:
        """
        Two-stage retrieval: fast retrieval + accurate re-ranking
        
        Stage 1: Bi-encoder retrieves top_k candidates (fast)
        Stage 2: Cross-encoder re-ranks to rerank_top_k (accurate)
        """
        # Stage 1: Initial retrieval
        query_embedding = self.embedder.encode(query)
        
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        candidates = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
        
        # Stage 2: Re-ranking
        if use_reranking and len(candidates) > 0:
            candidate_texts = [c.text for c, _ in candidates]
            reranked = self.reranker.rerank(query, candidate_texts, top_k=rerank_top_k)
            
            # Map back to chunks
            text_to_chunk = {c.text: c for c, _ in candidates}
            results = [
                (text_to_chunk[text], score)
                for text, score in reranked
            ]
            return results
        
        return candidates[:rerank_top_k]
    
    def query_with_context(
        self,
        query: str,
        conversation_history: List[str] = None,
        top_k: int = 3
    ) -> str:
        """
        Query with conversation context
        
        Real-world scenario: Chatbot with memory
        """
        # Expand query with context
        if conversation_history:
            expanded_query = self.expander.expand_with_context(query, conversation_history)
        else:
            expanded_query = query
        
        # Retrieve
        results = self.retrieve(expanded_query, top_k=top_k)
        
        # Format context
        context = "\n\n".join([
            f"[Source {i+1}] (score: {score:.3f})\n{chunk.text}"
            for i, (chunk, score) in enumerate(results)
        ])
        
        prompt = f"""Answer based on the context below.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt


if __name__ == "__main__":
    print("=== Semantic Chunking ===")
    
    text = """
    Machine learning is a field of AI. It focuses on building systems that learn from data.
    Deep learning uses neural networks. These networks have multiple layers.
    Python is a programming language. It's popular for data science and ML.
    """
    
    chunker = SemanticChunker(similarity_threshold=0.6)
    chunks = chunker.chunk(text)
    
    print(f"Created {len(chunks)} semantic chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {chunk.text[:80]}...")
    
    print("\n=== Re-Ranking ===")
    
    query = "What is deep learning?"
    documents = [
        "Python is a programming language used for many applications",
        "Deep learning is a subset of machine learning using neural networks",
        "Neural networks are inspired by the human brain structure",
        "Machine learning algorithms learn patterns from data"
    ]
    
    reranker = ReRanker()
    results = reranker.rerank(query, documents, top_k=3)
    
    print(f"Query: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | {doc}")
    
    print("\n=== Advanced RAG Pipeline ===")
    
    docs = [
        """Python is a high-level programming language. It emphasizes code readability 
        and simplicity. Python is widely used in web development, data science, and 
        machine learning. Popular frameworks include Django, Flask, NumPy, and Pandas.""",
        
        """Machine learning is a branch of artificial intelligence. It enables systems 
        to learn and improve from experience. Common techniques include supervised learning, 
        unsupervised learning, and reinforcement learning. Applications range from image 
        recognition to natural language processing.""",
        
        """Deep learning is a subset of machine learning. It uses artificial neural networks 
        with multiple layers. Deep learning excels at processing unstructured data like 
        images, audio, and text. Popular frameworks include TensorFlow, PyTorch, and Keras."""
    ]
    
    rag = AdvancedRAG()
    rag.add_documents(docs)
    
    # Test retrieval with re-ranking
    query = "What frameworks are used for deep learning?"
    results = rag.retrieve(query, top_k=5, rerank_top_k=2)
    
    print(f"\nQuery: {query}\n")
    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(f"Text: {chunk.text[:100]}...")
        print()
    
    # Test with conversation context
    print("\n=== Contextual Query ===")
    conversation = [
        "Tell me about programming languages",
        "I'm interested in data science"
    ]
    
    prompt = rag.query_with_context(
        "Which one should I learn?",
        conversation_history=conversation
    )
    
    print(prompt[:400] + "...")
