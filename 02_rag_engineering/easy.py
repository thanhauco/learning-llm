"""
RAG Engineering - Easy Level
Basic RAG pipeline with document loading and retrieval
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleChunker:
    """
    Split documents into chunks
    
    Why chunking matters:
    - LLMs have token limits
    - Smaller chunks = more precise retrieval
    - Larger chunks = more context
    """
    
    @staticmethod
    def chunk_by_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks by token count
        
        Real-world use: Most common chunking strategy
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """
        Split by sentences for semantic coherence
        
        Better for: Q&A, where answers are typically sentence-based
        """
        # Simple sentence splitting (production would use spaCy/NLTK)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ". ".join(sentences[i:i + sentences_per_chunk]) + "."
            chunks.append(chunk)
        
        return chunks


class SimpleRAG:
    """
    Basic RAG pipeline
    
    Flow: Query -> Retrieve relevant chunks -> Generate answer
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str], chunk_size: int = 512):
        """
        Add documents to the knowledge base
        
        Steps:
        1. Chunk documents
        2. Embed chunks
        3. Store for retrieval
        """
        chunker = SimpleChunker()
        
        # Chunk all documents
        for doc in documents:
            chunks = chunker.chunk_by_tokens(doc, chunk_size=chunk_size)
            self.chunks.extend(chunks)
        
        # Embed all chunks
        print(f"Embedding {len(self.chunks)} chunks...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        print("Done!")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant chunks for a query
        
        This is the "R" in RAG
        """
        if self.embeddings is None:
            raise ValueError("No documents added yet!")
        
        # Embed query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        Full RAG query: retrieve + format context
        
        In production, this context would be sent to an LLM
        """
        results = self.retrieve(question, top_k=top_k)
        
        # Format context
        context = "\n\n".join([
            f"[Source {i+1}] (relevance: {score:.2f})\n{chunk}"
            for i, (chunk, score) in enumerate(results)
        ])
        
        # In production, you'd send this to GPT/Claude:
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt


if __name__ == "__main__":
    print("=== Document Chunking ===")
    
    sample_doc = """
    Machine learning is a subset of artificial intelligence that focuses on 
    building systems that learn from data. Deep learning is a type of machine 
    learning that uses neural networks with multiple layers. These networks can 
    learn complex patterns in large datasets. Natural language processing is 
    another important area that helps computers understand human language.
    """
    
    chunker = SimpleChunker()
    
    # Token-based chunking
    token_chunks = chunker.chunk_by_tokens(sample_doc, chunk_size=20, overlap=5)
    print(f"Token chunks: {len(token_chunks)}")
    for i, chunk in enumerate(token_chunks):
        print(f"  Chunk {i+1}: {chunk[:50]}...")
    
    # Sentence-based chunking
    sentence_chunks = chunker.chunk_by_sentences(sample_doc, sentences_per_chunk=2)
    print(f"\nSentence chunks: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks):
        print(f"  Chunk {i+1}: {chunk[:50]}...")
    
    print("\n=== Simple RAG Pipeline ===")
    
    # Create knowledge base
    documents = [
        """Python is a high-level programming language known for its simplicity 
        and readability. It's widely used in data science, web development, and 
        automation. Python has a rich ecosystem of libraries like NumPy, Pandas, 
        and TensorFlow.""",
        
        """Machine learning involves training models on data to make predictions. 
        Supervised learning uses labeled data, while unsupervised learning finds 
        patterns in unlabeled data. Common algorithms include decision trees, 
        neural networks, and support vector machines.""",
        
        """Natural language processing (NLP) enables computers to understand and 
        generate human language. Key tasks include sentiment analysis, named entity 
        recognition, and machine translation. Modern NLP relies heavily on 
        transformer models like BERT and GPT."""
    ]
    
    # Initialize RAG
    rag = SimpleRAG()
    rag.add_documents(documents, chunk_size=50)
    
    # Query the system
    questions = [
        "What is Python used for?",
        "Explain supervised learning",
        "What are transformer models?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        # Retrieve relevant chunks
        results = rag.retrieve(question, top_k=2)
        
        for i, (chunk, score) in enumerate(results):
            print(f"\nResult {i+1} (score: {score:.3f}):")
            print(chunk[:150] + "...")
    
    print("\n=== Full RAG Prompt ===")
    prompt = rag.query("What programming language is good for data science?")
    print(prompt[:500] + "...")
