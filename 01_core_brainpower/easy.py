"""
Core Brainpower - Easy Level
Tokenization and basic embeddings
"""

import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


def tokenize_text(text: str, model: str = "gpt-4") -> List[int]:
    """
    Tokenize text using tiktoken (OpenAI's tokenizer)
    
    Example:
        tokens = tokenize_text("Hello, world!")
        print(f"Token count: {len(tokens)}")
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return tokens


def decode_tokens(tokens: List[int], model: str = "gpt-4") -> str:
    """Decode tokens back to text"""
    encoding = tiktoken.encoding_for_model(model)
    text = encoding.decode(tokens)
    return text


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text - crucial for API costs and context limits"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Get embedding vector for text
    
    This converts text into a dense vector representation
    where semantically similar texts have similar vectors
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode(text)
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Returns value between -1 and 1:
    - 1 means identical
    - 0 means orthogonal (no similarity)
    - -1 means opposite
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def find_most_similar(query: str, documents: List[str]) -> tuple[str, float]:
    """
    Find the most similar document to a query
    
    This is the foundation of semantic search
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)
    
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    max_idx = np.argmax(similarities)
    return documents[max_idx], similarities[max_idx]


if __name__ == "__main__":
    # Example 1: Tokenization
    print("=== Tokenization ===")
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenize_text(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Decoded: {decode_tokens(tokens)}")
    
    # Example 2: Token counting for cost estimation
    print("\n=== Token Counting ===")
    long_text = "AI is transforming the world. " * 100
    token_count = count_tokens(long_text)
    print(f"Token count: {token_count}")
    print(f"Estimated cost (GPT-4): ${token_count * 0.00003:.4f}")
    
    # Example 3: Embeddings and similarity
    print("\n=== Embeddings & Similarity ===")
    text1 = "I love programming"
    text2 = "I enjoy coding"
    text3 = "I hate vegetables"
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)
    
    print(f"Embedding dimension: {len(emb1)}")
    print(f"Similarity (programming vs coding): {cosine_similarity(emb1, emb2):.4f}")
    print(f"Similarity (programming vs vegetables): {cosine_similarity(emb1, emb3):.4f}")
    
    # Example 4: Semantic search
    print("\n=== Semantic Search ===")
    documents = [
        "Python is a programming language",
        "Dogs are loyal pets",
        "Machine learning uses neural networks",
        "Cats are independent animals"
    ]
    
    query = "What is a good pet?"
    best_doc, score = find_most_similar(query, documents)
    print(f"Query: {query}")
    print(f"Best match: {best_doc}")
    print(f"Score: {score:.4f}")
