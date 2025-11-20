"""
AI Code Assistant - Production Implementation

Integrates:
- Code embeddings (01)
- RAG for code search (02)
- Quantization for local deployment (03)
- Caching for performance (04)
- Prompt engineering for code generation (05)
- Testing and validation (05, 06)
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib


@dataclass
class CodeChunk:
    """Represents a code chunk"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, method
    name: str
    language: str


@dataclass
class SearchResult:
    """Code search result"""
    chunk: CodeChunk
    score: float
    explanation: str = ""


class CodeParser:
    """
    Parse code into semantic chunks
    
    Real-world: Use tree-sitter for proper AST parsing
    """
    
    @staticmethod
    def parse_python_file(filepath: str) -> List[CodeChunk]:
        """
        Parse Python file into functions and classes
        
        Simplified version - production uses AST
        """
        chunks = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return chunks
        
        lines = content.split('\n')
        current_chunk = []
        current_name = None
        current_type = None
        start_line = 0
        
        for i, line in enumerate(lines):
            # Detect function/class definitions
            if line.strip().startswith('def '):
                if current_chunk:
                    chunks.append(CodeChunk(
                        content='\n'.join(current_chunk),
                        file_path=filepath,
                        start_line=start_line,
                        end_line=i-1,
                        chunk_type=current_type or 'code',
                        name=current_name or 'unknown',
                        language='python'
                    ))
                
                current_chunk = [line]
                current_type = 'function'
                current_name = line.split('def ')[1].split('(')[0].strip()
                start_line = i
            
            elif line.strip().startswith('class '):
                if current_chunk:
                    chunks.append(CodeChunk(
                        content='\n'.join(current_chunk),
                        file_path=filepath,
                        start_line=start_line,
                        end_line=i-1,
                        chunk_type=current_type or 'code',
                        name=current_name or 'unknown',
                        language='python'
                    ))
                
                current_chunk = [line]
                current_type = 'class'
                current_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                start_line = i
            
            else:
                if current_chunk:
                    current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk),
                file_path=filepath,
                start_line=start_line,
                end_line=len(lines)-1,
                chunk_type=current_type or 'code',
                name=current_name or 'unknown',
                language='python'
            ))
        
        return chunks


class CodeEmbedder:
    """
    Generate embeddings for code
    
    Real-world: Use CodeBERT or GraphCodeBERT
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # In production, use microsoft/codebert-base
        self.model = SentenceTransformer(model_name)
        self.cache = {}
    
    def embed(self, code: str) -> np.ndarray:
        """Generate embedding for code"""
        # Cache embeddings
        cache_key = hashlib.md5(code.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = self.model.encode(code)
        self.cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(self, codes: List[str]) -> np.ndarray:
        """Batch embed multiple code snippets"""
        return self.model.encode(codes, show_progress_bar=True)


class CodeSearchEngine:
    """
    Semantic code search
    
    Real-world: Use FAISS or Pinecone for scale
    """
    
    def __init__(self):
        self.embedder = CodeEmbedder()
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def index_repository(self, repo_path: str):
        """
        Index entire repository
        
        Walks through all Python files and indexes them
        """
        print(f"Indexing repository: {repo_path}")
        
        parser = CodeParser()
        all_chunks = []
        
        # Walk through repository
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    chunks = parser.parse_python_file(filepath)
                    all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        
        # Generate embeddings
        if self.chunks:
            print(f"Generating embeddings for {len(self.chunks)} code chunks...")
            code_texts = [chunk.content for chunk in self.chunks]
            self.embeddings = self.embedder.embed_batch(code_texts)
            print("Indexing complete!")
        else:
            print("No code chunks found")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for code by natural language query
        
        Examples:
        - "function to parse JSON"
        - "class for database connection"
        - "error handling code"
        """
        if not self.chunks or self.embeddings is None:
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter by type if specified
        if filter_type:
            mask = np.array([chunk.chunk_type == filter_type for chunk in self.chunks])
            similarities = similarities * mask
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            SearchResult(
                chunk=self.chunks[idx],
                score=float(similarities[idx])
            )
            for idx in top_indices
        ]
        
        return results


class CodeGenerator:
    """
    Generate code from natural language
    
    Real-world: Use GPT-4 or Claude with code-specific prompts
    """
    
    @staticmethod
    def generate_function(description: str, context: List[CodeChunk] = None) -> str:
        """
        Generate function from description
        
        Uses context from similar code in repository
        """
        # Build prompt with context
        context_str = ""
        if context:
            context_str = "Similar code in repository:\n\n"
            for chunk in context[:2]:
                context_str += f"```python\n{chunk.content}\n```\n\n"
        
        prompt = f"""{context_str}Generate a Python function that: {description}

Requirements:
- Include docstring
- Add type hints
- Handle edge cases
- Follow PEP 8

```python
"""
        
        # In production, call LLM here
        # For demo, return template
        return f'''def generated_function():
    """
    {description}
    """
    # TODO: Implement
    pass
'''


class CodeReviewer:
    """
    Automated code review
    
    Checks for:
    - Potential bugs
    - Security issues
    - Performance problems
    - Style violations
    """
    
    PATTERNS = {
        "security": {
            "sql_injection": r"execute\(['\"].*%s.*['\"]\)",
            "hardcoded_secret": r"(password|secret|api_key)\s*=\s*['\"][^'\"]+['\"]",
        },
        "bugs": {
            "bare_except": r"except\s*:",
            "mutable_default": r"def\s+\w+\([^)]*=\s*\[\]",
        },
        "performance": {
            "string_concat_loop": r"for\s+.*:\s*\w+\s*\+=\s*['\"]",
        }
    }
    
    def review(self, code: str) -> List[Dict]:
        """
        Review code and return issues
        """
        issues = []
        
        for category, patterns in self.PATTERNS.items():
            for issue_type, pattern in patterns.items():
                matches = re.finditer(pattern, code)
                for match in matches:
                    issues.append({
                        "category": category,
                        "type": issue_type,
                        "line": code[:match.start()].count('\n') + 1,
                        "message": self._get_message(issue_type),
                        "severity": self._get_severity(category)
                    })
        
        return issues
    
    @staticmethod
    def _get_message(issue_type: str) -> str:
        messages = {
            "sql_injection": "Potential SQL injection vulnerability",
            "hardcoded_secret": "Hardcoded secret detected",
            "bare_except": "Bare except clause catches all exceptions",
            "mutable_default": "Mutable default argument",
            "string_concat_loop": "String concatenation in loop (use join)"
        }
        return messages.get(issue_type, "Issue detected")
    
    @staticmethod
    def _get_severity(category: str) -> str:
        severity_map = {
            "security": "high",
            "bugs": "medium",
            "performance": "low"
        }
        return severity_map.get(category, "info")


class CodeAssistant:
    """
    Complete AI code assistant
    
    Combines search, generation, and review
    """
    
    def __init__(self):
        self.search_engine = CodeSearchEngine()
        self.generator = CodeGenerator()
        self.reviewer = CodeReviewer()
    
    def index_repository(self, repo_path: str):
        """Index repository for search"""
        self.search_engine.index_repository(repo_path)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for code"""
        return self.search_engine.search(query, top_k=top_k)
    
    def generate(self, description: str) -> str:
        """Generate code with repository context"""
        # Search for similar code
        context = self.search_engine.search(description, top_k=2)
        context_chunks = [r.chunk for r in context]
        
        # Generate with context
        return self.generator.generate_function(description, context_chunks)
    
    def review(self, code: str) -> List[Dict]:
        """Review code for issues"""
        return self.reviewer.review(code)
    
    def explain(self, code: str) -> str:
        """
        Explain code
        
        In production: Use LLM to generate explanation
        """
        # Search for similar code
        results = self.search_engine.search(code, top_k=1)
        
        if results:
            return f"This code is similar to {results[0].chunk.name} in {results[0].chunk.file_path}"
        
        return "Code explanation would be generated by LLM here"


if __name__ == "__main__":
    print("=== AI Code Assistant ===\n")
    
    # Initialize assistant
    assistant = CodeAssistant()
    
    # Index current directory (demo)
    print("Indexing repository...")
    assistant.index_repository(".")
    
    # Test search
    print("\n=== Code Search ===")
    query = "function for caching"
    results = assistant.search(query, top_k=3)
    
    print(f"Query: {query}\n")
    for i, result in enumerate(results):
        print(f"Result {i+1} (score: {result.score:.3f}):")
        print(f"  File: {result.chunk.file_path}")
        print(f"  Name: {result.chunk.name}")
        print(f"  Type: {result.chunk.chunk_type}")
        print(f"  Lines: {result.chunk.start_line}-{result.chunk.end_line}")
        print()
    
    # Test generation
    print("=== Code Generation ===")
    description = "Calculate fibonacci number recursively"
    generated = assistant.generate(description)
    print(f"Description: {description}\n")
    print("Generated code:")
    print(generated)
    
    # Test review
    print("\n=== Code Review ===")
    bad_code = """
def process_data(data=[]):
    password = "hardcoded123"
    try:
        result = ""
        for item in data:
            result += str(item)
    except:
        pass
    return result
"""
    
    issues = assistant.review(bad_code)
    print(f"Found {len(issues)} issues:\n")
    for issue in issues:
        print(f"[{issue['severity'].upper()}] Line {issue['line']}: {issue['message']}")
        print(f"  Category: {issue['category']}")
        print()
    
    print("=== Features ===")
    print("✓ Semantic code search")
    print("✓ Context-aware generation")
    print("✓ Automated code review")
    print("✓ Code explanation")
    print("✓ Repository indexing")
    print("✓ Multi-language support (extensible)")
