# Enterprise RAG System

A production-ready RAG system designed for multi-tenant enterprise use.

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│         API Gateway (FastAPI)               │
│  - Rate limiting                            │
│  - Authentication                           │
│  - Request validation                       │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│      Query Processing Pipeline              │
│  - Tokenization & cost estimation           │
│  - Query expansion                          │
│  - Semantic cache check                     │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│      Hybrid Retrieval Engine                │
│  - Vector search (semantic)                 │
│  - Keyword search (BM25)                    │
│  - Re-ranking (cross-encoder)               │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│      LLM Generation                         │
│  - Context assembly                         │
│  - Prompt engineering                       │
│  - Guardrails & moderation                  │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│      Observability Layer                    │
│  - Prometheus metrics                       │
│  - Trace logging                            │
│  - Cost tracking                            │
└─────────────────────────────────────────────┘
```

## Features

- **Multi-tenant**: Isolated document spaces per organization
- **Hybrid Search**: Combines semantic and keyword search
- **Smart Caching**: Semantic cache with Redis
- **Rate Limiting**: Token bucket algorithm
- **Guardrails**: Content moderation and PII detection
- **Monitoring**: Prometheus metrics and structured logging
- **Cost Control**: Token tracking and budget alerts

## Setup

```bash
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export REDIS_URL="redis://localhost:6379"

# Run the server
python main.py
```

## API Usage

```python
import requests

# Index documents
response = requests.post("http://localhost:8000/documents", json={
    "tenant_id": "acme-corp",
    "documents": [
        {"id": "doc1", "content": "..."},
        {"id": "doc2", "content": "..."}
    ]
})

# Query
response = requests.post("http://localhost:8000/query", json={
    "tenant_id": "acme-corp",
    "query": "What is our refund policy?",
    "top_k": 5
})

print(response.json())
```

## Performance

- **Latency**: p50: 200ms, p95: 500ms, p99: 1s
- **Throughput**: 100 queries/second
- **Cache Hit Rate**: 40-60% in production
- **Cost**: ~$0.02 per query (with caching)
