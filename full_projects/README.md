# Full Production Projects

Complete end-to-end projects that integrate all concepts from the learning modules.

## Projects

### 1. Enterprise RAG System (`enterprise_rag/`)
A production-ready RAG system with:
- Multi-tenant document management
- Hybrid search (semantic + keyword)
- Re-ranking and query expansion
- Cost tracking and rate limiting
- Observability and monitoring
- Guardrails and content moderation

**Covers:** Tokenization, embeddings, RAG, caching, evaluation, production patterns

### 2. Fine-tuned Customer Support Bot (`customer_support_bot/`)
A specialized chatbot with:
- LoRA fine-tuning on support tickets
- 4-bit quantization for efficiency
- RAG for knowledge base integration
- Prompt engineering patterns
- PII detection and redaction
- Performance monitoring

**Covers:** Fine-tuning, quantization, LoRA, RAG vs fine-tuning, guardrails, privacy

### 3. Code Assistant with Semantic Cache (`code_assistant/`)
An intelligent code helper with:
- Semantic caching for repeated queries
- Context-aware code completion
- Multi-file code understanding
- Latency optimization (<500ms)
- Token budget management
- JSON schema validation

**Covers:** Caching strategies, latency optimization, attention mechanisms, prompt engineering

### 4. Document Intelligence Platform (`document_intelligence/`)
Process and analyze documents at scale:
- Batch processing pipeline
- Multiple chunking strategies
- Vector database integration (Pinecone, Chroma)
- RAGAS evaluation framework
- Adversarial prompt testing
- Cost optimization

**Covers:** All RAG concepts, evaluation, quality control, systems thinking

## Running the Projects

Each project has its own README with setup instructions and architecture details.

```bash
cd full_projects/<project_name>
pip install -r requirements.txt
python main.py
```
