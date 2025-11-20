# Full Production Projects

Three comprehensive projects that integrate all concepts from the learning modules.

## Projects

### 1. Enterprise RAG System (`enterprise_rag/`)
**Complexity: Advanced**

A production-ready RAG system for customer support with:
- Multi-source document ingestion (PDF, web, API)
- Hybrid search (semantic + keyword)
- Re-ranking and query expansion
- Caching and rate limiting
- Cost tracking and observability
- Guardrails and content moderation

**Covers:** Tokenization, embeddings, RAG, caching, evaluation, production monitoring

---

### 2. Fine-Tuned Code Assistant (`code_assistant/`)
**Complexity: Expert**

A specialized coding assistant with:
- LoRA fine-tuning on code datasets
- 4-bit quantization for efficiency
- Context-aware code completion
- RAG over documentation
- Prompt engineering patterns
- Latency optimization (<500ms)

**Covers:** Fine-tuning, LoRA/QLoRA, quantization, RAG, prompt engineering, latency optimization

---

### 3. Multi-Agent LLM System (`multi_agent_system/`)
**Complexity: Expert**

An orchestrated multi-agent system for complex tasks:
- Agent routing and orchestration
- Specialized agents (research, code, analysis)
- Shared KV cache across agents
- Rate-limit aware scheduling
- RAGAS evaluation framework
- Privacy-preserving data flows

**Covers:** All concepts - attention mechanisms, RAG, fine-tuning, caching, rate limiting, evaluation, security

---

## Setup

Each project has its own README with:
- Architecture overview
- Setup instructions
- Configuration options
- Usage examples
- Performance benchmarks

## Running Projects

```bash
cd full_projects/<project_name>
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python main.py
```
