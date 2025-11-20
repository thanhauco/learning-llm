# LLM Engineering Mastery

A comprehensive, hands-on learning path from basics to production-ready LLM systems covering all modern tools and techniques.

## 🎉 Project Complete Summary

### ✅ **10 Learning Sections** (01-10)
Each section covers a critical LLM engineering topic with easy, intermediate, and advanced levels:

1. **Core Brainpower** - Tokenization, embeddings, attention (✅ 100% complete)
2. **RAG Engineering** - Retrieval, chunking, re-ranking (✅ 100% complete)
3. **Model Optimization** - Quantization, LoRA, QLoRA (✅ 100% complete)
4. **Systems Thinking** - Caching, rate limiting, load balancing (✅ 100% complete)
5. **Quality Control** - Prompts, evaluation, testing (✅ 100% complete)
6. **Production** - Guardrails, security, observability (⏳ 33% complete)
7. **Distributed Training** - Ray, DeepSpeed, FSDP (⏳ 33% complete)
8. **Inference Optimization** - vLLM, TGI, continuous batching (⏳ 33% complete)
9. **Fast Fine-tuning** - Unsloth, Flash Attention (⏳ 33% complete)
10. **MLOps & Deployment** - BentoML, Modal, Replicate (⏳ 33% complete)

### ✅ **3 Full Production Projects** (100% complete)
Each with main code, README, Google Colab notebook, and sample data:

1. **Enterprise RAG System**
   - Semantic caching, hybrid search, re-ranking
   - PII detection, content moderation
   - Full monitoring and evaluation
   - ✅ [Runnable Colab Notebook](full_projects/01_enterprise_rag/enterprise_rag_colab.ipynb)

2. **AI Code Assistant**
   - Semantic code search
   - Context-aware generation
   - Automated code review
   - ✅ [Runnable Colab Notebook](full_projects/02_ai_code_assistant/code_assistant_colab.ipynb)

3. **Production Chatbot Platform**
   - Multi-turn conversations
   - Function calling, cost tracking
   - Auto-scaling, A/B testing
   - ✅ [Runnable Colab Notebook](full_projects/03_production_chatbot/chatbot_colab.ipynb)

### 📊 **Project Statistics**
- **30+ Python files** with production-ready code
- **10,000+ lines** of educational code
- **50+ concepts** covered from basics to advanced
- **100+ real-world examples**
- **3 Google Colab notebooks** ready to run
- **Sample data files** for all projects

## Project Structure

```
01_core_brainpower/     - Tokenization, embeddings, attention mechanisms
02_rag_engineering/     - RAG architecture, retrieval, chunking strategies
03_model_optimization/  - Quantization, LoRA, QLoRA, adapters
04_systems_thinking/    - Latency, caching, rate limiting
05_quality_control/     - Prompt engineering, evaluation, testing
06_production/          - Guardrails, security, observability
07_distributed_training/ - Ray, DeepSpeed, FSDP, multi-GPU training
08_inference_optimization/ - vLLM, TGI, TensorRT-LLM, continuous batching
09_fast_finetuning/     - Unsloth, Axolotl, Flash Attention optimization
10_mlops_deployment/    - BentoML, Modal, Replicate, serverless deployment
full_projects/          - 3 complete production-ready applications
```

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# Clone repository
git clone https://github.com/thanhauco/learning-llm
cd learning-llm

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run any example
python 01_core_brainpower/easy.py
```

### Option 2: Run in Google Colab (Recommended)
Click on any notebook to run in your browser with free GPU:
- [Enterprise RAG System](full_projects/01_enterprise_rag/enterprise_rag_colab.ipynb)
- [AI Code Assistant](full_projects/02_ai_code_assistant/code_assistant_colab.ipynb)
- [Production Chatbot](full_projects/03_production_chatbot/chatbot_colab.ipynb)

## Learning Path

### Recommended Order
1. **Sections 01-06** (Fundamentals) - 2-3 weeks
   - Core concepts, RAG, optimization, systems, quality, production
2. **Sections 07-10** (Advanced Scaling) - 1-2 weeks
   - Distributed training, inference optimization, fast fine-tuning, MLOps
3. **Full Projects** (Integration) - 1 week
   - Build complete systems integrating all concepts

**Total Time: 4-6 weeks to master**

### Each Section Contains
- `README.md` - Theory and concepts
- `easy.py` - Basic implementations
- `intermediate.py` - More complex patterns
- `advanced.py` - Production-ready code

### Progression
- **Easy:** Understand the concept with simple examples
- **Intermediate:** Learn advanced techniques and patterns
- **Advanced:** Production-ready implementations with optimization


## 🎯 What You'll Learn

### Core Skills (Sections 01-06)
- **Tokenization & Embeddings:** How LLMs process text
- **RAG Architecture:** Build retrieval-augmented generation systems
- **Model Optimization:** Quantization, LoRA, QLoRA for efficient inference
- **Systems Design:** Caching, rate limiting, load balancing
- **Quality Control:** Prompt engineering, evaluation, testing
- **Production:** Guardrails, security, monitoring

### Advanced Skills (Sections 07-10)
- **Distributed Training:** Scale training across multiple GPUs
- **Inference Optimization:** vLLM, continuous batching, 10-40x speedup
- **Fast Fine-tuning:** Unsloth, Flash Attention, 2-5x faster training
- **MLOps:** Deploy models with BentoML, Modal, Replicate

### Real-World Projects
- **Enterprise RAG:** Production-ready document Q&A system
- **AI Code Assistant:** Semantic code search and generation
- **Production Chatbot:** Scalable conversational AI platform

## 📚 Key Technologies Covered

- **LLM Frameworks:** Transformers, LangChain, LlamaIndex
- **Vector Databases:** FAISS, Chroma, Pinecone
- **Optimization:** bitsandbytes, PEFT, Unsloth
- **Inference:** vLLM, TGI, TensorRT-LLM
- **Distributed:** Ray, DeepSpeed, FSDP
- **Deployment:** BentoML, Modal, Replicate, FastAPI
- **Monitoring:** Prometheus, custom metrics

## 💡 Real-World Examples

Every section includes practical examples:
- Calculate token costs for API calls
- Build semantic search in 50 lines
- Implement caching for 80% latency reduction
- Deploy models with auto-scaling
- Evaluate RAG systems with RAGAS metrics
- Fine-tune LLaMA on consumer GPUs

## 🔥 Hot Topics Covered

- **Unsloth:** 2-5x faster fine-tuning
- **vLLM:** 10-40x faster inference with PagedAttention
- **Flash Attention 2:** Memory-efficient attention
- **QLoRA:** Fine-tune 70B models on single GPU
- **Continuous Batching:** Maximize throughput
- **Semantic Caching:** Reduce costs by 80%

## 📖 Additional Resources

- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Detailed completion status
- Each section's README.md - Theory and concepts
- Inline code comments - Implementation details
- Full project READMEs - Architecture and deployment guides

## 🤝 Contributing

This is a learning resource. Feel free to:
- Open issues for questions
- Submit PRs for improvements
- Share your implementations
- Suggest new topics

## 📝 License

MIT License - Feel free to use for learning and commercial projects.

## 🌟 Acknowledgments

Built with knowledge from:
- OpenAI, Anthropic, HuggingFace documentation
- Research papers on RAG, LoRA, vLLM
- Production experience from real-world LLM systems
- Community best practices

---

**Start your LLM engineering journey today!** 🚀

Begin with `01_core_brainpower/easy.py` or jump straight into the [Colab notebooks](full_projects/) for hands-on learning.
