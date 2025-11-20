# LLM Engineering Mastery

A comprehensive, hands-on learning path from basics to production-ready LLM systems.

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

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Learning Path

Each module builds on the previous one. Start with `01_core_brainpower` and work your way up.

Each folder contains:
- `README.md` - Theory and concepts
- `easy.py` - Basic implementations
- `intermediate.py` - More complex patterns
- `advanced.py` - Production-ready code
- `tests.py` - Validation and benchmarks
