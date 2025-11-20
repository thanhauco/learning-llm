# Inference Optimization - vLLM, TGI, TensorRT-LLM

## What You'll Learn

1. **vLLM** - Fast LLM inference with PagedAttention
2. **Text Generation Inference (TGI)** - HuggingFace production serving
3. **TensorRT-LLM** - NVIDIA optimized inference
4. **Continuous Batching** - Maximize throughput

## Key Concepts

- PagedAttention for memory efficiency
- Continuous batching vs static batching
- KV cache optimization
- Speculative decoding
- Tensor parallelism for inference

## Real-World Scenarios

- Serve 1000s of concurrent users
- Reduce latency from 5s to 500ms
- 10-20x higher throughput than naive serving
- Cost optimization for production

## Tools Covered

- **vLLM:** State-of-the-art inference engine
- **TGI:** Production-ready serving
- **TensorRT-LLM:** Maximum performance on NVIDIA GPUs
- **llama.cpp:** CPU inference
