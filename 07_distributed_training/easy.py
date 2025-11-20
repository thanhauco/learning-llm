"""
Distributed Training - Easy Level
Introduction to distributed training concepts and Ray basics
"""

import numpy as np
from typing import List, Dict
import time
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    num_gpus: int = 1


class ParallelismStrategies:
    """
    Understanding different parallelism strategies
    
    Real-world: Choose based on model size and hardware
    """
    
    @staticmethod
    def data_parallelism_demo():
        """
        Data Parallelism: Split data across GPUs
        
        Use when: Model fits on single GPU
        Example: Training ResNet on ImageNet
        
        GPU 0: Batch 0-31
        GPU 1: Batch 32-63
        GPU 2: Batch 64-95
        GPU 3: Batch 96-127
        
        Each GPU has full model copy
        Gradients are averaged across GPUs
        """
        print("=== Data Parallelism ===")
        print("Model size: 7B parameters (28GB)")
        print("Available: 4x A100 80GB")
        print("\nStrategy:")
        print("  - Each GPU: Full model (28GB)")
        print("  - Batch split: 128 / 4 = 32 per GPU")
        print("  - Speedup: ~4x (linear scaling)")
        print("  - Communication: Gradient sync after each step")
        
        return {
            "strategy": "data_parallel",
            "gpus": 4,
            "model_per_gpu": "28GB",
            "effective_batch_size": 128,
            "speedup": "4x"
        }
    
    @staticmethod
    def model_parallelism_demo():
        """
        Model Parallelism: Split model across GPUs
        
        Use when: Model doesn't fit on single GPU
        Example: Training GPT-3 175B
        
        GPU 0: Layers 0-23
        GPU 1: Layers 24-47
        GPU 2: Layers 48-71
        GPU 3: Layers 72-95
        
        Data flows through GPUs sequentially
        """
        print("\n=== Model Parallelism ===")
        print("Model size: 175B parameters (700GB)")
        print("Available: 4x A100 80GB")
        print("\nStrategy:")
        print("  - GPU 0: Layers 0-23 (175GB)")
        print("  - GPU 1: Layers 24-47 (175GB)")
        print("  - GPU 2: Layers 48-71 (175GB)")
        print("  - GPU 3: Layers 72-95 (175GB)")
        print("  - Speedup: ~1x (sequential processing)")
        print("  - Communication: Activations between layers")
        
        return {
            "strategy": "model_parallel",
            "gpus": 4,
            "model_per_gpu": "175GB",
            "effective_batch_size": 32,
            "speedup": "1x"
        }
    
    @staticmethod
    def pipeline_parallelism_demo():
        """
        Pipeline Parallelism: Combine data + model parallelism
        
        Use when: Very large models
        Example: Training GPT-3 with micro-batches
        
        Split model across GPUs + pipeline micro-batches
        Better GPU utilization than pure model parallelism
        """
        print("\n=== Pipeline Parallelism ===")
        print("Model size: 175B parameters")
        print("Available: 8x A100 80GB")
        print("\nStrategy:")
        print("  - Model split across 4 GPUs (vertical)")
        print("  - Data split across 2 replicas (horizontal)")
        print("  - Micro-batches pipeline through stages")
        print("  - Speedup: ~6x (better than pure model parallel)")
        print("  - Communication: Activations + gradients")
        
        return {
            "strategy": "pipeline_parallel",
            "gpus": 8,
            "model_replicas": 2,
            "pipeline_stages": 4,
            "speedup": "6x"
        }


class SimpleDistributedTrainer:
    """
    Simulate distributed training
    
    Real-world: Use PyTorch DDP, DeepSpeed, or FSDP
    """
    
    def __init__(self, num_gpus: int = 1):
        self.num_gpus = num_gpus
        self.global_step = 0
    
    def train_step(self, batch_size: int) -> Dict:
        """
        Simulate single training step
        
        In distributed training:
        1. Forward pass on each GPU
        2. Compute loss
        3. Backward pass
        4. All-reduce gradients
        5. Update weights
        """
        # Simulate computation time
        compute_time = 0.1  # 100ms per GPU
        
        # Simulate communication time (gradient sync)
        # Communication increases with more GPUs
        comm_time = 0.01 * (self.num_gpus - 1)
        
        total_time = compute_time + comm_time
        
        # Effective batch size scales with GPUs
        effective_batch_size = batch_size * self.num_gpus
        
        # Throughput (samples/second)
        throughput = effective_batch_size / total_time
        
        self.global_step += 1
        
        return {
            "step": self.global_step,
            "compute_time": compute_time,
            "comm_time": comm_time,
            "total_time": total_time,
            "throughput": throughput,
            "effective_batch_size": effective_batch_size
        }
    
    def estimate_training_time(
        self,
        total_samples: int,
        batch_size: int,
        epochs: int
    ) -> Dict:
        """
        Estimate total training time
        
        Real-world: Use this for capacity planning
        """
        steps_per_epoch = total_samples // (batch_size * self.num_gpus)
        total_steps = steps_per_epoch * epochs
        
        # Simulate one step to get timing
        step_result = self.train_step(batch_size)
        time_per_step = step_result["total_time"]
        
        total_time_seconds = total_steps * time_per_step
        total_time_hours = total_time_seconds / 3600
        
        return {
            "total_samples": total_samples,
            "batch_size_per_gpu": batch_size,
            "num_gpus": self.num_gpus,
            "effective_batch_size": batch_size * self.num_gpus,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_hours": total_time_hours,
            "throughput": step_result["throughput"]
        }


def calculate_gpu_requirements(
    model_params: int,
    batch_size: int,
    sequence_length: int = 2048,
    precision: str = "fp16"
) -> Dict:
    """
    Calculate GPU memory requirements
    
    Memory breakdown:
    1. Model weights
    2. Optimizer states (Adam: 2x model size)
    3. Gradients (same as model size)
    4. Activations (depends on batch size)
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }[precision]
    
    # Model weights
    model_memory = model_params * bytes_per_param
    
    # Optimizer states (Adam)
    optimizer_memory = model_params * 4 * 2  # FP32 for optimizer
    
    # Gradients
    gradient_memory = model_params * bytes_per_param
    
    # Activations (rough estimate)
    activation_memory = batch_size * sequence_length * model_params * 0.0001 * bytes_per_param
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        "model_gb": model_memory / (1024**3),
        "optimizer_gb": optimizer_memory / (1024**3),
        "gradients_gb": gradient_memory / (1024**3),
        "activations_gb": activation_memory / (1024**3),
        "total_gb": total_memory / (1024**3),
        "precision": precision
    }


if __name__ == "__main__":
    print("=== Parallelism Strategies ===\n")
    
    strategies = ParallelismStrategies()
    
    # Data parallelism
    data_parallel = strategies.data_parallelism_demo()
    
    # Model parallelism
    model_parallel = strategies.model_parallelism_demo()
    
    # Pipeline parallelism
    pipeline_parallel = strategies.pipeline_parallelism_demo()
    
    print("\n=== Distributed Training Simulation ===\n")
    
    # Compare single GPU vs multi-GPU
    configs = [1, 2, 4, 8]
    
    for num_gpus in configs:
        trainer = SimpleDistributedTrainer(num_gpus=num_gpus)
        
        # Estimate training time for LLaMA 7B
        estimate = trainer.estimate_training_time(
            total_samples=1_000_000,  # 1M samples
            batch_size=32,
            epochs=3
        )
        
        print(f"{num_gpus} GPU(s):")
        print(f"  Effective batch size: {estimate['effective_batch_size']}")
        print(f"  Steps per epoch: {estimate['steps_per_epoch']:,}")
        print(f"  Estimated time: {estimate['estimated_hours']:.1f} hours")
        print(f"  Throughput: {estimate['throughput']:.0f} samples/sec")
        print()
    
    print("=== GPU Memory Requirements ===\n")
    
    models = {
        "LLaMA 7B": 7_000_000_000,
        "LLaMA 13B": 13_000_000_000,
        "LLaMA 70B": 70_000_000_000,
    }
    
    for model_name, params in models.items():
        print(f"{model_name}:")
        
        for precision in ["fp32", "fp16", "int8"]:
            mem = calculate_gpu_requirements(params, batch_size=1, precision=precision)
            print(f"  {precision.upper()}: {mem['total_gb']:.1f} GB")
            
            # Determine GPU requirements
            if mem['total_gb'] <= 24:
                print(f"    → Fits on 1x RTX 4090")
            elif mem['total_gb'] <= 80:
                print(f"    → Fits on 1x A100 80GB")
            else:
                num_gpus = int(np.ceil(mem['total_gb'] / 80))
                print(f"    → Requires {num_gpus}x A100 80GB")
        print()
    
    print("=== Key Takeaways ===\n")
    print("1. Data Parallelism: Best for models that fit on 1 GPU")
    print("   - Linear speedup with more GPUs")
    print("   - Simple to implement")
    print("   - Most common approach")
    
    print("\n2. Model Parallelism: For models too large for 1 GPU")
    print("   - No speedup (sequential)")
    print("   - Complex to implement")
    print("   - Use only when necessary")
    
    print("\n3. Pipeline Parallelism: Best of both worlds")
    print("   - Good speedup for large models")
    print("   - Better GPU utilization")
    print("   - More complex implementation")
    
    print("\n4. Memory Optimization:")
    print("   - Use FP16/BF16 for 2x memory savings")
    print("   - Use gradient checkpointing for 3-5x savings")
    print("   - Use ZeRO for 4-8x savings")
    print("   - Use quantization (INT8/INT4) for 4-8x savings")
