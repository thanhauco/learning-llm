"""
Model Optimization - Intermediate Level
LoRA and parameter-efficient fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 8  # Rank of low-rank matrices
    alpha: int = 16  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Which layers to apply LoRA


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer
    
    Instead of fine-tuning all weights W:
    W' = W + BA
    
    Where:
    - W: frozen pretrained weights (d x d)
    - B: trainable matrix (d x r)
    - A: trainable matrix (r x d)
    - r << d (rank is much smaller than dimension)
    
    Real-world benefit:
    - 10,000x fewer parameters to train
    - Same performance as full fine-tuning
    - Can switch between multiple adapters
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x @ A^T @ B^T * scaling
        """
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adapter
    
    Real-world usage: Replace nn.Linear in transformer models
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        use_lora: bool = True
    ):
        super().__init__()
        
        # Original linear layer (frozen during training)
        self.linear = nn.Linear(in_features, out_features)
        self.use_lora = use_lora
        
        if use_lora:
            self.lora = LoRALayer(in_features, out_features, r, alpha)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: original output + LoRA adaptation
        """
        result = self.linear(x)
        
        if self.use_lora:
            result = result + self.lora(x)
        
        return result
    
    def freeze_base(self):
        """Freeze original weights, only train LoRA"""
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def calculate_lora_savings(
    model_params: int,
    lora_rank: int = 8,
    num_layers: int = 32
) -> Dict:
    """
    Calculate parameter savings with LoRA
    
    Example: LLaMA 7B
    - Full fine-tuning: 7B parameters
    - LoRA (r=8): ~4M parameters (0.06% of original)
    """
    # Assume each layer has attention (4 matrices) and FFN (2 matrices)
    matrices_per_layer = 6
    total_matrices = num_layers * matrices_per_layer
    
    # Each matrix: d x d parameters
    # LoRA adds: d x r + r x d = 2 * d * r parameters
    d = int(np.sqrt(model_params / (num_layers * matrices_per_layer)))
    
    full_params = model_params
    lora_params = total_matrices * 2 * d * lora_rank
    
    return {
        "full_params": full_params,
        "lora_params": lora_params,
        "reduction_factor": full_params / lora_params,
        "lora_percentage": (lora_params / full_params) * 100,
        "memory_savings_gb": (full_params - lora_params) * 4 / (1024**3)
    }


class MultiAdapterManager:
    """
    Manage multiple LoRA adapters
    
    Real-world scenario: One base model, multiple task-specific adapters
    Example: Customer support bot with adapters for different products
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.adapters: Dict[str, Dict] = {}
        self.active_adapter = None
    
    def add_adapter(self, name: str, adapter_weights: Dict):
        """Add a new adapter"""
        self.adapters[name] = adapter_weights
        print(f"Added adapter: {name}")
    
    def switch_adapter(self, name: str):
        """Switch to a different adapter"""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found")
        
        self.active_adapter = name
        # In production, load adapter weights into model
        print(f"Switched to adapter: {name}")
    
    def list_adapters(self) -> List[str]:
        """List all available adapters"""
        return list(self.adapters.keys())


if __name__ == "__main__":
    print("=== LoRA Basics ===\n")
    
    # Compare regular linear vs LoRA
    in_features = 4096
    out_features = 4096
    
    regular_linear = nn.Linear(in_features, out_features)
    lora_linear = LinearWithLoRA(in_features, out_features, r=8)
    lora_linear.freeze_base()
    
    regular_params = sum(p.numel() for p in regular_linear.parameters())
    lora_params = lora_linear.get_trainable_params()
    
    print(f"Regular Linear:")
    print(f"  Total parameters: {regular_params:,}")
    print(f"  Memory: {regular_params * 4 / (1024**2):.2f} MB")
    
    print(f"\nLinear with LoRA (r=8):")
    print(f"  Total parameters: {sum(p.numel() for p in lora_linear.parameters()):,}")
    print(f"  Trainable parameters: {lora_params:,}")
    print(f"  Reduction: {regular_params / lora_params:.1f}x")
    print(f"  Trainable: {(lora_params / regular_params) * 100:.2f}%")
    
    # Test forward pass
    x = torch.randn(1, 10, in_features)
    
    with torch.no_grad():
        out_regular = regular_linear(x)
        out_lora = lora_linear(x)
    
    print(f"\nOutput shapes match: {out_regular.shape == out_lora.shape}")
    
    # Real-world model calculations
    print("\n=== Real-World LoRA Savings ===\n")
    
    models = {
        "LLaMA 7B": 7_000_000_000,
        "LLaMA 13B": 13_000_000_000,
        "LLaMA 70B": 70_000_000_000,
    }
    
    for model_name, params in models.items():
        print(f"{model_name}:")
        
        for rank in [4, 8, 16, 32]:
            savings = calculate_lora_savings(params, lora_rank=rank)
            print(f"  LoRA r={rank}:")
            print(f"    Trainable params: {savings['lora_params']:,} ({savings['lora_percentage']:.3f}%)")
            print(f"    Reduction: {savings['reduction_factor']:.0f}x")
        print()
    
    # Multi-adapter example
    print("=== Multi-Adapter System ===\n")
    
    base_model = nn.Linear(100, 100)
    manager = MultiAdapterManager(base_model)
    
    # Simulate different adapters for different tasks
    manager.add_adapter("customer_support", {"weights": "..."})
    manager.add_adapter("code_generation", {"weights": "..."})
    manager.add_adapter("translation", {"weights": "..."})
    
    print(f"\nAvailable adapters: {manager.list_adapters()}")
    
    manager.switch_adapter("customer_support")
    print("Processing customer query...")
    
    manager.switch_adapter("code_generation")
    print("Generating code...")
    
    # LoRA vs Full Fine-tuning comparison
    print("\n=== LoRA vs Full Fine-tuning ===\n")
    
    print("Full Fine-tuning:")
    print("  ✓ Slightly better performance")
    print("  ✗ Requires 7B+ parameters in memory")
    print("  ✗ Slow training")
    print("  ✗ Can't easily switch between tasks")
    print("  ✗ Risk of catastrophic forgetting")
    
    print("\nLoRA:")
    print("  ✓ 99.9% fewer trainable parameters")
    print("  ✓ Fast training")
    print("  ✓ Multiple adapters for different tasks")
    print("  ✓ No catastrophic forgetting")
    print("  ✓ Easy to share (adapters are tiny)")
    print("  ≈ Similar performance to full fine-tuning")
    
    print("\n=== Practical Example ===")
    print("Training LLaMA 7B:")
    print("  Full fine-tuning: Requires 4x A100 80GB, 24+ hours")
    print("  LoRA (r=8): Runs on 1x RTX 4090 24GB, 4-6 hours")
    print("  Adapter size: ~25MB (easy to share on HuggingFace)")
