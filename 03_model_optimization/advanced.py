"""
Model Optimization - Advanced Level
QLoRA, mixed precision, and production optimization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    """
    QLoRA configuration
    
    QLoRA = Quantized LoRA
    - Base model in 4-bit
    - LoRA adapters in FP16/BF16
    - Enables fine-tuning 65B models on single 48GB GPU
    """
    bits: int = 4
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_double_quant: bool = True  # Double quantization for extra savings
    quant_type: str = "nf4"  # Normal Float 4-bit


class NF4Quantizer:
    """
    Normal Float 4-bit quantization
    
    Used in QLoRA paper
    Optimized for normally distributed weights
    """
    
    # NF4 quantization levels (optimized for normal distribution)
    NF4_LEVELS = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    @classmethod
    def quantize(cls, weights: np.ndarray) -> tuple:
        """
        Quantize to NF4
        
        Steps:
        1. Normalize weights to [-1, 1]
        2. Map to nearest NF4 level
        3. Store scale factor
        """
        # Calculate scale
        absmax = np.abs(weights).max()
        scale = absmax
        
        # Normalize
        normalized = weights / (scale + 1e-8)
        
        # Quantize to NF4 levels
        quantized = np.zeros_like(normalized, dtype=np.int8)
        for i, level in enumerate(cls.NF4_LEVELS):
            mask = (normalized >= level)
            if i < len(cls.NF4_LEVELS) - 1:
                mask &= (normalized < cls.NF4_LEVELS[i + 1])
            quantized[mask] = i
        
        return quantized, scale
    
    @classmethod
    def dequantize(cls, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize from NF4"""
        dequantized = np.zeros_like(quantized, dtype=np.float32)
        
        for i, level in enumerate(cls.NF4_LEVELS):
            mask = (quantized == i)
            dequantized[mask] = level * scale
        
        return dequantized


class DoubleQuantization:
    """
    Double quantization for extra memory savings
    
    Quantize the quantization constants themselves
    Used in QLoRA to save even more memory
    """
    
    @staticmethod
    def quantize_scales(scales: np.ndarray, bits: int = 8) -> tuple:
        """
        Quantize the scale factors
        
        Saves memory when you have many blocks
        """
        min_scale = scales.min()
        max_scale = scales.max()
        
        range_scale = max_scale - min_scale
        scale_scale = range_scale / (2 ** bits - 1)
        
        quantized_scales = np.round((scales - min_scale) / scale_scale)
        quantized_scales = quantized_scales.astype(np.uint8)
        
        return quantized_scales, scale_scale, min_scale


class QLoRALinear(nn.Module):
    """
    QLoRA linear layer
    
    Real-world: Fine-tune 70B models on consumer hardware
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QLoRAConfig
    ):
        super().__init__()
        self.config = config
        
        # Base weights (quantized, frozen)
        self.register_buffer(
            'weight_quantized',
            torch.randint(-8, 7, (out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer('weight_scale', torch.randn(1))
        
        # LoRA adapters (trainable, FP16)
        self.lora_A = nn.Parameter(
            torch.randn(config.lora_r, in_features, dtype=torch.float16) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, config.lora_r, dtype=torch.float16)
        )
        
        self.scaling = config.lora_alpha / config.lora_r
        self.dropout = nn.Dropout(config.lora_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Dequantize base weights on-the-fly
        2. Apply base transformation
        3. Add LoRA adaptation
        """
        # Dequantize base weights (happens in FP16 for speed)
        weight_dequant = self.weight_quantized.to(torch.float16) * self.weight_scale
        
        # Base transformation
        result = torch.nn.functional.linear(x, weight_dequant)
        
        # LoRA adaptation
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_result * self.scaling
        
        return result
    
    def get_memory_usage(self) -> Dict:
        """Calculate memory usage"""
        base_memory = self.weight_quantized.numel() * 1  # 1 byte per param (INT8)
        lora_memory = (self.lora_A.numel() + self.lora_B.numel()) * 2  # 2 bytes (FP16)
        
        return {
            "base_mb": base_memory / (1024**2),
            "lora_mb": lora_memory / (1024**2),
            "total_mb": (base_memory + lora_memory) / (1024**2)
        }


class MixedPrecisionTrainer:
    """
    Mixed precision training for efficiency
    
    Real-world: 2-3x faster training, 50% less memory
    """
    
    def __init__(self, model: nn.Module, use_amp: bool = True):
        self.model = model
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    def training_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Training step with automatic mixed precision
        
        Uses FP16 for forward/backward, FP32 for optimizer
        """
        optimizer.zero_grad()
        
        if self.use_amp:
            # Mixed precision context
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, labels)
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular FP32 training
            outputs = self.model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()
        
        return loss.item()


def calculate_qlora_memory(
    model_params: int,
    lora_rank: int = 64,
    base_bits: int = 4,
    lora_bits: int = 16
) -> Dict:
    """
    Calculate QLoRA memory requirements
    
    Example: LLaMA 70B
    - FP16: 140 GB
    - INT8: 70 GB
    - INT4: 35 GB
    - QLoRA (INT4 + LoRA): ~40 GB (fits on A100 80GB!)
    """
    # Base model memory (quantized)
    base_memory = model_params * (base_bits / 8)
    
    # LoRA memory (assume 0.1% of params are LoRA)
    lora_params = model_params * 0.001
    lora_memory = lora_params * (lora_bits / 8)
    
    # Optimizer states (Adam: 2x params for momentum + variance)
    optimizer_memory = lora_params * 4 * 2  # FP32 for optimizer
    
    # Gradients
    gradient_memory = lora_params * (lora_bits / 8)
    
    # Activations (rough estimate)
    activation_memory = model_params * 0.01 * 2  # Small fraction, FP16
    
    total_memory = (
        base_memory + lora_memory + optimizer_memory + 
        gradient_memory + activation_memory
    )
    
    return {
        "base_gb": base_memory / (1024**3),
        "lora_gb": lora_memory / (1024**3),
        "optimizer_gb": optimizer_memory / (1024**3),
        "gradients_gb": gradient_memory / (1024**3),
        "activations_gb": activation_memory / (1024**3),
        "total_gb": total_memory / (1024**3)
    }


if __name__ == "__main__":
    print("=== NF4 Quantization ===\n")
    
    # Test NF4 quantization
    weights = np.random.randn(1000, 1000).astype(np.float32)
    
    quantized, scale = NF4Quantizer.quantize(weights)
    dequantized = NF4Quantizer.dequantize(quantized, scale)
    
    error = np.mean(np.abs(weights - dequantized))
    print(f"Original memory: {weights.nbytes / (1024**2):.2f} MB")
    print(f"Quantized memory: {quantized.nbytes / (1024**2):.2f} MB")
    print(f"Compression: {weights.nbytes / quantized.nbytes:.1f}x")
    print(f"Mean absolute error: {error:.6f}")
    
    print("\n=== QLoRA Memory Calculations ===\n")
    
    models = {
        "LLaMA 7B": 7_000_000_000,
        "LLaMA 13B": 13_000_000_000,
        "LLaMA 70B": 70_000_000_000,
    }
    
    for model_name, params in models.items():
        print(f"{model_name}:")
        memory = calculate_qlora_memory(params, lora_rank=64)
        
        print(f"  Base model (4-bit): {memory['base_gb']:.1f} GB")
        print(f"  LoRA adapters: {memory['lora_gb']:.2f} GB")
        print(f"  Optimizer states: {memory['optimizer_gb']:.1f} GB")
        print(f"  Gradients: {memory['gradients_gb']:.2f} GB")
        print(f"  Activations: {memory['activations_gb']:.1f} GB")
        print(f"  Total: {memory['total_gb']:.1f} GB")
        
        # Determine feasibility
        if memory['total_gb'] <= 24:
            print(f"  ✓ Fits on RTX 4090 24GB")
        elif memory['total_gb'] <= 48:
            print(f"  ✓ Fits on A6000 48GB")
        elif memory['total_gb'] <= 80:
            print(f"  ✓ Fits on A100 80GB")
        else:
            print(f"  ✗ Requires multi-GPU setup")
        print()
    
    print("=== QLoRA vs Alternatives ===\n")
    
    print("Full Fine-tuning (FP16):")
    print("  LLaMA 70B: 140 GB + 280 GB (optimizer) = 420 GB")
    print("  Requires: 6x A100 80GB")
    print("  Cost: ~$20/hour")
    
    print("\nLoRA (FP16):")
    print("  LLaMA 70B: 140 GB + 1 GB (LoRA) = 141 GB")
    print("  Requires: 2x A100 80GB")
    print("  Cost: ~$7/hour")
    
    print("\nQLoRA (4-bit + FP16 LoRA):")
    print("  LLaMA 70B: 35 GB + 5 GB (LoRA+optimizer) = 40 GB")
    print("  Requires: 1x A100 80GB or 2x RTX 4090")
    print("  Cost: ~$3/hour or run locally")
    
    print("\n=== Practical Tips ===\n")
    print("1. Use QLoRA for large models (30B+)")
    print("2. Use LoRA for medium models (7B-30B)")
    print("3. Use full fine-tuning only if you have the budget")
    print("4. Start with r=8, increase to r=64 if needed")
    print("5. Monitor validation loss - QLoRA matches full fine-tuning")
    print("6. Use bitsandbytes library for production QLoRA")
