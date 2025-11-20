"""
Model Optimization - Easy Level
Understanding quantization and model compression basics
"""

import numpy as np
from typing import Tuple
import torch


def quantize_to_int8(weights: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Quantize FP32 weights to INT8
    
    Real-world benefit: 4x memory reduction, 2-4x faster inference
    Used by: llama.cpp, GGML, bitsandbytes
    
    Formula: quantized = round((weight - min) / scale)
    """
    # Calculate scale and zero point
    min_val = weights.min()
    max_val = weights.max()
    
    # Scale to fit in INT8 range [-128, 127]
    scale = (max_val - min_val) / 255
    zero_point = -128 - min_val / scale
    
    # Quantize
    quantized = np.round((weights - min_val) / scale + zero_point)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    
    return quantized, scale, zero_point


def dequantize_from_int8(
    quantized: np.ndarray,
    scale: float,
    zero_point: float
) -> np.ndarray:
    """
    Dequantize INT8 back to FP32
    
    Formula: weight = (quantized - zero_point) * scale + min
    """
    return (quantized.astype(np.float32) - zero_point) * scale


def quantize_to_int4(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Quantize to 4-bit (even more aggressive)
    
    Real-world benefit: 8x memory reduction
    Used by: GPTQ, AWQ, QLoRA
    
    4-bit range: [-8, 7]
    """
    min_val = weights.min()
    max_val = weights.max()
    
    scale = (max_val - min_val) / 15  # 4-bit has 16 values
    
    quantized = np.round((weights - min_val) / scale - 8)
    quantized = np.clip(quantized, -8, 7).astype(np.int8)
    
    return quantized, scale


def compare_quantization_quality(original: np.ndarray, quantized: np.ndarray) -> dict:
    """
    Measure quantization error
    
    Metrics:
    - MSE: Mean squared error
    - MAE: Mean absolute error
    - Max error: Worst case error
    """
    mse = np.mean((original - quantized) ** 2)
    mae = np.mean(np.abs(original - quantized))
    max_error = np.max(np.abs(original - quantized))
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "relative_error": mae / (np.abs(original).mean() + 1e-8)
    }


def calculate_memory_savings(
    num_parameters: int,
    original_bits: int = 32,
    quantized_bits: int = 8
) -> dict:
    """
    Calculate memory savings from quantization
    
    Real-world example:
    - LLaMA 7B: 7B params * 4 bytes = 28GB
    - LLaMA 7B INT8: 7B params * 1 byte = 7GB
    - LLaMA 7B INT4: 7B params * 0.5 bytes = 3.5GB
    """
    original_bytes = num_parameters * (original_bits // 8)
    quantized_bytes = num_parameters * (quantized_bits / 8)
    
    savings_bytes = original_bytes - quantized_bytes
    savings_percent = (savings_bytes / original_bytes) * 100
    
    return {
        "original_gb": original_bytes / (1024**3),
        "quantized_gb": quantized_bytes / (1024**3),
        "savings_gb": savings_bytes / (1024**3),
        "savings_percent": savings_percent,
        "compression_ratio": original_bytes / quantized_bytes
    }


class SimpleQuantizer:
    """
    Simple quantizer for demonstration
    
    Real-world: Use bitsandbytes, GPTQ, or AWQ
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.scale = None
        self.zero_point = None
    
    def quantize(self, weights: np.ndarray) -> np.ndarray:
        """Quantize weights"""
        if self.bits == 8:
            quantized, self.scale, self.zero_point = quantize_to_int8(weights)
        elif self.bits == 4:
            quantized, self.scale = quantize_to_int4(weights)
            self.zero_point = -8
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")
        
        return quantized
    
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize weights"""
        if self.scale is None:
            raise ValueError("Must quantize first")
        
        return dequantize_from_int8(quantized, self.scale, self.zero_point)


if __name__ == "__main__":
    print("=== Quantization Basics ===\n")
    
    # Simulate model weights
    np.random.seed(42)
    weights = np.random.randn(1000, 1000).astype(np.float32)
    
    print(f"Original weights shape: {weights.shape}")
    print(f"Original dtype: {weights.dtype}")
    print(f"Original memory: {weights.nbytes / (1024**2):.2f} MB")
    
    # INT8 quantization
    print("\n=== INT8 Quantization ===")
    quantized_int8, scale, zero_point = quantize_to_int8(weights)
    dequantized_int8 = dequantize_from_int8(quantized_int8, scale, zero_point)
    
    print(f"Quantized dtype: {quantized_int8.dtype}")
    print(f"Quantized memory: {quantized_int8.nbytes / (1024**2):.2f} MB")
    print(f"Memory reduction: {weights.nbytes / quantized_int8.nbytes:.1f}x")
    
    error_int8 = compare_quantization_quality(weights, dequantized_int8)
    print(f"Mean Absolute Error: {error_int8['mae']:.6f}")
    print(f"Relative Error: {error_int8['relative_error']:.2%}")
    
    # INT4 quantization
    print("\n=== INT4 Quantization ===")
    quantized_int4, scale_int4 = quantize_to_int4(weights)
    
    print(f"Quantized memory: {quantized_int4.nbytes / (1024**2):.2f} MB")
    print(f"Memory reduction: {weights.nbytes / quantized_int4.nbytes:.1f}x")
    print("Note: INT4 stored in INT8 array, actual savings would be 8x")
    
    # Real-world model examples
    print("\n=== Real-World Model Sizes ===")
    
    models = {
        "GPT-2 Small": 124_000_000,
        "GPT-2 Large": 774_000_000,
        "LLaMA 7B": 7_000_000_000,
        "LLaMA 13B": 13_000_000_000,
        "LLaMA 70B": 70_000_000_000,
    }
    
    for model_name, params in models.items():
        print(f"\n{model_name} ({params/1e9:.1f}B parameters):")
        
        # FP32
        fp32 = calculate_memory_savings(params, 32, 32)
        print(f"  FP32: {fp32['original_gb']:.1f} GB")
        
        # INT8
        int8 = calculate_memory_savings(params, 32, 8)
        print(f"  INT8: {int8['quantized_gb']:.1f} GB (saves {int8['savings_gb']:.1f} GB)")
        
        # INT4
        int4 = calculate_memory_savings(params, 32, 4)
        print(f"  INT4: {int4['quantized_gb']:.1f} GB (saves {int4['savings_gb']:.1f} GB)")
    
    # Practical example
    print("\n=== Practical Example ===")
    print("Running LLaMA 70B on consumer hardware:")
    print("  FP32: 280 GB - Impossible on consumer GPUs")
    print("  FP16: 140 GB - Requires 2x A100 80GB")
    print("  INT8: 70 GB - Fits on 1x A100 80GB")
    print("  INT4: 35 GB - Fits on 2x RTX 4090 24GB")
    
    # Using SimpleQuantizer
    print("\n=== SimpleQuantizer Demo ===")
    small_weights = np.random.randn(100, 100).astype(np.float32)
    
    quantizer = SimpleQuantizer(bits=8)
    quantized = quantizer.quantize(small_weights)
    restored = quantizer.dequantize(quantized)
    
    error = compare_quantization_quality(small_weights, restored)
    print(f"Quantization error: {error['relative_error']:.2%}")
    print(f"Memory saved: {(1 - quantized.nbytes / small_weights.nbytes) * 100:.1f}%")
