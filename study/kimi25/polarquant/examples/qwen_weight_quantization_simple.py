#!/usr/bin/env python3
"""
Qwen3.5-0.8B Model Weight Quantization with PolarQuant (Simplified)

This is a simplified demonstration of weight quantization for Qwen models.
"""

import numpy as np
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from polarquant import PolarQuant, PolarQuantConfig


def quantize_weight_matrix(weight, radius_bits=8, angle_bits=4, block_size=256):
    """
    Quantize a weight matrix using PolarQuant.
    
    Args:
        weight: Weight matrix of shape (out_features, in_features)
        radius_bits: Bits for radius quantization
        angle_bits: Bits for angle quantization
        block_size: Block size for quantization
        
    Returns:
        Tuple of (compressed_data, stats)
    """
    rows, cols = weight.shape
    
    # Pad to match block_size if needed
    if cols < block_size:
        padded = np.zeros((rows, block_size), dtype=weight.dtype)
        padded[:, :cols] = weight
        weight = padded
        cols = block_size
    
    # Create quantizer
    config = PolarQuantConfig(
        dimension=cols,
        radius_bits=radius_bits,
        angle_bits=angle_bits,
        seed=42
    )
    quantizer = PolarQuant(config)
    
    # Quantize each row
    compressed_rows = []
    start = time.time()
    
    for i, row in enumerate(weight):
        compressed = quantizer.compress(row)
        compressed_rows.append(compressed)
    
    elapsed = (time.time() - start) * 1000
    
    # Calculate stats
    original_bytes = rows * cols * 4  # float32
    compressed_bits = rows * (radius_bits + (cols - 1) * angle_bits)
    compressed_bytes = (compressed_bits + 7) // 8
    ratio = original_bytes / compressed_bytes
    
    stats = {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': ratio,
        'time_ms': elapsed,
    }
    
    return compressed_rows, stats, quantizer


def dequantize_weight_matrix(compressed_rows, original_shape, quantizer):
    """Dequantize weight matrix."""
    rows = []
    for compressed in compressed_rows:
        row = quantizer.decompress(compressed)
        rows.append(row)
    
    weight = np.stack(rows)
    return weight[:original_shape[0], :original_shape[1]]


def demo_qwen_weight_quantization():
    """Demonstrate weight quantization for Qwen3.5-0.8B."""
    print("=" * 70)
    print("Qwen3.5-0.8B Weight Quantization Demo (Simplified)")
    print("=" * 70)
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    
    if os.path.exists(model_path):
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        text_config = config.get('text_config', config)
    else:
        # Default config
        text_config = {
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 8,
            'intermediate_size': 3584,
            'vocab_size': 248320,
        }
    
    hidden_size = text_config['hidden_size']
    intermediate_size = text_config.get('intermediate_size', hidden_size * 4)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    
    # Test different weight matrices
    test_weights = [
        ("Q Projection", (hidden_size, hidden_size)),
        ("K Projection", (hidden_size, hidden_size)),
        ("V Projection", (hidden_size, hidden_size)),
        ("O Projection", (hidden_size, hidden_size)),
        ("MLP Gate", (intermediate_size, hidden_size)),
        ("MLP Up", (intermediate_size, hidden_size)),
        ("MLP Down", (hidden_size, intermediate_size)),
    ]
    
    print(f"\n{'Layer':<20} {'Shape':<20} {'Ratio':<10} {'MSE':<12} {'Time':<10}")
    print("-" * 75)
    
    total_original = 0
    total_compressed = 0
    
    for name, shape in test_weights:
        # Generate random weight matrix
        np.random.seed(42)
        weight = np.random.randn(*shape).astype(np.float32)
        weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-10)
        
        # Quantize
        compressed, stats, quantizer = quantize_weight_matrix(
            weight, radius_bits=8, angle_bits=4, block_size=256
        )
        
        # Dequantize
        reconstructed = dequantize_weight_matrix(compressed, shape, quantizer)
        
        # Compute error
        mse = np.mean((weight - reconstructed) ** 2)
        
        total_original += stats['original_bytes']
        total_compressed += stats['compressed_bytes']
        
        print(f"{name:<20} {str(shape):<20} {stats['compression_ratio']:<10.2f} "
              f"{mse:<12.6f} {stats['time_ms']:<10.1f}")
    
    # Overall stats
    overall_ratio = total_original / total_compressed
    
    print(f"\n{'='*70}")
    print("Overall Statistics:")
    print(f"{'='*70}")
    print(f"Total original size: {total_original / (1024**2):.2f} MB")
    print(f"Total compressed size: {total_compressed / (1024**2):.2f} MB")
    print(f"Overall compression ratio: {overall_ratio:.2f}x")
    print(f"Space saved: {(1 - 1/overall_ratio) * 100:.1f}%")


def demo_bit_width_comparison():
    """Compare different bit widths."""
    print(f"\n{'='*70}")
    print("Bit Width Comparison")
    print(f"{'='*70}")
    
    # Test matrix
    shape = (1024, 1024)
    np.random.seed(42)
    weight = np.random.randn(*shape).astype(np.float32)
    weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-10)
    
    bit_configs = [
        (6, 3, "Low (6+3)"),
        (8, 4, "Medium (8+4)"),
        (10, 6, "High (10+6)"),
    ]
    
    print(f"\n{'Config':<20} {'Ratio':<10} {'MSE':<12} {'Cosine':<10}")
    print("-" * 55)
    
    for r_bits, a_bits, label in bit_configs:
        compressed, stats, quantizer = quantize_weight_matrix(
            weight, radius_bits=r_bits, angle_bits=a_bits, block_size=256
        )
        reconstructed = dequantize_weight_matrix(compressed, shape, quantizer)
        
        mse = np.mean((weight - reconstructed) ** 2)
        
        # Cosine similarity
        cos_sims = []
        for i in range(min(100, shape[0])):
            cos_sim = np.dot(weight[i], reconstructed[i]) / (
                np.linalg.norm(weight[i]) * np.linalg.norm(reconstructed[i])
            )
            cos_sims.append(cos_sim)
        mean_cosine = np.mean(cos_sims)
        
        print(f"{label:<20} {stats['compression_ratio']:<10.2f} {mse:<12.6f} {mean_cosine:<10.4f}")


def demo_matrix_multiplication():
    """Demonstrate matrix multiplication with quantized weights."""
    print(f"\n{'='*70}")
    print("Matrix Multiplication with Quantized Weights")
    print(f"{'='*70}")
    
    in_features = 1024
    out_features = 1024
    batch_size = 32
    
    print(f"\nLinear layer: {in_features} -> {out_features}")
    print(f"Batch size: {batch_size}")
    
    # Create weight matrix
    np.random.seed(42)
    weight = np.random.randn(out_features, in_features).astype(np.float32)
    weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-10) * 0.02
    
    # Quantize
    compressed, stats, quantizer = quantize_weight_matrix(
        weight, radius_bits=8, angle_bits=4, block_size=256
    )
    
    # Create input
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    
    # Original computation
    start = time.time()
    y_original = x @ weight.T
    time_original = (time.time() - start) * 1000
    
    # Quantized computation
    start = time.time()
    weight_dequantized = dequantize_weight_matrix(compressed, weight.shape, quantizer)
    y_quantized = x @ weight_dequantized.T
    time_quantized = (time.time() - start) * 1000
    
    # Compare
    mse = np.mean((y_original - y_quantized) ** 2)
    relative_error = np.linalg.norm(y_original - y_quantized) / np.linalg.norm(y_original)
    
    print(f"\nResults:")
    print(f"  Original time: {time_original:.2f} ms")
    print(f"  Quantized time: {time_quantized:.2f} ms")
    print(f"  Output MSE: {mse:.6f}")
    print(f"  Relative error: {relative_error:.4f}")
    
    print(f"\nMemory:")
    print(f"  Original: {stats['original_bytes'] / 1024:.2f} KB")
    print(f"  Compressed: {stats['compressed_bytes'] / 1024:.2f} KB")
    print(f"  Ratio: {stats['compression_ratio']:.2f}x")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PolarQuant: Qwen Weight Quantization (Simple)".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    np.random.seed(42)
    
    demo_qwen_weight_quantization()
    demo_bit_width_comparison()
    demo_matrix_multiplication()
    
    print(f"\n{'='*70}")
    print("All demos completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
