#!/usr/bin/env python3
"""
Qwen3.5-0.8B Model Weight Quantization with PolarQuant

This example demonstrates how to use PolarQuant to quantize model weights
of Qwen3.5-0.8B from safetensors format.

Note: PolarQuant is designed for high-dimensional vector quantization (like KV cache).
For weight quantization, we adapt it by reshaping weight matrices appropriately.
"""

import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polarquant import PolarQuant, PolarQuantConfig


class BFloat16Converter:
    """Utility to convert bfloat16 to float32 and back."""
    
    @staticmethod
    def bfloat16_to_float32(data: bytes) -> np.ndarray:
        """Convert bfloat16 bytes to float32 array."""
        # bfloat16 is 16 bits, we need to pad to 32 bits
        # Read as uint16, then expand to uint32 with zeros in lower 16 bits
        uint16_array = np.frombuffer(data, dtype=np.uint16)
        # Pad to float32 by shifting left 16 bits
        uint32_array = uint16_array.astype(np.uint32) << 16
        return uint32_array.view(np.float32)
    
    @staticmethod
    def float32_to_bfloat16(arr: np.ndarray) -> bytes:
        """Convert float32 array to bfloat16 bytes."""
        # Take upper 16 bits of float32
        uint32_array = arr.view(np.uint32)
        uint16_array = (uint32_array >> 16).astype(np.uint16)
        return uint16_array.tobytes()


class WeightQuantizer:
    """
    Quantizer for neural network weights using PolarQuant.
    
    Since PolarQuant is designed for vectors, we quantize weights by:
    1. Reshaping weight matrices to vectors (if small enough)
    2. Or quantizing row-by-row/column-by-column for large matrices
    """
    
    def __init__(self, radius_bits: int = 8, angle_bits: int = 4, 
                 block_size: int = 256, use_hadamard: bool = False):
        """
        Initialize weight quantizer.
        
        Args:
            radius_bits: Bits for radius quantization
            angle_bits: Bits for angle quantization
            block_size: Block size for blocking large matrices
            use_hadamard: Use Hadamard rotation
        """
        self.radius_bits = radius_bits
        self.angle_bits = angle_bits
        self.block_size = block_size
        self.use_hadamard = use_hadamard
        
        self.quantizers = {}  # Cache quantizers by dimension
        self.stats = {
            'total_params': 0,
            'quantized_params': 0,
            'original_bytes': 0,
            'compressed_bytes': 0,
            'compression_time_ms': 0,
        }
    
    def _get_quantizer(self, dim: int) -> PolarQuant:
        """Get or create quantizer for given dimension."""
        if dim not in self.quantizers:
            config = PolarQuantConfig(
                dimension=dim,
                radius_bits=self.radius_bits,
                angle_bits=self.angle_bits,
                use_hadamard=self.use_hadamard,
                seed=42
            )
            self.quantizers[dim] = PolarQuant(config)
        return self.quantizers[dim]
    
    def quantize_vector(self, vec: np.ndarray) -> Tuple:
        """
        Quantize a single vector.
        
        Args:
            vec: Input vector
            
        Returns:
            Compressed representation
        """
        dim = len(vec)
        
        # For dimensions not matching block_size, pad or use appropriate size
        if dim <= self.block_size:
            # Pad to block_size if smaller
            if dim < self.block_size:
                padded = np.zeros(self.block_size, dtype=vec.dtype)
                padded[:dim] = vec
                vec = padded
            
            quantizer = self._get_quantizer(self.block_size)
            
            start = time.time()
            compressed = quantizer.compress(vec)
            elapsed = (time.time() - start) * 1000
            
            self.stats['compression_time_ms'] += elapsed
            self.stats['quantized_params'] += dim
            
            # Estimate sizes
            self.stats['original_bytes'] += dim * 4  # float32
            compressed_bits = self.radius_bits + (self.block_size - 1) * self.angle_bits
            self.stats['compressed_bytes'] += (compressed_bits + 7) // 8
            
            return compressed, dim  # Return dimension for dequantization
        else:
            # For very large vectors, quantize in blocks
            return self._quantize_large_vector(vec)
    
    def _quantize_large_vector(self, vec: np.ndarray) -> Tuple:
        """Quantize a large vector by blocking."""
        dim = len(vec)
        num_blocks = (dim + self.block_size - 1) // self.block_size
        compressed_blocks = []
        
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, dim)
            block = vec[start_idx:end_idx]
            
            # Pad last block if needed
            if len(block) < self.block_size:
                padded = np.zeros(self.block_size, dtype=block.dtype)
                padded[:len(block)] = block
                block = padded
            
            quantizer = self._get_quantizer(self.block_size)
            compressed = quantizer.compress(block)
            compressed_blocks.append(compressed)
        
        self.stats['quantized_params'] += dim
        self.stats['original_bytes'] += dim * 4
        compressed_bits = num_blocks * (self.radius_bits + (self.block_size - 1) * self.angle_bits)
        self.stats['compressed_bytes'] += (compressed_bits + 7) // 8
        
        return compressed_blocks, dim
    
    def dequantize_vector(self, compressed_data, original_dim: int) -> np.ndarray:
        """
        Dequantize vector.
        
        Args:
            compressed_data: Compressed representation
            original_dim: Original dimension
            
        Returns:
            Dequantized vector
        """
        if isinstance(compressed_data, list):
            # Large vector with blocks
            return self._dequantize_large_vector(compressed_data, original_dim)
        else:
            # Single block
            quantizer = self._get_quantizer(self.block_size)
            vec = quantizer.decompress(compressed_data)
            return vec[:original_dim]
    
    def _dequantize_large_vector(self, compressed_blocks: List, original_dim: int) -> np.ndarray:
        """Dequantize large vector from blocks."""
        result = []
        for compressed in compressed_blocks:
            quantizer = self._get_quantizer(self.block_size)
            block = quantizer.decompress(compressed)
            result.append(block)
        
        vec = np.concatenate(result)
        return vec[:original_dim]
    
    def quantize_matrix(self, matrix: np.ndarray, axis: int = 0) -> Tuple:
        """
        Quantize a matrix row-by-row or column-by-column.
        
        Args:
            matrix: Input matrix
            axis: 0 for row-wise, 1 for column-wise
            
        Returns:
            List of compressed representations
        """
        if axis == 0:
            # Row-wise
            compressed_rows = []
            for row in matrix:
                compressed, dim = self.quantize_vector(row)
                compressed_rows.append((compressed, dim))
            return compressed_rows, matrix.shape
        else:
            # Column-wise
            compressed_cols = []
            for col in matrix.T:
                compressed, dim = self.quantize_vector(col)
                compressed_cols.append((compressed, dim))
            return compressed_cols, matrix.shape
    
    def dequantize_matrix(self, compressed_data: List, original_shape: Tuple, axis: int = 0) -> np.ndarray:
        """Dequantize matrix."""
        if axis == 0:
            rows = []
            for compressed, dim in compressed_data:
                row = self.dequantize_vector(compressed, dim)
                rows.append(row)
            return np.stack(rows).reshape(original_shape)
        else:
            cols = []
            for compressed, dim in compressed_data:
                col = self.dequantize_vector(compressed, dim)
                cols.append(col)
            return np.stack(cols, axis=1).reshape(original_shape)
    
    def get_stats(self) -> Dict:
        """Get quantization statistics."""
        orig = self.stats['original_bytes']
        comp = self.stats['compressed_bytes']
        ratio = orig / max(comp, 1)
        
        return {
            'total_params': self.stats['total_params'],
            'quantized_params': self.stats['quantized_params'],
            'original_bytes': orig,
            'compressed_bytes': comp,
            'compression_ratio': ratio,
            'space_saved_pct': (1 - 1/ratio) * 100 if ratio > 0 else 0,
            'compression_time_ms': self.stats['compression_time_ms'],
        }


class QwenWeightQuantizer:
    """Quantizer specifically for Qwen model weights."""
    
    def __init__(self, model_path: str, radius_bits: int = 8, angle_bits: int = 4):
        self.model_path = model_path
        self.config = self._load_config()
        self.quantizer = WeightQuantizer(radius_bits, angle_bits)
        
        # Load tensor info from index
        self.tensor_info = self._load_tensor_index()
    
    def _load_config(self) -> dict:
        """Load model configuration."""
        config_path = os.path.join(self.model_path, 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_tensor_index(self) -> Dict:
        """Load tensor index."""
        index_path = os.path.join(self.model_path, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def quantize_attention_weights(self):
        """Quantize attention layer weights."""
        print("\nQuantizing attention weights...")
        
        # Extract dimensions from config
        text_config = self.config.get('text_config', self.config)
        hidden_size = text_config['hidden_size']
        num_heads = text_config['num_attention_heads']
        num_layers = text_config['num_hidden_layers']
        
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Num layers: {num_layers}")
        
        # Simulate quantizing Q, K, V projection weights
        # In real implementation, these would be loaded from safetensors
        head_dim = hidden_size // num_heads
        
        print(f"  Quantizing Q/K/V projections (simulated)...")
        for layer_idx in range(num_layers):
            # Q projection: [hidden_size, hidden_size]
            q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
            compressed_q, shape = self.quantizer.quantize_matrix(q_weight, axis=0)
            
            # K projection
            k_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
            compressed_k, _ = self.quantizer.quantize_matrix(k_weight, axis=0)
            
            # V projection
            v_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
            compressed_v, _ = self.quantizer.quantize_matrix(v_weight, axis=0)
            
            # O projection
            o_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
            compressed_o, _ = self.quantizer.quantize_matrix(o_weight, axis=0)
            
            if (layer_idx + 1) % 6 == 0:
                print(f"    Processed {layer_idx + 1}/{num_layers} layers...")
        
        print(f"  Attention weights quantized!")
    
    def quantize_mlp_weights(self):
        """Quantize MLP layer weights."""
        print("\nQuantizing MLP weights...")
        
        text_config = self.config.get('text_config', self.config)
        hidden_size = text_config['hidden_size']
        intermediate_size = text_config.get('intermediate_size', hidden_size * 4)
        num_layers = text_config['num_hidden_layers']
        
        print(f"  Intermediate size: {intermediate_size}")
        
        for layer_idx in range(num_layers):
            # Gate projection
            gate_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
            compressed_gate, _ = self.quantizer.quantize_matrix(gate_weight, axis=0)
            
            # Up projection
            up_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
            compressed_up, _ = self.quantizer.quantize_matrix(up_weight, axis=0)
            
            # Down projection
            down_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
            compressed_down, _ = self.quantizer.quantize_matrix(down_weight, axis=0)
            
            if (layer_idx + 1) % 6 == 0:
                print(f"    Processed {layer_idx + 1}/{num_layers} layers...")
        
        print(f"  MLP weights quantized!")
    
    def quantize_embedding_weights(self):
        """Quantize embedding weights."""
        print("\nQuantizing embedding weights...")
        
        text_config = self.config.get('text_config', self.config)
        vocab_size = text_config['vocab_size']
        hidden_size = text_config['hidden_size']
        
        print(f"  Vocab size: {vocab_size}")
        print(f"  Hidden size: {hidden_size}")
        
        # For demo, only quantize a subset of embeddings
        demo_vocab_size = min(10000, vocab_size)
        print(f"  Quantizing first {demo_vocab_size} embeddings (demo only)...")
        
        # Token embeddings
        embed_weight = np.random.randn(demo_vocab_size, hidden_size).astype(np.float32)
        
        # Quantize in chunks
        chunk_size = 1000
        num_chunks = (demo_vocab_size + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, demo_vocab_size)
            chunk = embed_weight[start_idx:end_idx]
            compressed, _ = self.quantizer.quantize_matrix(chunk, axis=0)
        
        print(f"  Embedding weights quantized!")
    
    def get_stats(self) -> Dict:
        """Get quantization statistics."""
        return self.quantizer.get_stats()


def demo_weight_quantization():
    """Demonstrate weight quantization for Qwen."""
    print("=" * 70)
    print("Qwen3.5-0.8B Weight Quantization Demo")
    print("=" * 70)
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    
    if not os.path.exists(model_path):
        print(f"\nModel not found at: {model_path}")
        print("Using simulated weights...")
        model_path = "."
    
    # Initialize quantizer
    quantizer = QwenWeightQuantizer(
        model_path=model_path,
        radius_bits=8,
        angle_bits=4
    )
    
    print(f"\nModel Configuration:")
    text_config = quantizer.config.get('text_config', quantizer.config)
    print(f"  Model type: {text_config.get('model_type', 'unknown')}")
    print(f"  Hidden size: {text_config['hidden_size']}")
    print(f"  Num layers: {text_config['num_hidden_layers']}")
    print(f"  Vocab size: {text_config['vocab_size']}")
    
    # Quantize different weight types
    quantizer.quantize_attention_weights()
    quantizer.quantize_mlp_weights()
    quantizer.quantize_embedding_weights()
    
    # Statistics
    stats = quantizer.get_stats()
    
    print(f"\n{'='*70}")
    print("Quantization Statistics:")
    print(f"{'='*70}")
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Quantized parameters: {stats['quantized_params']:,}")
    print(f"Original size: {stats['original_bytes'] / (1024**2):.2f} MB")
    print(f"Compressed size: {stats['compressed_bytes'] / (1024**2):.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Space saved: {stats['space_saved_pct']:.1f}%")
    print(f"Total compression time: {stats['compression_time_ms']:.2f} ms")


def demo_quality_analysis():
    """Analyze quantization quality on sample weights."""
    print(f"\n{'='*70}")
    print("Weight Quantization Quality Analysis")
    print(f"{'='*70}")
    
    # Test different matrix sizes
    test_configs = [
        (256, 256, "Small Attention"),
        (1024, 1024, "Large Attention"),
        (3584, 1024, "MLP Up"),
        (1024, 3584, "MLP Down"),
    ]
    
    bit_configs = [
        (8, 4, "Medium (8+4 bits)"),
        (10, 6, "High (10+6 bits)"),
        (12, 8, "Ultra High (12+8 bits)"),
    ]
    
    print(f"\n{'Matrix':<20} {'Bits':<25} {'Ratio':<10} {'MSE':<12} {'Cosine':<10}")
    print("-" * 80)
    
    for rows, cols, name in test_configs:
        # Generate test weight matrix
        np.random.seed(42)
        weight = np.random.randn(rows, cols).astype(np.float32)
        weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
        
        for r_bits, a_bits, label in bit_configs:
            quantizer = WeightQuantizer(radius_bits=r_bits, angle_bits=a_bits, block_size=256)
            
            # Quantize
            compressed, shape = quantizer.quantize_matrix(weight, axis=0)
            
            # Dequantize
            reconstructed = quantizer.dequantize_matrix(compressed, shape, axis=0)
            
            # Compute metrics
            mse = np.mean((weight - reconstructed) ** 2)
            
            # Cosine similarity per row
            cos_sims = []
            for i in range(min(rows, 100)):  # Sample 100 rows
                w_row = weight[i]
                r_row = reconstructed[i]
                cos_sim = np.dot(w_row, r_row) / (np.linalg.norm(w_row) * np.linalg.norm(r_row) + 1e-10)
                cos_sims.append(cos_sim)
            mean_cosine = np.mean(cos_sims)
            
            ratio = quantizer.get_stats()['compression_ratio']
            
            print(f"{name:<20} {label:<25} {ratio:<10.2f} {mse:<12.6f} {mean_cosine:<10.4f}")


def demo_block_size_comparison():
    """Compare different block sizes for quantization."""
    print(f"\n{'='*70}")
    print("Block Size Comparison")
    print(f"{'='*70}")
    
    # Test matrix
    np.random.seed(42)
    weight = np.random.randn(1024, 1024).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    
    block_sizes = [64, 128, 256, 512]
    
    print(f"\n{'Block Size':<15} {'Ratio':<10} {'MSE':<12} {'Time (ms)':<15}")
    print("-" * 55)
    
    for block_size in block_sizes:
        quantizer = WeightQuantizer(radius_bits=8, angle_bits=4, block_size=block_size)
        
        start = time.time()
        compressed, shape = quantizer.quantize_matrix(weight, axis=0)
        reconstructed = quantizer.dequantize_matrix(compressed, shape, axis=0)
        elapsed = (time.time() - start) * 1000
        
        mse = np.mean((weight - reconstructed) ** 2)
        ratio = quantizer.get_stats()['compression_ratio']
        
        print(f"{block_size:<15} {ratio:<10.2f} {mse:<12.6f} {elapsed:<15.2f}")


def demo_practical_usage():
    """Demonstrate practical usage of quantized weights."""
    print(f"\n{'='*70}")
    print("Practical Usage: Matrix Multiplication with Quantized Weights")
    print(f"{'='*70}")
    
    # Simulate a linear layer
    in_features = 1024
    out_features = 1024
    batch_size = 32
    
    print(f"\nLinear layer: {in_features} -> {out_features}")
    print(f"Batch size: {batch_size}")
    
    # Create weight matrix
    np.random.seed(42)
    weight = np.random.randn(out_features, in_features).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=1, keepdims=True) * 0.02
    
    # Quantize
    quantizer = WeightQuantizer(radius_bits=8, angle_bits=4, block_size=256)
    compressed, shape = quantizer.quantize_matrix(weight, axis=0)
    
    # Create input
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    
    # Original computation
    start = time.time()
    y_original = x @ weight.T
    time_original = (time.time() - start) * 1000
    
    # Quantized computation (dequantize on-the-fly)
    start = time.time()
    weight_dequantized = quantizer.dequantize_matrix(compressed, shape, axis=0)
    y_quantized = x @ weight_dequantized.T
    time_quantized = (time.time() - start) * 1000
    
    # Compare
    mse = np.mean((y_original - y_quantized) ** 2)
    relative_error = np.linalg.norm(y_original - y_quantized) / np.linalg.norm(y_original)
    
    print(f"\nResults:")
    print(f"  Original computation time: {time_original:.2f} ms")
    print(f"  Quantized computation time: {time_quantized:.2f} ms")
    print(f"  Output MSE: {mse:.6f}")
    print(f"  Relative error: {relative_error:.4f}")
    
    stats = quantizer.get_stats()
    print(f"\nMemory:")
    print(f"  Original weight size: {stats['original_bytes'] / 1024:.2f} KB")
    print(f"  Compressed weight size: {stats['compressed_bytes'] / 1024:.2f} KB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PolarQuant: Qwen Weight Quantization Demo".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    np.random.seed(42)
    
    demo_weight_quantization()
    demo_quality_analysis()
    demo_block_size_comparison()
    demo_practical_usage()
    
    print(f"\n{'='*70}")
    print("All demos completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
