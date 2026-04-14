#!/usr/bin/env python3
"""
Qwen3.5-0.8B Model Quantization Demo with PolarQuant

This example demonstrates how to use PolarQuant to quantize the KV cache
of a real Qwen3.5-0.8B model during inference.

Model specs from config.json:
- Hidden size: 1024
- Num attention heads: 8
- Num KV heads: 2 (GQA)
- Head dim: 256
- Num layers: 24
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import time

# Add parent directory to path for importing polarquant
sys.path.insert(0, str(Path(__file__).parent.parent))

from polarquant import PolarQuant, PolarQuantConfig


class QwenKVCacheQuantizer:
    """
    PolarQuant-based KV cache quantizer for Qwen models.
    
    Qwen3.5-0.8B uses:
    - GQA (Grouped Query Attention) with 2 KV heads
    - Head dimension: 256
    - 24 transformer layers
    """
    
    def __init__(self, model_path: str, radius_bits: int = 8, angle_bits: int = 4):
        """
        Initialize quantizer for Qwen model.
        
        Args:
            model_path: Path to model directory containing config.json
            radius_bits: Bits for radius quantization
            angle_bits: Bits for angle quantization
        """
        self.model_path = model_path
        self.config = self._load_config()
        
        # Extract model dimensions
        text_config = self.config.get('text_config', self.config)
        self.hidden_size = text_config['hidden_size']
        self.num_heads = text_config['num_attention_heads']
        self.num_kv_heads = text_config.get('num_key_value_heads', self.num_heads)
        self.head_dim = text_config.get('head_dim', self.hidden_size // self.num_heads)
        self.num_layers = text_config['num_hidden_layers']
        
        print(f"Model Configuration:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Num KV heads: {self.num_kv_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  Num layers: {self.num_layers}")
        
        # Initialize PolarQuant for each KV head
        self.kv_quantizers = []
        for layer_idx in range(self.num_layers):
            layer_quantizers = []
            for kv_head_idx in range(self.num_kv_heads):
                config = PolarQuantConfig(
                    dimension=self.head_dim,
                    radius_bits=radius_bits,
                    angle_bits=angle_bits,
                    seed=42 + layer_idx * self.num_kv_heads + kv_head_idx
                )
                quantizer = PolarQuant(config)
                layer_quantizers.append(quantizer)
            self.kv_quantizers.append(layer_quantizers)
        
        # KV cache storage: [layer][kv_head][seq_len]
        self.kv_cache = [
            [[] for _ in range(self.num_kv_heads)]
            for _ in range(self.num_layers)
        ]
        
        # Statistics
        self.compression_stats = {
            'original_bytes': 0,
            'compressed_bytes': 0,
            'compression_time_ms': 0,
        }
    
    def _load_config(self) -> dict:
        """Load model configuration."""
        config_path = os.path.join(self.model_path, 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def compress_kv(self, layer_idx: int, kv_head_idx: int, 
                    key: np.ndarray, value: np.ndarray) -> Tuple:
        """
        Compress key and value tensors.
        
        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            key: Key tensor of shape (head_dim,)
            value: Value tensor of shape (head_dim,)
            
        Returns:
            Tuple of (compressed_key, compressed_value)
        """
        quantizer = self.kv_quantizers[layer_idx][kv_head_idx]
        
        start = time.time()
        ck = quantizer.compress(key)
        cv = quantizer.compress(value)
        elapsed = (time.time() - start) * 1000
        
        self.compression_stats['compression_time_ms'] += elapsed
        
        # Estimate sizes
        self.compression_stats['original_bytes'] += 2 * key.nbytes
        # Compressed: radius_idx (radius_bits bits) + angle_indices (head_dim-1)*angle_bits bits
        # Convert to bytes (rounded up)
        config = quantizer.config
        compressed_key_bits = config.radius_bits + (self.head_dim - 1) * config.angle_bits
        compressed_size = 2 * ((compressed_key_bits + 7) // 8)  # 2 for key+value
        self.compression_stats['compressed_bytes'] += compressed_size
        
        return ck, cv
    
    def decompress_kv(self, layer_idx: int, kv_head_idx: int,
                      compressed_key, compressed_value) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompress key and value tensors.
        
        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            compressed_key: Compressed key
            compressed_value: Compressed value
            
        Returns:
            Tuple of (key, value)
        """
        quantizer = self.kv_quantizers[layer_idx][kv_head_idx]
        key = quantizer.decompress(compressed_key)
        value = quantizer.decompress(compressed_value)
        return key, value
    
    def append_to_cache(self, layer_idx: int, kv_head_idx: int,
                       key: np.ndarray, value: np.ndarray):
        """
        Append KV pair to cache with compression.
        
        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            key: Key tensor
            value: Value tensor
        """
        ck, cv = self.compress_kv(layer_idx, kv_head_idx, key, value)
        self.kv_cache[layer_idx][kv_head_idx].append((ck, cv))
    
    def get_from_cache(self, layer_idx: int, kv_head_idx: int, 
                       seq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get KV pair from cache with decompression.
        
        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            seq_idx: Sequence index
            
        Returns:
            Tuple of (key, value)
        """
        ck, cv = self.kv_cache[layer_idx][kv_head_idx][seq_idx]
        return self.decompress_kv(layer_idx, kv_head_idx, ck, cv)
    
    def compute_attention_score(self, layer_idx: int, kv_head_idx: int,
                               query: np.ndarray, seq_idx: int) -> float:
        """
        Compute attention score for a query against a cached key.
        
        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            query: Query tensor
            seq_idx: Sequence index
            
        Returns:
            Attention score
        """
        key, _ = self.get_from_cache(layer_idx, kv_head_idx, seq_idx)
        return np.dot(query, key)
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        orig = self.compression_stats['original_bytes']
        comp = self.compression_stats['compressed_bytes']
        ratio = orig / max(comp, 1)
        
        return {
            'original_bytes': orig,
            'compressed_bytes': comp,
            'compression_ratio': ratio,
            'space_saved_pct': (1 - 1/ratio) * 100 if ratio > 0 else 0,
            'compression_time_ms': self.compression_stats['compression_time_ms'],
        }
    
    def get_cache_size(self) -> int:
        """Get total number of cached KV pairs."""
        total = 0
        for layer in self.kv_cache:
            for head_cache in layer:
                total += len(head_cache)
        return total


def demo_qwen_kv_cache_simulation():
    """Simulate KV cache usage for Qwen3.5-0.8B."""
    print("=" * 70)
    print("Qwen3.5-0.8B KV Cache Quantization Demo")
    print("=" * 70)
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    
    if not os.path.exists(model_path):
        print(f"\nModel not found at: {model_path}")
        print("Using default configuration...")
        model_path = None
    
    # Initialize quantizer
    quantizer = QwenKVCacheQuantizer(
        model_path=model_path or ".",
        radius_bits=8,
        angle_bits=4
    )
    
    # Simulate generating a sequence
    seq_len = 100
    batch_size = 1
    
    print(f"\nSimulating inference with:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total layers: {quantizer.num_layers}")
    print(f"  KV heads per layer: {quantizer.num_kv_heads}")
    
    print(f"\nGenerating {seq_len} tokens...")
    
    for token_idx in range(seq_len):
        # For each token, generate KV for all layers and heads
        for layer_idx in range(quantizer.num_layers):
            for kv_head_idx in range(quantizer.num_kv_heads):
                # Simulate key and value vectors from model
                key = np.random.randn(quantizer.head_dim).astype(np.float32)
                value = np.random.randn(quantizer.head_dim).astype(np.float32)
                
                # Normalize (as in real attention)
                key = key / (np.linalg.norm(key) + 1e-10)
                value = value / (np.linalg.norm(value) + 1e-10)
                
                # Compress and store
                quantizer.append_to_cache(layer_idx, kv_head_idx, key, value)
        
        if (token_idx + 1) % 20 == 0:
            print(f"  Processed {token_idx + 1} tokens...")
    
    # Statistics
    stats = quantizer.get_compression_stats()
    total_kv_pairs = quantizer.get_cache_size()
    
    print(f"\n{'='*70}")
    print("Compression Statistics:")
    print(f"{'='*70}")
    print(f"Total KV pairs cached: {total_kv_pairs:,}")
    print(f"Original size: {stats['original_bytes'] / (1024**2):.2f} MB")
    print(f"Compressed size: {stats['compressed_bytes'] / (1024**2):.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Space saved: {stats['space_saved_pct']:.1f}%")
    print(f"Total compression time: {stats['compression_time_ms']:.2f} ms")
    print(f"Avg per token: {stats['compression_time_ms']/seq_len:.3f} ms")


def demo_attention_computation():
    """Demonstrate attention computation with compressed KV cache."""
    print(f"\n{'='*70}")
    print("Attention Computation with Compressed KV Cache")
    print(f"{'='*70}")
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    if not os.path.exists(model_path):
        model_path = "."
    
    quantizer = QwenKVCacheQuantizer(model_path=model_path, radius_bits=8, angle_bits=4)
    
    # Build a cache with some tokens
    seq_len = 50
    for token_idx in range(seq_len):
        for layer_idx in range(quantizer.num_layers):
            for kv_head_idx in range(quantizer.num_kv_heads):
                key = np.random.randn(quantizer.head_dim).astype(np.float32)
                value = np.random.randn(quantizer.head_dim).astype(np.float32)
                key = key / (np.linalg.norm(key) + 1e-10)
                value = value / (np.linalg.norm(value) + 1e-10)
                quantizer.append_to_cache(layer_idx, kv_head_idx, key, value)
    
    # Simulate attention computation for a new query
    layer_idx = 0
    kv_head_idx = 0
    query = np.random.randn(quantizer.head_dim).astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-10)
    
    print(f"\nComputing attention scores for layer {layer_idx}, head {kv_head_idx}")
    print(f"Query shape: {query.shape}")
    print(f"Cache size: {len(quantizer.kv_cache[layer_idx][kv_head_idx])} tokens")
    
    # Compute attention scores
    start = time.time()
    scores = []
    for seq_idx in range(seq_len):
        score = quantizer.compute_attention_score(layer_idx, kv_head_idx, query, seq_idx)
        scores.append(score)
    elapsed = (time.time() - start) * 1000
    
    scores = np.array(scores)
    
    print(f"\nAttention Statistics:")
    print(f"  Computation time: {elapsed:.2f} ms")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  Std score: {scores.std():.4f}")
    print(f"  Max score: {scores.max():.4f}")
    print(f"  Min score: {scores.min():.4f}")
    
    # Show top-k attention weights
    k = 5
    top_k_indices = np.argsort(scores)[-k:][::-1]
    print(f"\nTop-{k} attention positions:")
    for i, idx in enumerate(top_k_indices):
        print(f"  {i+1}. Position {idx}: score = {scores[idx]:.4f}")


def demo_quality_vs_compression_tradeoff():
    """Explore quality vs compression tradeoff for Qwen."""
    print(f"\n{'='*70}")
    print("Quality vs Compression Tradeoff Analysis")
    print(f"{'='*70}")
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    if not os.path.exists(model_path):
        model_path = "."
    
    # Load config to get head_dim
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        text_config = config.get('text_config', config)
        head_dim = text_config.get('head_dim', 256)
    else:
        head_dim = 256
    
    bit_configs = [
        (4, 2, "Ultra Low"),
        (6, 3, "Low"),
        (8, 4, "Medium"),
        (10, 6, "High"),
        (12, 8, "Ultra High"),
    ]
    
    # Generate test vectors
    np.random.seed(42)
    test_vectors = [np.random.randn(head_dim).astype(np.float32) for _ in range(10)]
    test_vectors = [v / (np.linalg.norm(v) + 1e-10) for v in test_vectors]
    
    print(f"\nTesting with head_dim={head_dim}, {len(test_vectors)} vectors")
    print(f"\n{'Config':<15} {'Radius':<8} {'Angle':<8} {'Ratio':<8} {'Cosine':<10}")
    print("-" * 60)
    
    for r_bits, a_bits, label in bit_configs:
        config = PolarQuantConfig(
            dimension=head_dim,
            radius_bits=r_bits,
            angle_bits=a_bits,
            seed=42
        )
        pq = PolarQuant(config)
        
        # Compress and decompress
        cosine_sims = []
        for vec in test_vectors:
            compressed = pq.compress(vec)
            recon = pq.decompress(compressed)
            sim = np.dot(vec, recon) / (np.linalg.norm(vec) * np.linalg.norm(recon))
            cosine_sims.append(sim)
        
        mean_cosine = np.mean(cosine_sims)
        ratio = pq.compression_ratio()
        
        print(f"{label:<15} {r_bits:<8} {a_bits:<8} {ratio:<8.2f} {mean_cosine:<10.4f}")


def demo_memory_usage_estimation():
    """Estimate memory usage for different sequence lengths."""
    print(f"\n{'='*70}")
    print("Memory Usage Estimation for Qwen3.5-0.8B")
    print(f"{'='*70}")
    
    model_path = "/Users/moyong/project/ai/models/Qwen/Qwen3.5-0.8B"
    if not os.path.exists(model_path):
        model_path = "."
    
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        text_config = config.get('text_config', config)
        num_layers = text_config['num_hidden_layers']
        num_kv_heads = text_config.get('num_key_value_heads', 2)
        head_dim = text_config.get('head_dim', 256)
    else:
        num_layers = 24
        num_kv_heads = 2
        head_dim = 256
    
    # Calculate bytes per KV pair
    bytes_per_float32 = 4
    original_bytes_per_kv = 2 * head_dim * bytes_per_float32  # key + value
    
    # PolarQuant compressed (8-bit radius, 4-bit angles)
    radius_bits = 8
    angle_bits = 4
    compressed_key_bits = radius_bits + (head_dim - 1) * angle_bits
    compressed_bytes_per_kv = 2 * ((compressed_key_bits + 7) // 8)  # 2 for key+value
    
    compression_ratio = original_bytes_per_kv / compressed_bytes_per_kv
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  KV heads per layer: {num_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    
    print(f"\nPer KV pair:")
    print(f"  Original: {original_bytes_per_kv} bytes")
    print(f"  Compressed: {compressed_bytes_per_kv} bytes")
    print(f"  Ratio: {compression_ratio:.2f}x")
    
    # Estimate for different sequence lengths
    seq_lengths = [1024, 4096, 16384, 65536, 131072]
    
    print(f"\n{'Seq Length':<12} {'Original':<15} {'Compressed':<15} {'Saved':<15}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        total_kv_pairs = seq_len * num_layers * num_kv_heads
        original_mb = total_kv_pairs * original_bytes_per_kv / (1024**2)
        compressed_mb = total_kv_pairs * compressed_bytes_per_kv / (1024**2)
        saved_mb = original_mb - compressed_mb
        
        print(f"{seq_len:<12} {original_mb:<15.1f} {compressed_mb:<15.1f} {saved_mb:<15.1f}")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PolarQuant: Qwen3.5-0.8B Quantization Demo".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    np.random.seed(42)
    
    demo_qwen_kv_cache_simulation()
    demo_attention_computation()
    demo_quality_vs_compression_tradeoff()
    demo_memory_usage_estimation()
    
    print(f"\n{'='*70}")
    print("All demos completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
