#!/usr/bin/env python3
"""
KV Cache Compression Demo for LLM Inference.

This example demonstrates how PolarQuant can be used to compress
KV (Key-Value) cache entries in transformer-based language models.

In LLM inference, the KV cache stores intermediate attention computations
for previous tokens. For long sequences, this can consume significant memory.
PolarQuant provides near-lossless compression with high compression ratios.
"""

import numpy as np
import time
from polarquant import PolarQuant, PolarQuantConfig


class SimulatedKVCache:
    """Simulates a KV cache for transformer attention."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 1000, use_compression: bool = True):
        """
        Initialize KV cache.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            use_compression: Whether to use PolarQuant compression
        """
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_compression = use_compression
        
        # Initialize PolarQuant if compression is enabled
        if use_compression:
            config = PolarQuantConfig(
                dimension=head_dim,
                radius_bits=8,
                angle_bits=4,
                seed=42
            )
            self.quantizer = PolarQuant(config)
            self.compression_ratio = self.quantizer.compression_ratio()
        
        # Storage
        self.keys = []
        self.values = []
        self.compressed_keys = []
        self.compressed_values = []
        
        # Statistics
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
    
    def append(self, key: np.ndarray, value: np.ndarray):
        """Add a new key-value pair to the cache."""
        if len(self.keys) >= self.max_seq_len:
            raise RuntimeError("KV cache is full")
        
        # Normalize (common in attention)
        key = key / (np.linalg.norm(key) + 1e-10)
        value = value / (np.linalg.norm(value) + 1e-10)
        
        # Store original for comparison
        self.keys.append(key.copy())
        self.values.append(value.copy())
        
        # Update original size (float32 = 4 bytes per element)
        self.total_original_bytes += 2 * key.nbytes  # key + value
        
        if self.use_compression:
            # Compress
            start = time.time()
            ck = self.quantizer.compress(key)
            cv = self.quantizer.compress(value)
            compress_time = time.time() - start
            
            self.compressed_keys.append(ck)
            self.compressed_values.append(cv)
            
            # Estimate compressed size
            # radius_idx: 4 bytes (int32)
            # angle_indices: (head_dim - 1) * 4 bytes
            compressed_size = 2 * (4 + (self.head_dim - 1) * 4)
            self.total_compressed_bytes += compressed_size
            
            return compress_time
        
        return 0.0
    
    def get_key(self, idx: int) -> np.ndarray:
        """Get key at index (decompresses if needed)."""
        if self.use_compression:
            return self.quantizer.decompress(self.compressed_keys[idx])
        return self.keys[idx]
    
    def get_value(self, idx: int) -> np.ndarray:
        """Get value at index (decompresses if needed)."""
        if self.use_compression:
            return self.quantizer.decompress(self.compressed_values[idx])
        return self.values[idx]
    
    def compute_attention_scores(self, query: np.ndarray) -> np.ndarray:
        """
        Compute attention scores for a query against all keys.
        
        Args:
            query: Query vector of shape (head_dim,)
            
        Returns:
            Attention scores of shape (seq_len,)
        """
        query = query / (np.linalg.norm(query) + 1e-10)
        
        scores = []
        for i in range(len(self.keys)):
            key = self.get_key(i)
            score = np.dot(query, key)
            scores.append(score)
        
        return np.array(scores)
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        if self.use_compression:
            actual_ratio = self.total_original_bytes / max(self.total_compressed_bytes, 1)
            return {
                'original_bytes': self.total_original_bytes,
                'compressed_bytes': self.total_compressed_bytes,
                'compression_ratio': actual_ratio,
                'space_saved_pct': (1 - 1/actual_ratio) * 100,
            }
        else:
            return {
                'original_bytes': self.total_original_bytes,
                'compressed_bytes': self.total_original_bytes,
                'compression_ratio': 1.0,
                'space_saved_pct': 0.0,
            }


def demo_kv_cache_compression():
    """Demonstrate KV cache compression."""
    print("=" * 70)
    print("KV Cache Compression Demo")
    print("=" * 70)
    
    # Typical LLM parameters
    head_dim = 64  # Common head dimension (e.g., 64, 128)
    seq_len = 100   # Simulate 100 tokens
    
    print(f"\nConfiguration:")
    print(f"  Head dimension: {head_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Quantization: 8-bit radius, 4-bit angles")
    
    # Create caches
    cache_compressed = SimulatedKVCache(head_dim, use_compression=True)
    cache_uncompressed = SimulatedKVCache(head_dim, use_compression=False)
    
    # Simulate generating KV pairs for a sequence
    print(f"\nGenerating {seq_len} KV pairs...")
    
    total_compress_time = 0.0
    
    for i in range(seq_len):
        # Simulate key and value from transformer layer
        key = np.random.randn(head_dim)
        value = np.random.randn(head_dim)
        
        # Add to uncompressed cache
        cache_uncompressed.append(key, value)
        
        # Add to compressed cache
        compress_time = cache_compressed.append(key, value)
        total_compress_time += compress_time
    
    print(f"Compression time: {total_compress_time*1000:.2f} ms")
    print(f"Average per token: {total_compress_time/seq_len*1000:.3f} ms")
    
    # Memory statistics
    print("\nMemory Usage:")
    stats = cache_compressed.get_memory_stats()
    print(f"  Original size: {stats['original_bytes'] / 1024:.2f} KB")
    print(f"  Compressed size: {stats['compressed_bytes'] / 1024:.2f} KB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Space saved: {stats['space_saved_pct']:.1f}%")
    
    # Test attention computation
    print("\nAttention Score Quality:")
    query = np.random.randn(head_dim)
    
    # Compute with uncompressed keys
    scores_uncompressed = cache_uncompressed.compute_attention_scores(query)
    
    # Compute with compressed keys
    start = time.time()
    scores_compressed = cache_compressed.compute_attention_scores(query)
    decompress_time = time.time() - start
    
    print(f"  Decompression time: {decompress_time*1000:.2f} ms")
    
    # Compare scores
    correlation = np.corrcoef(scores_uncompressed, scores_compressed)[0, 1]
    mae = np.mean(np.abs(scores_uncompressed - scores_compressed))
    
    print(f"  Score correlation: {correlation:.4f}")
    print(f"  Mean absolute error: {mae:.6f}")
    
    # Test top-k accuracy
    k = 5
    top_k_uncompressed = set(np.argsort(scores_uncompressed)[-k:])
    top_k_compressed = set(np.argsort(scores_compressed)[-k:])
    top_k_accuracy = len(top_k_uncompressed & top_k_compressed) / k
    
    print(f"  Top-{k} accuracy: {top_k_accuracy*100:.1f}%")


def demo_attention_head_comparison():
    """Compare different attention head dimensions."""
    print("\n" + "=" * 70)
    print("Attention Head Dimension Comparison")
    print("=" * 70)
    
    head_dims = [32, 64, 128, 256]
    seq_len = 50
    
    print(f"\n{'Head Dim':<12} {'Ratio':<10} {'Cosine':<10} {'Memory (KB)':<15}")
    print("-" * 50)
    
    for head_dim in head_dims:
        cache = SimulatedKVCache(head_dim, use_compression=True)
        
        # Populate cache
        for _ in range(seq_len):
            key = np.random.randn(head_dim)
            value = np.random.randn(head_dim)
            cache.append(key, value)
        
        # Test reconstruction quality
        query = np.random.randn(head_dim)
        query = query / np.linalg.norm(query)
        
        # Get original and reconstructed keys
        original_keys = [cache.keys[i] for i in range(seq_len)]
        reconstructed_keys = [cache.get_key(i) for i in range(seq_len)]
        
        # Compute cosine similarities
        cosine_sims = []
        for orig, recon in zip(original_keys, reconstructed_keys):
            sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
            cosine_sims.append(sim)
        
        mean_cosine = np.mean(cosine_sims)
        
        # Get memory stats
        stats = cache.get_memory_stats()
        
        print(f"{head_dim:<12} {stats['compression_ratio']:<10.2f} "
              f"{mean_cosine:<10.4f} {stats['compressed_bytes']/1024:<15.2f}")


def demo_long_sequence():
    """Demonstrate compression on a long sequence."""
    print("\n" + "=" * 70)
    print("Long Sequence Compression (10K tokens)")
    print("=" * 70)
    
    head_dim = 64
    seq_len = 10000
    
    print(f"\nSimulating {seq_len} tokens with head_dim={head_dim}")
    
    cache = SimulatedKVCache(head_dim, max_seq_len=seq_len+100, use_compression=True)
    
    # Generate KV pairs
    for i in range(seq_len):
        key = np.random.randn(head_dim)
        value = np.random.randn(head_dim)
        cache.append(key, value)
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1} tokens...")
    
    # Memory statistics
    stats = cache.get_memory_stats()
    
    print(f"\nMemory Usage:")
    print(f"  Original: {stats['original_bytes'] / (1024*1024):.2f} MB")
    print(f"  Compressed: {stats['compressed_bytes'] / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Space saved: {stats['space_saved_pct']:.1f}%")
    
    # Test random access
    print(f"\nRandom Access Test:")
    n_samples = 100
    sample_indices = np.random.choice(seq_len, n_samples, replace=False)
    
    start = time.time()
    for idx in sample_indices:
        _ = cache.get_key(idx)
        _ = cache.get_value(idx)
    elapsed = time.time() - start
    
    print(f"  Accessed {n_samples} random KV pairs in {elapsed*1000:.2f} ms")
    print(f"  Average access time: {elapsed/n_samples*1000:.3f} ms")


def demo_quality_vs_compression():
    """Explore quality vs compression tradeoff."""
    print("\n" + "=" * 70)
    print("Quality vs Compression Tradeoff")
    print("=" * 70)
    
    head_dim = 128
    seq_len = 100
    
    bit_configs = [
        (4, 2, "Low"),
        (6, 3, "Medium-Low"),
        (8, 4, "Medium"),
        (10, 6, "Medium-High"),
        (12, 8, "High"),
    ]
    
    print(f"\n{'Config':<15} {'Radius':<8} {'Angle':<8} {'Ratio':<8} {'Cosine':<10}")
    print("-" * 60)
    
    for r_bits, a_bits, label in bit_configs:
        # Create custom config
        config = PolarQuantConfig(
            dimension=head_dim,
            radius_bits=r_bits,
            angle_bits=a_bits,
            seed=42
        )
        
        cache = SimulatedKVCache(head_dim, use_compression=True)
        cache.quantizer = PolarQuant(config)
        cache.compression_ratio = cache.quantizer.compression_ratio()
        
        # Populate
        for _ in range(seq_len):
            key = np.random.randn(head_dim)
            value = np.random.randn(head_dim)
            cache.append(key, value)
        
        # Test quality
        cosine_sims = []
        for i in range(seq_len):
            orig = cache.keys[i]
            recon = cache.get_key(i)
            sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
            cosine_sims.append(sim)
        
        mean_cosine = np.mean(cosine_sims)
        
        print(f"{label:<15} {r_bits:<8} {a_bits:<8} "
              f"{cache.compression_ratio:<8.2f} {mean_cosine:<10.4f}")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PolarQuant: KV Cache Compression Demo".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    np.random.seed(42)
    
    demo_kv_cache_compression()
    demo_attention_head_comparison()
    demo_long_sequence()
    demo_quality_vs_compression()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
