#!/usr/bin/env python3
"""
Basic usage example for PolarQuant.

This example demonstrates the core functionality of PolarQuant:
- Configuration
- Compression and decompression
- Error metrics computation
- Batch processing
"""

import numpy as np
from polarquant import PolarQuant, PolarQuantConfig, PolarQuantBatch


def example_basic_compression():
    """Example 1: Basic compression and decompression."""
    print("=" * 60)
    print("Example 1: Basic Compression")
    print("=" * 60)
    
    # Configure quantizer
    config = PolarQuantConfig(
        dimension=64,
        radius_bits=8,
        angle_bits=4,
        seed=42
    )
    
    # Create quantizer
    pq = PolarQuant(config)
    
    # Create a random vector
    x = np.random.randn(64)
    x = x / np.linalg.norm(x)  # Normalize to unit vector
    
    print(f"Original vector dimension: {len(x)}")
    print(f"Original vector norm: {np.linalg.norm(x):.4f}")
    
    # Compress
    compressed = pq.compress(x)
    print(f"\nCompressed representation:")
    print(f"  Radius index: {compressed.radius_idx}")
    print(f"  Angle indices shape: {compressed.angle_indices.shape}")
    
    # Decompress
    x_reconstructed = pq.decompress(compressed)
    print(f"\nReconstructed vector norm: {np.linalg.norm(x_reconstructed):.4f}")
    
    # Compute error metrics
    errors = pq.compute_error(x, x_reconstructed)
    print(f"\nError Metrics:")
    print(f"  MSE: {errors['mse']:.6f}")
    print(f"  RMSE: {errors['rmse']:.6f}")
    print(f"  Cosine Similarity: {errors['cosine_similarity']:.4f}")
    print(f"  Relative Error: {errors['relative_error']:.4f}")
    
    # Compression ratio
    ratio = pq.compression_ratio()
    print(f"\nCompression Ratio: {ratio:.2f}x")
    print(f"Space Saved: {(1 - 1/ratio) * 100:.1f}%")


def example_different_dimensions():
    """Example 2: Testing different dimensions."""
    print("\n" + "=" * 60)
    print("Example 2: Different Dimensions")
    print("=" * 60)
    
    dimensions = [32, 64, 128, 256, 512]
    
    print(f"{'Dimension':<12} {'Ratio':<10} {'Cosine':<10}")
    print("-" * 35)
    
    for dim in dimensions:
        config = PolarQuantConfig(dimension=dim, radius_bits=8, angle_bits=4)
        pq = PolarQuant(config)
        
        x = np.random.randn(dim)
        x = x / np.linalg.norm(x)
        
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        ratio = pq.compression_ratio()
        cosine = np.dot(x, x_recon) / (np.linalg.norm(x) * np.linalg.norm(x_recon))
        
        print(f"{dim:<12} {ratio:<10.2f} {cosine:<10.4f}")


def example_different_bit_widths():
    """Example 3: Testing different bit widths."""
    print("\n" + "=" * 60)
    print("Example 3: Different Bit Widths")
    print("=" * 60)
    
    dim = 128
    bit_configs = [
        (4, 2),
        (6, 3),
        (8, 4),
        (10, 6),
        (12, 8),
    ]
    
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    
    print(f"{'Radius Bits':<12} {'Angle Bits':<12} {'Ratio':<10} {'Cosine':<10}")
    print("-" * 50)
    
    for r_bits, a_bits in bit_configs:
        config = PolarQuantConfig(dimension=dim, radius_bits=r_bits, angle_bits=a_bits)
        pq = PolarQuant(config)
        
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        ratio = pq.compression_ratio()
        cosine = np.dot(x, x_recon) / (np.linalg.norm(x) * np.linalg.norm(x_recon))
        
        print(f"{r_bits:<12} {a_bits:<12} {ratio:<10.2f} {cosine:<10.4f}")


def example_batch_processing():
    """Example 4: Batch processing."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    config = PolarQuantConfig(dimension=64, radius_bits=8, angle_bits=4)
    pq = PolarQuant(config)
    batch_processor = PolarQuantBatch(pq)
    
    # Generate batch of vectors
    n_vectors = 100
    X = np.random.randn(n_vectors, 64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize each row
    
    print(f"Processing {n_vectors} vectors...")
    
    # Compress batch
    compressed_list = batch_processor.compress_batch(X)
    print(f"Compressed {len(compressed_list)} vectors")
    
    # Decompress batch
    X_reconstructed = batch_processor.decompress_batch(compressed_list)
    print(f"Reconstructed shape: {X_reconstructed.shape}")
    
    # Compute batch error metrics
    errors = batch_processor.compute_batch_error(X, X_reconstructed)
    print(f"\nBatch Error Metrics:")
    print(f"  Mean MSE: {errors['mean_mse']:.6f}")
    print(f"  Mean RMSE: {errors['mean_rmse']:.6f}")
    print(f"  Mean Cosine: {errors['mean_cosine']:.4f}")
    print(f"  Min Cosine: {errors['min_cosine']:.4f}")


def example_save_load():
    """Example 5: Save and load quantizer."""
    print("\n" + "=" * 60)
    print("Example 5: Save and Load")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Create and configure quantizer
    config = PolarQuantConfig(dimension=64, radius_bits=8, angle_bits=4, seed=42)
    pq = PolarQuant(config)
    
    # Compress a vector
    x = np.random.randn(64)
    x = x / np.linalg.norm(x)
    
    compressed = pq.compress(x)
    x_recon1 = pq.decompress(compressed)
    
    # Save quantizer
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        pq.save(temp_path)
        print(f"Quantizer saved to: {temp_path}")
        
        # Load quantizer
        pq_loaded = PolarQuant.load(temp_path)
        print("Quantizer loaded successfully")
        
        # Decompress with loaded quantizer
        x_recon2 = pq_loaded.decompress(compressed)
        
        # Verify they match
        if np.allclose(x_recon1, x_recon2):
            print("✓ Reconstructions match!")
        else:
            print("✗ Reconstructions differ")
            
    finally:
        os.unlink(temp_path)


def example_hadamard_rotation():
    """Example 6: Hadamard rotation vs random rotation."""
    print("\n" + "=" * 60)
    print("Example 6: Hadamard vs Random Rotation")
    print("=" * 60)
    
    dim = 64
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    
    # Random rotation
    config_random = PolarQuantConfig(dimension=dim, use_hadamard=False, seed=42)
    pq_random = PolarQuant(config_random)
    
    # Hadamard rotation
    config_hadamard = PolarQuantConfig(dimension=dim, use_hadamard=True, seed=42)
    pq_hadamard = PolarQuant(config_hadamard)
    
    # Test both
    compressed_r = pq_random.compress(x)
    compressed_h = pq_hadamard.compress(x)
    
    x_recon_r = pq_random.decompress(compressed_r)
    x_recon_h = pq_hadamard.decompress(compressed_h)
    
    cosine_r = np.dot(x, x_recon_r) / (np.linalg.norm(x) * np.linalg.norm(x_recon_r))
    cosine_h = np.dot(x, x_recon_h) / (np.linalg.norm(x) * np.linalg.norm(x_recon_h))
    
    print(f"Random Rotation:   Cosine = {cosine_r:.4f}")
    print(f"Hadamard Rotation: Cosine = {cosine_h:.4f}")
    print(f"\nNote: Hadamard rotation is faster (O(d log d)) but may have slightly different quality")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  PolarQuant: Basic Usage Examples".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    np.random.seed(42)  # For reproducibility
    
    example_basic_compression()
    example_different_dimensions()
    example_different_bit_widths()
    example_batch_processing()
    example_save_load()
    example_hadamard_rotation()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
