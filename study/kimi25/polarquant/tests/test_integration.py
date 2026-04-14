"""
Integration tests for PolarQuant.

These tests verify end-to-end functionality with realistic use cases.
"""

import unittest
import numpy as np

from polarquant import PolarQuant, PolarQuantConfig


class TestKVCacheSimulation(unittest.TestCase):
    """Simulate KV cache compression as in LLM inference."""
    
    def setUp(self):
        """Set up typical KV cache dimensions."""
        # Typical dimensions for LLM KV cache
        self.head_dim = 64  # Attention head dimension
        self.config = PolarQuantConfig(
            dimension=self.head_dim,
            radius_bits=8,
            angle_bits=4,
            seed=42
        )
        self.pq = PolarQuant(self.config)
    
    def test_kv_cache_compression(self):
        """Test compression of KV cache entries."""
        # Simulate 100 KV pairs
        n_tokens = 100
        
        keys = []
        values = []
        compressed_keys = []
        compressed_values = []
        
        for i in range(n_tokens):
            # Generate random key and value
            key = np.random.randn(self.head_dim)
            value = np.random.randn(self.head_dim)
            
            # Normalize (common in attention)
            key = key / np.linalg.norm(key)
            value = value / np.linalg.norm(value)
            
            keys.append(key)
            values.append(value)
            
            # Compress
            compressed_keys.append(self.pq.compress(key))
            compressed_values.append(self.pq.compress(value))
        
        # Decompress and verify
        reconstructed_keys = [self.pq.decompress(ck) for ck in compressed_keys]
        reconstructed_values = [self.pq.decompress(cv) for cv in compressed_values]
        
        # Check reconstruction quality
        cosine_sims = []
        for orig, recon in zip(keys, reconstructed_keys):
            sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
            cosine_sims.append(sim)
        
        mean_cosine = np.mean(cosine_sims)
        self.assertGreater(mean_cosine, 0.85, 
                          f"Mean cosine similarity {mean_cosine} too low")
    
    def test_attention_score_preservation(self):
        """Test that attention scores are preserved after compression."""
        # Generate query and keys
        query = np.random.randn(self.head_dim)
        query = query / np.linalg.norm(query)
        
        n_keys = 50
        keys = [np.random.randn(self.head_dim) for _ in range(n_keys)]
        keys = [k / np.linalg.norm(k) for k in keys]
        
        # Compute original attention scores
        original_scores = [np.dot(query, k) for k in keys]
        
        # Compress and decompress keys
        compressed_keys = [self.pq.compress(k) for k in keys]
        reconstructed_keys = [self.pq.decompress(ck) for ck in compressed_keys]
        
        # Compute reconstructed attention scores
        reconstructed_scores = [np.dot(query, k) for k in reconstructed_keys]
        
        # Check correlation between original and reconstructed scores
        correlation = np.corrcoef(original_scores, reconstructed_scores)[0, 1]
        self.assertGreater(correlation, 0.9, 
                          f"Attention score correlation {correlation} too low")


class TestEmbeddingCompression(unittest.TestCase):
    """Test compression of embedding vectors."""
    
    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings (e.g., from BERT)."""
        config = PolarQuantConfig(dimension=768, radius_bits=8, angle_bits=4)
        pq = PolarQuant(config)
        
        # Generate random embeddings
        n_embeddings = 10
        embeddings = [np.random.randn(768) for _ in range(n_embeddings)]
        
        # Compress and decompress
        compressed = [pq.compress(e) for e in embeddings]
        reconstructed = [pq.decompress(c) for c in compressed]
        
        # Check quality
        cosine_sims = []
        for orig, recon in zip(embeddings, reconstructed):
            sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
            cosine_sims.append(sim)
        
        mean_cosine = np.mean(cosine_sims)
        self.assertGreater(mean_cosine, 0.9)
    
    def test_semantic_similarity_preservation(self):
        """Test that semantic similarity is preserved."""
        config = PolarQuantConfig(dimension=256, radius_bits=8, angle_bits=4)
        pq = PolarQuant(config)
        
        # Create similar and dissimilar vectors
        np.random.seed(42)  # For reproducibility
        base = np.random.randn(256)
        base = base / np.linalg.norm(base)
        
        # Create a very similar vector (small perturbation)
        similar = base + 0.05 * np.random.randn(256)
        similar = similar / np.linalg.norm(similar)
        
        dissimilar = np.random.randn(256)
        dissimilar = dissimilar / np.linalg.norm(dissimilar)
        
        # Compress
        base_compressed = pq.compress(base)
        similar_compressed = pq.compress(similar)
        dissimilar_compressed = pq.compress(dissimilar)
        
        # Decompress
        base_recon = pq.decompress(base_compressed)
        similar_recon = pq.decompress(similar_compressed)
        dissimilar_recon = pq.decompress(dissimilar_compressed)
        
        # Check similarities
        orig_sim = np.dot(base, similar)
        recon_sim = np.dot(base_recon, similar_recon)
        
        orig_dissim = np.dot(base, dissimilar)
        recon_dissim = np.dot(base_recon, dissimilar_recon)
        
        # Check that reconstruction preserves relative similarity
        # The reconstructed similarity should be correlated with original
        self.assertGreater(recon_sim, 0.5)  # Similar vectors should have positive similarity
        
        # Dissimilar vectors should remain dissimilar (near zero)
        self.assertLess(abs(recon_dissim), 0.6)


class TestCompressionRatios(unittest.TestCase):
    """Test compression ratios for different configurations."""
    
    def test_compression_ratio_calculation(self):
        """Verify compression ratio calculations."""
        test_cases = [
            (64, 8, 4),   # d=64, 8-bit radius, 4-bit angles
            (128, 8, 4),  # d=128
            (256, 8, 4),  # d=256
            (512, 8, 4),  # d=512
        ]
        
        for dim, r_bits, a_bits in test_cases:
            config = PolarQuantConfig(
                dimension=dim,
                radius_bits=r_bits,
                angle_bits=a_bits
            )
            pq = PolarQuant(config)
            
            ratio = pq.compression_ratio()
            
            # Expected: (dim * 32) / (r_bits + (dim-1) * a_bits)
            expected = (dim * 32) / (r_bits + (dim - 1) * a_bits)
            
            self.assertAlmostEqual(ratio, expected, places=5,
                                 msg=f"Failed for dim={dim}")
    
    def test_higher_dimensions_better_ratios(self):
        """Test that higher dimensions achieve better compression ratios."""
        dims = [32, 64, 128, 256, 512]
        ratios = []
        
        for dim in dims:
            config = PolarQuantConfig(dimension=dim, radius_bits=8, angle_bits=4)
            pq = PolarQuant(config)
            ratios.append(pq.compression_ratio())
        
        # Higher dimensions should have better (higher) compression ratios
        for i in range(len(ratios) - 1):
            self.assertGreaterEqual(
                ratios[i + 1], ratios[i],
                f"Dimension {dims[i+1]} should have better ratio than {dims[i]}"
            )


class TestRobustness(unittest.TestCase):
    """Test robustness to various inputs."""
    
    def test_very_small_values(self):
        """Test with very small values."""
        config = PolarQuantConfig(dimension=16)
        pq = PolarQuant(config)
        
        x = 1e-6 * np.random.randn(16)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
        self.assertTrue(np.all(np.isfinite(x_recon)))
    
    def test_very_large_values(self):
        """Test with very large values."""
        config = PolarQuantConfig(dimension=16)
        pq = PolarQuant(config)
        
        x = 1e6 * np.random.randn(16)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
        self.assertTrue(np.all(np.isfinite(x_recon)))
    
    def test_mixed_scale_values(self):
        """Test with mixed scale values."""
        config = PolarQuantConfig(dimension=16)
        pq = PolarQuant(config)
        
        x = np.random.randn(16)
        x[0] = 1e3  # Large value
        x[1] = 1e-3  # Small value
        
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
        self.assertTrue(np.all(np.isfinite(x_recon)))


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks."""
    
    def test_compression_speed(self):
        """Benchmark compression speed."""
        config = PolarQuantConfig(dimension=256)
        pq = PolarQuant(config)
        
        n_vectors = 100
        vectors = [np.random.randn(256) for _ in range(n_vectors)]
        
        import time
        start = time.time()
        compressed = [pq.compress(v) for v in vectors]
        elapsed = time.time() - start
        
        # Should be reasonably fast (adjust threshold as needed)
        self.assertLess(elapsed, 10.0, 
                       f"Compression took {elapsed:.2f}s, too slow")
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        from polarquant.core import PolarQuantBatch
        
        config = PolarQuantConfig(dimension=128)
        pq = PolarQuant(config)
        batch_processor = PolarQuantBatch(pq)
        
        # Large batch
        X = np.random.randn(100, 128)
        
        import time
        start = time.time()
        compressed = batch_processor.compress_batch(X)
        X_recon = batch_processor.decompress_batch(compressed)
        elapsed = time.time() - start
        
        self.assertEqual(X_recon.shape, (100, 128))
        self.assertLess(elapsed, 15.0,
                       f"Batch processing took {elapsed:.2f}s, too slow")


if __name__ == '__main__':
    unittest.main()
