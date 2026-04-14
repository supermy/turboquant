"""
Unit tests for core PolarQuant functionality.
"""

import unittest
import numpy as np
import tempfile
import os

from polarquant.core import PolarQuant, PolarQuantConfig, CompressedVector, PolarQuantBatch


class TestPolarQuantConfig(unittest.TestCase):
    """Test PolarQuantConfig."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = PolarQuantConfig(dimension=64, radius_bits=8, angle_bits=4)
        self.assertEqual(config.dimension, 64)
        self.assertEqual(config.radius_bits, 8)
        self.assertEqual(config.angle_bits, 4)
    
    def test_invalid_dimension(self):
        """Test invalid dimension."""
        with self.assertRaises(ValueError):
            PolarQuantConfig(dimension=1)
    
    def test_invalid_bits(self):
        """Test invalid bit settings."""
        with self.assertRaises(ValueError):
            PolarQuantConfig(dimension=10, radius_bits=0)
        with self.assertRaises(ValueError):
            PolarQuantConfig(dimension=10, radius_bits=20)
        with self.assertRaises(ValueError):
            PolarQuantConfig(dimension=10, angle_bits=0)


class TestCompressedVector(unittest.TestCase):
    """Test CompressedVector."""
    
    def test_size_calculation(self):
        """Test size calculation."""
        cv = CompressedVector(
            radius_idx=10,
            angle_indices=np.array([1, 2, 3, 4]),
            original_norm=1.0
        )
        self.assertGreater(cv.size_bits(), 0)
        self.assertGreater(cv.size_bytes(), 0)


class TestPolarQuant(unittest.TestCase):
    """Test main PolarQuant class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PolarQuantConfig(dimension=16, radius_bits=8, angle_bits=4, seed=42)
        self.pq = PolarQuant(self.config)
    
    def test_initialization(self):
        """Test PolarQuant initialization."""
        self.assertEqual(self.pq.dimension, 16)
        self.assertIsNotNone(self.pq.rotation_matrix)
        self.assertIsNotNone(self.pq.angle_centroids)
    
    def test_compression_decompression(self):
        """Test basic compression and decompression."""
        x = np.random.randn(16)
        
        compressed = self.pq.compress(x)
        self.assertIsInstance(compressed, CompressedVector)
        
        x_recon = self.pq.decompress(compressed)
        self.assertEqual(len(x_recon), 16)
    
    def test_reconstruction_quality(self):
        """Test reconstruction quality."""
        np.random.seed(42)
        x = np.random.randn(16)
        x = x / np.linalg.norm(x)  # Normalize
        
        compressed = self.pq.compress(x)
        x_recon = self.pq.decompress(compressed)
        
        # Check cosine similarity
        cosine_sim = np.dot(x, x_recon) / (np.linalg.norm(x) * np.linalg.norm(x_recon))
        self.assertGreater(cosine_sim, 0.8)  # Should be reasonably high
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        x = np.random.randn(10)  # Wrong dimension
        with self.assertRaises(ValueError):
            self.pq.compress(x)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        ratio = self.pq.compression_ratio()
        self.assertGreater(ratio, 1.0)  # Should achieve compression
        
        # Expected: 16*32 / (8 + 15*4) = 512 / 68 ≈ 7.5
        expected_ratio = (16 * 32) / (8 + 15 * 4)
        self.assertAlmostEqual(ratio, expected_ratio, places=5)
    
    def test_error_metrics(self):
        """Test error metric computation."""
        x = np.random.randn(16)
        compressed = self.pq.compress(x)
        x_recon = self.pq.decompress(compressed)
        
        errors = self.pq.compute_error(x, x_recon)
        
        self.assertIn('mse', errors)
        self.assertIn('rmse', errors)
        self.assertIn('cosine_similarity', errors)
        self.assertIn('relative_error', errors)
        
        self.assertGreaterEqual(errors['mse'], 0)
        self.assertGreaterEqual(errors['rmse'], 0)
    
    def test_zero_vector(self):
        """Test handling of zero vector."""
        x = np.zeros(16)
        compressed = self.pq.compress(x)
        x_recon = self.pq.decompress(compressed)
        
        # Should handle gracefully
        self.assertEqual(len(x_recon), 16)
    
    def test_save_load(self):
        """Test saving and loading."""
        # Compress a vector
        x = np.random.randn(16)
        compressed1 = self.pq.compress(x)
        x_recon1 = self.pq.decompress(compressed1)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            self.pq.save(temp_path)
            
            # Load
            pq_loaded = PolarQuant.load(temp_path)
            
            # Should produce same reconstruction
            x_recon2 = pq_loaded.decompress(compressed1)
            np.testing.assert_array_almost_equal(x_recon1, x_recon2)
        finally:
            os.unlink(temp_path)
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        config1 = PolarQuantConfig(dimension=16, seed=42)
        config2 = PolarQuantConfig(dimension=16, seed=42)
        
        pq1 = PolarQuant(config1)
        pq2 = PolarQuant(config2)
        
        x = np.random.randn(16)
        
        compressed1 = pq1.compress(x)
        compressed2 = pq2.compress(x)
        
        # Should produce same compressed representation
        self.assertEqual(compressed1.radius_idx, compressed2.radius_idx)
        np.testing.assert_array_equal(compressed1.angle_indices, compressed2.angle_indices)


class TestPolarQuantDifferentDimensions(unittest.TestCase):
    """Test with different dimensions."""
    
    def test_dimension_4(self):
        """Test with dimension 4."""
        config = PolarQuantConfig(dimension=4, radius_bits=6, angle_bits=3)
        pq = PolarQuant(config)
        
        x = np.random.randn(4)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 4)
    
    def test_dimension_64(self):
        """Test with dimension 64."""
        config = PolarQuantConfig(dimension=64, radius_bits=8, angle_bits=4)
        pq = PolarQuant(config)
        
        x = np.random.randn(64)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 64)
    
    def test_dimension_256(self):
        """Test with dimension 256."""
        config = PolarQuantConfig(dimension=256, radius_bits=8, angle_bits=4)
        pq = PolarQuant(config)
        
        x = np.random.randn(256)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 256)
        
        # Higher dimension should have better cosine similarity
        x_norm = x / np.linalg.norm(x)
        cosine_sim = np.dot(x_norm, x_recon) / np.linalg.norm(x_recon)
        self.assertGreater(cosine_sim, 0.9)


class TestPolarQuantDifferentBitWidths(unittest.TestCase):
    """Test with different bit widths."""
    
    def test_low_bits(self):
        """Test with low bit width."""
        config = PolarQuantConfig(dimension=16, radius_bits=4, angle_bits=2)
        pq = PolarQuant(config)
        
        x = np.random.randn(16)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
    
    def test_high_bits(self):
        """Test with high bit width."""
        config = PolarQuantConfig(dimension=16, radius_bits=12, angle_bits=8)
        pq = PolarQuant(config)
        
        x = np.random.randn(16)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
        
        # Higher bits should give better reconstruction
        x_norm = x / np.linalg.norm(x)
        cosine_sim = np.dot(x_norm, x_recon) / np.linalg.norm(x_recon)
        self.assertGreater(cosine_sim, 0.95)


class TestPolarQuantBatch(unittest.TestCase):
    """Test batch processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PolarQuantConfig(dimension=16, seed=42)
        self.pq = PolarQuant(self.config)
        self.batch_processor = PolarQuantBatch(self.pq)
    
    def test_batch_compression(self):
        """Test batch compression."""
        X = np.random.randn(10, 16)
        compressed_list = self.batch_processor.compress_batch(X)
        
        self.assertEqual(len(compressed_list), 10)
        for cv in compressed_list:
            self.assertIsInstance(cv, CompressedVector)
    
    def test_batch_decompression(self):
        """Test batch decompression."""
        X = np.random.randn(5, 16)
        compressed_list = self.batch_processor.compress_batch(X)
        X_recon = self.batch_processor.decompress_batch(compressed_list)
        
        self.assertEqual(X_recon.shape, (5, 16))
    
    def test_batch_error_metrics(self):
        """Test batch error metrics."""
        X = np.random.randn(10, 16)
        compressed_list = self.batch_processor.compress_batch(X)
        X_recon = self.batch_processor.decompress_batch(compressed_list)
        
        errors = self.batch_processor.compute_batch_error(X, X_recon)
        
        self.assertIn('mean_mse', errors)
        self.assertIn('mean_cosine', errors)
        self.assertGreater(errors['mean_cosine'], 0.8)


class TestPolarQuantWithHadamard(unittest.TestCase):
    """Test PolarQuant with Hadamard rotation."""
    
    def test_hadamard_rotation(self):
        """Test with Hadamard rotation enabled."""
        config = PolarQuantConfig(dimension=16, use_hadamard=True, seed=42)
        pq = PolarQuant(config)
        
        x = np.random.randn(16)
        compressed = pq.compress(x)
        x_recon = pq.decompress(compressed)
        
        self.assertEqual(len(x_recon), 16)
    
    def test_hadamard_vs_random(self):
        """Compare Hadamard vs random rotation."""
        x = np.random.randn(16)
        
        config_hadamard = PolarQuantConfig(dimension=16, use_hadamard=True, seed=42)
        config_random = PolarQuantConfig(dimension=16, use_hadamard=False, seed=42)
        
        pq_hadamard = PolarQuant(config_hadamard)
        pq_random = PolarQuant(config_random)
        
        # Both should work
        compressed_h = pq_hadamard.compress(x)
        compressed_r = pq_random.compress(x)
        
        x_recon_h = pq_hadamard.decompress(compressed_h)
        x_recon_r = pq_random.decompress(compressed_r)
        
        self.assertEqual(len(x_recon_h), 16)
        self.assertEqual(len(x_recon_r), 16)


if __name__ == '__main__':
    unittest.main()
