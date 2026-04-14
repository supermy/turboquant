"""
PolarQuant 工具函数单元测试

本模块测试 polarquant/utils.py 中的核心工具函数:
- 随机正交矩阵生成
- Hadamard 旋转
- 极坐标变换
- Lloyd-Max 量化
- Beta 分布参数计算

测试使用 unittest 框架，确保各函数正确性。
"""

import unittest
import numpy as np
from polarquant.utils import (
    random_orthogonal_matrix,    # 随机正交矩阵生成
    hadamard_rotation,           # Hadamard 旋转
    cartesian_to_polar,          # 笛卡尔转极坐标
    polar_to_cartesian,          # 极坐标转笛卡尔
    compute_lloyd_max_centroids, # 计算 Lloyd-Max 质心
    lloyd_max_quantize,          # Lloyd-Max 量化
    lloyd_max_dequantize,        # Lloyd-Max 反量化
    beta_distribution_params,    # Beta 分布参数
)


class TestRandomOrthogonalMatrix(unittest.TestCase):
    """测试随机正交矩阵生成"""
    
    def test_orthogonality(self):
        """测试生成的矩阵是正交矩阵 (Q @ Q.T = I)"""
        d = 10
        Q = random_orthogonal_matrix(d, seed=42)
        
        # 验证 Q @ Q.T = I
        identity = Q @ Q.T
        np.testing.assert_array_almost_equal(identity, np.eye(d), decimal=10)
    
    def test_determinant(self):
        """测试行列式为 +/- 1"""
        d = 5
        Q = random_orthogonal_matrix(d, seed=42)
        det = np.linalg.det(Q)
        self.assertAlmostEqual(abs(det), 1.0, places=10)
    
    def test_reproducibility(self):
        """测试相同种子产生相同矩阵"""
        d = 8
        Q1 = random_orthogonal_matrix(d, seed=123)
        Q2 = random_orthogonal_matrix(d, seed=123)
        np.testing.assert_array_equal(Q1, Q2)
    
    def test_different_seeds(self):
        """测试不同种子产生不同矩阵"""
        d = 8
        Q1 = random_orthogonal_matrix(d, seed=1)
        Q2 = random_orthogonal_matrix(d, seed=2)
        self.assertFalse(np.allclose(Q1, Q2))


class TestHadamardRotation(unittest.TestCase):
    """测试 Hadamard 旋转"""
    
    def test_norm_preservation(self):
        """测试 Hadamard 旋转保持范数"""
        d = 16
        x = np.random.randn(d)
        x_rotated = hadamard_rotation(x)
        
        np.testing.assert_almost_equal(
            np.linalg.norm(x), np.linalg.norm(x_rotated), decimal=10
        )
    
    def test_linearity(self):
        """测试 Hadamard 旋转是线性变换"""
        d = 8
        x1 = np.random.randn(d)
        x2 = np.random.randn(d)
        
        # 测试 H(a*x + b*y) = a*H(x) + b*H(y)
        a, b = 2.0, 3.0
        result1 = hadamard_rotation(a * x1 + b * x2)
        result2 = a * hadamard_rotation(x1) + b * hadamard_rotation(x2)
        
        np.testing.assert_array_almost_equal(result1, result2, decimal=10)
    
    def test_power_of_two(self):
        """测试 2 的幂次维度"""
        for d in [2, 4, 8, 16, 32]:
            x = np.random.randn(d)
            x_rotated = hadamard_rotation(x)
            self.assertEqual(len(x_rotated), d)
    
    def test_non_power_of_two(self):
        """测试非 2 的幂次维度"""
        d = 10
        x = np.random.randn(d)
        x_rotated = hadamard_rotation(x)
        self.assertEqual(len(x_rotated), d)


class TestPolarCoordinateTransform(unittest.TestCase):
    """测试笛卡尔/极坐标变换"""
    
    def test_roundtrip_2d(self):
        """测试 2D 往返转换"""
        x = np.array([3.0, 4.0])
        r, angles = cartesian_to_polar(x)
        x_recon = polar_to_cartesian(r, angles)
        
        np.testing.assert_array_almost_equal(x, x_recon, decimal=10)
    
    def test_roundtrip_3d(self):
        """测试 3D 往返转换"""
        x = np.array([1.0, 2.0, 2.0])
        r, angles = cartesian_to_polar(x)
        x_recon = polar_to_cartesian(r, angles)
        
        np.testing.assert_array_almost_equal(x, x_recon, decimal=10)
    
    def test_roundtrip_nd(self):
        """测试 nD 往返转换"""
        for d in [2, 4, 8, 16]:
            x = np.random.randn(d)
            r, angles = cartesian_to_polar(x)
            x_recon = polar_to_cartesian(r, angles)
            
            np.testing.assert_array_almost_equal(x, x_recon, decimal=8)
    
    def test_radius_computation(self):
        """测试半径计算正确"""
        x = np.array([3.0, 4.0])
        r, _ = cartesian_to_polar(x)
        self.assertAlmostEqual(r, 5.0, places=10)
    
    def test_zero_vector(self):
        """测试零向量处理"""
        x = np.zeros(5)
        r, angles = cartesian_to_polar(x)
        
        self.assertEqual(r, 0.0)
        np.testing.assert_array_equal(angles, np.zeros(4))
    
    def test_unit_vector(self):
        """测试单位向量"""
        d = 4
        x = np.array([1.0, 0.0, 0.0, 0.0])
        r, angles = cartesian_to_polar(x)
        
        self.assertAlmostEqual(r, 1.0, places=10)


class TestLloydMaxQuantization(unittest.TestCase):
    """测试 Lloyd-Max 量化"""
    
    def test_centroids_in_range(self):
        """测试质心在 [0, 1] 范围内"""
        alpha, beta_param = 2.0, 2.0
        bits = 4
        
        centroids = compute_lloyd_max_centroids(alpha, beta_param, bits)
        
        self.assertTrue(np.all(centroids >= 0))
        self.assertTrue(np.all(centroids <= 1))
        self.assertEqual(len(centroids), 2 ** bits)
    
    def test_quantization_dequantization(self):
        """测试量化后反量化"""
        alpha, beta_param = 2.0, 2.0
        bits = 3
        
        centroids = compute_lloyd_max_centroids(alpha, beta_param, bits)
        
        # 测试值
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        indices = lloyd_max_quantize(x, centroids)
        x_recon = lloyd_max_dequantize(indices, centroids)
        
        # 检查索引有效
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 2 ** bits))
        
        # 检查重建值是质心
        for val in x_recon:
            self.assertIn(val, centroids)
    
    def test_convergence(self):
        """测试 Lloyd-Max 算法收敛"""
        alpha, beta_param = 4.0, 4.0
        bits = 4
        
        # 不同迭代次数
        centroids_10 = compute_lloyd_max_centroids(alpha, beta_param, bits, n_iterations=10)
        centroids_100 = compute_lloyd_max_centroids(alpha, beta_param, bits, n_iterations=100)
        
        # 充分迭代后应相似
        np.testing.assert_array_almost_equal(centroids_10, centroids_100, decimal=1)


class TestBetaDistributionParams(unittest.TestCase):
    """测试 Beta 分布参数计算"""
    
    def test_symmetric_params(self):
        """测试参数对称 (alpha = beta)"""
        for d in [2, 4, 8, 16, 32]:
            alpha, beta_param = beta_distribution_params(d)
            self.assertEqual(alpha, beta_param)
            self.assertEqual(alpha, d / 2)
    
    def test_positive_params(self):
        """测试参数为正"""
        for d in [2, 4, 8]:
            alpha, beta_param = beta_distribution_params(d)
            self.assertGreater(alpha, 0)
            self.assertGreater(beta_param, 0)


if __name__ == '__main__':
    unittest.main()
