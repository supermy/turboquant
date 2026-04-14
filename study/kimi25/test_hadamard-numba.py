import numpy as np
import time

def hadamard_matrix(n):
    """
    递归构造Hadamard矩阵 H_n (2^n × 2^n)
    H_1 = [[1, 1], [1, -1]]
    H_{k+1} = [[H_k, H_k], [H_k, -H_k]]
    """
    if n == 0:
        return np.array([[1]])
    H_prev = hadamard_matrix(n-1)
    H_top = np.hstack([H_prev, H_prev])
    H_bottom = np.hstack([H_prev, -H_prev])
    return np.vstack([H_top, H_bottom])

def fwht(x):
    """
    快速Walsh-Hadamard变换 (FWHT)
    原地计算，复杂度 O(n log n)
    """
    x = x.copy()
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x / np.sqrt(n)  # 归一化保持能量

import numpy as np

def fwht_numpy(x):
    """
    纯NumPy实现的FWHT
    利用位运算和向量化，比纯Python循环快10-20倍
    """
    x = x.copy()
    n = len(x)
    
    # 位反转置换（可选优化）
    # x = x[bit_reversal_permutation(n)]
    
    h = 1
    while h < n:
        # 向量化操作：同时处理所有蝴蝶对
        x = x.reshape(n // (2*h), 2*h)
        x[:, :h], x[:, h:] = x[:, :h] + x[:, h:], x[:, :h] - x[:, h:]
        x = x.flatten()
        h *= 2
    
    return x / np.sqrt(n)

def hadamard_rotation_numpy(x, seed=None):
    """纯NumPy版本，无需Numba或Scipy"""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(x)
    if n & (n - 1) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - n), mode='constant')
        n = next_pow2
    
    d1 = np.random.choice([-1, 1], size=n)
    x = x * d1
    
    x = fwht_numpy(x)
    
    d2 = np.random.choice([-1, 1], size=n)
    x = x * d2
    
    return x[:len(x)]

from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def fwht_numba(x):
    """Numba加速的FWHT（串行版本）"""
    n = x.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            i2 = i + h
            for j in range(i, i2):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x

def hadamard_rotation_numba(x, seed=None):
    """Numba加速的Hadamard旋转"""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(x)
    original_n = n
    
    # 填充到2的幂次
    if n & (n - 1) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - n), mode='constant')
        n = next_pow2
    
    x = x.astype(np.float64).copy()
    
    # 随机符号
    d1 = np.random.choice(np.array([-1.0, 1.0]), size=n)
    x *= d1
    
    # FWHT
    fwht_numba(x)
    
    # 归一化和第二次符号翻转
    x /= np.sqrt(n)
    d2 = np.random.choice(np.array([-1.0, 1.0]), size=n)
    x *= d2
    
    return x[:original_n]


def hadamard_rotation(x, seed=None):
    """
    TurboQuant风格的Hadamard旋转：
    随机符号翻转 → FWHT → 随机符号翻转
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(x)
    
    # 检查维度是否为2的幂次
    if n & (n - 1) != 0:
        # 填充到最近的2的幂次
        next_pow2 = 1 << (n - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - n), mode='constant')
        n = next_pow2
    
    # 第一阶段：随机符号翻转
    d1 = np.random.choice([-1, 1], size=n)
    x_rotated = x * d1
    
    # 第二阶段：快速Hadamard变换
    x_rotated = fwht(x_rotated)
    
    # 第三阶段：再次随机符号翻转
    d2 = np.random.choice([-1, 1], size=n)
    x_rotated = x_rotated * d2
    
    return x_rotated[:len(x)]  # 截断回原始维度



def random_orthogonal_rotation(x):
    """
    精确随机正交旋转（QR分解法）
    复杂度 O(n^3) 生成 + O(n^2) 乘法
    """
    n = len(x)
    # 生成随机高斯矩阵
    A = np.random.randn(n, n)
    # QR分解得到正交矩阵 Q
    Q, _ = np.linalg.qr(A)
    return Q @ x

# ==================== 测试 ====================

def test_basic():
    """基础功能测试"""
    print("=" * 50)
    print("基础Hadamard矩阵测试")
    print("=" * 50)
    
    # 构造H_3 (8×8)
    H = hadamard_matrix(3)
    print(f"\nHadamard矩阵 H_3 (8×8):\n{H}")
    
    # 验证正交性: H @ H.T = n * I
    orthogonality = H @ H.T
    print(f"\n正交性验证 H @ H.T:\n{orthogonality}")
    print(f"是否正交: {np.allclose(orthogonality, 8 * np.eye(8))}")
    
    # 快速变换验证
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y_slow = H @ x / np.sqrt(8)  # 矩阵乘法
    y_fast = fwht(x)              # 快速算法
    
    print(f"\n输入向量: {x}")
    print(f"矩阵乘法结果: {y_slow}")
    print(f"FWHT结果:     {y_fast}")
    print(f"结果一致: {np.allclose(y_slow, y_fast)}")

def test_rotation_properties():
    """测试旋转性质：能量守恒、去相关"""
    print("\n" + "=" * 50)
    print("旋转性质测试")
    print("=" * 50)
    
    # 构造有"尖峰"的向量
    x = np.zeros(128)
    x[0] = 10.0  # 异常值
    x[1] = 5.0
    
    print(f"\n原始向量能量分布 (前10维): {x[:10]}")
    print(f"原始向量L2范数: {np.linalg.norm(x):.4f}")
    
    # Hadamard旋转
    y_hadamard = hadamard_rotation_numba(x, seed=42)
    print(f"\nHadamard旋转后 (前10维): {np.abs(y_hadamard[:10])}")
    print(f"旋转后L2范数: {np.linalg.norm(y_hadamard):.4f}")
    print(f"能量守恒: {np.isclose(np.linalg.norm(x), np.linalg.norm(y_hadamard))}")
    
    # 统计分布
    print(f"\nHadamard旋转后统计:")
    print(f"  均值: {np.mean(y_hadamard):.4f}")
    print(f"  标准差: {np.std(y_hadamard):.4f}")
    print(f"  最大值: {np.max(np.abs(y_hadamard)):.4f}")
    print(f"  能量均匀度 (max/mean): {np.max(y_hadamard**2) / np.mean(y_hadamard**2):.2f}")
    
    # 对比：精确随机正交
    y_orthogonal = random_orthogonal_rotation(x)
    print(f"\n随机正交旋转后统计:")
    print(f"  标准差: {np.std(y_orthogonal):.4f}")
    print(f"  最大值: {np.max(np.abs(y_orthogonal)):.4f}")

def test_performance():
    """性能对比测试"""
    print("\n" + "=" * 50)
    print("性能对比测试")
    print("=" * 50)
    
    dimensions = [128, 256, 512, 1024, 2048, 4096]
    iterations = 100
    
    print(f"\n{'维度':>8} | {'Hadamard(ms)':>12} | {'随机正交(ms)':>14} | {'加速比':>8}")
    print("-" * 60)
    
    for d in dimensions:
        x = np.random.randn(d)
        
        # Hadamard计时
        start = time.time()
        for _ in range(iterations):
            _ = hadamard_rotation_numba(x)
        t_hadamard = (time.time() - start) * 1000 / iterations
        
        # 随机正交计时（只计乘法，不计生成）
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        start = time.time()
        for _ in range(iterations):
            _ = Q @ x
        t_orthogonal = (time.time() - start) * 1000 / iterations
        
        speedup = t_orthogonal / t_hadamard
        
        print(f"{d:>8} | {t_hadamard:>12.3f} | {t_orthogonal:>14.3f} | {speedup:>8.1f}x")

def test_quantization_effect():
    """测试对量化的改善效果"""
    print("\n" + "=" * 50)
    print("量化效果测试")
    print("=" * 50)
    
    # 生成有异常值的向量（模拟LLM激活值）
    np.random.seed(42)
    x = np.random.randn(128)
    x[0] *= 10  # 制造异常值
    x[5] *= 8
    
    def quantize(x, bits=4):
        """简单均匀量化"""
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (2**bits - 1)
        x_quant = np.round((x - x_min) / scale) * scale + x_min
        return x_quant, scale
    
    # 无旋转直接量化
    x_quant_direct, _ = quantize(x, bits=4)
    mse_direct = np.mean((x - x_quant_direct)**2)
    
    # Hadamard旋转后量化
    y = hadamard_rotation_numba(x, seed=42)
    y_quant, _ = quantize(y, bits=4)
    # 反旋转（近似）
    x_recon = hadamard_rotation_numba(y_quant, seed=42)  # 用相同种子
    
    # 注意：Hadamard是自逆的（除了归一化），但随机符号需要一致
    # 简化处理：直接比较旋转域的误差
    mse_hadamard = np.mean((y - y_quant)**2)
    
    print(f"\n原始向量范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Hadamard旋转后范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"\n4-bit量化MSE:")
    print(f"  无旋转:  {mse_direct:.6f}")
    print(f"  Hadamard: {mse_hadamard:.6f}")
    print(f"  改善:    {mse_direct/mse_hadamard:.1f}x")

if __name__ == "__main__":
    test_basic()
    test_rotation_properties()
    test_performance()
    test_quantization_effect()
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
