"""
PolarQuant 算法工具函数模块

本模块提供 PolarQuant 算法的核心数学运算，包括:
- 随机正交矩阵生成: 用于分布归一化
- Hadamard 旋转: 快速结构化正交变换
- 笛卡尔/极坐标转换: 高维向量的坐标变换
- Lloyd-Max 量化: 基于 Beta 分布的最优标量量化

数学原理:
1. 随机旋转后，各坐标分量服从 Beta(d/2, d/2) 分布
2. 极坐标变换将向量分解为半径和角度
3. Lloyd-Max 算法迭代优化量化质心，最小化均方误差
"""

import numpy as np
from scipy.stats import beta
from typing import Tuple, Optional


def random_orthogonal_matrix(d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    使用 QR 分解生成随机正交矩阵
    
    随机旋转是 PolarQuant 的关键步骤，它确保旋转后的每个坐标
    服从 Beta(d/2, d/2) 分布，这是 Lloyd-Max 量化的理论基础。
    
    数学原理:
        - 随机矩阵 A 的元素服从标准正态分布 N(0, 1)
        - QR 分解: A = QR，其中 Q 是正交矩阵，R 是上三角矩阵
        - Q 的列向量是 A 列空间的正交基
        - 调整行列式为 1 确保是 proper rotation（非反射）
    
    参数:
        d: 矩阵维度，生成 d×d 正交矩阵
        seed: 随机种子，用于结果可复现
        
    返回:
        d×d 正交矩阵 Q，满足 Q @ Q.T = I（单位矩阵）
        
    示例:
        >>> Q = random_orthogonal_matrix(256, seed=42)
        >>> np.allclose(Q @ Q.T, np.eye(256))  # 验证正交性
        True
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机矩阵，元素服从标准正态分布
    A = np.random.randn(d, d)
    
    # QR 分解得到正交矩阵 Q
    Q, R = np.linalg.qr(A)
    
    # 确保行列式为 1（proper rotation）
    # 如果行列式为 -1，说明包含反射，需要调整
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q


def hadamard_rotation(x: np.ndarray) -> np.ndarray:
    """
    对输入向量应用 Hadamard 旋转
    
    Hadamard 旋转是一种结构化的正交变换，可以使用
    快速 Walsh-Hadamard 变换 (FWHT) 在 O(d log d) 时间内计算。
    
    与随机正交旋转相比:
    - 优点: 计算更快 O(d log d) vs O(d²)，无需存储矩阵
    - 缺点: 是确定性变换，随机性较弱，理论保证稍差
    
    算法步骤:
        1. 将输入向量填充到 2 的幂次长度
        2. 应用蝶形运算（butterfly operation）
        3. 归一化以保持范数
    
    参数:
        x: 输入向量，形状为 (d,)
        
    返回:
        旋转后的向量，形状为 (d,)
        
    参考:
        Fast Walsh-Hadamard Transform 是一种快速算法，
        类似于 FFT，用于计算 Hadamard 矩阵与向量的乘积
    """
    d = len(x)
    
    # 如果维度不是 2 的幂，填充到最近的 2 的幂
    # 使用位运算: 1 << (d-1).bit_length() 得到大于等于 d 的最小 2 的幂
    d_pad = 1 << (d - 1).bit_length()
    if d_pad != d:
        x_padded = np.zeros(d_pad)
        x_padded[:d] = x.copy()
    else:
        x_padded = x.copy()
    
    # 快速 Walsh-Hadamard 变换 (FWHT)
    # 使用蝶形运算，时间复杂度 O(d log d)
    h = 1
    while h < len(x_padded):
        # 对每个块进行蝶形运算
        for i in range(0, len(x_padded), h * 2):
            for j in range(i, i + h):
                # 蝶形运算: [a, b] → [a+b, a-b]
                a = x_padded[j]
                b = x_padded[j + h]
                x_padded[j] = a + b
                x_padded[j + h] = a - b
        h *= 2
    
    # 归一化: 保持向量范数不变
    # Hadamard 矩阵 H 满足 H @ H.T = d * I，所以需要除以 sqrt(d)
    x_padded = x_padded / np.sqrt(d_pad)
    
    return x_padded[:d]


def cartesian_to_polar(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    将笛卡尔坐标转换为极坐标
    
    对于 d 维向量，极坐标表示为:
        - r: 半径（L2 范数）
        - angles: d-1 个角度，前 d-2 个在 [0, π]，最后一个在 [0, 2π]
    
    坐标变换公式:
        x₀ = r · cos(θ₀)
        x₁ = r · sin(θ₀) · cos(θ₁)
        x₂ = r · sin(θ₀) · sin(θ₁) · cos(θ₂)
        ...
        x_{d-2} = r · sin(θ₀) · ... · sin(θ_{d-3}) · cos(θ_{d-2})
        x_{d-1} = r · sin(θ₀) · ... · sin(θ_{d-3}) · sin(θ_{d-2})
    
    逆变换使用递推公式:
        cos(θᵢ) = xᵢ / ||x[i:]||
        其中 ||x[i:]|| 是剩余分量的范数
    
    参数:
        x: 输入向量，形状为 (d,)
        
    返回:
        (radius, angles) 元组:
        - radius: 半径 r（非负浮点数）
        - angles: 角度数组，形状为 (d-1,)
        
    示例:
        >>> x = np.array([3.0, 4.0])  # 2D 向量
        >>> r, angles = cartesian_to_polar(x)
        >>> r  # 半径 = 5.0
        5.0
    """
    d = len(x)
    
    # 计算半径: r = ||x||₂
    r = np.linalg.norm(x)
    
    if r < 1e-10:
        # 零向量特殊情况
        return 0.0, np.zeros(d - 1)
    
    # 归一化到单位球面
    x_norm = x / r
    
    # 使用递推公式计算角度
    angles = np.zeros(d - 1)
    
    for i in range(d - 1):
        # 计算剩余分量的范数
        remaining_norm = np.linalg.norm(x_norm[i:])
        if remaining_norm < 1e-10:
            angles[i] = 0.0
        else:
            # cos(θᵢ) = xᵢ / ||x[i:]||
            cos_theta = np.clip(x_norm[i] / remaining_norm, -1.0, 1.0)
            angles[i] = np.arccos(cos_theta)
    
    # 根据最后一个坐标的符号调整最后一个角度
    # 确保角度覆盖完整的 [0, 2π] 范围
    if x[-1] < 0:
        angles[-1] = 2 * np.pi - angles[-1]
    
    return r, angles


def polar_to_cartesian(r: float, angles: np.ndarray) -> np.ndarray:
    """
    将极坐标转换回笛卡尔坐标
    
    使用递推公式:
        x₀ = r · cos(θ₀)
        x₁ = r · sin(θ₀) · cos(θ₁)
        ...
        x_{d-1} = r · sin(θ₀) · ... · sin(θ_{d-2})
    
    实现技巧:
        使用累积乘积 sin(θ₀) · sin(θ₁) · ... 避免重复计算
    
    参数:
        r: 半径（非负）
        angles: 角度数组，形状为 (d-1,)
        
    返回:
        笛卡尔坐标向量，形状为 (d,)
        
    注意:
        这是 cartesian_to_polar 的逆函数，但由于量化误差，
        可能存在微小的重建误差
    """
    d = len(angles) + 1
    x = np.zeros(d)
    
    if r < 1e-10:
        return x
    
    # 累积 sin 乘积
    sin_product = 1.0
    
    for i in range(d - 1):
        # xᵢ = r · sin(θ₀) · ... · sin(θ_{i-1}) · cos(θᵢ)
        x[i] = r * sin_product * np.cos(angles[i])
        sin_product *= np.sin(angles[i])
    
    # 最后一个分量: x_{d-1} = r · sin(θ₀) · ... · sin(θ_{d-2})
    x[-1] = r * sin_product
    
    return x


def compute_lloyd_max_centroids(alpha: float, beta_param: float, 
                                 bits: int, n_iterations: int = 100) -> np.ndarray:
    """
    计算 Beta 分布的 Lloyd-Max 量化质心
    
    Lloyd-Max 算法是一种迭代优化算法，用于找到最优的量化质心，
    使得量化后的均方误差 (MSE) 最小。
    
    算法步骤:
        1. 初始化质心（通常均匀分布）
        2. 计算决策边界（相邻质心的中点）
        3. 更新质心为条件期望: E[X | X ∈ cell]
        4. 重复 2-3 直到收敛
    
    数学公式:
        - 边界: bᵢ = (c_{i-1} + cᵢ) / 2
        - 新质心: cᵢ = E[X | bᵢ < X < b_{i+1}]
                 = ∫_{bᵢ}^{b_{i+1}} x·f(x)dx / ∫_{bᵢ}^{b_{i+1}} f(x)dx
    
    数值积分:
        使用 Simpson 法则计算积分，精度高且计算效率好
    
    参数:
        alpha: Beta 分布的 α 参数
        beta_param: Beta 分布的 β 参数
        bits: 量化比特数，决定量化级别数 (2^bits)
        n_iterations: Lloyd-Max 迭代次数，默认 100
        
    返回:
        质心数组，形状为 (2^bits,)
        
    参考:
        Lloyd-Max 量化是最优标量量化的标准算法，
        广泛应用于信号处理和压缩领域
    """
    n_levels = 2 ** bits
    
    # 初始化质心: 在 [0, 1] 区间均匀分布
    centroids = np.linspace(0, 1, n_levels)
    
    # Beta 分布概率密度函数 (PDF)
    def beta_pdf(x):
        return beta.pdf(x, alpha, beta_param)
    
    # Lloyd-Max 迭代
    for _ in range(n_iterations):
        # 步骤 1: 计算决策边界（相邻质心的中点）
        boundaries = np.zeros(n_levels + 1)
        boundaries[0] = 0.0
        boundaries[-1] = 1.0
        
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2
        
        # 步骤 2: 更新质心为条件期望
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            
            if a >= b:
                new_centroids[i] = centroids[i]
                continue
            
            # 使用 Simpson 法则进行数值积分
            n_points = 100
            x_grid = np.linspace(a, b, n_points)
            dx = (b - a) / (n_points - 1)
            
            pdf_vals = beta_pdf(x_grid)
            x_pdf_vals = x_grid * pdf_vals
            
            # Simpson 法则: ∫f(x)dx ≈ Δx/3 · [f₀ + 4Σf_{奇数} + 2Σf_{偶数} + f_n]
            integral_x_pdf = dx / 3 * (
                x_pdf_vals[0] + 
                4 * np.sum(x_pdf_vals[1:-1:2]) + 
                2 * np.sum(x_pdf_vals[2:-2:2]) + 
                x_pdf_vals[-1]
            )
            
            integral_pdf = dx / 3 * (
                pdf_vals[0] + 
                4 * np.sum(pdf_vals[1:-1:2]) + 
                2 * np.sum(pdf_vals[2:-2:2]) + 
                pdf_vals[-1]
            )
            
            # 条件期望: E[X | X ∈ [a,b]] = ∫x·f(x)dx / ∫f(x)dx
            if integral_pdf > 1e-10:
                new_centroids[i] = integral_x_pdf / integral_pdf
            else:
                new_centroids[i] = (a + b) / 2
        
        centroids = new_centroids
    
    return centroids


def lloyd_max_quantize(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    使用 Lloyd-Max 质心对值进行量化
    
    量化过程: 将每个值映射到最近的质心的索引
    
    参数:
        x: 输入值，范围应在 [0, 1] 内
        centroids: Lloyd-Max 质心数组
        
    返回:
        量化索引数组，类型为 uint32
        
    算法:
        对于每个值，找到与其绝对差最小的质心的索引
    """
    # 裁剪到 [0, 1] 范围，确保数值稳定性
    x_clipped = np.clip(x, 0, 1)
    
    # 找到每个值最近的质心
    # 使用广播: x[:, None] - centroids[None, :] 计算所有差值
    indices = np.argmin(np.abs(x_clipped[:, None] - centroids[None, :]), axis=1)
    
    return indices.astype(np.uint32)


def lloyd_max_dequantize(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    使用质心将量化索引反量化为值
    
    反量化过程: 将索引映射回对应的质心值
    
    参数:
        indices: 量化索引数组
        centroids: Lloyd-Max 质心数组
        
    返回:
        重建的值数组
        
    注意:
        由于量化是有损的，重建值与原始值存在量化误差
    """
    return centroids[indices]


def beta_distribution_params(d: int) -> Tuple[float, float]:
    """
    计算旋转后坐标的 Beta 分布参数
    
    理论依据:
        随机正交旋转后，当向量在单位球面上均匀分布时，
        每个坐标分量在 [-1, 1] 上的分布是 Beta(d/2, d/2) 的缩放版本。
        
        将坐标从 [-1, 1] 映射到 [0, 1]:
            x' = (x + 1) / 2
        则 x' 服从 Beta(d/2, d/2) 分布。
    
    参数:
        d: 向量维度
        
    返回:
        (alpha, beta) 参数元组，其中 alpha = beta = d/2
        
    参考:
        这是高维几何中的标准结果，与球面上均匀分布的性质相关
    """
    alpha = d / 2
    beta_param = d / 2
    return alpha, beta_param
