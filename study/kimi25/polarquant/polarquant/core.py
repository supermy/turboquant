"""
PolarQuant 核心实现模块

本模块实现了 PolarQuant 算法的主要功能，用于通过极坐标变换
对高维向量进行高效的无损量化压缩。

主要组件:
- PolarQuantConfig: 配置类，用于设置量化参数
- CompressedVector: 压缩向量类，存储量化后的数据
- PolarQuant: 主量化类，实现压缩/解压算法
- PolarQuantBatch: 批处理类，支持批量向量处理

算法流程:
1. 随机正交旋转: 将输入向量通过正交矩阵旋转，使各分量服从 Beta 分布
2. 极坐标变换: 将笛卡尔坐标 (x₁, x₂, ..., xₙ) 转换为 (r, θ₁, θ₂, ..., θₙ₋₁)
3. Lloyd-Max 量化: 基于 Beta 分布计算最优量化质心
4. 索引存储: 只存储量化后的整数索引，实现高压缩比

典型应用场景:
- LLM KV Cache 压缩 (6-8x 压缩比)
- 向量数据库嵌入压缩
- 神经网络权重量化
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pickle

# 从 utils 模块导入工具函数
from .utils import (
    random_orthogonal_matrix,    # 生成随机正交矩阵
    cartesian_to_polar,          # 笛卡尔坐标转极坐标
    polar_to_cartesian,          # 极坐标转笛卡尔坐标
    compute_lloyd_max_centroids, # 计算 Lloyd-Max 质心
    lloyd_max_quantize,          # Lloyd-Max 量化
    lloyd_max_dequantize,        # Lloyd-Max 反量化
    beta_distribution_params,    # Beta 分布参数计算
)


@dataclass
class PolarQuantConfig:
    """
    PolarQuant 量化配置类
    
    用于配置 PolarQuant 量化算法的各项参数
    
    属性说明:
        dimension (int): 向量维度 d，必须 >= 2
        radius_bits (int): 半径量化使用的比特数，范围 1-16，默认 8
        angle_bits (int): 角度量化使用的比特数，范围 1-16，默认 4
        use_hadamard (bool): 是否使用 Hadamard 旋转，默认 False
                            Hadamard 旋转计算更快 (O(d log d)) 但质量略低
        seed (int): 随机种子，用于结果可复现，默认 42
    
    使用示例:
        >>> config = PolarQuantConfig(dimension=256, radius_bits=8, angle_bits=4)
        >>> pq = PolarQuant(config)
    """
    dimension: int      # 向量维度
    radius_bits: int = 8    # 半径量化比特数
    angle_bits: int = 4     # 角度量化比特数
    use_hadamard: bool = False  # 是否使用 Hadamard 旋转
    seed: int = 42      # 随机种子
    
    def __post_init__(self):
        """
        配置验证
        
        确保各参数在有效范围内
        """
        if self.dimension < 2:
            raise ValueError("维度必须至少为 2")
        if self.radius_bits < 1 or self.radius_bits > 16:
            raise ValueError("radius_bits 必须在 1 到 16 之间")
        if self.angle_bits < 1 or self.angle_bits > 16:
            raise ValueError("angle_bits 必须在 1 到 16 之间")


class CompressedVector:
    """
    压缩向量表示类
    
    存储量化后的半径和角度索引，以及可选的原始向量范数
    
    属性:
        radius_idx (int): 半径的量化索引
        angle_indices (np.ndarray): 角度的量化索引数组，形状为 (dimension-1,)
        original_norm (float, optional): 原始向量的 L2 范数，用于可选的后续处理
    
    内存占用估算:
        - radius_idx: 4 bytes (int32)
        - angle_indices: (dimension-1) * 4 bytes (int32 数组)
        - 实际压缩后: radius_bits + (dimension-1) * angle_bits bits
    """
    
    def __init__(self, radius_idx: int, angle_indices: np.ndarray, 
                 original_norm: Optional[float] = None):
        self.radius_idx = radius_idx           # 半径量化索引
        self.angle_indices = angle_indices     # 角度量化索引数组
        self.original_norm = original_norm     # 原始向量范数（可选）
    
    def size_bits(self) -> int:
        """
        返回压缩表示的总比特数（估算值）
        
        注意: 这是简化估算，实际存储可能因编码方式而异
        """
        return 32 + len(self.angle_indices) * 32  # 简化为 int32 估算
    
    def size_bytes(self) -> int:
        """
        返回压缩表示的总字节数（估算值）
        """
        return (self.size_bits() + 7) // 8  # 向上取整到字节


class PolarQuant:
    """
    PolarQuant 主量化类
    
    实现 PolarQuant 算法的核心功能，包括:
    1. 随机正交旋转: 将向量分布归一化为 Beta 分布
    2. 极坐标变换: 笛卡尔坐标 ↔ 极坐标
    3. Lloyd-Max 量化: 基于 Beta 分布的最优标量量化
    4. 零元数据开销: 无需存储每个向量的缩放因子
    
    压缩比计算:
        原始大小: dimension * 32 bits (float32)
        压缩大小: radius_bits + (dimension-1) * angle_bits
        压缩比: (dimension * 32) / (radius_bits + (dimension-1) * angle_bits)
    
    使用示例:
        >>> config = PolarQuantConfig(dimension=256, radius_bits=8, angle_bits=4)
        >>> pq = PolarQuant(config)
        >>> 
        >>> # 压缩向量
        >>> x = np.random.randn(256)
        >>> compressed = pq.compress(x)
        >>> 
        >>> # 解压向量
        >>> x_recon = pq.decompress(compressed)
        >>> 
        >>> # 计算压缩比
        >>> ratio = pq.compression_ratio()
    """
    
    def __init__(self, config: PolarQuantConfig):
        """
        使用配置初始化 PolarQuant 量化器
        
        参数:
            config: PolarQuantConfig 配置实例
        
        初始化过程:
            1. 保存配置和维度信息
            2. 初始化旋转矩阵（随机正交或 Hadamard）
            3. 计算 Beta 分布参数 α = β = d/2
            4. 预计算 Lloyd-Max 角度质心
            5. 设置半径量化级别
        """
        self.config = config
        self.dimension = config.dimension
        
        # 初始化旋转矩阵
        if config.use_hadamard:
            # Hadamard 旋转: 使用快速 Walsh-Hadamard 变换
            # 优点: 计算复杂度 O(d log d)，无需存储矩阵
            # 缺点: 是结构化变换，随机性较弱
            self.rotation_matrix = None
        else:
            # 随机正交旋转: 使用 QR 分解生成随机正交矩阵
            # 优点: 完全随机，理论保证更好
            # 缺点: 需要存储 d×d 矩阵，计算复杂度 O(d²)
            self.rotation_matrix = random_orthogonal_matrix(
                config.dimension, seed=config.seed
            )
        
        # 计算 Beta 分布参数
        # 旋转后的坐标服从 Beta(d/2, d/2) 分布
        self.alpha, self.beta_param = beta_distribution_params(config.dimension)
        
        # 预计算 Lloyd-Max 质心用于角度量化
        # 基于 Beta(α, β) 分布计算最优量化质心
        self.angle_centroids = compute_lloyd_max_centroids(
            self.alpha, self.beta_param, config.angle_bits
        )
        
        # 半径量化级别数: 2^radius_bits
        self.radius_levels = 2 ** config.radius_bits
        
        # 半径量化的动态范围
        # 使用对数尺度量化以更好地处理大动态范围
        self.radius_min = 0.0
        self.radius_max = 10.0
        
    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """
        对输入向量应用旋转变换
        
        参数:
            x: 输入向量，形状为 (dimension,)
            
        返回:
            旋转后的向量
        """
        if self.config.use_hadamard:
            from .utils import hadamard_rotation
            return hadamard_rotation(x)
        else:
            # 矩阵乘法: R @ x
            return self.rotation_matrix @ x
    
    def _inverse_rotate(self, x: np.ndarray) -> np.ndarray:
        """
        应用逆旋转变换
        
        对于正交矩阵，逆矩阵等于转置: R⁻¹ = Rᵀ
        Hadamard 矩阵是自逆的（差一个归一化因子）
        
        参数:
            x: 旋转后的向量
            
        返回:
            原始坐标系中的向量
        """
        if self.config.use_hadamard:
            from .utils import hadamard_rotation
            return hadamard_rotation(x)
        else:
            # 正交矩阵的逆等于转置
            return self.rotation_matrix.T @ x
    
    def _quantize_radius(self, r: float) -> Tuple[int, float]:
        """
        使用对数尺度均匀量化半径
        
        使用对数尺度可以更好地处理大动态范围，
        因为半径可能跨越多个数量级
        
        参数:
            r: 半径值 (非负)
            
        返回:
            (量化索引, 重建半径值) 的元组
        """
        if r < 1e-10:
            # 处理零向量情况
            return 0, 0.0
        
        # 对数变换: log(r)
        log_r = np.log(max(r, 1e-10))
        log_min, log_max = np.log(1e-3), np.log(self.radius_max)
        
        # 归一化到 [0, 1] 区间
        log_r_norm = (log_r - log_min) / (log_max - log_min)
        log_r_norm = np.clip(log_r_norm, 0, 1)
        
        # 均匀量化到离散级别
        idx = int(log_r_norm * (self.radius_levels - 1))
        idx = min(idx, self.radius_levels - 1)
        
        # 反量化: 从索引重建半径值
        log_r_recon = log_min + idx / (self.radius_levels - 1) * (log_max - log_min)
        r_recon = np.exp(log_r_recon)
        
        return idx, r_recon
    
    def _dequantize_radius(self, idx: int) -> float:
        """
        从量化索引反量化半径
        
        参数:
            idx: 半径量化索引
            
        返回:
            重建的半径值
        """
        log_min, log_max = np.log(1e-3), np.log(self.radius_max)
        log_r = log_min + idx / (self.radius_levels - 1) * (log_max - log_min)
        return np.exp(log_r)
    
    def compress(self, x: np.ndarray) -> CompressedVector:
        """
        使用 PolarQuant 算法压缩向量
        
        压缩流程:
            1. 随机旋转: y = R @ x
            2. 极坐标变换: y → (r, θ₁, θ₂, ..., θₙ₋₁)
            3. 半径量化: r → idx_r (对数尺度)
            4. 角度量化: θᵢ → idx_θᵢ (Lloyd-Max)
            5. 返回压缩表示
        
        参数:
            x: 输入向量，形状为 (dimension,)
            
        返回:
            CompressedVector 压缩向量实例
            
        异常:
            ValueError: 如果输入维度与配置不匹配
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.dimension:
            raise ValueError(f"期望维度 {self.dimension}, 实际得到 {len(x)}")
        
        # 保存原始范数，用于可选的后续处理
        original_norm = np.linalg.norm(x)
        
        # 步骤 1: 随机旋转
        # 目的: 将向量分布归一化为 Beta 分布
        x_rotated = self._rotate(x)
        
        # 步骤 2: 转换为极坐标
        # 结果: r (半径), angles (d-1 个角度)
        r, angles = cartesian_to_polar(x_rotated)
        
        # 步骤 3: 量化半径 (对数尺度)
        radius_idx, _ = self._quantize_radius(r)
        
        # 步骤 4: 量化角度 (Lloyd-Max)
        # 将角度从 [0, π] 映射到 [0, 1] 用于 Beta 分布
        angles_normalized = angles / np.pi
        angles_normalized = np.clip(angles_normalized, 0, 1)
        
        # 使用预计算的 Lloyd-Max 质心进行量化
        angle_indices = lloyd_max_quantize(angles_normalized, self.angle_centroids)
        
        return CompressedVector(radius_idx, angle_indices, original_norm)
    
    def decompress(self, compressed: CompressedVector) -> np.ndarray:
        """
        解压向量
        
        解压流程:
            1. 反量化半径: idx_r → r
            2. 反量化角度: idx_θᵢ → θᵢ
            3. 极坐标转笛卡尔: (r, θ) → y
            4. 逆旋转: x = Rᵀ @ y
            5. 返回重建向量
        
        参数:
            compressed: CompressedVector 压缩向量实例
            
        返回:
            重建的向量，形状为 (dimension,)
        """
        # 步骤 1: 反量化半径
        r = self._dequantize_radius(compressed.radius_idx)
        
        # 步骤 2: 反量化角度
        angles = lloyd_max_dequantize(compressed.angle_indices, self.angle_centroids)
        angles = angles * np.pi  # 从 [0, 1] 反归一化到 [0, π]
        
        # 步骤 3: 极坐标转笛卡尔坐标
        x_rotated = polar_to_cartesian(r, angles)
        
        # 步骤 4: 逆旋转
        x_reconstructed = self._inverse_rotate(x_rotated)
        
        return x_reconstructed
    
    def compression_ratio(self) -> float:
        """
        计算压缩比
        
        压缩比 = 原始大小 / 压缩后大小
        
        原始大小: dimension × 32 bits (float32)
        压缩大小: radius_bits + (dimension-1) × angle_bits
        
        返回:
            压缩比，值越大表示压缩效果越好
            
        示例:
            对于 dimension=256, radius_bits=8, angle_bits=4:
            原始大小 = 256 × 32 = 8192 bits
            压缩大小 = 8 + 255 × 4 = 1028 bits
            压缩比 = 8192 / 1028 ≈ 7.97x
        """
        # 原始大小: float32 = 32 bits per element
        original_bits = self.dimension * 32
        
        # 压缩后大小
        compressed_bits = self.config.radius_bits + (self.dimension - 1) * self.config.angle_bits
        
        return original_bits / compressed_bits
    
    def compute_error(self, x: np.ndarray, x_reconstructed: np.ndarray) -> dict:
        """
        计算重建误差指标
        
        评估压缩质量的多项指标:
            - MSE: 均方误差
            - RMSE: 均方根误差
            - Cosine similarity: 余弦相似度 (方向保留)
            - Relative error: 相对误差
        
        参数:
            x: 原始向量
            x_reconstructed: 重建向量
            
        返回:
            包含各项误差指标的字典
        """
        # 均方误差 (Mean Squared Error)
        mse = np.mean((x - x_reconstructed) ** 2)
        rmse = np.sqrt(mse)
        
        # 余弦相似度 (Cosine Similarity)
        # 衡量方向保留程度，范围 [-1, 1]，越接近 1 越好
        norm_x = np.linalg.norm(x)
        norm_recon = np.linalg.norm(x_reconstructed)
        if norm_x > 1e-10 and norm_recon > 1e-10:
            cosine_sim = np.dot(x, x_reconstructed) / (norm_x * norm_recon)
        else:
            cosine_sim = 0.0
        
        # 相对误差 (Relative Error)
        if norm_x > 1e-10:
            relative_error = np.linalg.norm(x - x_reconstructed) / norm_x
        else:
            relative_error = 0.0
        
        return {
            'mse': mse,                          # 均方误差
            'rmse': rmse,                        # 均方根误差
            'cosine_similarity': cosine_sim,     # 余弦相似度
            'relative_error': relative_error,    # 相对误差
        }
    
    def save(self, path: str):
        """
        将量化器状态保存到文件
        
        保存内容包括:
            - 配置参数
            - 旋转矩阵
            - 角度质心
            - 半径范围
        
        参数:
            path: 保存文件路径
        """
        state = {
            'config': self.config,
            'rotation_matrix': self.rotation_matrix,
            'angle_centroids': self.angle_centroids,
            'radius_max': self.radius_max,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'PolarQuant':
        """
        从文件加载量化器
        
        参数:
            path: 保存文件路径
            
        返回:
            加载的 PolarQuant 实例
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # 创建新实例
        pq = cls(state['config'])
        pq.rotation_matrix = state['rotation_matrix']
        pq.angle_centroids = state['angle_centroids']
        pq.radius_max = state['radius_max']
        return pq


class PolarQuantBatch:
    """
    PolarQuant 批处理类
    
    用于高效地批量压缩/解压多个向量
    
    使用示例:
        >>> pq = PolarQuant(config)
        >>> batch_processor = PolarQuantBatch(pq)
        >>> 
        >>> # 批量压缩
        >>> X = np.random.randn(100, 256)  # 100 个向量
        >>> compressed_list = batch_processor.compress_batch(X)
        >>> 
        >>> # 批量解压
        >>> X_recon = batch_processor.decompress_batch(compressed_list)
    """
    
    def __init__(self, quantizer: PolarQuant):
        """
        初始化批处理器
        
        参数:
            quantizer: PolarQuant 量化器实例
        """
        self.quantizer = quantizer
    
    def compress_batch(self, X: np.ndarray) -> List[CompressedVector]:
        """
        批量压缩向量
        
        参数:
            X: 输入数组，形状为 (n_vectors, dimension)
            
        返回:
            CompressedVector 实例列表，长度为 n_vectors
        """
        return [self.quantizer.compress(x) for x in X]
    
    def decompress_batch(self, compressed_list: List[CompressedVector]) -> np.ndarray:
        """
        批量解压向量
        
        参数:
            compressed_list: CompressedVector 实例列表
            
        返回:
            重建的向量数组，形状为 (n_vectors, dimension)
        """
        return np.array([self.quantizer.decompress(c) for c in compressed_list])
    
    def compute_batch_error(self, X: np.ndarray, X_reconstructed: np.ndarray) -> dict:
        """
        计算批量误差指标
        
        参数:
            X: 原始向量数组，形状为 (n_vectors, dimension)
            X_reconstructed: 重建向量数组，形状为 (n_vectors, dimension)
            
        返回:
            包含平均误差指标的字典:
                - mean_mse: 平均均方误差
                - mean_rmse: 平均均方根误差
                - mean_cosine: 平均余弦相似度
                - min_cosine: 最小余弦相似度
        """
        mse_list = []
        cosine_list = []
        
        # 逐个向量计算误差
        for x, x_recon in zip(X, X_reconstructed):
            errors = self.quantizer.compute_error(x, x_recon)
            mse_list.append(errors['mse'])
            cosine_list.append(errors['cosine_similarity'])
        
        return {
            'mean_mse': np.mean(mse_list),           # 平均 MSE
            'mean_rmse': np.mean(np.sqrt(mse_list)), # 平均 RMSE
            'mean_cosine': np.mean(cosine_list),     # 平均余弦相似度
            'min_cosine': np.min(cosine_list),       # 最小余弦相似度
        }
