"""
PolarQuant: 基于极坐标变换的无损量化算法

这是 Google Research 提出的 PolarQuant 算法的 Python 实现，
用于高维向量（如 LLM 的 KV Cache）的高效压缩。

核心特性:
- 随机正交旋转: 将向量分布归一化为 Beta 分布
- 极坐标变换: 将笛卡尔坐标转换为 (半径, 角度)
- Lloyd-Max 量化: 针对 Beta 分布优化的标量量化
- 零元数据开销: 无需存储每个向量的缩放因子
- 近无损压缩: 高余弦相似度 (>0.95)

算法流程:
1. 随机旋转: 使用正交矩阵旋转输入向量
2. 极坐标转换: y = (r, θ₁, θ₂, ..., θ_{d-1})
3. Lloyd-Max 量化: 基于 Beta(d/2, d/2) 分布计算最优质心
4. 存储索引: 只存储量化后的整数索引

典型应用场景:
- LLM KV Cache 压缩 (6-8x 压缩比)
- 向量数据库嵌入压缩
- 神经网络权重量化

参考论文:
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- PolarQuant: Quantizing KV Caches with Polar Transformation (Google Research)

作者: PolarQuant Team
版本: 0.1.0
"""

# 从 core 模块导入核心类
from .core import PolarQuant, PolarQuantConfig, PolarQuantBatch

# 从 utils 模块导入工具函数
from .utils import (
    random_orthogonal_matrix,  # 生成随机正交矩阵
    cartesian_to_polar,         # 笛卡尔坐标转极坐标
    polar_to_cartesian,         # 极坐标转笛卡尔坐标
    lloyd_max_quantize,         # Lloyd-Max 量化
    lloyd_max_dequantize,       # Lloyd-Max 反量化
)

# 版本号
__version__ = "0.1.0"

# 公开 API 列表
__all__ = [
    # 核心类
    "PolarQuant",        # 主量化类
    "PolarQuantConfig",  # 配置类
    "PolarQuantBatch",   # 批处理类
    
    # 工具函数
    "random_orthogonal_matrix",  # 随机正交矩阵生成
    "cartesian_to_polar",        # 笛卡尔转极坐标
    "polar_to_cartesian",        # 极坐标转笛卡尔
    "lloyd_max_quantize",        # Lloyd-Max 量化
    "lloyd_max_dequantize",      # Lloyd-Max 反量化
]
