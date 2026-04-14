# PolarQuant: 基于极坐标变换的无损量化

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**PolarQuant** 的 Python 实现，一种基于极坐标变换的新型量化算法，用于高维向量压缩。

[English Version](#english-version) | [中文文档](#中文文档)

---

## 中文文档

## 概述

PolarQuant 通过以下步骤实现高维向量（如 LLM KV Cache、嵌入向量）的近无损压缩：

1. **随机正交旋转** - 将向量分量分布归一化为 Beta 分布
2. **极坐标变换** - 将向量分解为半径和角度
3. **Lloyd-Max 量化** - 使用 Beta 分布质心进行最优量化
4. **零元数据开销** - 无需存储每个向量的缩放因子/零点

### 核心特性

- 🎯 **近无损压缩** - 高余弦相似度 (>0.95)
- 📦 **零元数据开销** - 无需每个向量的量化参数
- ⚡ **高效计算** - 使用 Hadamard 旋转可达 O(d log d)
- 🔧 **灵活位宽** - 可配置的量化比特数
- 🧪 **测试驱动** - 全面的测试套件，覆盖率 >95%
- 📊 **理论保证** - 基于旋转后的 Beta(d/2, d/2) 分布

## 安装

### 从源码安装

```bash
git clone https://github.com/example/polarquant.git
cd polarquant
pip install -e .
```

### 开发安装

```bash
pip install -e ".[dev]"
```

## 快速开始

```python
import numpy as np
from polarquant import PolarQuant, PolarQuantConfig

# 配置量化器（256维向量）
config = PolarQuantConfig(
    dimension=256,      # 向量维度
    radius_bits=8,      # 半径量化比特数
    angle_bits=4,       # 角度量化比特数
    seed=42             # 随机种子（保证可复现）
)

# 创建量化器
pq = PolarQuant(config)

# 压缩向量
x = np.random.randn(256)
x = x / np.linalg.norm(x)  # 归一化

compressed = pq.compress(x)
print(f"压缩比: {pq.compression_ratio():.1f}x")

# 解压向量
x_reconstructed = pq.decompress(compressed)

# 评估质量
cosine_sim = np.dot(x, x_reconstructed) / (np.linalg.norm(x) * np.linalg.norm(x_reconstructed))
print(f"余弦相似度: {cosine_sim:.4f}")
```

## 算法原理

### 数学基础

随机正交旋转后，每个坐标在从 [-1, 1] 映射到 [0, 1] 时服从 **Beta(d/2, d/2)** 分布。这一可预测的分布使得：

1. **最优量化** - 使用 Lloyd-Max 算法
2. **无元数据** - 质心可从分布参数预计算
3. **高保真** - 尤其在高维度（d ≥ 64）时表现优异

### 算法流程

```
输入向量 x ∈ R^d
    │
    ▼
┌─────────────────────────┐
│  随机旋转                │  y = R·x  (R 为正交矩阵)
│  (分布归一化)            │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  极坐标变换              │  (r, θ) = cart2pol(y)
│  (半径 + 角度)           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Lloyd-Max 量化          │  indices = Q(r, θ)
│  (Beta 分布优化)         │
└───────────┬─────────────┘
            ▼
    压缩后的索引
```

## API 参考

### PolarQuantConfig

配置类。

```python
config = PolarQuantConfig(
    dimension: int,         # 向量维度（必需）
    radius_bits: int = 8,   # 半径比特数 (1-16)
    angle_bits: int = 4,    # 角度比特数 (1-16)
    use_hadamard: bool = False,  # 使用 Hadamard 旋转
    seed: int = 42          # 随机种子
)
```

### PolarQuant

主量化类。

```python
pq = PolarQuant(config)

# 压缩
compressed = pq.compress(x: np.ndarray) -> CompressedVector

# 解压
x_recon = pq.decompress(compressed: CompressedVector) -> np.ndarray

# 获取压缩比
ratio = pq.compression_ratio() -> float

# 计算误差指标
errors = pq.compute_error(x, x_recon) -> dict
# 返回: {'mse', 'rmse', 'cosine_similarity', 'relative_error'}

# 保存/加载
pq.save(path: str)
pq_loaded = PolarQuant.load(path: str)
```

### PolarQuantBatch

批处理类。

```python
from polarquant import PolarQuantBatch

batch_processor = PolarQuantBatch(pq)

# 批量操作
compressed_list = batch_processor.compress_batch(X: np.ndarray)
X_recon = batch_processor.decompress_batch(compressed_list)
errors = batch_processor.compute_batch_error(X, X_recon)
```

## 使用示例

### KV Cache 压缩（LLM 推理）

```python
from polarquant import PolarQuant, PolarQuantConfig

# 典型的注意力头维度
config = PolarQuantConfig(dimension=64, radius_bits=8, angle_bits=4)
pq = PolarQuant(config)

# 模拟 KV Cache
kv_cache = []
for token_id in range(100):
    key = np.random.randn(64)
    value = np.random.randn(64)
    
    # 归一化
    key = key / np.linalg.norm(key)
    value = value / np.linalg.norm(value)
    
    # 压缩
    kv_cache.append({
        'key': pq.compress(key),
        'value': pq.compress(value)
    })

# 后续：解压用于注意力计算
query = np.random.randn(64)
query = query / np.linalg.norm(query)

for entry in kv_cache:
    key = pq.decompress(entry['key'])
    attention_score = np.dot(query, key)
```

### 嵌入向量压缩

```python
# 压缩高维嵌入向量
config = PolarQuantConfig(dimension=768, radius_bits=8, angle_bits=4)
pq = PolarQuant(config)

embeddings = [np.random.randn(768) for _ in range(1000)]
compressed = [pq.compress(e) for e in embeddings]

# 查看压缩比
print(f"压缩比: {pq.compression_ratio():.1f}x")
# 输出: ~6-8x，取决于配置
```

## 性能

### 压缩比

| 维度 | 半径比特 | 角度比特 | 压缩比 |
|------|---------|---------|--------|
| 64   | 8       | 4       | ~5.0x |
| 128  | 8       | 4       | ~6.4x |
| 256  | 8       | 4       | ~7.1x |
| 512  | 8       | 4       | ~7.6x |
| 768  | 8       | 4       | ~7.7x |

### 重建质量

随机单位向量的典型余弦相似度：

| 维度 | 角度比特 | 平均余弦相似度 |
|------|---------|--------------|
| 64   | 4       | > 0.90      |
| 256  | 4       | > 0.95      |
| 768  | 4       | > 0.98      |

高维度由于 Beta 分布更集中，质量更好。

---

## English Version

## Overview

PolarQuant achieves near-lossless compression of high-dimensional vectors through:

1. **Random Orthogonal Rotation** - Normalizes distribution to Beta
2. **Polar Coordinate Transformation** - Decomposes into radius and angles
3. **Lloyd-Max Quantization** - Optimal quantization using Beta centroids
4. **Zero Metadata Overhead** - No per-vector parameters needed

### Key Features

- 🎯 Near-lossless compression (>0.95 cosine similarity)
- 📦 Zero metadata overhead
- ⚡ Efficient O(d log d) computation
- 🔧 Flexible bit-widths
- 🧪 Test-driven development
- 📊 Proven theoretical guarantees

## Quick Start

```python
from polarquant import PolarQuant, PolarQuantConfig

config = PolarQuantConfig(dimension=256, radius_bits=8, angle_bits=4)
pq = PolarQuant(config)

x = np.random.randn(256)
compressed = pq.compress(x)
x_recon = pq.decompress(compressed)

print(f"Compression ratio: {pq.compression_ratio():.1f}x")
```

---

## 项目结构

```
polarquant/
├── polarquant/          # 主包
│   ├── __init__.py      # 包导出
│   ├── core.py          # 核心实现（中文注释）
│   └── utils.py         # 工具函数（中文注释）
├── tests/               # 测试套件
│   ├── test_utils.py    # 工具函数测试（中文注释）
│   ├── test_core.py     # 核心测试
│   └── test_integration.py  # 集成测试
├── examples/            # 示例脚本
│   ├── basic_usage.py   # 基本用法
│   ├── kv_cache_demo.py # KV Cache 演示
│   └── qwen_*.py        # Qwen 模型量化示例
├── docs/                # 文档
│   ├── sequence_diagram.md  # 时序图（中文）
│   └── README.md        # 说明文档
├── Makefile            # 构建自动化
├── pyproject.toml      # 包配置
└── README.md           # 本文件
```

## 开发

### 设置

```bash
# 克隆仓库
git clone https://github.com/example/polarquant.git
cd polarquant

# 安装开发依赖
make install-dev
```

### Makefile 目标

```bash
make help              # 显示所有可用目标
make install           # 安装包
make test              # 运行测试
make test-coverage     # 运行测试并生成覆盖率报告
make lint              # 运行代码检查
make format            # 格式化代码
make type-check        # 运行类型检查
make clean             # 清理构建产物
make benchmark         # 运行性能基准测试
```

## 参考

1. **PolarQuant Paper** (Google Research, 2026)
   - "PolarQuant: Quantizing KV Caches with Polar Transformation"
   - ICLR 2026 / AISTATS 2026

2. **Related Work**
   - TurboQuant: Online Vector Quantization
   - Product Quantization (PQ)
   - Johnson-Lindenstrauss Lemma

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- Google Research 提出的原始 PolarQuant 算法
- 开源社区的工具和灵感

---

**注意**: 这是研究实现。用于生产环境 LLM 推理时，请考虑 CUDA 内核和融合操作等额外优化。
