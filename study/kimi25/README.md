# kimi25 - PolarQuant 无损量化实现

本目录包含 **PolarQuant** 算法的完整 Python 实现，基于 Google Research 提出的极坐标变换量化方法。

## 项目结构

```
kimi25/
├── polarquant/              # PolarQuant 核心实现
│   ├── polarquant/          # 主包
│   │   ├── __init__.py      # 包导出
│   │   ├── core.py          # 核心量化类
│   │   └── utils.py         # 工具函数
│   ├── tests/               # 测试套件
│   ├── examples/            # 使用示例
│   ├── docs/                # 文档和时序图
│   ├── README.md            # 详细说明
│   └── Makefile             # 构建工具
└── README.md                # 本文件
```

## 快速开始

### 进入项目目录

```bash
cd polarquant
```

### 安装

```bash
pip install -e .
```

### 运行示例

```bash
# 基本用法示例
python examples/basic_usage.py

# KV Cache 演示
python examples/kv_cache_demo.py

# Qwen 模型量化
python examples/qwen_quantization_demo.py
```

### 运行测试

```bash
make test
```

## PolarQuant 简介

PolarQuant 是一种基于**极坐标变换**的无损量化算法，特点：

- **高压缩比**: 6-8x 压缩比
- **近无损**: 余弦相似度 > 0.95
- **零元数据**: 无需存储量化参数
- **理论保证**: 基于 Beta 分布的最优量化

### 算法流程

```
输入向量 x ∈ R^d
    ↓
1. 随机旋转: y = Q · x
    ↓
2. 极坐标变换: (r, θ) = cart2pol(y)
    ↓
3. Lloyd-Max 量化
    ↓
压缩后的索引
```

## 核心文件

| 文件 | 说明 |
|------|------|
| `polarquant/core.py` | PolarQuant 主类实现 |
| `polarquant/utils.py` | 工具函数（旋转、极坐标变换、Lloyd-Max） |
| `tests/test_*.py` | 单元测试和集成测试 |
| `examples/*.py` | 使用示例 |
| `docs/sequence_diagram.md` | 时序图文档 |

## 更多信息

详见 `polarquant/README.md`
