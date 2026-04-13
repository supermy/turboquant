# TurboQuant - 高维向量压缩工具集

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **PolarQuant** 和 **TurboQuant** 算法的高维向量压缩工具集，专为 LLM KV Cache 和嵌入向量优化。

## 📖 目录

- [项目简介](#项目简介)
- [子项目概览](#子项目概览)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [性能对比](#性能对比)
- [应用场景](#应用场景)
- [参考文献](#参考文献)

## 项目简介

TurboQuant 实现了 Google Research 的 **PolarQuant** 和 **TurboQuant** 算法，通过极坐标变换和 Lloyd-Max 量化实现高维向量的高效压缩。特别适合：

- **LLM KV Cache 压缩** - 减少 50-80% 内存占用
- **嵌入向量压缩** - 保持语义相似度
- **向量检索优化** - 降低存储和带宽需求

### 核心算法流程

```
输入向量 x ∈ R^d
    │
    ▼
┌─────────────────────────┐
│  随机旋转               │  y = Q·x  (Q 为正交矩阵)
│  (分布归一化)           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  极坐标变换             │  (r, θ) = cart2pol(y)
│  (半径 + 角度)          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Lloyd-Max 量化         │  indices = Q(r, θ)
│  (Beta 分布优化)        │
└───────────┬─────────────┘
            ▼
    压缩后的索引
```

## 子项目概览

本项目包含五个独立的实现，分别针对不同的使用场景优化：

| 子项目 | 语言 | 特点 | 适用场景 |
|--------|------|------|----------|
| **根目录** | Rust | TurboQuant + RaBitQ 向量检索 | 向量搜索、ANN 索引 |
| **[google/](google/)** | Python | Google 官方实现，HuggingFace 集成 | 研究、实验、快速原型 |
| **[llamacpp/](llamacpp/)** | C++/CUDA | llama.cpp 集成，GPU 加速 | 生产环境、边缘设备 |
| **[faiss/](faiss/)** | C++/Python | FAISS 集成，SIMD 优化 | 向量检索、大规模搜索 |
| **[glm51/](glm51/)** | Rust | PolarQuant KV Cache 压缩 | 高性能计算、系统编程 |

### 根目录 Rust 实现 (TurboQuant + RaBitQ)

**特点**：
- ✅ TurboQuant Flat + SQ8 (98% 召回率)
- ✅ RaBitQ IVF + SQ8 (98% 召回率)
- ✅ 无需训练的 TurboQuant
- ✅ 极致压缩的 RaBitQ (1-bit)

**召回率测试结果**：

| 方法 | 存储 (bytes) | Recall@10 | 训练 |
|------|-------------|-----------|------|
| TurboQuant 4-bit | 64 | 86.3% | 否 |
| TurboQuant 6-bit | 96 | 93.4% | 否 |
| TurboQuant 4-bit + SQ8 | 192 | **98.5%** | SQ8 |
| RaBitQ 1-bit | 24 | 47.1% | 是 |
| RaBitQ 1-bit + SQ8 | 152 | **98.2%** | 是 |
| RaBitQ IVF + SQ8 | 152 | **98.7%** | 是 |

**快速开始**：
```bash
cargo build
cargo test -- --nocapture
cargo run
```

**详细文档**: [src/lib.rs](src/lib.rs)

### Google 官方实现 (`google/`)

**特点**：
- ✅ Google Research 官方代码
- ✅ HuggingFace Transformers 集成
- ✅ QJL 偏差校正
- ✅ 完整的测试和基准

**快速开始**：
```bash
cd google
make install
make demo
```

**详细文档**: [google/README.md](google/README.md)

### llama.cpp 实现 (`llamacpp/`)

**特点**：
- ✅ 与 llama.cpp Q4_0 格式兼容
- ✅ CUDA 内核支持
- ✅ CPU/CPU 混合执行
- ✅ 生产环境优化

**快速开始**：
```bash
cd llamacpp
make build
make test
```

**详细文档**: [llamacpp/README.md](llamacpp/README.md)

### FAISS 实现 (`faiss/`)

**特点**：
- ✅ 1-8 bit 量化支持
- ✅ AVX2/AVX-512/NEON 优化
- ✅ 无需训练数据
- ✅ 向量检索优化

**快速开始**：
```bash
cd faiss
make install
make demo
```

**详细文档**: [faiss/README.md](faiss/README.md)

### Rust 实现 (`glm51/`)

**特点**：
- ✅ 比 Python 快 10-50 倍
- ✅ 内存安全，无 GC
- ✅ 零成本抽象
- ✅ SIMD 自动向量化

**快速开始**：
```bash
cd glm51
cargo run --example kv_cache_demo
```

**详细文档**: [glm51/README.md](glm51/README.md)

## 核心特性

### 1. 高压缩比

| 维度 | 位数 | 压缩比 | 余弦相似度 |
|------|------|--------|------------|
| 64   | 4    | ~5.0x  | > 0.85     |
| 128  | 4    | ~6.4x  | > 0.95     |
| 256  | 4    | ~7.1x  | > 0.95     |
| 512  | 4    | ~7.6x  | > 0.98     |
| 768  | 4    | ~7.7x  | > 0.98     |

### 2. 语义保持

**量化后语义没有丢失，只是精度降低了**。这就像图像压缩：
- 原图：高清 1920x1080
- 压缩后：720p
- 内容（语义）还在，只是细节（精度）减少了

**实验数据验证**：

| 指标 | 结果 | 说明 |
|------|------|------|
| Pearson 相关性 | 0.9428 | 注意力分数高度相关 |
| Top-5 准确率 | 80% | 排名基本保持 |
| 余弦相似度 | > 0.95 | 高维时更好 |
| 平均绝对误差 | 0.028 | 误差很小 |

### 3. 零元数据开销

- 无需存储每个向量的缩放因子
- 无需存储零点偏移
- 质心可从 Beta 分布预计算

### 4. 数据无关

- 不需要训练数据
- 不需要校准集
- 随机种子确定旋转矩阵

## 快速开始

### 使用主 Makefile

```bash
# 显示帮助
make help

# 构建所有子项目
make all

# 运行所有测试
make test

# 运行基准测试
make benchmark

# 更新所有子项目代码
make update

# 清理所有项目
make clean
```

### 单独使用子项目

```bash
# Google 官方实现
cd google && make demo

# llama.cpp 实现
cd llamacpp && make test

# FAISS 实现
cd faiss && make demo

# Rust 实现
cd glm51 && cargo run --example kv_cache_demo
```

## 性能对比

### 各实现性能对比

| 实现 | 语言 | 压缩速度 | 解压速度 | 内存占用 |
|------|------|----------|----------|----------|
| Google | Python | ~2ms | ~1.5ms | 高 |
| llama.cpp | C++ | ~0.2ms | ~0.15ms | 低 |
| FAISS | C++ | ~0.1ms | ~0.08ms | 低 |
| **Rust** | Rust | **~0.1ms** | **~0.08ms** | **最低** |

### KV Cache 压缩效果

#### 基础测试 (100 tokens, 64维)

```
内存使用:
  原始大小: 100.00 KB
  压缩后大小: 50.00 KB
  节省空间: 50.0%

注意力分数质量:
  分数相关性 (Pearson): 0.9428
  Top-5 检索准确率: 80.0%
```

#### 长序列测试 (10K tokens, 64维)

```
内存使用:
  原始: 9.77 MB
  压缩后: 4.88 MB
  节省空间: 50.0%
```

### GSM8K 准确率

| 方法 | Bits | Accuracy |
|------|------|----------|
| FP16 | 16   | 56.2%    |
| Kivi | 4    | 55.8%    |
| **PolarQuant** | 4 | **56.0%** |
| **PolarQuant + QJL** | 4 | **56.1%** |

## 应用场景

### 1. LLM KV Cache 压缩

```
优势:
- 减少 50-80% 内存占用
- 支持更长上下文
- 降低 GPU 显存压力

适用:
- 长文本生成
- 多轮对话
- 文档问答
```

### 2. 嵌入向量压缩

```
优势:
- 保持语义相似度
- 降低存储成本
- 加速向量检索

适用:
- 语义搜索
- 推荐系统
- RAG 应用
```

### 3. 向量数据库优化

```
优势:
- 减少索引大小
- 降低内存带宽需求
- 加速相似度计算

适用:
- 大规模向量检索
- 实时推荐
- 相似图片搜索
```

## 项目结构

```
turboquant/
├── src/                      # 根目录 Rust 实现 (TurboQuant + RaBitQ)
│   ├── lib.rs                # 库入口
│   ├── utils.rs              # 工具函数
│   ├── hadamard.rs           # Hadamard 旋转
│   ├── lloyd_max.rs          # Lloyd-Max 量化
│   ├── sq8.rs                # SQ8 标量量化
│   ├── turboquant.rs         # TurboQuant Flat + SQ8
│   ├── rabitq.rs             # RaBitQ Flat + SQ8
│   ├── kmeans.rs             # KMeans 聚类
│   └── ivf.rs                # RaBitQ IVF + SQ8
│
├── benches/                  # 性能基准测试
│   └── turboquant_bench.rs
│
├── google/                   # Google 官方实现
│   ├── models/
│   │   ├── kernel4group.py
│   │   ├── modeling_llama_polar.py
│   │   └── modeling_llama_qjl.py
│   ├── utils/
│   │   └── metrics.py
│   └── Makefile
│
├── llamacpp/                 # llama.cpp 实现
│   ├── tq/
│   │   ├── tq_quants.h
│   │   ├── tq_quants.c
│   │   └── test_tq.c
│   ├── cuda/
│   │   ├── tq4-rotate.cu
│   │   └── tq4-set-rows.cu
│   └── Makefile
│
├── faiss/                    # FAISS 实现
│   ├── standalone_test/
│   │   ├── rabitq_faiss.h    # RaBitQ FAISS 复刻
│   │   ├── quantization.h    # TurboQuant 实现
│   │   └── test_cost_benefit.cpp
│   ├── scalar_quantizer/
│   │   ├── quantizers.h
│   │   ├── training.h
│   │   └── training.cpp
│   └── Makefile
│
├── glm51/                    # Rust PolarQuant 实现
│   ├── src/
│   │   ├── lib.rs
│   │   ├── config.rs
│   │   ├── quantizer.rs
│   │   └── batch.rs
│   ├── tests/
│   │   └── integration.rs
│   ├── examples/
│   │   └── kv_cache_demo.rs
│   └── Cargo.toml
│
├── Cargo.toml                # Rust 项目配置
├── Makefile                  # 主 Makefile
└── README.md                 # 本文件
```

## 常见问题

### Q: 压缩的向量参与注意力计算需要反量化吗？

**A: 是的，需要反量化。**

原因：
1. 注意力计算需要点积运算
2. 压缩存储的是索引，无法直接参与数学运算
3. 但反量化速度快，且语义保持良好

### Q: 量化后语义会丢失吗？

**A: 语义没有丢失，只是精度降低了。**

量化保持：
- ✅ 向量间的相对关系
- ✅ 余弦相似度 (> 0.95)
- ✅ Top-K 排序 (80% 准确率)
- ❌ 精确数值 (有损失)

### Q: 为什么高维效果更好？

**A: Beta 分布集中性。**

- 高维时，Beta(d/2, d/2) 分布集中在 0.5 附近
- 分布越集中，量化越精确
- 这就是为什么 LLM 的注意力头（64-128维）效果很好

### Q: 如何选择实现？

**A: 根据使用场景选择。**

| 场景 | 推荐实现 | 原因 |
|------|----------|------|
| 研究、实验 | Google | 官方实现，易于修改 |
| 生产环境 | llama.cpp | GPU 加速，生产优化 |
| 向量检索 | FAISS | SIMD 优化，大规模搜索 |
| 高性能计算 | Rust | 最快速度，内存安全 |

## 参考文献

1. **PolarQuant 论文**
   - "PolarQuant: Quantizing KV Caches with Polar Transformation"
   - Han et al., AISTATS 2026
   - arXiv:2502.02617

2. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

3. **QJL 论文**
   - "QJL: 1-Bit Quantized Johnson-Lindenstrauss"
   - Zandieh et al., ICLR 2026
   - arXiv:2406.03482

4. **官方代码库**
   - Google: https://github.com/ericshwu/PolarQuant
   - llama.cpp: https://github.com/ggml-org/llama.cpp/pull/20995
   - FAISS: https://github.com/facebookresearch/faiss/pull/5049

## 许可证

本项目采用 MIT 许可证。

## 致谢

- Google Research 提供的原始 PolarQuant 算法
- llama.cpp 和 FAISS 社区的优秀实现
- Rust 和 Python 社区的数值计算库
