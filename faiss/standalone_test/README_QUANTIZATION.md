# 量化索引本地实现

## 概述

本目录包含从 FAISS 抽取并本地实现的量化索引，包括：

1. **TurboQuant Flat** - Beta 分布量化
2. **RaBitQ Flat** - 随机二进制量化
3. **RaBitQ IVF** - 倒排索引版本

## 文件结构

```
standalone_test/
├── quantization.h           # 核心实现头文件
├── test_quantization.cpp    # 测试程序
├── ivf_turboquant.h         # IVF + TurboQuant 实现
├── test_ivf_turboquant.cpp  # IVF + TurboQuant 测试
└── README.md                # 本文件

rabitq/                      # FAISS RaBitQ 源码参考
├── IndexRaBitQ.h
├── IndexRaBitQ.cpp
├── RaBitQuantizer.h
├── RaBitQuantizer.cpp
└── RaBitQUtils.h
```

## 编译与运行

```bash
# 编译测试程序
cd standalone_test
g++ -std=c++17 -O3 -o test_quantization test_quantization.cpp -lm

# 运行测试
./test_quantization
```

## 测试结果

### 1. TurboQuant Flat (4-bit)

| 指标 | 值 |
|------|-----|
| 维度 | 128 |
| 数据量 | 5000 |
| 码大小 | 64 bytes |
| Recall@10 | 0.38 |

### 2. RaBitQ Flat (1-bit)

| 指标 | 值 |
|------|-----|
| 维度 | 128 |
| 数据量 | 5000 |
| 码大小 | 24 bytes |
| Recall@10 | 0.27 |

### 3. RaBitQ IVF (1-bit)

| 指标 | 值 |
|------|-----|
| 维度 | 128 |
| 数据量 | 5000 |
| 聚类数 | 100 |
| 码大小 | 24 bytes |
| nprobe=20 Recall@10 | 0.23 |

## 不同位宽对比

### TurboQuant Flat

| 位宽 | 码大小 | Recall@10 |
|------|--------|-----------|
| 2-bit | 32 bytes | 0.30 |
| 4-bit | 64 bytes | 0.40 |
| 8-bit | 128 bytes | 0.40 |

### RaBitQ Flat

| 位宽 | 码大小 | Recall@10 |
|------|--------|-----------|
| 1-bit | 24 bytes | 0.29 |
| 2-bit | 48 bytes | 0.29 |
| 4-bit | 80 bytes | 0.29 |

## 核心算法

### TurboQuant

1. **L2 归一化**: 将向量归一化到单位球面
2. **Hadamard 旋转**: 3轮 FWHT + 随机符号翻转
3. **Beta 分布量化**: 单位向量坐标服从 Beta 分布
4. **Lloyd-Max 迭代**: 构建最优量化码本

### RaBitQ

1. **随机旋转**: Hadamard 变换
2. **中心化**: 减去质心
3. **1-bit 量化**: 符号位量化
4. **因子存储**: `or_minus_c_l2sqr`, `dp_multiplier`
5. **距离估计**: 使用近似公式计算距离

## 使用示例

### TurboQuant Flat

```cpp
#include "quantization.h"

using namespace quant;

// 创建索引
TurboQuantFlatIndex index(128, 4);  // 128维, 4-bit

// 训练 (无需数据)
index.train(nb, xb.data());

// 添加数据
index.add(nb, xb.data());

// 搜索
std::vector<std::vector<size_t>> result_ids;
std::vector<std::vector<float>> result_dists;
index.search(nq, xq.data(), k, result_ids, result_dists);
```

### RaBitQ Flat

```cpp
// 创建索引
RaBitQFlatIndex index(128, 1);  // 128维, 1-bit

// 训练 (计算质心)
index.train(nb, xb.data());

// 添加数据
index.add(nb, xb.data());

// 搜索
index.search(nq, xq.data(), k, result_ids, result_dists);
```

### RaBitQ IVF

```cpp
// 创建索引
RaBitQIVFIndex index(128, 100, 1);  // 128维, 100聚类, 1-bit

// 训练
index.train(nb, xb.data());

// 添加数据
index.add(nb, xb.data());

// 搜索
index.search(nq, xq.data(), k, result_ids, result_dists, 20);  // nprobe=20
```

## 参考文献

1. **TurboQuant**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026

2. **RaBitQ**
   - "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search"
   - Gao & Long, SIGMOD 2025
   - https://arxiv.org/abs/2405.12497

3. **FAISS**
   - https://github.com/facebookresearch/faiss
