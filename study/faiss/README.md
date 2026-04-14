# FAISS TurboQuant 实现

本目录包含从 FAISS PR #5049 抽取的 TurboQuant 相关代码。

## 文件结构

```
faiss/
├── scalar_quantizer/           # TurboQuant 量化器
│   ├── quantizers.h            # 量化器定义
│   ├── training.h              # 训练接口
│   ├── training.cpp            # 训练实现（核心算法）
│   ├── sq-avx2.cpp             # AVX2 SIMD 优化
│   ├── sq-avx512.cpp           # AVX-512 SIMD 优化
│   ├── sq-neon.cpp             # ARM NEON 优化
│   └── test_scalar_quantizer.cpp  # 测试代码
│
├── vector_transform/           # 向量变换 (RHT)
│   ├── VectorTransform.h       # 变换接口定义
│   ├── VectorTransform.cpp     # 变换实现
│   └── README.md               # RHT 详细说明
│
└── Makefile                    # 构建脚本
```

## TurboQuant 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAISS TurboQuant 完整流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  index_factory(d, "L2norm,HR,SQtqmse4")                                     │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────┐                                                            │
│  │  L2norm     │ ◀── NormalizationTransform: x = x / ||x||                  │
│  └─────────────┘                                                            │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  HR (HadamardRotation)                                      │            │
│  │                                                             │            │
│  │  // 三轮随机符号翻转 + FWHT                                  │            │
│  │  for each vector x:                                         │            │
│  │    x₁ = D₁ · FWHT(x)      // Round 1                        │            │
│  │    x₂ = D₂ · FWHT(x₁)     // Round 2                        │            │
│  │    x₃ = D₃ · FWHT(x₂)     // Round 3                        │            │
│  │    return x₃ / (d * √d)   // Normalize                      │            │
│  │                                                             │            │
│  │  复杂度: O(d log d)                                         │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────┐                                │
│  │  SQtqmse4 (TurboQuant MSE 4-bit)        │                                │
│  │                                         │                                │
│  │  // Beta 分布量化                        │                                │
│  │  for each coordinate x[i]:              │                                │
│  │    idx = binary_search(boundaries, x[i])│                                │
│  │    code[i] = centroids[idx]             │                                │
│  │                                         │                                │
│  │  无需训练数据，解析计算代码本             │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 向量变换 (vector_transform/)

| 组件 | 类名 | 复杂度 | 说明 |
|------|------|--------|------|
| `RR` | `RandomRotationMatrix` | O(d²) | 随机旋转矩阵 (QR 分解) |
| `HR` | `HadamardRotation` | **O(d log d)** | 快速 Hadamard 旋转 (FWHT) |
| `L2norm` | `NormalizationTransform` | O(d) | L2 归一化 |

### 2. 量化器 (scalar_quantizer/)

| 类型 | 位数 | 代码大小 (d=128) | 说明 |
|------|------|------------------|------|
| `QT_1bit_tqmse` | 1 | 16 bytes | 二值量化 |
| `QT_2bit_tqmse` | 2 | 32 bytes | 2-bit 量化 |
| `QT_3bit_tqmse` | 3 | 48 bytes | 3-bit 量化 |
| `QT_4bit_tqmse` | 4 | 64 bytes | 4-bit 量化 |
| `QT_8bit_tqmse` | 8 | 128 bytes | 8-bit 量化 |

## 使用方式

### Python 接口

```python
import faiss
import numpy as np

# 方式 1: 仅 Beta 分布量化 (无旋转)
index1 = faiss.index_factory(128, 'SQtqmse4')

# 方式 2: L2归一化 + 随机旋转 + Beta分布量化
index2 = faiss.index_factory(128, 'L2norm,RR,SQtqmse4')

# 方式 3: L2归一化 + Hadamard旋转 + Beta分布量化 (推荐)
index3 = faiss.index_factory(128, 'L2norm,HR,SQtqmse4')

# 训练（无需数据，仅确定维度）
xb = np.random.randn(1000, 128).astype('float32')
faiss.normalize_L2(xb)
index3.train(xb)
index3.add(xb)

# 搜索
xq = np.random.randn(10, 128).astype('float32')
faiss.normalize_L2(xq)
D, I = index3.search(xq, 10)
```

### C++ 接口

```cpp
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/VectorTransform.h>

// 创建 Hadamard 旋转
faiss::HadamardRotation hr(128);

// 创建 TurboQuant 量化器
faiss::ScalarQuantizer sq(128, faiss::ScalarQuantizer::QT_4bit_tqmse);

// 训练（无需数据）
sq.train(0, nullptr);

// 编码
std::vector<float> rotated(128);
hr.apply(1, x, rotated.data());

std::vector<uint8_t> codes(sq.code_size);
sq.compute_codes(rotated.data(), codes.data(), 1);
```

## 核心算法

### Fast Walsh-Hadamard Transform (FWHT)

```cpp
// O(n log n) 复杂度
static void fwht_inplace(float* buf, size_t n) {
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < n; i += step * 2) {
            for (size_t j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j] = a + b;
                buf[j + step] = a - b;
            }
        }
    }
}
```

### Beta 分布量化

```cpp
// 单位向量坐标的边缘分布
// p(x) ∝ (1 - x²)^((d-3)/2),  x ∈ [-1, 1]

void build_TurboQuantMSECodebook(
    size_t d, size_t nbits,
    std::vector<float>& centroids,
    std::vector<float>& boundaries
) {
    // 1. 离散化 Beta 分布
    // 2. Lloyd-Max 迭代优化质心
    // 3. 计算边界
}
```

## 性能数据

### 基准测试结果 (macOS CPU)

```
===== TurboQuant
training time: 0.002 s
encode time: 0.080 s
reconstruction error: 0.009
recall@1: 0.7189
code_size: 50 B/vector
```

### 与 PQ/RQ 对比

| 方法 | 训练时间 | 编码时间 | 重建误差 | Recall@1 |
|------|----------|----------|----------|----------|
| PQ | 0.594s | 0.317s | 0.010 | 0.7036 |
| RQ (beam=32) | 105.3s | 105.3s | 0.014 | 0.7067 |
| **TurboQuant** | **0.002s** | **0.080s** | **0.009** | **0.7189** |

## 与其他实现的对比

| 特性 | FAISS | llama.cpp | Google |
|------|-------|-----------|--------|
| **量化方式** | Beta 分布 | Beta 分布 | Polar + Lloyd-Max |
| **位宽支持** | 1-8 bit | 4 bit | 3-4 bit |
| **旋转方式** | HR (O(d log d)) | 随机正交 (O(d²)) | Hadamard |
| **SIMD 支持** | AVX2/AVX-512/NEON | CUDA | CUDA |
| **训练需求** | 无 | 无 | 无 |

## 快速开始

```bash
# 安装 FAISS
pip install faiss-cpu

# 运行演示
make demo

# 下载最新代码
make download
```

## 参考文献

1. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

2. **FAISS PR #5049**
   - https://github.com/facebookresearch/faiss/pull/5049

3. **Fast Walsh-Hadamard Transform**
   - https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

## 许可证

代码来自 FAISS 项目，遵循 MIT 许可证。
