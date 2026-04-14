# FAISS Vector Transform - RHT (Randomized Hadamard Transform)

## 概述

FAISS 通过 `index_factory` 支持两种旋转预处理：

| 组件 | 类名 | 复杂度 | 说明 |
|------|------|--------|------|
| `RR` | `RandomRotationMatrix` | O(d²) | 随机旋转矩阵 (QR 分解) |
| `HR` | `HadamardRotation` | **O(d log d)** | 快速 Hadamard 旋转 (FWHT) |

## 使用方式

```python
import faiss

# 方式 1: 随机旋转 (O(d²))
index = faiss.index_factory(d, "L2norm,RR,SQtqmse4")

# 方式 2: Hadamard 旋转 (O(d log d)) - 推荐
index = faiss.index_factory(d, "L2norm,HR,SQtqmse4")
```

## 核心代码

### 1. RandomRotationMatrix (随机旋转矩阵)

**文件**: `VectorTransform.h:115-127`, `VectorTransform.cpp:333-361`

```cpp
struct RandomRotationMatrix : LinearTransform {
    RandomRotationMatrix(int d_in_val, int d_out_val);
    void init(int seed);  // 使用 QR 分解生成正交矩阵
};
```

**实现原理**:
1. 生成随机高斯矩阵 `G ~ N(0, 1)`
2. 对 `G` 进行 QR 分解: `G = QR`
3. `Q` 即为随机正交矩阵

### 2. HadamardRotation (快速 Hadamard 旋转)

**文件**: `VectorTransform.h:129-150`, `VectorTransform.cpp:364-491`

```cpp
struct HadamardRotation : VectorTransform {
    uint32_t seed;
    std::vector<float> signs1, signs2, signs3;  // 三轮随机符号翻转
    
    explicit HadamardRotation(int d, uint32_t seed = 12345);
    void apply_noalloc(idx_t n, const float* x, float* xt) const override;
};
```

**实现原理**:
```
H = D₃ · FWHT · D₂ · FWHT · D₁ · FWHT

其中:
- FWHT: 快速 Walsh-Hadamard 变换 (O(d log d))
- D₁, D₂, D₃: 随机对角矩阵 (元素为 ±1)
```

**核心算法**:
```cpp
// Fast Walsh-Hadamard Transform (O(n log n))
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
│  │  for each vector x:                                         │            │
│  │    x₁ = D₁ · FWHT(x)      // Round 1                        │            │
│  │    x₂ = D₂ · FWHT(x₁)     // Round 2                        │            │
│  │    x₃ = D₃ · FWHT(x₂)     // Round 3                        │            │
│  │    return x₃ / (d * √d)   // Normalize                      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────┐                                │
│  │  SQtqmse4 (TurboQuant MSE 4-bit)        │                                │
│  │                                         │                                │
│  │  // Beta 分布量化 (见 scalar_quantizer/) │                                │
│  │  for each coordinate x[i]:              │                                │
│  │    idx = binary_search(boundaries, x[i])│                                │
│  │    code[i] = centroids[idx]             │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 性能对比

| 方法 | 预处理复杂度 | 内存占用 | 适用场景 |
|------|-------------|---------|---------|
| RandomRotation (RR) | O(d²) | O(d²) | 低维度 (d < 1024) |
| HadamardRotation (HR) | O(d log d) | O(d) | 高维度 (d ≥ 1024) |

## 参考文献

- [FAISS Wiki: The index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
- [TurboQuant Paper](https://arxiv.org/abs/2502.12539)
