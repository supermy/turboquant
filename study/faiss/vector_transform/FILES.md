# FAISS Vector Transform 文件列表

## 下载的文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `VectorTransform.h` | 10KB | 向量变换头文件 |
| `VectorTransform.cpp` | 43KB | 向量变换实现 |

## 包含的类

### 1. RandomRotationMatrix (随机旋转矩阵)

```cpp
struct RandomRotationMatrix : LinearTransform {
    RandomRotationMatrix(int d_in_val, int d_out_val);
    void init(int seed);  // 使用 QR 分解生成正交矩阵
};
```

- **复杂度**: O(d²)
- **内存**: O(d²)
- **适用**: 低维度 (d < 1024)

### 2. HadamardRotation (快速 Hadamard 旋转)

```cpp
struct HadamardRotation : VectorTransform {
    uint32_t seed;
    std::vector<float> signs1, signs2, signs3;  // 三轮随机符号翻转
    
    explicit HadamardRotation(int d, uint32_t seed = 12345);
};
```

- **复杂度**: O(d log d)
- **内存**: O(d)
- **适用**: 高维度 (d ≥ 1024)

### 3. 其他变换类

- `LinearTransform`: 线性变换基类
- `PCAMatrix`: PCA 降维 + 可选白化 + 随机旋转
- `OPQMatrix`: 优化乘积量化
- `NormalizationTransform`: L2 归一化
- `RemapDimensionsTransform`: 维度映射

## index_factory 用法

```python
import faiss

# 随机旋转 (O(d²))
index = faiss.index_factory(d, "RR64,Flat")

# Hadamard 旋转 (O(d log d))
index = faiss.index_factory(d, "HR64,Flat")

# TurboQuant 完整流程
index = faiss.index_factory(d, "L2norm,HR,SQtqmse4")
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

### 三轮 Hadamard 旋转

```cpp
// H = D₃ · FWHT · D₂ · FWHT · D₁ · FWHT
void HadamardRotation::apply_noalloc(idx_t n, const float* x, float* xt) const {
    // Round 1: D₁ · FWHT
    for (size_t j = 0; j < d; j++) xo[j] = xi[j] * signs1[j];
    fwht_inplace(xo, p);
    
    // Round 2: D₂ · FWHT
    for (size_t j = 0; j < p; j++) xo[j] *= signs2[j];
    fwht_inplace(xo, p);
    
    // Round 3: D₃ · FWHT + Normalize
    for (size_t j = 0; j < p; j++) xo[j] *= signs3[j];
    fwht_inplace(xo, p);
    
    // Normalize: scale = 1 / (p * √p)
    for (size_t j = 0; j < p; j++) xo[j] *= total_scale;
}
```

## 参考文献

- [FAISS Wiki: The index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
- [Fast Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
