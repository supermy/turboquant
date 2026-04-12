# FAISS TurboQuant 实现

本目录包含从 FAISS PR #5049 抽取的 TurboQuant 相关代码。

## 文件结构

```
faiss/
└── scalar_quantizer/
    ├── quantizers.h      # TurboQuant 量化器定义
    ├── training.h        # 训练接口
    ├── training.cpp      # 训练实现（核心算法）
    ├── sq-avx2.cpp       # AVX2 SIMD 优化
    ├── sq-avx512.cpp     # AVX-512 SIMD 优化
    ├── sq-neon.cpp       # ARM NEON 优化
    └── test_scalar_quantizer.cpp  # 测试代码
```

## 核心算法

### TurboQuant MSE 量化器

FAISS 实现了 `QuantizerTurboQuantMSE` 模板类，支持 1-8 bit 量化：

```cpp
template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantMSE : ScalarQuantizer::SQuantizer {
    // 编码向量
    void encode_vector(const float* x, uint8_t* code) const;
    
    // 解码向量
    void decode_vector(const uint8_t* code, float* x) const;
    
    // 重建单个分量
    float reconstruct_component(const uint8_t* code, size_t i) const;
};
```

### 关键特性

1. **无需训练数据**
   - 代码本从 Beta 分布解析计算
   - 只需维度 `d` 和位数 `nbits`

2. **Beta 分布最优量化**
   - 使用 Lloyd-Max 算法计算最优质心
   - 支持 1-8 bit 量化

3. **SIMD 优化**
   - AVX2: 8 个分量并行处理
   - AVX-512: 16 个分量并行处理
   - ARM NEON: 8 个分量并行处理

## API 说明

### 训练接口

```cpp
// 构建 TurboQuant MSE 代码本
void train_TurboQuantMSE(
    size_t d,        // 向量维度
    size_t nbits,    // 每分量位数 (1-8)
    std::vector<float>& trained  // 输出: [质心, 边界]
);
```

### 使用示例

```cpp
#include <faiss/impl/ScalarQuantizer.h>

// 创建 TurboQuant 量化器
faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_4bit_tqmse);

// 训练（无需数据）
sq.train(0, nullptr);

// 编码
std::vector<uint8_t> codes(sq.code_size * n);
sq.compute_codes(x, codes.data(), n);

// 解码
std::vector<float> decoded(n * d);
sq.decode(codes.data(), decoded.data(), n);
```

### Python 接口

```python
import faiss

# 创建索引
index = faiss.index_factory(128, 'SQtqmse4')

# 训练（无需数据）
index.train(xb)  # xb 仅用于确定维度

# 添加向量
index.add(xb)

# 搜索
D, I = index.search(xq, k)
```

## 支持的量化类型

| 类型 | 位数 | 代码大小 (d=128) | 说明 |
|------|------|------------------|------|
| `QT_1bit_tqmse` | 1 | 16 bytes | 二值量化 |
| `QT_2bit_tqmse` | 2 | 32 bytes | 2-bit 量化 |
| `QT_3bit_tqmse` | 3 | 48 bytes | 3-bit 量化 |
| `QT_4bit_tqmse` | 4 | 64 bytes | 4-bit 量化 |
| `QT_8bit_tqmse` | 8 | 128 bytes | 8-bit 量化 |

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

**关键优势**：
- 训练时间几乎为零
- 编码速度最快
- 重建误差最低
- Recall 最高

## SIMD 优化细节

### AVX2 实现

```cpp
// 4-bit 解包到 8 个 uint32
FAISS_ALWAYS_INLINE __m256i unpack_8x4bit_to_u32(const uint8_t* code, int i) {
    const uint32_t packed = load_u32(code + (i >> 1));
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i indices = _mm256_srlv_epi32(_mm256_set1_epi32(packed), shifts);
    return _mm256_and_si256(indices, _mm256_set1_epi32(0xf));
}

// 使用 gather 指令查表
FAISS_ALWAYS_INLINE simd8float32 reconstruct_8_components(const uint8_t* code, int i) const {
    const __m256i indices = unpack_8x4bit_to_u32(code, i);
    return simd8float32(_mm256_i32gather_ps(centroids, indices, sizeof(float)));
}
```

### ARM NEON 实现

```cpp
// 4-bit 解包
void unpack_8x4bit_to_u8(const uint8_t* code, int i, uint8_t out[8]) {
    const uint32_t packed = load_u32(code + (i >> 1));
    for (size_t j = 0; j < 8; ++j) {
        out[j] = (packed >> (4 * j)) & 0xf;
    }
}

// 查表重建
simd8float32 reconstruct_8_components(const uint8_t* code, int i) const {
    uint8_t indices[8];
    unpack_8x4bit_to_u8(code, i, indices);
    return gather_8_components(centroids, indices);
}
```

## 与 llama.cpp 实现的对比

| 特性 | FAISS | llama.cpp |
|------|-------|-----------|
| **量化方式** | Beta 分布最优 | Beta 分布最优 |
| **位宽支持** | 1-8 bit | 4 bit |
| **旋转矩阵** | 无 | 随机正交旋转 |
| **SIMD 支持** | AVX2/AVX-512/NEON | CUDA |
| **训练需求** | 无 | 无 |
| **集成方式** | ScalarQuantizer 扩展 | GGML 类型扩展 |

## 算法原理

### Beta 分布

对于 d 维单位向量，随机旋转后每个坐标的边缘分布：

```
p(x) ∝ (1 - x²)^((d-3)/2),  x ∈ [-1, 1]
```

这是 Beta((d-1)/2, (d-1)/2) 分布在 [-1, 1] 上的投影。

### Lloyd-Max 量化

1. **初始化**: 等质量划分
2. **迭代**:
   - 计算边界：b_i = (c_i + c_{i+1}) / 2
   - 更新质心：c_i = E[X | b_{i-1} < X < b_i]
3. **收敛**: 最大质心变化 < 1e-8

### 代码本结构

```
trained = [centroids[k], boundaries[k-1]]

其中:
- k = 2^nbits
- centroids: 量化质心
- boundaries: Voronoi 边界
```

## 参考文献

1. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

2. **FAISS PR #5049**
   - https://github.com/facebookresearch/faiss/pull/5049

## 许可证

代码来自 FAISS 项目，遵循 MIT 许可证。
