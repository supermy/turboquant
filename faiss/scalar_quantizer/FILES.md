# FAISS TurboQuant 实现文件清单

## 文件结构

```
faiss/scalar_quantizer/
├── quantizers.h              # TurboQuant 量化器定义 (11K)
├── training.h                # 训练接口 (1.5K)
├── training.cpp              # 训练实现 - 核心算法 (12K)
├── sq-avx2.cpp               # AVX2 SIMD 优化 (20K)
├── sq-avx512.cpp             # AVX-512 SIMD 优化 (20K)
├── sq-neon.cpp               # ARM NEON 优化 (19K)
└── test_scalar_quantizer.cpp # 测试代码 (15K)
```

## 文件详情

### quantizers.h (11K)
- TurboQuant MSE 量化器定义
- 量化器模板类
- 支持 1-8 bit 量化

**核心类**:
```cpp
template <int NBits, SIMDLevel SL>
struct QuantizerTurboQuantMSE : ScalarQuantizer::SQuantizer {
    static constexpr size_t kCentroidsCount = size_t(1) << NBits;
    
    const float* centroids;
    const float* boundaries;
    
    void encode_vector(const float* x, uint8_t* code) const;
    void decode_vector(const uint8_t* code, float* x) const;
};
```

### training.h (1.5K)
- 训练接口定义
- TurboQuant MSE 训练函数

**核心函数**:
```cpp
void train_TurboQuantMSE(
    size_t d,        // 向量维度
    size_t nbits,    // 每分量位数 (1-8)
    std::vector<float>& trained  // 输出: [质心, 边界]
);
```

### training.cpp (12K)
- TurboQuant 训练实现
- Beta 分布代码本计算
- Lloyd-Max 迭代优化

**核心算法**:
- Beta 分布: p(x) ∝ (1 - x²)^((d-3)/2)
- Lloyd-Max 迭代
- 理论 MSE 界: √3·π/2 · (1/2^nbits)²

### sq-avx2.cpp (20K)
- AVX2 SIMD 优化实现
- 8 个分量并行处理
- Gather 指令查表

**核心函数**:
```cpp
FAISS_ALWAYS_INLINE __m256i unpack_8x4bit_to_u32(const uint8_t* code, int i);
FAISS_ALWAYS_INLINE simd8float32 reconstruct_8_components(const uint8_t* code, int i) const;
```

### sq-avx512.cpp (20K)
- AVX-512 SIMD 优化实现
- 16 个分量并行处理
- 更高吞吐量

**核心函数**:
```cpp
FAISS_ALWAYS_INLINE __m512i unpack_16x4bit_to_u32(const uint8_t* code, int i);
FAISS_ALWAYS_INLINE simd16float32 reconstruct_16_components(const uint8_t* code, int i) const;
```

### sq-neon.cpp (19K)
- ARM NEON SIMD 优化
- 8 个分量并行处理
- 移动设备优化

**核心函数**:
```cpp
void unpack_8x4bit_to_u8(const uint8_t* code, int i, uint8_t out[8]);
simd8float32 gather_8_components(const float* codebook, const uint8_t indices[8]);
```

### test_scalar_quantizer.cpp (15K)
- 完整的测试套件
- 编码/解码测试
- SIMD 路径一致性测试

**测试内容**:
- TQMSE 编码/解码
- 精度排序验证
- 非 SIMD 维度测试
- SIMD 分发选择
- SIMD 距离路径一致性

## 性能数据

### 压缩比

| 位数 | 代码大小 (d=128) | 压缩比 |
|------|------------------|--------|
| 1    | 16 bytes         | 32x    |
| 2    | 32 bytes         | 16x    |
| 3    | 48 bytes         | 10.7x  |
| 4    | 64 bytes         | 8x     |
| 8    | 128 bytes        | 4x     |

### 精度

| 位数 | MSE (理论) | MSE (实际) | 余弦相似度 |
|------|------------|------------|------------|
| 1    | 0.170      | ~0.175     | > 0.70     |
| 2    | 0.042      | ~0.045     | > 0.85     |
| 3    | 0.011      | ~0.012     | > 0.92     |
| 4    | 0.0026     | ~0.003     | > 0.96     |
| 8    | 0.0001     | ~0.0001    | > 0.99     |

## 使用示例

### C++ API

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

### Python API

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

## SIMD 优化对比

| SIMD 级别 | 并行度 | 平台 | 性能提升 |
|-----------|--------|------|----------|
| NONE      | 1      | 通用  | 1x       |
| AVX2      | 8      | x86   | ~8x      |
| AVX-512   | 16     | x86   | ~16x     |
| NEON      | 8      | ARM   | ~8x      |

## 参考文献

1. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

2. **FAISS PR #5049**
   - https://github.com/facebookresearch/faiss/pull/5049

## 许可证

代码来自 FAISS 项目，遵循 MIT 许可证。
