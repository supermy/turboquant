# llama.cpp TurboQuant 实现

本目录包含从 llama.cpp PR #20995 抽取的 TurboQuant (TQ4_0) 相关代码。

## 文件结构

```
llamacpp/
├── tq/
│   ├── tq_quants.h      # 头文件 - API 定义
│   ├── tq_quants.c      # CPU 实现 - 核心量化算法
│   └── test_tq.c        # 测试文件
└── cuda/
    ├── tq4-rotate.cu    # CUDA 旋转内核
    └── tq4-set-rows.cu  # CUDA SET_ROWS 内核
```

## 核心算法

### TQ4_0: TurboQuant 4-bit 量化

TQ4_0 实现了以下流程：

```
输入向量 x ∈ R^d
    │
    ▼
┌─────────────────────────┐
│  随机正交旋转           │  y = Π^T @ x
│  (QR 分解生成)          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Beta 分布量化          │  使用 Lloyd-Max 最优质心
│  (16级量化)             │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Q4_0 块打包            │  18 bytes / 32 values
│  (与 llama.cpp 兼容)    │
└───────────┬─────────────┘
            ▼
    压缩后的 KV Cache
```

### 关键特性

1. **随机正交旋转**
   - 通过 QR 分解生成随机正交矩阵 Π
   - 旋转后每个坐标遵循 Beta(d/2, d/2) 分布
   - 分布集中，量化误差小

2. **Beta 最优量化**
   - 使用 Lloyd-Max 算法计算最优质心
   - 16 级量化（4-bit）
   - 理论 MSE 界：√3·π/2 · (1/4^4) ≈ 0.0106

3. **Q4_0 兼容格式**
   - 与 llama.cpp 的 Q4_0 块布局相同
   - 18 bytes 存储 32 个值
   - FP16 缩放因子 + 4-bit 量化值

## API 说明

### 初始化

```c
// 初始化旋转矩阵
// head_dim: 注意力头维度（通常 128）
// seed: 随机种子
void tq_init(int head_dim, uint64_t seed);

// 释放资源
void tq_free(void);

// 获取头维度
int tq_head_dim(void);
```

### 量化/反量化

```c
// 量化一行浮点数
void quantize_row_tq4_0_ref(const float * x, void * y, int64_t k);

// 反量化
void dequantize_row_tq4_0(const void * x, float * y, int64_t k);

// 点积（用于注意力计算）
void vec_dot_tq4_0_q8_0(int n, float * s, ...);
```

## CUDA 实现

### tq4-rotate.cu

预旋转内核，在 SET_ROWS 之前应用旋转：

```cuda
// 旋转 KV 数据
void tq_cuda_rotate_before_set_rows(float * data, int n_elements, void * stream);

// 反旋转（解压后）
void tq_cuda_unrotate_after_dequant(float * data, int n_elements, void * stream);
```

### tq4-set-rows.cu

完整的 CUDA 量化内核：

```cuda
// 旋转 + 量化一个 head 向量
__global__ void k_tq4_0_set_rows(
    const float * src,
    const int64_t * indices,
    block_q4_0 * dst,
    const float * rot_t,
    const int head_dim,
    ...
);
```

## 性能数据

### 压缩比

| 维度 | 原始大小 | 压缩后 | 压缩比 |
|------|----------|--------|--------|
| 128  | 512 bytes | 72 bytes | 7.1x |
| 256  | 1024 bytes | 144 bytes | 7.1x |

### 重建质量

- **MSE**: ~0.01（理论界 0.0106）
- **余弦相似度**: > 0.95
- **范数误差**: < 0.01

## 使用示例

### C 语言

```c
#include "tq_quants.h"

int main() {
    // 初始化
    tq_init(128, 42);
    
    // 量化
    float input[128] = {...};
    block_q4_0 output[4];  // 128 / 32 = 4 blocks
    quantize_row_tq4_0_ref(input, output, 128);
    
    // 反量化
    float reconstructed[128];
    dequantize_row_tq4_0(output, reconstructed, 128);
    
    // 清理
    tq_free();
    return 0;
}
```

### 编译测试

```bash
cd llamacpp/tq
gcc -O2 -lm -o test_tq test_tq.c tq_quants.c
./test_tq
```

## 与 PolarQuant 的区别

| 特性 | PolarQuant | TurboQuant (TQ4_0) |
|------|------------|-------------------|
| **旋转方式** | 随机正交 | 随机正交 |
| **量化方式** | 极坐标 + Lloyd-Max | Beta 分布 + Lloyd-Max |
| **存储格式** | 自定义 | Q4_0 兼容 |
| **集成方式** | 独立库 | llama.cpp 内置 |
| **GPU 支持** | 需自行实现 | CUDA 内核已提供 |

## 参考文献

1. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

2. **llama.cpp PR #20995**
   - https://github.com/ggml-org/llama.cpp/pull/20995

## 许可证

代码来自 llama.cpp 项目，遵循 MIT 许可证。
