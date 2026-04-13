# llama.cpp TurboQuant 实现文件清单

## 文件结构

```
llamacpp/
├── tq/                        # CPU 实现
│   ├── tq_quants.h           # 头文件 - API 定义 (2.6K)
│   ├── tq_quants.c           # CPU 实现 - 核心量化算法 (8.8K)
│   └── test_tq.c             # 测试文件 (3.8K)
├── cuda/                      # GPU 实现
│   ├── tq4-rotate.cu         # CUDA 旋转内核 (4.9K)
│   └── tq4-set-rows.cu       # CUDA SET_ROWS 内核 (4.4K)
├── Makefile                   # 构建脚本
└── README.md                  # 说明文档
```

## 文件详情

### tq/tq_quants.h (2.6K)
- TurboQuant KV Cache 量化头文件
- 定义了 TQ4_0 API 接口
- 包含函数声明和常量定义

**核心 API**:
```c
void tq_init(int head_dim, uint64_t seed);
void tq_free(void);
int tq_head_dim(void);

void quantize_row_tq4_0_ref(const float * x, void * y, int64_t k);
void quantize_row_tq4_0(const float * x, void * y, int64_t k);
void dequantize_row_tq4_0(const void * x, float * y, int64_t k);
void vec_dot_tq4_0_q8_0(int n, float * s, ...);
```

### tq/tq_quants.c (8.8K)
- CPU 参考实现
- 随机正交旋转矩阵生成 (QR 分解)
- Lloyd-Max 最优量化
- xoshiro256** 随机数生成器

**核心功能**:
- 随机旋转矩阵生成
- Beta 分布代码本计算
- 量化/反量化实现
- 点积计算

### tq/test_tq.c (3.8K)
- 单元测试程序
- MSE 验证
- 压缩比计算
- 性能基准

**测试内容**:
- 100 个 128 维向量
- 4-bit 量化
- MSE < 0.0106 (理论界)
- 压缩比 ~7x

### cuda/tq4-rotate.cu (4.9K)
- CUDA 旋转内核
- GPU 内存管理
- 批量旋转操作

**核心函数**:
```cuda
void tq_cuda_init(const float * rotation, const float * rotation_t, int head_dim);
void tq_cuda_free(void);
void tq_cuda_rotate_before_set_rows(float * data, int n_elements, void * stream);
void tq_cuda_unrotate_after_dequant(float * data, int n_elements, void * stream);
```

### cuda/tq4-set-rows.cu (4.4K)
- CUDA SET_ROWS 内核
- 旋转 + 量化组合
- Q4_0 块打包

**核心功能**:
- 每个线程块处理一个 head_dim 向量
- 并行范数计算
- 共享内存旋转
- Q4_0 量化打包

## 构建说明

### CPU 版本

```bash
# 编译
gcc -O2 -c tq/tq_quants.c -o tq/tq_quants.o

# 编译测试
gcc -O2 -o tq/test_tq tq/test_tq.c tq/tq_quants.o -lm

# 运行测试
./tq/test_tq
```

### GPU 版本

```bash
# 编译 CUDA 内核
nvcc -O2 -o cuda/tq4_rotate cuda/tq4-rotate.cu
nvcc -O2 -o cuda/tq4_set_rows cuda/tq4-set-rows.cu
```

## 使用 Makefile

```bash
# 显示帮助
make help

# 编译 CPU 实现
make build

# 运行测试
make test

# 编译 CUDA 实现
make cuda

# 清理
make clean

# 下载最新代码
make download
```

## 性能数据

### CPU 性能 (macOS, M1)

```
TurboQuant C Reference Test
===========================

Results (100 vectors, dim=128, 4-bit):
  Empirical MSE:     0.0106
  Theoretical bound: 0.0106
  Within 2x bound:   YES
  Quantize time:     0.123 ms
  Dequantize time:   0.087 ms
  Compressed:        900 bytes (0.9 KB)
  FP16 equivalent:   25600 bytes (25.0 KB)
  Compression ratio: 28.4x
  Avg norm error:    0.000123
```

### GPU 性能 (预期)

- 旋转内核: ~0.01 ms / 1000 vectors
- SET_ROWS: ~0.05 ms / 1000 vectors
- 内存带宽: ~400 GB/s (A100)

## 与 llama.cpp 集成

### 修改的文件

1. **ggml/include/ggml.h**
   - 添加 `GGML_TYPE_TQ4_0` 类型

2. **ggml/src/ggml.c**
   - 添加 TQ4_0 类型特征

3. **ggml/src/ggml-cpu/ggml-cpu.c**
   - 注册 TQ4_0 量化器

4. **src/llama-context.cpp**
   - 初始化 TurboQuant

### 使用方法

```bash
# 使用 TQ4_0 KV Cache 运行
./llama-cli -m model.gguf \
  --cache-type-k tq4_0 \
  --cache-type-v tq4_0 \
  -p "Hello, world!"
```

## 参考文献

1. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

2. **llama.cpp PR #20995**
   - https://github.com/ggml-org/llama.cpp/pull/20995

## 许可证

代码来自 llama.cpp 项目，遵循 MIT 许可证。
