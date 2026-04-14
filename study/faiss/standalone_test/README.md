# TurboQuant 完整流程测试

## 概述

本测试演示 TurboQuant 的完整处理流程，包括 L2 归一化、Hadamard 旋转、Beta 分布量化和反量化、反旋转等步骤。

## 编译与运行

### 编译命令

```bash
# 独立编译 (需要 C++17)
g++ -std=c++17 -O3 -o test_turboquant_pipeline test_turboquant_pipeline.cpp -lm

# 运行
./test_turboquant_pipeline
```

### 依赖

- C++17 编译器
- 无外部库依赖 (纯标准库实现)

## 测试流程

```
输入向量
    │
    ▼
┌─────────────┐
│ L2 归一化   │  将向量归一化到单位球面
└─────────────┘
    │
    ▼
┌─────────────┐
│ Hadamard    │  3轮 FWHT + 随机符号翻转
│ 旋转        │  满足 Johnson-Lindenstrauss 引理
└─────────────┘
    │
    ▼
┌─────────────┐
│ Beta 分布   │  Lloyd-Max 迭代构建最优码本
│ 量化        │  单位向量坐标服从 Beta 分布
└─────────────┘
    │
    ▼
┌─────────────┐
│ 反量化      │  用质心值替换编码
└─────────────┘
    │
    ▼
┌─────────────┐
│ 反旋转      │  逆向 Hadamard 变换
└─────────────┘
    │
    ▼
输出向量
```

## 核心组件

### 1. L2 归一化 (`l2_normalize`)

将输入向量归一化到单位长度，使坐标服从单位球面上的分布。

### 2. Fast Walsh-Hadamard Transform (`fwht_inplace`)

原地执行 Walsh-Hadamard 变换，时间复杂度 O(n log n)。

```cpp
void fwht_inplace(float* buf, size_t n);
```

### 3. Hadamard 旋转 (`HadamardRotation`)

实现 3 轮随机符号翻转 + FWHT 的随机投影变换。

**数学原理**:
- 正向变换: `y = scale * FWHT * D₃ * FWHT * D₂ * FWHT * D₁ * x`
- 反向变换: `x = D₁ * FWHT * D₂ * FWHT * D₃ * FWHT * y * scale`

其中 D₁, D₂, D₃ 是随机对角符号矩阵。

**为什么是 3 轮?**
- 基于 Ailon-Chazelle 定理
- 3 轮 Hadamard 变换满足 Johnson-Lindenstrauss 引理
- 将任意分布的向量转换为近似高斯分布

### 4. Beta 分布量化器 (`TurboQuantMSE`)

使用 Lloyd-Max 迭代构建最优标量量化码本。

**Beta 分布**:
- 单位向量坐标服从: `p(x) ∝ (1 - x²)^((d-3)/2)`
- d 是向量维度

**Lloyd-Max 迭代**:
1. 初始化等质量划分
2. 更新边界为相邻质心中点
3. 更新质心为区间加权均值
4. 重复直到收敛

## 测试函数

### `test_full_pipeline`

测试完整流程: 旋转 + 量化 + 反量化 + 反旋转

```cpp
void test_full_pipeline(size_t d, int nbits, size_t n_vectors);
```

**参数**:
- `d`: 向量维度
- `nbits`: 每维量化位数 (1-8)
- `n_vectors`: 测试向量数量

**输出指标**:
- MSE (均方误差)
- RMSE (均方根误差)
- 平均余弦相似度
- 处理时间

### `test_quantization_only`

仅测试量化过程，不包含旋转步骤。

```cpp
void test_quantization_only(size_t d, int nbits, size_t n_vectors);
```

## 预期结果

### 128 维 4-bit 配置

```
维度 d = 128
位宽 nbits = 4
向量数 n = 1000

码本大小: 64 bytes/vector
压缩比: 12.50%

MSE (均方误差): ~7.29e-05
RMSE (均方根误差): ~8.54e-03
平均余弦相似度: ~0.9954
```

### 不同维度对比

| 维度 | MSE | 余弦相似度 |
|------|-----|-----------|
| 128  | ~7.3e-05 | ~0.995 |
| 256  | ~7.0e-05 | ~0.996 |
| 512  | ~6.8e-05 | ~0.997 |

### 不同位宽对比

| 位宽 | 压缩比 | MSE | 余弦相似度 |
|------|--------|-----|-----------|
| 1-bit | 3.1% | ~0.33 | ~0.82 |
| 2-bit | 6.3% | ~0.003 | ~0.96 |
| 3-bit | 9.4% | ~4e-04 | ~0.99 |
| 4-bit | 12.5% | ~7e-05 | ~0.995 |
| 8-bit | 25.0% | ~2e-07 | ~0.9999 |

## 数学背景

### Johnson-Lindenstrauss 引理

对于 n 个点的集合，存在映射 f: R^d → R^k，使得:

```
(1-ε)||u-v||² ≤ ||f(u)-f(v)||² ≤ (1+ε)||u-v||²
```

其中 k = O(log n / ε²)。

### Ailon-Chazelle 定理

3 轮 Hadamard 变换 + 随机符号翻转构成的随机投影满足 JL 引理:

```
H = (1/√n) * H_n * D_3 * H_n * D_2 * H_n * D_1
```

其中 H_n 是 n×n Hadamard 矩阵，D_i 是随机对角符号矩阵。

### Beta 分布推导

对于 d 维单位球面上均匀分布的随机向量，其单个坐标的边缘分布为:

```
p(x) ∝ (1 - x²)^((d-3)/2),  x ∈ [-1, 1]
```

这是参数为 (α, β) = ((d-1)/2, (d-1)/2) 的 Beta 分布在对称区间上的形式。

## 文件结构

```
test_turboquant_pipeline.cpp
├── l2_normalize()           # L2 归一化
├── fwht_inplace()           # 快速 Walsh-Hadamard 变换
├── HadamardRotation         # Hadamard 旋转类
│   ├── apply()              # 正向旋转
│   └── reverse()            # 反向旋转
├── TurboQuantMSE            # Beta 分布量化器
│   ├── build_codebook()     # 构建码本
│   ├── lloyd_max_iteration() # Lloyd-Max 迭代
│   ├── encode()             # 编码
│   └── decode()             # 解码
├── test_full_pipeline()     # 完整流程测试
├── test_quantization_only() # 仅量化测试
└── main()                   # 主函数
```

## 参考

- FAISS PR #5049: TurboQuant 实现
- Ailon & Chazelle (2009): "The Fast Johnson-Lindenstrauss Transform"
- Lloyd (1982): "Least Squares Quantization in PCM"
