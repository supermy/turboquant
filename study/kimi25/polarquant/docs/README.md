# PolarQuant 时序图文档

本文档包含 PolarQuant 算法的可视化时序图和流程图。

## 文件说明

| 文件 | 说明 |
|------|------|
| `sequence_diagram.png` | 时序图 (PNG格式) |
| `algorithm_flow.png` | 算法流程图 (PNG格式) |
| `sequence_diagram.puml` | PlantUML 源文件 |
| `sequence_diagram.md` | Mermaid 时序图 (Markdown) |

## 时序图说明

### 参与者 (Participants)

```
User/Caller (用户/调用者)
    ↓
PolarQuantConfig (配置类)
    ↓
PolarQuant (主量化类)
    ↓
Utils Module (工具模块)
    ↓
CompressedVector (压缩向量类)
```

### 阶段 1: 初始化 (Initialization)

```
1. 用户创建配置
   - dimension: 向量维度
   - radius_bits: 半径量化比特数
   - angle_bits: 角度量化比特数

2. 初始化 PolarQuant
   ↓
3. 生成随机正交矩阵 Q
   - 使用 QR 分解
   - Q @ Q^T = I (单位矩阵)
   
4. 计算 Lloyd-Max 质心
   - 基于 Beta(d/2, d/2) 分布
   - 预计算最优量化质心
   
5. 返回量化器实例
```

### 阶段 2: 压缩 (Compression)

```
输入: 向量 x ∈ R^d

步骤 1: 随机旋转
   y = Q @ x
   目的: 将分布归一化为 Beta 分布

步骤 2: 极坐标变换
   (r, θ) = cartesian_to_polar(y)
   r: 半径 (L2 范数)
   θ: d-1 个角度

步骤 3a: 半径量化 (对数尺度)
   idx_r = quantize_radius(r)
   使用对数尺度处理大动态范围

步骤 3b: 角度量化 (Lloyd-Max)
   idx_θ = lloyd_max_quantize(θ)
   基于 Beta 分布的最优量化

步骤 4: 存储索引
   返回 CompressedVector(idx_r, idx_θ)
```

### 阶段 3: 解压 (Decompression)

```
输入: CompressedVector

步骤 1: 反量化半径
   r = dequantize_radius(idx_r)

步骤 2: 反量化角度
   θ = lloyd_max_dequantize(idx_θ)

步骤 3: 极坐标转笛卡尔
   y = polar_to_cartesian(r, θ)

步骤 4: 逆旋转
   x = Q^T @ y
   
输出: 重建向量 x_reconstructed
```

## 算法流程图

```
┌─────────────────────────────────────────┐
│  Input Vector x ∈ R^d                   │
│  (输入向量)                              │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 1: Random Rotation                │
│  (步骤1: 随机旋转)                       │
│  y = Q · x                              │
│  目的: 归一化分布为 Beta(d/2, d/2)       │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 2: Polar Transform                │
│  (步骤2: 极坐标变换)                     │
│  (r, θ) = cart2pol(y)                   │
│  笛卡尔坐标 → (半径, 角度)               │
└─────────────────┬───────────────────────┘
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
┌───────────────┐   ┌───────────────┐
│ Step 3a:      │   │ Step 3b:      │
│ Radius Quant  │   │ Angle Quant   │
│ (半径量化)     │   │ (角度量化)     │
│ Log-scale     │   │ Lloyd-Max     │
│ r → idx_r     │   │ θ → idx_θ     │
└───────┬───────┘   └───────┬───────┘
        └─────────┬─────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 4: Store Indices                  │
│  (步骤4: 存储索引)                       │
│  (idx_r, idx_θ)                         │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Compressed Representation              │
│  (压缩表示)                              │
│  High Compression Ratio (~8x)           │
└─────────────────────────────────────────┘
```

## 关键概念

### Q: Random Orthogonal Matrix (随机正交矩阵)
- **生成方式**: QR 分解
- **性质**: Q @ Q^T = I (单位矩阵)
- **作用**: 将向量分布归一化为 Beta 分布

### Lloyd-Max Quantization (Lloyd-Max 量化)
- **目标**: 最小化均方误差 (MSE)
- **方法**: 迭代优化量化质心
- **输入**: Beta(d/2, d/2) 分布
- **输出**: 最优量化质心

### Compression Ratio (压缩比)
```
原始大小: d × 32 bits (float32)
压缩大小: r_bits + (d-1) × a_bits

示例: d=256, r_bits=8, a_bits=4
原始: 256 × 32 = 8192 bits
压缩: 8 + 255 × 4 = 1028 bits
压缩比: 8192 / 1028 ≈ 7.97x
```

## 使用说明

### 生成时序图

```bash
cd docs
python generate_sequence_diagram.py
```

### 查看时序图

直接打开 `sequence_diagram.png` 和 `algorithm_flow.png` 查看。

## 参考

- TurboQuant Paper (ICLR 2026)
- PolarQuant: Quantizing KV Caches with Polar Transformation
