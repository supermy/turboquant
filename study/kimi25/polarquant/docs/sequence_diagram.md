# PolarQuant 时序图

## 1. 初始化阶段

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant Config as "配置类"
    participant PQ as "主量化类"
    participant Rot as "旋转模块"
    participant Lloyd as "量化器"

    User->>Config: 创建配置
    Config-->>User: 配置对象
    
    User->>PQ: PolarQuant(配置)
    activate PQ
    
    PQ->>Rot: 生成正交矩阵(d, 种子)
    activate Rot
    Rot-->>PQ: 旋转矩阵Q
    deactivate Rot
    
    PQ->>Lloyd: 计算质心(α, β, 比特数)
    activate Lloyd
    Lloyd-->>PQ: 角度质心
    deactivate Lloyd
    
    PQ-->>User: 量化器就绪
    deactivate PQ
```

## 2. 压缩阶段

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant PQ as "主量化类"
    participant Rot as "旋转模块"
    participant Polar as "极坐标变换"
    participant Lloyd as "量化器"
    participant CV as "压缩向量"

    User->>PQ: 压缩(x)
    activate PQ
    
    Note over PQ: 步骤1: 随机旋转
    PQ->>Rot: 旋转(x)
    activate Rot
    Rot-->>PQ: x_旋转后 = Q * x
    deactivate Rot
    
    Note over PQ: 步骤2: 极坐标变换
    PQ->>Polar: 笛卡尔转极坐标(x_旋转后)
    activate Polar
    Polar-->>PQ: (半径, 角度)
    deactivate Polar
    
    Note over PQ: 步骤3: 半径量化
    PQ->>PQ: 量化半径(半径)
    activate PQ
    PQ-->>PQ: 半径索引
    deactivate PQ
    
    Note over PQ: 步骤4: 角度量化
    PQ->>Lloyd: 最大量化(角度, 质心)
    activate Lloyd
    Lloyd-->>PQ: 角度索引
    deactivate Lloyd
    
    PQ->>CV: 创建压缩向量()
    activate CV
    CV-->>PQ: 已压缩
    deactivate CV
    
    PQ-->>User: 压缩后的向量
    deactivate PQ
```

## 3. 解压阶段

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant PQ as "主量化类"
    participant Lloyd as "量化器"
    participant Polar as "极坐标变换"
    participant Rot as "旋转模块"

    User->>PQ: 解压(已压缩)
    activate PQ
    
    Note over PQ: 步骤1: 反量化半径
    PQ->>PQ: 反量化半径(半径索引)
    activate PQ
    PQ-->>PQ: 半径
    deactivate PQ
    
    Note over PQ: 步骤2: 反量化角度
    PQ->>Lloyd: 最大反量化(角度索引, 质心)
    activate Lloyd
    Lloyd-->>PQ: 角度
    deactivate Lloyd
    
    Note over PQ: 步骤3: 极坐标转笛卡尔
    PQ->>Polar: 极坐标转笛卡尔(半径, 角度)
    activate Polar
    Polar-->>PQ: x_旋转后
    deactivate Polar
    
    Note over PQ: 步骤4: 逆旋转
    PQ->>Rot: 逆旋转(x_旋转后)
    activate Rot
    Rot-->>PQ: x = Q^T * x_旋转后
    deactivate Rot
    
    PQ-->>User: 重建的向量
    deactivate PQ
```

## 4. 批处理阶段

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant Batch as "批处理类"
    participant PQ as "主量化类"

    User->>PQ: 创建批处理(量化器)
    activate PQ
    PQ-->>User: 批处理器
    deactivate PQ
    
    User->>Batch: 批量压缩(X)
    activate Batch
    
    loop 每个向量 x in X
        Batch->>PQ: 压缩(x)
        activate PQ
        PQ-->>Batch: 压缩后的向量
        deactivate PQ
    end
    
    Batch-->>User: 压缩后的向量列表
    deactivate Batch
```

## 5. 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant Config as "配置类"
    participant PQ as "主量化类"
    participant Utils as "工具函数"
    participant CV as "压缩向量"

    User->>Config: 创建配置
    Config-->>User: 配置
    User->>PQ: 初始化
    PQ->>Utils: 生成正交矩阵()
    Utils-->>PQ: Q
    PQ->>Utils: 计算质心()
    Utils-->>PQ: 质心
    PQ-->>User: 量化器

    User->>PQ: 压缩(x)
    PQ->>Utils: 旋转(x)
    Utils-->>PQ: x_旋转后
    PQ->>Utils: 笛卡尔转极坐标()
    Utils-->>PQ: (半径, 角度)
    PQ->>PQ: 量化半径()
    PQ-->>PQ: 半径索引
    PQ->>Utils: 最大量化()
    Utils-->>PQ: 角度索引
    PQ->>CV: 创建
    CV-->>PQ: 已压缩
    PQ-->>User: 已压缩

    User->>PQ: 解压(已压缩)
    PQ->>PQ: 反量化半径()
    PQ-->>PQ: 半径
    PQ->>Utils: 最大反量化()
    Utils-->>PQ: 角度
    PQ->>Utils: 极坐标转笛卡尔()
    Utils-->>PQ: x_旋转后
    PQ->>Utils: 逆旋转()
    Utils-->>PQ: 重建的向量
    PQ-->>User: 重建的向量
```

---

## 参与者说明

| 名称 | 对应代码 | 职责 |
|------|----------|------|
| 用户 | User code | 调用 PolarQuant API |
| 配置类 | `core.py` 配置类 | 存储量化参数 |
| 主量化类 | `core.py` 主类 | 实现压缩/解压算法 |
| 旋转模块 | `utils.py` | 生成旋转矩阵 Q |
| 极坐标变换 | `utils.py` | 坐标系转换 |
| 量化器 | `utils.py` | 计算质心和量化 |
| 压缩向量 | `core.py` 数据类 | 存储压缩结果 |
| 工具函数 | `utils.py` | 数学运算工具 |
| 批处理类 | `core.py` | 批量处理向量 |

## 关键步骤说明

**阶段 1: 初始化**
1. 用户创建配置
2. 初始化主量化类
3. 生成随机正交矩阵 Q (QR分解)
4. 计算 Lloyd-Max 质心 (基于 Beta 分布)
5. 返回量化器实例

**阶段 2: 压缩**
1. 随机旋转: y = Q @ x (将分布归一化为 Beta 分布)
2. 极坐标变换: (半径, 角度) = 笛卡尔转极坐标(y)
3. 半径量化: 半径 → 半径索引 (对数尺度)
4. 角度量化: 角度 → 角度索引 (Lloyd-Max 最优量化)
5. 存储索引: 返回压缩向量

**阶段 3: 解压**
1. 反量化半径: 半径索引 → 半径
2. 反量化角度: 角度索引 → 角度
3. 极坐标转笛卡尔: (半径, 角度) → y
4. 逆旋转: x = Q^T @ y
5. 返回重建向量

**阶段 4: 批处理**
- 对多个向量循环处理
- 复用同一个量化器实例

## 压缩比计算

```
原始大小: d × 32 bits (float32)
压缩大小: r_bits + (d-1) × a_bits

示例: d=256, r_bits=8, a_bits=4
原始: 256 × 32 = 8192 bits
压缩: 8 + 255 × 4 = 1028 bits
压缩比: 8192 / 1028 ≈ 7.97x
```

## 核心公式

**随机旋转:**
```
y = Q · x
Q: 随机正交矩阵 (Q @ Q^T = I)
```

**极坐标变换:**
```
x_0 = 半径 · cos(角度_0)
x_1 = 半径 · sin(角度_0) · cos(角度_1)
x_2 = 半径 · sin(角度_0) · sin(角度_1) · cos(角度_2)
...
```

**Lloyd-Max 量化:**
```
边界: b_i = (c_{i-1} + c_i) / 2
质心: c_i = E[X | b_i < X < b_{i+1}]
```
