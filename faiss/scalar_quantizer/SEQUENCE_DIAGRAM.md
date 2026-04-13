# FAISS TurboQuant 时序图

## 1. 训练阶段时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant SQ as "ScalarQuantizer"
    participant Trainer as "train_TurboQuantMSE"
    participant Builder as "build_Codebook"
    participant Beta as "Beta分布计算"
    participant Lloyd as "Lloyd-Max迭代"

    User->>SQ: train(n, x)
    activate SQ
    
    SQ->>Trainer: train_TurboQuantMSE(d, nbits)
    activate Trainer
    
    Trainer->>Builder: build_TurboQuantMSECodebook(d, nbits)
    activate Builder
    
    Note over Builder: 检查参数有效性
    
    alt d == 1
        Builder->>Builder: 生成端点质心
        Builder-->>Trainer: centroids, boundaries
    else d > 1
        Builder->>Beta: 计算Beta分布
        activate Beta
        
        Note over Beta: 离散化密度函数
        Note over Beta: p(x) ∝ (1-x²)^((d-3)/2)
        
        Beta->>Beta: 构建网格 (ngrid点)
        Beta->>Beta: 计算累积质量
        Beta->>Beta: 计算累积一阶矩
        
        Beta-->>Builder: prefix_w, prefix_wx
        deactivate Beta
        
        Builder->>Lloyd: Lloyd-Max迭代优化
        activate Lloyd
        
        loop 最多100次迭代
            Lloyd->>Lloyd: 计算Voronoi边界
            Lloyd->>Lloyd: 更新质心
            Lloyd->>Lloyd: 检查收敛 (<1e-8)
        end
        
        Lloyd-->>Builder: 最优质心
        deactivate Lloyd
        
        Builder->>Builder: 排序质心
        Builder->>Builder: 计算边界
    end
    
    Builder-->>Trainer: centroids[k], boundaries[k-1]
    deactivate Builder
    
    Trainer->>Trainer: 打包训练结果
    Trainer-->>SQ: trained向量
    deactivate Trainer
    
    SQ-->>User: 训练完成
    deactivate SQ
```

## 2. 量化阶段时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant SQ as "ScalarQuantizer"
    participant Quant as "QuantizerTurboQuantMSE"
    participant Select as "select_index"
    participant Encode as "encode_index"
    participant Code as "压缩码"

    User->>SQ: compute_codes(x, codes, n)
    activate SQ
    
    SQ->>Quant: encode_vector(x, code)
    activate Quant
    
    loop 每个分量 i
        Quant->>Select: select_index(x[i])
        activate Select
        
        Note over Select: 二分查找边界
        Select->>Select: upper_bound(boundaries, x[i])
        Select-->>Quant: index
        deactivate Select
        
        Quant->>Encode: encode_index(index, code, i)
        activate Encode
        
        Note over Encode: 计算位偏移
        Note over Encode: bit_offset = i * nbits
        Note over Encode: 打包到字节
        
        Encode->>Code: 写入压缩码
        Encode-->>Quant: 完成
        deactivate Encode
    end
    
    Quant-->>SQ: 压缩码
    deactivate Quant
    
    SQ-->>User: codes数组
    deactivate SQ
```

## 3. 反量化阶段时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant SQ as "ScalarQuantizer"
    participant Quant as "QuantizerTurboQuantMSE"
    participant Decode as "decode_index"
    participant Lookup as "质心查表"
    participant Output as "输出向量"

    User->>SQ: decode(codes, x, n)
    activate SQ
    
    SQ->>Quant: decode_vector(code, x)
    activate Quant
    
    loop 每个分量 i
        Quant->>Decode: decode_index(code, i)
        activate Decode
        
        Note over Decode: 计算位偏移
        Note over Decode: bit_offset = i * nbits
        Note over Decode: 从字节解包
        
        Decode->>Decode: 读取索引
        Decode-->>Quant: index
        deactivate Decode
        
        Quant->>Lookup: centroids[index]
        activate Lookup
        Lookup-->>Quant: value
        deactivate Lookup
        
        Quant->>Output: x[i] = value
    end
    
    Quant-->>SQ: 解压向量
    deactivate Quant
    
    SQ-->>User: x数组
    deactivate SQ
```

## 4. SIMD 优化时序图 (AVX2)

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant Quant as "QuantizerTurboQuantMSE<AVX2>"
    participant Unpack as "unpack_8xNbit"
    participant Gather as "_mm256_i32gather_ps"
    participant Output as "simd8float32"

    User->>Quant: reconstruct_8_components(code, i)
    activate Quant
    
    alt 1-bit
        Quant->>Unpack: unpack_8x1bit_to_u32(code, i)
    else 2-bit
        Quant->>Unpack: unpack_8x2bit_to_u32(code, i)
    else 3-bit
        Quant->>Unpack: unpack_8x3bit_to_u32(code, i)
    else 4-bit
        Quant->>Unpack: unpack_8x4bit_to_u32(code, i)
    else 8-bit
        Quant->>Unpack: _mm256_cvtepu8_epi32
    end
    
    activate Unpack
    Note over Unpack: 加载压缩数据
    Note over Unpack: 解包到8个uint32
    Unpack-->>Quant: __m256i indices
    deactivate Unpack
    
    Quant->>Gather: gather操作
    activate Gather
    Note over Gather: 一次加载8个质心
    Note over Gather: _mm256_i32gather_ps(centroids, indices, 4)
    Gather-->>Quant: __m256 values
    deactivate Gather
    
    Quant->>Output: simd8float32(values)
    activate Output
    Output-->>User: 8个float值
    deactivate Output
    
    deactivate Quant
```

## 5. 完整流程时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant Index as "IndexScalarQuantizer"
    participant SQ as "ScalarQuantizer"
    participant Trainer as "训练模块"
    participant Quant as "量化器"
    participant DC as "距离计算器"

    User->>Index: 创建索引
    activate Index
    Index->>SQ: ScalarQuantizer(d, QT_4bit_tqmse)
    activate SQ
    SQ-->>Index: 初始化完成
    deactivate SQ
    Index-->>User: 索引对象
    deactivate Index

    User->>Index: train(xb)
    activate Index
    Index->>SQ: train(n, xb)
    activate SQ
    SQ->>Trainer: train_TurboQuantMSE(d, 4)
    activate Trainer
    Note over Trainer: 构建Beta分布代码本
    Note over Trainer: Lloyd-Max优化
    Trainer-->>SQ: trained参数
    deactivate Trainer
    SQ->>Quant: 创建量化器
    activate Quant
    Quant-->>SQ: 量化器就绪
    deactivate Quant
    SQ-->>Index: 训练完成
    deactivate SQ
    Index-->>User: 训练完成
    deactivate Index

    User->>Index: add(xb)
    activate Index
    Index->>SQ: compute_codes(xb, codes, n)
    activate SQ
    SQ->>Quant: encode_vector(x, code)
    activate Quant
    loop 每个向量
        Quant->>Quant: select_index(x[i])
        Quant->>Quant: encode_index(idx, code, i)
    end
    Quant-->>SQ: codes
    deactivate Quant
    SQ-->>Index: codes
    deactivate SQ
    Index-->>User: 添加完成
    deactivate Index

    User->>Index: search(xq, k)
    activate Index
    Index->>DC: get_distance_computer()
    activate DC
    DC->>DC: set_query(xq)
    loop 每个数据库向量
        DC->>Quant: decode_vector(code, x)
        Quant-->>DC: 解压向量
        DC->>DC: 计算距离
    end
    DC-->>Index: distances
    deactivate DC
    Index-->>User: D, I
    deactivate Index
```

## 6. SIMD 分发时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant SQ as "ScalarQuantizer"
    participant Dispatch as "sq_select_quantizer"
    participant AVX512 as "AVX-512量化器"
    participant AVX2 as "AVX2量化器"
    participant NEON as "NEON量化器"
    participant Fallback as "标量量化器"

    User->>SQ: select_quantizer()
    activate SQ
    SQ->>Dispatch: sq_select_quantizer<SIMDLevel>()
    activate Dispatch
    
    alt AVX-512可用 且 d%16==0
        Dispatch->>AVX512: new QuantizerTurboQuantMSE<NBits, AVX512>
        activate AVX512
        AVX512-->>Dispatch: AVX-512量化器
        deactivate AVX512
    else AVX2可用 且 d%8==0
        Dispatch->>AVX2: new QuantizerTurboQuantMSE<NBits, AVX2>
        activate AVX2
        AVX2-->>Dispatch: AVX2量化器
        deactivate AVX2
    else NEON可用 且 d%8==0
        Dispatch->>NEON: new QuantizerTurboQuantMSE<NBits, NEON>
        activate NEON
        NEON-->>Dispatch: NEON量化器
        deactivate NEON
    else 其他情况
        Dispatch->>Fallback: new QuantizerTurboQuantMSE<NBits, NONE>
        activate Fallback
        Fallback-->>Dispatch: 标量量化器
        deactivate Fallback
    end
    
    Dispatch-->>SQ: 量化器实例
    deactivate Dispatch
    SQ-->>User: 量化器
    deactivate SQ
```

## 7. 距离计算时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as "用户"
    participant DC as "DCTemplate"
    participant Quant as "QuantizerTurboQuantMSE"
    participant SIMD as "SIMD操作"
    participant Accum as "累加器"

    User->>DC: query_to_code(code)
    activate DC
    
    DC->>Quant: reconstruct_8/16_components(code, i)
    activate Quant
    
    alt SIMD路径
        Quant->>SIMD: 解包索引
        activate SIMD
        SIMD->>SIMD: gather质心
        SIMD-->>Quant: SIMD向量
        deactivate SIMD
    else 标量路径
        Quant->>Quant: decode_index(code, i)
        Quant->>Quant: centroids[index]
    end
    
    Quant-->>DC: 解压分量
    deactivate Quant
    
    DC->>DC: 计算差值
    DC->>SIMD: SIMD点积
    activate SIMD
    SIMD->>Accum: 累加结果
    activate Accum
    Accum-->>SIMD: sum
    deactivate Accum
    SIMD-->>DC: 距离
    deactivate SIMD
    
    DC-->>User: distance
    deactivate DC
```

---

## 参与者说明

| 名称 | 对应代码 | 职责 |
|------|----------|------|
| 用户 | User code | 调用 FAISS API |
| ScalarQuantizer | ScalarQuantizer | 标量量化器主类 |
| train_TurboQuantMSE | training.cpp | TurboQuant 训练 |
| build_Codebook | training.cpp | 构建代码本 |
| Beta分布计算 | training.cpp | 计算 Beta 分布密度 |
| Lloyd-Max迭代 | training.cpp | Lloyd-Max 优化 |
| QuantizerTurboQuantMSE | quantizers.h | TurboQuant 量化器 |
| select_index | quantizers.h | 选择量化索引 |
| encode_index | quantizers.h | 编码索引到字节 |
| decode_index | quantizers.h | 从字节解码索引 |
| SIMD操作 | sq-*.cpp | SIMD 优化操作 |

## 关键步骤说明

**训练阶段**：
1. 检查参数有效性 (nbits ≤ 8)
2. 特殊情况处理 (d == 1)
3. 离散化 Beta 分布密度
4. Lloyd-Max 迭代优化
5. 排序并计算边界

**量化阶段**：
1. 对每个分量选择最近质心
2. 二分查找确定索引
3. 打包索引到压缩码

**反量化阶段**：
1. 从压缩码解包索引
2. 查表获取质心值
3. 组装输出向量

**SIMD 优化**：
1. 批量解包索引 (8/16个)
2. Gather 指令批量查表
3. SIMD 向量操作

## 性能优化点

1. **训练阶段**：
   - 无需训练数据
   - 解析计算代码本
   - 时间复杂度 O(nbits × iterations)

2. **量化阶段**：
   - 二分查找 O(log k)
   - 位打包减少内存
   - SIMD 并行处理

3. **反量化阶段**：
   - 查表操作 O(1)
   - SIMD Gather 加速
   - 缓存友好

4. **距离计算**：
   - 避免完全解压
   - SIMD 点积
   - 批量处理
