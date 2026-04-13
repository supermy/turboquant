# TurboQuant 时序图

## 1. TurboQuant Flat + SQ8 搜索时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Index as TurboQuantFlatIndex
    participant Hadamard as HadamardRotation
    participant LloydMax as LloydMaxQuantizer
    participant SQ8 as SQ8Quantizer
    participant Heap as BinaryHeap

    Client->>Index: search(queries, n, k, refine_factor)
    
    rect rgb(240, 248, 255)
        Note over Index: 步骤1: 归一化查询向量
        Index->>Index: l2_normalize(queries)
    end
    
    rect rgb(255, 250, 240)
        Note over Index,Hadamard: 步骤2: Hadamard旋转
        Index->>Hadamard: apply_batch(queries)
        Hadamard->>Hadamard: 3轮随机符号翻转 + FWHT
        Hadamard-->>Index: rotated_queries
    end
    
    rect rgb(240, 255, 240)
        Note over Index,Heap: 步骤3: 粗排 (Lloyd-Max距离)
        loop 每个查询
            Index->>Heap: 创建 k1 大小的堆
            loop 每个编码向量
                Index->>LloydMax: compute_distance(code, query)
                LloydMax-->>Index: dist
                alt 堆未满 或 dist更小
                    Index->>Heap: push((dist, id))
                end
            end
        end
    end
    
    rect rgb(255, 240, 245)
        Note over Index,Heap: 步骤4: 精排 (SQ8距离)
        loop 每个候选
            Index->>SQ8: compute_distance(sq8_code, query)
            SQ8-->>Index: refined_dist
            alt 堆未满 或 dist更小
                Index->>Heap: push((refined_dist, id))
            end
        end
    end
    
    Index-->>Client: Vec<Vec<(id, dist)>>
```

## 2. RaBitQ IVF + SQ8 搜索时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Index as RaBitQIVFIndex
    participant KMeans as KMeans
    participant Codec as RaBitQCodec
    participant SQ8 as SQ8Quantizer
    participant Heap as BinaryHeap

    Client->>Index: search(queries, n, k, nprobe, refine_factor)
    
    rect rgb(240, 248, 255)
        Note over Index,KMeans: 步骤1: 找到最近的nprobe个聚类
        Index->>KMeans: nearest_clusters(query, nprobe)
        KMeans-->>Index: Vec<(dist, cluster_id)>
    end
    
    rect rgb(255, 250, 240)
        Note over Index,Heap: 步骤2: 在聚类中搜索 (RaBitQ距离)
        loop 每个查询
            Index->>Heap: 创建 k1 大小的堆
            loop 每个最近聚类
                Index->>Index: compute_query_factors(query, centroid)
                loop 每个聚类中的向量
                    Index->>Codec: compute_distance(code, query_fac)
                    Codec-->>Index: dist
                    alt 堆未满 或 dist更小
                        Index->>Heap: push((dist, id))
                    end
                end
            end
        end
    end
    
    rect rgb(240, 255, 240)
        Note over Index,Heap: 步骤3: 精排 (SQ8距离)
        loop 每个候选
            Index->>Index: 查找候选所属聚类
            Index->>SQ8: compute_distance(sq8_code, query)
            SQ8-->>Index: refined_dist
            alt 堆未满 或 dist更小
                Index->>Heap: push((refined_dist, id))
            end
        end
    end
    
    Index-->>Client: Vec<Vec<(id, dist)>>
```

## 3. 向量编码时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Index as Index
    participant Hadamard as HadamardRotation
    participant LloydMax as LloydMaxQuantizer
    participant SQ8 as SQ8Quantizer

    Client->>Index: add(data, n)
    
    rect rgb(240, 248, 255)
        Note over Index: 步骤1: L2归一化
        Index->>Index: l2_normalize(data)
    end
    
    rect rgb(255, 250, 240)
        Note over Index,Hadamard: 步骤2: Hadamard旋转
        Index->>Hadamard: apply_batch(data)
        Hadamard->>Hadamard: 第1轮: 随机符号 + FWHT
        Hadamard->>Hadamard: 第2轮: 随机符号 + FWHT
        Hadamard->>Hadamard: 第3轮: 随机符号 + FWHT
        Hadamard->>Hadamard: 缩放: * 1/(d*√d)
        Hadamard-->>Index: rotated_data
    end
    
    rect rgb(240, 255, 240)
        Note over Index,LloydMax: 步骤3: Lloyd-Max编码
        loop 每个向量
            Index->>LloydMax: select_index(x[i])
            LloydMax-->>Index: idx
            Index->>LloydMax: encode_index(idx, code, i)
        end
    end
    
    rect rgb(255, 240, 245)
        Note over Index,SQ8: 步骤4: SQ8编码 (可选)
        Index->>SQ8: encode(x, code)
        SQ8->>SQ8: 线性映射到 [0, 255]
    end
    
    Index-->>Client: 编码完成
```

## 4. Lloyd-Max 量化器初始化时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Q as LloydMaxQuantizer

    Client->>Q: new(d, nbits)
    
    rect rgb(240, 248, 255)
        Note over Q: 步骤1: 初始化参数
        Q->>Q: k = 2^nbits
        Q->>Q: centroids = vec![0.0; k]
        Q->>Q: boundaries = vec![0.0; k-1]
    end
    
    rect rgb(255, 250, 240)
        Note over Q: 步骤2: Lloyd-Max迭代
        Q->>Q: 计算Beta分布权重
        Q->>Q: 初始分割: 等权重分割
        
        loop 最多100次迭代
            Q->>Q: 更新边界: boundary = (centroid[i] + centroid[i+1]) / 2
            Q->>Q: 更新分割点
            Q->>Q: 更新中心: 加权均值
            Q->>Q: 检查收敛: max_delta < 1e-8
        end
    end
    
    rect rgb(240, 255, 240)
        Note over Q: 步骤3: 计算边界
        Q->>Q: boundaries[i] = (centroids[i] + centroids[i+1]) / 2
    end
    
    Q-->>Client: 量化器就绪
```

## 5. Hadamard 旋转时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant H as HadamardRotation
    participant FWHT as FWHT

    Client->>H: apply(x)
    
    rect rgb(240, 248, 255)
        Note over H: 准备缓冲区
        H->>H: buf = [0.0; d_out]
        H->>H: 复制输入并应用第一轮符号
    end
    
    rect rgb(255, 250, 240)
        Note over H,FWHT: 第1轮: 符号翻转 + FWHT
        H->>H: buf[i] *= signs1[i]
        H->>FWHT: fwht_inplace(buf)
        FWHT->>FWHT: 蝶形运算 O(n log n)
        FWHT-->>H: buf
    end
    
    rect rgb(240, 255, 240)
        Note over H,FWHT: 第2轮: 符号翻转 + FWHT
        H->>H: buf[i] *= signs2[i]
        H->>FWHT: fwht_inplace(buf)
        FWHT-->>H: buf
    end
    
    rect rgb(255, 240, 245)
        Note over H,FWHT: 第3轮: 符号翻转 + FWHT
        H->>H: buf[i] *= signs3[i]
        H->>FWHT: fwht_inplace(buf)
        FWHT-->>H: buf
    end
    
    rect rgb(248, 248, 255)
        Note over H: 缩放
        H->>H: buf[i] *= scale (1/(d*√d))
    end
    
    H-->>Client: rotated vector
```

## 6. 模块依赖关系图

```mermaid
graph TB
    subgraph 核心模块
        utils[utils.rs<br/>工具函数]
        hadamard[hadamard.rs<br/>Hadamard旋转]
        lloyd_max[lloyd_max.rs<br/>Lloyd-Max量化]
        sq8[sq8.rs<br/>SQ8量化]
    end
    
    subgraph 索引模块
        turboquant[turboquant.rs<br/>TurboQuant Flat]
        rabitq[rabitq.rs<br/>RaBitQ Flat]
        kmeans[kmeans.rs<br/>KMeans聚类]
        ivf[ivf.rs<br/>RaBitQ IVF]
    end
    
    subgraph 应用
        main[main.rs<br/>性能测试]
    end
    
    utils --> hadamard
    utils --> lloyd_max
    utils --> sq8
    utils --> kmeans
    utils --> turboquant
    utils --> rabitq
    
    hadamard --> turboquant
    lloyd_max --> turboquant
    sq8 --> turboquant
    sq8 --> rabitq
    sq8 --> ivf
    
    rabitq --> ivf
    kmeans --> ivf
    
    turboquant --> main
    rabitq --> main
    ivf --> main
```

## 7. 数据流图

```mermaid
flowchart TB
    subgraph 输入
        data[原始向量<br/>FP32]
        query[查询向量<br/>FP32]
    end
    
    subgraph TurboQuant编码
        norm1[L2归一化]
        rot1[Hadamard旋转<br/>3轮FWHT]
        lm[Lloyd-Max编码<br/>4/6/8-bit]
        sq8_1[SQ8编码<br/>可选]
    end
    
    subgraph RaBitQ编码
        centroid[计算质心]
        sign[符号位编码<br/>1-bit]
        factor[计算因子<br/>dp_multiplier]
        sq8_2[SQ8编码<br/>可选]
    end
    
    subgraph 搜索
        coarse[粗排<br/>量化距离]
        refine[精排<br/>SQ8距离]
        result[返回Top-K]
    end
    
    data --> norm1 --> rot1 --> lm --> sq8_1
    data --> centroid --> sign --> factor --> sq8_2
    
    query --> norm1
    query --> coarse
    
    lm --> coarse
    sign --> coarse
    sq8_1 --> refine
    sq8_2 --> refine
    
    coarse --> refine --> result
```
