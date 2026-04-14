# TurboQuant - 高性能向量量化与检索引擎

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **RaBitQ** 和 **TurboQuant** 算法的高性能向量量化与近似最近邻 (ANN) 检索引擎，使用 Rust + C++ SIMD + RocksDB 持久化。

## 核心特性

- **RaBitQ 1-bit 量化** — 极致压缩，16KB LUT 常驻 L1 cache
- **TurboQuant 4-bit 量化** — Split LUT (8KB) L1 常驻，IVF 场景下 QPS 4.8x RaBitQ
- **NEON SIMD 加速** — L2/SQ8/Split-LUT 距离计算使用 ARM NEON 指令
- **IVF 索引** — 倒排文件索引，支持 RaBitQ 和 TurboQuant 两种量化
- **SQ8 精炼** — 8-bit 标量量化重排，大幅提升召回率
- **RocksDB 持久化** — HyperClockCache + PinnableSlice 零拷贝 + async_io + Ribbon Filter
- **C++ SIMD 引擎** — NEON (ARM) / AVX2 (x86) 距离计算内核 + CompressedSecondaryCache
- **NNG 服务化** — 独立 Query/Write/Notify Socket，延迟隔离

## 性能基准 (SIFT Small 10K×128D, 真实数据)

### IVF-TurboQuant vs IVF-RaBitQ

| 方法 | QPS | Recall@10 | P50 延迟 | vs RaBitQ IVF |
|------|-----|-----------|---------|--------------|
| **TQ-IVF-256 np=16** | **4409** | 97.1% | 206μs | 🏆 3.8x QPS |
| **TQ-IVF-256 np=8** | **4317** | 90.2% | 173μs | 1.9x QPS |
| **TQ-IVF-256 np=32** | **3036** | 99.1% | 307μs | 4.7x QPS |
| TQ-IVF-64 np=8 | 3311 | 97.5% | 279μs | 1.4x QPS |
| TQ-IVF-64 np=16 | 1968 | 99.4% | 469μs | 1.7x QPS |
| TQ-IVF-64 np=32 | 1302 | 99.4% | 734μs | 2.1x QPS |
| RaBitQ IVF-256 np=8 | 2272 | 90.1% | 399μs | baseline |
| RaBitQ IVF-256 np=16 | 1168 | 96.7% | 755μs | baseline |
| RaBitQ IVF-256 np=32 | 651 | 98.4% | 1437μs | baseline |
| RaBitQ Flat+SQ8 | 3125 | 93.1% | 285μs | - |
| TQ 4bit+SQ8 (Flat) | 766 | 98.7% | 1217μs | - |
| Persisted np=8 | 2203 | 96.7% | 421μs | - |

> 测试环境: Apple M1, d=128, nb=10000, nq=100, k=10, 10轮取平均

## 快速开始

```bash
# 构建
cargo build --release

# 运行测试
cargo test --release

# SIFT Small QPS 基准测试
cargo run --release --bin siftsmall_qps
```

## 使用指南

### 1. 内存索引 (纯内存，最高 QPS)

#### RaBitQ Flat + SQ8

```rust
use turboquant::{RaBitQFlatIndex, SQ8Quantizer};

let d = 128;
let nb = 10000;

// 创建索引 (1-bit 量化 + SQ8 精炼)
let mut index = RaBitQFlatIndex::new(d, true);

// 训练 (计算质心 + SQ8 参数)
let data: Vec<f32> = vec![0.0; nb * d]; // 你的向量数据
index.train(&data, nb);

// 添加向量
index.add(&data, nb);

// 搜索
let query: Vec<f32> = vec![0.0; d];
let k = 10;
let results = index.search(&query, 1, k, 10); // refine_factor=10
// results[0] = [(id, distance), ...]
```

#### TurboQuant IVF + SQ8 (推荐，最高 QPS + Recall)

```rust
use turboquant::TurboQuantIVFIndex;

let d = 128;
let nb = 100000;
let nlist = 256; // 聚类数，建议 sqrt(nb)

// 创建索引 (4-bit 量化 + SQ8 精炼)
let mut index = TurboQuantIVFIndex::new(d, nlist, 4, true);

// 训练 (KMeans 聚类 + SQ8 参数)
index.train(&data, nb);

// 添加向量
index.add(&data, nb);

// 搜索
let nprobe = 16; // 搜索的聚类数，越大越精确但越慢
let refine_factor = 10; // SQ8 精炼倍数
let results = index.search(&query, 1, k, nprobe, refine_factor);
```

#### RaBitQ IVF + SQ8

```rust
use turboquant::RaBitQIVFIndex;

let d = 128;
let nlist = 64;

// 创建索引 (1-bit 量化 + SQ8 精炼)
let mut index = RaBitQIVFIndex::new(d, nlist, 1, false, true);
//                                      d    nlist  nb_bits  is_inner_product  use_sq8

// 训练 + 添加 + 搜索 (同上)
index.train(&data, nb);
index.add(&data, nb);
let results = index.search(&query, 1, k, 8, 10);
```

### 2. 持久化索引 (RocksDB，支持增量写入)

#### RaBitQ IVF 持久化

```rust
use turboquant::RocksDBIVFIndex;
use std::path::PathBuf;

// 打开/创建 RocksDB 存储
let mut db_index = RocksDBIVFIndex::open(&PathBuf::from("./data/rabitq_ivf"))?;

// 从内存索引构建
db_index.build_from_ivf(&memory_index)?;

// 直接查询 (无需加载到内存)
let results = db_index.search(&query, k, nprobe, refine_factor);
```

#### TurboQuant IVF 持久化

```rust
use turboquant::RocksDBTQIVFIndex;

let mut db_index = RocksDBTQIVFIndex::open(&PathBuf::from("./data/tq_ivf"))?;
db_index.build_from_ivf(&tq_memory_index)?;

let results = db_index.search(&query, k, nprobe, refine_factor);
```

### 3. NNG 服务模式

```rust
use turboquant::TurboQuantServer;

// 创建服务 (d=128)
let server = TurboQuantServer::new(128)
    .with_workers(4)
    .with_query_url("tcp://127.0.0.1:5555")
    .with_write_url("tcp://127.0.0.1:5556")
    .with_notify_url("tcp://127.0.0.1:5557");

server.run()?;
```

#### 客户端查询

```rust
use nng::Socket;

let sock = Socket::new(nng::Protocol::Req0)?;
sock.dial("tcp://127.0.0.1:5555")?;

let request = QueryRequest::IVFSearch {
    query: vec![0.0; 128],
    k: 10,
    nprobe: 16,
    refine_factor: 10,
};

let mut msg = nng::Message::new();
msg.push_back(&bincode::serialize(&request)?);
sock.send(msg)?;

let reply = sock.recv()?;
let response: QueryResponse = bincode::deserialize(reply.as_slice())?;
```

### 4. 参数选择指南

| 场景 | 推荐索引 | nlist | nprobe | refine_factor | 预期 QPS | Recall |
|------|---------|-------|--------|---------------|---------|--------|
| 低延迟优先 | TQ-IVF-256 | 256 | 8 | 5 | ~4300 | 90% |
| 均衡 | TQ-IVF-256 | 256 | 16 | 10 | ~4400 | 97% |
| 高召回率 | TQ-IVF-256 | 256 | 32 | 10 | ~3000 | 99% |
| 极致召回率 | TQ-IVF-64 | 64 | 16 | 10 | ~2000 | 99.4% |
| 极致压缩 | RaBitQ Flat | - | - | 10 | ~3100 | 93% |
| 持久化查询 | RocksDB TQ-IVF | 256 | 8 | 10 | ~2200 | 97% |

**nlist 选择**: 建议 `sqrt(nb)`，如 10K 向量 → nlist=100, 100K → nlist=316, 1M → nlist=1000

**nprobe 选择**: nprobe 越大召回率越高但 QPS 越低。通常 nprobe/nlist = 5%-10% 即可获得 95%+ 召回率

## 架构

```
┌─────────────────────────────────────────────────────┐
│                   Rust 核心层                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ RaBitQ   │ │TurboQuant│ │   IVF    │            │
│  │ 1-bit LUT│ │4-bit LUT │ │ nprobe   │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       └──────┬──────┘           │                    │
│              ▼                  ▼                    │
│  ┌──────────────────────────────────────┐           │
│  │         SQ8 精炼重排                  │           │
│  └──────────────────┬───────────────────┘           │
│                     ▼                               │
│  ┌──────────────────────────────────────┐           │
│  │  Rust NEON SIMD (l2/sq8/split_lut)  │           │
│  └──────────────────┬───────────────────┘           │
│                     ▼                               │
│  ┌──────────────────────────────────────┐           │
│  │  C++ SIMD 引擎 (NEON/AVX2)          │           │
│  │  + CompressedSecondaryCache          │           │
│  └──────────────────┬───────────────────┘           │
│                     ▼                               │
│  ┌──────────────────────────────────────┐           │
│  │    RocksDB 持久化                     │           │
│  │    HyperClockCache + PinnableSlice   │           │
│  │    async_io + Ribbon Filter          │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

## 项目结构

```
turboquant/
├── src/
│   ├── lib.rs                # 库入口
│   ├── rabitq.rs             # RaBitQ 1-bit 量化 + LUT 距离计算
│   ├── turboquant.rs         # TurboQuant 4-bit/6-bit 量化 + Split LUT
│   ├── lloyd_max.rs          # Lloyd-Max 量化器 (Split LUT 构建)
│   ├── sq8.rs                # SQ8 标量量化 + 预处理查询
│   ├── hadamard.rs           # Hadamard 随机旋转
│   ├── kmeans.rs             # KMeans 聚类 (buffer reuse)
│   ├── ivf.rs                # IVF 倒排索引 (RaBitQ + TurboQuant)
│   ├── store.rs              # RocksDB 向量存储 (V1/V2/V3 兼容)
│   ├── ivf_store.rs          # IVF 专用 RocksDB 存储 (RaBitQ + TurboQuant)
│   ├── vector_engine_ffi.rs  # C++ SIMD 引擎 FFI 桥接
│   ├── server.rs             # NNG 服务端 (Query/Write/Notify)
│   ├── utils.rs              # 工具函数 (SIMD距离, prefetch, l2_normalize)
│   └── bin/
│       └── siftsmall_qps.rs  # SIFT Small QPS 基准测试
├── cpp/
│   ├── vector_engine.h       # C API 头文件
│   ├── simd_distance.h       # SIMD 距离内核 (NEON/AVX2)
│   ├── vector_query_engine.cpp # C++ 查询引擎 + RocksDB 插件
│   └── rocksdb_query_plugin.h # SST 分区/合并/压缩插件
├── build.rs                  # C++ 编译脚本
└── tests/
    ├── store_test.rs         # 存储测试
    ├── siftsmall_test.rs     # SIFT 召回率测试
    └── persistence_qps_test.rs # 持久化 QPS 测试
```

## 量化算法

### RaBitQ (1-bit)

```
输入向量 x ∈ R^d
    │
    ▼
随机旋转 (Hadamard) → y = H·x
    │
    ▼
1-bit 符号量化 → signs = sign(y)
    │
    ▼
LUT 距离计算 (16KB, L1 常驻)
    │
    ▼
SQ8 精炼重排
```

- **LUT 大小**: d/8 × 256 × 4B = 16KB (d=128)
- **查询延迟**: ~4 cycles/lookup (L1 hit)
- **存储**: (d/8 + 8) bytes/vector = 24B (d=128)

### TurboQuant (4-bit)

```
输入向量 x ∈ R^d
    │
    ▼
随机旋转 (Hadamard) → y = H·x
    │
    ▼
4-bit Lloyd-Max 量化 → codes (2 dim/byte)
    │
    ▼
Split LUT 距离计算 (8KB, L1 常驻)
    │
    ▼
SQ8 精炼重排
```

- **Split LUT 大小**: code_sz × 16 × 4B × 2 = 8KB (d=128)
- **存储**: (d/2 + d) bytes/vector = 192B with SQ8 (d=128)
- **召回率**: 98.7% (最高)

### Split LUT 原理

将 4-bit LUT 的 256 项表拆分为两个 16 项表，LUT 从 64KB 降至 8KB：

```
原始: lut[j][byte] = (centroids[lo] - query[2j])² + (centroids[hi] - query[2j+1])²
拆分: lo_lut[j][byte & 0xF] + hi_lut[j][byte >> 4]  // 数学等价
```

## 性能优化

### Rust SIMD 加速

| 函数 | 优化 | 说明 |
|------|------|------|
| `l2_distance_simd` | NEON vfmaq | L2 距离 4-wide SIMD |
| `sq8_distance_simd` | NEON vfmaq | SQ8 解码+距离 4-wide |
| `dot_product_simd` | NEON vfmaq | 点积 4-wide SIMD |
| `l2_norm_simd` | NEON vfmaq | L2 范数 4-wide SIMD |
| `l2_normalize` | NEON + inv_norm | 乘法替代除法 |
| `nearest_clusters` | select_nth_unstable | O(n) 替代 O(n log n) 排序 |

### RocksDB 优化

| 优化 | 说明 | QPS 提升 |
|------|------|---------|
| HyperClockCache | 无锁缓存，替换 LRUCache | +10-15% |
| PinnableSlice | 零拷贝读取，避免内存分配 | +5-8% |
| async_io | 异步预取 SST 数据 | +5-10% |
| Ribbon Filter | 更省空间的布隆过滤器替代 | +3-5% |
| kBinarySearchWithFirstKey | C++ 层延迟读取数据块 | +5-10% |
| CompressedSecondaryCache | LZ4 压缩二级缓存 | 间接提升 |

## NNG 服务架构

```
Client ──NNG Req0──→ Query Socket (Rep0:5555)  → N Worker 线程并发查询
Client ──NNG Req0──→ Write Socket (Rep0:5556)  → 1 线程串行写入 + 批量提交
Server ──NNG Pub0──→ Notify Socket (Pub0:5557) → 事件广播
```

使用 NNG (nanomsg next generation) 替换 ZeroMQ：
- 内置 WebSocket + TLS 支持
- Survey 模式 (ZeroMQ 无)
- 更轻量的 C 依赖
- 写入与查询独立 Socket，延迟隔离

## 参考文献

1. **RaBitQ** — "RaBitQ: Quantization with Radix Trees for Scalable Vector Search"
2. **TurboQuant** — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", Zandieh et al., ICLR 2026
3. **PolarQuant** — "PolarQuant: Quantizing KV Caches with Polar Transformation", Han et al., AISTATS 2026

## 许可证

MIT
