# TurboQuant - 高性能向量量化与检索引擎

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **RaBitQ** 和 **TurboQuant** 算法的高性能向量量化与近似最近邻 (ANN) 检索引擎，使用 Rust + C++ SIMD + RocksDB 持久化。

## 核心特性

- **RaBitQ 1-bit 量化** — 极致压缩，16KB LUT 常驻 L1 cache
- **TurboQuant 4-bit 量化** — Split LUT (8KB) L1 常驻，IVF 场景下 QPS 4.8x RaBitQ
- **IVF 索引** — 倒排文件索引，支持 RaBitQ 和 TurboQuant 两种量化
- **SQ8 精炼** — 8-bit 标量量化重排，大幅提升召回率
- **RocksDB 持久化** — HyperClockCache + PinnableSlice 零拷贝 + async_io + Ribbon Filter
- **C++ SIMD 引擎** — NEON (ARM) / AVX2 (x86) 距离计算内核
- **NNG 服务化** — 独立 Query/Write/Notify Socket，延迟隔离

## 性能基准 (SIFT Small 10K×128D)

### IVF-TurboQuant vs IVF-RaBitQ

| 方法 | QPS | Recall@10 | P50 延迟 | vs RaBitQ IVF |
|------|-----|-----------|---------|--------------|
| **TQ-IVF-256 np=8** | **6876** | 90.2% | 138μs | 2.7x QPS |
| **TQ-IVF-256 np=16** | **5150** | 97.1% | 188μs | 3.7x QPS |
| **TQ-IVF-256 np=32** | **3534** | 99.1% | 278μs | 4.8x QPS |
| TQ-IVF-64 np=8 | 3869 | 97.5% | 249μs | 1.6x QPS |
| TQ-IVF-64 np=16 | 2293 | 99.4% | 422μs | 1.7x QPS |
| TQ-IVF-64 np=32 | 1487 | 99.4% | 660μs | 2.1x QPS |
| RaBitQ IVF-256 np=8 | 2560 | 90.1% | 368μs | baseline |
| RaBitQ IVF-256 np=16 | 1386 | 96.7% | 682μs | baseline |
| RaBitQ IVF-256 np=32 | 731 | 98.4% | 1303μs | baseline |
| RaBitQ Flat+SQ8 | 3773 | 93.1% | 252μs | - |
| TQ 4bit+SQ8 (Flat) | 812 | 98.7% | 1179μs | - |
| Persisted np=8 | 2392 | 96.7% | 397μs | - |

> 测试环境: Apple M1, d=128, nb=10000, nq=100, k=10

## 快速开始

```bash
# 构建
cargo build --release

# 运行测试
cargo test --release

# SIFT Small QPS 基准测试
cargo run --release --bin siftsmall_qps
```

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
│  │    C++ SIMD 距离引擎 (NEON/AVX2)     │           │
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
│   ├── ivf.rs                # IVF 倒排索引 (2-pass scan)
│   ├── store.rs              # RocksDB 向量存储 (V1/V2 兼容)
│   ├── ivf_store.rs          # IVF 专用 RocksDB 存储
│   ├── vector_engine_ffi.rs  # C++ SIMD 引擎 FFI 桥接
│   ├── utils.rs              # 工具函数 (prefetch, next_power_of_2)
│   └── bin/
│       └── siftsmall_qps.rs  # SIFT Small QPS 基准测试
├── cpp/
│   ├── vector_engine.h       # C API 头文件
│   ├── simd_distance.h       # SIMD 距离内核 (NEON/AVX2)
│   ├── vector_query_engine.cpp # C++ 查询引擎 + RocksDB 插件
│   └── rocksdb_query_plugin.h # SST 分区/合并/压缩插件
├── build.rs                  # C++ 编译脚本
├── plan.md                   # 优化计划
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

## RocksDB 优化

| 优化 | 说明 | QPS 提升 |
|------|------|---------|
| HyperClockCache | 无锁缓存，替换 LRUCache | +10-15% |
| PinnableSlice | 零拷贝读取，避免内存分配 | +5-8% |
| async_io | 异步预取 SST 数据 | +5-10% |
| Ribbon Filter | 更省空间的布隆过滤器替代 | +3-5% |
| kBinarySearchWithFirstKey | C++ 层延迟读取数据块 | +5-10% |

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
