//! TurboQuant: 高性能向量量化库
//!
//! 本库实现了两种主要的向量量化方法:
//! - **TurboQuant**: 基于 Hadamard 旋转和 Lloyd-Max 量化的快速向量压缩
//! - **RaBitQ**: 随机二进制量化，支持 IVF 索引加速
//!
//! # 核心特性
//! - 无需训练的 TurboQuant (4-bit, 6-bit, 8-bit)
//! - 极致压缩的 RaBitQ (1-bit)
//! - SQ8 refinement 两阶段搜索，提升召回率
//! - IVF 聚类索引加速大规模搜索
//!
//! # 召回率测试结果
//! | 方法 | 存储 | Recall@10 |
//! |------|------|-----------|
//! | TurboQuant 4-bit | 64B | 86.3% |
//! | TurboQuant 6-bit | 96B | 93.4% |
//! | TurboQuant 4-bit + SQ8 | 192B | 98.5% |
//! | RaBitQ 1-bit + SQ8 | 152B | 98.2% |
//! | RaBitQ IVF + SQ8 | 152B | 98.7% |

pub mod hadamard;
pub mod ivf;
pub mod ivf_store;
pub mod kmeans;
pub mod lloyd_max;
pub mod rabitq;
pub mod sift;
pub mod sq8;
pub mod store;
pub mod turboquant;
pub mod utils;
pub mod vector_engine_ffi;

#[cfg(feature = "nng")]
pub mod server;

pub use hadamard::HadamardRotation;
pub use ivf::RaBitQIVFIndex;
pub use ivf::TurboQuantIVFIndex;
pub use ivf_store::RocksDBIVFIndex;
pub use ivf_store::RocksDBTQIVFIndex;
pub use kmeans::KMeans;
pub use lloyd_max::LloydMaxQuantizer;
pub use rabitq::{RaBitQCodec, RaBitQFlatIndex};
pub use sift::SiftSmallDataset;
pub use sq8::SQ8Quantizer;
pub use store::{IndexMeta, IndexType, StoreStats, VectorStore};
pub use turboquant::TurboQuantFlatIndex;
pub use utils::*;
pub use vector_engine_ffi::VectorEngine;

#[cfg(feature = "nng")]
pub use server::MemoryIndex;
#[cfg(feature = "nng")]
pub use server::PersistedIndex;
#[cfg(feature = "nng")]
pub use server::TurboQuantServer;
#[cfg(feature = "nng")]
pub use server::VectorEngineService;
