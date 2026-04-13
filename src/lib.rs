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

pub mod utils;
pub mod hadamard;
pub mod lloyd_max;
pub mod sq8;
pub mod turboquant;
pub mod rabitq;
pub mod kmeans;
pub mod ivf;
pub mod sift;
pub mod store;

pub use utils::*;
pub use hadamard::HadamardRotation;
pub use lloyd_max::LloydMaxQuantizer;
pub use sq8::SQ8Quantizer;
pub use turboquant::TurboQuantFlatIndex;
pub use rabitq::{RaBitQCodec, RaBitQFlatIndex};
pub use kmeans::KMeans;
pub use ivf::RaBitQIVFIndex;
pub use sift::SiftSmallDataset;
pub use store::{VectorStore, IndexMeta, IndexType, StoreStats};
