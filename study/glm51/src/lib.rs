// PolarQuant: 基于极坐标变换的无损量化算法 (Rust 实现)
//
// 基于 Google Research 论文 "PolarQuant: Quantizing KV Caches with Polar Transformation"
// 核心思想：通过随机正交旋转归一化分布 → 极坐标变换 → Lloyd-Max 量化
// 实现零元数据开销的高维向量近无损压缩

// 子模块声明
pub mod batch;      // 批量处理模块
pub mod config;     // 配置模块（Builder 模式）
pub mod error;      // 错误类型模块
pub mod quantizer;  // 核心量化器模块
pub mod utils;      // 数学工具函数模块

// 公共 API 导出
pub use batch::{BatchErrorMetrics, PolarQuantBatch};
pub use config::PolarQuantConfig;
pub use error::{PolarQuantError, Result};
pub use quantizer::{CompressedVector, ErrorMetrics, PolarQuant};
pub use utils::{
    beta_distribution_params, cartesian_to_polar, compute_lloyd_max_centroids, hadamard_rotation,
    lloyd_max_dequantize, lloyd_max_quantize, polar_to_cartesian, random_orthogonal_matrix,
};
