// PolarQuant 错误类型定义
// 使用 thiserror 库实现符合 Rust 惯例的错误类型

use thiserror::Error;

/// PolarQuant 统一错误类型
#[derive(Error, Debug)]
pub enum PolarQuantError {
    /// 无效的向量维度（必须 >= 2）
    #[error("Invalid dimension: {0}, must be >= 2")]
    InvalidDimension(usize),
    /// 无效的半径量化位数（必须在 1-16 之间）
    #[error("Invalid radius_bits: {0}, must be between 1 and 16")]
    InvalidRadiusBits(u32),
    /// 无效的角度量化位数（必须在 1-16 之间）
    #[error("Invalid angle_bits: {0}, must be between 1 and 16")]
    InvalidAngleBits(u32),
    /// 向量维度不匹配
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// IO 错误（文件读写）
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// 序列化/反序列化错误
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// 统一 Result 类型别名
pub type Result<T> = std::result::Result<T, PolarQuantError>;
