// PolarQuant 配置模块
// 使用 Builder 模式构建配置，支持链式调用

use serde::{Deserialize, Serialize};

use crate::error::{PolarQuantError, Result};

/// PolarQuant 量化配置
///
/// # 字段说明
/// - `dimension`: 向量维度 d（必须 >= 2）
/// - `radius_bits`: 半径量化位数（默认 8，范围 1-16）
/// - `angle_bits`: 角度量化位数（默认 4，范围 1-16）
/// - `use_hadamard`: 是否使用 Hadamard 旋转替代随机正交矩阵（默认 false）
/// - `seed`: 随机种子，确保可复现性（默认 42）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarQuantConfig {
    pub dimension: usize,
    pub radius_bits: u32,
    pub angle_bits: u32,
    pub use_hadamard: bool,
    pub seed: u64,
}

impl PolarQuantConfig {
    /// 使用默认参数创建配置
    pub fn new(dimension: usize) -> Result<Self> {
        Self::builder(dimension).build()
    }

    /// 创建 Builder 实例，开始链式配置
    pub fn builder(dimension: usize) -> PolarQuantConfigBuilder {
        PolarQuantConfigBuilder {
            dimension,
            radius_bits: 8,
            angle_bits: 4,
            use_hadamard: false,
            seed: 42,
        }
    }

    /// 校验配置参数合法性
    pub fn validate(&self) -> Result<()> {
        if self.dimension < 2 {
            return Err(PolarQuantError::InvalidDimension(self.dimension));
        }
        if self.radius_bits < 1 || self.radius_bits > 16 {
            return Err(PolarQuantError::InvalidRadiusBits(self.radius_bits));
        }
        if self.angle_bits < 1 || self.angle_bits > 16 {
            return Err(PolarQuantError::InvalidAngleBits(self.angle_bits));
        }
        Ok(())
    }
}

/// PolarQuant 配置 Builder（链式调用模式）
///
/// # 示例
/// ```ignore
/// let config = PolarQuantConfig::builder(256)
///     .radius_bits(8)
///     .angle_bits(4)
///     .use_hadamard(false)
///     .seed(42)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct PolarQuantConfigBuilder {
    dimension: usize,
    radius_bits: u32,
    angle_bits: u32,
    use_hadamard: bool,
    seed: u64,
}

impl PolarQuantConfigBuilder {
    /// 设置半径量化位数
    pub fn radius_bits(mut self, bits: u32) -> Self {
        self.radius_bits = bits;
        self
    }

    /// 设置角度量化位数
    pub fn angle_bits(mut self, bits: u32) -> Self {
        self.angle_bits = bits;
        self
    }

    /// 是否使用 Hadamard 旋转（O(d log d) 快速变换，替代 O(d²) 随机矩阵乘法）
    pub fn use_hadamard(mut self, use_hadamard: bool) -> Self {
        self.use_hadamard = use_hadamard;
        self
    }

    /// 设置随机种子（确保同一种子产生相同的旋转矩阵）
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// 构建最终配置，自动校验参数合法性
    pub fn build(self) -> Result<PolarQuantConfig> {
        let config = PolarQuantConfig {
            dimension: self.dimension,
            radius_bits: self.radius_bits,
            angle_bits: self.angle_bits,
            use_hadamard: self.use_hadamard,
            seed: self.seed,
        };
        config.validate()?;
        Ok(config)
    }
}
