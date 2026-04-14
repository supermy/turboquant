// PolarQuant 核心量化器实现
//
// 完整的压缩/解压管线：
//   压缩: x → 随机旋转 → 极坐标变换 → Lloyd-Max 量化 → 压缩索引
//   解压: 压缩索引 → 反量化 → 极坐标逆变换 → 逆旋转 → x_recon
//
// 核心优势：无需存储 per-vector 的 scale/zero-point 元数据，
// 因为 Beta 分布参数可从维度 d 解析计算。

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::config::PolarQuantConfig;
use crate::error::{PolarQuantError, Result};
use crate::utils::{
    beta_distribution_params, cartesian_to_polar, compute_lloyd_max_centroids, hadamard_rotation,
    lloyd_max_dequantize, lloyd_max_quantize, polar_to_cartesian, random_orthogonal_matrix,
    vector_norm,
};

/// 压缩后的向量表示
///
/// 仅存储量化索引，无需存储 scale/zero-point 元数据。
/// - `radius_idx`: 半径量化索引
/// - `angle_indices`: d-1 个角度量化索引
/// - `original_norm`: 原始向量的 L2 范数（用于评估，非必需）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    pub radius_idx: u32,
    pub angle_indices: Vec<u32>,
    pub original_norm: f64,
}

impl CompressedVector {
    /// 计算压缩后的比特数
    pub fn size_bits(&self, radius_bits: u32, angle_bits: u32) -> usize {
        radius_bits as usize + self.angle_indices.len() * angle_bits as usize
    }

    /// 计算压缩后的字节数（向上取整）
    pub fn size_bytes(&self, radius_bits: u32, angle_bits: u32) -> usize {
        self.size_bits(radius_bits, angle_bits).div_ceil(8)
    }
}

/// PolarQuant 量化器
///
/// 核心数据结构，包含：
/// - 旋转矩阵（随机正交或 Hadamard）
/// - 预计算的 Lloyd-Max 角度质心
/// - 半径量化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarQuant {
    /// 量化配置
    pub config: PolarQuantConfig,
    /// 随机正交旋转矩阵（Hadamard 模式下为 None）
    #[serde(with = "matrix_serde")]
    pub rotation_matrix: Option<DMatrix<f64>>,
    /// 预计算的 Lloyd-Max 角度量化质心
    pub angle_centroids: Vec<f64>,
    /// 半径量化级数 = 2^radius_bits
    pub radius_levels: u32,
    /// 半径量化下界（对数尺度）
    pub radius_min: f64,
    /// 半径量化上界（对数尺度）
    pub radius_max: f64,
}

/// nalgebra DMatrix 的自定义序列化/反序列化
/// 将矩阵序列化为 (行数, 列数, 数据) 元组
mod matrix_serde {
    use nalgebra::DMatrix;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(
        matrix: &Option<DMatrix<f64>>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match matrix {
            Some(m) => {
                let (nrows, ncols) = m.shape();
                let data: Vec<f64> = m.iter().cloned().collect();
                serializer.serialize_some(&(nrows, ncols, data))
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<Option<DMatrix<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<(usize, usize, Vec<f64>)> = Option::deserialize(deserializer)?;
        Ok(opt.map(|(nrows, ncols, data)| DMatrix::from_vec(nrows, ncols, data)))
    }
}

impl PolarQuant {
    /// 创建新的 PolarQuant 量化器
    ///
    /// 初始化步骤：
    /// 1. 根据配置生成旋转矩阵（随机正交或 Hadamard）
    /// 2. 根据 Beta(d/2, d/2) 分布预计算 Lloyd-Max 质心
    /// 3. 设置半径量化参数
    pub fn new(config: PolarQuantConfig) -> Result<Self> {
        config.validate()?;

        // 生成旋转矩阵（Hadamard 模式下不需要显式存储矩阵）
        let rotation_matrix = if config.use_hadamard {
            None
        } else {
            Some(random_orthogonal_matrix(config.dimension, config.seed))
        };

        // 预计算 Lloyd-Max 角度量化质心
        let (alpha, beta_param) = beta_distribution_params(config.dimension);
        let angle_centroids = compute_lloyd_max_centroids(alpha, beta_param, config.angle_bits, 100);
        let radius_levels = 1u32 << config.radius_bits;

        Ok(Self {
            config,
            rotation_matrix,
            angle_centroids,
            radius_levels,
            radius_min: 1e-3,
            radius_max: 10.0,
        })
    }

    /// 设置半径量化上界（链式调用）
    pub fn with_radius_max(mut self, radius_max: f64) -> Self {
        self.radius_max = radius_max;
        self
    }

    /// 正向旋转：x → R·x（随机正交矩阵乘法或 Hadamard 变换）
    fn rotate(&self, x: &[f64]) -> Vec<f64> {
        if self.config.use_hadamard {
            hadamard_rotation(x)
        } else {
            let x_vec = nalgebra::DVector::from(x.to_vec());
            let result = self.rotation_matrix.as_ref().unwrap() * &x_vec;
            result.iter().cloned().collect()
        }
    }

    /// 逆向旋转：x → R^T·x（Hadamard 自逆，随机矩阵用转置）
    fn inverse_rotate(&self, x: &[f64]) -> Vec<f64> {
        if self.config.use_hadamard {
            hadamard_rotation(x) // Hadamard 矩阵自逆：H^T = H
        } else {
            let x_vec = nalgebra::DVector::from(x.to_vec());
            let result = self.rotation_matrix.as_ref().unwrap().transpose() * &x_vec;
            result.iter().cloned().collect()
        }
    }

    /// 半径量化（对数尺度均匀量化）
    ///
    /// 在对数空间均匀量化，更好地覆盖不同量级的半径值。
    /// 返回 (量化索引, 反量化后的半径值)
    fn quantize_radius(&self, r: f64) -> (u32, f64) {
        if r < 1e-10 {
            return (0, 0.0);
        }

        // 对数空间归一化到 [0, 1]
        let log_r = r.ln().max(1e-10f64.ln());
        let log_min = self.radius_min.ln();
        let log_max = self.radius_max.ln();

        let log_r_norm = ((log_r - log_min) / (log_max - log_min)).clamp(0.0, 1.0);

        // 均匀量化
        let idx = (log_r_norm * (self.radius_levels - 1) as f64).floor() as u32;
        let idx = idx.min(self.radius_levels - 1);

        // 反量化：从索引恢复半径
        let log_r_recon =
            log_min + idx as f64 / (self.radius_levels - 1) as f64 * (log_max - log_min);
        let r_recon = log_r_recon.exp();

        (idx, r_recon)
    }

    /// 半径反量化：从索引恢复半径值
    fn dequantize_radius(&self, idx: u32) -> f64 {
        let log_min = self.radius_min.ln();
        let log_max = self.radius_max.ln();
        let log_r =
            log_min + idx as f64 / (self.radius_levels - 1) as f64 * (log_max - log_min);
        log_r.exp()
    }

    /// 压缩向量
    ///
    /// 完整压缩管线：
    /// 1. 随机旋转（分布归一化）
    /// 2. 极坐标变换（分离半径和角度）
    /// 3. 半径对数量化
    /// 4. 角度 Lloyd-Max 量化
    pub fn compress(&self, x: &[f64]) -> Result<CompressedVector> {
        // 维度检查
        if x.len() != self.config.dimension {
            return Err(PolarQuantError::DimensionMismatch {
                expected: self.config.dimension,
                actual: x.len(),
            });
        }

        // 保存原始范数（用于评估）
        let original_norm = vector_norm(x);

        // 步骤 1: 随机旋转
        let x_rotated = self.rotate(x);

        // 步骤 2: 极坐标变换
        let (r, angles) = cartesian_to_polar(&x_rotated);

        // 步骤 3: 半径量化
        let (radius_idx, _) = self.quantize_radius(r);

        // 步骤 4: 角度量化
        // 将 [0, π] 归一化到 [0, 1]，然后 Lloyd-Max 量化
        let angles_normalized: Vec<f64> = angles
            .iter()
            .map(|&a| (a / std::f64::consts::PI).clamp(0.0, 1.0))
            .collect();
        let angle_indices = lloyd_max_quantize(&angles_normalized, &self.angle_centroids);

        Ok(CompressedVector {
            radius_idx,
            angle_indices,
            original_norm,
        })
    }

    /// 解压向量
    ///
    /// 完整解压管线（compress 的逆过程）：
    /// 1. 半径反量化
    /// 2. 角度反量化 + 反归一化
    /// 3. 极坐标逆变换
    /// 4. 逆旋转
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f64> {
        // 步骤 1: 半径反量化
        let r = self.dequantize_radius(compressed.radius_idx);

        // 步骤 2: 角度反量化 + 反归一化 [0,1] → [0,π]
        let angles_normalized =
            lloyd_max_dequantize(&compressed.angle_indices, &self.angle_centroids);
        let angles: Vec<f64> = angles_normalized
            .iter()
            .map(|&a| a * std::f64::consts::PI)
            .collect();

        // 步骤 3: 极坐标逆变换
        let x_rotated = polar_to_cartesian(r, &angles);

        // 步骤 4: 逆旋转
        self.inverse_rotate(&x_rotated)
    }

    /// 计算理论压缩比
    ///
    /// 压缩比 = 原始比特数 / 压缩后比特数
    /// 原始: d × 32 (f32) 或 d × 64 (f64)
    /// 压缩: radius_bits + (d-1) × angle_bits
    pub fn compression_ratio(&self) -> f64 {
        let original_bits = self.config.dimension as f64 * 32.0;
        let compressed_bits =
            self.config.radius_bits as f64 + (self.config.dimension - 1) as f64 * self.config.angle_bits as f64;
        original_bits / compressed_bits
    }

    /// 计算重建误差指标
    ///
    /// 返回多种误差度量：
    /// - MSE: 均方误差
    /// - RMSE: 均方根误差
    /// - cosine_similarity: 余弦相似度
    /// - relative_error: 相对误差 (||x - x'|| / ||x||)
    pub fn compute_error(&self, x: &[f64], x_reconstructed: &[f64]) -> ErrorMetrics {
        let n = x.len().min(x_reconstructed.len());

        // 均方误差 MSE
        let mse: f64 = x[..n]
            .iter()
            .zip(x_reconstructed[..n].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;

        let rmse = mse.sqrt();

        let norm_x = vector_norm(x);
        let norm_recon = vector_norm(x_reconstructed);

        // 余弦相似度
        let cosine_similarity = if norm_x > 1e-10 && norm_recon > 1e-10 {
            x[..n]
                .iter()
                .zip(x_reconstructed[..n].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                / (norm_x * norm_recon)
        } else {
            0.0
        };

        // 相对误差
        let relative_error = if norm_x > 1e-10 {
            let diff_norm: f64 = x[..n]
                .iter()
                .zip(x_reconstructed[..n].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            diff_norm / norm_x
        } else {
            0.0
        };

        ErrorMetrics {
            mse,
            rmse,
            cosine_similarity,
            relative_error,
        }
    }

    /// 将量化器保存到二进制文件（bincode 序列化）
    pub fn save(&self, path: &str) -> Result<()> {
        let encoded =
            bincode::serialize(self).map_err(|e| PolarQuantError::Serialization(e.to_string()))?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// 从二进制文件加载量化器
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let decoded: PolarQuant =
            bincode::deserialize(&data).map_err(|e| PolarQuantError::Serialization(e.to_string()))?;
        Ok(decoded)
    }
}

/// 重建误差指标
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// 均方误差
    pub mse: f64,
    /// 均方根误差
    pub rmse: f64,
    /// 余弦相似度
    pub cosine_similarity: f64,
    /// 相对误差
    pub relative_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 创建测试用量化器
    fn make_test_quantizer(dim: usize) -> PolarQuant {
        let config = PolarQuantConfig::new(dim).unwrap();
        PolarQuant::new(config).unwrap()
    }

    /// 测试量化器初始化
    #[test]
    fn test_initialization() {
        let pq = make_test_quantizer(16);
        assert_eq!(pq.config.dimension, 16);
        assert!(pq.rotation_matrix.is_some());
        assert!(!pq.angle_centroids.is_empty());
    }

    /// 测试压缩/解压基本流程
    #[test]
    fn test_compress_decompress() {
        let pq = make_test_quantizer(16);
        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();

        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);

        assert_eq!(x_recon.len(), 16);
    }

    /// 测试重建质量（归一化向量）
    #[test]
    fn test_reconstruction_quality() {
        let config = PolarQuantConfig::builder(16).seed(42).build().unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();
        let norm = vector_norm(&x);
        let x_norm: Vec<f64> = x.iter().map(|&v| v / norm).collect();

        let compressed = pq.compress(&x_norm).unwrap();
        let x_recon = pq.decompress(&compressed);

        let errors = pq.compute_error(&x_norm, &x_recon);
        assert!(
            errors.cosine_similarity > 0.8,
            "cosine similarity too low: {}",
            errors.cosine_similarity
        );
    }

    /// 测试维度不匹配错误
    #[test]
    fn test_dimension_mismatch() {
        let pq = make_test_quantizer(16);
        let x = vec![1.0; 10];
        let result = pq.compress(&x);
        assert!(result.is_err());
    }

    /// 测试压缩比计算
    #[test]
    fn test_compression_ratio() {
        let pq = make_test_quantizer(16);
        let ratio = pq.compression_ratio();
        assert!(ratio > 1.0);

        // 期望: (16 * 32) / (8 + 15 * 4) = 512 / 68 ≈ 7.53
        let expected = (16.0 * 32.0) / (8.0 + 15.0 * 4.0);
        assert_relative_eq!(ratio, expected, epsilon = 1e-5);
    }

    /// 测试误差指标计算
    #[test]
    fn test_error_metrics() {
        let pq = make_test_quantizer(16);
        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();
        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);

        let errors = pq.compute_error(&x, &x_recon);
        assert!(errors.mse >= 0.0);
        assert!(errors.rmse >= 0.0);
        assert!(errors.cosine_similarity >= -1.0 && errors.cosine_similarity <= 1.0);
    }

    /// 测试零向量处理
    #[test]
    fn test_zero_vector() {
        let pq = make_test_quantizer(16);
        let x = vec![0.0; 16];
        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);
        assert_eq!(x_recon.len(), 16);
    }

    /// 测试序列化/反序列化
    #[test]
    fn test_save_load() {
        let pq = make_test_quantizer(16);
        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();

        let compressed = pq.compress(&x).unwrap();
        let x_recon1 = pq.decompress(&compressed);

        let temp_path = "/tmp/polarquant_test_save.bin";
        pq.save(temp_path).unwrap();

        let pq_loaded = PolarQuant::load(temp_path).unwrap();
        let x_recon2 = pq_loaded.decompress(&compressed);

        for (r1, r2) in x_recon1.iter().zip(x_recon2.iter()) {
            assert_relative_eq!(r1, r2, epsilon = 1e-10);
        }

        let _ = std::fs::remove_file(temp_path);
    }

    /// 测试可复现性（同一种子产生相同结果）
    #[test]
    fn test_reproducibility() {
        let config1 = PolarQuantConfig::builder(16).seed(42).build().unwrap();
        let config2 = PolarQuantConfig::builder(16).seed(42).build().unwrap();

        let pq1 = PolarQuant::new(config1).unwrap();
        let pq2 = PolarQuant::new(config2).unwrap();

        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();

        let c1 = pq1.compress(&x).unwrap();
        let c2 = pq2.compress(&x).unwrap();

        assert_eq!(c1.radius_idx, c2.radius_idx);
        assert_eq!(c1.angle_indices, c2.angle_indices);
    }

    /// 测试 Hadamard 旋转模式
    #[test]
    fn test_hadamard_mode() {
        let config = PolarQuantConfig::builder(16)
            .use_hadamard(true)
            .seed(42)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();
        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);

        assert_eq!(x_recon.len(), 16);
    }
}
