// PolarQuant 批量处理模块
//
// 提供批量压缩/解压和批量误差评估功能，
// 适用于 LLM KV Cache 等需要处理大量向量的场景。

use crate::quantizer::{CompressedVector, PolarQuant};

/// PolarQuant 批量处理器
///
/// 封装 PolarQuant 量化器，提供批量操作接口。
/// 适用于 KV Cache 中大量 Key/Value 向量的压缩场景。
pub struct PolarQuantBatch<'a> {
    quantizer: &'a PolarQuant,
}

impl<'a> PolarQuantBatch<'a> {
    /// 创建批量处理器
    pub fn new(quantizer: &'a PolarQuant) -> Self {
        Self { quantizer }
    }

    /// 批量压缩向量
    ///
    /// 对每个向量调用 quantizer.compress()，跳过维度不匹配的向量。
    pub fn compress_batch(&self, x: &[Vec<f64>]) -> Vec<CompressedVector> {
        x.iter()
            .filter_map(|v| self.quantizer.compress(v).ok())
            .collect()
    }

    /// 批量解压向量
    pub fn decompress_batch(&self, compressed: &[CompressedVector]) -> Vec<Vec<f64>> {
        compressed
            .iter()
            .map(|c| self.quantizer.decompress(c))
            .collect()
    }

    /// 计算批量重建误差指标
    ///
    /// 返回聚合指标：
    /// - mean_mse: 平均均方误差
    /// - mean_rmse: 平均均方根误差
    /// - mean_cosine: 平均余弦相似度
    /// - min_cosine: 最小余弦相似度（最差情况）
    pub fn compute_batch_error(&self, x: &[Vec<f64>], x_reconstructed: &[Vec<f64>]) -> BatchErrorMetrics {
        let n = x.len().min(x_reconstructed.len());
        let mut mse_list = Vec::with_capacity(n);
        let mut cosine_list = Vec::with_capacity(n);

        for i in 0..n {
            let errors = self.quantizer.compute_error(&x[i], &x_reconstructed[i]);
            mse_list.push(errors.mse);
            cosine_list.push(errors.cosine_similarity);
        }

        // 计算聚合统计量
        let mean_mse = if mse_list.is_empty() {
            0.0
        } else {
            mse_list.iter().sum::<f64>() / mse_list.len() as f64
        };
        let mean_rmse = mean_mse.sqrt();
        let mean_cosine = if cosine_list.is_empty() {
            0.0
        } else {
            cosine_list.iter().sum::<f64>() / cosine_list.len() as f64
        };
        let min_cosine = cosine_list
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        BatchErrorMetrics {
            mean_mse,
            mean_rmse,
            mean_cosine,
            min_cosine,
        }
    }
}

/// 批量误差指标
#[derive(Debug, Clone)]
pub struct BatchErrorMetrics {
    /// 平均均方误差
    pub mean_mse: f64,
    /// 平均均方根误差
    pub mean_rmse: f64,
    /// 平均余弦相似度
    pub mean_cosine: f64,
    /// 最小余弦相似度（最差情况）
    pub min_cosine: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolarQuantConfig;

    /// 创建测试用量化器
    fn make_test_quantizer() -> PolarQuant {
        let config = PolarQuantConfig::new(16).unwrap();
        PolarQuant::new(config).unwrap()
    }

    /// 生成确定性随机向量（用于测试）
    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| (0..dim).map(|i| ((i + 1) as f64 * 0.1).sin()).collect())
            .collect()
    }

    /// 测试批量压缩
    #[test]
    fn test_batch_compress() {
        let pq = make_test_quantizer();
        let batch = PolarQuantBatch::new(&pq);
        let vectors = generate_random_vectors(10, 16);

        let compressed = batch.compress_batch(&vectors);
        assert_eq!(compressed.len(), 10);
    }

    /// 测试批量解压
    #[test]
    fn test_batch_decompress() {
        let pq = make_test_quantizer();
        let batch = PolarQuantBatch::new(&pq);
        let vectors = generate_random_vectors(5, 16);

        let compressed = batch.compress_batch(&vectors);
        let reconstructed = batch.decompress_batch(&compressed);

        assert_eq!(reconstructed.len(), 5);
        for v in &reconstructed {
            assert_eq!(v.len(), 16);
        }
    }

    /// 测试批量误差指标计算
    #[test]
    fn test_batch_error_metrics() {
        let pq = make_test_quantizer();
        let batch = PolarQuantBatch::new(&pq);
        let vectors = generate_random_vectors(10, 16);

        let compressed = batch.compress_batch(&vectors);
        let reconstructed = batch.decompress_batch(&compressed);

        let errors = batch.compute_batch_error(&vectors, &reconstructed);
        assert!(errors.mean_mse >= 0.0);
        assert!(errors.mean_cosine > 0.5);
    }
}
