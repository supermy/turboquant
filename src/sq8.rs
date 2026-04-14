//! SQ8 标量量化器
//!
//! 8-bit 标量量化，用于两阶段搜索的 refinement 阶段。
//! 每个维度独立量化，存储为 1 字节。
//!
//! # 特点
//! - 简单高效: 线性映射到 [0, 255]
//! - 高精度: 相对误差 < 5%
//! - 用于 refinement: 提升召回率

/// SQ8 标量量化器
///
/// 每个维度独立量化，存储为 1 字节。
#[derive(Clone)]
pub struct SQ8Quantizer {
    /// 向量维度
    pub d: usize,
    /// 每维最小值
    pub vmin: Vec<f32>,
    /// 每维最大值
    pub vmax: Vec<f32>,
    /// 预计算 scale: (vmax - vmin) / 255.0
    pub scale: Vec<f32>,
}

impl SQ8Quantizer {
    /// 创建新的 SQ8 量化器
    ///
    /// # 参数
    /// - `d`: 向量维度
    pub fn new(d: usize) -> Self {
        Self {
            d,
            vmin: vec![f32::MAX; d],
            vmax: vec![f32::MIN; d],
            scale: vec![0.0; d],
        }
    }

    /// 训练量化器
    ///
    /// 计算每个维度的最小值和最大值。
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `n`: 数据点数量
    pub fn train(&mut self, data: &[f32], n: usize) {
        // 初始化
        self.vmin.fill(f32::MAX);
        self.vmax.fill(f32::MIN);

        // 统计每维的最小值和最大值
        for i in 0..n {
            for j in 0..self.d {
                let val = data[i * self.d + j];
                self.vmin[j] = self.vmin[j].min(val);
                self.vmax[j] = self.vmax[j].max(val);
            }
        }

        // 避免零区间
        for j in 0..self.d {
            if self.vmax[j] - self.vmin[j] < 1e-6 {
                self.vmax[j] = self.vmin[j] + 1e-6;
            }
        }

        for j in 0..self.d {
            self.scale[j] = (self.vmax[j] - self.vmin[j]) / 255.0;
        }
    }

    /// 计算编码大小
    ///
    /// # 返回值
    /// 每个向量的编码字节数 (= d)
    pub fn code_size(&self) -> usize {
        self.d
    }

    /// 编码向量
    ///
    /// # 参数
    /// - `x`: 输入向量
    /// - `code`: 输出编码 (d 字节)
    pub fn encode(&self, x: &[f32], code: &mut [u8]) {
        for j in 0..self.d {
            // 线性映射到 [0, 1]
            let normalized = ((x[j] - self.vmin[j]) / (self.vmax[j] - self.vmin[j]))
                .clamp(0.0, 1.0);
            // 量化到 [0, 255]
            code[j] = (normalized * 255.0) as u8;
        }
    }

    /// 解码向量
    ///
    /// # 参数
    /// - `code`: 编码 (d 字节)
    /// - `x`: 输出向量
    pub fn decode(&self, code: &[u8], x: &mut [f32]) {
        for j in 0..self.d {
            x[j] = self.vmin[j] + code[j] as f32 * self.scale[j];
        }
    }

    /// 计算编码向量与查询的距离
    ///
    /// # 参数
    /// - `code`: 编码向量
    /// - `query`: 查询向量 (原始空间)
    ///
    /// # 返回值
    /// L2 距离平方
    pub fn compute_distance(&self, code: &[u8], query: &[f32]) -> f32 {
        crate::utils::sq8_distance_simd(code, query, &self.vmin, &self.scale, self.d)
    }

    pub fn compute_distance_preprocessed(&self, code: &[u8], query_preprocessed: &[f32]) -> f32 {
        let mut dist = 0.0f32;
        for j in 0..self.d {
            let diff = code[j] as f32 - query_preprocessed[j];
            dist += diff * diff;
        }
        dist * self.scale_sum_sq()
    }

    pub fn preprocess_query(&self, query: &[f32]) -> Vec<f32> {
        query.iter().zip(self.vmin.iter()).zip(self.scale.iter())
            .map(|((q, v), s)| (q - v) / s)
            .collect()
    }

    fn scale_sum_sq(&self) -> f32 {
        let mut sum = 0.0f32;
        for s in &self.scale {
            sum += s * s;
        }
        sum / self.d as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::l2_distance;

    /// 测试 SQ8 编解码往返
    #[test]
    fn test_sq8_roundtrip() {
        let d = 128;
        let mut sq8 = SQ8Quantizer::new(d);
        // 生成训练数据
        let data: Vec<f32> = (0..1000 * d).map(|i| (i as f32 * 0.001).sin()).collect();
        sq8.train(&data, 1000);

        // 编码解码
        let x = &data[0..d];
        let mut code = vec![0u8; sq8.code_size()];
        sq8.encode(x, &mut code);
        let mut decoded = vec![0.0f32; d];
        sq8.decode(&code, &mut decoded);

        // 检查相对误差
        let orig_norm = crate::utils::l2_norm(x);
        let err = l2_distance(x, &decoded).sqrt();
        assert!(err / orig_norm < 0.07, "SQ8 相对误差过大: {}", err / orig_norm);
    }

    /// 测试 SQ8 距离精度
    #[test]
    fn test_sq8_distance_accuracy() {
        let d = 64;
        let mut sq8 = SQ8Quantizer::new(d);
        let data: Vec<f32> = (0..500 * d).map(|i| (i as f32 * 0.01).cos()).collect();
        sq8.train(&data, 500);

        let a = &data[0..d];
        let b = &data[d..2 * d];

        let mut code_a = vec![0u8; sq8.code_size()];
        sq8.encode(a, &mut code_a);

        // 比较 SQ8 距离与真实距离
        let true_dist = l2_distance(a, b);
        let sq8_dist = sq8.compute_distance(&code_a, b);
        assert!((true_dist - sq8_dist).abs() / true_dist < 0.1);
    }
}
