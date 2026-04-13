//! RaBitQ: 随机二进制量化
//!
//! 实现 FAISS RaBitQ 算法，使用 1-bit 编码进行极致压缩。
//! 支持可选的 SQ8 refinement 和 IVF 索引。
//!
//! # 核心思想
//! 1. 相对于质心编码: 编码向量与质心的差值
//! 2. 符号位编码: 只保留每个分量的正负号
//! 3. 距离估计: 使用数学公式估计 L2 距离
//!
//! # 召回率
//! - 1-bit: ~47%
//! - 1-bit + SQ8: ~98%

use std::collections::BinaryHeap;

use crate::sq8::SQ8Quantizer;
use crate::utils::{l2_distance, l2_norm_sq, FloatOrd};

/// 向量符号因子
///
/// 存储编码向量的元数据，用于距离估计。
#[derive(Clone, Debug)]
pub struct SignBitFactors {
    /// ||or - c||²: 向量到质心的距离平方
    pub or_minus_c_l2sqr: f32,
    /// dp_multiplier: 距离估计乘数
    pub dp_multiplier: f32,
}

/// 查询因子数据
///
/// 存储查询相关的预计算值，加速距离计算。
#[derive(Clone, Debug)]
pub struct QueryFactorsData {
    /// c1 = 2 / √d
    pub c1: f32,
    /// c34 = Σ(qr - c) / √d
    pub c34: f32,
    /// ||qr - c||²: 查询到质心的距离平方
    pub qr_to_c_l2sqr: f32,
    /// qr - c: 查询减去质心
    pub rotated_q: Vec<f32>,
}

/// RaBitQ 编解码器
///
/// 实现向量到 1-bit 编码的转换和距离估计。
pub struct RaBitQCodec {
    /// 向量维度
    pub d: usize,
    /// 量化位数 (通常为 1)
    pub nb_bits: usize,
    /// 是否为内积距离
    pub is_inner_product: bool,
}

impl RaBitQCodec {
    /// 创建新的编解码器
    pub fn new(d: usize, nb_bits: usize, is_inner_product: bool) -> Self {
        Self {
            d,
            nb_bits,
            is_inner_product,
        }
    }

    /// 计算编码大小
    ///
    /// 编码 = 符号位 (d/8 字节) + 因子 (8 字节)
    pub fn code_size(&self) -> usize {
        let base_size = (self.d + 7) / 8;
        let factor_size = std::mem::size_of::<SignBitFactors>();
        base_size + factor_size
    }

    /// 计算向量的中间值
    ///
    /// 用于编码时计算符号因子。
    fn compute_vector_intermediate_values(
        &self,
        x: &[f32],
        centroid: Option<&[f32]>,
    ) -> (f32, f32, f32) {
        let mut norm_l2sqr = 0.0f32;
        let mut or_l2sqr = 0.0f32;
        let mut dp_oo = 0.0f32;

        for j in 0..self.d {
            let x_val = x[j];
            let centroid_val = centroid.map_or(0.0, |c| c[j]);
            let or_minus_c = x_val - centroid_val;

            norm_l2sqr += or_minus_c * or_minus_c;
            or_l2sqr += x_val * x_val;

            // 累加绝对值 (符号位之和)
            if or_minus_c > 0.0 {
                dp_oo += or_minus_c;
            } else {
                dp_oo -= or_minus_c;
            }
        }

        (norm_l2sqr, or_l2sqr, dp_oo)
    }

    /// 编码向量
    ///
    /// # 参数
    /// - `x`: 输入向量
    /// - `centroid`: 质心 (可选)
    /// - `code`: 输出编码
    ///
    /// # 编码结构
    /// - 前 d/8 字节: 符号位
    /// - 后 8 字节: SignBitFactors
    pub fn encode(&self, x: &[f32], centroid: Option<&[f32]>, code: &mut [u8]) {
        code.fill(0);

        let (norm_l2sqr, or_l2sqr, dp_oo) = self.compute_vector_intermediate_values(x, centroid);

        // 编码符号位
        for i in 0..self.d {
            let or_minus_c = x[i] - centroid.map_or(0.0, |c| c[i]);
            if or_minus_c > 0.0 {
                code[i / 8] |= 1 << (i % 8);
            }
        }

        // 计算符号因子
        let sqrt_norm_l2 = norm_l2sqr.sqrt();
        let inv_d_sqrt = 1.0 / (self.d as f32).sqrt();
        let inv_norm_l2 = if norm_l2sqr < 1e-10 { 1.0 } else { 1.0 / sqrt_norm_l2 };

        let normalized_dp = dp_oo * inv_norm_l2 * inv_d_sqrt;
        let inv_dp_oo = if normalized_dp.abs() < 1e-10 { 1.0 } else { 1.0 / normalized_dp };

        let factors = SignBitFactors {
            or_minus_c_l2sqr: if self.is_inner_product {
                norm_l2sqr - or_l2sqr
            } else {
                norm_l2sqr
            },
            dp_multiplier: inv_dp_oo * sqrt_norm_l2,
        };

        // 写入因子
        let base_size = (self.d + 7) / 8;
        let factors_bytes = unsafe {
            std::slice::from_raw_parts(
                &factors as *const SignBitFactors as *const u8,
                std::mem::size_of::<SignBitFactors>(),
            )
        };
        code[base_size..base_size + std::mem::size_of::<SignBitFactors>()].copy_from_slice(factors_bytes);
    }

    /// 计算距离
    ///
    /// 使用 RaBitQ 距离估计公式:
    /// dist = ||or-c||² + ||qr-c||² - 2 * dp_multiplier * final_dot
    ///
    /// # 参数
    /// - `code`: 编码向量
    /// - `query_fac`: 查询因子
    ///
    /// # 返回值
    /// 估计的 L2 距离平方
    pub fn compute_distance(
        &self,
        code: &[u8],
        query_fac: &QueryFactorsData,
    ) -> f32 {
        let base_size = (self.d + 7) / 8;
        let factors = unsafe {
            let ptr = code[base_size..].as_ptr() as *const SignBitFactors;
            &*ptr
        };

        // 计算点积估计
        let mut dot_qo = 0.0f32;
        for i in 0..self.d {
            let bit = (code[i / 8] >> (i % 8)) & 1;
            if bit != 0 {
                dot_qo += query_fac.rotated_q[i];
            }
        }

        let final_dot = query_fac.c1 * dot_qo - query_fac.c34;

        // 距离估计公式
        let dist = factors.or_minus_c_l2sqr + query_fac.qr_to_c_l2sqr
            - 2.0 * factors.dp_multiplier * final_dot;

        if self.is_inner_product {
            -0.5 * (dist - query_fac.qr_to_c_l2sqr)
        } else {
            dist.max(0.0)
        }
    }
}

/// 计算查询因子
///
/// 预计算查询相关的值，加速后续距离计算。
///
/// # 参数
/// - `query`: 查询向量
/// - `d`: 维度
/// - `centroid`: 质心
/// - `_is_inner_product`: 是否为内积距离
///
/// # 返回值
/// 查询因子数据
pub fn compute_query_factors(
    query: &[f32],
    d: usize,
    centroid: Option<&[f32]>,
    _is_inner_product: bool,
) -> QueryFactorsData {
    let qr_to_c_l2sqr = match centroid {
        Some(c) => l2_distance(query, c),
        None => l2_norm_sq(query),
    };

    let rotated_q: Vec<f32> = match centroid {
        Some(c) => query.iter().zip(c.iter()).map(|(&q, &c)| q - c).collect(),
        None => query.to_vec(),
    };

    let inv_d = 1.0 / (d as f32).sqrt();
    let sum_q: f32 = rotated_q.iter().sum();

    QueryFactorsData {
        c1: 2.0 * inv_d,
        c34: sum_q * inv_d,
        qr_to_c_l2sqr,
        rotated_q,
    }
}

/// RaBitQ Flat 索引
///
/// 支持可选 SQ8 refinement 的平面索引。
pub struct RaBitQFlatIndex {
    /// 向量维度
    pub d: usize,
    /// 量化位数
    pub nb_bits: usize,
    /// 是否为内积距离
    pub is_inner_product: bool,
    /// 数据质心
    centroid: Vec<f32>,
    /// 编解码器
    codec: RaBitQCodec,
    /// 编码后的向量
    codes: Vec<u8>,
    /// 可选的 SQ8 量化器
    sq8: Option<SQ8Quantizer>,
    /// SQ8 编码后的向量
    sq8_codes: Vec<u8>,
    /// 总向量数
    ntotal: usize,
}

impl RaBitQFlatIndex {
    /// 创建新的 RaBitQ Flat 索引
    pub fn new(d: usize, nb_bits: usize, is_inner_product: bool, use_sq8: bool) -> Self {
        let codec = RaBitQCodec::new(d, nb_bits, is_inner_product);
        let sq8 = if use_sq8 { Some(SQ8Quantizer::new(d)) } else { None };

        Self {
            d,
            nb_bits,
            is_inner_product,
            centroid: vec![0.0; d],
            codec,
            codes: Vec::new(),
            sq8,
            sq8_codes: Vec::new(),
            ntotal: 0,
        }
    }

    /// 训练索引
    ///
    /// 计算数据质心和 SQ8 参数。
    pub fn train(&mut self, data: &[f32], n: usize) {
        // 计算质心
        self.centroid.fill(0.0);
        for i in 0..n {
            for j in 0..self.d {
                self.centroid[j] += data[i * self.d + j];
            }
        }
        for j in 0..self.d {
            self.centroid[j] /= n as f32;
        }

        // 训练 SQ8
        if let Some(ref mut sq8) = self.sq8 {
            sq8.train(data, n);
        }
    }

    /// 添加向量到索引
    pub fn add(&mut self, data: &[f32], n: usize) {
        let code_sz = self.codec.code_size();
        self.codes.resize((self.ntotal + n) * code_sz, 0);

        if let Some(ref sq8) = self.sq8 {
            let sq8_sz = sq8.code_size();
            self.sq8_codes.resize((self.ntotal + n) * sq8_sz, 0);
        }

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];

            // RaBitQ 编码
            self.codec.encode(xi, Some(&self.centroid), &mut self.codes[(self.ntotal + i) * code_sz..(self.ntotal + i + 1) * code_sz]);

            // SQ8 编码
            if let Some(ref sq8) = self.sq8 {
                let sq8_sz = sq8.code_size();
                sq8.encode(xi, &mut self.sq8_codes[(self.ntotal + i) * sq8_sz..(self.ntotal + i + 1) * sq8_sz]);
            }
        }

        self.ntotal += n;
    }

    /// 搜索最近邻
    ///
    /// 两阶段搜索:
    /// 1. 粗排: RaBitQ 距离估计
    /// 2. 精排: SQ8 距离计算 (可选)
    pub fn search(&self, queries: &[f32], n: usize, k: usize, refine_factor: usize) -> Vec<Vec<(usize, f32)>> {
        let code_sz = self.codec.code_size();
        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &queries[q * self.d..(q + 1) * self.d];
            let query_fac = compute_query_factors(query, self.d, Some(&self.centroid), self.is_inner_product);

            // 粗排
            let k1 = if self.sq8.is_some() {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for i in 0..self.ntotal {
                let code = &self.codes[i * code_sz..(i + 1) * code_sz];
                let dist = self.codec.compute_distance(code, &query_fac);

                if heap.len() < k1 {
                    heap.push((FloatOrd(dist), i));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((FloatOrd(dist), i));
                }
            }

            let candidates: Vec<(f32, usize)> = heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

            // 精排
            let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

            if let Some(ref sq8) = self.sq8 {
                let sq8_sz = sq8.code_size();
                for (_, idx) in &candidates {
                    let sq8_code = &self.sq8_codes[*idx * sq8_sz..(*idx + 1) * sq8_sz];
                    let refined_dist = sq8.compute_distance(sq8_code, query);

                    if final_heap.len() < k {
                        final_heap.push((FloatOrd(refined_dist), *idx));
                    } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                        final_heap.pop();
                        final_heap.push((FloatOrd(refined_dist), *idx));
                    }
                }
            } else {
                for (dist, idx) in candidates {
                    if final_heap.len() < k {
                        final_heap.push((FloatOrd(dist), idx));
                    } else if dist < final_heap.peek().unwrap().0 .0 {
                        final_heap.pop();
                        final_heap.push((FloatOrd(dist), idx));
                    }
                }
            }

            let mut result: Vec<(usize, f32)> = final_heap
                .into_iter()
                .map(|(FloatOrd(d), i)| (i, d))
                .collect();
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results.push(result);
        }

        results
    }

    /// 获取总向量数
    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    /// 获取编码大小
    pub fn code_size(&self) -> usize {
        self.codec.code_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    /// 测试 RaBitQ Flat 1-bit 召回率
    #[test]
    fn test_rabitq_flat_1bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQFlatIndex::new(d, 1, false, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ Flat 1-bit Recall@{}: {:.4}", k, recall);
    }

    /// 测试 RaBitQ Flat + SQ8 召回率
    #[test]
    fn test_rabitq_flat_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ Flat 1-bit + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.85, "RaBitQ Flat + SQ8 recall too low: {}", recall);
    }
}
