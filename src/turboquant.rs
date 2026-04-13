//! TurboQuant Flat 索引
//!
//! 基于 Hadamard 旋转和 Lloyd-Max 量化的向量索引。
//! 支持可选的 SQ8 refinement 两阶段搜索。
//!
//! # 核心思想
//! 1. Hadamard 旋转: 将向量分量归一化为 Beta 分布
//! 2. Lloyd-Max 量化: 针对分布优化量化中心
//! 3. SQ8 refinement: 两阶段搜索提升召回率
//!
//! # 召回率
//! - 4-bit: ~86%
//! - 6-bit: ~93%
//! - 4-bit + SQ8: ~98%

use std::collections::BinaryHeap;

use crate::hadamard::HadamardRotation;
use crate::lloyd_max::LloydMaxQuantizer;
use crate::sq8::SQ8Quantizer;
use crate::utils::{l2_normalize, next_power_of_2, FloatOrd};

/// TurboQuant Flat 索引
///
/// 支持可选 SQ8 refinement 的向量索引。
pub struct TurboQuantFlatIndex {
    /// 向量维度
    pub d: usize,
    /// 每维量化位数
    pub nbits: usize,
    /// Hadamard 旋转器
    pub rotation: HadamardRotation,
    /// Lloyd-Max 量化器
    pub quantizer: LloydMaxQuantizer,
    /// 编码后的向量
    pub codes: Vec<u8>,
    /// 可选的 SQ8 量化器
    pub sq8: Option<SQ8Quantizer>,
    /// SQ8 编码后的向量
    pub sq8_codes: Vec<u8>,
    /// 总向量数
    pub ntotal: usize,
}

impl TurboQuantFlatIndex {
    /// 创建新的 TurboQuant 索引
    ///
    /// # 参数
    /// - `d`: 向量维度
    /// - `nbits`: 每维量化位数 (4, 6, 8)
    /// - `use_sq8`: 是否使用 SQ8 refinement
    ///
    /// # 返回值
    /// 初始化的索引
    pub fn new(d: usize, nbits: usize, use_sq8: bool) -> Self {
        let d_rotated = next_power_of_2(d);
        let rotation = HadamardRotation::new(d, 12345);
        let quantizer = LloydMaxQuantizer::new(d_rotated, nbits);
        let sq8 = if use_sq8 { Some(SQ8Quantizer::new(d)) } else { None };

        Self {
            d,
            nbits,
            rotation,
            quantizer,
            codes: Vec::new(),
            sq8,
            sq8_codes: Vec::new(),
            ntotal: 0,
        }
    }

    /// 训练索引
    ///
    /// 仅 SQ8 需要训练，TurboQuant 本身无需训练。
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `n`: 数据点数量
    pub fn train(&mut self, data: &[f32], n: usize) {
        if let Some(ref mut sq8) = self.sq8 {
            sq8.train(data, n);
        }
    }

    /// 添加向量到索引
    ///
    /// # 参数
    /// - `data`: 待添加的向量
    /// - `n`: 向量数量
    ///
    /// # 过程
    /// 1. L2 归一化
    /// 2. Hadamard 旋转
    /// 3. Lloyd-Max 编码
    /// 4. (可选) SQ8 编码
    pub fn add(&mut self, data: &[f32], n: usize) {
        // 步骤 1: L2 归一化
        let mut x_normalized = data.to_vec();
        for i in 0..n {
            l2_normalize(&mut x_normalized[i * self.d..(i + 1) * self.d]);
        }

        // 步骤 2: Hadamard 旋转
        let x_rotated = self.rotation.apply_batch(n, &x_normalized);

        // 步骤 3: Lloyd-Max 编码
        let code_sz = self.quantizer.code_size();
        self.codes.resize((self.ntotal + n) * code_sz, 0);

        for i in 0..n {
            let xi = &x_rotated[i * self.rotation.d_out..(i + 1) * self.rotation.d_out];
            self.quantizer.encode(xi, &mut self.codes[(self.ntotal + i) * code_sz..(self.ntotal + i + 1) * code_sz]);
        }

        // 步骤 4: SQ8 编码 (可选)
        if let Some(ref sq8) = self.sq8 {
            let sq8_sz = sq8.code_size();
            self.sq8_codes.resize((self.ntotal + n) * sq8_sz, 0);
            for i in 0..n {
                let xi = &data[i * self.d..(i + 1) * self.d];
                sq8.encode(xi, &mut self.sq8_codes[(self.ntotal + i) * sq8_sz..(self.ntotal + i + 1) * sq8_sz]);
            }
        }

        self.ntotal += n;
    }

    /// 搜索最近邻
    ///
    /// # 参数
    /// - `queries`: 查询向量
    /// - `n`: 查询数量
    /// - `k`: 返回数量
    /// - `refine_factor`: refinement 倍数
    ///
    /// # 返回值
    /// 每个查询的 k 个最近邻 (索引, 距离)
    ///
    /// # 过程
    /// 1. 归一化查询
    /// 2. Hadamard 旋转
    /// 3. 粗排: Lloyd-Max 距离计算
    /// 4. 精排: SQ8 距离计算 (可选)
    pub fn search(&self, queries: &[f32], n: usize, k: usize, refine_factor: usize) -> Vec<Vec<(usize, f32)>> {
        // 步骤 1: 归一化查询
        let mut x_normalized = queries.to_vec();
        for i in 0..n {
            l2_normalize(&mut x_normalized[i * self.d..(i + 1) * self.d]);
        }

        // 步骤 2: Hadamard 旋转
        let x_rotated = self.rotation.apply_batch(n, &x_normalized);
        let code_sz = self.quantizer.code_size();

        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &x_rotated[q * self.rotation.d_out..(q + 1) * self.rotation.d_out];

            // 步骤 3: 粗排 - 使用 Lloyd-Max 距离
            let k1 = if self.sq8.is_some() {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for i in 0..self.ntotal {
                let code = &self.codes[i * code_sz..(i + 1) * code_sz];
                let dist = self.quantizer.compute_distance(code, query);

                if heap.len() < k1 {
                    heap.push((FloatOrd(dist), i));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((FloatOrd(dist), i));
                }
            }

            let candidates: Vec<(f32, usize)> = heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

            // 步骤 4: 精排 - 使用 SQ8 距离 (可选)
            let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

            if let Some(ref sq8) = self.sq8 {
                let sq8_sz = sq8.code_size();
                let orig_query = &queries[q * self.d..(q + 1) * self.d];
                for (_, idx) in &candidates {
                    let sq8_code = &self.sq8_codes[*idx * sq8_sz..(*idx + 1) * sq8_sz];
                    let refined_dist = sq8.compute_distance(sq8_code, orig_query);

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
        self.quantizer.code_size()
    }

    /// 获取总存储大小
    pub fn total_storage(&self) -> usize {
        let base = self.ntotal * self.quantizer.code_size();
        let sq8_storage = if self.sq8.is_some() {
            self.ntotal * self.d
        } else {
            0
        };
        base + sq8_storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    /// 测试 TurboQuant 4-bit 召回率
    #[test]
    fn test_turboquant_4bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 4, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 4-bit Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.5, "TurboQuant 4-bit recall too low: {}", recall);
    }

    /// 测试 TurboQuant 4-bit + SQ8 召回率
    #[test]
    fn test_turboquant_4bit_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 4-bit + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.9, "TurboQuant 4-bit + SQ8 recall too low: {}", recall);
    }

    /// 测试 TurboQuant 6-bit 召回率
    #[test]
    fn test_turboquant_6bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 6, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 6-bit Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.8, "TurboQuant 6-bit recall too low: {}", recall);
    }
}
