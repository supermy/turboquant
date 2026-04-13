//! RaBitQ IVF 索引
//!
//! 结合 IVF (倒排文件索引) 和 RaBitQ 量化的大规模向量索引。
//!
//! # 核心思想
//! 1. IVF 分区: 使用 KMeans 将向量分区
//! 2. RaBitQ 编码: 每个分区独立编码
//! 3. SQ8 refinement: 两阶段搜索提升召回率
//!
//! # 召回率
//! - RaBitQ IVF + SQ8: ~98%

use std::collections::{BinaryHeap, HashMap};

use crate::kmeans::KMeans;
use crate::rabitq::{RaBitQCodec, compute_query_factors};
use crate::sq8::SQ8Quantizer;
use crate::utils::FloatOrd;

/// 聚类数据
///
/// 存储单个聚类的编码向量和 ID。
struct ClusterData {
    /// RaBitQ 编码
    codes: Vec<u8>,
    /// 原始向量 ID
    ids: Vec<usize>,
    /// SQ8 编码
    sq8_codes: Vec<u8>,
}

/// RaBitQ IVF 索引
///
/// 支持大规模向量搜索的倒排索引。
pub struct RaBitQIVFIndex {
    /// 向量维度
    pub d: usize,
    /// 聚类数量
    pub nlist: usize,
    /// 量化位数
    pub nb_bits: usize,
    /// 是否为内积距离
    pub is_inner_product: bool,

    /// KMeans 聚类器
    kmeans: KMeans,
    /// 每个聚类的编解码器
    codecs: Vec<RaBitQCodec>,
    /// 聚类中心
    cluster_centroids: Vec<Vec<f32>>,
    /// 聚类数据
    clusters: Vec<ClusterData>,
    /// 每个聚类的 SQ8 量化器
    sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    /// 总向量数
    ntotal: usize,
}

impl RaBitQIVFIndex {
    /// 创建新的 RaBitQ IVF 索引
    ///
    /// # 参数
    /// - `d`: 向量维度
    /// - `nlist`: 聚类数量
    /// - `nb_bits`: 量化位数
    /// - `is_inner_product`: 是否为内积距离
    /// - `use_sq8`: 是否使用 SQ8 refinement
    pub fn new(d: usize, nlist: usize, nb_bits: usize, is_inner_product: bool, use_sq8: bool) -> Self {
        let kmeans = KMeans::new(d, nlist, 20);
        let codecs: Vec<RaBitQCodec> = (0..nlist)
            .map(|_| RaBitQCodec::new(d, nb_bits, is_inner_product))
            .collect();
        let cluster_centroids = vec![vec![0.0; d]; nlist];
        let clusters = (0..nlist)
            .map(|_| ClusterData {
                codes: Vec::new(),
                ids: Vec::new(),
                sq8_codes: Vec::new(),
            })
            .collect();
        let sq8_quantizers: Vec<Option<SQ8Quantizer>> = if use_sq8 {
            (0..nlist).map(|_| Some(SQ8Quantizer::new(d))).collect()
        } else {
            (0..nlist).map(|_| None).collect()
        };

        Self {
            d,
            nlist,
            nb_bits,
            is_inner_product,
            kmeans,
            codecs,
            cluster_centroids,
            clusters,
            sq8_quantizers,
            ntotal: 0,
        }
    }

    /// 训练索引
    ///
    /// 1. 训练 KMeans 聚类
    /// 2. 训练每个聚类的 SQ8 量化器
    pub fn train(&mut self, data: &[f32], n: usize) {
        // 训练 KMeans
        self.kmeans.train(data, n, 42);

        // 复制聚类中心
        for i in 0..self.nlist {
            self.cluster_centroids[i].copy_from_slice(
                &self.kmeans.centroids[i * self.d..(i + 1) * self.d],
            );
        }

        // 训练每个聚类的 SQ8
        let use_sq8 = self.sq8_quantizers[0].is_some();
        if use_sq8 {
            let mut cluster_data: Vec<Vec<f32>> = vec![Vec::new(); self.nlist];
            for i in 0..n {
                let cluster_id = self.kmeans.assign_cluster(&data[i * self.d..(i + 1) * self.d]);
                cluster_data[cluster_id].extend_from_slice(&data[i * self.d..(i + 1) * self.d]);
            }

            for c in 0..self.nlist {
                let n_in_cluster = cluster_data[c].len() / self.d;
                if n_in_cluster > 0 {
                    if let Some(ref mut sq8) = self.sq8_quantizers[c] {
                        sq8.train(&cluster_data[c], n_in_cluster);
                    }
                }
            }
        }
    }

    /// 添加向量到索引
    pub fn add(&mut self, data: &[f32], n: usize) {
        let code_sz = self.codecs[0].code_size();

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];
            let cluster_id = self.kmeans.assign_cluster(xi);

            // RaBitQ 编码
            let mut code = vec![0u8; code_sz];
            self.codecs[cluster_id].encode(xi, Some(&self.cluster_centroids[cluster_id]), &mut code);
            self.clusters[cluster_id].codes.extend_from_slice(&code);

            // SQ8 编码
            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                let mut sq8_code = vec![0u8; sq8.code_size()];
                sq8.encode(xi, &mut sq8_code);
                self.clusters[cluster_id].sq8_codes.extend_from_slice(&sq8_code);
            }

            // 记录 ID
            self.clusters[cluster_id].ids.push(self.ntotal + i);
        }

        self.ntotal += n;
    }

    /// 搜索最近邻
    ///
    /// # 参数
    /// - `queries`: 查询向量
    /// - `n`: 查询数量
    /// - `k`: 返回数量
    /// - `nprobe`: 搜索的聚类数量
    /// - `refine_factor`: refinement 倍数
    ///
    /// # 过程
    /// 1. 找到最近的 nprobe 个聚类
    /// 2. 在这些聚类中搜索
    /// 3. SQ8 refinement
    pub fn search(
        &self,
        queries: &[f32],
        n: usize,
        k: usize,
        nprobe: usize,
        refine_factor: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let code_sz = self.codecs[0].code_size();
        let use_sq8 = self.sq8_quantizers[0].is_some();
        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &queries[q * self.d..(q + 1) * self.d];

            // 步骤 1: 找到最近的聚类
            let nearest = self.kmeans.nearest_clusters(query, nprobe);

            // 步骤 2: 在这些聚类中搜索
            let k1 = if use_sq8 {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for (_, cluster_id) in &nearest {
                let cluster = &self.clusters[*cluster_id];
                let n_vectors = cluster.ids.len();
                if n_vectors == 0 {
                    continue;
                }

                let query_fac = compute_query_factors(
                    query,
                    self.d,
                    Some(&self.cluster_centroids[*cluster_id]),
                    self.is_inner_product,
                );

                for v in 0..n_vectors {
                    let code = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                    let dist = self.codecs[*cluster_id].compute_distance(code, &query_fac);

                    if heap.len() < k1 {
                        heap.push((FloatOrd(dist), cluster.ids[v]));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((FloatOrd(dist), cluster.ids[v]));
                    }
                }
            }

            let candidates: Vec<(f32, usize)> = heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

            // 步骤 3: SQ8 refinement
            let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

            if use_sq8 {
                // 构建 ID 到聚类的映射
                let id_to_cluster: HashMap<usize, usize> = self
                    .clusters
                    .iter()
                    .enumerate()
                    .flat_map(|(c, cluster)| cluster.ids.iter().map(move |&id| (id, c)))
                    .collect();

                for (_, idx) in &candidates {
                    if let Some(&cluster_id) = id_to_cluster.get(idx) {
                        let cluster = &self.clusters[cluster_id];
                        let pos = cluster.ids.iter().position(|&id| id == *idx);
                        if let Some(pos) = pos {
                            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                                let sq8_sz = sq8.code_size();
                                let sq8_code = &cluster.sq8_codes[pos * sq8_sz..(pos + 1) * sq8_sz];
                                let refined_dist = sq8.compute_distance(sq8_code, query);

                                if final_heap.len() < k {
                                    final_heap.push((FloatOrd(refined_dist), *idx));
                                } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                                    final_heap.pop();
                                    final_heap.push((FloatOrd(refined_dist), *idx));
                                }
                            }
                        }
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
        self.codecs[0].code_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    /// 测试 RaBitQ IVF + SQ8 召回率
    #[test]
    fn test_rabitq_ivf_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;
        let nlist = 64;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, nlist.min(64), 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ IVF + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.85, "RaBitQ IVF + SQ8 recall too low: {}", recall);
    }
}
