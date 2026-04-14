//! RaBitQ IVF 索引 + TurboQuant IVF 索引
//!
//! 结合 IVF (倒排文件索引) 和量化方法的大规模向量索引。
//!
//! # 两种 IVF 变体
//! 1. RaBitQIVFIndex: IVF + RaBitQ 1-bit 量化 (高速粗排)
//! 2. TurboQuantIVFIndex: IVF + TurboQuant 4-bit 量化 (高召回率)
//!
//! # 核心思想
//! 1. IVF 分区: 使用 KMeans 将向量分区
//! 2. 量化编码: 每个分区独立编码
//! 3. SQ8 refinement: 两阶段搜索提升召回率
//!
//! # 召回率
//! - RaBitQ IVF + SQ8: ~98%
//! - TurboQuant IVF + SQ8: ~98.5%

use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::kmeans::KMeans;
use crate::rabitq::{RaBitQCodec, QueryFactorsData, compute_query_factors_into};
use crate::sq8::SQ8Quantizer;
use crate::utils::{prefetch_read, FloatOrd};

/// 聚类数据 (SoA 布局)
///
/// 符号位和因子分离存储，粗排只读 signs，减少内存带宽。
pub struct ClusterData {
    /// 符号位 (d/8 字节/向量，粗排热路径)
    pub signs: Vec<u8>,
    /// SignBitFactors (8 字节/向量，精排冷路径)
    pub factors: Vec<u8>,
    /// RaBitQ 编码 (兼容旧接口，signs + factors 交织)
    pub codes: Vec<u8>,
    /// 原始向量 ID (u32 足够，4B vs 8B)
    pub ids: Vec<u32>,
    /// SQ8 编码
    pub sq8_codes: Vec<u8>,
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
    pub kmeans: KMeans,
    /// 每个聚类的编解码器
    pub codecs: Vec<RaBitQCodec>,
    /// 聚类中心
    pub cluster_centroids: Vec<Vec<f32>>,
    /// 聚类数据
    pub clusters: Vec<ClusterData>,
    /// 每个聚类的 SQ8 量化器
    pub sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    /// 总向量数
    pub ntotal: usize,
    /// ID 到 (cluster_id, position_in_cluster) 的预构建索引 (Vec 替代 HashMap)
    id_to_pos: Vec<(usize, usize)>,
    /// 聚类偏移前缀和 (cluster_offsets[c] = 前 c 个聚类的向量总数)
    cluster_offsets: Vec<usize>,
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
                signs: Vec::new(),
                factors: Vec::new(),
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
            id_to_pos: Vec::new(),
            cluster_offsets: vec![0; nlist + 1],
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
        let signs_sz = self.codecs[0].signs_size();

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];
            let cluster_id = self.kmeans.assign_cluster(xi);

            let pos_in_cluster = self.clusters[cluster_id].ids.len();

            let mut code = vec![0u8; code_sz];
            self.codecs[cluster_id].encode(xi, Some(&self.cluster_centroids[cluster_id]), &mut code);

            self.clusters[cluster_id].signs.extend_from_slice(&code[..signs_sz]);
            self.clusters[cluster_id].factors.extend_from_slice(&code[signs_sz..signs_sz + 8]);
            self.clusters[cluster_id].codes.extend_from_slice(&code);

            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                let mut sq8_code = vec![0u8; sq8.code_size()];
                sq8.encode(xi, &mut sq8_code);
                self.clusters[cluster_id].sq8_codes.extend_from_slice(&sq8_code);
            }

            let id = self.ntotal + i;
            self.clusters[cluster_id].ids.push(id as u32);

            if id >= self.id_to_pos.len() {
                self.id_to_pos.resize(id + 1, (0, 0));
            }
            self.id_to_pos[id] = (cluster_id, pos_in_cluster);
        }

        self.ntotal += n;
        self.rebuild_cluster_offsets();
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
        let signs_sz = self.codecs[0].signs_size();
        let use_sq8 = self.sq8_quantizers[0].is_some();

        let results: Vec<Vec<(usize, f32)>> = (0..n)
            .into_par_iter()
            .map(|q| {
                let query = &queries[q * self.d..(q + 1) * self.d];
                let mut nearest_buf = Vec::with_capacity(nprobe);
                self.kmeans.nearest_clusters_into(query, nprobe, &mut nearest_buf);

                let k1 = if use_sq8 {
                    (k * refine_factor).min(self.ntotal)
                } else {
                    k
                };

                let mut query_fac = QueryFactorsData::with_capacity(self.d);

                if use_sq8 {
                    let mut rabitq_heap: BinaryHeap<(FloatOrd, (usize, usize))> = BinaryHeap::with_capacity(k1);

                    for (_, cluster_id) in &nearest_buf {
                        let cluster = &self.clusters[*cluster_id];
                        let n_vectors = cluster.ids.len();
                        if n_vectors == 0 {
                            continue;
                        }

                        compute_query_factors_into(
                            query,
                            self.d,
                            Some(&self.cluster_centroids[*cluster_id]),
                            &mut query_fac,
                        );

                        for v in 0..n_vectors {
                            if v + 2 < n_vectors {
                                unsafe {
                                    prefetch_read(cluster.signs.as_ptr().add((v + 1) * signs_sz));
                                    prefetch_read(cluster.factors.as_ptr().add((v + 1) * 8));
                                }
                            }
                            let signs = &cluster.signs[v * signs_sz..(v + 1) * signs_sz];
                            let dot_qo = self.codecs[*cluster_id].compute_distance_signs_only(signs, &query_fac);

                            let factors = &cluster.factors[v * 8..(v + 1) * 8];
                            let or_minus_c_l2sqr = f32::from_le_bytes(factors[0..4].try_into().unwrap());
                            let dp_multiplier = f32::from_le_bytes(factors[4..8].try_into().unwrap());
                            let dist = self.codecs[*cluster_id].compute_distance_with_factors(
                                dot_qo, or_minus_c_l2sqr, dp_multiplier, &query_fac,
                            );

                            if rabitq_heap.len() < k1 {
                                rabitq_heap.push((FloatOrd(dist), (*cluster_id, v)));
                            } else if dist < rabitq_heap.peek().unwrap().0 .0 {
                                rabitq_heap.pop();
                                rabitq_heap.push((FloatOrd(dist), (*cluster_id, v)));
                            }
                        }
                    }

                    let rabitq_candidates: Vec<(f32, usize, usize)> = rabitq_heap
                        .into_iter()
                        .map(|(FloatOrd(d), (cid, v))| (d, cid, v))
                        .collect();

                    let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);
                    for (_, cluster_id, v) in &rabitq_candidates {
                        if let Some(ref sq8) = self.sq8_quantizers[*cluster_id] {
                            let sq8_sz = sq8.code_size();
                            let sq8_code = &self.clusters[*cluster_id].sq8_codes[*v * sq8_sz..(*v + 1) * sq8_sz];
                            let refined_dist = sq8.compute_distance(sq8_code, query);
                            let id = self.clusters[*cluster_id].ids[*v] as usize;

                            if final_heap.len() < k {
                                final_heap.push((FloatOrd(refined_dist), id));
                            } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                                final_heap.pop();
                                final_heap.push((FloatOrd(refined_dist), id));
                            }
                        }
                    }

                    let mut result: Vec<(usize, f32)> = final_heap
                        .into_iter()
                        .map(|(FloatOrd(d), i)| (i, d))
                        .collect();
                    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    result
                } else {
                    let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

                    for (_, cluster_id) in &nearest_buf {
                        let cluster = &self.clusters[*cluster_id];
                        let n_vectors = cluster.ids.len();
                        if n_vectors == 0 {
                            continue;
                        }

                        compute_query_factors_into(
                            query,
                            self.d,
                            Some(&self.cluster_centroids[*cluster_id]),
                            &mut query_fac,
                        );

                        for v in 0..n_vectors {
                            if v + 2 < n_vectors {
                                unsafe {
                                    prefetch_read(cluster.signs.as_ptr().add((v + 1) * signs_sz));
                                    prefetch_read(cluster.factors.as_ptr().add((v + 1) * 8));
                                }
                            }
                            let signs = &cluster.signs[v * signs_sz..(v + 1) * signs_sz];
                            let dot_qo = self.codecs[*cluster_id].compute_distance_signs_only(signs, &query_fac);

                            let factors = &cluster.factors[v * 8..(v + 1) * 8];
                            let or_minus_c_l2sqr = f32::from_le_bytes(factors[0..4].try_into().unwrap());
                            let dp_multiplier = f32::from_le_bytes(factors[4..8].try_into().unwrap());
                            let dist = self.codecs[*cluster_id].compute_distance_with_factors(
                                dot_qo, or_minus_c_l2sqr, dp_multiplier, &query_fac,
                            );

                            if heap.len() < k1 {
                                heap.push((FloatOrd(dist), cluster.ids[v] as usize));
                            } else if dist < heap.peek().unwrap().0 .0 {
                                heap.pop();
                                heap.push((FloatOrd(dist), cluster.ids[v] as usize));
                            }
                        }
                    }

                    let mut result: Vec<(usize, f32)> = heap
                        .into_iter()
                        .map(|(FloatOrd(d), i)| (i, d))
                        .collect();
                    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    result
                }
            })
            .collect();

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

    /// 重建 ID 到位置的索引
    ///
    /// 从 RocksDB 加载索引后调用，重建 id_to_pos 映射。
    pub fn rebuild_id_index(&mut self) {
        self.id_to_pos.clear();
        self.id_to_pos.resize(self.ntotal, (0, 0));
        for c in 0..self.nlist {
            for (pos, &id) in self.clusters[c].ids.iter().enumerate() {
                if (id as usize) < self.id_to_pos.len() {
                    self.id_to_pos[id as usize] = (c, pos);
                }
            }
        }
        self.rebuild_cluster_offsets();
    }

    fn rebuild_cluster_offsets(&mut self) {
        self.cluster_offsets = vec![0; self.nlist + 1];
        for c in 0..self.nlist {
            self.cluster_offsets[c + 1] = self.cluster_offsets[c] + self.clusters[c].ids.len();
        }
    }
}

/// TurboQuant IVF 聚类数据
pub struct TQClusterData {
    /// TurboQuant 编码 (code_sz 字节/向量)
    pub codes: Vec<u8>,
    /// 原始向量 ID
    pub ids: Vec<u32>,
    /// SQ8 编码
    pub sq8_codes: Vec<u8>,
}

/// TurboQuant IVF 索引
///
/// IVF + TurboQuant 4-bit 量化 + SQ8 精炼。
/// 相比 RaBitQ IVF:
/// - 4-bit 量化精度更高，粗排召回率更高
/// - Split LUT (8KB) L1 常驻，减少 cache miss
/// - 无需 per-cluster codec (TurboQuant 全局量化)
/// - 支持 early stop
pub struct TurboQuantIVFIndex {
    pub d: usize,
    pub nlist: usize,
    pub nbits: usize,
    pub kmeans: KMeans,
    pub rotation: crate::hadamard::HadamardRotation,
    pub quantizer: crate::lloyd_max::LloydMaxQuantizer,
    pub cluster_centroids: Vec<Vec<f32>>,
    pub clusters: Vec<TQClusterData>,
    pub sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    pub ntotal: usize,
    id_to_pos: Vec<(usize, usize)>,
    cluster_offsets: Vec<usize>,
}

impl TurboQuantIVFIndex {
    pub fn new(d: usize, nlist: usize, nbits: usize, use_sq8: bool) -> Self {
        let kmeans = KMeans::new(d, nlist, 20);
        let d_rotated = crate::utils::next_power_of_2(d);
        let rotation = crate::hadamard::HadamardRotation::new(d, 12345);
        let quantizer = crate::lloyd_max::LloydMaxQuantizer::new(d_rotated, nbits);
        let cluster_centroids = vec![vec![0.0; d]; nlist];
        let clusters = (0..nlist)
            .map(|_| TQClusterData {
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
            nbits,
            kmeans,
            rotation,
            quantizer,
            cluster_centroids,
            clusters,
            sq8_quantizers,
            ntotal: 0,
            id_to_pos: Vec::new(),
            cluster_offsets: vec![0; nlist + 1],
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.kmeans.train(data, n, 42);

        for i in 0..self.nlist {
            self.cluster_centroids[i].copy_from_slice(
                &self.kmeans.centroids[i * self.d..(i + 1) * self.d],
            );
        }

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

    pub fn add(&mut self, data: &[f32], n: usize) {
        let code_sz = self.quantizer.code_size();

        let mut x_normalized = data.to_vec();
        for i in 0..n {
            crate::utils::l2_normalize(&mut x_normalized[i * self.d..(i + 1) * self.d]);
        }
        let x_rotated = self.rotation.apply_batch(n, &x_normalized);

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];
            let cluster_id = self.kmeans.assign_cluster(xi);

            let pos_in_cluster = self.clusters[cluster_id].ids.len();

            let xi_rotated = &x_rotated[i * self.rotation.d_out..(i + 1) * self.rotation.d_out];
            let mut code = vec![0u8; code_sz];
            self.quantizer.encode(xi_rotated, &mut code);
            self.clusters[cluster_id].codes.extend_from_slice(&code);

            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                let mut sq8_code = vec![0u8; sq8.code_size()];
                sq8.encode(xi, &mut sq8_code);
                self.clusters[cluster_id].sq8_codes.extend_from_slice(&sq8_code);
            }

            let id = self.ntotal + i;
            self.clusters[cluster_id].ids.push(id as u32);

            if id >= self.id_to_pos.len() {
                self.id_to_pos.resize(id + 1, (0, 0));
            }
            self.id_to_pos[id] = (cluster_id, pos_in_cluster);
        }

        self.ntotal += n;
        self.rebuild_cluster_offsets();
    }

    pub fn search(
        &self,
        queries: &[f32],
        n: usize,
        k: usize,
        nprobe: usize,
        refine_factor: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let use_sq8 = self.sq8_quantizers[0].is_some();
        let code_sz = self.quantizer.code_size();
        let nbits = self.quantizer.nbits;
        let use_split_lut = nbits == 4;

        let results: Vec<Vec<(usize, f32)>> = (0..n)
            .into_par_iter()
            .map(|q| {
                let query = &queries[q * self.d..(q + 1) * self.d];
                let mut nearest_buf = Vec::with_capacity(nprobe);
                self.kmeans.nearest_clusters_into(query, nprobe, &mut nearest_buf);

                let k1 = if use_sq8 {
                    (k * refine_factor).min(self.ntotal)
                } else {
                    k
                };

                let mut query_rotated = vec![0.0f32; self.rotation.d_out];
                let mut q_normalized = query.to_vec();
                crate::utils::l2_normalize(&mut q_normalized);
                self.rotation.apply_into(&q_normalized, &mut query_rotated);

                let mut heap: BinaryHeap<(FloatOrd, (usize, usize))> = BinaryHeap::with_capacity(k1);

                if use_split_lut {
                    let (lo_lut, hi_lut) = self.quantizer.build_split_lut(&query_rotated);

                    for (_, cluster_id) in &nearest_buf {
                        let cluster = &self.clusters[*cluster_id];
                        let n_vectors = cluster.ids.len();
                        if n_vectors == 0 {
                            continue;
                        }

                        for v in 0..n_vectors {
                            if v + 2 < n_vectors {
                                unsafe {
                                    prefetch_read(cluster.codes.as_ptr().add((v + 1) * code_sz));
                                }
                            }
                            let code = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                            let mut dist = 0.0f32;
                            for j in 0..code_sz {
                                let byte = code[j] as usize;
                                dist += lo_lut[j][byte & 0xF] + hi_lut[j][byte >> 4];
                                if j & 0xF == 0xF && heap.len() >= k1 && dist > heap.peek().unwrap().0 .0 {
                                    dist = f32::INFINITY;
                                    break;
                                }
                            }
                            if dist.is_finite() {
                                if heap.len() < k1 {
                                    heap.push((FloatOrd(dist), (*cluster_id, v)));
                                } else if dist < heap.peek().unwrap().0 .0 {
                                    heap.pop();
                                    heap.push((FloatOrd(dist), (*cluster_id, v)));
                                }
                            }
                        }
                    }
                } else {
                    for (_, cluster_id) in &nearest_buf {
                        let cluster = &self.clusters[*cluster_id];
                        let n_vectors = cluster.ids.len();
                        if n_vectors == 0 {
                            continue;
                        }

                        for v in 0..n_vectors {
                            let code = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                            let dist = self.quantizer.compute_distance(code, &query_rotated);

                            if heap.len() < k1 {
                                heap.push((FloatOrd(dist), (*cluster_id, v)));
                            } else if dist < heap.peek().unwrap().0 .0 {
                                heap.pop();
                                heap.push((FloatOrd(dist), (*cluster_id, v)));
                            }
                        }
                    }
                }

                let candidates: Vec<(f32, usize, usize)> = heap
                    .into_iter()
                    .map(|(FloatOrd(d), (cid, v))| (d, cid, v))
                    .collect();

                if !use_sq8 {
                    let mut result: Vec<(usize, f32)> = candidates
                        .into_iter()
                        .take(k)
                        .map(|(d, cid, v)| (self.clusters[cid].ids[v] as usize, d))
                        .collect();
                    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    return result;
                }

                let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);
                for (_, cluster_id, v) in &candidates {
                    if let Some(ref sq8) = self.sq8_quantizers[*cluster_id] {
                        let sq8_sz = sq8.code_size();
                        let sq8_code = &self.clusters[*cluster_id].sq8_codes[*v * sq8_sz..(*v + 1) * sq8_sz];
                        let refined_dist = sq8.compute_distance(sq8_code, query);
                        let id = self.clusters[*cluster_id].ids[*v] as usize;

                        if final_heap.len() < k {
                            final_heap.push((FloatOrd(refined_dist), id));
                        } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                            final_heap.pop();
                            final_heap.push((FloatOrd(refined_dist), id));
                        }
                    }
                }

                let mut result: Vec<(usize, f32)> = final_heap
                    .into_iter()
                    .map(|(FloatOrd(d), i)| (i, d))
                    .collect();
                result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                result
            })
            .collect();

        results
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    pub fn code_size(&self) -> usize {
        self.quantizer.code_size()
    }

    pub fn rebuild_id_index(&mut self) {
        self.id_to_pos.clear();
        self.id_to_pos.resize(self.ntotal, (0, 0));
        for c in 0..self.nlist {
            for (pos, &id) in self.clusters[c].ids.iter().enumerate() {
                if (id as usize) < self.id_to_pos.len() {
                    self.id_to_pos[id as usize] = (c, pos);
                }
            }
        }
        self.rebuild_cluster_offsets();
    }

    fn rebuild_cluster_offsets(&mut self) {
        self.cluster_offsets = vec![0; self.nlist + 1];
        for c in 0..self.nlist {
            self.cluster_offsets[c + 1] = self.cluster_offsets[c] + self.clusters[c].ids.len();
        }
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
