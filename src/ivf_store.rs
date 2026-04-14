//! IVF 感知存储的直接查询索引
//!
//! 基于 RocksDB 的 IVF 索引，支持直接从 RocksDB 查询，
//! 无需全量加载到内存。
//!
//! # 核心设计
//!
//! - 聚类前缀键: `[cluster_id:u16 LE][local_id:u32 LE][0x00:u16]`
//! - 仅质心和编解码器在内存中（O(nlist × d)）
//! - 查询时范围扫描 nprobe 个聚类的 codes
//! - SQ8 精排使用 multi_get 批量读取
//!
//! # RocksDB 插件机制利用
//!
//! - PrefixExtractor: 前缀 Bloom Filter
//! - Block Cache + Bloom Filter: 热数据缓存
//! - ReadOptions: 范围扫描 + 预读
//! - CuckooTable: sq8 CF 点查 O(1)
//!
//! # 索引类型
//!
//! - `RocksDBIVFIndex`: RaBitQ 1-bit 量化 IVF
//! - `RocksDBTQIVFIndex`: TurboQuant 4-bit 量化 IVF

use std::collections::BinaryHeap;
use std::path::Path;

use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, CuckooTableOptions, Direction, IteratorMode, Options,
    ReadOptions, SliceTransform, WriteBatch, DB,
};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor};

use crate::rabitq::{compute_query_factors_into, QueryFactorsData, RaBitQCodec};
use crate::sq8::SQ8Quantizer;
use crate::utils::{l2_distance_simd, FloatOrd};

const CF_RABITQ_SIGNS: &str = "rabitq_signs";
const CF_RABITQ_FACTORS: &str = "rabitq_factors";
const CF_RABITQ_SQ8: &str = "rabitq_sq8";
const CF_CENTROIDS: &str = "centroids";
const CF_CLUSTER_META: &str = "cluster_meta";

const BLOCK_CACHE_SIZE: usize = 512 * 1024 * 1024;

/// 聚类元数据
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClusterMeta {
    pub count: u32,
    pub start_local_id: u32,
}

/// RocksDB IVF 直接查询索引
///
/// 质心和编解码器在内存中，向量编码在 RocksDB 中。
/// 查询时只读取 nprobe 个聚类的数据，内存占用 O(nlist × d)。
pub struct RocksDBIVFIndex {
    db: DB,
    d: usize,
    nlist: usize,
    nb_bits: usize,
    is_inner_product: bool,
    use_sq8: bool,
    ntotal: usize,
    centroids: Vec<f32>,
    codecs: Vec<RaBitQCodec>,
    sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    cluster_counts: Vec<usize>,
    cluster_offsets: Vec<usize>,
}

impl RocksDBIVFIndex {
    /// 打开或创建 RocksDB IVF 索引
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.increase_parallelism(4);
        db_opts.set_max_background_jobs(4);
        db_opts.set_write_buffer_size(64 * 1024 * 1024);
        db_opts.set_max_write_buffer_number(3);
        db_opts.set_level_compaction_dynamic_level_bytes(true);
        db_opts.set_max_open_files(-1);
        db_opts.optimize_level_style_compaction(256 * 1024 * 1024);
        db_opts.set_use_fsync(false);

        let shared_cache = rocksdb::Cache::new_hyper_clock_cache(BLOCK_CACHE_SIZE, 0);

        let signs_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::None);
            opts.set_write_buffer_size(16 * 1024 * 1024);
            opts.set_target_file_size_base(16 * 1024 * 1024);
            opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(2));

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&shared_cache);
            block_opts.set_ribbon_filter(8.0);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            block_opts.set_block_size(2 * 1024);
            block_opts.set_format_version(5);
            block_opts.set_index_type(BlockBasedIndexType::BinarySearch);
            opts.set_block_based_table_factory(&block_opts);

            ColumnFamilyDescriptor::new(CF_RABITQ_SIGNS, opts)
        };

        let factors_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            opts.set_write_buffer_size(8 * 1024 * 1024);

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&shared_cache);
            block_opts.set_ribbon_filter(10.0);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_block_size(1 * 1024);
            block_opts.set_format_version(5);
            opts.set_block_based_table_factory(&block_opts);

            ColumnFamilyDescriptor::new(CF_RABITQ_FACTORS, opts)
        };

        let sq8_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            let mut cuckoo_opts = CuckooTableOptions::default();
            cuckoo_opts.set_hash_ratio(0.9);
            cuckoo_opts.set_max_search_depth(100);
            opts.set_cuckoo_table_factory(&cuckoo_opts);
            ColumnFamilyDescriptor::new(CF_RABITQ_SQ8, opts)
        };

        let centroids_cf = ColumnFamilyDescriptor::new(CF_CENTROIDS, Options::default());
        let cluster_meta_cf = ColumnFamilyDescriptor::new(CF_CLUSTER_META, Options::default());

        let db = DB::open_cf_descriptors(
            &db_opts,
            path,
            vec![
                ColumnFamilyDescriptor::new("default", Options::default()),
                signs_cf,
                factors_cf,
                sq8_cf,
                centroids_cf,
                cluster_meta_cf,
            ],
        )
        .map_err(|e| format!("打开RocksDB失败: {}", e))?;

        let mut index = Self {
            db,
            d: 0,
            nlist: 0,
            nb_bits: 1,
            is_inner_product: false,
            use_sq8: false,
            ntotal: 0,
            centroids: vec![],
            codecs: vec![],
            sq8_quantizers: vec![],
            cluster_counts: vec![],
            cluster_offsets: vec![],
        };

        if let Ok(meta_bytes) = index.db.get(b"meta") {
            if let Some(bytes) = meta_bytes {
                if let Ok(meta) = bincode::deserialize::<crate::store::IndexMeta>(&bytes) {
                    index.d = meta.d;
                    index.nlist = meta.nlist;
                    index.nb_bits = meta.nbits;
                    index.is_inner_product = meta.is_inner_product;
                    index.use_sq8 = meta.use_sq8;
                    index.ntotal = meta.ntotal;
                    index.centroids = meta.ivf_centroids;
                    index.codecs = (0..meta.nlist)
                        .map(|_| RaBitQCodec::new(meta.d, meta.nbits, meta.is_inner_product))
                        .collect();

                    if meta.use_sq8 {
                        index.sq8_quantizers = (0..meta.nlist)
                            .map(|c| {
                                let mut sq8 = SQ8Quantizer::new(meta.d);
                                if c < meta.ivf_sq8_vmin.len() && c < meta.ivf_sq8_vmax.len() {
                                    sq8.vmin = meta.ivf_sq8_vmin[c].clone();
                                    sq8.vmax = meta.ivf_sq8_vmax[c].clone();
                                    sq8.scale = (0..meta.d)
                                        .map(|j| (sq8.vmax[j] - sq8.vmin[j]) / 255.0)
                                        .collect();
                                }
                                Some(sq8)
                            })
                            .collect();
                    }

                    let mut cluster_counts = vec![0usize; meta.nlist];
                    if let Ok(cf) = index.cf(CF_CLUSTER_META) {
                        for c in 0..meta.nlist {
                            let meta_key = (c as u16).to_le_bytes();
                            if let Ok(Some(val)) = index.db.get_cf(cf, &meta_key[..]) {
                                if let Ok(cm) = bincode::deserialize::<ClusterMeta>(&val) {
                                    cluster_counts[c] = cm.count as usize;
                                }
                            }
                        }
                    }

                    let mut cluster_offsets = vec![0usize; meta.nlist + 1];
                    let mut offset = 0usize;
                    for c in 0..meta.nlist {
                        cluster_offsets[c] = offset;
                        offset += cluster_counts[c];
                    }
                    *cluster_offsets.last_mut().unwrap() = offset;

                    index.cluster_counts = cluster_counts;
                    index.cluster_offsets = cluster_offsets;
                }
            }
        }

        Ok(index)
    }

    /// 从内存 IVF 索引构建 RocksDB IVF 索引
    pub fn build_from_ivf(&mut self, index: &crate::ivf::RaBitQIVFIndex) -> Result<(), String> {
        self.d = index.d;
        self.nlist = index.nlist;
        self.nb_bits = index.nb_bits;
        self.is_inner_product = index.is_inner_product;
        self.use_sq8 = index.sq8_quantizers[0].is_some();
        self.ntotal = index.ntotal;

        self.centroids = index.kmeans.centroids.clone();
        self.codecs = index.codecs.clone();
        self.sq8_quantizers = index.sq8_quantizers.iter().map(|opt| opt.clone()).collect();
        self.cluster_counts = index.clusters.iter().map(|c| c.ids.len()).collect();
        let mut offset = 0usize;
        self.cluster_offsets = index
            .clusters
            .iter()
            .map(|c| {
                let o = offset;
                offset += c.ids.len();
                o
            })
            .collect();

        let signs_sz = index.codecs[0].signs_size();
        let code_sz = index.codecs[0].code_size();
        let mut batch = WriteBatch::default();

        let cf_signs = self.cf(CF_RABITQ_SIGNS)?;
        let cf_factors = self.cf(CF_RABITQ_FACTORS)?;
        let cf_sq8 = self.cf(CF_RABITQ_SQ8)?;
        let cf_centroids = self.cf(CF_CENTROIDS)?;
        let cf_cluster_meta = self.cf(CF_CLUSTER_META)?;

        for c in 0..index.nlist {
            let cluster = &index.clusters[c];
            let n_vectors = cluster.ids.len();

            let meta = ClusterMeta {
                count: n_vectors as u32,
                start_local_id: 0,
            };
            let meta_bytes =
                bincode::serialize(&meta).map_err(|e| format!("序列化ClusterMeta失败: {}", e))?;
            let meta_key = (c as u16).to_le_bytes();
            batch.put_cf(cf_cluster_meta, &meta_key[..], &meta_bytes);

            for v in 0..n_vectors {
                let key = Self::cluster_key(c as u16, v as u32);
                let code_val = &cluster.codes[v * code_sz..(v + 1) * code_sz];

                batch.put_cf(cf_signs, &key, &code_val[..signs_sz]);
                batch.put_cf(cf_factors, &key, &code_val[signs_sz..signs_sz + 8]);

                if self.use_sq8 {
                    let sq8_sz = index.d;
                    let sq8_key = (cluster.ids[v] as u64).to_le_bytes();
                    let sq8_val = &cluster.sq8_codes[v * sq8_sz..(v + 1) * sq8_sz];
                    batch.put_cf(cf_sq8, &sq8_key, sq8_val);
                }
            }

            let centroid_key = format!("c{}", c);
            let centroid_val = &index.cluster_centroids[c];
            let encoded =
                bincode::serialize(centroid_val).map_err(|e| format!("序列化质心失败: {}", e))?;
            batch.put_cf(cf_centroids, centroid_key.as_bytes(), &encoded);
        }

        let index_meta = crate::store::IndexMeta {
            index_type: crate::store::IndexType::RaBitQIVF,
            d: index.d,
            ntotal: index.ntotal,
            nbits: index.nb_bits,
            use_sq8: self.use_sq8,
            is_inner_product: index.is_inner_product,
            nlist: index.nlist,
            hadamard_seed: 0,
            kmeans_niter: index.kmeans.niter,
            sq8_vmin: vec![],
            sq8_vmax: vec![],
            rabitq_centroid: vec![],
            ivf_centroids: index.kmeans.centroids.clone(),
            ivf_cluster_ids: index.clusters.iter().map(|c| c.ids.clone()).collect(),
            ivf_sq8_vmin: if self.use_sq8 {
                index
                    .sq8_quantizers
                    .iter()
                    .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmin.clone()))
                    .collect()
            } else {
                vec![]
            },
            ivf_sq8_vmax: if self.use_sq8 {
                index
                    .sq8_quantizers
                    .iter()
                    .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmax.clone()))
                    .collect()
            } else {
                vec![]
            },
            storage_version: 2,
        };
        let meta_bytes =
            bincode::serialize(&index_meta).map_err(|e| format!("序列化元数据失败: {}", e))?;
        batch.put(b"meta", &meta_bytes);

        self.db
            .write(batch)
            .map_err(|e| format!("批量写入失败: {}", e))?;

        Ok(())
    }

    /// 直接从 RocksDB 执行 IVF 查询
    ///
    /// 只读取 nprobe 个聚类的数据，内存占用 O(nlist × d)。
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        refine_factor: usize,
    ) -> Vec<(usize, f32)> {
        if self.d == 0 || self.nlist == 0 {
            return vec![];
        }

        let nearest_clusters = self.nearest_clusters(query, nprobe);

        let k1 = if self.use_sq8 {
            (k * refine_factor).min(self.ntotal)
        } else {
            k
        };

        let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);
        let cf_signs = match self.cf(CF_RABITQ_SIGNS) {
            Ok(cf) => cf,
            Err(_) => return vec![],
        };

        let mut query_fac = QueryFactorsData::with_capacity(self.d);

        for (_, cluster_id) in &nearest_clusters {
            compute_query_factors_into(
                query,
                self.d,
                Some(&self.centroids[*cluster_id * self.d..(*cluster_id + 1) * self.d]),
                &mut query_fac,
            );

            let start = Self::cluster_key(*cluster_id as u16, 0);
            let end = Self::cluster_key((*cluster_id + 1) as u16, 0);

            let mut read_opts = ReadOptions::default();
            read_opts.set_verify_checksums(false);
            read_opts.fill_cache(true);
            read_opts.set_readahead_size(256 * 1024);
            read_opts.set_async_io(true);
            read_opts.set_iterate_lower_bound(&start);
            read_opts.set_iterate_upper_bound(&end);

            let iter = self.db.iterator_cf_opt(
                cf_signs,
                read_opts,
                IteratorMode::From(&start, Direction::Forward),
            );

            let mut local_id: usize = 0;
            for item in iter {
                if let Ok((_key, signs_val)) = item {
                    let global_id = self.decode_global_id(*cluster_id, local_id);
                    let dot_qo = self.codecs[*cluster_id]
                        .compute_distance_signs_only(&signs_val, &query_fac);

                    if heap.len() < k1 {
                        heap.push((FloatOrd(dot_qo), global_id));
                    } else if dot_qo < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((FloatOrd(dot_qo), global_id));
                    }
                    local_id += 1;
                }
            }
        }

        let candidates: Vec<(f32, usize)> =
            heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

        let cf_factors = match self.cf(CF_RABITQ_FACTORS) {
            Ok(cf) => cf,
            Err(_) => return vec![],
        };

        let mut refined_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);
        for (dot_qo, idx) in &candidates {
            let cluster_id = self.find_cluster_for_id(*idx);
            if let Some(cid) = cluster_id {
                compute_query_factors_into(
                    query,
                    self.d,
                    Some(&self.centroids[cid * self.d..(cid + 1) * self.d]),
                    &mut query_fac,
                );

                let local_id = *idx - self.cluster_offset(cid);
                let key = Self::cluster_key(cid as u16, local_id as u32);
                if let Ok(Some(factors_val)) = self.db.get_pinned_cf(cf_factors, &key) {
                    let or_minus_c_l2sqr =
                        f32::from_le_bytes(factors_val[..4].try_into().unwrap_or([0u8; 4]));
                    let dp_multiplier =
                        f32::from_le_bytes(factors_val[4..8].try_into().unwrap_or([0u8; 4]));
                    let dist = self.codecs[cid].compute_distance_with_factors(
                        *dot_qo,
                        or_minus_c_l2sqr,
                        dp_multiplier,
                        &query_fac,
                    );

                    if !self.use_sq8 {
                        if refined_heap.len() < k {
                            refined_heap.push((FloatOrd(dist), *idx));
                        } else if dist < refined_heap.peek().unwrap().0 .0 {
                            refined_heap.pop();
                            refined_heap.push((FloatOrd(dist), *idx));
                        }
                    } else {
                        if refined_heap.len() < k1 {
                            refined_heap.push((FloatOrd(dist), *idx));
                        } else if dist < refined_heap.peek().unwrap().0 .0 {
                            refined_heap.pop();
                            refined_heap.push((FloatOrd(dist), *idx));
                        }
                    }
                }
            }
        }

        if !self.use_sq8 {
            let mut result: Vec<(usize, f32)> = refined_heap
                .into_iter()
                .map(|(FloatOrd(d), i)| (i, d))
                .collect();
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return result;
        }

        let cf_sq8 = match self.cf(CF_RABITQ_SQ8) {
            Ok(cf) => cf,
            Err(_) => return vec![],
        };

        let sq8_keys: Vec<[u8; 8]> = refined_heap
            .iter()
            .map(|(_, idx)| (*idx as u64).to_le_bytes())
            .collect();

        let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

        for (i, (_, idx)) in candidates.iter().enumerate() {
            let cluster_id = self.find_cluster_for_id(*idx);
            if let Some(cid) = cluster_id {
                if let Some(ref sq8) = self.sq8_quantizers[cid] {
                    let key = &sq8_keys[i];
                    if let Ok(Some(val)) = self.db.get_pinned_cf(cf_sq8, key) {
                        let refined_dist = sq8.compute_distance(&val, query);
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

        let mut result: Vec<(usize, f32)> = final_heap
            .into_iter()
            .map(|(FloatOrd(d), i)| (i, d))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result
    }

    fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<(f32, usize)> {
        let mut dists: Vec<(f32, usize)> = (0..self.nlist)
            .map(|c| {
                let centroid = &self.centroids[c * self.d..(c + 1) * self.d];
                (l2_distance_simd(query, centroid), c)
            })
            .collect();
        let np = nprobe.min(self.nlist);
        if np < self.nlist {
            dists.select_nth_unstable_by(np, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        dists.truncate(np);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists
    }

    fn find_cluster_for_id(&self, id: usize) -> Option<usize> {
        if self.cluster_offsets.is_empty() {
            return None;
        }
        let idx = self.cluster_offsets.partition_point(|&offset| offset <= id);
        if idx == 0 {
            return None;
        }
        let c = idx - 1;
        if id < self.cluster_offsets[c] + self.cluster_counts[c] {
            Some(c)
        } else {
            None
        }
    }

    fn cluster_offset(&self, cluster_id: usize) -> usize {
        if cluster_id < self.cluster_offsets.len() {
            self.cluster_offsets[cluster_id]
        } else {
            0
        }
    }

    fn decode_global_id(&self, cluster_id: usize, local_id: usize) -> usize {
        self.cluster_offset(cluster_id) + local_id
    }

    fn cluster_key(cluster_id: u16, local_id: u32) -> [u8; 8] {
        let mut key = [0u8; 8];
        key[0..2].copy_from_slice(&cluster_id.to_le_bytes());
        key[2..6].copy_from_slice(&local_id.to_le_bytes());
        key
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily, String> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| format!("Column Family '{}' 不存在", name))
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }
}

// ==================== TurboQuant IVF 持久化索引 ====================

const CF_TQ_CODES: &str = "tq_codes";
const CF_TQ_SQ8: &str = "tq_sq8";

/// RocksDB TurboQuant IVF 直接查询索引
///
/// 使用 TurboQuant 4-bit 量化 + Split LUT 的 IVF 索引。
/// 相比 RaBitQ IVF:
/// - 4-bit 量化精度更高，粗排召回率更高
/// - Split LUT (8KB) L1 常驻，减少 cache miss
/// - 支持 early stop
pub struct RocksDBTQIVFIndex {
    db: DB,
    d: usize,
    nlist: usize,
    nbits: usize,
    use_sq8: bool,
    ntotal: usize,
    centroids: Vec<f32>,
    quantizer: crate::lloyd_max::LloydMaxQuantizer,
    rotation: crate::hadamard::HadamardRotation,
    sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    cluster_counts: Vec<usize>,
    cluster_offsets: Vec<usize>,
}

impl RocksDBTQIVFIndex {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.increase_parallelism(4);
        db_opts.set_max_background_jobs(4);
        db_opts.set_write_buffer_size(64 * 1024 * 1024);
        db_opts.set_max_write_buffer_number(3);
        db_opts.set_level_compaction_dynamic_level_bytes(true);
        db_opts.set_max_open_files(-1);
        db_opts.optimize_level_style_compaction(256 * 1024 * 1024);
        db_opts.set_use_fsync(false);

        let shared_cache = rocksdb::Cache::new_hyper_clock_cache(BLOCK_CACHE_SIZE, 0);

        let codes_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::None);
            opts.set_write_buffer_size(16 * 1024 * 1024);
            opts.set_target_file_size_base(16 * 1024 * 1024);
            opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(2));

            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&shared_cache);
            block_opts.set_ribbon_filter(8.0);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            block_opts.set_block_size(4 * 1024);
            block_opts.set_format_version(5);
            block_opts.set_index_type(BlockBasedIndexType::BinarySearch);
            opts.set_block_based_table_factory(&block_opts);

            ColumnFamilyDescriptor::new(CF_TQ_CODES, opts)
        };

        let sq8_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            let mut cuckoo_opts = CuckooTableOptions::default();
            cuckoo_opts.set_hash_ratio(0.9);
            cuckoo_opts.set_max_search_depth(100);
            opts.set_cuckoo_table_factory(&cuckoo_opts);
            ColumnFamilyDescriptor::new(CF_TQ_SQ8, opts)
        };

        let centroids_cf = ColumnFamilyDescriptor::new(CF_CENTROIDS, Options::default());
        let cluster_meta_cf = ColumnFamilyDescriptor::new(CF_CLUSTER_META, Options::default());

        let db = DB::open_cf_descriptors(
            &db_opts,
            path,
            vec![
                ColumnFamilyDescriptor::new("default", Options::default()),
                codes_cf,
                sq8_cf,
                centroids_cf,
                cluster_meta_cf,
            ],
        )
        .map_err(|e| format!("打开RocksDB失败: {}", e))?;

        let mut index = Self {
            db,
            d: 0,
            nlist: 0,
            nbits: 4,
            use_sq8: false,
            ntotal: 0,
            centroids: vec![],
            quantizer: crate::lloyd_max::LloydMaxQuantizer::new(128, 4),
            rotation: crate::hadamard::HadamardRotation::new(128, 12345),
            sq8_quantizers: vec![],
            cluster_counts: vec![],
            cluster_offsets: vec![],
        };

        if let Ok(meta_bytes) = index.db.get(b"meta") {
            if let Some(bytes) = meta_bytes {
                if let Ok(meta) = bincode::deserialize::<crate::store::IndexMeta>(&bytes) {
                    let d_rotated = crate::utils::next_power_of_2(meta.d);
                    index.d = meta.d;
                    index.nlist = meta.nlist;
                    index.nbits = meta.nbits;
                    index.use_sq8 = meta.use_sq8;
                    index.ntotal = meta.ntotal;
                    index.quantizer =
                        crate::lloyd_max::LloydMaxQuantizer::new(d_rotated, meta.nbits);
                    index.rotation =
                        crate::hadamard::HadamardRotation::new(meta.d, meta.hadamard_seed);
                    index.centroids = meta.ivf_centroids;

                    if meta.use_sq8 {
                        index.sq8_quantizers = (0..meta.nlist)
                            .map(|c| {
                                let mut sq8 = SQ8Quantizer::new(meta.d);
                                if c < meta.ivf_sq8_vmin.len() && c < meta.ivf_sq8_vmax.len() {
                                    sq8.vmin = meta.ivf_sq8_vmin[c].clone();
                                    sq8.vmax = meta.ivf_sq8_vmax[c].clone();
                                    sq8.scale = (0..meta.d)
                                        .map(|j| (sq8.vmax[j] - sq8.vmin[j]) / 255.0)
                                        .collect();
                                }
                                Some(sq8)
                            })
                            .collect();
                    }

                    let mut cluster_counts = vec![0usize; meta.nlist];
                    if let Ok(cf) = index.cf(CF_CLUSTER_META) {
                        for c in 0..meta.nlist {
                            let meta_key = (c as u16).to_le_bytes();
                            if let Ok(Some(val)) = index.db.get_cf(cf, &meta_key[..]) {
                                if let Ok(cm) = bincode::deserialize::<ClusterMeta>(&val) {
                                    cluster_counts[c] = cm.count as usize;
                                }
                            }
                        }
                    }

                    let mut cluster_offsets = vec![0usize; meta.nlist + 1];
                    let mut offset = 0usize;
                    for c in 0..meta.nlist {
                        cluster_offsets[c] = offset;
                        offset += cluster_counts[c];
                    }
                    *cluster_offsets.last_mut().unwrap() = offset;

                    index.cluster_counts = cluster_counts;
                    index.cluster_offsets = cluster_offsets;
                }
            }
        }

        Ok(index)
    }

    pub fn build_from_ivf(&mut self, index: &crate::ivf::TurboQuantIVFIndex) -> Result<(), String> {
        self.d = index.d;
        self.nlist = index.nlist;
        self.nbits = index.nbits;
        self.use_sq8 = index.sq8_quantizers[0].is_some();
        self.ntotal = index.ntotal;

        self.centroids = index.cluster_centroids.iter().flatten().copied().collect();
        self.quantizer = index.quantizer.clone();
        self.rotation = index.rotation.clone();
        self.sq8_quantizers = index.sq8_quantizers.iter().map(|opt| opt.clone()).collect();
        self.cluster_counts = index.clusters.iter().map(|c| c.ids.len()).collect();
        let mut offset = 0usize;
        self.cluster_offsets = index
            .clusters
            .iter()
            .map(|c| {
                let o = offset;
                offset += c.ids.len();
                o
            })
            .collect();

        let code_sz = self.quantizer.code_size();
        let mut batch = WriteBatch::default();

        let cf_codes = self.cf(CF_TQ_CODES)?;
        let cf_sq8 = self.cf(CF_TQ_SQ8)?;
        let cf_centroids = self.cf(CF_CENTROIDS)?;
        let cf_cluster_meta = self.cf(CF_CLUSTER_META)?;

        for c in 0..index.nlist {
            let cluster = &index.clusters[c];
            let n_vectors = cluster.ids.len();

            let meta = ClusterMeta {
                count: n_vectors as u32,
                start_local_id: 0,
            };
            let meta_bytes =
                bincode::serialize(&meta).map_err(|e| format!("序列化ClusterMeta失败: {}", e))?;
            let meta_key = (c as u16).to_le_bytes();
            batch.put_cf(cf_cluster_meta, &meta_key[..], &meta_bytes);

            for v in 0..n_vectors {
                let key = Self::cluster_key(c as u16, v as u32);
                let code_val = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                batch.put_cf(cf_codes, &key, code_val);

                if self.use_sq8 {
                    let sq8_sz = index.d;
                    let sq8_key = (cluster.ids[v] as u64).to_le_bytes();
                    let sq8_val = &cluster.sq8_codes[v * sq8_sz..(v + 1) * sq8_sz];
                    batch.put_cf(cf_sq8, &sq8_key, sq8_val);
                }
            }

            let centroid_key = format!("c{}", c);
            let centroid_val = &index.cluster_centroids[c];
            let encoded =
                bincode::serialize(centroid_val).map_err(|e| format!("序列化质心失败: {}", e))?;
            batch.put_cf(cf_centroids, centroid_key.as_bytes(), &encoded);
        }

        let index_meta = crate::store::IndexMeta {
            index_type: crate::store::IndexType::TurboQuant,
            d: index.d,
            ntotal: index.ntotal,
            nbits: index.nbits,
            use_sq8: self.use_sq8,
            is_inner_product: false,
            nlist: index.nlist,
            hadamard_seed: 12345,
            kmeans_niter: 20,
            sq8_vmin: vec![],
            sq8_vmax: vec![],
            rabitq_centroid: vec![],
            ivf_centroids: index.cluster_centroids.iter().flatten().copied().collect(),
            ivf_cluster_ids: index.clusters.iter().map(|c| c.ids.clone()).collect(),
            ivf_sq8_vmin: if self.use_sq8 {
                index
                    .sq8_quantizers
                    .iter()
                    .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmin.clone()))
                    .collect()
            } else {
                vec![]
            },
            ivf_sq8_vmax: if self.use_sq8 {
                index
                    .sq8_quantizers
                    .iter()
                    .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmax.clone()))
                    .collect()
            } else {
                vec![]
            },
            storage_version: 3,
        };
        let meta_bytes =
            bincode::serialize(&index_meta).map_err(|e| format!("序列化元数据失败: {}", e))?;
        batch.put(b"meta", &meta_bytes);

        self.db
            .write(batch)
            .map_err(|e| format!("批量写入失败: {}", e))?;

        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        refine_factor: usize,
    ) -> Vec<(usize, f32)> {
        if self.d == 0 || self.nlist == 0 {
            return vec![];
        }

        let nearest_clusters = self.nearest_clusters(query, nprobe);

        let k1 = if self.use_sq8 {
            (k * refine_factor).min(self.ntotal)
        } else {
            k
        };

        let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);
        let cf_codes = match self.cf(CF_TQ_CODES) {
            Ok(cf) => cf,
            Err(_) => return vec![],
        };

        let mut query_normalized = query.to_vec();
        crate::utils::l2_normalize(&mut query_normalized);
        let mut query_rotated = vec![0.0f32; self.rotation.d_out];
        self.rotation
            .apply_into(&query_normalized, &mut query_rotated);

        let (lo_lut, hi_lut) = self.quantizer.build_split_lut(&query_rotated);
        let code_sz = self.quantizer.code_size();

        for (_, cluster_id) in &nearest_clusters {
            let start = Self::cluster_key(*cluster_id as u16, 0);
            let end = Self::cluster_key((*cluster_id + 1) as u16, 0);

            let mut read_opts = ReadOptions::default();
            read_opts.set_verify_checksums(false);
            read_opts.fill_cache(true);
            read_opts.set_readahead_size(256 * 1024);
            read_opts.set_async_io(true);
            read_opts.set_iterate_lower_bound(&start);
            read_opts.set_iterate_upper_bound(&end);

            let iter = self.db.iterator_cf_opt(
                cf_codes,
                read_opts,
                IteratorMode::From(&start, Direction::Forward),
            );

            let mut local_id: usize = 0;
            for item in iter {
                if let Ok((_key, code_val)) = item {
                    let global_id = self.decode_global_id(*cluster_id, local_id);

                    let mut dist = 0.0f32;
                    for j in 0..code_sz {
                        let byte = code_val[j] as usize;
                        dist += lo_lut[j][byte & 0xF] + hi_lut[j][byte >> 4];
                        if j & 0xF == 0xF && heap.len() >= k1 && dist > heap.peek().unwrap().0 .0 {
                            dist = f32::INFINITY;
                            break;
                        }
                    }

                    if dist.is_finite() {
                        if heap.len() < k1 {
                            heap.push((FloatOrd(dist), global_id));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((FloatOrd(dist), global_id));
                        }
                    }
                    local_id += 1;
                }
            }
        }

        if !self.use_sq8 {
            let mut result: Vec<(usize, f32)> =
                heap.into_iter().map(|(FloatOrd(d), i)| (i, d)).collect();
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return result;
        }

        let candidates: Vec<(f32, usize)> =
            heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

        let cf_sq8 = match self.cf(CF_TQ_SQ8) {
            Ok(cf) => cf,
            Err(_) => return vec![],
        };

        let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

        for (_, idx) in &candidates {
            let cluster_id = self.find_cluster_for_id(*idx);
            if let Some(cid) = cluster_id {
                if let Some(ref sq8) = self.sq8_quantizers[cid] {
                    let key = (*idx as u64).to_le_bytes();
                    if let Ok(Some(val)) = self.db.get_pinned_cf(cf_sq8, &key) {
                        let refined_dist = sq8.compute_distance(&val, query);
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

        let mut result: Vec<(usize, f32)> = final_heap
            .into_iter()
            .map(|(FloatOrd(d), i)| (i, d))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result
    }

    fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<(f32, usize)> {
        let mut dists: Vec<(f32, usize)> = (0..self.nlist)
            .map(|c| {
                let centroid = &self.centroids[c * self.d..(c + 1) * self.d];
                (l2_distance_simd(query, centroid), c)
            })
            .collect();
        let np = nprobe.min(self.nlist);
        if np < self.nlist {
            dists.select_nth_unstable_by(np, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        dists.truncate(np);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists
    }

    fn find_cluster_for_id(&self, id: usize) -> Option<usize> {
        if self.cluster_offsets.is_empty() {
            return None;
        }
        let idx = self.cluster_offsets.partition_point(|&offset| offset <= id);
        if idx == 0 {
            return None;
        }
        let c = idx - 1;
        if id < self.cluster_offsets[c] + self.cluster_counts[c] {
            Some(c)
        } else {
            None
        }
    }

    fn cluster_offset(&self, cluster_id: usize) -> usize {
        if cluster_id < self.cluster_offsets.len() {
            self.cluster_offsets[cluster_id]
        } else {
            0
        }
    }

    fn decode_global_id(&self, cluster_id: usize, local_id: usize) -> usize {
        self.cluster_offset(cluster_id) + local_id
    }

    fn cluster_key(cluster_id: u16, local_id: u32) -> [u8; 8] {
        let mut key = [0u8; 8];
        key[0..2].copy_from_slice(&cluster_id.to_le_bytes());
        key[2..6].copy_from_slice(&local_id.to_le_bytes());
        key
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily, String> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| format!("Column Family '{}' 不存在", name))
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }
}
