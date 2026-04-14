//! RocksDB 持久化模块
//!
//! 基于 RocksDB Column Family 插件机制实现向量索引的持久化存储。
//!
//! # V2 存储架构 (按数据特征分 CF)
//!
//! ```text
//! RocksDB
//! ├── "default" CF         → 索引元数据 (IndexMeta)
//! │
//! ├── RaBitQ 专用 (小值、无压缩、小块):
//! │   ├── "rabitq_signs"   → 符号位 (d/8 B/向量, 无压缩, 2KB块, 全局Bloom)
//! │   ├── "rabitq_factors" → SignBitFactors (8 B/向量, LZ4, 1KB块)
//! │   └── "rabitq_sq8"     → SQ8精排 (d B/向量, CuckooTable O(1))
//! │
//! ├── TurboQuant 专用 (中值、LZ4、大块):
//! │   ├── "tq_codes"       → Lloyd-Max编码 (LZ4, 8KB块, TwoLevelIndex)
//! │   └── "tq_sq8"         → SQ8精排 (CuckooTable O(1))
//! │
//! ├── IVF 共用:
//! │   ├── "centroids"      → 质心数据
//! │   └── "cluster_meta"   → 聚类元数据
//! │
//! └── V1 兼容 (旧格式):
//!     ├── "codes"          → 旧格式编码
//!     ├── "sq8"            → 旧格式SQ8
//!     └── "factors"        → 旧格式因子
//! ```
//!
//! # 设计原则
//!
//! - RaBitQ 符号位 16B/向量，不可压缩，无压缩 + 2KB 小块
//! - TurboQuant 编码 64B/向量，可压缩，LZ4 + 8KB 块
//! - SQ8 精排数据，CuckooTable O(1) 点查
//! - 冷热分离: signs(热) vs factors(冷) vs sq8(冷)

use std::path::Path;

use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, CuckooTableOptions, Env,
    ReadOptions, SliceTransform, WriteBatch, DB,
};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options};
use serde::{Deserialize, Serialize};

use crate::ivf::RaBitQIVFIndex;
use crate::rabitq::RaBitQFlatIndex;
use crate::sq8::SQ8Quantizer;
use crate::turboquant::TurboQuantFlatIndex;

const CF_RABITQ_SIGNS: &str = "rabitq_signs";
const CF_RABITQ_FACTORS: &str = "rabitq_factors";
const CF_RABITQ_SQ8: &str = "rabitq_sq8";
const CF_TQ_CODES: &str = "tq_codes";
const CF_TQ_SQ8: &str = "tq_sq8";
const CF_CENTROIDS: &str = "centroids";
const CF_CLUSTER_META: &str = "cluster_meta";

const CF_CODES_V1: &str = "codes";
const CF_SQ8_V1: &str = "sq8";
const CF_FACTORS_V1: &str = "factors";

const BLOCK_CACHE_SIZE: usize = 512 * 1024 * 1024;
const BLOOM_BITS_PER_KEY: f64 = 10.0;
const RATE_LIMITER_BYTES_PER_SEC: i64 = 100 * 1024 * 1024;

/// 索引类型标识
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    TurboQuant,
    RaBitQFlat,
    RaBitQIVF,
}

/// 索引元数据
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexMeta {
    pub index_type: IndexType,
    pub d: usize,
    pub ntotal: usize,
    pub nbits: usize,
    pub use_sq8: bool,
    pub is_inner_product: bool,
    pub nlist: usize,
    pub hadamard_seed: u64,
    pub kmeans_niter: usize,
    pub sq8_vmin: Vec<f32>,
    pub sq8_vmax: Vec<f32>,
    pub rabitq_centroid: Vec<f32>,
    pub ivf_centroids: Vec<f32>,
    pub ivf_cluster_ids: Vec<Vec<u32>>,
    pub ivf_sq8_vmin: Vec<Vec<f32>>,
    pub ivf_sq8_vmax: Vec<Vec<f32>>,
    pub storage_version: u32,
}

/// RocksDB 性能统计
#[derive(Debug, Default)]
pub struct RocksDBStats {
    pub block_cache_hit_rate: f64,
    pub bloom_filter_useful: u64,
    pub bloom_filter_checked: u64,
    pub get_p50_us: f64,
    pub block_read_count: u64,
}

/// RocksDB 向量存储
pub struct VectorStore {
    db: DB,
}

impl VectorStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.increase_parallelism(4);
        db_opts.set_max_background_jobs(4);
        db_opts.set_write_buffer_size(64 * 1024 * 1024);
        db_opts.set_max_write_buffer_number(3);
        db_opts.set_target_file_size_base(64 * 1024 * 1024);
        db_opts.set_level_compaction_dynamic_level_bytes(true);
        db_opts.set_max_open_files(-1);
        db_opts.optimize_level_style_compaction(256 * 1024 * 1024);
        db_opts.enable_statistics();
        db_opts.set_stats_dump_period_sec(60);
        db_opts.set_ratelimiter(RATE_LIMITER_BYTES_PER_SEC, 100_000, 10);

        let mut env = Env::new().map_err(|e| format!("创建Env失败: {}", e))?;
        env.set_background_threads(4);
        env.set_high_priority_background_threads(2);
        env.set_low_priority_background_threads(2);
        db_opts.set_env(&env);

        let cache = rocksdb::Cache::new_hyper_clock_cache(BLOCK_CACHE_SIZE, 0);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            Self::build_rabitq_signs_cf(&cache),
            Self::build_rabitq_factors_cf(&cache),
            Self::build_rabitq_sq8_cf(),
            Self::build_tq_codes_cf(&cache),
            Self::build_tq_sq8_cf(),
            ColumnFamilyDescriptor::new(CF_CENTROIDS, Options::default()),
            ColumnFamilyDescriptor::new(CF_CLUSTER_META, Options::default()),
            Self::build_v1_codes_cf(&cache),
            Self::build_v1_sq8_cf(),
            ColumnFamilyDescriptor::new(CF_FACTORS_V1, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors)
            .map_err(|e| format!("打开RocksDB失败: {}", e))?;

        Ok(Self { db })
    }

    // ─── RaBitQ 专用 CF 配置 ───

    fn build_rabitq_signs_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_write_buffer_size(16 * 1024 * 1024);
        opts.set_target_file_size_base(16 * 1024 * 1024);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(2));

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        block_opts.set_hybrid_ribbon_filter(8.0, 1);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        block_opts.set_block_size(2 * 1024);
        block_opts.set_format_version(5);
        block_opts.set_index_type(BlockBasedIndexType::BinarySearch);
        opts.set_block_based_table_factory(&block_opts);

        ColumnFamilyDescriptor::new(CF_RABITQ_SIGNS, opts)
    }

    fn build_rabitq_factors_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024);
        opts.set_target_file_size_base(8 * 1024 * 1024);

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        block_opts.set_hybrid_ribbon_filter(BLOOM_BITS_PER_KEY, 1);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_block_size(1 * 1024);
        block_opts.set_format_version(5);
        opts.set_block_based_table_factory(&block_opts);

        ColumnFamilyDescriptor::new(CF_RABITQ_FACTORS, opts)
    }

    fn build_rabitq_sq8_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let mut cuckoo_opts = CuckooTableOptions::default();
        cuckoo_opts.set_hash_ratio(0.9);
        cuckoo_opts.set_max_search_depth(100);
        opts.set_cuckoo_table_factory(&cuckoo_opts);
        ColumnFamilyDescriptor::new(CF_RABITQ_SQ8, opts)
    }

    // ─── TurboQuant 专用 CF 配置 ───

    fn build_tq_codes_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(64 * 1024 * 1024);
        opts.set_max_write_buffer_number(3);
        opts.set_target_file_size_base(64 * 1024 * 1024);

        opts.set_compaction_filter("vector_cleanup", |_level, _key, value| {
            if value.is_empty() {
                rocksdb::CompactionDecision::Remove
            } else {
                rocksdb::CompactionDecision::Keep
            }
        });

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        block_opts.set_hybrid_ribbon_filter(BLOOM_BITS_PER_KEY, 1);
        block_opts.set_partition_filters(true);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        block_opts.set_block_size(8 * 1024);
        block_opts.set_format_version(5);
        block_opts.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
        opts.set_block_based_table_factory(&block_opts);

        ColumnFamilyDescriptor::new(CF_TQ_CODES, opts)
    }

    fn build_tq_sq8_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let mut cuckoo_opts = CuckooTableOptions::default();
        cuckoo_opts.set_hash_ratio(0.9);
        cuckoo_opts.set_max_search_depth(100);
        opts.set_cuckoo_table_factory(&cuckoo_opts);
        ColumnFamilyDescriptor::new(CF_TQ_SQ8, opts)
    }

    // ─── V1 兼容 CF 配置 ───

    fn build_v1_codes_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(2));

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        block_opts.set_hybrid_ribbon_filter(BLOOM_BITS_PER_KEY, 1);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_block_size(4 * 1024);
        block_opts.set_format_version(5);
        block_opts.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
        opts.set_block_based_table_factory(&block_opts);

        ColumnFamilyDescriptor::new(CF_CODES_V1, opts)
    }

    fn build_v1_sq8_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let mut cuckoo_opts = CuckooTableOptions::default();
        cuckoo_opts.set_hash_ratio(0.9);
        cuckoo_opts.set_max_search_depth(100);
        opts.set_cuckoo_table_factory(&cuckoo_opts);
        ColumnFamilyDescriptor::new(CF_SQ8_V1, opts)
    }

    // ─── 通用工具方法 ───

    fn read_opts_for_load() -> ReadOptions {
        let mut opts = ReadOptions::default();
        opts.set_verify_checksums(false);
        opts.fill_cache(true);
        opts.set_readahead_size(64 * 1024);
        opts.set_async_io(true);
        opts
    }

    pub fn get_rocksdb_stats(&self) -> RocksDBStats {
        RocksDBStats::default()
    }

    fn save_meta(&self, meta: &IndexMeta) -> Result<(), String> {
        let encoded = bincode::serialize(meta)
            .map_err(|e| format!("序列化元数据失败: {}", e))?;
        self.db
            .put(b"meta", &encoded)
            .map_err(|e| format!("写入元数据失败: {}", e))
    }

    fn load_meta(&self) -> Result<IndexMeta, String> {
        let val = self
            .db
            .get(b"meta")
            .map_err(|e| format!("读取元数据失败: {}", e))?
            .ok_or_else(|| "元数据不存在".to_string())?;
        bincode::deserialize(&val).map_err(|e| format!("反序列化元数据失败: {}", e))
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily, String> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| format!("Column Family '{}' 不存在", name))
    }

    // ─── V2: TurboQuant 专用存储 ───

    /// 保存 TurboQuant 索引 (V2: tq_codes + tq_sq8)
    pub fn save_turboquant(&self, index: &TurboQuantFlatIndex) -> Result<(), String> {
        let meta = IndexMeta {
            index_type: IndexType::TurboQuant,
            d: index.d,
            ntotal: index.ntotal,
            nbits: index.nbits,
            use_sq8: index.sq8.is_some(),
            is_inner_product: false,
            nlist: 0,
            hadamard_seed: 12345,
            kmeans_niter: 0,
            sq8_vmin: index.sq8.as_ref().map_or(vec![], |s| s.vmin.clone()),
            sq8_vmax: index.sq8.as_ref().map_or(vec![], |s| s.vmax.clone()),
            rabitq_centroid: vec![],
            ivf_centroids: vec![],
            ivf_cluster_ids: vec![],
            ivf_sq8_vmin: vec![],
            ivf_sq8_vmax: vec![],
            storage_version: 2,
        };
        self.save_meta(&meta)?;

        let code_sz = index.code_size();
        let mut batch = WriteBatch::default();
        let cf = self.cf(CF_TQ_CODES)?;
        for i in 0..index.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = &index.codes[i * code_sz..(i + 1) * code_sz];
            batch.put_cf(cf, &key, val);
        }

        if index.sq8.is_some() {
            let sq8_sz = index.d;
            let cf = self.cf(CF_TQ_SQ8)?;
            for i in 0..index.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = &index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz];
                batch.put_cf(cf, &key, val);
            }
        }

        self.db.write(batch).map_err(|e| format!("批量写入失败: {}", e))?;
        Ok(())
    }

    /// 加载 TurboQuant 索引
    pub fn load_turboquant(&self) -> Result<TurboQuantFlatIndex, String> {
        let meta = self.load_meta()?;
        if meta.index_type != IndexType::TurboQuant {
            return Err(format!("索引类型不匹配: 期望 TurboQuant, 实际 {:?}", meta.index_type));
        }

        let mut index = TurboQuantFlatIndex::new(meta.d, meta.nbits, meta.use_sq8);
        index.ntotal = meta.ntotal;

        let code_sz = index.code_size();
        index.codes.resize(meta.ntotal * code_sz, 0);

        let read_opts = Self::read_opts_for_load();
        let cf_name = if meta.storage_version >= 2 { CF_TQ_CODES } else { CF_CODES_V1 };
        let cf = self.cf(cf_name)?;
        for i in 0..meta.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = self
                .db
                .get_pinned_cf_opt(cf, &key, &read_opts)
                .map_err(|e| format!("读取codes失败: {}", e))?
                .ok_or_else(|| format!("codes缺失: id={}", i))?;
            index.codes[i * code_sz..(i + 1) * code_sz].copy_from_slice(&val[..code_sz]);
        }

        if meta.use_sq8 {
            let sq8_sz = meta.d;
            index.sq8_codes.resize(meta.ntotal * sq8_sz, 0);
            let mut sq8 = SQ8Quantizer::new(meta.d);
            sq8.vmin = meta.sq8_vmin;
            sq8.vmax = meta.sq8_vmax;
            sq8.scale = (0..meta.d).map(|j| (sq8.vmax[j] - sq8.vmin[j]) / 255.0).collect();
            index.sq8 = Some(sq8);

            let cf_name = if meta.storage_version >= 2 { CF_TQ_SQ8 } else { CF_SQ8_V1 };
            let cf = self.cf(cf_name)?;
            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = self
                    .db
                    .get_pinned_cf_opt(cf, &key, &read_opts)
                    .map_err(|e| format!("读取sq8失败: {}", e))?
                    .ok_or_else(|| format!("sq8缺失: id={}", i))?;
                index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz].copy_from_slice(&val[..sq8_sz]);
            }
        }

        Ok(index)
    }

    // ─── V2: RaBitQ Flat 专用存储 ───

    /// 保存 RaBitQ Flat 索引 (V2: signs/factors/sq8 分离)
    pub fn save_rabitq_flat(&self, index: &RaBitQFlatIndex) -> Result<(), String> {
        let meta = IndexMeta {
            index_type: IndexType::RaBitQFlat,
            d: index.d,
            ntotal: index.ntotal,
            nbits: index.nb_bits,
            use_sq8: index.sq8.is_some(),
            is_inner_product: index.is_inner_product,
            nlist: 0,
            hadamard_seed: 0,
            kmeans_niter: 0,
            sq8_vmin: index.sq8.as_ref().map_or(vec![], |s| s.vmin.clone()),
            sq8_vmax: index.sq8.as_ref().map_or(vec![], |s| s.vmax.clone()),
            rabitq_centroid: index.centroid.clone(),
            ivf_centroids: vec![],
            ivf_cluster_ids: vec![],
            ivf_sq8_vmin: vec![],
            ivf_sq8_vmax: vec![],
            storage_version: 2,
        };
        self.save_meta(&meta)?;

        let signs_sz = index.codec.signs_size();
        let factors_sz = 8;
        let code_sz = index.codec.code_size();
        let mut batch = WriteBatch::default();

        let cf_signs = self.cf(CF_RABITQ_SIGNS)?;
        let cf_factors = self.cf(CF_RABITQ_FACTORS)?;

        for i in 0..index.ntotal {
            let key = (i as u64).to_le_bytes();
            let code = &index.codes[i * code_sz..(i + 1) * code_sz];

            batch.put_cf(cf_signs, &key, &code[..signs_sz]);
            batch.put_cf(cf_factors, &key, &code[signs_sz..signs_sz + factors_sz]);
        }

        if index.sq8.is_some() {
            let sq8_sz = index.d;
            let cf = self.cf(CF_RABITQ_SQ8)?;
            for i in 0..index.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = &index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz];
                batch.put_cf(cf, &key, val);
            }
        }

        self.db.write(batch).map_err(|e| format!("批量写入失败: {}", e))?;
        Ok(())
    }

    /// 加载 RaBitQ Flat 索引
    pub fn load_rabitq_flat(&self) -> Result<RaBitQFlatIndex, String> {
        let meta = self.load_meta()?;
        if meta.index_type != IndexType::RaBitQFlat {
            return Err(format!("索引类型不匹配: 期望 RaBitQFlat, 实际 {:?}", meta.index_type));
        }

        let mut index = RaBitQFlatIndex::new(meta.d, meta.nbits, meta.is_inner_product, meta.use_sq8);
        index.ntotal = meta.ntotal;
        index.centroid = meta.rabitq_centroid;

        let code_sz = index.codec.code_size();
        index.codes.resize(meta.ntotal * code_sz, 0);

        let read_opts = Self::read_opts_for_load();

        if meta.storage_version >= 2 {
            let signs_sz = index.codec.signs_size();
            let cf_signs = self.cf(CF_RABITQ_SIGNS)?;
            let cf_factors = self.cf(CF_RABITQ_FACTORS)?;

            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let signs_val = self
                    .db
                    .get_pinned_cf_opt(cf_signs, &key, &read_opts)
                    .map_err(|e| format!("读取signs失败: {}", e))?
                    .ok_or_else(|| format!("signs缺失: id={}", i))?;
                index.codes[i * code_sz..i * code_sz + signs_sz].copy_from_slice(&signs_val[..signs_sz]);

                let factors_val = self
                    .db
                    .get_pinned_cf_opt(cf_factors, &key, &read_opts)
                    .map_err(|e| format!("读取factors失败: {}", e))?
                    .ok_or_else(|| format!("factors缺失: id={}", i))?;
                index.codes[i * code_sz + signs_sz..i * code_sz + signs_sz + 8]
                    .copy_from_slice(&factors_val[..8]);
            }
        } else {
            let cf = self.cf(CF_CODES_V1)?;
            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = self
                    .db
                    .get_pinned_cf_opt(cf, &key, &read_opts)
                    .map_err(|e| format!("读取codes失败: {}", e))?
                    .ok_or_else(|| format!("codes缺失: id={}", i))?;
                index.codes[i * code_sz..(i + 1) * code_sz].copy_from_slice(&val[..code_sz]);
            }
        }

        if meta.use_sq8 {
            let sq8_sz = meta.d;
            index.sq8_codes.resize(meta.ntotal * sq8_sz, 0);
            let mut sq8 = SQ8Quantizer::new(meta.d);
            sq8.vmin = meta.sq8_vmin;
            sq8.vmax = meta.sq8_vmax;
            sq8.scale = (0..meta.d).map(|j| (sq8.vmax[j] - sq8.vmin[j]) / 255.0).collect();
            index.sq8 = Some(sq8);

            let cf_name = if meta.storage_version >= 2 { CF_RABITQ_SQ8 } else { CF_SQ8_V1 };
            let cf = self.cf(cf_name)?;
            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = self
                    .db
                    .get_pinned_cf_opt(cf, &key, &read_opts)
                    .map_err(|e| format!("读取sq8失败: {}", e))?
                    .ok_or_else(|| format!("sq8缺失: id={}", i))?;
                index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz].copy_from_slice(&val[..sq8_sz]);
            }
        }

        Ok(index)
    }

    // ─── V2: RaBitQ IVF 专用存储 ───

    pub fn save_rabitq_ivf(&self, index: &RaBitQIVFIndex) -> Result<(), String> {
        let use_sq8 = index.sq8_quantizers[0].is_some();
        let sq8_vmin: Vec<Vec<f32>> = if use_sq8 {
            index.sq8_quantizers.iter().map(|opt| opt.as_ref().map_or(vec![], |s| s.vmin.clone())).collect()
        } else { vec![] };
        let sq8_vmax: Vec<Vec<f32>> = if use_sq8 {
            index.sq8_quantizers.iter().map(|opt| opt.as_ref().map_or(vec![], |s| s.vmax.clone())).collect()
        } else { vec![] };
        let cluster_ids: Vec<Vec<u32>> = index.clusters.iter().map(|c| c.ids.clone()).collect();

        let meta = IndexMeta {
            index_type: IndexType::RaBitQIVF,
            d: index.d, ntotal: index.ntotal, nbits: index.nb_bits, use_sq8,
            is_inner_product: index.is_inner_product, nlist: index.nlist,
            hadamard_seed: 0, kmeans_niter: index.kmeans.niter,
            sq8_vmin: vec![], sq8_vmax: vec![], rabitq_centroid: vec![],
            ivf_centroids: index.kmeans.centroids.clone(),
            ivf_cluster_ids: cluster_ids, ivf_sq8_vmin: sq8_vmin, ivf_sq8_vmax: sq8_vmax,
            storage_version: 2,
        };
        self.save_meta(&meta)?;

        let signs_sz = index.codecs[0].signs_size();
        let code_sz = index.codecs[0].code_size();
        let mut batch = WriteBatch::default();
        let cf_signs = self.cf(CF_RABITQ_SIGNS)?;
        let cf_factors = self.cf(CF_RABITQ_FACTORS)?;
        let cf_centroids = self.cf(CF_CENTROIDS)?;

        for c in 0..index.nlist {
            let cluster = &index.clusters[c];
            for v in 0..cluster.ids.len() {
                let id = cluster.ids[v];
                let key = (id as u64).to_le_bytes();
                let code = &cluster.codes[v * code_sz..(v + 1) * code_sz];

                batch.put_cf(cf_signs, &key, &code[..signs_sz]);
                batch.put_cf(cf_factors, &key, &code[signs_sz..signs_sz + 8]);

                if use_sq8 {
                    let sq8_sz = index.d;
                    let cf = self.cf(CF_RABITQ_SQ8)?;
                    let sq8_val = &cluster.sq8_codes[v * sq8_sz..(v + 1) * sq8_sz];
                    batch.put_cf(cf, &key, sq8_val);
                }
            }

            let centroid_key = format!("c{}", c);
            let encoded = bincode::serialize(&index.cluster_centroids[c])
                .map_err(|e| format!("序列化质心失败: {}", e))?;
            batch.put_cf(cf_centroids, centroid_key.as_bytes(), &encoded);
        }

        self.db.write(batch).map_err(|e| format!("批量写入失败: {}", e))?;
        Ok(())
    }

    pub fn load_rabitq_ivf(&self) -> Result<RaBitQIVFIndex, String> {
        let meta = self.load_meta()?;
        if meta.index_type != IndexType::RaBitQIVF {
            return Err(format!("索引类型不匹配: 期望 RaBitQIVF, 实际 {:?}", meta.index_type));
        }

        let mut index = RaBitQIVFIndex::new(
            meta.d, meta.nlist, meta.nbits, meta.is_inner_product, meta.use_sq8,
        );
        index.kmeans.centroids = meta.ivf_centroids;
        index.kmeans.niter = meta.kmeans_niter;

        for c in 0..meta.nlist {
            index.cluster_centroids[c].copy_from_slice(
                &index.kmeans.centroids[c * meta.d..(c + 1) * meta.d],
            );
        }

        if meta.use_sq8 {
            for c in 0..meta.nlist {
                if let Some(ref mut sq8) = index.sq8_quantizers[c] {
                    sq8.vmin = meta.ivf_sq8_vmin[c].clone();
                    sq8.vmax = meta.ivf_sq8_vmax[c].clone();
                    sq8.scale = (0..meta.d).map(|j| (sq8.vmax[j] - sq8.vmin[j]) / 255.0).collect();
                }
            }
        }

        let code_sz = index.codecs[0].code_size();
        let signs_sz = index.codecs[0].signs_size();
        let read_opts = Self::read_opts_for_load();

        if meta.storage_version >= 2 {
            let cf_signs = self.cf(CF_RABITQ_SIGNS)?;
            let cf_factors = self.cf(CF_RABITQ_FACTORS)?;

            for c in 0..meta.nlist {
                let ids = &meta.ivf_cluster_ids[c];
                let n_vectors = ids.len();
                index.clusters[c].ids = ids.clone();
                index.clusters[c].signs.resize(n_vectors * signs_sz, 0);
                index.clusters[c].factors.resize(n_vectors * 8, 0);
                index.clusters[c].codes.resize(n_vectors * code_sz, 0);
                if meta.use_sq8 { index.clusters[c].sq8_codes.resize(n_vectors * meta.d, 0); }

                for v in 0..n_vectors {
                    let id = ids[v];
                    let key = (id as u64).to_le_bytes();

                    let signs_val = self.db.get_pinned_cf_opt(cf_signs, &key, &read_opts)
                        .map_err(|_| format!("读取signs失败: id={}", id))?
                        .ok_or_else(|| format!("signs缺失: id={}", id))?;
                    index.clusters[c].signs[v * signs_sz..(v + 1) * signs_sz].copy_from_slice(&signs_val[..signs_sz]);
                    index.clusters[c].codes[v * code_sz..v * code_sz + signs_sz].copy_from_slice(&signs_val[..signs_sz]);

                    let factors_val = self.db.get_pinned_cf_opt(cf_factors, &key, &read_opts)
                        .map_err(|_| format!("读取factors失败: id={}", id))?
                        .ok_or_else(|| format!("factors缺失: id={}", id))?;
                    index.clusters[c].factors[v * 8..(v + 1) * 8].copy_from_slice(&factors_val[..8]);
                    index.clusters[c].codes[v * code_sz + signs_sz..v * code_sz + signs_sz + 8].copy_from_slice(&factors_val[..8]);

                    if meta.use_sq8 {
                        let cf = self.cf(CF_RABITQ_SQ8)?;
                        let sq8_val = self.db.get_pinned_cf_opt(cf, &key, &read_opts)
                            .map_err(|_| format!("读取sq8失败: id={}", id))?
                            .ok_or_else(|| format!("sq8缺失: id={}", id))?;
                        index.clusters[c].sq8_codes[v * meta.d..(v + 1) * meta.d].copy_from_slice(&sq8_val[..meta.d]);
                    }
                }
            }
        } else {
            let cf_codes = self.cf(CF_CODES_V1)?;
            let cf_sq8 = self.cf(CF_SQ8_V1)?;

            for c in 0..meta.nlist {
                let ids = &meta.ivf_cluster_ids[c];
                let n_vectors = ids.len();
                index.clusters[c].ids = ids.clone();
                index.clusters[c].signs.resize(n_vectors * signs_sz, 0);
                index.clusters[c].factors.resize(n_vectors * 8, 0);
                index.clusters[c].codes.resize(n_vectors * code_sz, 0);
                if meta.use_sq8 { index.clusters[c].sq8_codes.resize(n_vectors * meta.d, 0); }

                for v in 0..n_vectors {
                    let id = ids[v];
                    let key = (id as u64).to_le_bytes();
                    let code_val = self.db.get_pinned_cf_opt(cf_codes, &key, &read_opts)
                        .map_err(|_| format!("读取codes失败: id={}", id))?
                        .ok_or_else(|| format!("codes缺失: id={}", id))?;
                    index.clusters[c].codes[v * code_sz..(v + 1) * code_sz].copy_from_slice(&code_val[..code_sz]);
                    index.clusters[c].signs[v * signs_sz..(v + 1) * signs_sz].copy_from_slice(&code_val[..signs_sz]);
                    index.clusters[c].factors[v * 8..(v + 1) * 8].copy_from_slice(&code_val[signs_sz..signs_sz + 8]);

                    if meta.use_sq8 {
                        let sq8_val = self.db.get_pinned_cf_opt(cf_sq8, &key, &read_opts)
                            .map_err(|_| format!("读取sq8失败: id={}", id))?
                            .ok_or_else(|| format!("sq8缺失: id={}", id))?;
                        index.clusters[c].sq8_codes[v * meta.d..(v + 1) * meta.d].copy_from_slice(&sq8_val[..meta.d]);
                    }
                }
            }
        }

        index.ntotal = meta.ntotal;
        index.rebuild_id_index();
        Ok(index)
    }

    // ─── 增量操作 ───

    pub fn insert_turboquant_vector(&self, id: u64, code: &[u8], sq8_code: Option<&[u8]>) -> Result<(), String> {
        let key = id.to_le_bytes();
        let cf = self.cf(CF_TQ_CODES)?;
        self.db.put_cf(cf, &key, code).map_err(|e| format!("写入codes失败: id={}", id))?;
        if let Some(sq8) = sq8_code {
            let cf = self.cf(CF_TQ_SQ8)?;
            self.db.put_cf(cf, &key, sq8).map_err(|e| format!("写入sq8失败: id={}", id))?;
        }
        Ok(())
    }

    pub fn insert_rabitq_ivf_vector(&self, id: u64, cluster_id: usize, code: &[u8], sq8_code: Option<&[u8]>) -> Result<(), String> {
        let key = id.to_le_bytes();
        let signs_sz = code.len().saturating_sub(8);
        if signs_sz > 0 {
            let cf = self.cf(CF_RABITQ_SIGNS)?;
            self.db.put_cf(cf, &key, &code[..signs_sz]).map_err(|e| format!("写入signs失败: id={}", id))?;
        }
        if code.len() >= signs_sz + 8 {
            let cf = self.cf(CF_RABITQ_FACTORS)?;
            self.db.put_cf(cf, &key, &code[signs_sz..signs_sz + 8]).map_err(|e| format!("写入factors失败: id={}", id))?;
        }
        if let Some(sq8) = sq8_code {
            let cf = self.cf(CF_RABITQ_SQ8)?;
            self.db.put_cf(cf, &key, sq8).map_err(|e| format!("写入sq8失败: id={}", id))?;
        }
        let cf = self.cf(CF_FACTORS_V1)?;
        let cluster_bytes = bincode::serialize(&cluster_id).map_err(|e| format!("序列化cluster_id失败: {}", e))?;
        self.db.put_cf(cf, &key, &cluster_bytes).map_err(|e| format!("写入factors失败: id={}", id))?;
        Ok(())
    }

    pub fn get_code(&self, id: u64) -> Result<Option<Vec<u8>>, String> {
        let key = id.to_le_bytes();
        if let Ok(cf) = self.cf(CF_TQ_CODES) {
            if let Ok(Some(val)) = self.db.get_cf(cf, &key) {
                return Ok(Some(val));
            }
        }
        if let Ok(cf) = self.cf(CF_CODES_V1) {
            if let Ok(Some(val)) = self.db.get_cf(cf, &key) {
                return Ok(Some(val));
            }
        }
        if let (Ok(cf_signs), Ok(cf_factors)) = (self.cf(CF_RABITQ_SIGNS), self.cf(CF_RABITQ_FACTORS)) {
            let signs = self.db.get_cf(cf_signs, &key);
            let factors = self.db.get_cf(cf_factors, &key);
            if let (Ok(Some(s)), Ok(Some(f))) = (signs, factors) {
                let mut code = s;
                code.extend_from_slice(&f);
                return Ok(Some(code));
            }
        }
        Ok(None)
    }

    pub fn get_sq8_code(&self, id: u64) -> Result<Option<Vec<u8>>, String> {
        let key = id.to_le_bytes();
        for cf_name in [CF_TQ_SQ8, CF_SQ8_V1, CF_RABITQ_SQ8] {
            if let Ok(cf) = self.cf(cf_name) {
                if let Ok(Some(val)) = self.db.get_cf(cf, &key) {
                    return Ok(Some(val));
                }
            }
        }
        Ok(None)
    }

    pub fn delete_vector(&self, id: u64) -> Result<(), String> {
        let key = id.to_le_bytes();
        for cf_name in [CF_RABITQ_SIGNS, CF_RABITQ_FACTORS, CF_RABITQ_SQ8, CF_TQ_CODES, CF_TQ_SQ8, CF_CODES_V1, CF_SQ8_V1, CF_FACTORS_V1] {
            if let Ok(cf) = self.cf(cf_name) {
                let _ = self.db.delete_cf(cf, &key);
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> Result<StoreStats, String> {
        let meta = self.load_meta()?;
        let mut code_count = 0usize;
        let cf_name = match meta.index_type {
            IndexType::TurboQuant => CF_TQ_CODES,
            _ => CF_RABITQ_SIGNS,
        };
        if let Ok(cf) = self.cf(cf_name) {
            let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter { if item.is_ok() { code_count += 1; } }
        }
        Ok(StoreStats { index_type: meta.index_type, d: meta.d, ntotal: meta.ntotal, code_count, sq8_count: code_count, use_sq8: meta.use_sq8 })
    }
}

#[derive(Debug)]
pub struct StoreStats {
    pub index_type: IndexType,
    pub d: usize,
    pub ntotal: usize,
    pub code_count: usize,
    pub sq8_count: usize,
    pub use_sq8: bool,
}
