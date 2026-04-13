//! RocksDB 持久化模块
//!
//! 基于 RocksDB Column Family 插件机制实现向量索引的持久化存储。
//!
//! # 存储架构
//!
//! ```text
//! RocksDB
//! ├── "default" CF: 索引元数据 (IndexMeta)
//! ├── "codes" CF:    量化编码 (id -> codes_bytes)
//! ├── "sq8" CF:      SQ8 编码 (id -> sq8_bytes)
//! ├── "centroids" CF: 质心数据 (cluster_id -> centroid_bytes)
//! └── "factors" CF:   量化因子 (id -> factors_bytes)
//! ```
//!
//! # Column Family 设计理念
//!
//! - 不同类型数据分CF存储，利用RocksDB的LSM-Tree分层压缩
//! - 量化编码CF可配置独立压缩策略 (如ZSTD)，最大化存储效率
//! - SQ8编码CF使用轻量压缩 (如LZ4)，保证读取速度
//! - 质心/因子数据量小，使用默认压缩
//!
//! # 序列化策略
//!
//! - 元数据: bincode (确定性编码，跨平台兼容)
//! - 编码数据: 原始字节 (零拷贝，无需反序列化)
//! - 质心/因子: bincode (确定性编码)

use std::path::Path;

use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};

use crate::ivf::RaBitQIVFIndex;
use crate::rabitq::RaBitQFlatIndex;
use crate::sq8::SQ8Quantizer;
use crate::turboquant::TurboQuantFlatIndex;

const CF_CODES: &str = "codes";
const CF_SQ8: &str = "sq8";
const CF_CENTROIDS: &str = "centroids";
const CF_FACTORS: &str = "factors";

/// 索引类型标识
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    TurboQuant,
    RaBitQFlat,
    RaBitQIVF,
}

/// 索引元数据
///
/// 存储在 "default" Column Family 中，键为 "meta"。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexMeta {
    /// 索引类型
    pub index_type: IndexType,
    /// 向量维度
    pub d: usize,
    /// 总向量数
    pub ntotal: usize,
    /// 量化位数
    pub nbits: usize,
    /// 是否使用 SQ8
    pub use_sq8: bool,
    /// 是否内积距离
    pub is_inner_product: bool,
    /// IVF 聚类数 (仅 IVF)
    pub nlist: usize,
    /// Hadamard 旋转种子
    pub hadamard_seed: u64,
    /// KMeans 迭代次数 (仅 IVF)
    pub kmeans_niter: usize,
    /// SQ8 vmin
    pub sq8_vmin: Vec<f32>,
    /// SQ8 vmax
    pub sq8_vmax: Vec<f32>,
    /// RaBitQ 质心 (Flat/IVF)
    pub rabitq_centroid: Vec<f32>,
    /// IVF 聚类质心
    pub ivf_centroids: Vec<f32>,
    /// IVF 每个聚类的向量ID列表
    pub ivf_cluster_ids: Vec<Vec<usize>>,
    /// IVF 每个聚类的 SQ8 vmin
    pub ivf_sq8_vmin: Vec<Vec<f32>>,
    /// IVF 每个聚类的 SQ8 vmax
    pub ivf_sq8_vmax: Vec<Vec<f32>>,
}

/// RocksDB 向量存储
///
/// 封装 RocksDB 实例，提供向量索引的持久化读写接口。
pub struct VectorStore {
    db: DB,
}

impl VectorStore {
    /// 打开或创建向量存储
    ///
    /// # 参数
    /// - `path`: 存储目录路径
    ///
    /// # 返回值
    /// VectorStore 实例
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let codes_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
            ColumnFamilyDescriptor::new(CF_CODES, opts)
        };

        let sq8_cf = {
            let mut opts = Options::default();
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            ColumnFamilyDescriptor::new(CF_SQ8, opts)
        };

        let centroids_cf = {
            ColumnFamilyDescriptor::new(CF_CENTROIDS, Options::default())
        };

        let factors_cf = {
            ColumnFamilyDescriptor::new(CF_FACTORS, Options::default())
        };

        let db = DB::open_cf_descriptors(&db_opts, path, vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            codes_cf,
            sq8_cf,
            centroids_cf,
            factors_cf,
        ]).map_err(|e| format!("打开RocksDB失败: {}", e))?;

        Ok(Self { db })
    }

    /// 保存 TurboQuant Flat 索引
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
        };

        self.save_meta(&meta)?;

        let code_sz = index.code_size();
        let cf = self.cf(CF_CODES)?;
        for i in 0..index.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = &index.codes[i * code_sz..(i + 1) * code_sz];
            self.db.put_cf(cf, &key, val).map_err(|e| format!("写入codes失败: {}", e))?;
        }

        if index.sq8.is_some() {
            let sq8_sz = index.d;
            let cf = self.cf(CF_SQ8)?;
            for i in 0..index.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = &index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz];
                self.db.put_cf(cf, &key, val).map_err(|e| format!("写入sq8失败: {}", e))?;
            }
        }

        Ok(())
    }

    /// 加载 TurboQuant Flat 索引
    pub fn load_turboquant(&self) -> Result<TurboQuantFlatIndex, String> {
        let meta = self.load_meta()?;

        if meta.index_type != IndexType::TurboQuant {
            return Err(format!("索引类型不匹配: 期望 TurboQuant, 实际 {:?}", meta.index_type));
        }

        let mut index = TurboQuantFlatIndex::new(meta.d, meta.nbits, meta.use_sq8);
        index.ntotal = meta.ntotal;

        let code_sz = index.code_size();
        index.codes.resize(meta.ntotal * code_sz, 0);

        let cf = self.cf(CF_CODES)?;
        for i in 0..meta.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = self.db.get_cf(cf, &key).map_err(|e| format!("读取codes失败: {}", e))?
                .ok_or_else(|| format!("codes缺失: id={}", i))?;
            index.codes[i * code_sz..(i + 1) * code_sz].copy_from_slice(&val[..code_sz]);
        }

        if meta.use_sq8 {
            let sq8_sz = meta.d;
            index.sq8_codes.resize(meta.ntotal * sq8_sz, 0);

            let mut sq8 = SQ8Quantizer::new(meta.d);
            sq8.vmin = meta.sq8_vmin;
            sq8.vmax = meta.sq8_vmax;
            index.sq8 = Some(sq8);

            let cf = self.cf(CF_SQ8)?;
            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = self.db.get_cf(cf, &key).map_err(|e| format!("读取sq8失败: {}", e))?
                    .ok_or_else(|| format!("sq8缺失: id={}", i))?;
                index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz].copy_from_slice(&val[..sq8_sz]);
            }
        }

        Ok(index)
    }

    /// 保存 RaBitQ Flat 索引
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
        };

        self.save_meta(&meta)?;

        let code_sz = index.code_size();
        let cf = self.cf(CF_CODES)?;
        for i in 0..index.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = &index.codes[i * code_sz..(i + 1) * code_sz];
            self.db.put_cf(cf, &key, val).map_err(|e| format!("写入codes失败: {}", e))?;
        }

        if index.sq8.is_some() {
            let sq8_sz = index.d;
            let cf = self.cf(CF_SQ8)?;
            for i in 0..index.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = &index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz];
                self.db.put_cf(cf, &key, val).map_err(|e| format!("写入sq8失败: {}", e))?;
            }
        }

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

        let code_sz = index.code_size();
        index.codes.resize(meta.ntotal * code_sz, 0);

        let cf = self.cf(CF_CODES)?;
        for i in 0..meta.ntotal {
            let key = (i as u64).to_le_bytes();
            let val = self.db.get_cf(cf, &key).map_err(|e| format!("读取codes失败: {}", e))?
                .ok_or_else(|| format!("codes缺失: id={}", i))?;
            index.codes[i * code_sz..(i + 1) * code_sz].copy_from_slice(&val[..code_sz]);
        }

        if meta.use_sq8 {
            let sq8_sz = meta.d;
            index.sq8_codes.resize(meta.ntotal * sq8_sz, 0);

            let mut sq8 = SQ8Quantizer::new(meta.d);
            sq8.vmin = meta.sq8_vmin;
            sq8.vmax = meta.sq8_vmax;
            index.sq8 = Some(sq8);

            let cf = self.cf(CF_SQ8)?;
            for i in 0..meta.ntotal {
                let key = (i as u64).to_le_bytes();
                let val = self.db.get_cf(cf, &key).map_err(|e| format!("读取sq8失败: {}", e))?
                    .ok_or_else(|| format!("sq8缺失: id={}", i))?;
                index.sq8_codes[i * sq8_sz..(i + 1) * sq8_sz].copy_from_slice(&val[..sq8_sz]);
            }
        }

        Ok(index)
    }

    /// 保存 RaBitQ IVF 索引
    pub fn save_rabitq_ivf(&self, index: &RaBitQIVFIndex) -> Result<(), String> {
        let use_sq8 = index.sq8_quantizers[0].is_some();

        let sq8_vmin: Vec<Vec<f32>> = if use_sq8 {
            index.sq8_quantizers.iter()
                .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmin.clone()))
                .collect()
        } else {
            vec![]
        };

        let sq8_vmax: Vec<Vec<f32>> = if use_sq8 {
            index.sq8_quantizers.iter()
                .map(|opt| opt.as_ref().map_or(vec![], |s| s.vmax.clone()))
                .collect()
        } else {
            vec![]
        };

        let cluster_ids: Vec<Vec<usize>> = index.clusters.iter().map(|c| c.ids.clone()).collect();

        let meta = IndexMeta {
            index_type: IndexType::RaBitQIVF,
            d: index.d,
            ntotal: index.ntotal,
            nbits: index.nb_bits,
            use_sq8,
            is_inner_product: index.is_inner_product,
            nlist: index.nlist,
            hadamard_seed: 0,
            kmeans_niter: index.kmeans.niter,
            sq8_vmin: vec![],
            sq8_vmax: vec![],
            rabitq_centroid: vec![],
            ivf_centroids: index.kmeans.centroids.clone(),
            ivf_cluster_ids: cluster_ids,
            ivf_sq8_vmin: sq8_vmin,
            ivf_sq8_vmax: sq8_vmax,
        };

        self.save_meta(&meta)?;

        let code_sz = index.code_size();
        let cf_codes = self.cf(CF_CODES)?;
        let cf_sq8 = self.cf(CF_SQ8)?;

        for c in 0..index.nlist {
            let cluster = &index.clusters[c];
            let n_vectors = cluster.ids.len();

            for v in 0..n_vectors {
                let id = cluster.ids[v];
                let key = (id as u64).to_le_bytes();

                let code_val = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                self.db.put_cf(cf_codes, &key, code_val)
                    .map_err(|_| format!("写入codes失败: cluster={} id={}", c, id))?;

                if use_sq8 {
                    let sq8_sz = index.d;
                    let sq8_val = &cluster.sq8_codes[v * sq8_sz..(v + 1) * sq8_sz];
                    self.db.put_cf(cf_sq8, &key, sq8_val)
                        .map_err(|_| format!("写入sq8失败: cluster={} id={}", c, id))?;
                }
            }

            let centroid_key = format!("c{}", c);
            let centroid_val = &index.cluster_centroids[c];
            let cf_centroids = self.cf(CF_CENTROIDS)?;
            let encoded = bincode::serialize(centroid_val)
                .map_err(|e| format!("序列化质心失败: {}", e))?;
            self.db.put_cf(cf_centroids, centroid_key.as_bytes(), &encoded)
                .map_err(|e| format!("写入质心失败: {}", e))?;
        }

        Ok(())
    }

    /// 加载 RaBitQ IVF 索引
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
                }
            }
        }

        let code_sz = index.code_size();
        let cf_codes = self.cf(CF_CODES)?;
        let cf_sq8 = self.cf(CF_SQ8)?;

        for c in 0..meta.nlist {
            let ids = &meta.ivf_cluster_ids[c];
            let n_vectors = ids.len();

            index.clusters[c].ids = ids.clone();
            index.clusters[c].codes.resize(n_vectors * code_sz, 0);

            if meta.use_sq8 {
                index.clusters[c].sq8_codes.resize(n_vectors * meta.d, 0);
            }

            for v in 0..n_vectors {
                let id = ids[v];
                let key = (id as u64).to_le_bytes();

                let code_val = self.db.get_cf(cf_codes, &key)
                    .map_err(|_| format!("读取codes失败: id={}", id))?
                    .ok_or_else(|| format!("codes缺失: id={}", id))?;
                index.clusters[c].codes[v * code_sz..(v + 1) * code_sz]
                    .copy_from_slice(&code_val[..code_sz]);

                if meta.use_sq8 {
                    let sq8_val = self.db.get_cf(cf_sq8, &key)
                        .map_err(|_| format!("读取sq8失败: id={}", id))?
                        .ok_or_else(|| format!("sq8缺失: id={}", id))?;
                    index.clusters[c].sq8_codes[v * meta.d..(v + 1) * meta.d]
                        .copy_from_slice(&sq8_val[..meta.d]);
                }
            }
        }

        index.ntotal = meta.ntotal;

        Ok(index)
    }

    /// 增量插入单条向量到 TurboQuant 索引
    ///
    /// 仅写入 RocksDB，不修改内存索引。
    /// 适用于在线写入场景，后续通过 load 重建索引。
    pub fn insert_turboquant_vector(
        &self,
        id: u64,
        code: &[u8],
        sq8_code: Option<&[u8]>,
    ) -> Result<(), String> {
        let key = id.to_le_bytes();
        let cf = self.cf(CF_CODES)?;
        self.db.put_cf(cf, &key, code)
            .map_err(|e| format!("写入codes失败: id={}", id))?;

        if let Some(sq8) = sq8_code {
            let cf = self.cf(CF_SQ8)?;
            self.db.put_cf(cf, &key, sq8)
                .map_err(|e| format!("写入sq8失败: id={}", id))?;
        }

        Ok(())
    }

    /// 增量插入单条向量到 RaBitQ IVF 索引
    pub fn insert_rabitq_ivf_vector(
        &self,
        id: u64,
        cluster_id: usize,
        code: &[u8],
        sq8_code: Option<&[u8]>,
    ) -> Result<(), String> {
        let key = id.to_le_bytes();
        let cf = self.cf(CF_CODES)?;
        self.db.put_cf(cf, &key, code)
            .map_err(|e| format!("写入codes失败: id={}", id))?;

        if let Some(sq8) = sq8_code {
            let cf = self.cf(CF_SQ8)?;
            self.db.put_cf(cf, &key, sq8)
                .map_err(|e| format!("写入sq8失败: id={}", id))?;
        }

        let cf = self.cf(CF_FACTORS)?;
        let cluster_bytes = bincode::serialize(&cluster_id)
            .map_err(|e| format!("序列化cluster_id失败: {}", e))?;
        self.db.put_cf(cf, &key, &cluster_bytes)
            .map_err(|e| format!("写入factors失败: id={}", id))?;

        Ok(())
    }

    /// 读取单条向量编码
    pub fn get_code(&self, id: u64) -> Result<Option<Vec<u8>>, String> {
        let key = id.to_le_bytes();
        let cf = self.cf(CF_CODES)?;
        self.db.get_cf(cf, &key).map_err(|e| format!("读取codes失败: {}", e))
    }

    /// 读取单条 SQ8 编码
    pub fn get_sq8_code(&self, id: u64) -> Result<Option<Vec<u8>>, String> {
        let key = id.to_le_bytes();
        let cf = self.cf(CF_SQ8)?;
        self.db.get_cf(cf, &key).map_err(|e| format!("读取sq8失败: {}", e))
    }

    /// 删除单条向量
    pub fn delete_vector(&self, id: u64) -> Result<(), String> {
        let key = id.to_le_bytes();

        let cf = self.cf(CF_CODES)?;
        self.db.delete_cf(cf, &key).map_err(|e| format!("删除codes失败: {}", e))?;

        let cf = self.cf(CF_SQ8)?;
        self.db.delete_cf(cf, &key).map_err(|e| format!("删除sq8失败: {}", e))?;

        let cf = self.cf(CF_FACTORS)?;
        self.db.delete_cf(cf, &key).map_err(|e| format!("删除factors失败: {}", e))?;

        Ok(())
    }

    /// 获取存储统计信息
    pub fn stats(&self) -> Result<StoreStats, String> {
        let meta = self.load_meta()?;

        let mut code_count = 0usize;
        let cf = self.cf(CF_CODES)?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        for item in iter {
            if item.is_ok() { code_count += 1; }
        }

        let mut sq8_count = 0usize;
        let cf = self.cf(CF_SQ8)?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        for item in iter {
            if item.is_ok() { sq8_count += 1; }
        }

        Ok(StoreStats {
            index_type: meta.index_type,
            d: meta.d,
            ntotal: meta.ntotal,
            code_count,
            sq8_count,
            use_sq8: meta.use_sq8,
        })
    }

    fn save_meta(&self, meta: &IndexMeta) -> Result<(), String> {
        let encoded = bincode::serialize(meta)
            .map_err(|e| format!("序列化元数据失败: {}", e))?;
        self.db.put(b"meta", &encoded)
            .map_err(|e| format!("写入元数据失败: {}", e))
    }

    fn load_meta(&self) -> Result<IndexMeta, String> {
        let val = self.db.get(b"meta")
            .map_err(|e| format!("读取元数据失败: {}", e))?
            .ok_or_else(|| "元数据不存在".to_string())?;
        bincode::deserialize(&val)
            .map_err(|e| format!("反序列化元数据失败: {}", e))
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily, String> {
        self.db.cf_handle(name)
            .ok_or_else(|| format!("Column Family '{}' 不存在", name))
    }
}

/// 存储统计信息
#[derive(Debug)]
pub struct StoreStats {
    pub index_type: IndexType,
    pub d: usize,
    pub ntotal: usize,
    pub code_count: usize,
    pub sq8_count: usize,
    pub use_sq8: bool,
}
