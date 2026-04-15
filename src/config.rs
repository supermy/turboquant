//! TurboQuant 统一配置系统
//!
//! 所有参数提供默认值，支持 TOML 配置文件，持久化路径可配置。
//!
//! # 使用方式
//!
//! ```ignore
//! use turboquant::config::TurboConfig;
//!
//! let config = TurboConfig::default();
//! let config = TurboConfig::load_from_file("turboquant.toml")?;
//! let config = TurboConfig::load_from_str(toml_str)?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MB: usize = 1024 * 1024;
const KB: usize = 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TurboConfig {
    pub index: IndexConfig,
    pub rocksdb: RocksDBConfig,
    #[cfg(feature = "nng")]
    pub server: ServerConfig,
    pub storage: StorageConfig,
}

impl Default for TurboConfig {
    fn default() -> Self {
        Self {
            index: IndexConfig::default(),
            rocksdb: RocksDBConfig::default(),
            #[cfg(feature = "nng")]
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

impl TurboConfig {
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("读取配置文件失败: {}", e))?;
        Self::load_from_str(&content)
    }

    pub fn load_from_str(s: &str) -> Result<Self, String> {
        let config: TurboConfig =
            toml::from_str(s).map_err(|e| format!("解析配置文件失败: {}", e))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let content = toml::to_string_pretty(self).map_err(|e| format!("序列化配置失败: {}", e))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("创建配置目录失败: {}", e))?;
        }
        std::fs::write(path, content).map_err(|e| format!("写入配置文件失败: {}", e))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.index.validate()?;
        self.rocksdb.validate()?;
        self.storage.validate()
    }

    pub fn generate_default_toml() -> String {
        let config = Self::default();
        toml::to_string_pretty(&config).unwrap_or_default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub d: usize,
    pub nlist: usize,
    pub nbits: usize,
    pub use_sq8: bool,
    pub refine_factor: usize,
    pub hadamard_seed: u64,
    pub kmeans_niter: usize,
    pub kmeans_seed: u64,
    pub is_inner_product: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            d: 128,
            nlist: 64,
            nbits: 4,
            use_sq8: true,
            refine_factor: 10,
            hadamard_seed: 12345,
            kmeans_niter: 20,
            kmeans_seed: 42,
            is_inner_product: false,
        }
    }
}

impl IndexConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.d == 0 {
            return Err("index.d 不能为 0".to_string());
        }
        if self.nlist == 0 {
            return Err("index.nlist 不能为 0".to_string());
        }
        if self.nbits == 0 || self.nbits > 8 {
            return Err("index.nbits 必须在 1..=8 范围内".to_string());
        }
        if self.kmeans_niter == 0 {
            return Err("index.kmeans_niter 不能为 0".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RocksDBConfig {
    pub block_cache_size_mb: usize,
    pub write_buffer_size_mb: usize,
    pub max_write_buffer_number: i32,
    pub max_background_jobs: i32,
    pub target_file_size_base_mb: usize,
    pub max_open_files: i32,
    pub level_compaction_dynamic_level_bytes: bool,
    pub optimize_level_style_compaction_mb: usize,
    pub use_fsync: bool,
    pub stats_dump_period_sec: u32,
    pub rate_limiter_bytes_per_sec_mb: i64,
    pub env_background_threads: i32,
    pub env_high_priority_threads: i32,
    pub env_low_priority_threads: i32,
    pub bloom_bits_per_key: f64,
    pub readahead_size_kb: usize,
    pub async_io: bool,
    pub verify_checksums: bool,
    pub cuckoo_hash_ratio: f64,
    pub cuckoo_max_search_depth: u32,
    pub rabitq_signs_block_size_kb: usize,
    pub rabitq_signs_write_buffer_mb: usize,
    pub rabitq_factors_block_size_kb: usize,
    pub rabitq_factors_write_buffer_mb: usize,
    pub tq_codes_block_size_kb: usize,
    pub tq_codes_write_buffer_mb: usize,
    pub v1_codes_block_size_kb: usize,
}

impl Default for RocksDBConfig {
    fn default() -> Self {
        Self {
            block_cache_size_mb: 512,
            write_buffer_size_mb: 64,
            max_write_buffer_number: 3,
            max_background_jobs: 4,
            target_file_size_base_mb: 64,
            max_open_files: -1,
            level_compaction_dynamic_level_bytes: true,
            optimize_level_style_compaction_mb: 256,
            use_fsync: false,
            stats_dump_period_sec: 60,
            rate_limiter_bytes_per_sec_mb: 100,
            env_background_threads: 4,
            env_high_priority_threads: 2,
            env_low_priority_threads: 2,
            bloom_bits_per_key: 10.0,
            readahead_size_kb: 64,
            async_io: true,
            verify_checksums: false,
            cuckoo_hash_ratio: 0.9,
            cuckoo_max_search_depth: 100,
            rabitq_signs_block_size_kb: 2,
            rabitq_signs_write_buffer_mb: 16,
            rabitq_factors_block_size_kb: 1,
            rabitq_factors_write_buffer_mb: 8,
            tq_codes_block_size_kb: 8,
            tq_codes_write_buffer_mb: 64,
            v1_codes_block_size_kb: 4,
        }
    }
}

impl RocksDBConfig {
    pub fn block_cache_size(&self) -> usize {
        self.block_cache_size_mb * MB
    }

    pub fn write_buffer_size(&self) -> usize {
        self.write_buffer_size_mb * MB
    }

    pub fn target_file_size_base(&self) -> usize {
        self.target_file_size_base_mb * MB
    }

    pub fn optimize_level_style_compaction(&self) -> usize {
        self.optimize_level_style_compaction_mb * MB
    }

    pub fn rate_limiter_bytes_per_sec(&self) -> i64 {
        self.rate_limiter_bytes_per_sec_mb * MB as i64
    }

    pub fn readahead_size(&self) -> usize {
        self.readahead_size_kb * KB
    }

    pub fn rabitq_signs_block_size(&self) -> usize {
        self.rabitq_signs_block_size_kb * KB
    }

    pub fn rabitq_signs_write_buffer(&self) -> usize {
        self.rabitq_signs_write_buffer_mb * MB
    }

    pub fn rabitq_factors_block_size(&self) -> usize {
        self.rabitq_factors_block_size_kb * KB
    }

    pub fn rabitq_factors_write_buffer(&self) -> usize {
        self.rabitq_factors_write_buffer_mb * MB
    }

    pub fn tq_codes_block_size(&self) -> usize {
        self.tq_codes_block_size_kb * KB
    }

    pub fn tq_codes_write_buffer(&self) -> usize {
        self.tq_codes_write_buffer_mb * MB
    }

    pub fn v1_codes_block_size(&self) -> usize {
        self.v1_codes_block_size_kb * KB
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.block_cache_size_mb == 0 {
            return Err("rocksdb.block_cache_size_mb 不能为 0".to_string());
        }
        if self.write_buffer_size_mb == 0 {
            return Err("rocksdb.write_buffer_size_mb 不能为 0".to_string());
        }
        if self.max_write_buffer_number < 1 {
            return Err("rocksdb.max_write_buffer_number 必须 >= 1".to_string());
        }
        if self.max_background_jobs < 1 {
            return Err("rocksdb.max_background_jobs 必须 >= 1".to_string());
        }
        if self.cuckoo_hash_ratio <= 0.0 || self.cuckoo_hash_ratio > 1.0 {
            return Err("rocksdb.cuckoo_hash_ratio 必须在 (0, 1] 范围内".to_string());
        }
        Ok(())
    }
}

#[cfg(feature = "nng")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub query_url: String,
    pub write_url: String,
    pub notify_url: String,
    pub n_workers: usize,
    pub d: usize,
}

#[cfg(feature = "nng")]
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            query_url: "tcp://127.0.0.1:5555".to_string(),
            write_url: "tcp://127.0.0.1:5556".to_string(),
            notify_url: "tcp://127.0.0.1:5557".to_string(),
            n_workers: 4,
            d: 128,
        }
    }
}

#[cfg(feature = "nng")]
impl ServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_workers == 0 {
            return Err("server.n_workers 不能为 0".to_string());
        }
        if self.d == 0 {
            return Err("server.d 不能为 0".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            db_path: PathBuf::from("data/turboquant.db"),
        }
    }
}

impl StorageConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    pub fn ensure_dirs(&self) -> Result<(), String> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("创建数据库目录失败: {}", e))?;
        }
        std::fs::create_dir_all(&self.data_dir).map_err(|e| format!("创建数据目录失败: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = TurboConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_default_values() {
        let config = TurboConfig::default();
        assert_eq!(config.index.d, 128);
        assert_eq!(config.index.nlist, 64);
        assert_eq!(config.index.nbits, 4);
        assert!(config.index.use_sq8);
        assert_eq!(config.index.refine_factor, 10);
        assert_eq!(config.index.hadamard_seed, 12345);
        assert_eq!(config.index.kmeans_niter, 20);
        assert_eq!(config.index.kmeans_seed, 42);
        assert!(!config.index.is_inner_product);

        assert_eq!(config.rocksdb.block_cache_size_mb, 512);
        assert_eq!(config.rocksdb.write_buffer_size_mb, 64);
        assert_eq!(config.rocksdb.max_write_buffer_number, 3);
        assert_eq!(config.rocksdb.max_background_jobs, 4);
        assert_eq!(config.rocksdb.bloom_bits_per_key, 10.0);
        assert!(!config.rocksdb.use_fsync);
        assert_eq!(config.rocksdb.cuckoo_hash_ratio, 0.9);
        assert_eq!(config.rocksdb.cuckoo_max_search_depth, 100);
        assert_eq!(config.rocksdb.rabitq_signs_block_size_kb, 2);
        assert_eq!(config.rocksdb.rabitq_factors_block_size_kb, 1);
        assert_eq!(config.rocksdb.tq_codes_block_size_kb, 8);

        #[cfg(feature = "nng")]
        {
            assert_eq!(config.server.query_url, "tcp://127.0.0.1:5555");
            assert_eq!(config.server.write_url, "tcp://127.0.0.1:5556");
            assert_eq!(config.server.notify_url, "tcp://127.0.0.1:5557");
            assert_eq!(config.server.n_workers, 4);
            assert_eq!(config.server.d, 128);
        }

        assert_eq!(config.storage.data_dir, PathBuf::from("data"));
        assert_eq!(config.storage.db_path, PathBuf::from("data/turboquant.db"));
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = TurboConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: TurboConfig = toml::from_str(&toml_str).unwrap();
        assert!(parsed.validate().is_ok());

        assert_eq!(parsed.index.d, config.index.d);
        assert_eq!(parsed.index.nlist, config.index.nlist);
        assert_eq!(parsed.index.nbits, config.index.nbits);
        assert_eq!(parsed.index.use_sq8, config.index.use_sq8);
        assert_eq!(parsed.index.hadamard_seed, config.index.hadamard_seed);
        assert_eq!(
            parsed.rocksdb.block_cache_size_mb,
            config.rocksdb.block_cache_size_mb
        );
        assert_eq!(
            parsed.rocksdb.write_buffer_size_mb,
            config.rocksdb.write_buffer_size_mb
        );
        assert_eq!(parsed.storage.data_dir, config.storage.data_dir);
        assert_eq!(parsed.storage.db_path, config.storage.db_path);
    }

    #[test]
    fn test_load_from_str() {
        let toml_str = r#"
[index]
d = 256
nlist = 128
nbits = 6
use_sq8 = false
refine_factor = 5
hadamard_seed = 99999
kmeans_niter = 30
kmeans_seed = 100
is_inner_product = true

[rocksdb]
block_cache_size_mb = 1024
write_buffer_size_mb = 128
max_background_jobs = 8
bloom_bits_per_key = 12.0

[storage]
data_dir = "/tmp/turboquant_data"
db_path = "/tmp/turboquant_data/myindex.db"
"#;
        let config = TurboConfig::load_from_str(toml_str).unwrap();
        assert_eq!(config.index.d, 256);
        assert_eq!(config.index.nlist, 128);
        assert_eq!(config.index.nbits, 6);
        assert!(!config.index.use_sq8);
        assert_eq!(config.index.refine_factor, 5);
        assert_eq!(config.index.hadamard_seed, 99999);
        assert_eq!(config.index.kmeans_niter, 30);
        assert_eq!(config.index.kmeans_seed, 100);
        assert!(config.index.is_inner_product);

        assert_eq!(config.rocksdb.block_cache_size_mb, 1024);
        assert_eq!(config.rocksdb.write_buffer_size_mb, 128);
        assert_eq!(config.rocksdb.max_background_jobs, 8);
        assert_eq!(config.rocksdb.bloom_bits_per_key, 12.0);

        assert_eq!(
            config.storage.data_dir,
            PathBuf::from("/tmp/turboquant_data")
        );
        assert_eq!(
            config.storage.db_path,
            PathBuf::from("/tmp/turboquant_data/myindex.db")
        );
    }

    #[test]
    fn test_load_partial_config() {
        let toml_str = r#"
[index]
d = 64
"#;
        let config = TurboConfig::load_from_str(toml_str).unwrap();
        assert_eq!(config.index.d, 64);
        assert_eq!(config.index.nlist, 64);
        assert_eq!(config.index.nbits, 4);
        assert_eq!(config.rocksdb.block_cache_size_mb, 512);
    }

    #[test]
    fn test_load_empty_config() {
        let toml_str = "";
        let config = TurboConfig::load_from_str(toml_str).unwrap();
        assert_eq!(config.index.d, 128);
        assert_eq!(config.rocksdb.block_cache_size_mb, 512);
    }

    #[test]
    fn test_validation_errors() {
        let mut config = TurboConfig::default();

        config.index.d = 0;
        assert!(config.validate().is_err());
        config.index.d = 128;

        config.index.nbits = 0;
        assert!(config.validate().is_err());
        config.index.nbits = 4;

        config.index.nbits = 9;
        assert!(config.validate().is_err());
        config.index.nbits = 4;

        config.rocksdb.block_cache_size_mb = 0;
        assert!(config.validate().is_err());
        config.rocksdb.block_cache_size_mb = 512;

        config.rocksdb.max_write_buffer_number = 0;
        assert!(config.validate().is_err());
        config.rocksdb.max_write_buffer_number = 3;

        config.rocksdb.cuckoo_hash_ratio = 0.0;
        assert!(config.validate().is_err());
        config.rocksdb.cuckoo_hash_ratio = 0.9;

        config.rocksdb.cuckoo_hash_ratio = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rocksdb_size_helpers() {
        let config = RocksDBConfig::default();
        assert_eq!(config.block_cache_size(), 512 * 1024 * 1024);
        assert_eq!(config.write_buffer_size(), 64 * 1024 * 1024);
        assert_eq!(config.target_file_size_base(), 64 * 1024 * 1024);
        assert_eq!(config.optimize_level_style_compaction(), 256 * 1024 * 1024);
        assert_eq!(config.rate_limiter_bytes_per_sec(), 100 * 1024 * 1024);
        assert_eq!(config.readahead_size(), 64 * 1024);
        assert_eq!(config.rabitq_signs_block_size(), 2 * 1024);
        assert_eq!(config.rabitq_signs_write_buffer(), 16 * 1024 * 1024);
        assert_eq!(config.rabitq_factors_block_size(), 1 * 1024);
        assert_eq!(config.rabitq_factors_write_buffer(), 8 * 1024 * 1024);
        assert_eq!(config.tq_codes_block_size(), 8 * 1024);
        assert_eq!(config.tq_codes_write_buffer(), 64 * 1024 * 1024);
        assert_eq!(config.v1_codes_block_size(), 4 * 1024);
    }

    #[test]
    fn test_save_and_load_file() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let config_path = tmp_dir.path().join("test_config.toml");

        let config = TurboConfig::default();
        config.save_to_file(&config_path).unwrap();

        let loaded = TurboConfig::load_from_file(&config_path).unwrap();
        assert_eq!(loaded.index.d, config.index.d);
        assert_eq!(loaded.index.nlist, config.index.nlist);
        assert_eq!(
            loaded.rocksdb.block_cache_size_mb,
            config.rocksdb.block_cache_size_mb
        );
        assert_eq!(loaded.storage.data_dir, config.storage.data_dir);
    }

    #[test]
    fn test_generate_default_toml() {
        let toml_str = TurboConfig::generate_default_toml();
        assert!(toml_str.contains("[index]"));
        assert!(toml_str.contains("[rocksdb]"));
        assert!(toml_str.contains("[storage]"));
        assert!(toml_str.contains("d = 128"));
        assert!(toml_str.contains("nlist = 64"));

        let parsed = TurboConfig::load_from_str(&toml_str).unwrap();
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn test_storage_ensure_dirs() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let storage = StorageConfig {
            data_dir: tmp_dir.path().join("data"),
            db_path: tmp_dir.path().join("data/subdir/index.db"),
        };
        storage.ensure_dirs().unwrap();
        assert!(tmp_dir.path().join("data").exists());
        assert!(tmp_dir.path().join("data/subdir").exists());
    }
}
