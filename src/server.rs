//! NNG 服务架构
//!
//! 使用 NNG (nanomsg next generation) 替换 ZeroMQ，提供写入/查询/通知接口。
//!
//! # 架构设计
//!
//! - Query Socket (Rep0): 查询服务，多 Worker 线程并发
//! - Write Socket (Rep0): 写入服务，单线程串行化 + 批量提交
//! - Notify Socket (Pub0): 事件广播
//!
//! # NNG vs ZeroMQ
//!
//! | 维度 | NNG | ZeroMQ |
//! |------|-----|--------|
//! | ROUTER 模式 | 无 (用 Rep0 + 多实例替代) | ZMQ_ROUTER |
//! | WebSocket | 原生支持 | 不支持 |
//! | TLS | 内置 | 需要 CurveZMQ |
//! | Survey 模式 | 原生支持 | 不支持 |
//! | C 依赖 | CMake + NNG C 库 | 纯 Rust (zeromq crate) |

use std::sync::Arc;
use std::thread;
use std::time::Instant;
use parking_lot::RwLock;

use crate::ivf::{RaBitQIVFIndex, TurboQuantIVFIndex};
use crate::ivf_store::{RocksDBIVFIndex, RocksDBTQIVFIndex};
use crate::turboquant::TurboQuantFlatIndex;
use crate::rabitq::RaBitQFlatIndex;

const DEFAULT_QUERY_URL: &str = "tcp://127.0.0.1:5555";
const DEFAULT_WRITE_URL: &str = "tcp://127.0.0.1:5556";
const DEFAULT_NOTIFY_URL: &str = "tcp://127.0.0.1:5557";

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum QueryRequest {
    IVFSearch {
        query: Vec<f32>,
        k: u32,
        nprobe: u32,
        refine_factor: u32,
    },
    FlatSearch {
        query: Vec<f32>,
        k: u32,
    },
    PersistedIVFSearch {
        query: Vec<f32>,
        k: u32,
        nprobe: u32,
        refine_factor: u32,
    },
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct QueryResponse {
    pub results: Vec<(u32, f32)>,
    pub latency_us: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WriteRequest {
    Insert {
        vectors: Vec<f32>,
        n: u32,
        ids: Option<Vec<u32>>,
    },
    Delete {
        ids: Vec<u32>,
    },
    BuildIVFIndex {
        nlist: u32,
        index_type: u8,
        quantization: u8,
    },
    PersistIndex {
        path: String,
    },
    LoadIndex {
        path: String,
    },
    Flush,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WriteResponse {
    Inserted { ids: Vec<u32> },
    Deleted { count: u32 },
    IndexBuilt { nlist: u32, ntotal: u32 },
    IndexPersisted { path: String },
    IndexLoaded { ntotal: u32 },
    Flushed,
    Error { message: String },
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum NotifyEvent {
    IndexBuilt { nlist: u32, ntotal: u32 },
    IndexPersisted { path: String },
    WriteProgress { inserted: u32, pending: u32 },
    Error { message: String },
}

/// 内存索引引擎
pub enum MemoryIndex {
    None,
    RaBitQIVF(RaBitQIVFIndex),
    TurboQuantIVF(TurboQuantIVFIndex),
    RaBitQFlat(RaBitQFlatIndex),
    TurboQuantFlat(TurboQuantFlatIndex),
}

/// 持久化索引引擎
pub enum PersistedIndex {
    None,
    RaBitQIVF(RocksDBIVFIndex),
    TurboQuantIVF(RocksDBTQIVFIndex),
}

/// 向量引擎服务
pub struct VectorEngineService {
    d: usize,
    memory_index: RwLock<MemoryIndex>,
    persisted_index: RwLock<PersistedIndex>,
    pending_vectors: RwLock<Vec<f32>>,
}

impl VectorEngineService {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            memory_index: RwLock::new(MemoryIndex::None),
            persisted_index: RwLock::new(PersistedIndex::None),
            pending_vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, vectors: &[f32], n: usize) -> Vec<u32> {
        let mut pending = self.pending_vectors.write();
        let start_id = (pending.len() / self.d) as u32;
        pending.extend_from_slice(vectors);
        (start_id..start_id + n as u32).collect()
    }

    pub fn build_ivf_index(&self, nlist: usize, index_type: u8, quantization: u8) -> Result<(usize, usize), String> {
        let pending = self.pending_vectors.read();
        let n = pending.len() / self.d;
        if n == 0 {
            return Err("没有待索引的向量".to_string());
        }

        let use_sq8 = true;
        match index_type {
            0 => {
                let mut index = RaBitQIVFIndex::new(self.d, nlist, 1, false, use_sq8);
                index.train(&pending, n);
                index.add(&pending, n);
                let ntotal = index.ntotal();
                *self.memory_index.write() = MemoryIndex::RaBitQIVF(index);
                Ok((nlist, ntotal))
            }
            1 => {
                let mut index = TurboQuantIVFIndex::new(self.d, nlist, quantization as usize, use_sq8);
                index.train(&pending, n);
                index.add(&pending, n);
                let ntotal = index.ntotal();
                *self.memory_index.write() = MemoryIndex::TurboQuantIVF(index);
                Ok((nlist, ntotal))
            }
            _ => Err(format!("不支持的索引类型: {}", index_type)),
        }
    }

    pub fn persist(&self, path: &str) -> Result<(), String> {
        let mem_idx = self.memory_index.read();
        match &*mem_idx {
            MemoryIndex::RaBitQIVF(index) => {
                let mut persisted = RocksDBIVFIndex::open(&std::path::PathBuf::from(path))?;
                persisted.build_from_ivf(index)?;
                *self.persisted_index.write() = PersistedIndex::RaBitQIVF(persisted);
                Ok(())
            }
            MemoryIndex::TurboQuantIVF(index) => {
                let mut persisted = RocksDBTQIVFIndex::open(&std::path::PathBuf::from(path))?;
                persisted.build_from_ivf(index)?;
                *self.persisted_index.write() = PersistedIndex::TurboQuantIVF(persisted);
                Ok(())
            }
            _ => Err("没有内存索引可持久化".to_string()),
        }
    }

    pub fn search_memory(&self, query: &[f32], k: usize, nprobe: usize, refine_factor: usize) -> Vec<(u32, f32)> {
        let mem_idx = self.memory_index.read();
        match &*mem_idx {
            MemoryIndex::RaBitQIVF(index) => {
                let results = index.search(query, 1, k, nprobe, refine_factor);
                results.into_iter().next().unwrap_or_default().into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            MemoryIndex::TurboQuantIVF(index) => {
                let results = index.search(query, 1, k, nprobe, refine_factor);
                results.into_iter().next().unwrap_or_default().into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            MemoryIndex::RaBitQFlat(index) => {
                let results = index.search(query, 1, k, 1);
                results.into_iter().next().unwrap_or_default().into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            MemoryIndex::TurboQuantFlat(index) => {
                let results = index.search(query, 1, k, 1);
                results.into_iter().next().unwrap_or_default().into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            MemoryIndex::None => vec![],
        }
    }

    pub fn search_persisted(&self, query: &[f32], k: usize, nprobe: usize, refine_factor: usize) -> Vec<(u32, f32)> {
        let pers_idx = self.persisted_index.read();
        match &*pers_idx {
            PersistedIndex::RaBitQIVF(index) => {
                let results = index.search(query, k, nprobe, refine_factor);
                results.into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            PersistedIndex::TurboQuantIVF(index) => {
                let results = index.search(query, k, nprobe, refine_factor);
                results.into_iter().map(|(id, dist)| (id as u32, dist)).collect()
            }
            PersistedIndex::None => vec![],
        }
    }

    pub fn ntotal(&self) -> usize {
        let pending = self.pending_vectors.read();
        pending.len() / self.d
    }

    pub fn load_index(&self, path: &str) -> Result<usize, String> {
        let store = crate::store::VectorStore::open(&std::path::PathBuf::from(path))?;
        let meta = store.load_meta()?;

        match meta.index_type {
            crate::store::IndexType::RaBitQIVF => {
                let index = store.load_rabitq_ivf()?;
                let ntotal = index.ntotal();
                *self.persisted_index.write() = PersistedIndex::RaBitQIVF(
                    crate::ivf_store::RocksDBIVFIndex::open(&std::path::PathBuf::from(path))?
                );
                Ok(ntotal)
            }
            crate::store::IndexType::TurboQuant => {
                let persisted = crate::ivf_store::RocksDBTQIVFIndex::open(&std::path::PathBuf::from(path))?;
                let ntotal = persisted.ntotal();
                *self.persisted_index.write() = PersistedIndex::TurboQuantIVF(persisted);
                Ok(ntotal)
            }
            _ => Err(format!("不支持的索引类型: {:?}", meta.index_type)),
        }
    }
}

pub struct TurboQuantServer {
    query_url: String,
    write_url: String,
    notify_url: String,
    n_workers: usize,
    d: usize,
}

impl TurboQuantServer {
    pub fn new(d: usize) -> Self {
        Self {
            query_url: DEFAULT_QUERY_URL.to_string(),
            write_url: DEFAULT_WRITE_URL.to_string(),
            notify_url: DEFAULT_NOTIFY_URL.to_string(),
            n_workers: 4,
            d,
        }
    }

    pub fn with_query_url(mut self, url: &str) -> Self {
        self.query_url = url.to_string();
        self
    }

    pub fn with_write_url(mut self, url: &str) -> Self {
        self.write_url = url.to_string();
        self
    }

    pub fn with_notify_url(mut self, url: &str) -> Self {
        self.notify_url = url.to_string();
        self
    }

    pub fn with_workers(mut self, n: usize) -> Self {
        self.n_workers = n;
        self
    }

    pub fn run(self) -> Result<(), String> {
        let engine = Arc::new(VectorEngineService::new(self.d));
        let query_url = self.query_url.clone();
        let write_url = self.write_url.clone();
        let notify_url = self.notify_url.clone();

        let notify_sock = nng::Socket::new(nng::Protocol::Pub0)
            .map_err(|e| format!("创建 Pub0 socket 失败: {}", e))?;
        notify_sock.listen(&notify_url)
            .map_err(|e| format!("Pub0 listen 失败: {}", e))?;

        let write_sock = nng::Socket::new(nng::Protocol::Rep0)
            .map_err(|e| format!("创建 Rep0 (write) socket 失败: {}", e))?;
        write_sock.listen(&write_url)
            .map_err(|e| format!("Rep0 (write) listen 失败: {}", e))?;

        let query_sockets: Vec<nng::Socket> = (0..self.n_workers)
            .map(|i| {
                let sock = nng::Socket::new(nng::Protocol::Rep0)
                    .map_err(|e| format!("创建 Rep0 (query) socket 失败: {}", e))?;
                let port = 5555 + i;
                let url = format!("tcp://127.0.0.1:{}", port);
                sock.listen(&url)
                    .map_err(|e| format!("Rep0 (query) listen 失败: {}", e))?;
                Ok(sock)
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut worker_handles = Vec::new();
        for sock in query_sockets {
            let engine = engine.clone();
            let handle = thread::spawn(move || {
                Self::query_worker(sock, engine);
            });
            worker_handles.push(handle);
        }

        Self::write_worker(write_sock, notify_sock, engine);

        for handle in worker_handles {
            let _ = handle.join();
        }

        Ok(())
    }

    fn query_worker(sock: nng::Socket, engine: Arc<VectorEngineService>) {
        loop {
            match sock.recv() {
                Ok(msg) => {
                    let req: Result<QueryRequest, _> = bincode::deserialize(msg.as_slice());
                    let resp = match req {
                        Ok(QueryRequest::IVFSearch { query, k, nprobe, refine_factor }) => {
                            let t0 = Instant::now();
                            let results = engine.search_memory(&query, k as usize, nprobe as usize, refine_factor as usize);
                            QueryResponse {
                                results,
                                latency_us: t0.elapsed().as_micros() as u64,
                            }
                        }
                        Ok(QueryRequest::FlatSearch { query, k }) => {
                            let t0 = Instant::now();
                            let results = engine.search_memory(&query, k as usize, 1, 1);
                            QueryResponse {
                                results,
                                latency_us: t0.elapsed().as_micros() as u64,
                            }
                        }
                        Ok(QueryRequest::PersistedIVFSearch { query, k, nprobe, refine_factor }) => {
                            let t0 = Instant::now();
                            let results = engine.search_persisted(&query, k as usize, nprobe as usize, refine_factor as usize);
                            QueryResponse {
                                results,
                                latency_us: t0.elapsed().as_micros() as u64,
                            }
                        }
                        Err(_) => QueryResponse {
                            results: vec![],
                            latency_us: 0,
                        },
                    };

                    if let Ok(resp_bytes) = bincode::serialize(&resp) {
                        let mut reply = nng::Message::new();
                        reply.push_back(&resp_bytes);
                        let _ = sock.send(reply);
                    }
                }
                Err(_) => break,
            }
        }
    }

    fn write_worker(sock: nng::Socket, notify_sock: nng::Socket, engine: Arc<VectorEngineService>) {
        loop {
            match sock.recv() {
                Ok(msg) => {
                    let req: Result<WriteRequest, _> = bincode::deserialize(msg.as_slice());
                    let resp = match req {
                        Ok(WriteRequest::Insert { vectors, n, ids }) => {
                            let inserted_ids = engine.insert(&vectors, n as usize);
                            WriteResponse::Inserted { ids: inserted_ids }
                        }
                        Ok(WriteRequest::Delete { ids }) => {
                            WriteResponse::Deleted { count: 0 }
                        }
                        Ok(WriteRequest::BuildIVFIndex { nlist, index_type, quantization }) => {
                            match engine.build_ivf_index(nlist as usize, index_type, quantization) {
                                Ok((nl, nt)) => {
                                    let event = NotifyEvent::IndexBuilt { nlist: nl as u32, ntotal: nt as u32 };
                                    if let Ok(event_bytes) = bincode::serialize(&event) {
                                        let mut notify_msg = nng::Message::new();
                                        notify_msg.push_back(&event_bytes);
                                        let _ = notify_sock.send(notify_msg);
                                    }
                                    WriteResponse::IndexBuilt { nlist: nl as u32, ntotal: nt as u32 }
                                }
                                Err(e) => WriteResponse::Error { message: e },
                            }
                        }
                        Ok(WriteRequest::PersistIndex { path }) => {
                            match engine.persist(&path) {
                                Ok(()) => {
                                    let event = NotifyEvent::IndexPersisted { path: path.clone() };
                                    if let Ok(event_bytes) = bincode::serialize(&event) {
                                        let mut notify_msg = nng::Message::new();
                                        notify_msg.push_back(&event_bytes);
                                        let _ = notify_sock.send(notify_msg);
                                    }
                                    WriteResponse::IndexPersisted { path }
                                }
                                Err(e) => WriteResponse::Error { message: e },
                            }
                        }
                        Ok(WriteRequest::LoadIndex { path }) => {
                            match engine.load_index(&path) {
                                Ok(ntotal) => WriteResponse::IndexLoaded { ntotal: ntotal as u32 },
                                Err(e) => WriteResponse::Error { message: e },
                            }
                        }
                        Ok(WriteRequest::Flush) => WriteResponse::Flushed,
                        Err(_) => WriteResponse::Error { message: "反序列化失败".to_string() },
                    };

                    if let Ok(resp_bytes) = bincode::serialize(&resp) {
                        let mut reply = nng::Message::new();
                        reply.push_back(&resp_bytes);
                        let _ = sock.send(reply);
                    }
                }
                Err(_) => break,
            }
        }
    }
}

impl Default for TurboQuantServer {
    fn default() -> Self {
        Self::new(128)
    }
}
