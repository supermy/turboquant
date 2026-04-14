# MergeOperator 查询插件计划

## 核心思想

**将查询计算下推到 C++ RocksDB 插件层，Rust 只发送查询请求、接收结果。**

当前查询流程的问题：
1. **多轮 FFI 开销**：Rust 侧每步都通过 rocksdb crate → librocksdb-sys C API → RocksDB C++ 读取数据，再返回 Rust 计算距离，再读下一批数据
2. **数据搬运**：signs/factors/sq8 数据从 RocksDB block cache → C API → Rust 堆内存，大量无意义拷贝
3. **串行化**：Rust 侧的距离计算是标量实现，无法利用 NEON/AVX2 SIMD
4. **多次 IO**：粗排读 signs → 精排读 factors → SQ8 读 sq8_codes，三阶段各自独立 IO

**新架构**：C++ 查询插件直接在 RocksDB 内部执行查询，一次 FFI 调用返回最终 top-K 结果。

```
当前:  Rust ─get_cf─→ RocksDB ─→ 返回 signs ─→ Rust 计算 dot_qo
       Rust ─get_cf─→ RocksDB ─→ 返回 factors ─→ Rust 计算 dist
       Rust ─get_cf─→ RocksDB ─→ 返回 sq8 ─→ Rust 计算 refined_dist
       (3 轮 FFI + 3 轮数据搬运 + 标量计算)

新:    Rust ─search(query, k, nprobe)─→ C++ 查询插件
       C++ 插件: SIMD LUT 构建 + 范围扫描 signs + 读 factors + 读 sq8 + top-K
       ─→ 返回 Vec<(id, dist)>
       (1 轮 FFI + 零数据搬运 + SIMD 计算)
```

---

## 架构设计

```
┌─────────────────────────────────────────────────────┐
│                    Rust 应用层                       │
│  ivf.rs / ivf_store.rs / rabitq.rs / turboquant.rs  │
├─────────────────────────────────────────────────────┤
│              Rust FFI 桥接层                         │
│  src/vector_engine_ffi.rs                           │
│  ┌─────────────────────────────────────────────┐    │
│  │  VectorEngine::ivf_search(                  │    │
│  │    &self, query, k, nprobe, refine_factor)  │    │
│  │  → Vec<(u32, f32)>                          │    │
│  │                                             │    │
│  │  VectorEngine::flat_search(                 │    │
│  │    &self, query, k)                         │    │
│  │  → Vec<(u32, f32)>                          │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│              C++ 向量查询引擎                        │
│  cpp/vector_query_engine.h / .cpp                   │
│  ┌─────────────────────────────────────────────┐    │
│  │  VectorQueryEngine {                        │    │
│  │    db: rocksdb::DB*                         │    │
│  │    codecs: vector<RaBitQCodec>              │    │
│  │    sq8_quantizers: vector<SQ8Quantizer>     │    │
│  │    centroids: vector<float>                 │    │
│  │    cluster_counts: vector<uint32_t>         │    │
│  │    cluster_offsets: vector<uint64_t>        │    │
│  │                                             │    │
│  │    ivf_search(query, k, nprobe, refine)     │    │
│  │    flat_search(query, k)                    │    │
│  │  }                                          │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│            C++ SIMD 距离计算内核                     │
│  cpp/simd_distance.h / .cpp                         │
│  ┌─────────────────────────────────────────────┐    │
│  │  rabitq_signs_distance_neon(signs, lut, n)  │    │
│  │  rabitq_full_distance_neon(...)             │    │
│  │  sq8_distance_neon(sq8, query, vmin, scale) │    │
│  │  lloyd_max_4bit_lut_distance_neon(...)      │    │
│  │  l2_distance_neon(a, b, d)                  │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│          C++ RocksDB 查询插件层                      │
│  cpp/rocksdb_query_plugin.h / .cpp                  │
│  ┌─────────────────────────────────────────────┐    │
│  │  VectorSstPartitioner (按 cluster 分区)     │    │
│  │  VectorMergeOperator (查询结果缓存合并)     │    │
│  │  VectorCompactionFilter (向量感知压缩)      │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│                 RocksDB 8.10.0                       │
│              (librocksdb-sys)                        │
└─────────────────────────────────────────────────────┘
```

---

## C++ 查询引擎 API

### C API (供 Rust FFI 调用)

```c
// ==================== 引擎生命周期 ====================

// 创建查询引擎 (打开 RocksDB + 加载元数据)
// path: RocksDB 数据库路径
// 返回: 引擎句柄, NULL 表示失败
void* vq_engine_open(const char* path);

// 关闭查询引擎
void vq_engine_close(void* engine);

// ==================== IVF 查询 ====================

// IVF 查询参数
typedef struct {
    const float* query;       // 查询向量, [d] f32
    int d;                    // 维度
    int k;                    // 返回 top-K
    int nprobe;               // 探测聚类数
    int refine_factor;        // SQ8 精排扩大因子
    int use_sq8;              // 是否使用 SQ8 精排
} IVFSearchParams;

// IVF 查询结果
typedef struct {
    uint32_t id;              // 向量全局 ID
    float distance;           // 距离
} QueryResult;

// 执行 IVF 查询
// 返回: 结果数组, 调用者需 vq_results_free() 释放
// n_results: 输出结果数量
QueryResult* vq_ivf_search(void* engine, const IVFSearchParams* params, int* n_results);

// ==================== 扁平查询 ====================

typedef struct {
    const float* query;       // 查询向量, [d] f32
    int d;                    // 维度
    int k;                    // 返回 top-K
    int index_type;           // 0=RaBitQ, 1=TurboQuant
} FlatSearchParams;

QueryResult* vq_flat_search(void* engine, const FlatSearchParams* params, int* n_results);

// ==================== 批量查询 ====================

// 批量 IVF 查询 (多查询并行)
// queries: [n_queries * d] f32
// 返回: [n_queries] 个结果数组, 每个由 n_results_per_query[i] 指定长度
QueryResult* vq_ivf_batch_search(
    void* engine,
    const float* queries, int n_queries,
    int d, int k, int nprobe, int refine_factor, int use_sq8,
    int** n_results_per_query);

// ==================== 内存管理 ====================

void vq_results_free(QueryResult* results);
void vq_n_results_free(int* n_results);
```

### C++ 内部实现

```cpp
class VectorQueryEngine {
public:
    // 打开 RocksDB, 加载元数据
    Status Open(const std::string& path);

    // IVF 查询核心实现
    std::vector<QueryResult> IVFSearch(
        const float* query, int d, int k,
        int nprobe, int refine_factor, bool use_sq8);

    // 扁平查询核心实现
    std::vector<QueryResult> FlatSearch(
        const float* query, int d, int k, IndexType type);

private:
    rocksdb::DB* db_;

    // 内存中的元数据 (O(nlist * d))
    std::vector<float> centroids_;         // [nlist * d]
    std::vector<RaBitQCodec> codecs_;      // [nlist]
    std::vector<SQ8Quantizer> sq8_quants_; // [nlist]
    std::vector<uint32_t> cluster_counts_; // [nlist]
    std::vector<uint64_t> cluster_offsets_;// [nlist+1] 前缀和

    int d_;
    int nlist_;
    int ntotal_;
    bool is_inner_product_;
    bool use_sq8_;

    // ==================== 查询阶段 ====================

    // 阶段1: 找到 nprobe 个最近聚类 (SIMD l2_distance)
    std::vector<int> NearestClusters(const float* query, int nprobe);

    // 阶段2: RaBitQ signs + factors 范围扫描 (SIMD LUT)
    // 一次扫描同时读 signs + factors, 计算 RaBitQ 完整距离
    struct Candidate {
        float distance;
        uint32_t global_id;
        uint16_t cluster_id;
        uint16_t local_id;
    };
    std::vector<Candidate> RaBitQScan(
        const float* query, const std::vector<int>& clusters,
        int k1);

    // 阶段3: SQ8 精排 (SIMD fma)
    std::vector<QueryResult> SQ8Refine(
        const float* query, const std::vector<Candidate>& candidates, int k);

    // ==================== SIMD 内核 ====================

    // RaBitQ LUT 构建 + 批量 signs 距离
    void BuildRaBitQLUT(const float* query, const float* centroid,
                        float* lut, float* c1_c34_qr_l2sqr);

    // SIMD 批量 signs-only 距离计算
    void BatchSignsDistanceNEON(const uint8_t* signs_data,
                                int n_vectors, int signs_size,
                                const float* lut,
                                float* distances);

    // SIMD SQ8 距离计算
    void BatchSQ8DistanceNEON(const uint8_t* sq8_codes,
                              const float* query,
                              const float* vmin, const float* scale,
                              int n_vectors, int d,
                              float* distances);
};
```

---

## 查询执行流程 (C++ 内部)

```cpp
std::vector<QueryResult> VectorQueryEngine::IVFSearch(
    const float* query, int d, int k,
    int nprobe, int refine_factor, bool use_sq8)
{
    int k1 = use_sq8 ? std::min(k * refine_factor, ntotal_) : k;

    // === 阶段1: 最近聚类选择 ===
    auto nearest = NearestClusters(query, nprobe);
    // SIMD l2_distance, O(nlist * d / 4) NEON

    // === 阶段2: RaBitQ signs + factors 范围扫描 ===
    auto candidates = RaBitQScan(query, nearest, k1);

    if (!use_sq8) {
        // 直接返回 top-K
        std::partial_sort(candidates.begin(),
                         candidates.begin() + std::min(k, (int)candidates.size()),
                         candidates.end());
        // 转换为 QueryResult 返回
        return ConvertToResults(candidates, k);
    }

    // === 阶段3: SQ8 精排 ===
    return SQ8Refine(query, candidates, k);
}

std::vector<Candidate> VectorQueryEngine::RaBitQScan(
    const float* query, const std::vector<int>& clusters, int k1)
{
    // 预分配结果
    std::vector<Candidate> top_k1;
    top_k1.reserve(k1 + 1);
    float max_dist = std::numeric_limits<float>::max();

    auto* cf_signs = db_->DefaultColumnFamily(); // rabitq_signs CF
    auto* cf_factors = db_->DefaultColumnFamily(); // rabitq_factors CF

    for (int cluster_id : clusters) {
        const float* centroid = &centroids_[cluster_id * d_];
        const auto& codec = codecs_[cluster_id];

        // 构建 SIMD LUT (NEON)
        alignas(16) float lut[16][256]; // d=128: 16KB LUT
        float c1, c34, qr_to_c_l2sqr;
        BuildRaBitQLUT(query, centroid, &lut[0][0], &c1, &c34, &qr_to_c_l2sqr);

        // 范围扫描 signs CF
        std::string start = ClusterKey(cluster_id, 0);
        std::string end = ClusterKey(cluster_id + 1, 0);

        rocksdb::ReadOptions opts;
        opts.verify_checksums = false;
        opts.fill_cache = true;
        opts.readahead_size = 256 * 1024;
        opts.iterate_lower_bound = &start;
        opts.iterate_upper_bound = &end;

        auto iter = db_->NewIterator(opts, cf_signs);

        // 批量读取 signs + 计算 SIMD 距离
        int local_id = 0;
        for (iter->Seek(start); iter->Valid() && iter->key().compare(end) < 0; iter->Next()) {
            auto signs = iter->value().data();
            int signs_size = d_ / 8;

            // SIMD signs-only 距离
            float dot_qo = SIMDSignsDistance(signs, &lut[0][0], signs_size);

            // 读 factors (8 bytes: or_minus_c_l2sqr + dp_multiplier)
            auto fkey = ClusterKey(cluster_id, local_id);
            std::string factors_val;
            db_->Get(rocksdb::ReadOptions(), cf_factors, fkey, &factors_val);

            float or_minus_c_l2sqr, dp_multiplier;
            memcpy(&or_minus_c_l2sqr, factors_val.data(), 4);
            memcpy(&dp_multiplier, factors_val.data() + 4, 4);

            // 完整 RaBitQ 距离
            float final_dot = c1 * dot_qo - c34;
            float dist = or_minus_c_l2sqr + qr_to_c_l2sqr
                       - 2.0f * dp_multiplier * final_dot;
            if (!is_inner_product_) dist = std::max(0.0f, dist);

            // Top-K 堆维护
            uint32_t global_id = cluster_offsets_[cluster_id] + local_id;
            if ((int)top_k1.size() < k1) {
                top_k1.push_back({dist, global_id, (uint16_t)cluster_id, (uint16_t)local_id});
                if ((int)top_k1.size() == k1) {
                    max_dist = std::max_element(top_k1.begin(), top_k1.end(),
                        [](const auto& a, const auto& b) { return a.distance < b.distance; })->distance;
                }
            } else if (dist < max_dist) {
                auto it = std::max_element(top_k1.begin(), top_k1.end(),
                    [](const auto& a, const auto& b) { return a.distance < b.distance; });
                *it = {dist, global_id, (uint16_t)cluster_id, (uint16_t)local_id};
                max_dist = std::max_element(top_k1.begin(), top_k1.end(),
                    [](const auto& a, const auto& b) { return a.distance < b.distance; })->distance;
            }
            local_id++;
        }
        delete iter;
    }
    return top_k1;
}
```

---

## SIMD 内核实现

### RaBitQ Signs 距离 (NEON)

```cpp
// d=128: signs_size=16 bytes, LUT=16*256*4=16KB
// NEON: 4-wide f32 累加
float SIMDSignsDistanceNEON(const uint8_t* signs,
                             const float* lut, // [signs_size][256]
                             int signs_size)
{
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < signs_size; i++) {
        uint8_t byte_val = signs[i];
        float32x4_t vals = vld1q_f32(&lut[i * 256 + byte_val * 4]);
        // 注意: 实际实现需要 4 个 byte_val 对应的 LUT 值
        // 优化: 一次处理 4 个字节
        sum = vaddq_f32(sum, vals);
    }
    // 水平求和
    float result;
    #if __aarch64__
    result = vaddvq_f32(sum);
    #else
    float32x2_t s = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
    result = vget_lane_f32(vpadd_f32(s, s), 0);
    #endif
    return result;
}

// 更优: 4 字节批量处理
float SIMDSignsDistanceNEON_4x(const uint8_t* signs,
                                const float* lut,
                                int signs_size)
{
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < signs_size; i += 4) {
        // 一次读 4 个 signs 字节
        uint8x8_t bytes = vld1_u8(&signs[i]);
        // 从 LUT 中 gather 4 个值
        float32x4_t v0 = vld1q_f32(&lut[(i+0) * 256 + vget_lane_u8(bytes, 0)]);
        float32x4_t v1 = vld1q_f32(&lut[(i+1) * 256 + vget_lane_u8(bytes, 1)]);
        float32x4_t v2 = vld1q_f32(&lut[(i+2) * 256 + vget_lane_u8(bytes, 2)]);
        float32x4_t v3 = vld1q_f32(&lut[(i+3) * 256 + vget_lane_u8(bytes, 3)]);
        sum = vaddq_f32(sum, v0);
        sum = vaddq_f32(sum, v1);
        sum = vaddq_f32(sum, v2);
        sum = vaddq_f32(sum, v3);
    }
    return vaddvq_f32(sum);
}
```

### SQ8 距离 (NEON)

```cpp
// SQ8: decoded[j] = vmin[j] + code[j] * scale[j]
// dist = Σ (decoded[j] - query[j])^2
float BatchSQ8DistanceNEON(const uint8_t* sq8_codes,
                            const float* query,
                            const float* vmin,
                            const float* scale,
                            int d)
{
    float32x4_t sum = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 3 < d; j += 4) {
        // 加载 4 个 u8 并转为 f32
        uint8x8_t codes = vld1_u8(&sq8_codes[j]);
        uint32x4_t codes32 = vmovl_u16(vget_low_u16(vmovl_u8(codes)));
        float32x4_t codes_f = vcvtq_f32_u32(codes32);

        float32x4_t s = vld1q_f32(&scale[j]);
        float32x4_t v = vld1q_f32(&vmin[j]);
        float32x4_t q = vld1q_f32(&query[j]);

        // decoded = vmin + code * scale
        float32x4_t decoded = vfmaq_f32(v, codes_f, s);
        // diff = decoded - query
        float32x4_t diff = vsubq_f32(decoded, q);
        // diff^2
        sum = vfmaq_f32(sum, diff, diff);
    }
    return vaddvq_f32(sum);
}
```

### L2 距离 (NEON)

```cpp
float L2DistanceNEON(const float* a, const float* b, int d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }
    return vaddvq_f32(sum);
}
```

---

## RocksDB 查询插件

### VectorSstPartitioner (按 cluster 分区 SST)

```cpp
class VectorSstPartitioner : public rocksdb::SstPartitioner {
public:
    explicit VectorSstPartitioner(int prefix_len)
        : prefix_len_(prefix_len) {}

    const char* Name() const override { return "VectorSstPartitioner"; }

    // 判断两个 key 是否应在同一个 SST 分区
    PartitionerResult ShouldPartition(
        const SstPartitioner::Request& request) override {
        // 比较前 prefix_len_ 字节 (cluster_id)
        if (request.current_user_key.size() >= prefix_len_ &&
            request.next_user_key.size() >= prefix_len_) {
            auto cur_prefix = request.current_user_key.substr(0, prefix_len_);
            auto next_prefix = request.next_user_key.substr(0, prefix_len_);
            if (cur_prefix != next_prefix) {
                return PartitionerResult::kPartition;
            }
        }
        return PartitionerResult::kNotPartition;
    }

    bool CanDoTrivialMove(const Slice& smallest_user_key,
                          const Slice& largest_user_key) const override {
        return false; // 不允许 trivial move, 确保分区边界
    }

private:
    int prefix_len_; // 2 for cluster_id:u16
};
```

### VectorMergeOperator (查询结果缓存)

```cpp
class VectorMergeOperator : public rocksdb::MergeOperator {
public:
    const char* Name() const override { return "VectorMergeOperator"; }

    // 合并逻辑: 用于 cluster_meta 的原子更新
    // cluster_meta 格式: [count:u32][start_local_id:u32]
    bool FullMergeV2(const MergeOperationInput& merge_in,
                     MergeOperationOutput* merge_out) const override {
        if (merge_in.key.starts_with("meta_")) {
            // cluster_meta 合并: 累加 count
            uint32_t count = 0;
            uint32_t start_id = 0;
            if (merge_in.existing_value) {
                memcpy(&count, merge_in.existing_value->data(), 4);
                memcpy(&start_id, merge_in.existing_value->data() + 4, 4);
            }
            for (auto& operand : merge_in.operand_list) {
                uint32_t delta;
                memcpy(&delta, operand.data(), 4);
                count += delta;
            }
            std::string result(8, '\0');
            memcpy(&result[0], &count, 4);
            memcpy(&result[4], &start_id, 4);
            merge_out->new_value = result;
            return true;
        }
        return false;
    }

    // 增量合并
    bool PartialMergeMulti(const Slice& key,
                           const std::deque<Slice>& operand_list,
                           std::string* new_value,
                           Logger* logger) const override {
        uint32_t total_delta = 0;
        for (auto& op : operand_list) {
            uint32_t delta;
            memcpy(&delta, op.data(), 4);
            total_delta += delta;
        }
        new_value->resize(4);
        memcpy(&(*new_value)[0], &total_delta, 4);
        return true;
    }
};
```

### VectorCompactionFilter (向量感知压缩)

```cpp
class VectorCompactionFilter : public rocksdb::CompactionFilter {
public:
    const char* Name() const override { return "VectorCompactionFilter"; }

    Decision FilterV2(int level, const Slice& key, ValueType value_type,
                      const Slice& existing_value, std::string* new_value,
                      std::string* skip_until) const override {
        // 如果 key 对应的 cluster 已被标记删除, 清理该向量数据
        // 通过检查 cluster_meta CF 中的删除标记
        if (IsDeletedCluster(key)) {
            return Decision::kRemove;
        }
        return Decision::kKeep;
    }

private:
    bool IsDeletedCluster(const Slice& key) const {
        // 从 key 前缀提取 cluster_id, 检查是否已删除
        // ...
    }
};
```

---

## Rust FFI 桥接

### src/vector_engine_ffi.rs

```rust
use std::ffi::CString;
use std::os::raw::c_int;
use std::path::Path;

#[repr(C)]
pub struct CQueryResult {
    pub id: u32,
    pub distance: f32,
}

#[repr(C)]
pub struct CIVFSearchParams {
    pub query: *const f32,
    pub d: c_int,
    pub k: c_int,
    pub nprobe: c_int,
    pub refine_factor: c_int,
    pub use_sq8: c_int,
}

extern "C" {
    fn vq_engine_open(path: *const i8) -> *mut std::ffi::c_void;
    fn vq_engine_close(engine: *mut std::ffi::c_void);
    fn vq_ivf_search(
        engine: *mut std::ffi::c_void,
        params: *const CIVFSearchParams,
        n_results: *mut c_int,
    ) -> *mut CQueryResult;
    fn vq_results_free(results: *mut CQueryResult);
    fn vq_flat_search(
        engine: *mut std::ffi::c_void,
        query: *const f32,
        d: c_int,
        k: c_int,
        index_type: c_int,
        n_results: *mut c_int,
    ) -> *mut CQueryResult;
}

pub struct VectorEngine {
    handle: *mut std::ffi::c_void,
}

impl VectorEngine {
    pub fn open(path: &Path) -> Result<Self, String> {
        let c_path = CString::new(path.to_str().ok_or("invalid path")?)
            .map_err(|e| e.to_string())?;
        let handle = unsafe { vq_engine_open(c_path.as_ptr()) };
        if handle.is_null() {
            return Err("failed to open vector engine".into());
        }
        Ok(Self { handle })
    }

    pub fn ivf_search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        refine_factor: usize,
        use_sq8: bool,
    ) -> Vec<(u32, f32)> {
        let d = query.len();
        let params = CIVFSearchParams {
            query: query.as_ptr(),
            d: d as c_int,
            k: k as c_int,
            nprobe: nprobe as c_int,
            refine_factor: refine_factor as c_int,
            use_sq8: use_sq8 as c_int,
        };
        let mut n_results: c_int = 0;
        let results = unsafe {
            vq_ivf_search(self.handle, &params, &mut n_results)
        };
        if results.is_null() || n_results == 0 {
            return vec![];
        }
        let vec: Vec<(u32, f32)> = unsafe {
            std::slice::from_raw_parts(results, n_results as usize)
                .iter()
                .map(|r| (r.id, r.distance))
                .collect()
        };
        unsafe { vq_results_free(results) };
        vec
    }

    pub fn flat_search(
        &self,
        query: &[f32],
        k: usize,
        index_type: usize,
    ) -> Vec<(u32, f32)> {
        let d = query.len();
        let mut n_results: c_int = 0;
        let results = unsafe {
            vq_flat_search(
                self.handle,
                query.as_ptr(),
                d as c_int,
                k as c_int,
                index_type as c_int,
                &mut n_results,
            )
        };
        if results.is_null() || n_results == 0 {
            return vec![];
        }
        let vec: Vec<(u32, f32)> = unsafe {
            std::slice::from_raw_parts(results, n_results as usize)
                .iter()
                .map(|r| (r.id, r.distance))
                .collect()
        };
        unsafe { vq_results_free(results) };
        vec
    }
}

impl Drop for VectorEngine {
    fn drop(&mut self) {
        unsafe { vq_engine_close(self.handle) };
    }
}

unsafe impl Send for VectorEngine {}
unsafe impl Sync for VectorEngine {}
```

---

## 构建系统

### build.rs

```rust
fn main() {
    let rocksdb_include = std::env::var("OUT_DIR")
        .map(|d| format!("{}/rocksdb/include", d))
        .unwrap_or_else(|_| "/usr/local/include".into());

    cc::Build::new()
        .cpp(true)
        .file("cpp/vector_query_engine.cpp")
        .file("cpp/simd_distance.cpp")
        .file("cpp/rocksdb_query_plugin.cpp")
        .include(&rocksdb_include)
        .flag("-std=c++17")
        .flag("-O3")
        .flag_if_supported("-march=native")
        .flag_if_supported("-D__STDC_FORMAT_MACROS")
        .compile("vector_engine");

    println!("cargo:rustc-link-lib=static=vector_engine");
    println!("cargo:rustc-link-lib=rocksdb");
}
```

### Cargo.toml 新增依赖

```toml
[dependencies]
cc = { version = "1", features = ["parallel"] }
```

---

## 剩余 HIGH/MEDIUM 优化 (同时实施)

| 编号 | 优化项 | 预估 QPS 提升 |
|------|--------|---------------|
| H7 | find_cluster_for_id 二分查找 + cluster_offsets 前缀和 | 5-10% |
| M1 | nearest_clusters_into buffer 复用 | 2-3% |
| M2 | ClusterData.codes 冗余字段移除 | 2-4% |
| M3 | SQ8 compute_distance 查找表优化 | 2-5% |
| M4 | Hadamard apply_batch buffer 复用 | 2-3% |
| M9 | RocksDB 加载 multi_get 批量读取 | 2-5% |

---

## 实施阶段

### Phase 1: 基础设施
1. 添加 `cc` 依赖到 Cargo.toml
2. 创建 `build.rs`
3. 创建 `cpp/` 目录和头文件
4. 创建 `src/vector_engine_ffi.rs`
5. 验证编译链路

### Phase 2: C++ SIMD 距离内核
1. 实现 `simd_distance.h/.cpp` (NEON 优先)
2. RaBitQ LUT 构建 + signs 批量距离
3. SQ8 批量距离
4. L2 距离
5. 单元测试

### Phase 3: C++ 查询引擎
1. 实现 `vector_query_engine.h/.cpp`
2. IVF 查询 (三阶段漏斗)
3. 扁平查询
4. 元数据加载 (从 RocksDB 读取 centroids/codecs/cluster_meta)
5. Rust FFI 绑定 + 集成测试

### Phase 4: RocksDB 查询插件
1. VectorSstPartitioner (按 cluster 分区)
2. VectorMergeOperator (cluster_meta 原子更新)
3. VectorCompactionFilter (向量感知压缩)
4. 集成到 store.rs / ivf_store.rs

### Phase 5: 剩余 HIGH/MEDIUM 优化
1. H7: find_cluster_for_id 二分查找
2. M1: nearest_clusters_into buffer 复用
3. M2: ClusterData.codes 移除
4. M3: SQ8 查找表优化
5. M4: Hadamard buffer 复用
6. M9: multi_get 批量读取

### Phase 6: 集成测试 + 性能验证

---

## 实施状态 (2026-04-14 更新)

### ✅ 已完成

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 基础设施: cc依赖 + build.rs + cpp/目录 + vector_engine_ffi.rs | ✅ 编译通过 |
| Phase 2 | C++ SIMD 距离内核: NEON LUT/signs/SQ8/L2 (simd_distance.h) | ✅ 编译通过 |
| Phase 3 | C++ 查询引擎: IVF三阶段漏斗 + 元数据加载 (vector_query_engine.cpp) | ✅ 编译通过 |
| Phase 4 | RocksDB 查询插件: SstPartitioner + MergeOperator + CompactionFilter (rocksdb_query_plugin.h) | ✅ 编译通过 |
| Phase 5 | HIGH/MEDIUM 优化: H7/M1/M3/M4 | ✅ 测试通过 |
| Phase 6 | 集成测试: 16/16 通过 | ✅ |

### 新增文件

| 文件 | 说明 |
|------|------|
| `cpp/vector_engine.h` | C API 头文件 (vq_engine_open/close, vq_ivf_search, vq_flat_search, vq_ivf_batch_search) |
| `cpp/simd_distance.h` | SIMD 距离计算内核 (NEON/AVX2 条件编译) |
| `cpp/vector_query_engine.cpp` | C++ 查询引擎 + C API 实现 |
| `cpp/rocksdb_query_plugin.h` | RocksDB 插件 (VectorSstPartitioner, VectorMergeOperator, VectorCompactionFilter) |
| `build.rs` | C++ 编译脚本 (cc crate) |
| `src/vector_engine_ffi.rs` | Rust FFI 桥接 (VectorEngine struct) |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `Cargo.toml` | 添加 `cc = "1"` 到 build-dependencies |
| `src/lib.rs` | 添加 `pub mod vector_engine_ffi;` + `pub use VectorEngine` |
| `src/ivf_store.rs` | H7: cluster_offsets 前缀和 + find_cluster_for_id 二分查找 O(log nlist) |
| `src/kmeans.rs` | M1: nearest_clusters_into buffer 复用 |
| `src/sq8.rs` | M3: compute_distance_preprocessed + preprocess_query |
| `src/hadamard.rs` | M4: apply_into + apply_batch buffer 复用 |

### ⏳ 待实施

| 优化项 | 预估提升 | 说明 |
|--------|----------|------|
| M2: ClusterData.codes 移除 | 2-4% | codes 与 signs+factors 重复，但 store.rs/ivf_store.rs 大量使用，需谨慎 |
| M9: multi_get 批量读取 | 2-5% | rocksdb crate API 限制，需重构加载逻辑 |
| C++ 查询引擎集成测试 | - | 需要实际 RocksDB 数据验证端到端正确性 |
| 性能基准测试 | - | 对比 Rust 标量 vs C++ SIMD 查询 QPS |

---

## SIFT Small 基准测试结果 (2026-04-14)

数据集: 10000 × 128D, 100 queries, k=10, 10轮取平均

| 方法 | QPS | P50(us) | P99(us) | Recall@10 |
|------|-----|---------|---------|-----------|
| **RaBitQ Flat+SQ8** | **3492** | 262 | 649 | 93.1% |
| IVF-64 np=8 | 2472 | 376 | 843 | 96.7% |
| IVF-256 np=8 | 2436 | 382 | 880 | 90.1% |
| IVF-64 np=16 | 1215 | 726 | 1773 | 98.5% |
| IVF-256 np=16 | 1296 | 731 | 1515 | 96.7% |
| TQ 4bit+SQ8 | 782 | 1198 | 2481 | 98.7% |
| Persisted np=8 | 1776 | 437 | 2219 | 96.7% |
| Persisted np=16 | 944 | 850 | 3769 | 98.5% |

### 关键发现

1. **RaBitQ Flat 最快 (3492 QPS)** — 1-bit LUT 查表极快，但 recall 只有 93.1%
2. **IVF-64 np=8 性价比最高** — 96.7% recall 下 2472 QPS
3. **TurboQuant 最慢但 recall 最高** — 4-bit LUT 64KB 溢出 L1 cache
4. **持久化查询比内存慢 ~30%** — RocksDB 读取开销

### 已修复 Bug

- **持久化 recall 32.5% → 96.7%** — store.rs V2 加载路径未填充 signs/factors 字段

### 进一步优化方向

1. **C++ SIMD 引擎端到端集成** — 当前 C++ 引擎已编译但未集成到 Rust 查询路径
2. **RaBitQ Flat recall 提升** — 增大 refine_factor 或使用 IVF 结构
3. **TurboQuant LUT 优化** — 4-bit 量化 64KB LUT 溢出 L1，考虑直接计算替代
4. **持久化查询优化** — 减少 RocksDB 读取次数，使用 C++ VectorQueryEngine

---

## 预估 QPS 提升

| 优化项 | 预估提升 |
|--------|----------|
| C++ SIMD 距离计算 (NEON) | 15-30% |
| 消除 FFI 数据搬运 (1轮 vs 3轮) | 10-20% |
| VectorSstPartitioner | 5-10% |
| VectorCompactionFilter | 3-5% |
| find_cluster_for_id 二分查找 | 5-10% |
| ClusterData.codes 移除 | 2-4% |
| SQ8 查找表优化 | 2-5% |
| nearest_clusters buffer 复用 | 2-3% |
| Hadamard buffer 复用 | 2-3% |
| multi_get 批量读取 | 2-5% |
| **综合预估** | **50-80%** |

---

## 一、RocksDB 8.10.0 新 API 优化方案

### 版本对应关系

| Rust Crate | 版本 |
|---|---|
| `rocksdb` | 0.22.0 |
| `librocksdb-sys` | 0.16.0 |
| **C++ RocksDB** | **8.10.0** (非 9.10) |

> 注意: 用户认为 rocksdb 0.22 对应 C++ 9.10，实际对应 **8.10.0**。
> 验证来源: `Cargo.lock` 中 `librocksdb-sys = "0.16.0+8.10.0"`，以及 `version.h` 中 `ROCKSDB_MAJOR=8, ROCKSDB_MINOR=10, ROCKSDB_PATCH=0`。

### Rust 绑定覆盖情况

| API | C++ 8.10 状态 | Rust 0.22 绑定 | 可用性 |
|-----|-------------|---------------|--------|
| HyperClockCache | ✅ 生产可用 | ✅ `Cache::new_hyper_clock_cache()` | **可直接使用** |
| async_io | ✅ 生产可用 | ✅ `ReadOptions::set_async_io()` | **可直接使用** |
| PinnableSlice | ✅ 生产可用 | ✅ `DBPinnableSlice` + `get_pinned()` | **可直接使用** |
| Ribbon Filter | ✅ 生产可用 | ✅ `set_ribbon_filter()` / `set_hybrid_ribbon_filter()` | **可直接使用** |
| Wide Columns | ⚠️ 部分NotSupported | ❌ C API (c.h) 未暴露 | **需 C++ 层绕过** |
| kBinarySearchWithFirstKey | ✅ 生产可用 | ❌ 枚举缺少变体 | **需 PR 或手动 FFI** |
| CompressedSecondaryCache | ✅ 生产可用 | ❌ C API 未暴露 | **需 C++ 层绕过** |

### 可直接使用的 4 个 API 优化方案

#### 1. HyperClockCache 替换 LRUCache (预估 QPS +10-15%)

**原理**: HyperClockCache 是无锁缓存实现，在高并发读场景下 CPU 效率远优于 LRUCache。向量查询是典型的读密集型负载，block cache 命中率直接决定 QPS。

**当前问题**: `ivf_store.rs` 中使用默认 LRUCache，高并发下锁竞争严重。

```rust
// 当前 (ivf_store.rs)
let block_cache = Cache::new_lru_cache(256 * 1024 * 1024); // 256MB LRU

// 优化后
let block_cache = Cache::new_hyper_clock_cache(
    256 * 1024 * 1024,  // capacity: 256MB
    4096,               // estimated_entry_charge: ~4KB (RocksDB block size)
);
```

**关键参数**:
- `estimated_entry_charge`: 设为 RocksDB block_size (~4096)，用于内部分片计算
- 设为 0 可启用动态调整（实验性），但建议固定值以获得稳定性能

**适用场景**: 所有 Column Family 的 block_cache（signs_cf, factors_cf, sq8_cf, centroids_cf 等）

#### 2. async_io + optimize_multiget_for_io (预估 QPS +5-10%)

**原理**: 启用异步 I/O 后，RocksDB 在顺序扫描和 MultiGet 时会异步预取 SST 文件数据，减少 I/O 等待时间。向量查询的 IVF 范围扫描正是顺序读取模式。

```rust
// 查询时设置 ReadOptions
let mut read_opts = ReadOptions::default();
read_opts.set_async_io(true);
read_opts.set_readahead_size(256 * 1024); // 256KB 预读

// DBOptions 层面
let mut opts = DBOptions::default();
// optimize_multiget_for_io 默认已启用，无需额外设置
```

**适用场景**:
- IVF 范围扫描 (signs/factors CF 的 Iterator 扫描)
- 元数据批量加载 (centroids/cluster_meta 的 MultiGet)
- 持久化查询的所有 RocksDB 读取操作

#### 3. PinnableSlice 零拷贝读取 (预估 QPS +5-8%)

**原理**: `get_pinned()` 返回 `DBPinnableSlice`，直接引用 block cache 中的数据，避免 `get()` 的内存拷贝。向量数据（signs 16B + factors 8B）频繁读取，零拷贝可显著减少内存分配和拷贝开销。

```rust
// 当前 (store.rs / ivf_store.rs)
let value = db.get_cf(cf, key)?; // 拷贝整个 value 到 Vec<u8>

// 优化后
let value = db.get_pinned_cf(cf, key)?; // 零拷贝，直接引用 block cache
if let Some(pinned) = value {
    let signs = &pinned[..signs_size];
    let factors = &pinned[signs_size..signs_size + 8];
    // 直接在 block cache 数据上计算距离，无需拷贝
}
```

**适用场景**:
- 所有 `db.get_cf()` 调用替换为 `db.get_pinned_cf()`
- signs/factors/sq8 数据读取
- 元数据读取 (centroids, cluster_meta)

#### 4. Ribbon Filter 替换 Bloom Filter (预估空间 -30%，间接 QPS +3-5%)

**原理**: Ribbon Filter 在相同误判率下使用更少空间（约节省 30%），减少 SST 文件体积和 I/O 量。

```rust
// 当前
block_opts.set_bloom_filter(10.0, true); // 10 bits/key Bloom

// 优化后
block_opts.set_hybrid_ribbon_filter(9.0, 1); // 9 bits/key Ribbon + L0 Bloom
// bloom_before_level=1: L0 用 Bloom (flush 时构建快), L1+ 用 Ribbon (空间省)
```

**适用场景**: 所有 Column Family 的 BlockBasedTableOptions

### 需 C++ 层绕过的 3 个 API

#### 5. kBinarySearchWithFirstKey (预估 QPS +5-10%)

**原理**: 在索引中存储每个数据块的第一个 key，允许迭代器延迟读取数据块直到实际需要。对 IVF 范围扫描（按 cluster_id 前缀范围读取）特别有效——可以跳过不相关数据块。

**绕过方案**: 在 C++ VectorQueryEngine 中直接使用 `kBinarySearchWithFirstKey`，Rust 侧无需绑定。

```cpp
// C++ VectorQueryEngine::Open()
rocksdb::BlockBasedTableOptions table_opts;
table_opts.index_type = rocksdb::BlockBasedTableOptions::kBinarySearchWithFirstKey;
table_opts.index_shortening = rocksdb::IndexShorteningMode::kNoShortening;
```

#### 6. CompressedSecondaryCache (预估缓存容量 +2-3x)

**原理**: 二级缓存使用 LZ4 压缩存储淘汰的 block cache 条目，相当于用 CPU 换内存，在内存有限时大幅提升有效缓存容量。

**绕过方案**: 在 C++ VectorQueryEngine 中创建 CompressedSecondaryCache，附加到 HyperClockCache。

```cpp
// C++ VectorQueryEngine::Open()
auto sec_cache = rocksdb::NewCompressedSecondaryCache(
    512 * 1024 * 1024,  // 512MB 二级缓存
    -1, false, 0.5, 0.0, nullptr, true,
    rocksdb::kDefaultCacheMetadataChargePolicy,
    rocksdb::CompressionType::kLZ4Compression);
auto cache = rocksdb::NewHyperClockCache(
    256 * 1024 * 1024, 4096);
// 将 sec_cache 附加到 cache...
```

#### 7. Wide Columns (PutEntity/GetEntity)

**原理**: 允许一个 key 下存储多个命名列，可将 signs + factors 合并为一个 Wide Column Entity，一次读取替代两次 Get。

**限制**: C API 未暴露，且 GetEntity 在 8.10 中标记为 NotSupported。

**替代方案**: 当前 signs + factors 分两个 CF 存储的设计已经足够高效（通过 Iterator 顺序读取 signs，同时按需 Get factors）。Wide Columns 的主要价值在于减少 CF 数量和管理复杂度，但 QPS 提升有限，暂不实施。

### 实施优先级

| 优先级 | API | 预估 QPS 提升 | 实施难度 | 说明 |
|--------|-----|-------------|---------|------|
| P0 | HyperClockCache | +10-15% | 低 | Rust 绑定已有，一行代码替换 |
| P0 | PinnableSlice | +5-8% | 低 | Rust 绑定已有，替换 get → get_pinned |
| P1 | async_io | +5-10% | 低 | Rust 绑定已有，ReadOptions 设置 |
| P1 | Ribbon Filter | +3-5% | 低 | Rust 绑定已有，替换 Bloom |
| P2 | kBinarySearchWithFirstKey | +5-10% | 中 | 需 C++ 层设置 |
| P2 | CompressedSecondaryCache | 间接提升 | 中 | 需 C++ 层创建 |
| P3 | Wide Columns | 有限 | 高 | C API 未暴露，暂不实施 |

---

## 二、RaBitQ vs TurboQuant QPS 差距分析与优化方案

### 性能差距

| 方法 | QPS | Recall@10 | QPS 倍率 |
|------|-----|-----------|---------|
| RaBitQ Flat+SQ8 | 3492 | 93.1% | **4.5x** |
| TQ 4bit+SQ8 | 782 | 98.7% | 1.0x |

### 根因分析: LUT Cache Thrashing

#### RaBitQ LUT (16KB, L1 常驻)

```
d=128, 1-bit 量化:
- base_size = 128/8 = 16 字节
- LUT = 16 × [f32; 256] = 16 × 1024B = 16KB
- Apple M1 L1D = 128KB → LUT 占 12.5%, 完全常驻
- 查表延迟: ~4 cycles (L1 hit)
```

#### TurboQuant LUT (64KB, 溢出 L1)

```
d=128, 4-bit 量化:
- code_sz = 128/2 = 64 字节 (每字节编码 2 个 4-bit 索引)
- LUT = 64 × [f32; 256] = 64 × 1024B = 64KB
- Apple M1 L1D = 128KB → LUT 占 50%, 加上 codes + heap 数据, 严重溢出
- 查表延迟: ~12 cycles (L2 hit, L1 miss)
- 每 vector 查表次数: 64 次 (4x RaBitQ)
- 总查表延迟: 64 × 12 = 768 cycles vs 16 × 4 = 64 cycles → 12x 差距
```

#### 延迟分解

| 阶段 | RaBitQ (cycles) | TurboQuant (cycles) | 倍率 |
|------|----------------|---------------------|------|
| LUT 构建 | ~500 | ~2000 | 4x |
| LUT 查表 | 16×4 = 64 | 64×12 = 768 | 12x |
| 因子校正 | ~20 | ~0 | - |
| **总计算** | ~584 | ~2768 | **4.7x** |

**结论**: QPS 差距的 80% 来自 LUT 查表阶段——TurboQuant 的 64KB LUT 溢出 L1 cache，每次查表需从 L2 读取，延迟是 L1 的 3 倍。

### 优化方案: Split LUT (lo/hi nibble 分离)

#### 核心思想

将 4-bit LUT 的 256 项表拆分为两个 16 项表:
- **lo_table[16]**: 低 4 位索引对应的距离贡献
- **hi_table[16]**: 高 4 位索引对应的距离贡献

```
原始 LUT:  [f32; 256] × 64 字节 = 64KB
  → 每字节查 1 次, 256 项表, 每项 4B

Split LUT: [f32; 16] × 64 × 2 = 16KB + 16KB = 32KB
  → 每字节查 2 次 (lo + hi), 16 项表, 每项 4B
  → lo_table: 64 × 16 × 4B = 4KB (完全 L1 常驻)
  → hi_table: 64 × 16 × 4B = 4KB (完全 L1 常驻)
  → 总计: 8KB (远小于 16KB RaBitQ LUT)
```

#### 数学等价性证明

```
原始: dist = Σ_j lut[j][code[j]]  (j=0..63)

其中 code[j] = hi_nibble << 4 | lo_nibble
lut[j][code[j]] = (centroids[lo_nibble] - query[2j])^2
                 + (centroids[hi_nibble] - query[2j+1])^2

Split:
lo_table[j][lo_nibble] = (centroids[lo_nibble] - query[2j])^2
hi_table[j][hi_nibble] = (centroids[hi_nibble] - query[2j+1])^2

dist = Σ_j (lo_table[j][code[j] & 0xF] + hi_table[j][code[j] >> 4])
     = Σ_j (lo_table[j][lo_nibble] + hi_table[j][hi_nibble])
     = Σ_j lut[j][code[j]]  ✓ 完全等价
```

#### 实现方案

```rust
// lloyd_max.rs: 新增 build_split_lut 方法
pub fn build_split_lut(&self, query: &[f32]) -> (Vec<[f32; 16]>, Vec<[f32; 16]>) {
    let code_sz = (self.d + 1) / 2;
    let mut lo_lut = vec![[0.0f32; 16]; code_sz];
    let mut hi_lut = vec![[0.0f32; 16]; code_sz];

    for j in 0..code_sz {
        let dim_lo = j * 2;
        let dim_hi = j * 2 + 1;
        for idx in 0..16u32 {
            if dim_lo < self.d {
                let diff = self.centroids[idx as usize] - query[dim_lo];
                lo_lut[j][idx as usize] = diff * diff;
            }
            if dim_hi < self.d {
                let diff = self.centroids[idx as usize] - query[dim_hi];
                hi_lut[j][idx as usize] = diff * diff;
            }
        }
    }
    (lo_lut, hi_lut)
}

// turboquant.rs: 使用 split LUT 搜索
fn search_with_split_lut(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let (lo_lut, hi_lut) = self.quantizer.build_split_lut(query);
    let code_sz = (self.d + 1) / 2;
    let k1 = k.max(10);
    let mut heap = BinaryHeap::with_capacity(k1 + 1);

    for i in 0..self.ntotal {
        let code = &self.codes[i * code_sz..(i + 1) * code_sz];
        let mut dist = 0.0f32;
        for j in 0..code_sz {
            let byte = code[j] as usize;
            dist += lo_lut[j][byte & 0xF] + hi_lut[j][byte >> 4];
        }
        // ... top-k 堆维护 ...
    }
    // ...
}
```

#### 预估性能提升

| 指标 | 原始 LUT | Split LUT | 提升 |
|------|---------|-----------|------|
| LUT 大小 | 64KB | 8KB | 8x 缩小 |
| L1 命中率 | ~50% (溢出) | ~100% (常驻) | 2x |
| 查表延迟 | ~12 cycles (L2) | ~4 cycles (L1) | 3x |
| 每 vector 查表次数 | 64 | 128 (2×64) | 0.5x |
| **总查表延迟** | 768 cycles | 512 cycles | **1.5x** |
| **预估 QPS** | 782 | ~1200-1500 | **1.5-2x** |

> 注意: Split LUT 查表次数翻倍 (64→128)，但每次查表延迟降低 3x (12→4 cycles)，净效果为 768→512 cycles，约 1.5x 提升。加上 L1 cache 空出后对其他数据（codes, heap）的正面影响，实际提升可能达到 1.5-2x。

#### 进一步优化: Split LUT + Early Stop

Split LUT 天然支持更细粒度的 early stop:
- 原始: 每 16 字节 (chunk) 检查一次 → 4 次检查/向量
- Split: 每 16 字节 × 2 (lo+hi) 检查一次 → 4 次检查/向量 (不变)
- 但可以更激进: 每 8 字节检查一次 → 8 次检查/向量，提前终止概率更高

### 其他 TurboQuant 优化方向

#### 2. SIMD 直接计算替代 LUT (预估 QPS +2-3x, 但复杂度高)

完全放弃 LUT，使用 SIMD 直接计算 4-bit 距离:
```cpp
// NEON: 一次处理 16 个 4-bit 索引
// 1. 从 codes 加载 8 字节 = 16 个 4-bit 索引
// 2. 拆分为 lo_nibbles 和 hi_nibbles
// 3. 使用 TBL 指令从 centroids gather 值
// 4. 计算差值平方和
```
优势: 无 LUT 构建开销，无 cache 压力
劣势: 实现复杂，NEON TBL 指令有限制

#### 3. 2-bit 量化折中 (预估 QPS +2x, recall -2%)

将 4-bit 量化降为 2-bit:
- LUT 大小: 64 × [f32; 4] = 1KB (完全 L1 常驻)
- 每 vector 查表: 256 次 × 4 cycles = 1024 cycles
- 但 2-bit 量化 recall 损失较大

---

## 三、ZeroMQ 写入/查询接口架构设计

### 结论: 写入与查询使用独立 Socket (推荐)

### 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client 应用                               │
│                                                                  │
│  ┌──────────────┐                          ┌──────────────┐     │
│  │  Query Client │                          │  Write Client │     │
│  └──────┬───────┘                          └──────┬───────┘     │
│         │ ZMQ_REQ                                 │ ZMQ_REQ     │
└─────────┼─────────────────────────────────────────┼─────────────┘
          │                                         │
          ▼                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TurboQuant Server                            │
│                                                                  │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Query Socket    │                    │  Write Socket    │   │
│  │  ZMQ_ROUTER      │                    │  ZMQ_ROUTER      │   │
│  │  Port 5555       │                    │  Port 5556       │   │
│  └────────┬─────────┘                    └────────┬─────────┘   │
│           │                                       │             │
│           ▼                                       ▼             │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Query Workers   │                    │  Write Worker    │   │
│  │  (N threads)     │                    │  (1 thread)      │   │
│  │  ┌────────────┐  │                    │  ┌────────────┐  │   │
│  │  │ Worker 1   │  │                    │  │  Sequencer │  │   │
│  │  │ ivf_search │  │                    │  │  写入队列   │  │   │
│  │  ├────────────┤  │                    │  │  批量提交   │  │   │
│  │  │ Worker 2   │  │                    │  └────────────┘  │   │
│  │  │ ivf_search │  │                    └────────┬─────────┘   │
│  │  ├────────────┤  │                             │             │
│  │  │ Worker N   │  │                             ▼             │
│  │  │ ivf_search │  │                    ┌──────────────────┐   │
│  │  └────────────┘  │                    │  Index Builder   │   │
│  └──────────────────┘                    │  (后台线程)       │   │
│                                          │  KMeans + 量化    │   │
│                                          └────────┬─────────┘   │
│                                                   │             │
│           ┌───────────────────────────────────────┘             │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Shared Engine                          │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  VectorEngine (Arc<RwLock<VectorEngine>>)       │     │   │
│  │  │  - RocksDB (HyperClockCache + PinnableSlice)    │     │   │
│  │  │  - C++ SIMD 查询引擎                            │     │   │
│  │  │  - IVF 索引 (RaBitQ / TurboQuant)              │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  Notify Socket   │  ZMQ_PUB Port 5557                       │
│  │  (事件通知)       │  - index_built 事件                      │
│  │                  │  - write_ack 事件                         │
│  └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

### 为什么使用独立 Socket?

| 维度 | 独立 Socket | 共享 Socket |
|------|-----------|------------|
| **延迟隔离** | ✅ 查询不受写入背压影响 | ❌ 写入高峰时查询延迟飙升 |
| **并发模型** | ✅ 查询 N 线程并行, 写入 1 线程串行 | ❌ 需复杂锁机制区分读写 |
| **QoS 控制** | ✅ 独立限流、优先级 | ❌ 难以区分优先级 |
| **水平扩展** | ✅ 查询节点可独立扩容 | ❌ 读写耦合 |
| **协议设计** | ✅ 各自优化序列化格式 | ❌ 混合协议复杂 |
| **RocksDB 锁** | ✅ 写入串行化避免写锁争用 | ❌ 多写线程争用写锁 |
| **实现复杂度** | 中等 | 高 |

### Socket 类型选择

| Socket | 类型 | 说明 |
|--------|------|------|
| Query Socket | `ZMQ_ROUTER` | 支持多客户端并发查询，异步回复 |
| Write Socket | `ZMQ_ROUTER` | 支持多客户端写入请求，串行处理 |
| Notify Socket | `ZMQ_PUB` | 广播索引构建完成等事件 |

> 为什么不用 `ZMQ_REP`? REP 是同步的，一次只能处理一个请求，无法并发。ROUTER 可以异步处理多个请求，适合高 QPS 场景。

### 协议设计

#### Query 协议 (Port 5555)

```rust
// 请求
#[derive(Serialize)]
enum QueryRequest {
    IVFSearch {
        query: Vec<f32>,       // [d] f32
        k: u32,
        nprobe: u32,
        refine_factor: u32,
    },
    FlatSearch {
        query: Vec<f32>,       // [d] f32
        k: u32,
        index_type: u8,       // 0=RaBitQ, 1=TurboQuant
    },
    BatchIVFSearch {
        queries: Vec<f32>,     // [n*d] f32
        n: u32,
        k: u32,
        nprobe: u32,
    },
}

// 响应
#[derive(Serialize)]
struct QueryResponse {
    results: Vec<(u32, f32)>,  // (id, distance) pairs
    latency_us: u64,
}
```

#### Write 协议 (Port 5556)

```rust
// 请求
#[derive(Serialize)]
enum WriteRequest {
    Insert {
        vectors: Vec<f32>,     // [n*d] f32
        n: u32,
        ids: Option<Vec<u32>>, // 可选指定 ID
    },
    Delete {
        ids: Vec<u32>,
    },
    BuildIndex {
        nlist: u32,
        index_type: u8,       // 0=RaBitQ+SQ8, 1=TurboQuant+SQ8
        quantization: u8,     // 4 or 6 (TurboQuant)
    },
    Flush,
}

// 响应
#[derive(Serialize)]
enum WriteResponse {
    Inserted { ids: Vec<u32> },
    Deleted { count: u32 },
    IndexBuilt { nlist: u32, ntotal: u32 },
    Flushed,
    Error { message: String },
}
```

#### Notify 协议 (Port 5557)

```rust
#[derive(Serialize)]
enum NotifyEvent {
    IndexBuilt { nlist: u32, ntotal: u32 },
    WriteProgress { inserted: u32, pending: u32 },
    Error { message: String },
}
```

### 写入流程

```
Client ──ZMQ_REQ──→ Write Socket (ZMQ_ROUTER:5556)
                         │
                         ▼
                    Write Worker (单线程)
                         │
                    ┌────┤
                    │    │
                    ▼    ▼
              立即写入    缓冲区累积
              RocksDB    (batch_size 阈值)
                    │    │
                    └────┤
                         │
                         ▼
                    写入 RocksDB (WriteBatch)
                         │
                         ▼
                    累积到阈值?
                    ┌────┤
                    │Yes │No
                    ▼    ▼
              触发索引    返回 ACK
              后台构建    (ZMQ_ROUTER reply)
                    │
                    ▼
              Notify Socket
              (ZMQ_PUB:5557)
              "index_built"
```

### 查询流程

```
Client ──ZMQ_REQ──→ Query Socket (ZMQ_ROUTER:5555)
                         │
                         ▼
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Worker 1    Worker 2    Worker N
              │          │          │
              ▼          ▼          ▼
          Arc<RwLock<VectorEngine>>::read()
              │          │          │
              ▼          ▼          ▼
          C++ SIMD IVF Search (1 FFI call)
              │          │          │
              ▼          ▼          ▼
          ZMQ_ROUTER reply → Client
```

### Rust 实现骨架

```rust
use zeromq::{RouterSocket, PubSocket};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use parking_lot::RwLock;

pub struct TurboQuantServer {
    engine: Arc<RwLock<VectorEngine>>,
    query_socket: RouterSocket,
    write_socket: RouterSocket,
    notify_socket: PubSocket,
}

impl TurboQuantServer {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                // 查询: 多线程并发处理
                msg = self.query_socket.recv() => {
                    let engine = self.engine.clone();
                    tokio::spawn(async move {
                        let req: QueryRequest = decode(&msg);
                        let guard = engine.read();
                        let resp = guard.query(req);
                        // reply via ROUTER identity frame
                    });
                }
                // 写入: 单线程串行处理
                msg = self.write_socket.recv() => {
                    let req: WriteRequest = decode(&msg);
                    let mut guard = self.engine.write();
                    let resp = guard.write(req);
                    // reply + notify
                }
            }
        }
    }
}
```

### 依赖新增

```toml
[dependencies]
zeromq = "0.4"           # ZeroMQ 绑定
tokio = { version = "1", features = ["full"] }  # 异步运行时
parking_lot = "0.12"     # 高性能 RwLock
serde = { version = "1", features = ["derive"] }
bincode = "1"            # 二进制序列化 (低延迟)
```

### 实施阶段

| Phase | 内容 | 说明 |
|-------|------|------|
| Phase 7 | ZeroMQ 基础设施 | 添加依赖, 创建 server 模块 |
| Phase 8 | Query Socket 实现 | ROUTER + 多 Worker 线程池 |
| Phase 9 | Write Socket 实现 | ROUTER + 单线程串行化 + 批量写入 |
| Phase 10 | Notify Socket 实现 | PUB 广播事件通知 |
| Phase 11 | 集成测试 + 性能验证 | 端到端 QPS 测试 |

---

## 综合优化实施计划

### 优先级排序

| 优先级 | 优化项 | 预估 QPS 提升 | 实施难度 | Phase |
|--------|--------|-------------|---------|-------|
| **P0** | HyperClockCache 替换 LRUCache | +10-15% | 低 | 已有 Rust 绑定 |
| **P0** | PinnableSlice 零拷贝 | +5-8% | 低 | 已有 Rust 绑定 |
| **P0** | TurboQuant Split LUT | +50-100% | 中 | 需修改 lloyd_max.rs + turboquant.rs |
| **P1** | async_io | +5-10% | 低 | 已有 Rust 绑定 |
| **P1** | Ribbon Filter | +3-5% | 低 | 已有 Rust 绑定 |
| **P1** | C++ kBinarySearchWithFirstKey | +5-10% | 中 | C++ 层设置 |
| **P2** | ZeroMQ Query Socket | 服务化 | 中 | 新模块 |
| **P2** | ZeroMQ Write Socket | 服务化 | 中 | 新模块 |
| **P3** | C++ CompressedSecondaryCache | 间接提升 | 中 | C++ 层创建 |
| **P3** | ZeroMQ Notify Socket | 事件通知 | 低 | 新模块 |

### 预估综合 QPS 提升

| 场景 | 当前 QPS | 优化后预估 | 提升倍率 |
|------|---------|-----------|---------|
| RaBitQ IVF np=8 | 2472 | ~3200 | 1.3x |
| TurboQuant 4bit | 782 | ~1500 | 1.9x |
| Persisted np=8 | 1776 | ~2800 | 1.6x |

---

## 四、NNG 替换 ZeroMQ 服务架构 (已实施)

### 实施状态: ✅ 完成

使用 `nng = "1.0"` crate 替换 ZeroMQ，创建了 [server.rs](src/server.rs) 模块。

### NNG vs ZeroMQ 对比

| 维度 | NNG | ZeroMQ |
|------|-----|--------|
| ROUTER 模式 | 无 (用多 Rep0 实例替代) | ZMQ_ROUTER |
| WebSocket | ✅ 原生支持 | ❌ 不支持 |
| TLS | ✅ 内置 | 需要 CurveZMQ |
| Survey 模式 | ✅ 原生支持 | ❌ 不支持 |
| C 依赖 | CMake + NNG C 库 | 纯 Rust (zeromq crate) |
| 异步支持 | 通过 runng crate | 原生 async/await |

### 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client 应用                               │
│                                                                  │
│  ┌──────────────┐                          ┌──────────────┐     │
│  │  Query Client │                          │  Write Client │     │
│  └──────┬───────┘                          └──────┬───────┘     │
│         │ NNG Req0                                │ NNG Req0    │
└─────────┼─────────────────────────────────────────┼─────────────┘
          │                                         │
          ▼                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TurboQuant Server                            │
│                                                                  │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Query Sockets   │                    │  Write Socket    │   │
│  │  N Rep0 实例     │                    │  Rep0 (单实例)   │   │
│  │  Port 5555+      │                    │  Port 5556       │   │
│  └────────┬─────────┘                    └────────┬─────────┘   │
│           │                                       │             │
│           ▼                                       ▼             │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Query Workers   │                    │  Write Worker    │   │
│  │  (N threads)     │                    │  (1 thread)      │   │
│  └──────────────────┘                    └────────┬─────────┘   │
│                                                   │             │
│           ┌───────────────────────────────────────┘             │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Shared Engine                          │   │
│  │  VectorEngine (Arc<RwLock<VectorEngine>>)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  Notify Socket   │  NNG Pub0 Port 5557                      │
│  │  (事件通知)       │  - index_built 事件                      │
│  └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

### 协议设计

```rust
// Query 协议
enum QueryRequest {
    IVFSearch { query: Vec<f32>, k: u32, nprobe: u32, refine_factor: u32 },
    FlatSearch { query: Vec<f32>, k: u32 },
}

struct QueryResponse {
    results: Vec<(u32, f32)>,
    latency_us: u64,
}

// Write 协议
enum WriteRequest {
    Insert { vectors: Vec<f32>, n: u32, ids: Option<Vec<u32>> },
    Delete { ids: Vec<u32> },
    BuildIndex { nlist: u32, index_type: u8, quantization: u8 },
    Flush,
}

// Notify 协议
enum NotifyEvent {
    IndexBuilt { nlist: u32, ntotal: u32 },
    WriteProgress { inserted: u32, pending: u32 },
}
```

---

## 五、IVF-TurboQuant 实现 (已实施)

### 实施状态: ✅ 完成

在 [ivf.rs](src/ivf.rs) 中新增 `TurboQuantIVFIndex`，支持 IVF + TurboQuant 4-bit 量化 + SQ8 精炼。

### 核心差异

| 特性 | RaBitQIVFIndex | TurboQuantIVFIndex |
|------|----------------|---------------------|
| 量化方法 | 1-bit 符号 + 8字节因子 | 4-bit Lloyd-Max |
| LUT 大小 | 16KB (d=128) | 8KB Split LUT |
| L1 命中率 | ~100% | ~100% |
| Early Stop | ❌ 不支持 | ✅ 支持 |
| 粗排精度 | 较低 | 较高 |
| 存储/向量 | 24B | 64B |

### Split LUT 原理

将 4-bit LUT 的 256 项表拆分为两个 16 项表：

```
原始 LUT:  [f32; 256] × 64 字节 = 64KB
  → 每字节查 1 次, 256 项表, 每项 4B

Split LUT: [f32; 16] × 64 × 2 = 8KB
  → 每字节查 2 次 (lo + hi), 16 项表
  → lo_table[j][byte & 0xF] + hi_table[j][byte >> 4]
  → 数学上完全等价
```

### 基准测试结果 (SIFT Small 10K×128D)

| 方法 | QPS | Recall@10 | vs RaBitQ IVF |
|------|-----|-----------|---------------|
| **TQ-IVF-256 np=8** | **6876** | 90.2% | **2.7x** QPS |
| **TQ-IVF-256 np=16** | **5150** | 97.1% | **3.7x** QPS |
| **TQ-IVF-256 np=32** | **3534** | 99.1% | **4.8x** QPS |
| TQ-IVF-64 np=8 | 3869 | 97.5% | 1.6x QPS |
| TQ-IVF-64 np=16 | 2293 | 99.4% | 1.7x QPS |
| TQ-IVF-64 np=32 | 1487 | 99.4% | 2.1x QPS |
| RaBitQ IVF-256 np=8 | 2560 | 90.1% | baseline |
| RaBitQ IVF-256 np=16 | 1386 | 96.7% | baseline |
| RaBitQ IVF-256 np=32 | 731 | 98.4% | baseline |

**关键发现**: IVF-TurboQuant 在所有 nprobe 配置下都同时实现了更高的 QPS 和更高的 Recall！

### 性能提升原因

1. **Split LUT (8KB)** 比 RaBitQ LUT (16KB) 更小，L1 命中率更高
2. **Early stop** 在 IVF 场景下效果显著（每个 cluster 内的向量数更少，top-k 堆更容易满）
3. **4-bit 量化精度**更高，粗排召回率更高，SQ8 精炼效果更好

---

## 六、RocksDB 优化实施总结 (已实施)

### 实施状态: ✅ 完成

| 优化 | 文件 | 效果 |
|------|------|------|
| HyperClockCache | ivf_store.rs, store.rs | 无锁缓存替换 LRUCache |
| PinnableSlice | store.rs | 所有 get_cf → get_pinned_cf，零拷贝读取 |
| Split LUT | lloyd_max.rs, turboquant.rs | 4-bit LUT 从 64KB 降至 8KB |
| async_io | ivf_store.rs, store.rs | ReadOptions 异步预取 |
| Ribbon Filter | ivf_store.rs, store.rs | set_hybrid_ribbon_filter 替换 Bloom |
| kBinarySearchWithFirstKey | vector_query_engine.cpp | C++ 层延迟读取数据块 |

### QPS 提升效果

| 方法 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| RaBitQ Flat+SQ8 | 3492 | 3773 | +8% |
| IVF-64 np=8 | 2472 | 2470 | ≈0% |
| TQ 4bit+SQ8 | 782 | 812 | +4% |
| **Persisted np=8** | **1776** | **2392** | **+35%** |

---

## 七、待实施项

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P2 | IVF-TurboQuant 持久化 (ivf_store.rs 扩展) | 待实施 |
| P2 | NNG 服务端完整实现 (集成 VectorEngine) | 待实施 |
| P3 | C++ CompressedSecondaryCache | 待实施 |
| P3 | 大规模数据集测试 (SIFT 1M) | 待实施 |
| **ZeroMQ 服务化后** | - | 取决于网络延迟 | - |
