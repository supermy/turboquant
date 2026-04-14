#include "vector_engine.h"
#include "simd_distance.h"

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/iterator.h>
#include <rocksdb/sst_partitioner.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/compaction_filter.h>
#include <rocksdb/table.h>

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <queue>
#include <numeric>
#include <memory>

namespace vq {

struct RaBitQCodecData {
    float c1;
    float c34;
    bool is_inner_product;
};

struct SQ8QuantizerData {
    std::vector<float> vmin;
    std::vector<float> scale;
    int d;
};

struct LloydMaxQuantizerData {
    std::vector<float> centroids;
    int nbits;
    int d;
    int code_size;
};

class VectorQueryEngine {
public:
    VectorQueryEngine() : db_(nullptr), d_(0), nlist_(0), ntotal_(0),
                          is_inner_product_(false), use_sq8_(false) {}
    ~VectorQueryEngine() {
        if (db_) {
            delete db_;
            db_ = nullptr;
        }
    }

    bool Open(const std::string& path) {
        rocksdb::Options opts;
        opts.create_if_missing = false;
        opts.create_missing_column_families = false;

        auto build_table_opts = []() -> rocksdb::BlockBasedTableOptions {
            rocksdb::BlockBasedTableOptions table_opts;
            table_opts.index_type = rocksdb::BlockBasedTableOptions::kBinarySearchWithFirstKey;
            table_opts.format_version = 5;
            table_opts.cache_index_and_filter_blocks = true;
            table_opts.pin_l0_filter_and_index_blocks_in_cache = true;
            return table_opts;
        };

        auto make_cf_opts = [&]() -> rocksdb::Options {
            rocksdb::Options cf_opts;
            auto* table_opts = new rocksdb::BlockBasedTableOptions(build_table_opts());
            cf_opts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(*table_opts));
            delete table_opts;
            return cf_opts;
        };

        std::vector<rocksdb::ColumnFamilyDescriptor> cfs;
        cfs.emplace_back("default", rocksdb::Options());
        cfs.emplace_back("rabitq_signs", make_cf_opts());
        cfs.emplace_back("rabitq_factors", make_cf_opts());
        cfs.emplace_back("rabitq_sq8", rocksdb::Options());
        cfs.emplace_back("centroids", rocksdb::Options());
        cfs.emplace_back("cluster_meta", rocksdb::Options());
        cfs.emplace_back("tq_codes", make_cf_opts());
        cfs.emplace_back("tq_sq8", rocksdb::Options());

        std::vector<rocksdb::ColumnFamilyHandle*> handles;
        rocksdb::Status s = rocksdb::DB::Open(opts, path, cfs, &handles, &db_);
        if (!s.ok()) {
            return false;
        }

        for (auto* h : handles) {
            cf_map_[h->GetName()] = h;
        }

        LoadMeta();
        return true;
    }

    std::vector<VQQueryResult> IVFSearch(
        const float* query, int d, int k,
        int nprobe, int refine_factor, bool use_sq8)
    {
        int k1 = use_sq8 ? std::min(k * refine_factor, ntotal_) : k;

        auto nearest = NearestClusters(query, nprobe);
        auto candidates = RaBitQScan(query, nearest, k1);

        if (!use_sq8) {
            std::partial_sort(candidates.begin(),
                             candidates.begin() + std::min(k, (int)candidates.size()),
                             candidates.end(),
                             [](const auto& a, const auto& b) { return a.distance < b.distance; });

            std::vector<VQQueryResult> results;
            int n = std::min(k, (int)candidates.size());
            results.reserve(n);
            for (int i = 0; i < n; i++) {
                results.push_back({candidates[i].global_id, candidates[i].distance});
            }
            return results;
        }

        return SQ8Refine(query, candidates, k);
    }

    std::vector<VQQueryResult> FlatSearch(
        const float* query, int d, int k, int index_type)
    {
        std::vector<VQQueryResult> results;
        return results;
    }

private:
    struct Candidate {
        float distance;
        uint32_t global_id;
        uint16_t cluster_id;
        uint16_t local_id;
    };

    rocksdb::DB* db_;
    std::unordered_map<std::string, rocksdb::ColumnFamilyHandle*> cf_map_;

    int d_;
    int nlist_;
    int ntotal_;
    bool is_inner_product_;
    bool use_sq8_;

    std::vector<float> centroids_;
    std::vector<RaBitQCodecData> codecs_;
    std::vector<SQ8QuantizerData> sq8_quants_;
    std::vector<uint32_t> cluster_counts_;
    std::vector<uint64_t> cluster_offsets_;

    void LoadMeta() {
        auto* meta_cf = GetCF("default");
        if (!meta_cf) return;

        std::string meta_val;
        db_->Get(rocksdb::ReadOptions(), meta_cf, rocksdb::Slice("meta"), &meta_val);
        if (meta_val.size() < 32) return;

        size_t offset = 0;
        memcpy(&d_, meta_val.data() + offset, 4); offset += 4;
        int ntotal_i;
        memcpy(&ntotal_i, meta_val.data() + offset, 4); offset += 4;
        ntotal_ = ntotal_i;
        int nlist_i;
        memcpy(&nlist_i, meta_val.data() + offset, 4); offset += 4;
        nlist_ = nlist_i;
        memcpy(&is_inner_product_, meta_val.data() + offset, 1); offset += 1;
        memcpy(&use_sq8_, meta_val.data() + offset, 1); offset += 1;

        LoadCentroids();
        LoadClusterMeta();
        LoadSQ8Quantizers();
    }

    void LoadCentroids() {
        auto* cf = GetCF("centroids");
        if (!cf) return;

        centroids_.resize(nlist_ * d_);
        for (int c = 0; c < nlist_; c++) {
            std::string key = "c" + std::to_string(c);
            std::string val;
            db_->Get(rocksdb::ReadOptions(), cf, rocksdb::Slice(key), &val);
            if (val.size() >= d_ * sizeof(float)) {
                memcpy(&centroids_[c * d_], val.data(), d_ * sizeof(float));
            }
        }

        codecs_.resize(nlist_);
        for (int c = 0; c < nlist_; c++) {
            float inv_d = 1.0f / sqrtf((float)d_);
            codecs_[c].c1 = 2.0f * inv_d;
            codecs_[c].c34 = 0.0f;
            codecs_[c].is_inner_product = is_inner_product_;
        }
    }

    void LoadClusterMeta() {
        auto* cf = GetCF("cluster_meta");
        if (!cf) return;

        cluster_counts_.resize(nlist_);
        cluster_offsets_.resize(nlist_ + 1);
        uint64_t offset = 0;

        for (int c = 0; c < nlist_; c++) {
            std::string key((const char*)&c, 2);
            std::string val;
            db_->Get(rocksdb::ReadOptions(), cf, rocksdb::Slice(key), &val);
            if (val.size() >= 4) {
                uint32_t count;
                memcpy(&count, val.data(), 4);
                cluster_counts_[c] = count;
            } else {
                cluster_counts_[c] = 0;
            }
            cluster_offsets_[c] = offset;
            offset += cluster_counts_[c];
        }
        cluster_offsets_[nlist_] = offset;
    }

    void LoadSQ8Quantizers() {
        if (!use_sq8_) return;
        sq8_quants_.resize(nlist_);
        for (int c = 0; c < nlist_; c++) {
            sq8_quants_[c].d = d_;
            sq8_quants_[c].vmin.resize(d_);
            sq8_quants_[c].scale.resize(d_);
        }
    }

    rocksdb::ColumnFamilyHandle* GetCF(const std::string& name) {
        auto it = cf_map_.find(name);
        return it != cf_map_.end() ? it->second : nullptr;
    }

    static std::string ClusterKey(uint16_t cluster_id, uint32_t local_id) {
        std::string key(8, '\0');
        memcpy(&key[0], &cluster_id, 2);
        memcpy(&key[2], &local_id, 4);
        return key;
    }

    std::vector<int> NearestClusters(const float* query, int nprobe) {
        std::vector<std::pair<float, int>> dists(nlist_);
        for (int c = 0; c < nlist_; c++) {
            dists[c] = {l2_distance_neon(query, &centroids_[c * d_], d_), c};
        }
        int np = std::min(nprobe, nlist_);
        std::partial_sort(dists.begin(), dists.begin() + np, dists.end());
        std::vector<int> result(np);
        for (int i = 0; i < np; i++) {
            result[i] = dists[i].second;
        }
        return result;
    }

    std::vector<Candidate> RaBitQScan(
        const float* query, const std::vector<int>& clusters, int k1)
    {
        std::vector<Candidate> top_k1;
        top_k1.reserve(k1 + 1);
        float max_dist = std::numeric_limits<float>::max();

        auto* cf_signs = GetCF("rabitq_signs");
        auto* cf_factors = GetCF("rabitq_factors");
        if (!cf_signs || !cf_factors) return top_k1;

        int signs_size = (d_ + 7) / 8;
        std::vector<float> lut_buf(signs_size * 256);

        for (int cluster_id : clusters) {
            const float* centroid = &centroids_[cluster_id * d_];
            const auto& codec = codecs_[cluster_id];

            float c1, c34, qr_to_c_l2sqr;
            build_rabitq_lut(query, centroid, d_, lut_buf.data(), &c1, &c34, &qr_to_c_l2sqr);

            std::string start = ClusterKey((uint16_t)cluster_id, 0);
            std::string end = ClusterKey((uint16_t)(cluster_id + 1), 0);

            rocksdb::ReadOptions opts;
            opts.verify_checksums = false;
            opts.fill_cache = true;
            opts.readahead_size = 256 * 1024;

            auto iter = db_->NewIterator(opts, cf_signs);
            int local_id = 0;

            for (iter->Seek(start); iter->Valid(); iter->Next()) {
                auto key = iter->key();
                if (key.compare(rocksdb::Slice(end)) >= 0) break;

                auto signs_val = iter->value();
                const uint8_t* signs = (const uint8_t*)signs_val.data();

                float dot_qo = rabitq_signs_distance(signs, lut_buf.data(), signs_size);

                std::string fkey = ClusterKey((uint16_t)cluster_id, (uint32_t)local_id);
                std::string factors_val;
                rocksdb::ReadOptions fopts;
                fopts.verify_checksums = false;
                fopts.fill_cache = true;
                db_->Get(fopts, cf_factors, rocksdb::Slice(fkey), &factors_val);

                float or_minus_c_l2sqr = 0.0f, dp_multiplier = 1.0f;
                if (factors_val.size() >= 8) {
                    memcpy(&or_minus_c_l2sqr, factors_val.data(), 4);
                    memcpy(&dp_multiplier, factors_val.data() + 4, 4);
                }

                float dist = rabitq_full_distance(
                    dot_qo, or_minus_c_l2sqr, dp_multiplier,
                    c1, c34, qr_to_c_l2sqr, is_inner_product_);

                uint32_t global_id = (uint32_t)(cluster_offsets_[cluster_id] + local_id);

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

    std::vector<VQQueryResult> SQ8Refine(
        const float* query, const std::vector<Candidate>& candidates, int k)
    {
        auto* cf_sq8 = GetCF("rabitq_sq8");
        if (!cf_sq8) return {};

        std::vector<std::pair<float, uint32_t>> refined;
        refined.reserve(candidates.size());

        for (const auto& cand : candidates) {
            if (cand.cluster_id < (int)sq8_quants_.size()) {
                auto& sq8 = sq8_quants_[cand.cluster_id];
                uint64_t gid = cand.global_id;
                std::string key((const char*)&gid, 8);
                std::string val;
                rocksdb::ReadOptions opts;
                opts.verify_checksums = false;
                opts.fill_cache = true;
                db_->Get(opts, cf_sq8, rocksdb::Slice(key), &val);

                if (val.size() >= (size_t)d_) {
                    float dist = sq8_distance(
                        (const uint8_t*)val.data(), query,
                        sq8.vmin.data(), sq8.scale.data(), d_);
                    refined.push_back({dist, cand.global_id});
                }
            }
        }

        std::partial_sort(refined.begin(),
                         refined.begin() + std::min(k, (int)refined.size()),
                         refined.end());

        std::vector<VQQueryResult> results;
        int n = std::min(k, (int)refined.size());
        results.reserve(n);
        for (int i = 0; i < n; i++) {
            results.push_back({refined[i].second, refined[i].first});
        }
        return results;
    }
};

} // namespace vq

// ==================== C API Implementation ====================

static vq::VectorQueryEngine* to_engine(void* p) {
    return static_cast<vq::VectorQueryEngine*>(p);
}

void* vq_engine_open(const char* path) {
    auto* engine = new vq::VectorQueryEngine();
    if (!engine->Open(std::string(path))) {
        delete engine;
        return nullptr;
    }
    return engine;
}

void vq_engine_close(void* engine) {
    delete to_engine(engine);
}

VQQueryResult* vq_ivf_search(void* engine, const VQIVFSearchParams* params, int* n_results) {
    auto results = to_engine(engine)->IVFSearch(
        params->query, params->d, params->k,
        params->nprobe, params->refine_factor, params->use_sq8 != 0);

    *n_results = (int)results.size();
    if (results.empty()) return nullptr;

    auto* out = (VQQueryResult*)malloc(results.size() * sizeof(VQQueryResult));
    memcpy(out, results.data(), results.size() * sizeof(VQQueryResult));
    return out;
}

VQQueryResult* vq_flat_search(void* engine, const VQFlatSearchParams* params, int* n_results) {
    auto results = to_engine(engine)->FlatSearch(
        params->query, params->d, params->k, params->index_type);

    *n_results = (int)results.size();
    if (results.empty()) return nullptr;

    auto* out = (VQQueryResult*)malloc(results.size() * sizeof(VQQueryResult));
    memcpy(out, results.data(), results.size() * sizeof(VQQueryResult));
    return out;
}

VQQueryResult* vq_ivf_batch_search(
    void* engine,
    const float* queries, int n_queries,
    int d, int k, int nprobe, int refine_factor, int use_sq8,
    int** n_results_per_query)
{
    size_t total = 0;
    std::vector<std::vector<VQQueryResult>> all_results(n_queries);

    for (int q = 0; q < n_queries; q++) {
        VQIVFSearchParams params;
        params.query = queries + q * d;
        params.d = d;
        params.k = k;
        params.nprobe = nprobe;
        params.refine_factor = refine_factor;
        params.use_sq8 = use_sq8;
        all_results[q] = to_engine(engine)->IVFSearch(
            params.query, params.d, params.k,
            params.nprobe, params.refine_factor, params.use_sq8 != 0);
        total += all_results[q].size();
    }

    auto* n_arr = (int*)malloc(n_queries * sizeof(int));
    auto* out = (VQQueryResult*)malloc(total * sizeof(VQQueryResult));
    size_t offset = 0;
    for (int q = 0; q < n_queries; q++) {
        n_arr[q] = (int)all_results[q].size();
        memcpy(out + offset, all_results[q].data(), all_results[q].size() * sizeof(VQQueryResult));
        offset += all_results[q].size();
    }
    *n_results_per_query = n_arr;
    return out;
}

void vq_results_free(VQQueryResult* results) {
    free(results);
}

void vq_n_results_free(int* n_results) {
    free(n_results);
}
