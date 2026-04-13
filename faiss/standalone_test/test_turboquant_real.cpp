/*
 * TurboQuant Flat + SQ8 真实数据测试
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_turboquant_real test_turboquant_real.cpp -lm
 */

#include "quantization.h"

using namespace quant;

void print_separator(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

// 生成具有聚类结构的数据 (模拟真实数据)
void generate_clustered_data(
        float* data, size_t n, size_t d, size_t n_clusters,
        float cluster_std = 0.1f, uint32_t seed = 42) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // 生成聚类中心
    std::vector<float> centroids(n_clusters * d);
    for (size_t c = 0; c < n_clusters; c++) {
        for (size_t j = 0; j < d; j++) {
            centroids[c * d + j] = dist(rng);
        }
        // 归一化聚类中心
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += centroids[c * d + j] * centroids[c * d + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < d; j++) {
            centroids[c * d + j] /= norm;
        }
    }
    
    // 生成数据点
    std::normal_distribution<float> noise_dist(0.0f, cluster_std);
    for (size_t i = 0; i < n; i++) {
        size_t cluster_id = i % n_clusters;
        const float* center = centroids.data() + cluster_id * d;
        
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] = center[j] + noise_dist(rng);
        }
        
        // 归一化
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] /= norm;
        }
    }
}

// 生成查询向量 (从数据中采样 + 噪声)
void generate_queries(
        float* queries, const float* data, size_t n_data, size_t n_queries, size_t d,
        float noise_level = 0.05f, uint32_t seed = 123) {
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> idx_dist(0, n_data - 1);
    std::normal_distribution<float> noise_dist(0.0f, noise_level);
    
    for (size_t i = 0; i < n_queries; i++) {
        size_t src_idx = idx_dist(rng);
        
        for (size_t j = 0; j < d; j++) {
            queries[i * d + j] = data[src_idx * d + j] + noise_dist(rng);
        }
        
        // 归一化
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += queries[i * d + j] * queries[i * d + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < d; j++) {
            queries[i * d + j] /= norm;
        }
    }
}

// SQ8 量化器
class SQ8Quantizer {
public:
    size_t d;
    std::vector<float> vmin, vmax;
    
    SQ8Quantizer(size_t d_val) : d(d_val), vmin(d_val), vmax(d_val) {}
    
    void train(size_t n, const float* x) {
        for (size_t j = 0; j < d; j++) {
            vmin[j] = std::numeric_limits<float>::max();
            vmax[j] = std::numeric_limits<float>::lowest();
        }
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                float val = x[i * d + j];
                vmin[j] = std::min(vmin[j], val);
                vmax[j] = std::max(vmax[j], val);
            }
        }
        
        for (size_t j = 0; j < d; j++) {
            if (vmax[j] - vmin[j] < 1e-6f) {
                vmax[j] = vmin[j] + 1e-6f;
            }
        }
    }
    
    size_t code_size() const { return d; }
    
    void encode(const float* x, uint8_t* code) const {
        for (size_t j = 0; j < d; j++) {
            float val = x[j];
            float normalized = (val - vmin[j]) / (vmax[j] - vmin[j]);
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            code[j] = static_cast<uint8_t>(normalized * 255.0f);
        }
    }
    
    float compute_distance(const uint8_t* code, const float* query) const {
        float dist = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float decoded = vmin[j] + (code[j] / 255.0f) * (vmax[j] - vmin[j]);
            float diff = decoded - query[j];
            dist += diff * diff;
        }
        return dist;
    }
};

// TurboQuant Flat + SQ8 索引
class TurboQuantFlatSQ8Index {
public:
    size_t d;
    int nbits;
    
    std::unique_ptr<HadamardRotation> rotation;
    std::unique_ptr<TurboQuantMSE> quantizer;
    std::unique_ptr<SQ8Quantizer> sq8;
    
    std::vector<uint8_t> codes;
    std::vector<uint8_t> sq8_codes;
    size_t ntotal;
    
    TurboQuantFlatSQ8Index(size_t d_val, int nbits_val = 4)
        : d(d_val), nbits(nbits_val), ntotal(0) {
        
        size_t d_rotated = next_power_of_2(d);
        rotation = std::make_unique<HadamardRotation>(d);
        quantizer = std::make_unique<TurboQuantMSE>(d_rotated, nbits);
        sq8 = std::make_unique<SQ8Quantizer>(d);
    }
    
    void train(size_t n, const float* x) {
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        sq8->train(n, x_normalized.data());
        
        std::cout << "TurboQuant Flat + SQ8: 训练完成" << std::endl;
    }
    
    void add(size_t n, const float* x) {
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x_normalized.data(), x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        codes.resize((ntotal + n) * code_sz);
        sq8_codes.resize((ntotal + n) * sq8->code_size());
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x_rotated.data() + i * d_rotated;
            quantizer->encode(xi, codes.data() + (ntotal + i) * code_sz);
            
            sq8->encode(x_normalized.data() + i * d, sq8_codes.data() + (ntotal + i) * sq8->code_size());
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists,
                size_t refine_factor = 1) const {
        
        result_ids.resize(n);
        result_dists.resize(n);
        
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x_normalized.data(), x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x_rotated.data() + q * d_rotated;
            const float* query_orig = x_normalized.data() + q * d;
            
            // 第一阶段: TurboQuant 搜索
            size_t k1 = (refine_factor > 1) ? std::min(k * refine_factor, ntotal) : k;
            std::priority_queue<std::pair<float, size_t>> top_k1;
            
            for (size_t i = 0; i < ntotal; i++) {
                const uint8_t* code = codes.data() + i * code_sz;
                float dist = quantizer->compute_distance(code, query);
                
                if (top_k1.size() < k1) {
                    top_k1.push({dist, i});
                } else if (dist < top_k1.top().first) {
                    top_k1.pop();
                    top_k1.push({dist, i});
                }
            }
            
            // 第二阶段: SQ8 refinement
            std::vector<std::pair<float, size_t>> candidates;
            while (!top_k1.empty()) {
                candidates.push_back(top_k1.top());
                top_k1.pop();
            }
            
            std::priority_queue<std::pair<float, size_t>> final_top_k;
            
            if (refine_factor > 1) {
                for (auto& [tq_dist, idx] : candidates) {
                    const uint8_t* sq8_code = sq8_codes.data() + idx * sq8->code_size();
                    float refined_dist = sq8->compute_distance(sq8_code, query_orig);
                    
                    if (final_top_k.size() < k) {
                        final_top_k.push({refined_dist, idx});
                    } else if (refined_dist < final_top_k.top().first) {
                        final_top_k.pop();
                        final_top_k.push({refined_dist, idx});
                    }
                }
            } else {
                for (auto& [dist, idx] : candidates) {
                    final_top_k.push({dist, idx});
                }
            }
            
            result_ids[q].resize(k);
            result_dists[q].resize(k);
            
            for (size_t i = k; i > 0; i--) {
                if (!final_top_k.empty()) {
                    result_dists[q][i - 1] = final_top_k.top().first;
                    result_ids[q][i - 1] = final_top_k.top().second;
                    final_top_k.pop();
                } else {
                    result_dists[q][i - 1] = -1;
                    result_ids[q][i - 1] = -1;
                }
            }
        }
    }
    
    size_t get_ntotal() const { return ntotal; }
    size_t get_code_size() const { return quantizer->code_size(); }
};

// ============================================================
// 测试 1: TurboQuant Flat (4-bit) - 聚类数据
// ============================================================

void test_turboquant_flat() {
    print_separator("测试 1: TurboQuant Flat (4-bit) - 聚类数据");
    
    size_t d = 128;
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    size_t n_clusters = 1000;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  位宽 = 4-bit" << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::cout << "生成聚类数据..." << std::endl;
    generate_clustered_data(xb.data(), nb, d, n_clusters, 0.1f, 42);
    generate_queries(xq.data(), xb.data(), nb, nq, d, 0.05f, 123);
    
    // 计算真实最近邻
    std::cout << "计算真实最近邻..." << std::endl;
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    // TurboQuant Flat
    TurboQuantFlatIndex index(d, 4);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "训练时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "添加时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    std::cout << "\n索引统计:" << std::endl;
    std::cout << "  总向量数: " << index.get_ntotal() << std::endl;
    std::cout << "  码大小: " << index.get_code_size() << " bytes" << std::endl;
    
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    
    start = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, result_ids, result_dists);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " ms" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() * 1000.0 / nq) << " μs" << std::endl;
    
    // 计算召回率
    size_t total_recall = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<size_t> found(k);
        for (size_t i = 0; i < k; i++) found[i] = result_ids[q][i];
        std::sort(found.begin(), found.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                total_recall++;
            }
        }
    }
    
    float recall = static_cast<float>(total_recall) / (nq * k);
    std::cout << "\nRecall@" << k << ": " << std::fixed << std::setprecision(4) << recall << std::endl;
}

// ============================================================
// 测试 2: TurboQuant Flat + SQ8 Refinement - 聚类数据
// ============================================================

void test_turboquant_flat_sq8() {
    print_separator("测试 2: TurboQuant Flat + SQ8 Refinement - 聚类数据");
    
    size_t d = 128;
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    size_t n_clusters = 1000;
    size_t refine_factor = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  位宽 = 4-bit + SQ8" << std::endl;
    std::cout << "  refine_factor = " << refine_factor << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::cout << "生成聚类数据..." << std::endl;
    generate_clustered_data(xb.data(), nb, d, n_clusters, 0.1f, 42);
    generate_queries(xq.data(), xb.data(), nb, nq, d, 0.05f, 123);
    
    std::cout << "计算真实最近邻..." << std::endl;
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    // TurboQuant Flat + SQ8
    TurboQuantFlatSQ8Index index(d, 4);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "训练时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "添加时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    std::cout << "\n索引统计:" << std::endl;
    std::cout << "  总向量数: " << index.get_ntotal() << std::endl;
    std::cout << "  TurboQuant 码大小: " << index.get_code_size() << " bytes" << std::endl;
    std::cout << "  SQ8 码大小: " << index.sq8->code_size() << " bytes" << std::endl;
    std::cout << "  总码大小: " << (index.get_code_size() + index.sq8->code_size()) << " bytes" << std::endl;
    
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    
    start = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, result_ids, result_dists, refine_factor);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " ms" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() * 1000.0 / nq) << " μs" << std::endl;
    
    // 计算召回率
    size_t total_recall = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<size_t> found(k);
        for (size_t i = 0; i < k; i++) found[i] = result_ids[q][i];
        std::sort(found.begin(), found.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                total_recall++;
            }
        }
    }
    
    float recall = static_cast<float>(total_recall) / (nq * k);
    std::cout << "\nRecall@" << k << ": " << std::fixed << std::setprecision(4) << recall << std::endl;
}

// ============================================================
// 测试 3: 不同位宽对比
// ============================================================

void test_different_bits() {
    print_separator("测试 3: 不同位宽对比 (TurboQuant Flat)");
    
    size_t d = 128;
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    size_t n_clusters = 1000;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::cout << "生成聚类数据..." << std::endl;
    generate_clustered_data(xb.data(), nb, d, n_clusters, 0.1f, 42);
    generate_queries(xq.data(), xb.data(), nb, nq, d, 0.05f, 123);
    
    std::cout << "计算真实最近邻..." << std::endl;
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    std::cout << "\n位宽 | 码大小 | Recall@" << k << std::endl;
    std::cout << "-----|--------|----------" << std::endl;
    
    for (int nbits : {2, 4, 6, 8}) {
        TurboQuantFlatIndex index(d, nbits);
        index.train(nb, xb.data());
        index.add(nb, xb.data());
        
        std::vector<std::vector<size_t>> result_ids;
        std::vector<std::vector<float>> result_dists;
        index.search(nq, xq.data(), k, result_ids, result_dists);
        
        size_t total_recall = 0;
        for (size_t q = 0; q < nq; q++) {
            std::vector<size_t> found(k);
            for (size_t i = 0; i < k; i++) found[i] = result_ids[q][i];
            std::sort(found.begin(), found.end());
            for (size_t i = 0; i < k; i++) {
                if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                    total_recall++;
                }
            }
        }
        
        float recall = static_cast<float>(total_recall) / (nq * k);
        
        std::cout << std::setw(4) << nbits << "-bit | " 
                  << std::setw(6) << index.get_code_size() << "B | "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
}

// ============================================================
// 测试 4: refine_factor 对召回率的影响
// ============================================================

void test_refine_factor_effect() {
    print_separator("测试 4: refine_factor 对召回率的影响 (TurboQuant + SQ8)");
    
    size_t d = 128;
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    size_t n_clusters = 1000;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::cout << "生成聚类数据..." << std::endl;
    generate_clustered_data(xb.data(), nb, d, n_clusters, 0.1f, 42);
    generate_queries(xq.data(), xb.data(), nb, nq, d, 0.05f, 123);
    
    std::cout << "计算真实最近邻..." << std::endl;
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    TurboQuantFlatSQ8Index index(d, 4);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    
    std::cout << "\nrefine_factor | Recall@" << k << std::endl;
    std::cout << "--------------|----------" << std::endl;
    
    for (size_t rf : {1, 3, 5, 10, 20, 50, 100}) {
        std::vector<std::vector<size_t>> result_ids;
        std::vector<std::vector<float>> result_dists;
        index.search(nq, xq.data(), k, result_ids, result_dists, rf);
        
        size_t total_recall = 0;
        for (size_t q = 0; q < nq; q++) {
            std::vector<size_t> found(k);
            for (size_t i = 0; i < k; i++) found[i] = result_ids[q][i];
            std::sort(found.begin(), found.end());
            for (size_t i = 0; i < k; i++) {
                if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                    total_recall++;
                }
            }
        }
        
        float recall = static_cast<float>(total_recall) / (nq * k);
        
        std::cout << std::setw(13) << rf << " | " 
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         TurboQuant Flat + SQ8 真实数据测试                   ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  TurboQuant: Beta 分布量化 + Hadamard 旋转                   ║" << std::endl;
    std::cout << "║  SQ8: 8-bit 标量量化 refinement                              ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_turboquant_flat();
    test_turboquant_flat_sq8();
    test_different_bits();
    test_refine_factor_effect();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
