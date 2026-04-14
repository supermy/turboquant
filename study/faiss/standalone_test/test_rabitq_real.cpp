/*
 * RaBitQ 真实数据测试 (使用合成聚类数据模拟 SIFT1M)
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_rabitq_real test_rabitq_real.cpp -lm
 */

#include "rabitq_faiss.h"

using namespace rabitq_faiss;

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

// ============================================================
// 测试 1: RaBitQ Flat (1-bit) - 聚类数据
// ============================================================

void test_rabitq_flat_clustered() {
    print_separator("测试 1: RaBitQ Flat (1-bit) - 聚类数据");
    
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
    std::cout << "  聚类数 = " << n_clusters << std::endl;
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
            dists[i] = {fvec_L2sqr(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    // RaBitQ Flat
    RaBitQFlatIndex index(d, 1, false, false);
    
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
    
    std::cout << "\n目标: ~0.76 (76%)" << std::endl;
    if (recall >= 0.70) {
        std::cout << "✓ 召回率达标!" << std::endl;
    } else {
        std::cout << "⚠ 召回率: " << recall << std::endl;
    }
}

// ============================================================
// 测试 2: RaBitQ Flat + SQ8 Refinement - 聚类数据
// ============================================================

void test_rabitq_flat_sq8_clustered() {
    print_separator("测试 2: RaBitQ Flat + SQ8 Refinement - 聚类数据");
    
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
    std::cout << "  聚类数 = " << n_clusters << std::endl;
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
            dists[i] = {fvec_L2sqr(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    // RaBitQ Flat + SQ8
    RaBitQFlatIndex index(d, 1, false, true);
    
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
    std::cout << "  RaBitQ 码大小: " << index.get_code_size() << " bytes" << std::endl;
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
    
    std::cout << "\n目标: ~0.94-0.95 (94-95%)" << std::endl;
    if (recall >= 0.90) {
        std::cout << "✓ 召回率达标!" << std::endl;
    } else {
        std::cout << "⚠ 召回率: " << recall << std::endl;
    }
}

// ============================================================
// 测试 3: RaBitQ IVF + SQ8 - 聚类数据
// ============================================================

void test_rabitq_ivf_sq8_clustered() {
    print_separator("测试 3: RaBitQ IVF + SQ8 - 聚类数据");
    
    size_t d = 128;
    size_t nlist = 256;
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    size_t n_clusters = 1000;
    size_t nprobe = 64;
    size_t refine_factor = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  探测数 nprobe = " << nprobe << std::endl;
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
            dists[i] = {fvec_L2sqr(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    // RaBitQ IVF + SQ8
    RaBitQIVFIndex index(d, nlist, 1, false, true);
    
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
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe, refine_factor);
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
    std::cout << "\nRecall@" << k << " (nprobe=" << nprobe << ", refine=" << refine_factor << "): " 
              << std::fixed << std::setprecision(4) << recall << std::endl;
}

// ============================================================
// 测试 4: 不同 refine_factor 对召回率的影响
// ============================================================

void test_refine_factor_effect_clustered() {
    print_separator("测试 4: refine_factor 对召回率的影响 (聚类数据)");
    
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
            dists[i] = {fvec_L2sqr(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) gt_ids[q][i] = dists[i].second;
    }
    
    RaBitQFlatIndex index(d, 1, false, true);
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
    std::cout << "║         RaBitQ 真实数据测试 (聚类数据模拟)                    ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  目标: 纯 RaBitQ (1-bit) 召回率 ~76%                          ║" << std::endl;
    std::cout << "║        配合 SQ8 refinement 后，召回率 ~94-95%                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_rabitq_flat_clustered();
    test_rabitq_flat_sq8_clustered();
    test_rabitq_ivf_sq8_clustered();
    test_refine_factor_effect_clustered();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
