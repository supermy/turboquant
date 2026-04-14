/*
 * RaBitQ 原样复刻测试
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_rabitq_faiss test_rabitq_faiss.cpp -lm
 */

#include "rabitq_faiss.h"

using namespace rabitq_faiss;

void print_separator(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// 测试 1: RaBitQ Flat (1-bit) 召回率
// ============================================================

void test_rabitq_flat() {
    print_separator("测试 1: RaBitQ Flat (1-bit) 召回率");
    
    size_t d = 128;
    size_t nb = 10000;
    size_t nq = 100;
    size_t k = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    // 计算真实最近邻
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
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
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
        std::cout << "⚠ 召回率偏低，需要检查实现" << std::endl;
    }
}

// ============================================================
// 测试 2: RaBitQ Flat + SQ8 Refinement 召回率
// ============================================================

void test_rabitq_flat_sq8() {
    print_separator("测试 2: RaBitQ Flat + SQ8 Refinement 召回率");
    
    size_t d = 128;
    size_t nb = 10000;
    size_t nq = 100;
    size_t k = 10;
    size_t refine_factor = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  refine_factor = " << refine_factor << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    // 计算真实最近邻
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
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
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
        std::cout << "⚠ 召回率偏低， 銶检查实现" << std::endl;
    }
}

// ============================================================
// 测试 3: RaBitQ IVF 召回率
// ============================================================

void test_rabitq_ivf() {
    print_separator("测试 3: RaBitQ IVF 召回率");
    
    size_t d = 128;
    size_t nlist = 100;
    size_t nb = 10000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 30;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  探测数 nprobe = " << nprobe << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    // 计算真实最近邻
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
    
    // RaBitQ IVF
    RaBitQIVFIndex index(d, nlist, 1, false, false);
    
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
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
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
    std::cout << "\nRecall@" << k << " (nprobe=" << nprobe << "): " << std::fixed << std::setprecision(4) << recall << std::endl;
}

// ============================================================
// 测试 4: RaBitQ IVF + SQ8 Refinement 召回率
// ============================================================

void test_rabitq_ivf_sq8() {
    print_separator("测试 4: RaBitQ IVF + SQ8 Refinement 召回率");
    
    size_t d = 128;
    size_t nlist = 100;
    size_t nb = 10000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 30;
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
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    // 计算真实最近邻
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
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
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
    std::cout << "\nRecall@" << k << " (nprobe=" << nprobe << ", refine=" << refine_factor << " << : " 
                  << std::fixed << std::setprecision(4) << recall << std::endl;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         RaBitQ 原样复刻测试                              ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  目标: 纯 RaBitQ (1-bit) 召回率 ~76%                      ║" << std::endl;
    std::cout << "║        配合 SQ8 refinement 后， 召回率 ~94-95%       ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_rabitq_flat();
    test_rabitq_flat_sq8();
    test_rabitq_ivf();
    test_rabitq_ivf_sq8();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
