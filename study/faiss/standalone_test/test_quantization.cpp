/*
 * 量化索引测试程序
 * 
 * 测试:
 * 1. TurboQuant Flat
 * 2. RaBitQ Flat
 * 3. RaBitQ IVF
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_quantization test_quantization.cpp -lm
 */

#include "quantization.h"

using namespace quant;

void print_separator(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// 测试 1: TurboQuant Flat
// ============================================================

void test_turboquant_flat() {
    print_separator("测试 1: TurboQuant Flat");
    
    size_t d = 128;
    int nbits = 4;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  位宽 nbits = " << nbits << std::endl;
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
    
    TurboQuantFlatIndex index(d, nbits);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "训练时间: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " μs" << std::endl;
    
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
    
    std::cout << "搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
    // 计算召回率
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
// 测试 2: RaBitQ Flat
// ============================================================

void test_rabitq_flat() {
    print_separator("测试 2: RaBitQ Flat");
    
    size_t d = 128;
    int nbits = 1;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  位宽 nbits = " << nbits << std::endl;
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
    
    RaBitQFlatIndex index(d, nbits);
    
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
    
    std::cout << "搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
    // 计算召回率
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
// 测试 3: RaBitQ IVF
// ============================================================

void test_rabitq_ivf() {
    print_separator("测试 3: RaBitQ IVF");
    
    size_t d = 128;
    size_t nlist = 100;
    int nbits = 1;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 20;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  位宽 nbits = " << nbits << std::endl;
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
    
    RaBitQIVFIndex index(d, nlist, nbits);
    
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
    
    std::cout << "搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
    // 计算召回率
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
// 测试 4: 不同位宽对比
// ============================================================

void test_different_nbits() {
    print_separator("测试 4: 不同位宽对比");
    
    size_t d = 128;
    size_t nb = 3000;
    size_t nq = 50;
    size_t k = 10;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    // 计算真实最近邻
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
    
    std::cout << "\n=== TurboQuant Flat ===" << std::endl;
    std::cout << "位宽 | 码大小 | Recall@" << k << std::endl;
    std::cout << "-----|--------|----------" << std::endl;
    
    for (int nbits : {2, 4, 8}) {
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
        std::cout << std::setw(4) << nbits << " | " 
                  << std::setw(6) << index.get_code_size() << " | "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
    
    std::cout << "\n=== RaBitQ Flat ===" << std::endl;
    std::cout << "位宽 | 码大小 | Recall@" << k << std::endl;
    std::cout << "-----|--------|----------" << std::endl;
    
    for (int nbits : {1, 2, 4}) {
        RaBitQFlatIndex index(d, nbits);
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
        std::cout << std::setw(4) << nbits << " | " 
                  << std::setw(6) << index.get_code_size() << " | "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
}

// ============================================================
// 测试 5: nprobe 对召回率的影响
// ============================================================

void test_nprobe_effect() {
    print_separator("测试 5: nprobe 对召回率的影响");
    
    size_t d = 128;
    size_t nlist = 100;
    int nbits = 1;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
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
    
    RaBitQIVFIndex index(d, nlist, nbits);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    
    std::cout << "\nnprobe | Recall@" << k << " | 说明" << std::endl;
    std::cout << "-------|----------|------" << std::endl;
    
    for (size_t nprobe : {1, 5, 10, 20, 50, 100}) {
        std::vector<std::vector<size_t>> result_ids;
        std::vector<std::vector<float>> result_dists;
        index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
        
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
        std::string note = (nprobe == nlist) ? "全探测" : "";
        
        std::cout << std::setw(6) << nprobe << " | " 
                  << std::fixed << std::setprecision(4) << recall << "   | " 
                  << note << std::endl;
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         量化索引测试套件                                      ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  1. TurboQuant Flat - Beta 分布量化                          ║" << std::endl;
    std::cout << "║  2. RaBitQ Flat - 随机二进制量化                             ║" << std::endl;
    std::cout << "║  3. RaBitQ IVF - 倒排索引版本                                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_turboquant_flat();
    test_rabitq_flat();
    test_rabitq_ivf();
    test_different_nbits();
    test_nprobe_effect();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
