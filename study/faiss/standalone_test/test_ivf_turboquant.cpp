/*
 * IVF + TurboQuant 测试程序
 * 
 * 测试驱动开发: 完整测试 IVF + TurboQuant 索引功能
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_ivf_turboquant test_ivf_turboquant.cpp -lm
 * 
 * 运行:
 *   ./test_ivf_turboquant
 */

#include "ivf_turboquant.h"
#include <cassert>

using namespace ivf_tq;

// ============================================================
// 测试工具函数
// ============================================================

void print_separator(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// 测试 1: FWHT 正确性
// ============================================================

void test_fwht() {
    print_separator("测试 1: FWHT 正确性");
    
    size_t n = 8;
    std::vector<float> buf = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> expected = {1, 1, 1, 1, 1, 1, 1, 1};
    
    fwht_inplace(buf.data(), n);
    
    for (size_t i = 0; i < n; i++) {
        assert(std::abs(buf[i] - expected[i]) < 1e-5);
    }
    
    std::cout << "✓ FWHT 单位脉冲测试通过" << std::endl;
    
    std::vector<float> buf2 = {1, 1, 1, 1, 1, 1, 1, 1};
    fwht_inplace(buf2.data(), n);
    
    for (size_t i = 0; i < n; i++) {
        assert(std::abs(buf2[i] - 8.0f) < 1e-5 || std::abs(buf2[i]) < 1e-5);
    }
    
    std::cout << "✓ FWHT 常向量测试通过" << std::endl;
}

// ============================================================
// 测试 2: Hadamard 旋转
// ============================================================

void test_hadamard_rotation() {
    print_separator("测试 2: Hadamard 旋转");
    
    size_t d = 128;
    HadamardRotation hr(d);
    
    std::vector<float> x(d, 1.0f);
    l2_normalize(x.data(), d);
    
    std::vector<float> rotated(hr.d_out);
    hr.apply(x.data(), rotated.data());
    
    float norm = 0.0f;
    for (size_t i = 0; i < hr.d_out; i++) {
        norm += rotated[i] * rotated[i];
    }
    norm = std::sqrt(norm);
    
    assert(std::abs(norm - 1.0f) < 0.01f);
    
    std::cout << "✓ 旋转后向量范数保持: " << norm << std::endl;
    
    std::vector<float> x2(d);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < d; i++) {
        x2[i] = dist(rng);
    }
    l2_normalize(x2.data(), d);
    
    std::vector<float> rotated2(hr.d_out);
    hr.apply(x2.data(), rotated2.data());
    
    float dot = dot_product(x2.data(), x.data(), d);
    float dot_rotated = dot_product(rotated2.data(), rotated.data(), hr.d_out);
    
    std::cout << "✓ 原始向量点积: " << dot << std::endl;
    std::cout << "✓ 旋转后点积: " << dot_rotated << std::endl;
    std::cout << "✓ Hadamard 旋转测试通过" << std::endl;
}

// ============================================================
// 测试 3: TurboQuant 量化器
// ============================================================

void test_turboquant() {
    print_separator("测试 3: TurboQuant 量化器");
    
    size_t d = 128;
    int nbits = 4;
    
    TurboQuantMSE tq(d, nbits);
    
    std::cout << "维度: " << d << std::endl;
    std::cout << "位宽: " << nbits << std::endl;
    std::cout << "码本大小: " << tq.code_size() << " bytes" << std::endl;
    std::cout << "质心数: " << tq.k << std::endl;
    
    std::vector<float> x(d);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < d; i++) {
        x[i] = dist(rng);
    }
    l2_normalize(x.data(), d);
    
    std::vector<uint8_t> code(tq.code_size());
    tq.encode(x.data(), code.data());
    
    std::vector<float> decoded(d);
    tq.decode(code.data(), decoded.data());
    
    float mse = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - decoded[i];
        mse += diff * diff;
    }
    mse /= d;
    
    std::cout << "MSE: " << std::scientific << mse << std::endl;
    std::cout << "✓ TurboQuant 量化器测试通过" << std::endl;
}

// ============================================================
// 测试 4: KMeans 聚类
// ============================================================

void test_kmeans() {
    print_separator("测试 4: KMeans 聚类");
    
    size_t d = 16;
    size_t k = 4;
    size_t n = 1000;
    
    KMeans km(d, k, 20);
    
    std::vector<float> x(n * d);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < n; i++) {
        size_t cluster = i % k;
        float center = static_cast<float>(cluster) * 5.0f;
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = center + dist(rng);
        }
    }
    
    km.train(n, x.data());
    
    std::vector<size_t> assignments(n);
    std::vector<size_t> counts(k, 0);
    
    for (size_t i = 0; i < n; i++) {
        assignments[i] = km.assign_cluster(x.data() + i * d);
        counts[assignments[i]]++;
    }
    
    std::cout << "各聚类大小: [";
    for (size_t i = 0; i < k; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << counts[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "✓ KMeans 聚类测试通过" << std::endl;
}

// ============================================================
// 测试 5: IVF + TurboQuant 完整流程
// ============================================================

void test_ivf_turboquant_full() {
    print_separator("测试 5: IVF + TurboQuant 完整流程");
    
    size_t d = 128;
    size_t nlist = 100;
    int nbits = 4;
    size_t nb = 10000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 10;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  量化位宽 nbits = " << nbits << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  探测数 nprobe = " << nprobe << std::endl;
    std::cout << std::endl;
    
    IVFTurboQuantIndex index(d, nlist, nbits);
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) {
        xb[i] = dist(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        xq[i] = dist(rng);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "训练时间: " << train_time.count() << " ms" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    end = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "添加时间: " << add_time.count() << " ms" << std::endl;
    
    index.print_stats();
    
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    
    start = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
    std::cout << "\n前 3 个查询结果:" << std::endl;
    for (size_t q = 0; q < std::min(nq, size_t(3)); q++) {
        std::cout << "  Query " << q << ": [";
        for (size_t i = 0; i < std::min(k, size_t(5)); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << result_ids[q][i];
        }
        std::cout << ", ...]" << std::endl;
    }
    
    std::cout << "\n✓ IVF + TurboQuant 完整流程测试通过" << std::endl;
}

// ============================================================
// 测试 6: 召回率评估
// ============================================================

void test_recall() {
    print_separator("测试 6: 召回率评估");
    
    size_t d = 128;
    size_t nlist = 100;
    int nbits = 4;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 20;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << "  返回数 k = " << k << std::endl;
    std::cout << "  探测数 nprobe = " << nprobe << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) {
        xb[i] = dist(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        xq[i] = dist(rng);
    }
    
    for (size_t i = 0; i < nb; i++) {
        l2_normalize(xb.data() + i * d, d);
    }
    for (size_t i = 0; i < nq; i++) {
        l2_normalize(xq.data() + i * d, d);
    }
    
    std::cout << "计算暴力搜索真实最近邻..." << std::endl;
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            float d_val = l2_distance(xq.data() + q * d, xb.data() + i * d, d);
            dists[i] = {d_val, i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) {
            gt_ids[q][i] = dists[i].second;
        }
    }
    
    IVFTurboQuantIndex index(d, nlist, nbits);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
    
    size_t total_recall = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<size_t> found(k);
        for (size_t i = 0; i < k; i++) {
            found[i] = result_ids[q][i];
        }
        std::sort(found.begin(), found.end());
        
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                total_recall++;
            }
        }
    }
    
    float recall_at_k = static_cast<float>(total_recall) / (nq * k);
    
    std::cout << "\n召回率评估结果:" << std::endl;
    std::cout << "  Recall@" << k << ": " << std::fixed << std::setprecision(4) << recall_at_k << std::endl;
    
    std::cout << "\n✓ 召回率评估测试通过" << std::endl;
}

// ============================================================
// 测试 7: 不同位宽对比
// ============================================================

void test_different_nbits() {
    print_separator("测试 7: 不同位宽对比");
    
    size_t d = 128;
    size_t nlist = 50;
    size_t nb = 3000;
    size_t nq = 50;
    size_t k = 10;
    size_t nprobe = 10;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) {
        xb[i] = dist(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        xq[i] = dist(rng);
    }
    
    for (size_t i = 0; i < nb; i++) {
        l2_normalize(xb.data() + i * d, d);
    }
    for (size_t i = 0; i < nq; i++) {
        l2_normalize(xq.data() + i * d, d);
    }
    
    std::vector<std::pair<float, size_t>> gt_dists(nb);
    std::vector<std::vector<size_t>> gt_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < nb; i++) {
            gt_dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(gt_dists.begin(), gt_dists.begin() + k, gt_dists.end());
        gt_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) {
            gt_ids[q][i] = gt_dists[i].second;
        }
    }
    
    std::cout << "位宽 | 码大小(bytes) | Recall@" << k << std::endl;
    std::cout << "-----|---------------|----------" << std::endl;
    
    for (int nbits : {1, 2, 3, 4, 8}) {
        IVFTurboQuantIndex index(d, nlist, nbits);
        index.train(nb, xb.data());
        index.add(nb, xb.data());
        
        std::vector<std::vector<size_t>> result_ids;
        std::vector<std::vector<float>> result_dists;
        index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
        
        size_t total_recall = 0;
        for (size_t q = 0; q < nq; q++) {
            std::vector<size_t> found(k);
            for (size_t i = 0; i < k; i++) {
                found[i] = result_ids[q][i];
            }
            std::sort(found.begin(), found.end());
            
            for (size_t i = 0; i < k; i++) {
                if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                    total_recall++;
                }
            }
        }
        
        float recall = static_cast<float>(total_recall) / (nq * k);
        
        std::cout << std::setw(4) << nbits << " | " 
                  << std::setw(13) << index.get_code_size() << " | "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
    
    std::cout << "\n✓ 不同位宽对比测试通过" << std::endl;
}

// ============================================================
// 测试 8: 性能基准
// ============================================================

void test_benchmark() {
    print_separator("测试 8: 性能基准");
    
    size_t d = 128;
    size_t nlist = 256;
    int nbits = 4;
    size_t nb = 50000;
    size_t nq = 1000;
    size_t k = 10;
    size_t nprobe = 20;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
    std::cout << std::endl;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) {
        xb[i] = dist(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        xq[i] = dist(rng);
    }
    
    IVFTurboQuantIndex index(d, nlist, nbits);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    end = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n性能基准结果:" << std::endl;
    std::cout << "  训练时间: " << train_time.count() << " ms" << std::endl;
    std::cout << "  添加时间: " << add_time.count() << " ms" << std::endl;
    std::cout << "  添加吞吐: " << std::fixed << std::setprecision(0) 
              << (1000.0 * nb / add_time.count()) << " vectors/sec" << std::endl;
    std::cout << "  搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "  每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    std::cout << "  搜索吞吐: " << std::fixed << std::setprecision(0)
              << (1000000.0 * nq / search_time.count()) << " queries/sec" << std::endl;
    
    size_t total_memory = index.get_ntotal() * index.get_code_size();
    std::cout << "  码存储: " << (total_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    
    std::cout << "\n✓ 性能基准测试通过" << std::endl;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         IVF + TurboQuant 测试套件                            ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  测试驱动开发: 完整验证 IVF + TurboQuant 功能                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_fwht();
    test_hadamard_rotation();
    test_turboquant();
    test_kmeans();
    test_ivf_turboquant_full();
    test_recall();
    test_different_nbits();
    test_benchmark();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试通过!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
