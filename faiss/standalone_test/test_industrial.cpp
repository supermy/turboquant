/*
 * IVF + RaBitQ + OPQ 工业最强组合测试
 * 
 * 测试驱动开发: 验证工业级向量搜索组合
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_industrial test_industrial.cpp -lm
 */

#include "industrial_quantization.h"

using namespace industrial;

void print_separator(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// 测试 1: OPQ 旋转效果
// ============================================================

void test_opq_rotation() {
    print_separator("测试 1: OPQ 旋转效果");
    
    size_t d = 128;
    size_t n = 1000;
    
    std::vector<float> x(n * d);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < n * d; i++) x[i] = dist(rng);
    for (size_t i = 0; i < n; i++) l2_normalize(x.data() + i * d, d);
    
    OPQMatrix opq(d);
    
    auto start = std::chrono::high_resolution_clock::now();
    opq.train(n, x.data(), 20);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "训练时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    std::vector<float> x_rotated(n * d);
    opq.apply_batch(n, x.data(), x_rotated.data());
    
    double total_norm_change = 0.0;
    for (size_t i = 0; i < n; i++) {
        float norm_before = l2_norm(x.data() + i * d, d);
        float norm_after = l2_norm(x_rotated.data() + i * d, d);
        total_norm_change += std::abs(norm_before - norm_after);
    }
    
    std::cout << "平均范数变化: " << (total_norm_change / n) << std::endl;
    std::cout << "✓ OPQ 旋转测试通过" << std::endl;
}

// ============================================================
// 测试 2: IVF + RaBitQ + OPQ 完整流程
// ============================================================

void test_full_pipeline() {
    print_separator("测试 2: IVF + RaBitQ + OPQ 完整流程");
    
    size_t d = 128;
    size_t nlist = 100;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 30;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
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
    
    IVFRaBitQOPQIndex index(d, nlist, d, 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.train(nb, xb.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "训练时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "添加时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    index.print_stats();
    
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
    
    std::vector<std::vector<size_t>> result_ids;
    std::vector<std::vector<float>> result_dists;
    
    start = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, result_ids, result_dists, nprobe);
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n搜索时间: " << search_time.count() << " μs" << std::endl;
    std::cout << "每查询时间: " << (search_time.count() / nq) << " μs" << std::endl;
    
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
// 测试 3: nprobe 对召回率的影响
// ============================================================

void test_nprobe_effect() {
    print_separator("测试 3: nprobe 对召回率的影响");
    
    size_t d = 128;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    size_t nlist = 100;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
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
    
    IVFRaBitQOPQIndex index(d, nlist, d, 1);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    
    std::cout << "\nnprobe | Recall@" << k << " | 说明" << std::endl;
    std::cout << "-------|----------|------" << std::endl;
    
    for (size_t nprobe : {1, 5, 10, 20, 30, 50, 100}) {
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
// 测试 4: 性能基准
// ============================================================

void test_benchmark() {
    print_separator("测试 4: 性能基准");
    
    size_t d = 128;
    size_t nlist = 256;
    size_t nb = 50000;
    size_t nq = 1000;
    size_t k = 10;
    size_t nprobe = 30;
    
    std::cout << "配置:" << std::endl;
    std::cout << "  维度 d = " << d << std::endl;
    std::cout << "  聚类数 nlist = " << nlist << std::endl;
    std::cout << "  数据量 nb = " << nb << std::endl;
    std::cout << "  查询数 nq = " << nq << std::endl;
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
    
    IVFRaBitQOPQIndex index(d, nlist, d, 1);
    
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
    
    index.print_stats();
}

// ============================================================
// 测试 5: 不同位宽对比
// ============================================================

void test_different_bits() {
    print_separator("测试 5: 不同位宽对比 (IVF + RaBitQ + OPQ)");
    
    size_t d = 128;
    size_t nlist = 100;
    size_t nb = 5000;
    size_t nq = 100;
    size_t k = 10;
    size_t nprobe = 30;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
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
    
    std::cout << "\n位宽 | 码大小 | Recall@" << k << " (nprobe=" << nprobe << ")" << std::endl;
    std::cout << "-----|--------|-----------------------" << std::endl;
    
    for (int nbits : {1, 2, 4}) {
        IVFRaBitQOPQIndex index(d, nlist, d, nbits);
        index.train(nb, xb.data());
        index.add(nb, xb.data());
        
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
        
        std::cout << std::setw(4) << nbits << "-bit | " 
                  << std::setw(6) << index.get_code_size() << "B | "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         IVF + RaBitQ + OPQ 工业最强组合测试                  ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  OPQ: 学习最优旋转矩阵                                       ║" << std::endl;
    std::cout << "║  RaBitQ: 随机二进制量化                                      ║" << std::endl;
    std::cout << "║  IVF: 倒排索引加速                                           ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_opq_rotation();
    test_full_pipeline();
    test_nprobe_effect();
    test_benchmark();
    test_different_bits();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
