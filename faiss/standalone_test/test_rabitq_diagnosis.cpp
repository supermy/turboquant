/*
 * RaBitQ 诊断测试
 * 
 * 分析 RaBitQ 召回率低的原因
 */

#include "quantization.h"

using namespace quant;

void diagnose_rabitq() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         RaBitQ 诊断测试                                      ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    size_t d = 128;
    size_t nb = 1000;
    size_t nq = 10;
    size_t k = 10;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    // L2 归一化
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "1. 原始向量暴力搜索" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<std::vector<size_t>> gt_ids(nq);
    std::vector<std::vector<float>> gt_dists(nq);
    
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq.data() + q * d, xb.data() + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        
        gt_ids[q].resize(k);
        gt_dists[q].resize(k);
        for (size_t i = 0; i < k; i++) {
            gt_ids[q][i] = dists[i].second;
            gt_dists[q][i] = dists[i].first;
        }
        
        std::cout << "Query " << q << ": 前3个距离 = " 
                  << gt_dists[q][0] << ", " << gt_dists[q][1] << ", " << gt_dists[q][2] << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "2. Hadamard 旋转后向量暴力搜索" << std::endl;
    std::cout << "========================================" << std::endl;
    
    size_t d_rotated = next_power_of_2(d);
    HadamardRotation hr(d);
    
    std::vector<float> xb_rotated(nb * d_rotated);
    std::vector<float> xq_rotated(nq * d_rotated);
    hr.apply_batch(nb, xb.data(), xb_rotated.data());
    hr.apply_batch(nq, xq.data(), xq_rotated.data());
    
    std::vector<std::vector<size_t>> rotated_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {l2_distance(xq_rotated.data() + q * d_rotated, 
                                    xb_rotated.data() + i * d_rotated, d_rotated), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        
        rotated_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) {
            rotated_ids[q][i] = dists[i].second;
        }
    }
    
    size_t rotated_recall = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<size_t> found(k);
        for (size_t i = 0; i < k; i++) found[i] = rotated_ids[q][i];
        std::sort(found.begin(), found.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                rotated_recall++;
            }
        }
    }
    std::cout << "旋转后 Recall@" << k << ": " << std::fixed << std::setprecision(4) 
              << (float)rotated_recall / (nq * k) << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "3. RaBitQ 量化后距离分析" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 计算质心
    std::vector<float> centroid(d_rotated, 0.0f);
    for (size_t i = 0; i < nb; i++) {
        for (size_t j = 0; j < d_rotated; j++) {
            centroid[j] += xb_rotated[i * d_rotated + j];
        }
    }
    for (size_t j = 0; j < d_rotated; j++) {
        centroid[j] /= nb;
    }
    
    // 编码所有向量
    RaBitQuantizer quantizer(d_rotated, 1);
    quantizer.centroid = centroid;
    
    size_t code_sz = quantizer.code_size();
    std::vector<uint8_t> codes(nb * code_sz);
    
    for (size_t i = 0; i < nb; i++) {
        quantizer.encode(xb_rotated.data() + i * d_rotated, codes.data() + i * code_sz);
    }
    
    // 比较真实距离 vs 量化距离
    std::cout << "\nQuery 0 的距离对比:" << std::endl;
    std::cout << "向量ID | 真实距离 | 量化距离 | 差异" << std::endl;
    std::cout << "-------|----------|----------|------" << std::endl;
    
    const float* query = xq_rotated.data();
    float qr_to_c_l2sqr = 0.0f;
    for (size_t j = 0; j < d_rotated; j++) {
        float diff = query[j] - centroid[j];
        qr_to_c_l2sqr += diff * diff;
    }
    
    for (size_t i = 0; i < 10; i++) {
        float true_dist = l2_distance(query, xb_rotated.data() + i * d_rotated, d_rotated);
        float quant_dist = quantizer.compute_distance(codes.data() + i * code_sz, query, qr_to_c_l2sqr);
        float diff = true_dist - quant_dist;
        
        std::cout << std::setw(6) << i << " | " 
                  << std::fixed << std::setprecision(4) << true_dist << "   | "
                  << quant_dist << "   | "
                  << std::showpos << diff << std::noshowpos << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "4. 量化后暴力搜索" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<std::vector<size_t>> quant_ids(nq);
    for (size_t q = 0; q < nq; q++) {
        const float* qvec = xq_rotated.data() + q * d_rotated;
        
        float qr_l2sqr = 0.0f;
        for (size_t j = 0; j < d_rotated; j++) {
            float diff = qvec[j] - centroid[j];
            qr_l2sqr += diff * diff;
        }
        
        std::vector<std::pair<float, size_t>> dists(nb);
        for (size_t i = 0; i < nb; i++) {
            dists[i] = {quantizer.compute_distance(codes.data() + i * code_sz, qvec, qr_l2sqr), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        
        quant_ids[q].resize(k);
        for (size_t i = 0; i < k; i++) {
            quant_ids[q][i] = dists[i].second;
        }
    }
    
    size_t quant_recall = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<size_t> found(k);
        for (size_t i = 0; i < k; i++) found[i] = quant_ids[q][i];
        std::sort(found.begin(), found.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(found.begin(), found.end(), gt_ids[q][i])) {
                quant_recall++;
            }
        }
    }
    std::cout << "量化后 Recall@" << k << ": " << std::fixed << std::setprecision(4) 
              << (float)quant_recall / (nq * k) << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "5. 问题分析" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n召回率分解:" << std::endl;
    std::cout << "  旋转后召回率: " << std::fixed << std::setprecision(4) 
              << (float)rotated_recall / (nq * k) << std::endl;
    std::cout << "  量化后召回率: " << (float)quant_recall / (nq * k) << std::endl;
    
    if ((float)rotated_recall / (nq * k) > 0.9) {
        std::cout << "\n✓ Hadamard 旋转保持距离关系良好" << std::endl;
    } else {
        std::cout << "\n⚠ Hadamard 旋转导致距离关系变化" << std::endl;
    }
    
    if ((float)quant_recall / (nq * k) < 0.5) {
        std::cout << "⚠ 量化误差较大，需要检查距离计算公式" << std::endl;
    }
}

int main() {
    diagnose_rabitq();
    return 0;
}
