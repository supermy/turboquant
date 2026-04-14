/*
 * RaBitQ 详细诊断测试
 */

#include "quantization.h"

using namespace quant;

void detailed_diagnosis() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         RaBitQ 详细诊断                                      ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    size_t d = 128;
    size_t nb = 100;
    size_t nq = 1;
    
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < nb * d; i++) xb[i] = dist(rng);
    for (size_t i = 0; i < nq * d; i++) xq[i] = dist(rng);
    
    for (size_t i = 0; i < nb; i++) l2_normalize(xb.data() + i * d, d);
    for (size_t i = 0; i < nq; i++) l2_normalize(xq.data() + i * d, d);
    
    size_t d_rotated = next_power_of_2(d);
    HadamardRotation hr(d);
    
    std::vector<float> xb_rotated(nb * d_rotated);
    std::vector<float> xq_rotated(nq * d_rotated);
    hr.apply_batch(nb, xb.data(), xb_rotated.data());
    hr.apply_batch(nq, xq.data(), xq_rotated.data());
    
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
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "1. 质心统计" << std::endl;
    std::cout << "========================================" << std::endl;
    float centroid_norm = 0.0f;
    for (size_t j = 0; j < d_rotated; j++) {
        centroid_norm += centroid[j] * centroid[j];
    }
    std::cout << "质心范数: " << std::sqrt(centroid_norm) << std::endl;
    std::cout << "质心前5维: ";
    for (size_t j = 0; j < 5; j++) {
        std::cout << centroid[j] << " ";
    }
    std::cout << std::endl;
    
    // 编码第一个向量
    RaBitQuantizer quantizer(d_rotated, 1);
    quantizer.centroid = centroid;
    
    size_t code_sz = quantizer.code_size();
    std::vector<uint8_t> code(code_sz);
    
    const float* vec0 = xb_rotated.data();
    quantizer.encode(vec0, code.data());
    
    const float* factors = reinterpret_cast<const float*>(code.data() + (d_rotated + 7) / 8);
    float or_minus_c_l2sqr = factors[0];
    float dp_multiplier = factors[1];
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "2. 向量0 编码分析" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 手动计算
    float norm_L2sqr_manual = 0.0f;
    float dp_oO_manual = 0.0f;
    int bit_count = 0;
    for (size_t i = 0; i < d_rotated; i++) {
        float or_minus_c = vec0[i] - centroid[i];
        norm_L2sqr_manual += or_minus_c * or_minus_c;
        if (or_minus_c > 0) {
            dp_oO_manual += or_minus_c;
            bit_count++;
        } else {
            dp_oO_manual -= or_minus_c;
        }
    }
    
    float sqrt_norm_L2 = std::sqrt(norm_L2sqr_manual);
    float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d_rotated));
    float normalized_dp = dp_oO_manual / sqrt_norm_L2 * inv_d_sqrt;
    float dp_multiplier_manual = sqrt_norm_L2 / normalized_dp;
    
    std::cout << "||or - c||^2: " << norm_L2sqr_manual << std::endl;
    std::cout << "||or - c||: " << sqrt_norm_L2 << std::endl;
    std::cout << "dp_oO (sum|or-c|): " << dp_oO_manual << std::endl;
    std::cout << "normalized_dp: " << normalized_dp << std::endl;
    std::cout << "dp_multiplier (手动): " << dp_multiplier_manual << std::endl;
    std::cout << "dp_multiplier (编码): " << dp_multiplier << std::endl;
    std::cout << "bit count: " << bit_count << " / " << d_rotated << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "3. 查询向量分析" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const float* query = xq_rotated.data();
    
    float qr_to_c_l2sqr = 0.0f;
    float sum_q_minus_c = 0.0f;
    for (size_t i = 0; i < d_rotated; i++) {
        float q_minus_c = query[i] - centroid[i];
        qr_to_c_l2sqr += q_minus_c * q_minus_c;
        sum_q_minus_c += q_minus_c;
    }
    
    std::cout << "||qr - c||^2: " << qr_to_c_l2sqr << std::endl;
    std::cout << "||qr - c||: " << std::sqrt(qr_to_c_l2sqr) << std::endl;
    std::cout << "sum(qr - c): " << sum_q_minus_c << std::endl;
    
    // 计算 dot_qo
    float dot_qo = 0.0f;
    uint64_t sum_bits = 0;
    for (size_t i = 0; i < d_rotated; i++) {
        bool bit = (code[i / 8] & (1 << (i % 8))) != 0;
        if (bit) {
            float q_minus_c = query[i] - centroid[i];
            dot_qo += q_minus_c;
            sum_bits++;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "4. 距离计算分析" << std::endl;
    std::cout << "========================================" << std::endl;
    
    float c1 = 2.0f * inv_d_sqrt;
    float c34 = sum_q_minus_c * inv_d_sqrt;
    
    float final_dot = c1 * dot_qo - c34;
    
    std::cout << "c1 = 2 / sqrt(d) = " << c1 << std::endl;
    std::cout << "c34 = sum(qr-c) / sqrt(d) = " << c34 << std::endl;
    std::cout << "dot_qo = " << dot_qo << std::endl;
    std::cout << "sum_bits = " << sum_bits << std::endl;
    std::cout << "final_dot = c1 * dot_qo - c34 = " << final_dot << std::endl;
    
    float quant_dist = or_minus_c_l2sqr + qr_to_c_l2sqr - 2.0f * dp_multiplier * final_dot;
    
    // 计算真实距离
    float true_dist = 0.0f;
    for (size_t i = 0; i < d_rotated; i++) {
        float diff = query[i] - vec0[i];
        true_dist += diff * diff;
    }
    
    std::cout << "\n量化距离: " << quant_dist << std::endl;
    std::cout << "真实距离: " << true_dist << std::endl;
    std::cout << "差异: " << (quant_dist - true_dist) << std::endl;
    
    // FAISS 论文中的公式分析
    std::cout << "\n========================================" << std::endl;
    std::cout << "5. 公式验证" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 根据 RaBitQ 论文，距离估计公式为：
    // ||or - q||^2 ≈ ||or - c||^2 + ||qr - c||^2 - 2 * ||or - c|| * ||qr - c|| * <q̃, õ>
    // 其中 <q̃, õ> 是归一化后的内积估计
    
    float sqrt_or_minus_c = std::sqrt(or_minus_c_l2sqr);
    float sqrt_qr_minus_c = std::sqrt(qr_to_c_l2sqr);
    
    // 理想情况下，final_dot 应该接近 <qr-c, or-c> / (||qr-c|| * ||or-c||)
    float true_cosine = 0.0f;
    for (size_t i = 0; i < d_rotated; i++) {
        true_cosine += (query[i] - centroid[i]) * (vec0[i] - centroid[i]);
    }
    true_cosine /= (sqrt_qr_minus_c * sqrt_or_minus_c);
    
    float estimated_cosine = final_dot * dp_multiplier / (sqrt_qr_minus_c * sqrt_or_minus_c);
    
    std::cout << "真实余弦相似度 <qr-c, or-c>: " << true_cosine << std::endl;
    std::cout << "估计余弦相似度: " << estimated_cosine << std::endl;
    std::cout << "余弦相似度差异: " << (estimated_cosine - true_cosine) << std::endl;
    
    // 使用真实余弦计算距离
    float ideal_dist = or_minus_c_l2sqr + qr_to_c_l2sqr - 2 * sqrt_or_minus_c * sqrt_qr_minus_c * true_cosine;
    std::cout << "\n理想距离 (使用真实余弦): " << ideal_dist << std::endl;
    std::cout << "量化距离 (使用估计余弦): " << quant_dist << std::endl;
    std::cout << "真实距离: " << true_dist << std::endl;
}

int main() {
    detailed_diagnosis();
    return 0;
}
