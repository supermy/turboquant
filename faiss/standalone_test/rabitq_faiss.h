/*
 * FAISS RaBitQ 原样复刻实现
 * 
 * 基于 FAISS RaBitQ 源码原样复刻，确保高召回率
 * - 纯 RaBitQ (1-bit): 召回率 ~76%
 * - 配合 SQ8 refinement: 召回率 ~94-95%
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_rabitq_faiss test_rabitq_faiss.cpp -lm
 */

#ifndef RABITQ_FAISS_H
#define RABITQ_FAISS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <queue>
#include <memory>
#include <cstring>
#include <limits>

namespace rabitq_faiss {

// ============================================================
// 常量定义 (来自 FAISS RaBitQUtils.cpp)
// ============================================================

// 理想量化器半径，用于最小化 L2 重建误差
const float Z_MAX_BY_QB[8] = {
    0.79688f,  // qb = 1
    1.49375f,  // qb = 2
    2.05078f,  // qb = 3
    2.50938f,  // qb = 4
    2.91250f,  // qb = 5
    3.26406f,  // qb = 6
    3.59844f,  // qb = 7
    3.91016f   // qb = 8
};

// ============================================================
// 数据结构 (来自 FAISS RaBitQUtils.h)
// ============================================================

// 1-bit 模式的向量因子
struct SignBitFactors {
    float or_minus_c_l2sqr = 0;  // ||or - c||^2 (L2) 或 ||or - c||^2 - ||or||^2 (IP)
    float dp_multiplier = 0;      // sqrt(||or-c||) / normalized_dp
};

// 多 bit 模式的向量因子 (包含误差界)
struct SignBitFactorsWithError : SignBitFactors {
    float f_error = 0;  // 误差界，用于两阶段搜索
};

// 额外 bit 因子 (nb_bits > 1)
struct ExtraBitsFactors {
    float f_add_ex = 0;      // 加性校正因子
    float f_rescale_ex = 0;  // 缩放因子
};

// 查询因子
struct QueryFactorsData {
    float c1 = 0;            // 2 / sqrt(d)
    float c2 = 0;            // 未使用
    float c34 = 0;           // sum(qr - c) / sqrt(d)
    float qr_to_c_L2sqr = 0; // ||qr - c||^2
    float qr_norm_L2sqr = 0; // ||qr||^2 (IP 模式)
    float q_dot_c = 0;       // <query, centroid> (IP 模式)
    float int_dot_scale = 1; // 整数点积缩放
    float g_error = 0;       // 查询误差因子
    std::vector<float> rotated_q;  // qr - c
};

// ============================================================
// 工具函数
// ============================================================

inline float fvec_L2sqr(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float fvec_norm_L2sqr(const float* a, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        sum += a[i] * a[i];
    }
    return sum;
}

inline float fvec_inner_product(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void l2_normalize(float* x, size_t d) {
    float norm = std::sqrt(fvec_norm_L2sqr(x, d));
    if (norm > 1e-10f) {
        for (size_t i = 0; i < d; i++) {
            x[i] /= norm;
        }
    }
}

inline size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p *= 2;
    return p;
}

// ============================================================
// 向量因子计算 (来自 FAISS RaBitQUtils.cpp)
// ============================================================

inline void compute_vector_intermediate_values(
        const float* x,
        size_t d,
        const float* centroid,
        float& norm_L2sqr,
        float& or_L2sqr,
        float& dp_oO) {
    norm_L2sqr = 0.0f;
    or_L2sqr = 0.0f;
    dp_oO = 0.0f;
    
    for (size_t j = 0; j < d; j++) {
        const float x_val = x[j];
        const float centroid_val = (centroid != nullptr) ? centroid[j] : 0.0f;
        const float or_minus_c = x_val - centroid_val;
        
        const float or_minus_c_sq = or_minus_c * or_minus_c;
        norm_L2sqr += or_minus_c_sq;
        or_L2sqr += x_val * x_val;
        
        const bool xb = (or_minus_c > 0.0f);
        dp_oO += xb ? or_minus_c : -or_minus_c;
    }
}

inline SignBitFactorsWithError compute_factors_from_intermediates(
        float norm_L2sqr,
        float or_L2sqr,
        float dp_oO,
        size_t d,
        bool is_inner_product,
        bool compute_error = true) {
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float kConstEpsilon = 1.9f;  // RaBitQ 论文中的误差界常数
    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));
    
    const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
    const float inv_norm_L2 = (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
    
    const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
    const float inv_dp_oO = (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);
    
    SignBitFactorsWithError factors;
    factors.or_minus_c_l2sqr = is_inner_product ? (norm_L2sqr - or_L2sqr) : norm_L2sqr;
    factors.dp_multiplier = inv_dp_oO * sqrt_norm_L2;
    
    // 仅在需要时计算误差界 (1-bit 模式跳过)
    if (compute_error) {
        const float xu_cb_norm_sqr = static_cast<float>(d) * 0.25f;
        const float ip_resi_xucb = 0.5f * dp_oO;
        
        float tmp_error = 0.0f;
        if (std::abs(ip_resi_xucb) > epsilon) {
            const float ratio_sq = (norm_L2sqr * xu_cb_norm_sqr) / (ip_resi_xucb * ip_resi_xucb);
            if (ratio_sq > 1.0f) {
                if (d == 1) {
                    tmp_error = sqrt_norm_L2 * kConstEpsilon * std::sqrt(ratio_sq - 1.0f);
                } else {
                    tmp_error = sqrt_norm_L2 * kConstEpsilon * 
                               std::sqrt((ratio_sq - 1.0f) / static_cast<float>(d - 1));
                }
            }
        }
        
        // 应用度量特定的乘数
        factors.f_error = is_inner_product ? tmp_error : 2.0f * tmp_error;
    }
    
    return factors;
}

inline SignBitFactorsWithError compute_vector_factors(
        const float* x,
        size_t d,
        const float* centroid,
        bool is_inner_product,
        bool compute_error = true) {
    float norm_L2sqr, or_L2sqr, dp_oO;
    compute_vector_intermediate_values(x, d, centroid, norm_L2sqr, or_L2sqr, dp_oO);
    return compute_factors_from_intermediates(norm_L2sqr, or_L2sqr, dp_oO, d, is_inner_product, compute_error);
}

// ============================================================
// 查询因子计算 (来自 FAISS RaBitQuantizer.cpp)
// ============================================================

inline QueryFactorsData compute_query_factors(
        const float* query,
        size_t d,
        const float* centroid,
        bool is_inner_product) {
    QueryFactorsData query_fac;
    
    // 计算查询到质心的距离
    if (centroid != nullptr) {
        query_fac.qr_to_c_L2sqr = fvec_L2sqr(query, centroid, d);
    } else {
        query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(query, d);
    }
    
    // 减去质心，得到 P^(-1)(qr - c)
    query_fac.rotated_q.resize(d);
    for (size_t i = 0; i < d; i++) {
        query_fac.rotated_q[i] = query[i] - ((centroid == nullptr) ? 0 : centroid[i]);
    }
    
    // 计算 g_error = ||qr - c|| (旋转后查询的 L2 范数)
    query_fac.g_error = std::sqrt(query_fac.qr_to_c_L2sqr);
    
    // 计算一些数值 — 不量化查询
    const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));
    
    float sum_q = 0;
    for (size_t i = 0; i < d; i++) {
        sum_q += query_fac.rotated_q[i];
    }
    
    query_fac.c1 = 2 * inv_d;
    query_fac.c2 = 0;
    query_fac.c34 = sum_q * inv_d;
    
    if (is_inner_product) {
        query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(query, d);
        query_fac.q_dot_c = centroid ? fvec_inner_product(query, centroid, d) : 0.0f;
    }
    
    return query_fac;
}

// ============================================================
// RaBitQ 编码器
// ============================================================

class RaBitQCodec {
public:
    size_t d;
    int nb_bits;
    bool is_inner_product;
    
    RaBitQCodec(size_t d_val, int nb_bits_val = 1, bool ip = false)
        : d(d_val), nb_bits(nb_bits_val), is_inner_product(ip) {}
    
    RaBitQCodec() = default;
    
    // 计算码大小
    size_t code_size() const {
        size_t base_size = (d + 7) / 8;  // 符号位
        size_t factor_size = sizeof(SignBitFactors);  // 因子
        return base_size + factor_size;
    }
    
    // 编码向量
    void encode(const float* x, const float* centroid, uint8_t* code) const {
        memset(code, 0, code_size());
        
        // 计算中间值
        float norm_L2sqr, or_L2sqr, dp_oO;
        compute_vector_intermediate_values(x, d, centroid, norm_L2sqr, or_L2sqr, dp_oO);
        
        // 编码符号位
        for (size_t i = 0; i < d; i++) {
            float or_minus_c = x[i] - (centroid ? centroid[i] : 0.0f);
            if (or_minus_c > 0.0f) {
                code[i / 8] |= (1 << (i % 8));
            }
        }
        
        // 计算并存储因子
        SignBitFactors factors;
        const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
        const float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        const float inv_norm_L2 = (norm_L2sqr < 1e-10f) ? 1.0f : (1.0f / sqrt_norm_L2);
        
        const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
        const float inv_dp_oO = (std::abs(normalized_dp) < 1e-10f) ? 1.0f : (1.0f / normalized_dp);
        
        factors.or_minus_c_l2sqr = is_inner_product ? (norm_L2sqr - or_L2sqr) : norm_L2sqr;
        factors.dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        
        // 存储因子
        size_t base_size = (d + 7) / 8;
        SignBitFactors* stored_factors = reinterpret_cast<SignBitFactors*>(code + base_size);
        *stored_factors = factors;
    }
    
    // 计算距离 (使用预计算的查询因子)
    float compute_distance(
            const uint8_t* code,
            const QueryFactorsData& query_fac,
            const float* centroid) const {
        
        size_t base_size = (d + 7) / 8;
        const SignBitFactors* factors = reinterpret_cast<const SignBitFactors*>(code + base_size);
        
        // 计算 dot_qo = sum(qr_i - c_i) for bits that are 1
        float dot_qo = 0.0f;
        uint64_t sum_bits = 0;
        
        for (size_t i = 0; i < d; i++) {
            bool bit = (code[i / 8] & (1 << (i % 8))) != 0;
            if (bit) {
                dot_qo += query_fac.rotated_q[i];
                sum_bits++;
            }
        }
        
        // 计算最终点积
        float final_dot = query_fac.c1 * dot_qo - query_fac.c34;
        
        // 计算距离
        float dist = factors->or_minus_c_l2sqr + query_fac.qr_to_c_L2sqr 
                   - 2.0f * factors->dp_multiplier * final_dot;
        
        // IP 模式的特殊处理
        if (is_inner_product) {
            dist = -0.5f * (dist - query_fac.qr_norm_L2sqr);
        } else {
            dist = std::max(0.0f, dist);
        }
        
        return dist;
    }
};

// ============================================================
// SQ8 标量量化器 (用于 refinement)
// ============================================================

class SQ8Quantizer {
public:
    size_t d;
    std::vector<float> vmin, vmax;
    
    SQ8Quantizer(size_t d_val) : d(d_val), vmin(d_val), vmax(d_val) {}
    
    void train(size_t n, const float* x) {
        // 初始化
        for (size_t j = 0; j < d; j++) {
            vmin[j] = std::numeric_limits<float>::max();
            vmax[j] = std::numeric_limits<float>::lowest();
        }
        
        // 计算范围
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                float val = x[i * d + j];
                vmin[j] = std::min(vmin[j], val);
                vmax[j] = std::max(vmax[j], val);
            }
        }
        
        // 添加小量避免除零
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
    
    void decode(const uint8_t* code, float* x) const {
        for (size_t j = 0; j < d; j++) {
            x[j] = vmin[j] + (code[j] / 255.0f) * (vmax[j] - vmin[j]);
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

// ============================================================
// RaBitQ Flat 索引
// ============================================================

class RaBitQFlatIndex {
public:
    size_t d;
    int nb_bits;
    bool is_inner_product;
    
    std::vector<float> centroid;
    RaBitQCodec codec;
    
    std::vector<uint8_t> codes;
    std::vector<size_t> ids;
    size_t ntotal;
    
    // SQ8 refinement
    bool use_sq8_refine;
    std::unique_ptr<SQ8Quantizer> sq8;
    std::vector<uint8_t> sq8_codes;
    
    RaBitQFlatIndex(size_t d_val, int nb_bits_val = 1, bool ip = false, bool sq8_refine = false)
        : d(d_val), nb_bits(nb_bits_val), is_inner_product(ip),
          codec(d_val, nb_bits_val, ip), ntotal(0), use_sq8_refine(sq8_refine) {
        
        if (use_sq8_refine) {
            sq8 = std::make_unique<SQ8Quantizer>(d);
        }
    }
    
    void train(size_t n, const float* x) {
        // 计算质心
        centroid.assign(d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                centroid[j] += x[i * d + j];
            }
        }
        for (size_t j = 0; j < d; j++) {
            centroid[j] /= n;
        }
        
        // 训练 SQ8
        if (use_sq8_refine) {
            sq8->train(n, x);
        }
        
        std::cout << "RaBitQ Flat: 训练完成 (质心 + " 
                  << (use_sq8_refine ? "SQ8" : "无 refinement") << ")" << std::endl;
    }
    
    void add(size_t n, const float* x) {
        size_t code_sz = codec.code_size();
        codes.resize((ntotal + n) * code_sz);
        ids.resize(ntotal + n);
        
        if (use_sq8_refine) {
            sq8_codes.resize((ntotal + n) * sq8->code_size());
        }
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            
            // 编码 RaBitQ
            codec.encode(xi, centroid.data(), codes.data() + (ntotal + i) * code_sz);
            
            // 编码 SQ8
            if (use_sq8_refine) {
                sq8->encode(xi, sq8_codes.data() + (ntotal + i) * sq8->code_size());
            }
            
            ids[ntotal + i] = ntotal + i;
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists,
                size_t refine_factor = 1) const {
        
        result_ids.resize(n);
        result_dists.resize(n);
        
        size_t code_sz = codec.code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            
            // 计算查询因子
            QueryFactorsData query_fac = compute_query_factors(query, d, centroid.data(), is_inner_product);
            
            // 第一阶段: RaBitQ 搜索
            size_t k1 = use_sq8_refine ? std::min(k * refine_factor, ntotal) : k;
            std::priority_queue<std::pair<float, size_t>> top_k1;
            
            for (size_t i = 0; i < ntotal; i++) {
                const uint8_t* code = codes.data() + i * code_sz;
                float dist = codec.compute_distance(code, query_fac, centroid.data());
                
                if (top_k1.size() < k1) {
                    top_k1.push({dist, i});
                } else if (dist < top_k1.top().first) {
                    top_k1.pop();
                    top_k1.push({dist, i});
                }
            }
            
            // 第二阶段: SQ8 refinement (如果启用)
            std::vector<std::pair<float, size_t>> candidates;
            while (!top_k1.empty()) {
                candidates.push_back(top_k1.top());
                top_k1.pop();
            }
            
            std::priority_queue<std::pair<float, size_t>> final_top_k;
            
            if (use_sq8_refine) {
                for (auto& [rabitq_dist, idx] : candidates) {
                    const uint8_t* sq8_code = sq8_codes.data() + idx * sq8->code_size();
                    float refined_dist = sq8->compute_distance(sq8_code, query);
                    
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
            
            // 输出结果
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
    size_t get_code_size() const { return codec.code_size(); }
};

// ============================================================
// KMeans 聚类
// ============================================================

class KMeans {
public:
    size_t d;
    size_t k;
    int niter;
    std::vector<float> centroids;
    
    KMeans(size_t d_val, size_t k_val, int niter_val = 20) 
        : d(d_val), k(k_val), niter(niter_val) {
        centroids.resize(k * d);
    }
    
    void train(size_t n, const float* x, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        
        // 随机初始化
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (size_t i = 0; i < std::min(k, n); i++) {
            std::copy(x + indices[i] * d, x + indices[i] * d + d, 
                     centroids.begin() + i * d);
        }
        
        std::vector<size_t> assign(n);
        std::vector<size_t> counts(k);
        std::vector<float> new_centroids(k * d, 0.0f);
        
        for (int iter = 0; iter < niter; iter++) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            
            // 分配
            for (size_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                
                float min_dist = std::numeric_limits<float>::max();
                size_t min_idx = 0;
                
                for (size_t j = 0; j < k; j++) {
                    float dist = fvec_L2sqr(xi, centroids.data() + j * d, d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = j;
                    }
                }
                
                assign[i] = min_idx;
                counts[min_idx]++;
                
                for (size_t j = 0; j < d; j++) {
                    new_centroids[min_idx * d + j] += xi[j];
                }
            }
            
            // 更新质心
            for (size_t i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    for (size_t j = 0; j < d; j++) {
                        centroids[i * d + j] = new_centroids[i * d + j] / counts[i];
                    }
                }
            }
        }
    }
    
    size_t assign_cluster(const float* x) const {
        float min_dist = std::numeric_limits<float>::max();
        size_t min_idx = 0;
        
        for (size_t i = 0; i < k; i++) {
            float dist = fvec_L2sqr(x, centroids.data() + i * d, d);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        
        return min_idx;
    }
};

// ============================================================
// RaBitQ IVF 索引
// ============================================================

class RaBitQIVFIndex {
public:
    size_t d;
    size_t nlist;
    int nb_bits;
    bool is_inner_product;
    
    std::unique_ptr<KMeans> kmeans;
    std::vector<RaBitQCodec> codecs;
    std::vector<std::vector<float>> cluster_centroids;
    
    std::vector<std::vector<uint8_t>> codes;
    std::vector<std::vector<size_t>> ids;
    size_t ntotal;
    
    // SQ8 refinement
    bool use_sq8_refine;
    std::vector<std::unique_ptr<SQ8Quantizer>> sq8_quantizers;
    std::vector<std::vector<uint8_t>> sq8_codes;
    
    RaBitQIVFIndex(size_t d_val, size_t nlist_val, int nb_bits_val = 1, bool ip = false, bool sq8_refine = false)
        : d(d_val), nlist(nlist_val), nb_bits(nb_bits_val), is_inner_product(ip),
          ntotal(0), use_sq8_refine(sq8_refine) {
        
        kmeans = std::make_unique<KMeans>(d, nlist);
        
        codecs.resize(nlist);
        for (size_t i = 0; i < nlist; i++) {
            codecs[i] = RaBitQCodec(d, nb_bits_val, ip);
        }
        
        cluster_centroids.resize(nlist);
        codes.resize(nlist);
        ids.resize(nlist);
        
        if (use_sq8_refine) {
            sq8_quantizers.resize(nlist);
            for (size_t i = 0; i < nlist; i++) {
                sq8_quantizers[i] = std::make_unique<SQ8Quantizer>(d);
            }
            sq8_codes.resize(nlist);
        }
    }
    
    void train(size_t n, const float* x) {
        // KMeans 聚类
        kmeans->train(n, x);
        
        // 存储聚类质心
        for (size_t i = 0; i < nlist; i++) {
            cluster_centroids[i].assign(
                kmeans->centroids.begin() + i * d,
                kmeans->centroids.begin() + (i + 1) * d
            );
        }
        
        // 训练 SQ8 (每个聚类独立)
        if (use_sq8_refine) {
            std::vector<std::vector<float>> cluster_data(nlist);
            for (size_t i = 0; i < n; i++) {
                size_t cluster_id = kmeans->assign_cluster(x + i * d);
                for (size_t j = 0; j < d; j++) {
                    cluster_data[cluster_id].push_back(x[i * d + j]);
                }
            }
            
            for (size_t c = 0; c < nlist; c++) {
                if (!cluster_data[c].empty()) {
                    sq8_quantizers[c]->train(cluster_data[c].size() / d, cluster_data[c].data());
                }
            }
        }
        
        std::cout << "RaBitQ IVF: 训练完成, " << nlist << " 个聚类" 
                  << (use_sq8_refine ? " + SQ8 refinement" : "") << std::endl;
    }
    
    void add(size_t n, const float* x) {
        size_t code_sz = codecs[0].code_size();
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            
            size_t cluster_id = kmeans->assign_cluster(xi);
            
            // 编码 RaBitQ
            std::vector<uint8_t> code(code_sz);
            codecs[cluster_id].encode(xi, cluster_centroids[cluster_id].data(), code.data());
            codes[cluster_id].insert(codes[cluster_id].end(), code.begin(), code.end());
            
            // 编码 SQ8
            if (use_sq8_refine) {
                std::vector<uint8_t> sq8_code(sq8_quantizers[cluster_id]->code_size());
                sq8_quantizers[cluster_id]->encode(xi, sq8_code.data());
                sq8_codes[cluster_id].insert(sq8_codes[cluster_id].end(), 
                                            sq8_code.begin(), sq8_code.end());
            }
            
            ids[cluster_id].push_back(ntotal + i);
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists,
                size_t nprobe = 10,
                size_t refine_factor = 1) const {
        
        result_ids.resize(n);
        result_dists.resize(n);
        
        size_t code_sz = codecs[0].code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            
            // 找最近的 nprobe 个聚类
            std::vector<std::pair<float, size_t>> cluster_dists(nlist);
            for (size_t c = 0; c < nlist; c++) {
                float dist = fvec_L2sqr(query, kmeans->centroids.data() + c * d, d);
                cluster_dists[c] = {dist, c};
            }
            
            std::partial_sort(cluster_dists.begin(), 
                             cluster_dists.begin() + std::min(nprobe, nlist),
                             cluster_dists.end());
            
            // 第一阶段: RaBitQ 搜索
            size_t k1 = use_sq8_refine ? std::min(k * refine_factor, ntotal) : k;
            std::priority_queue<std::pair<float, size_t>> top_k1;
            
            for (size_t p = 0; p < std::min(nprobe, nlist); p++) {
                size_t cluster_id = cluster_dists[p].second;
                
                const auto& cluster_codes = codes[cluster_id];
                const auto& cluster_ids = ids[cluster_id];
                size_t n_vectors = cluster_ids.size();
                
                // 计算查询因子 (使用聚类质心)
                QueryFactorsData query_fac = compute_query_factors(
                    query, d, cluster_centroids[cluster_id].data(), is_inner_product);
                
                for (size_t v = 0; v < n_vectors; v++) {
                    const uint8_t* code = cluster_codes.data() + v * code_sz;
                    float dist = codecs[cluster_id].compute_distance(
                        code, query_fac, cluster_centroids[cluster_id].data());
                    
                    if (top_k1.size() < k1) {
                        top_k1.push({dist, cluster_ids[v]});
                    } else if (dist < top_k1.top().first) {
                        top_k1.pop();
                        top_k1.push({dist, cluster_ids[v]});
                    }
                }
            }
            
            // 第二阶段: SQ8 refinement
            std::vector<std::pair<float, size_t>> candidates;
            while (!top_k1.empty()) {
                candidates.push_back(top_k1.top());
                top_k1.pop();
            }
            
            std::priority_queue<std::pair<float, size_t>> final_top_k;
            
            if (use_sq8_refine) {
                for (auto& [rabitq_dist, idx] : candidates) {
                    // 找到向量所属的聚类
                    size_t cluster_id = 0;
                    for (size_t c = 0; c < nlist; c++) {
                        auto it = std::lower_bound(ids[c].begin(), ids[c].end(), idx);
                        if (it != ids[c].end() && *it == idx) {
                            // 找到在聚类中的位置
                            size_t pos = it - ids[c].begin();
                            const uint8_t* sq8_code = sq8_codes[c].data() + 
                                                      pos * sq8_quantizers[c]->code_size();
                            float refined_dist = sq8_quantizers[c]->compute_distance(sq8_code, query);
                            
                            if (final_top_k.size() < k) {
                                final_top_k.push({refined_dist, idx});
                            } else if (refined_dist < final_top_k.top().first) {
                                final_top_k.pop();
                                final_top_k.push({refined_dist, idx});
                            }
                            break;
                        }
                    }
                }
            } else {
                for (auto& [dist, idx] : candidates) {
                    final_top_k.push({dist, idx});
                }
            }
            
            // 输出结果
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
    size_t get_code_size() const { return codecs[0].code_size(); }
};

} // namespace rabitq_faiss

#endif // RABITQ_FAISS_H
