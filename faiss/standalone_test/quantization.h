/*
 * TurboQuant Flat & RaBitQ 实现
 * 
 * 包含:
 * 1. TurboQuant Flat - 纯量化索引
 * 2. RaBitQ Flat - 随机二进制量化
 * 3. RaBitQ IVF - 倒排索引版本
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_quantization test_quantization.cpp -lm
 */

#ifndef QUANTIZATION_H
#define QUANTIZATION_H

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

namespace quant {

// ============================================================
// 工具函数
// ============================================================

inline float dot_product(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float l2_distance(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float l2_norm(const float* x, size_t d) {
    return std::sqrt(dot_product(x, x, d));
}

inline void l2_normalize(float* x, size_t d) {
    float norm = l2_norm(x, d);
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
// Fast Walsh-Hadamard Transform (FWHT)
// ============================================================

inline void fwht_inplace(float* buf, size_t n) {
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < n; i += step * 2) {
            for (size_t j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j] = a + b;
                buf[j + step] = a - b;
            }
        }
    }
}

// ============================================================
// Hadamard 旋转 (3轮 FWHT + 随机符号翻转)
// ============================================================

class HadamardRotation {
public:
    size_t d_in;
    size_t d_out;
    uint32_t seed;
    std::vector<float> signs1, signs2, signs3;
    float scale;
    
    explicit HadamardRotation(size_t d, uint32_t seed_val = 12345) 
        : d_in(d), seed(seed_val) {
        d_out = next_power_of_2(d);
        
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1);
        
        signs1.resize(d_out);
        signs2.resize(d_out);
        signs3.resize(d_out);
        
        for (size_t i = 0; i < d_out; i++) {
            signs1[i] = dist(rng) ? 1.0f : -1.0f;
            signs2[i] = dist(rng) ? 1.0f : -1.0f;
            signs3[i] = dist(rng) ? 1.0f : -1.0f;
        }
        
        scale = 1.0f / (d_out * std::sqrt(static_cast<float>(d_out)));
    }
    
    void apply(const float* x, float* x_rotated) const {
        std::vector<float> buf(d_out);
        
        for (size_t i = 0; i < d_in; i++) buf[i] = x[i] * signs1[i];
        for (size_t i = d_in; i < d_out; i++) buf[i] = 0.0f;
        
        fwht_inplace(buf.data(), d_out);
        
        for (size_t i = 0; i < d_out; i++) buf[i] *= signs2[i];
        fwht_inplace(buf.data(), d_out);
        
        for (size_t i = 0; i < d_out; i++) buf[i] *= signs3[i];
        fwht_inplace(buf.data(), d_out);
        
        for (size_t i = 0; i < d_out; i++) {
            x_rotated[i] = buf[i] * scale;
        }
    }
    
    void apply_batch(size_t n, const float* x, float* x_rotated) const {
        for (size_t i = 0; i < n; i++) {
            apply(x + i * d_in, x_rotated + i * d_out);
        }
    }
};

// ============================================================
// TurboQuant MSE 量化器 (Beta 分布)
// ============================================================

class TurboQuantMSE {
public:
    size_t d;
    int nbits;
    size_t k;
    std::vector<float> centroids;
    std::vector<float> boundaries;
    
    TurboQuantMSE(size_t d_val, int nbits_val) : d(d_val), nbits(nbits_val) {
        k = size_t(1) << nbits;
        build_codebook();
    }
    
    void build_codebook() {
        centroids.resize(k);
        boundaries.resize(k - 1);
        
        if (d == 1) {
            for (size_t i = 0; i < k; i++) {
                centroids[i] = (i < k / 2) ? -1.0f : 1.0f;
            }
        } else {
            lloyd_max_iteration();
        }
        
        for (size_t i = 0; i < k - 1; i++) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
    }
    
    void lloyd_max_iteration() {
        const int ngrid = 32768;
        const double step = 2.0 / ngrid;
        const double alpha = 0.5 * (static_cast<double>(d) - 3.0);
        
        std::vector<double> xs(ngrid);
        std::vector<double> prefix_w(ngrid + 1, 0.0);
        std::vector<double> prefix_wx(ngrid + 1, 0.0);
        
        for (size_t i = 0; i < ngrid; i++) {
            double x = -1.0 + (i + 0.5) * step;
            double one_minus_x2 = std::max(0.0, 1.0 - x * x);
            double w = (alpha == 0.0) ? 1.0 : std::pow(one_minus_x2, alpha);
            if (!std::isfinite(w) || w < 0.0) w = 0.0;
            
            xs[i] = x;
            prefix_w[i + 1] = prefix_w[i] + w;
            prefix_wx[i + 1] = prefix_wx[i] + w * x;
        }
        
        auto range_mean = [&](size_t i0, size_t i1, double fallback) {
            double w = prefix_w[i1] - prefix_w[i0];
            if (w <= 0.0) return fallback;
            return (prefix_wx[i1] - prefix_wx[i0]) / w;
        };
        
        std::vector<size_t> cuts(k + 1, 0);
        cuts[k] = ngrid;
        double total_w = prefix_w.back();
        
        for (size_t i = 1; i < k; i++) {
            double target = total_w * i / k;
            cuts[i] = std::lower_bound(prefix_w.begin(), prefix_w.end(), target) - prefix_w.begin();
            cuts[i] = std::min(cuts[i], static_cast<size_t>(ngrid));
        }
        
        std::vector<double> centroids_d(k);
        for (size_t i = 0; i < k; i++) {
            double left = -1.0 + 2.0 * i / k;
            double right = -1.0 + 2.0 * (i + 1) / k;
            centroids_d[i] = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
        }
        
        std::vector<double> boundaries_d(k - 1);
        for (int iter = 0; iter < 100; iter++) {
            for (size_t i = 0; i < k - 1; i++) {
                boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
            }
            
            cuts[0] = 0;
            cuts[k] = ngrid;
            for (size_t i = 1; i < k; i++) {
                cuts[i] = std::upper_bound(xs.begin(), xs.end(), boundaries_d[i - 1]) - xs.begin();
            }
            
            double max_delta = 0.0;
            for (size_t i = 0; i < k; i++) {
                double left = (i == 0) ? -1.0 : boundaries_d[i - 1];
                double right = (i + 1 == k) ? 1.0 : boundaries_d[i];
                double c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
                c = std::min(std::max(c, left), right);
                max_delta = std::max(max_delta, std::abs(c - centroids_d[i]));
                centroids_d[i] = c;
            }
            
            if (max_delta < 1e-8) break;
        }
        
        std::sort(centroids_d.begin(), centroids_d.end());
        for (size_t i = 0; i < k; i++) {
            centroids[i] = static_cast<float>(centroids_d[i]);
        }
    }
    
    size_t code_size() const {
        return (d * nbits + 7) / 8;
    }
    
    uint8_t select_index(float x) const {
        return static_cast<uint8_t>(
            std::upper_bound(boundaries.begin(), boundaries.end(), x) - boundaries.begin());
    }
    
    void encode_index(uint8_t idx, uint8_t* code, size_t i) const {
        size_t bit_offset = i * nbits;
        size_t byte_offset = bit_offset >> 3;
        size_t bit_shift = bit_offset & 7;
        uint16_t mask = (1 << nbits) - 1;
        uint16_t packed = static_cast<uint16_t>(idx & mask) << bit_shift;
        code[byte_offset] |= packed & 0xff;
        if (bit_shift + nbits > 8) {
            code[byte_offset + 1] |= packed >> 8;
        }
    }
    
    uint8_t decode_index(const uint8_t* code, size_t i) const {
        size_t bit_offset = i * nbits;
        size_t byte_offset = bit_offset >> 3;
        size_t bit_shift = bit_offset & 7;
        uint16_t mask = (1 << nbits) - 1;
        
        uint16_t packed = code[byte_offset];
        if (bit_shift + nbits > 8) {
            packed |= static_cast<uint16_t>(code[byte_offset + 1]) << 8;
        }
        return static_cast<uint8_t>((packed >> bit_shift) & mask);
    }
    
    void encode(const float* x, uint8_t* code) const {
        std::fill(code, code + code_size(), 0);
        for (size_t i = 0; i < d; i++) {
            uint8_t idx = select_index(x[i]);
            encode_index(idx, code, i);
        }
    }
    
    void decode(const uint8_t* code, float* x) const {
        for (size_t i = 0; i < d; i++) {
            uint8_t idx = decode_index(code, i);
            x[i] = centroids[idx];
        }
    }
    
    float compute_distance(const uint8_t* code, const float* query) const {
        float dist = 0.0f;
        for (size_t i = 0; i < d; i++) {
            uint8_t idx = decode_index(code, i);
            float diff = centroids[idx] - query[i];
            dist += diff * diff;
        }
        return dist;
    }
};

// ============================================================
// TurboQuant Flat 索引
// ============================================================

class TurboQuantFlatIndex {
public:
    size_t d;
    int nbits;
    
    std::unique_ptr<HadamardRotation> rotation;
    std::unique_ptr<TurboQuantMSE> quantizer;
    
    std::vector<uint8_t> codes;
    size_t ntotal;
    
    TurboQuantFlatIndex(size_t d_val, int nbits_val = 4)
        : d(d_val), nbits(nbits_val), ntotal(0) {
        
        size_t d_rotated = next_power_of_2(d);
        rotation = std::make_unique<HadamardRotation>(d);
        quantizer = std::make_unique<TurboQuantMSE>(d_rotated, nbits);
    }
    
    void train(size_t n, const float* x) {
        // TurboQuant 不需要训练数据
        std::cout << "TurboQuant Flat: 无需训练" << std::endl;
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
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x_rotated.data() + i * d_rotated;
            quantizer->encode(xi, codes.data() + (ntotal + i) * code_sz);
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists) const {
        
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
            
            std::priority_queue<std::pair<float, size_t>> top_k;
            
            for (size_t i = 0; i < ntotal; i++) {
                const uint8_t* code = codes.data() + i * code_sz;
                float dist = quantizer->compute_distance(code, query);
                
                if (top_k.size() < k) {
                    top_k.push({dist, i});
                } else if (dist < top_k.top().first) {
                    top_k.pop();
                    top_k.push({dist, i});
                }
            }
            
            result_ids[q].resize(k);
            result_dists[q].resize(k);
            
            for (size_t i = k; i > 0; i--) {
                if (!top_k.empty()) {
                    result_dists[q][i - 1] = top_k.top().first;
                    result_ids[q][i - 1] = top_k.top().second;
                    top_k.pop();
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
// RaBitQ 量化器
// ============================================================

class RaBitQuantizer {
public:
    size_t d;
    int nbits;
    
    std::vector<float> centroid;
    
    RaBitQuantizer(size_t d_val, int nbits_val = 1) 
        : d(d_val), nbits(nbits_val) {}
    
    size_t code_size() const {
        size_t ex_bits = nbits - 1;
        size_t base_size = (d + 7) / 8 + 8;
        size_t ex_size = ex_bits > 0 ? (d * ex_bits + 7) / 8 + 8 : 0;
        return base_size + ex_size;
    }
    
    void train(size_t n, const float* x) {
        centroid.assign(d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                centroid[j] += x[i * d + j];
            }
        }
        if (n > 0) {
            for (size_t j = 0; j < d; j++) {
                centroid[j] /= n;
            }
        }
    }
    
    void encode(const float* x, uint8_t* code) const {
        memset(code, 0, code_size());
        
        float norm_L2sqr = 0.0f;
        float or_L2sqr = 0.0f;
        float dp_oO = 0.0f;
        
        const float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        
        for (size_t i = 0; i < d; i++) {
            float or_minus_c = x[i] - centroid[i];
            norm_L2sqr += or_minus_c * or_minus_c;
            or_L2sqr += x[i] * x[i];
            
            if (or_minus_c > 0) {
                code[i / 8] |= (1 << (i % 8));
                dp_oO += or_minus_c;
            } else {
                dp_oO -= or_minus_c;
            }
        }
        
        float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
        float inv_norm_L2 = (norm_L2sqr < 1e-10f) ? 1.0f : (1.0f / sqrt_norm_L2);
        
        float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
        float inv_dp_oO = (std::abs(normalized_dp) < 1e-10f) ? 1.0f : (1.0f / normalized_dp);
        
        float or_minus_c_l2sqr = norm_L2sqr;
        float dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        
        float* factors = reinterpret_cast<float*>(code + (d + 7) / 8);
        factors[0] = or_minus_c_l2sqr;
        factors[1] = dp_multiplier;
    }
    
    void decode(const uint8_t* code, float* x) const {
        const float* factors = reinterpret_cast<const float*>(code + (d + 7) / 8);
        float or_minus_c_l2sqr = factors[0];
        float dp_multiplier = factors[1];
        
        float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        float sqrt_norm = std::sqrt(or_minus_c_l2sqr);
        
        for (size_t i = 0; i < d; i++) {
            bool bit = (code[i / 8] & (1 << (i % 8))) != 0;
            float sign = bit ? 1.0f : -1.0f;
            x[i] = sign * sqrt_norm * inv_d_sqrt + centroid[i];
        }
    }
    
    float compute_distance(const uint8_t* code, const float* query_rotated, 
                           float qr_to_c_l2sqr) const {
        const float* factors = reinterpret_cast<const float*>(code + (d + 7) / 8);
        float or_minus_c_l2sqr = factors[0];
        float dp_multiplier = factors[1];
        
        const float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        
        float dot_qo = 0.0f;
        uint64_t sum_bits = 0;
        
        for (size_t i = 0; i < d; i++) {
            bool bit = (code[i / 8] & (1 << (i % 8))) != 0;
            if (bit) {
                float q_minus_c = query_rotated[i] - centroid[i];
                dot_qo += q_minus_c;
                sum_bits++;
            }
        }
        
        float sum_q_minus_c = 0.0f;
        for (size_t i = 0; i < d; i++) {
            sum_q_minus_c += (query_rotated[i] - centroid[i]);
        }
        
        float c1 = 2.0f * inv_d_sqrt;
        float c34 = sum_q_minus_c * inv_d_sqrt;
        
        float final_dot = c1 * dot_qo - c34;
        
        float dist = or_minus_c_l2sqr + qr_to_c_l2sqr - 2.0f * dp_multiplier * final_dot;
        
        return std::max(0.0f, dist);
    }
};

// ============================================================
// RaBitQ Flat 索引
// ============================================================

class RaBitQFlatIndex {
public:
    size_t d;
    int nbits;
    
    std::unique_ptr<HadamardRotation> rotation;
    std::unique_ptr<RaBitQuantizer> quantizer;
    
    std::vector<uint8_t> codes;
    size_t ntotal;
    
    RaBitQFlatIndex(size_t d_val, int nbits_val = 1)
        : d(d_val), nbits(nbits_val), ntotal(0) {
        
        rotation = std::make_unique<HadamardRotation>(d);
        quantizer = std::make_unique<RaBitQuantizer>(d, nbits);
    }
    
    void train(size_t n, const float* x) {
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        quantizer->train(n, x_rotated.data());
        std::cout << "RaBitQ Flat: 训练完成 (计算质心)" << std::endl;
    }
    
    void add(size_t n, const float* x) {
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        codes.resize((ntotal + n) * code_sz);
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x_rotated.data() + i * d_rotated;
            quantizer->encode(xi, codes.data() + (ntotal + i) * code_sz);
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists) const {
        
        result_ids.resize(n);
        result_dists.resize(n);
        
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x_rotated.data() + q * d_rotated;
            
            float qr_to_c_l2sqr = 0.0f;
            for (size_t j = 0; j < d_rotated; j++) {
                float diff = query[j] - quantizer->centroid[j];
                qr_to_c_l2sqr += diff * diff;
            }
            
            std::priority_queue<std::pair<float, size_t>> top_k;
            
            for (size_t i = 0; i < ntotal; i++) {
                const uint8_t* code = codes.data() + i * code_sz;
                float dist = quantizer->compute_distance(code, query, qr_to_c_l2sqr);
                
                if (top_k.size() < k) {
                    top_k.push({dist, i});
                } else if (dist < top_k.top().first) {
                    top_k.pop();
                    top_k.push({dist, i});
                }
            }
            
            result_ids[q].resize(k);
            result_dists[q].resize(k);
            
            for (size_t i = k; i > 0; i--) {
                if (!top_k.empty()) {
                    result_dists[q][i - 1] = top_k.top().first;
                    result_ids[q][i - 1] = top_k.top().second;
                    top_k.pop();
                }
            }
        }
    }
    
    size_t get_ntotal() const { return ntotal; }
    size_t get_code_size() const { return quantizer->code_size(); }
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
            
            for (size_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                
                float min_dist = std::numeric_limits<float>::max();
                size_t min_idx = 0;
                
                for (size_t j = 0; j < k; j++) {
                    float dist = l2_distance(xi, centroids.data() + j * d, d);
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
            float dist = l2_distance(x, centroids.data() + i * d, d);
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
    int nbits;
    
    std::unique_ptr<HadamardRotation> rotation;
    std::unique_ptr<KMeans> kmeans;
    std::unique_ptr<RaBitQuantizer> quantizer;
    
    std::vector<std::vector<uint8_t>> codes;
    std::vector<std::vector<size_t>> ids;
    std::vector<std::vector<float>> cluster_centroids;
    size_t ntotal;
    
    RaBitQIVFIndex(size_t d_val, size_t nlist_val, int nbits_val = 1)
        : d(d_val), nlist(nlist_val), nbits(nbits_val), ntotal(0) {
        
        rotation = std::make_unique<HadamardRotation>(d);
        kmeans = std::make_unique<KMeans>(rotation->d_out, nlist);
        quantizer = std::make_unique<RaBitQuantizer>(rotation->d_out, nbits);
        
        codes.resize(nlist);
        ids.resize(nlist);
        cluster_centroids.resize(nlist);
    }
    
    void train(size_t n, const float* x) {
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        kmeans->train(n, x_rotated.data());
        
        for (size_t i = 0; i < nlist; i++) {
            cluster_centroids[i].assign(
                kmeans->centroids.begin() + i * d_rotated,
                kmeans->centroids.begin() + (i + 1) * d_rotated
            );
        }
        
        std::cout << "RaBitQ IVF: 训练完成, " << nlist << " 个聚类" << std::endl;
    }
    
    void add(size_t n, const float* x) {
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x_rotated.data() + i * d_rotated;
            
            size_t cluster_id = kmeans->assign_cluster(xi);
            
            std::vector<uint8_t> code(code_sz);
            
            RaBitQuantizer local_q(d_rotated, nbits);
            local_q.centroid = cluster_centroids[cluster_id];
            local_q.encode(xi, code.data());
            
            codes[cluster_id].insert(codes[cluster_id].end(), 
                                     code.begin(), code.end());
            ids[cluster_id].push_back(ntotal + i);
        }
        
        ntotal += n;
    }
    
    void search(size_t n, const float* x, size_t k, 
                std::vector<std::vector<size_t>>& result_ids,
                std::vector<std::vector<float>>& result_dists,
                size_t nprobe = 10) const {
        
        result_ids.resize(n);
        result_dists.resize(n);
        
        size_t d_rotated = rotation->d_out;
        std::vector<float> x_rotated(n * d_rotated);
        rotation->apply_batch(n, x, x_rotated.data());
        
        size_t code_sz = quantizer->code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x_rotated.data() + q * d_rotated;
            
            std::vector<std::pair<float, size_t>> cluster_dists(nlist);
            for (size_t c = 0; c < nlist; c++) {
                float dist = l2_distance(query, kmeans->centroids.data() + c * d_rotated, d_rotated);
                cluster_dists[c] = {dist, c};
            }
            
            std::partial_sort(cluster_dists.begin(), 
                             cluster_dists.begin() + std::min(nprobe, nlist),
                             cluster_dists.end());
            
            std::priority_queue<std::pair<float, size_t>> top_k;
            
            for (size_t p = 0; p < std::min(nprobe, nlist); p++) {
                size_t cluster_id = cluster_dists[p].second;
                
                const auto& cluster_codes = codes[cluster_id];
                const auto& cluster_ids = ids[cluster_id];
                
                size_t n_vectors = cluster_ids.size();
                
                RaBitQuantizer local_q(d_rotated, nbits);
                local_q.centroid = const_cast<std::vector<float>&>(cluster_centroids[cluster_id]);
                
                float qr_to_c_l2sqr = 0.0f;
                for (size_t j = 0; j < d_rotated; j++) {
                    float diff = query[j] - local_q.centroid[j];
                    qr_to_c_l2sqr += diff * diff;
                }
                
                for (size_t v = 0; v < n_vectors; v++) {
                    const uint8_t* code = cluster_codes.data() + v * code_sz;
                    float dist = local_q.compute_distance(code, query, qr_to_c_l2sqr);
                    
                    if (top_k.size() < k) {
                        top_k.push({dist, cluster_ids[v]});
                    } else if (dist < top_k.top().first) {
                        top_k.pop();
                        top_k.push({dist, cluster_ids[v]});
                    }
                }
            }
            
            result_ids[q].resize(k);
            result_dists[q].resize(k);
            
            for (size_t i = k; i > 0; i--) {
                if (!top_k.empty()) {
                    result_dists[q][i - 1] = top_k.top().first;
                    result_ids[q][i - 1] = top_k.top().second;
                    top_k.pop();
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

} // namespace quant

#endif // QUANTIZATION_H
