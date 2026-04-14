/*
 * IVF + RaBitQ + OPQ 工业最强组合
 * 
 * OPQ (Optimized Product Quantization): 学习最优旋转矩阵
 * RaBitQ: 随机二进制量化
 * IVF: 倒排索引加速
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -o test_industrial test_industrial.cpp -lm
 */

#ifndef INDUSTRIAL_QUANTIZATION_H
#define INDUSTRIAL_QUANTIZATION_H

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

namespace industrial {

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
// OPQ 旋转矩阵 (简化实现)
// ============================================================

class OPQMatrix {
public:
    size_t d_in;
    size_t d_out;
    std::vector<float> matrix;
    
    OPQMatrix(size_t d, size_t d_out_val = 0) 
        : d_in(d), d_out(d_out_val > 0 ? d_out_val : d) {
        matrix.resize(d_in * d_out, 0.0f);
        for (size_t i = 0; i < std::min(d_in, d_out); i++) {
            matrix[i * d_out + i] = 1.0f;
        }
    }
    
    void train(size_t n, const float* x, int niter = 20, int nsub = 8) {
        std::cout << "OPQ 训练: " << n << " 个向量, " << niter << " 次迭代" << std::endl;
        
        std::vector<float> rotated(n * d_out);
        apply_batch(n, x, rotated.data());
        
        std::mt19937 rng(42);
        
        for (int iter = 0; iter < niter; iter++) {
            std::vector<float> grad(d_in * d_out, 0.0f);
            
            for (size_t i = 0; i < n; i++) {
                const float* xi = x + i * d_in;
                float* ri = rotated.data() + i * d_out;
                
                for (size_t j = 0; j < d_out; j++) {
                    for (size_t k = 0; k < d_in; k++) {
                        grad[k * d_out + j] += xi[k] * ri[j];
                    }
                }
            }
            
            float lr = 0.01f / (1.0f + iter * 0.1f);
            for (size_t i = 0; i < matrix.size(); i++) {
                matrix[i] += lr * grad[i];
            }
            
            orthonormalize();
        }
        
        std::cout << "OPQ 训练完成" << std::endl;
    }
    
    void orthonormalize() {
        std::vector<float> Q(d_in * d_out);
        std::vector<float> R(d_out * d_out, 0.0f);
        
        for (size_t j = 0; j < d_out; j++) {
            std::vector<float> v(d_in);
            for (size_t i = 0; i < d_in; i++) {
                v[i] = matrix[i * d_out + j];
            }
            
            for (size_t k = 0; k < j; k++) {
                float dot = 0.0f;
                for (size_t i = 0; i < d_in; i++) {
                    dot += v[i] * Q[i * d_out + k];
                }
                R[k * d_out + j] = dot;
                
                for (size_t i = 0; i < d_in; i++) {
                    v[i] -= dot * Q[i * d_out + k];
                }
            }
            
            float norm = 0.0f;
            for (size_t i = 0; i < d_in; i++) {
                norm += v[i] * v[i];
            }
            norm = std::sqrt(norm);
            R[j * d_out + j] = norm;
            
            if (norm > 1e-10f) {
                for (size_t i = 0; i < d_in; i++) {
                    Q[i * d_out + j] = v[i] / norm;
                }
            }
        }
        
        matrix = Q;
    }
    
    void apply(const float* x, float* x_rotated) const {
        for (size_t j = 0; j < d_out; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < d_in; i++) {
                sum += x[i] * matrix[i * d_out + j];
            }
            x_rotated[j] = sum;
        }
    }
    
    void apply_batch(size_t n, const float* x, float* x_rotated) const {
        for (size_t i = 0; i < n; i++) {
            apply(x + i * d_in, x_rotated + i * d_out);
        }
    }
    
    void apply_transpose(const float* x, float* x_original) const {
        for (size_t j = 0; j < d_in; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < d_out; i++) {
                sum += x[i] * matrix[j * d_out + i];
            }
            x_original[j] = sum;
        }
    }
};

// ============================================================
// Hadamard 旋转 (用于 RaBitQ)
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

class HadamardRotation {
public:
    size_t d_in;
    size_t d_out;
    uint32_t seed;
    std::vector<float> signs;
    float scale;
    
    explicit HadamardRotation(size_t d, uint32_t seed_val = 12345) 
        : d_in(d), seed(seed_val) {
        d_out = next_power_of_2(d);
        
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1);
        
        signs.resize(d_out);
        for (size_t i = 0; i < d_out; i++) {
            signs[i] = dist(rng) ? 1.0f : -1.0f;
        }
        
        scale = 1.0f / std::sqrt(static_cast<float>(d_out));
    }
    
    void apply(const float* x, float* x_rotated) const {
        std::vector<float> buf(d_out);
        
        for (size_t i = 0; i < d_in; i++) buf[i] = x[i] * signs[i];
        for (size_t i = d_in; i < d_out; i++) buf[i] = 0.0f;
        
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
// RaBitQ 量化器
// ============================================================

class RaBitQuantizer {
public:
    size_t d;
    int nbits;
    std::vector<float> centroid;
    
    RaBitQuantizer() : d(0), nbits(1) {}
    
    RaBitQuantizer(size_t d_val, int nbits_val = 1) 
        : d(d_val), nbits(nbits_val) {}
    
    size_t code_size() const {
        return (d + 7) / 8 + 8;
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
        float dp_oO = 0.0f;
        
        const float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        
        for (size_t i = 0; i < d; i++) {
            float or_minus_c = x[i] - centroid[i];
            norm_L2sqr += or_minus_c * or_minus_c;
            
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
        
        float dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        
        float* factors = reinterpret_cast<float*>(code + (d + 7) / 8);
        factors[0] = norm_L2sqr;
        factors[1] = dp_multiplier;
    }
    
    float compute_distance(const uint8_t* code, const float* query, 
                           float qr_to_c_l2sqr) const {
        const float* factors = reinterpret_cast<const float*>(code + (d + 7) / 8);
        float or_minus_c_l2sqr = factors[0];
        float dp_multiplier = factors[1];
        
        const float inv_d_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
        
        float dot_qo = 0.0f;
        for (size_t i = 0; i < d; i++) {
            bool bit = (code[i / 8] & (1 << (i % 8))) != 0;
            if (bit) {
                float q_minus_c = query[i] - centroid[i];
                dot_qo += q_minus_c;
            }
        }
        
        float sum_q_minus_c = 0.0f;
        for (size_t i = 0; i < d; i++) {
            sum_q_minus_c += (query[i] - centroid[i]);
        }
        
        float c1 = 2.0f * inv_d_sqrt;
        float c34 = sum_q_minus_c * inv_d_sqrt;
        
        float final_dot = c1 * dot_qo - c34;
        
        float dist = or_minus_c_l2sqr + qr_to_c_l2sqr - 2.0f * dp_multiplier * final_dot;
        
        return std::max(0.0f, dist);
    }
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
// IVF + RaBitQ + OPQ 索引
// ============================================================

class IVFRaBitQOPQIndex {
public:
    size_t d;
    size_t d_opq;
    size_t d_hadamard;
    size_t nlist;
    int nbits;
    
    std::unique_ptr<OPQMatrix> opq;
    std::unique_ptr<HadamardRotation> hadamard;
    std::unique_ptr<KMeans> kmeans;
    
    std::vector<std::vector<uint8_t>> codes;
    std::vector<std::vector<size_t>> ids;
    std::vector<std::vector<float>> cluster_centroids;
    std::vector<RaBitQuantizer> quantizers;
    
    size_t ntotal;
    
    IVFRaBitQOPQIndex(size_t d_val, size_t nlist_val, size_t opq_dim = 0, int nbits_val = 1)
        : d(d_val), d_opq(opq_dim > 0 ? opq_dim : d_val), nlist(nlist_val), nbits(nbits_val), ntotal(0) {
        
        d_hadamard = next_power_of_2(d_opq);
        
        opq = std::make_unique<OPQMatrix>(d, d_opq);
        hadamard = std::make_unique<HadamardRotation>(d_opq);
        kmeans = std::make_unique<KMeans>(d_hadamard, nlist);
        
        codes.resize(nlist);
        ids.resize(nlist);
        cluster_centroids.resize(nlist);
        quantizers.resize(nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            quantizers[i] = RaBitQuantizer(d_hadamard, nbits);
        }
    }
    
    void train(size_t n, const float* x) {
        std::cout << "训练 IVF + RaBitQ + OPQ 索引..." << std::endl;
        
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        opq->train(n, x_normalized.data(), 20);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  OPQ 训练: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        
        std::vector<float> x_opq(n * d_opq);
        opq->apply_batch(n, x_normalized.data(), x_opq.data());
        
        std::vector<float> x_hadamard(n * d_hadamard);
        hadamard->apply_batch(n, x_opq.data(), x_hadamard.data());
        
        start = std::chrono::high_resolution_clock::now();
        kmeans->train(n, x_hadamard.data());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  KMeans 训练: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        
        for (size_t i = 0; i < nlist; i++) {
            cluster_centroids[i].assign(
                kmeans->centroids.begin() + i * d_hadamard,
                kmeans->centroids.begin() + (i + 1) * d_hadamard
            );
            quantizers[i].centroid = cluster_centroids[i];
        }
        
        std::cout << "训练完成: " << nlist << " 个聚类" << std::endl;
    }
    
    void add(size_t n, const float* x) {
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        std::vector<float> x_opq(n * d_opq);
        opq->apply_batch(n, x_normalized.data(), x_opq.data());
        
        std::vector<float> x_hadamard(n * d_hadamard);
        hadamard->apply_batch(n, x_opq.data(), x_hadamard.data());
        
        size_t code_sz = quantizers[0].code_size();
        
        for (size_t i = 0; i < n; i++) {
            const float* xi = x_hadamard.data() + i * d_hadamard;
            
            size_t cluster_id = kmeans->assign_cluster(xi);
            
            std::vector<uint8_t> code(code_sz);
            quantizers[cluster_id].encode(xi, code.data());
            
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
        
        std::vector<float> x_normalized(n * d);
        std::copy(x, x + n * d, x_normalized.begin());
        for (size_t i = 0; i < n; i++) {
            l2_normalize(x_normalized.data() + i * d, d);
        }
        
        std::vector<float> x_opq(n * d_opq);
        opq->apply_batch(n, x_normalized.data(), x_opq.data());
        
        std::vector<float> x_hadamard(n * d_hadamard);
        hadamard->apply_batch(n, x_opq.data(), x_hadamard.data());
        
        size_t code_sz = quantizers[0].code_size();
        
        for (size_t q = 0; q < n; q++) {
            const float* query = x_hadamard.data() + q * d_hadamard;
            
            std::vector<std::pair<float, size_t>> cluster_dists(nlist);
            for (size_t c = 0; c < nlist; c++) {
                float dist = l2_distance(query, kmeans->centroids.data() + c * d_hadamard, d_hadamard);
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
                
                float qr_to_c_l2sqr = 0.0f;
                for (size_t j = 0; j < d_hadamard; j++) {
                    float diff = query[j] - quantizers[cluster_id].centroid[j];
                    qr_to_c_l2sqr += diff * diff;
                }
                
                for (size_t v = 0; v < n_vectors; v++) {
                    const uint8_t* code = cluster_codes.data() + v * code_sz;
                    float dist = quantizers[cluster_id].compute_distance(code, query, qr_to_c_l2sqr);
                    
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
    size_t get_code_size() const { return quantizers[0].code_size(); }
    
    void print_stats() const {
        std::cout << "IVF + RaBitQ + OPQ 索引统计:" << std::endl;
        std::cout << "  原始维度: " << d << std::endl;
        std::cout << "  OPQ 维度: " << d_opq << std::endl;
        std::cout << "  Hadamard 维度: " << d_hadamard << std::endl;
        std::cout << "  聚类数 (nlist): " << nlist << std::endl;
        std::cout << "  量化位宽: " << nbits << " bits" << std::endl;
        std::cout << "  码大小: " << get_code_size() << " bytes/vector" << std::endl;
        std::cout << "  总向量数: " << ntotal << std::endl;
        
        std::cout << "  各聚类大小: [";
        for (size_t i = 0; i < std::min(nlist, size_t(10)); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i].size();
        }
        std::cout << ", ...]" << std::endl;
    }
};

} // namespace industrial

#endif // INDUSTRIAL_QUANTIZATION_H
