/*
 * TurboQuant 完整流程测试
 * 
 * 演示: 输入向量 -> L2归一化 -> Hadamard旋转 -> Beta分布量化 -> 反量化 -> 反旋转 -> 输出向量
 * 
 * 编译命令:
 *   g++ -std=c++17 -O3 -I/path/to/faiss -o test_turboquant_pipeline test_turboquant_pipeline.cpp -lfaiss -lm
 * 
 * 或使用 FAISS 源码编译:
 *   cd faiss/build && cmake .. && make
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

// ============================================================
// 1. L2 归一化
// ============================================================
void l2_normalize(float* x, size_t d) {
    float norm = 0.0f;
    for (size_t i = 0; i < d; i++) {
        norm += x[i] * x[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < d; i++) {
            x[i] /= norm;
        }
    }
}

// ============================================================
// 2. Fast Walsh-Hadamard Transform (FWHT)
// ============================================================
void fwht_inplace(float* buf, size_t n) {
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
// 3. Hadamard 旋转 (三轮随机符号翻转 + FWHT)
// ============================================================
class HadamardRotation {
public:
    size_t d;
    uint32_t seed;
    std::vector<float> signs1, signs2, signs3;
    float scale;
    
    explicit HadamardRotation(size_t d, uint32_t seed = 12345) 
        : d(d), seed(seed) {
        // 找到 >= d 的最小 2 的幂
        size_t p = 1;
        while (p < d) p *= 2;
        
        // 生成三轮随机符号
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1);
        
        signs1.resize(p);
        signs2.resize(p);
        signs3.resize(p);
        
        for (size_t i = 0; i < p; i++) {
            signs1[i] = dist(rng) ? 1.0f : -1.0f;
            signs2[i] = dist(rng) ? 1.0f : -1.0f;
            signs3[i] = dist(rng) ? 1.0f : -1.0f;
        }
        
        // 归一化因子: 1 / (p * sqrt(p))
        scale = 1.0f / (p * std::sqrt(static_cast<float>(p)));
    }
    
    // 正向旋转: x -> x'
    void apply(const float* x, float* x_rotated) const {
        size_t p = signs1.size();
        
        // 复制并填充到 p 维
        std::vector<float> buf(p);
        for (size_t i = 0; i < d; i++) buf[i] = x[i];
        for (size_t i = d; i < p; i++) buf[i] = 0.0f;
        
        // Round 1: D1 * FWHT
        for (size_t i = 0; i < p; i++) buf[i] *= signs1[i];
        fwht_inplace(buf.data(), p);
        
        // Round 2: D2 * FWHT
        for (size_t i = 0; i < p; i++) buf[i] *= signs2[i];
        fwht_inplace(buf.data(), p);
        
        // Round 3: D3 * FWHT + Scale
        for (size_t i = 0; i < p; i++) buf[i] *= signs3[i];
        fwht_inplace(buf.data(), p);
        for (size_t i = 0; i < p; i++) buf[i] *= scale;
        
        // 复制结果
        for (size_t i = 0; i < d; i++) x_rotated[i] = buf[i];
    }
    
    // 反向旋转: x' -> x
    void reverse(const float* x_rotated, float* x) const {
        size_t p = signs1.size();
        
        // 复制并填充到 p 维
        std::vector<float> buf(p);
        for (size_t i = 0; i < d; i++) buf[i] = x_rotated[i];
        for (size_t i = d; i < p; i++) buf[i] = 0.0f;
        
        // FAISS 正向变换:
        // y = scale * FWHT * D₃ * FWHT * D₂ * FWHT * D₁ * x
        //
        // 反向变换:
        // x = D₁ * FWHT * D₂ * FWHT * D₃ * FWHT * y * scale
        // 
        // 验证: H⁻¹ * H = D₁ * FWHT * D₂ * FWHT * D₃ * FWHT * scale * scale * FWHT * D₃ * FWHT * D₂ * FWHT * D₁
        //              = D₁ * FWHT * D₂ * FWHT * D₃ * FWHT * FWHT * D₃ * FWHT * D₂ * FWHT * D₁ * scale²
        //              = D₁ * FWHT * D₂ * FWHT * D₃ * (p*I) * D₃ * FWHT * D₂ * FWHT * D₁ * scale²
        //              = p * D₁ * FWHT * D₂ * FWHT * D₃ * D₃ * FWHT * D₂ * FWHT * D₁ * scale²
        //              = p * D₁ * FWHT * D₂ * FWHT * I * FWHT * D₂ * FWHT * D₁ * scale²
        //              = p² * D₁ * FWHT * D₂ * FWHT * FWHT * D₂ * FWHT * D₁ * scale²
        //              = p² * D₁ * FWHT * D₂ * (p*I) * D₂ * FWHT * D₁ * scale²
        //              = p³ * D₁ * FWHT * FWHT * D₁ * scale²
        //              = p⁴ * I * scale²
        //              = I  (因为 scale = 1/(p√p), scale² = 1/p³)
        
        // Round 3 reverse: FWHT * scale
        fwht_inplace(buf.data(), p);
        for (size_t i = 0; i < p; i++) buf[i] *= scale;
        
        // Round 2 reverse: D3
        for (size_t i = 0; i < p; i++) buf[i] *= signs3[i];
        
        // Round 1 reverse: FWHT
        fwht_inplace(buf.data(), p);
        
        // Round 0 reverse: D2
        for (size_t i = 0; i < p; i++) buf[i] *= signs2[i];
        
        // Round -1 reverse: FWHT
        fwht_inplace(buf.data(), p);
        
        // Final: D1
        for (size_t i = 0; i < p; i++) buf[i] *= signs1[i];
        
        // 复制结果
        for (size_t i = 0; i < d; i++) x[i] = buf[i];
    }
};

// ============================================================
// 4. Beta 分布量化器
// ============================================================
class TurboQuantMSE {
public:
    size_t d;
    int nbits;
    size_t k;  // 2^nbits
    std::vector<float> centroids;
    std::vector<float> boundaries;
    
    TurboQuantMSE(size_t d, int nbits) : d(d), nbits(nbits) {
        k = size_t(1) << nbits;
        build_codebook();
    }
    
    // 构建 Beta 分布码本
    void build_codebook() {
        centroids.resize(k);
        boundaries.resize(k - 1);
        
        if (d == 1) {
            // 1-D 特殊情况
            for (size_t i = 0; i < k; i++) {
                centroids[i] = (i < k / 2) ? -1.0f : 1.0f;
            }
        } else {
            // Beta 分布: p(x) ∝ (1 - x²)^((d-3)/2)
            // 使用 Lloyd-Max 迭代
            lloyd_max_iteration();
        }
        
        // 计算边界
        for (size_t i = 0; i < k - 1; i++) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
    }
    
    // Lloyd-Max 迭代
    void lloyd_max_iteration() {
        const int ngrid = 32768;
        const double step = 2.0 / ngrid;
        const double alpha = 0.5 * (static_cast<double>(d) - 3.0);
        
        // 离散化 Beta 分布
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
        
        // 初始化等质量划分
        std::vector<size_t> cuts(k + 1, 0);
        cuts[k] = ngrid;
        double total_w = prefix_w.back();
        
        for (size_t i = 1; i < k; i++) {
            double target = total_w * i / k;
            cuts[i] = std::lower_bound(prefix_w.begin(), prefix_w.end(), target) - prefix_w.begin();
            cuts[i] = std::min(cuts[i], static_cast<size_t>(ngrid));
        }
        
        // 初始质心
        std::vector<double> centroids_d(k);
        for (size_t i = 0; i < k; i++) {
            double left = -1.0 + 2.0 * i / k;
            double right = -1.0 + 2.0 * (i + 1) / k;
            centroids_d[i] = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
        }
        
        // Lloyd-Max 迭代
        std::vector<double> boundaries_d(k - 1);
        for (int iter = 0; iter < 100; iter++) {
            // 更新边界
            for (size_t i = 0; i < k - 1; i++) {
                boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
            }
            
            // 更新划分
            cuts[0] = 0;
            cuts[k] = ngrid;
            for (size_t i = 1; i < k; i++) {
                cuts[i] = std::upper_bound(xs.begin(), xs.end(), boundaries_d[i - 1]) - xs.begin();
            }
            
            // 更新质心
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
        
        // 排序并复制
        std::sort(centroids_d.begin(), centroids_d.end());
        for (size_t i = 0; i < k; i++) {
            centroids[i] = static_cast<float>(centroids_d[i]);
        }
    }
    
    // 量化: x -> code
    void encode(const float* x, uint8_t* code) const {
        for (size_t i = 0; i < d; i++) {
            uint8_t idx = select_index(x[i]);
            encode_index(idx, code, i);
        }
    }
    
    // 反量化: code -> x
    void decode(const uint8_t* code, float* x) const {
        for (size_t i = 0; i < d; i++) {
            uint8_t idx = decode_index(code, i);
            x[i] = centroids[idx];
        }
    }
    
private:
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
};

// ============================================================
// 5. 完整流程测试
// ============================================================
void test_full_pipeline(size_t d, int nbits, size_t n_vectors) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TurboQuant 完整流程测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "维度 d = " << d << std::endl;
    std::cout << "位宽 nbits = " << nbits << std::endl;
    std::cout << "向量数 n = " << n_vectors << std::endl;
    std::cout << std::endl;
    
    // 初始化组件
    HadamardRotation hr(d);
    TurboQuantMSE tq(d, nbits);
    
    // 计算码本大小
    size_t code_size = (d * nbits + 7) / 8;
    std::cout << "码本大小: " << code_size << " bytes/vector" << std::endl;
    std::cout << "压缩比: " << std::fixed << std::setprecision(2) 
              << (100.0 * code_size / (d * sizeof(float))) << "%" << std::endl;
    std::cout << std::endl;
    
    // 生成随机向量
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> original(n_vectors * d);
    for (size_t i = 0; i < n_vectors * d; i++) {
        original[i] = dist(rng);
    }
    
    // 归一化
    for (size_t i = 0; i < n_vectors; i++) {
        l2_normalize(&original[i * d], d);
    }
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    
    // 处理每个向量
    std::vector<float> reconstructed(n_vectors * d);
    std::vector<uint8_t> codes(n_vectors * code_size, 0);
    
    for (size_t i = 0; i < n_vectors; i++) {
        float* x_in = &original[i * d];
        float* x_out = &reconstructed[i * d];
        uint8_t* code = &codes[i * code_size];
        
        // 中间缓冲区
        std::vector<float> rotated(d);
        std::vector<float> quantized(d);
        
        // Step 1: Hadamard 旋转
        hr.apply(x_in, rotated.data());
        
        // Step 2: 量化
        tq.encode(rotated.data(), code);
        
        // Step 3: 反量化
        tq.decode(code, quantized.data());
        
        // Step 4: 反旋转
        hr.reverse(quantized.data(), x_out);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 计算重建误差
    double mse = 0.0;
    for (size_t i = 0; i < n_vectors * d; i++) {
        double diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    mse /= (n_vectors * d);
    
    // 计算余弦相似度
    double avg_cosine = 0.0;
    for (size_t i = 0; i < n_vectors; i++) {
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (size_t j = 0; j < d; j++) {
            float v1 = original[i * d + j];
            float v2 = reconstructed[i * d + j];
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        avg_cosine += dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    avg_cosine /= n_vectors;
    
    // 打印结果
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "结果:" << std::endl;
    std::cout << "  MSE (均方误差): " << std::scientific << std::setprecision(6) << mse << std::endl;
    std::cout << "  RMSE (均方根误差): " << std::scientific << std::setprecision(6) << std::sqrt(mse) << std::endl;
    std::cout << "  平均余弦相似度: " << std::fixed << std::setprecision(4) << avg_cosine << std::endl;
    std::cout << "  处理时间: " << duration.count() << " μs" << std::endl;
    std::cout << "  每向量时间: " << (duration.count() / n_vectors) << " μs" << std::endl;
    std::cout << std::endl;
    
    // 打印第一个向量的详细信息
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "第一个向量详情:" << std::endl;
    std::cout << "  原始向量 (前10维): [";
    for (size_t i = 0; i < std::min(size_t(10), d); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << original[i];
    }
    std::cout << ", ...]" << std::endl;
    
    std::cout << "  重建向量 (前10维): [";
    for (size_t i = 0; i < std::min(size_t(10), d); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << reconstructed[i];
    }
    std::cout << ", ...]" << std::endl;
    
    std::cout << "  压缩码 (前10字节): [";
    for (size_t i = 0; i < std::min(size_t(10), code_size); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(codes[i]);
    }
    std::cout << std::dec << ", ...]" << std::endl;
}

// ============================================================
// 6. 仅量化测试 (无旋转)
// ============================================================
void test_quantization_only(size_t d, int nbits, size_t n_vectors) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TurboQuant 仅量化测试 (无旋转)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "维度 d = " << d << std::endl;
    std::cout << "位宽 nbits = " << nbits << std::endl;
    std::cout << "向量数 n = " << n_vectors << std::endl;
    std::cout << std::endl;
    
    // 初始化量化器
    TurboQuantMSE tq(d, nbits);
    
    size_t code_size = (d * nbits + 7) / 8;
    
    // 生成随机向量
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> original(n_vectors * d);
    for (size_t i = 0; i < n_vectors * d; i++) {
        original[i] = dist(rng);
    }
    
    // 归一化
    for (size_t i = 0; i < n_vectors; i++) {
        l2_normalize(&original[i * d], d);
    }
    
    // 处理
    std::vector<float> reconstructed(n_vectors * d);
    std::vector<uint8_t> codes(n_vectors * code_size, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < n_vectors; i++) {
        tq.encode(&original[i * d], &codes[i * code_size]);
        tq.decode(&codes[i * code_size], &reconstructed[i * d]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 计算误差
    double mse = 0.0;
    for (size_t i = 0; i < n_vectors * d; i++) {
        double diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    mse /= (n_vectors * d);
    
    std::cout << "MSE: " << std::scientific << std::setprecision(6) << mse << std::endl;
    std::cout << "RMSE: " << std::scientific << std::setprecision(6) << std::sqrt(mse) << std::endl;
    std::cout << "处理时间: " << duration.count() << " μs" << std::endl;
}

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         TurboQuant 完整流程 C++ 测试                         ║" << std::endl;
    std::cout << "║                                                              ║" << std::endl;
    std::cout << "║  流程: 输入 -> L2归一化 -> Hadamard旋转 -> Beta量化          ║" << std::endl;
    std::cout << "║        -> 反量化 -> 反旋转 -> 输出                            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    // 测试不同配置
    test_full_pipeline(128, 4, 1000);   // 128维, 4-bit
    test_full_pipeline(256, 4, 1000);   // 256维, 4-bit
    test_full_pipeline(512, 4, 1000);   // 512维, 4-bit
    
    // 测试不同位宽
    test_full_pipeline(128, 1, 1000);   // 1-bit
    test_full_pipeline(128, 2, 1000);   // 2-bit
    test_full_pipeline(128, 3, 1000);   // 3-bit
    test_full_pipeline(128, 8, 1000);   // 8-bit
    
    // 对比: 仅量化 vs 完整流程
    test_quantization_only(128, 4, 1000);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "所有测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
