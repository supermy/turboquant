#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstdlib>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#define VQ_ALLOCA _alloca
#else
#include <alloca.h>
#define VQ_ALLOCA alloca
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
#include <arm_neon.h>
#define VQ_NEON 1
#elif defined(__AVX2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64)
#include <immintrin.h>
#define VQ_AVX2 1
#endif

namespace vq {

inline float l2_distance_neon(const float* a, const float* b, int d) {
    float result = 0.0f;
#if VQ_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }
    result = vaddvq_f32(sum);
    for (; i < d; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
#elif VQ_AVX2
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (; i < d; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
#else
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
#endif
    return result;
}

inline void build_rabitq_lut(
    const float* query, const float* centroid, int d,
    float* lut, float* c1_out, float* c34_out, float* qr_to_c_l2sqr_out)
{
    float inv_d = 1.0f / sqrtf((float)d);
    float qr_to_c_l2sqr = 0.0f;
    float sum_q = 0.0f;
    int base_size = (d + 7) / 8;

    float* rotated_q = (float*)VQ_ALLOCA(d * sizeof(float));

    if (centroid) {
        for (int i = 0; i < d; i++) {
            float diff = query[i] - centroid[i];
            rotated_q[i] = diff;
            qr_to_c_l2sqr += diff * diff;
        }
    } else {
        for (int i = 0; i < d; i++) {
            rotated_q[i] = query[i];
            qr_to_c_l2sqr += query[i] * query[i];
        }
    }

    for (int i = 0; i < d; i++) {
        sum_q += rotated_q[i];
    }

    for (int byte_idx = 0; byte_idx < base_size; byte_idx++) {
        float* table = lut + byte_idx * 256;
        for (int byte_val = 0; byte_val < 256; byte_val++) {
            float acc = 0.0f;
            for (int bit = 0; bit < 8; bit++) {
                int dim = byte_idx * 8 + bit;
                if (dim < d && (byte_val >> bit) & 1) {
                    acc += rotated_q[dim];
                }
            }
            table[byte_val] = acc;
        }
    }

    *c1_out = 2.0f * inv_d;
    *c34_out = sum_q * inv_d;
    *qr_to_c_l2sqr_out = qr_to_c_l2sqr;
}

inline float rabitq_signs_distance(const uint8_t* signs, const float* lut, int signs_size) {
    float dot_qo = 0.0f;
#if VQ_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < signs_size; i += 4) {
        float32x4_t v0 = vld1q_f32(&lut[(i + 0) * 256 + signs[i + 0]]);
        float32x4_t v1 = vld1q_f32(&lut[(i + 1) * 256 + signs[i + 1]]);
        float32x4_t v2 = vld1q_f32(&lut[(i + 2) * 256 + signs[i + 2]]);
        float32x4_t v3 = vld1q_f32(&lut[(i + 3) * 256 + signs[i + 3]]);
        sum = vaddq_f32(sum, v0);
        sum = vaddq_f32(sum, v1);
        sum = vaddq_f32(sum, v2);
        sum = vaddq_f32(sum, v3);
    }
    dot_qo = vaddvq_f32(sum);
    for (; i < signs_size; i++) {
        dot_qo += lut[i * 256 + signs[i]];
    }
#else
    for (int i = 0; i < signs_size; i++) {
        dot_qo += lut[i * 256 + signs[i]];
    }
#endif
    return dot_qo;
}

inline float rabitq_full_distance(
    float dot_qo, float or_minus_c_l2sqr, float dp_multiplier,
    float c1, float c34, float qr_to_c_l2sqr, bool is_inner_product)
{
    float final_dot = c1 * dot_qo - c34;
    float dist = or_minus_c_l2sqr + qr_to_c_l2sqr - 2.0f * dp_multiplier * final_dot;
    if (is_inner_product) {
        return -0.5f * (dist - qr_to_c_l2sqr);
    }
    return dist > 0.0f ? dist : 0.0f;
}

inline float sq8_distance(
    const uint8_t* code, const float* query,
    const float* vmin, const float* scale, int d)
{
    float dist = 0.0f;
#if VQ_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 3 < d; j += 4) {
        uint32_t c0 = code[j + 0];
        uint32_t c1 = code[j + 1];
        uint32_t c2 = code[j + 2];
        uint32_t c3 = code[j + 3];
        float32x4_t codes_f = { (float)c0, (float)c1, (float)c2, (float)c3 };
        float32x4_t s = vld1q_f32(&scale[j]);
        float32x4_t v = vld1q_f32(&vmin[j]);
        float32x4_t q = vld1q_f32(&query[j]);
        float32x4_t decoded = vfmaq_f32(v, codes_f, s);
        float32x4_t diff = vsubq_f32(decoded, q);
        sum = vfmaq_f32(sum, diff, diff);
    }
    dist = vaddvq_f32(sum);
    for (; j < d; j++) {
        float decoded = vmin[j] + (float)code[j] * scale[j];
        float diff = decoded - query[j];
        dist += diff * diff;
    }
#else
    for (int j = 0; j < d; j++) {
        float decoded = vmin[j] + (float)code[j] * scale[j];
        float diff = decoded - query[j];
        dist += diff * diff;
    }
#endif
    return dist;
}

inline float lloyd_max_4bit_distance(
    const uint8_t* code, int code_size,
    const float* lut)
{
    float dist = 0.0f;
#if VQ_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < code_size; i += 4) {
        float32x4_t v0 = vld1q_f32(&lut[(i + 0) * 256 + code[i + 0]]);
        float32x4_t v1 = vld1q_f32(&lut[(i + 1) * 256 + code[i + 1]]);
        float32x4_t v2 = vld1q_f32(&lut[(i + 2) * 256 + code[i + 2]]);
        float32x4_t v3 = vld1q_f32(&lut[(i + 3) * 256 + code[i + 3]]);
        sum = vaddq_f32(sum, v0);
        sum = vaddq_f32(sum, v1);
        sum = vaddq_f32(sum, v2);
        sum = vaddq_f32(sum, v3);
    }
    dist = vaddvq_f32(sum);
    for (; i < code_size; i++) {
        dist += lut[i * 256 + code[i]];
    }
#else
    for (int i = 0; i < code_size; i++) {
        dist += lut[i * 256 + code[i]];
    }
#endif
    return dist;
}

inline void build_lloyd_max_4bit_lut(
    const float* query, int d,
    const float* centroids, int n_centroids,
    float* lut, int code_size)
{
    for (int byte_idx = 0; byte_idx < code_size; byte_idx++) {
        float* table = lut + byte_idx * 256;
        for (int byte_val = 0; byte_val < 256; byte_val++) {
            int idx_lo = byte_val & 0x0F;
            int idx_hi = (byte_val >> 4) & 0x0F;
            int dim_lo = byte_idx * 2;
            int dim_hi = byte_idx * 2 + 1;
            float acc = 0.0f;
            if (dim_lo < d) {
                float diff = centroids[idx_lo] - query[dim_lo];
                acc += diff * diff;
            }
            if (dim_hi < d) {
                float diff = centroids[idx_hi] - query[dim_hi];
                acc += diff * diff;
            }
            table[byte_val] = acc;
        }
    }
}

} // namespace vq
