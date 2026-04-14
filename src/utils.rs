use rand::Rng;
use rand::SeedableRng;
use rand_distr::Distribution;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatOrd(pub f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

#[inline(always)]
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut vsum = vdupq_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            vsum = vfmaq_f32(vsum, diff, diff);
            i += 4;
        }
        sum = vaddvq_f32(vsum);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            use std::arch::x86_64::*;
            let mut vsum = _mm256_setzero_ps();
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let diff = _mm256_sub_ps(va, vb);
                vsum = _mm256_fmadd_ps(diff, diff, vsum);
                i += 8;
            }
            let hi = _mm256_extractf128_ps(vsum, 1);
            let lo = _mm256_castps256_ps128(vsum);
            let sum128 = _mm_add_ps(hi, lo);
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum128);
            sum = result[0] + result[1] + result[2] + result[3];
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            while i + 4 <= n {
                let diff0 = a[i] - b[i];
                let diff1 = a[i + 1] - b[i + 1];
                let diff2 = a[i + 2] - b[i + 2];
                let diff3 = a[i + 3] - b[i + 3];
                sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                i += 4;
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        while i + 4 <= n {
            let diff0 = a[i] - b[i];
            let diff1 = a[i + 1] - b[i + 1];
            let diff2 = a[i + 2] - b[i + 2];
            let diff3 = a[i + 3] - b[i + 3];
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            i += 4;
        }
    }

    while i < n {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut vsum = vdupq_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vsum = vfmaq_f32(vsum, va, vb);
            i += 4;
        }
        sum = vaddvq_f32(vsum);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            use std::arch::x86_64::*;
            let mut vsum = _mm256_setzero_ps();
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                vsum = _mm256_fmadd_ps(va, vb, vsum);
                i += 8;
            }
            let hi = _mm256_extractf128_ps(vsum, 1);
            let lo = _mm256_castps256_ps128(vsum);
            let sum128 = _mm_add_ps(hi, lo);
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum128);
            sum = result[0] + result[1] + result[2] + result[3];
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            while i + 4 <= n {
                sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
                i += 4;
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        while i + 4 <= n {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            i += 4;
        }
    }

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn sq8_distance_simd(code: &[u8], query: &[f32], vmin: &[f32], scale: &[f32], d: usize) -> f32 {
    let mut dist = 0.0f32;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut vsum = vdupq_f32(0.0);
        while i + 4 <= d {
            let codes = vld1_u8(code.as_ptr().add(i));
            let codes32 = vmovl_u16(vget_low_u16(vmovl_u8(codes)));
            let codes_f = vcvtq_f32_u32(codes32);
            let s = vld1q_f32(scale.as_ptr().add(i));
            let v = vld1q_f32(vmin.as_ptr().add(i));
            let q = vld1q_f32(query.as_ptr().add(i));
            let decoded = vfmaq_f32(v, codes_f, s);
            let diff = vsubq_f32(decoded, q);
            vsum = vfmaq_f32(vsum, diff, diff);
            i += 4;
        }
        dist = vaddvq_f32(vsum);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            use std::arch::x86_64::*;
            let mut vsum = _mm256_setzero_ps();
            while i + 8 <= d {
                let codes_low = _mm_loadl_epi64(unsafe { &*(code.as_ptr().add(i) as *const __m128i) });
                let codes_i32 = _mm256_cvtepu8_epi32(codes_low);
                let codes_f = _mm256_cvtepi32_ps(codes_i32);
                let s = _mm256_loadu_ps(scale.as_ptr().add(i));
                let v = _mm256_loadu_ps(vmin.as_ptr().add(i));
                let q = _mm256_loadu_ps(query.as_ptr().add(i));
                let decoded = _mm256_fmadd_ps(codes_f, s, v);
                let diff = _mm256_sub_ps(decoded, q);
                vsum = _mm256_fmadd_ps(diff, diff, vsum);
                i += 8;
            }
            let hi = _mm256_extractf128_ps(vsum, 1);
            let lo = _mm256_castps256_ps128(vsum);
            let sum128 = _mm_add_ps(hi, lo);
            let mut result = [0.0f32; 4];
            unsafe { _mm_storeu_ps(result.as_mut_ptr(), sum128); }
            dist = result[0] + result[1] + result[2] + result[3];
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            while i + 4 <= d {
                for j in 0..4 {
                    let decoded = vmin[i + j] + code[i + j] as f32 * scale[i + j];
                    let diff = decoded - query[i + j];
                    dist += diff * diff;
                }
                i += 4;
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        while i + 4 <= d {
            for j in 0..4 {
                let decoded = vmin[i + j] + code[i + j] as f32 * scale[i + j];
                let diff = decoded - query[i + j];
                dist += diff * diff;
            }
            i += 4;
        }
    }

    while i < d {
        let decoded = vmin[i] + code[i] as f32 * scale[i];
        let diff = decoded - query[i];
        dist += diff * diff;
        i += 1;
    }
    dist
}

pub fn l2_norm_sq(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}

pub fn l2_norm(x: &[f32]) -> f32 {
    l2_norm_sq(x).sqrt()
}

pub fn l2_normalize(x: &mut [f32]) {
    let norm = l2_norm_simd(x);
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        for v in x.iter_mut() {
            *v *= inv_norm;
        }
    }
}

#[inline(always)]
pub fn l2_norm_simd(x: &[f32]) -> f32 {
    let n = x.len();
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut vsum = vdupq_f32(0.0);
        while i + 4 <= n {
            let vx = vld1q_f32(x.as_ptr().add(i));
            vsum = vfmaq_f32(vsum, vx, vx);
            i += 4;
        }
        sum = vaddvq_f32(vsum);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            use std::arch::x86_64::*;
            let mut vsum = _mm256_setzero_ps();
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                vsum = _mm256_fmadd_ps(vx, vx, vsum);
                i += 8;
            }
            let hi = _mm256_extractf128_ps(vsum, 1);
            let lo = _mm256_castps256_ps128(vsum);
            let sum128 = _mm_add_ps(hi, lo);
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum128);
            sum = result[0] + result[1] + result[2] + result[3];
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            while i + 4 <= n {
                sum += x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 2] * x[i + 2] + x[i + 3] * x[i + 3];
                i += 4;
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        while i + 4 <= n {
            sum += x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 2] * x[i + 2] + x[i + 3] * x[i + 3];
            i += 4;
        }
    }

    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }
    sum.sqrt()
}

pub fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

#[inline(always)]
pub unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse")]
        {
            std::arch::x86_64::_mm_prefetch::<3>(ptr as *const i8);
        }
        #[cfg(not(target_feature = "sse"))]
        {
            let _ = ptr;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

pub fn generate_clustered_data(
    n: usize,
    d: usize,
    n_clusters: usize,
    cluster_std: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];

    let mut centroids = vec![0.0f32; n_clusters * d];
    for c in 0..n_clusters {
        for j in 0..d {
            centroids[c * d + j] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let norm = l2_norm(&centroids[c * d..(c + 1) * d]);
        for j in 0..d {
            centroids[c * d + j] /= norm;
        }
    }

    let normal = rand_distr::Normal::new(0.0, cluster_std as f64).unwrap();
    for i in 0..n {
        let cluster_id = i % n_clusters;
        let center = &centroids[cluster_id * d..(cluster_id + 1) * d];

        for j in 0..d {
            data[i * d + j] = center[j] + normal.sample(&mut rng) as f32;
        }

        let norm = l2_norm(&data[i * d..(i + 1) * d]);
        for j in 0..d {
            data[i * d + j] /= norm;
        }
    }

    data
}

pub fn generate_queries(
    data: &[f32],
    n_data: usize,
    n_queries: usize,
    d: usize,
    noise_level: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut queries = vec![0.0f32; n_queries * d];
    let normal = rand_distr::Normal::new(0.0, noise_level as f64).unwrap();

    for i in 0..n_queries {
        let src_idx = rng.gen_range(0..n_data);

        for j in 0..d {
            queries[i * d + j] = data[src_idx * d + j] + normal.sample(&mut rng) as f32;
        }

        let norm = l2_norm(&queries[i * d..(i + 1) * d]);
        for j in 0..d {
            queries[i * d + j] /= norm;
        }
    }

    queries
}

pub fn compute_ground_truth(data: &[f32], queries: &[f32], n_data: usize, n_queries: usize, d: usize, k: usize) -> Vec<Vec<usize>> {
    let mut gt = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let query = &queries[q * d..(q + 1) * d];
        let mut dists: Vec<(f32, usize)> = (0..n_data)
            .map(|i| (l2_distance_simd(query, &data[i * d..(i + 1) * d]), i))
            .collect();
        dists.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        gt.push(dists.iter().map(|&(_, i)| i).collect());
    }
    gt
}

pub fn compute_recall(result_ids: &[Vec<usize>], gt_ids: &[Vec<usize>], nq: usize, k: usize) -> f32 {
    let mut total_recall = 0usize;
    for q in 0..nq {
        let mut found = result_ids[q].clone();
        found.sort();
        for i in 0..k {
            if found.binary_search(&gt_ids[q][i]).is_ok() {
                total_recall += 1;
            }
        }
    }
    total_recall as f32 / (nq * k) as f32
}
