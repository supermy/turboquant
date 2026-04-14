// PolarQuant 核心数学工具函数
//
// 包含：
// - 随机正交矩阵生成（QR 分解法）
// - Hadamard 快速旋转（FWHT）
// - Cartesian ↔ Polar 坐标变换
// - Beta 分布参数计算
// - Lloyd-Max 最优量化（Beta 分布优化）
// - Lanczos 近似 log(Gamma) 函数
// - 正则化不完全 Beta 函数（数值积分）

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/// 生成随机正交矩阵（QR 分解法）
///
/// PolarQuant 的核心前提：随机正交旋转后，每个坐标分量
/// 投影到 [0,1] 后服从 Beta(d/2, d/2) 分布。
/// 这使得我们可以预先计算最优量化质心，无需存储元数据。
///
/// # 算法步骤
/// 1. 生成 d×d 随机高斯矩阵 A
/// 2. 对 A 进行 QR 分解，得到正交矩阵 Q
/// 3. 修正 R 对角线符号，确保 Q 的列方向一致
/// 4. 确保 det(Q) = 1（纯旋转，无反射）
///
/// # 参数
/// - `d`: 矩阵维度
/// - `seed`: 随机种子，确保可复现性
pub fn random_orthogonal_matrix(d: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // 步骤 1: 生成 d×d 随机高斯矩阵
    let mut a = DMatrix::<f64>::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            a[(i, j)] = rng.sample(StandardNormal);
        }
    }

    // 步骤 2: QR 分解
    let qr = a.qr();
    let q = qr.q();
    let r = qr.r();

    // 步骤 3: 修正 R 对角线符号，确保 Q 的列方向一致
    let mut diag_signs = DVector::<f64>::zeros(d);
    for i in 0..d {
        diag_signs[i] = if r[(i, i)] >= 0.0 { 1.0 } else { -1.0 };
    }

    let signs_matrix = DMatrix::<f64>::from_diagonal(&diag_signs);
    let q = q * &signs_matrix;

    // 步骤 4: 确保行列式为 +1（纯旋转矩阵）
    if q.determinant() < 0.0 {
        let mut q = q;
        for i in 0..d {
            q[(i, 0)] = -q[(i, 0)];
        }
        q
    } else {
        q
    }
}

/// Hadamard 旋转（快速 Walsh-Hadamard 变换，FWHT）
///
/// Hadamard 旋转是结构化正交变换，时间复杂度 O(d log d)，
/// 远优于随机正交矩阵的 O(d²) 矩阵乘法。
/// 且 Hadamard 矩阵是自逆的（H^T = H），逆变换即正变换。
///
/// # 算法
/// 对输入向量执行蝶形运算（butterfly operation），
/// 然后除以 sqrt(d_pad) 归一化以保持范数不变。
///
/// # 注意
/// 如果输入维度 d 不是 2 的幂，自动零填充到 d_pad = 2^ceil(log2(d))
pub fn hadamard_rotation(x: &[f64]) -> Vec<f64> {
    let d = x.len();

    // 零填充到 2 的幂次
    let d_pad = d.next_power_of_two();
    let mut x_padded = vec![0.0f64; d_pad];
    x_padded[..d].copy_from_slice(x);

    // 蝶形运算：Fast Walsh-Hadamard Transform
    let mut h = 1usize;
    while h < d_pad {
        for i in (0..d_pad).step_by(h * 2) {
            for j in i..(i + h) {
                let a = x_padded[j];
                let b = x_padded[j + h];
                x_padded[j] = a + b;
                x_padded[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // 归一化：除以 sqrt(d_pad) 保持 L2 范数不变
    let scale = 1.0 / (d_pad as f64).sqrt();
    for v in x_padded.iter_mut() {
        *v *= scale;
    }

    // 截断回原始维度
    x_padded[..d].to_vec()
}

/// Cartesian 坐标 → Polar 坐标变换
///
/// 将 d 维向量 x 分解为：
/// - r: 半径（L2 范数）
/// - angles: d-1 个角度，前 d-2 个在 [0, π]，最后一个在 [0, 2π]
///
/// # 变换公式
/// ```text
/// x_0 = r * cos(θ_0)
/// x_1 = r * sin(θ_0) * cos(θ_1)
/// x_2 = r * sin(θ_0) * sin(θ_1) * cos(θ_2)
/// ...
/// x_{d-2} = r * sin(θ_0) * ... * sin(θ_{d-3}) * cos(θ_{d-2})
/// x_{d-1} = r * sin(θ_0) * ... * sin(θ_{d-3}) * sin(θ_{d-2})
/// ```
///
/// # 返回
/// (半径 r, 角度向量 angles)
pub fn cartesian_to_polar(x: &[f64]) -> (f64, Vec<f64>) {
    let d = x.len();

    // 计算半径（L2 范数）
    let r = vector_norm(x);

    // 零向量特殊处理
    if r < 1e-10 {
        return (0.0, vec![0.0; d - 1]);
    }

    // 归一化到单位球面
    let x_norm: Vec<f64> = x.iter().map(|&v| v / r).collect();

    // 递推计算角度
    let mut angles = vec![0.0f64; d - 1];

    for i in 0..d - 1 {
        // 剩余分量的范数
        let remaining_norm = vector_norm(&x_norm[i..]);
        if remaining_norm < 1e-10 {
            angles[i] = 0.0;
        } else {
            // cos(θ_i) = x_i / ||x[i:]||
            let cos_theta = (x_norm[i] / remaining_norm).clamp(-1.0, 1.0);
            angles[i] = cos_theta.acos();
        }
    }

    // 最后一个角度：根据最后一个坐标的符号调整到 [0, 2π]
    if x.last().copied().unwrap_or(0.0) < 0.0 {
        if let Some(last) = angles.last_mut() {
            *last = 2.0 * std::f64::consts::PI - *last;
        }
    }

    (r, angles)
}

/// Polar 坐标 → Cartesian 坐标变换
///
/// 将 (半径 r, 角度 angles) 还原为 d 维向量。
/// 这是 cartesian_to_polar 的逆变换。
pub fn polar_to_cartesian(r: f64, angles: &[f64]) -> Vec<f64> {
    let d = angles.len() + 1;
    let mut x = vec![0.0f64; d];

    // 零半径特殊处理
    if r < 1e-10 {
        return x;
    }

    // 递推计算：累积 sin 乘积
    let mut sin_product = 1.0f64;

    for i in 0..d - 1 {
        x[i] = r * sin_product * angles[i].cos();
        sin_product *= angles[i].sin();
    }

    // 最后一个分量 = r * sin(θ_0) * sin(θ_1) * ... * sin(θ_{d-2})
    x[d - 1] = r * sin_product;

    x
}

/// 计算 Beta 分布参数
///
/// 随机正交旋转后，每个坐标分量投影到 [0,1] 后服从 Beta(d/2, d/2) 分布。
/// 对称参数 α = β = d/2，维度越高分布越集中在 0.5 附近。
pub fn beta_distribution_params(d: usize) -> (f64, f64) {
    let alpha = d as f64 / 2.0;
    let beta_param = d as f64 / 2.0;
    (alpha, beta_param)
}

/// Beta 分布概率密度函数 (PDF)
///
/// f(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
/// 其中 B(α, β) = Γ(α)Γ(β) / Γ(α+β) 是 Beta 函数
pub fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let ln_beta = log_gamma(alpha) + log_gamma(beta) - log_gamma(alpha + beta);
    let ln_pdf = (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_beta;
    ln_pdf.exp()
}

/// Beta 分布累积分布函数 (CDF)
///
/// 使用数值积分（中点法）计算正则化不完全 Beta 函数
pub fn beta_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    regularized_incomplete_beta(x, alpha, beta)
}

/// 计算 Lloyd-Max 最优量化质心
///
/// Lloyd-Max 算法迭代优化量化边界和质心，最小化均方误差。
/// 对于 Beta(α, β) 分布，质心可以不依赖数据预先计算。
///
/// # 算法步骤（每次迭代）
/// 1. 计算相邻质心中点作为 Voronoi 边界
/// 2. 对每个量化单元，计算条件期望 E[X | X ∈ cell] 作为新质心
/// 3. 使用 Simpson 法则数值积分
///
/// # 参数
/// - `alpha`, `beta_param`: Beta 分布参数
/// - `bits`: 量化位数（量化级数 = 2^bits）
/// - `n_iterations`: Lloyd-Max 迭代次数
pub fn compute_lloyd_max_centroids(
    alpha: f64,
    beta_param: f64,
    bits: u32,
    n_iterations: usize,
) -> Vec<f64> {
    let n_levels = 1usize << bits;

    // 初始化：均匀分布在 [0, 1]
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| (i as f64 + 0.5) / n_levels as f64)
        .collect();

    for _ in 0..n_iterations {
        // 步骤 1: 计算量化边界（相邻质心的中点）
        let mut boundaries = vec![0.0f64; n_levels + 1];
        boundaries[0] = 0.0;
        boundaries[n_levels] = 1.0;

        for i in 1..n_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }

        // 步骤 2: 更新质心为条件期望 E[X | X ∈ cell]
        let mut new_centroids = vec![0.0f64; n_levels];
        for i in 0..n_levels {
            let a = boundaries[i];
            let b = boundaries[i + 1];

            if a >= b {
                new_centroids[i] = centroids[i];
                continue;
            }

            // Simpson 法则数值积分
            let n_points = 200usize;
            let dx = (b - a) / (n_points - 1) as f64;

            let mut integral_x_pdf = 0.0f64; // ∫ x * f(x) dx
            let mut integral_pdf = 0.0f64;   // ∫ f(x) dx

            for k in 0..n_points {
                let x = a + k as f64 * dx;
                let pdf_val = beta_pdf(x, alpha, beta_param);

                // Simpson 权重：端点 1，奇数点 4，偶数点 2
                let weight = if k == 0 || k == n_points - 1 {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };

                integral_x_pdf += weight * x * pdf_val;
                integral_pdf += weight * pdf_val;
            }

            integral_x_pdf *= dx / 3.0;
            integral_pdf *= dx / 3.0;

            // 条件期望 = ∫ x*f(x)dx / ∫ f(x)dx
            if integral_pdf > 1e-10 {
                new_centroids[i] = integral_x_pdf / integral_pdf;
            } else {
                new_centroids[i] = (a + b) / 2.0;
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// Lloyd-Max 量化：将连续值映射到最近的质心索引
///
/// 对每个输入值，找到距离最近的质心，返回其索引。
pub fn lloyd_max_quantize(x: &[f64], centroids: &[f64]) -> Vec<u32> {
    x.iter()
        .map(|&v| {
            let v_clamped = v.clamp(0.0, 1.0);
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (*a - v_clamped).abs();
                    let db = (*b - v_clamped).abs();
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        })
        .collect()
}

/// Lloyd-Max 反量化：将质心索引映射回质心值
pub fn lloyd_max_dequantize(indices: &[u32], centroids: &[f64]) -> Vec<f64> {
    indices.iter().map(|&idx| centroids[idx as usize]).collect()
}

/// 计算向量的 L2 范数
pub fn vector_norm(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Lanczos 近似计算 log(Γ(x))
///
/// 使用 Lanczos 逼近法，精度约 15 位有效数字。
/// 当 x < 0.5 时使用反射公式：Γ(x) = π / (sin(πx) * Γ(1-x))
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let lanczos_g = 7.0;
    // Lanczos 系数（g=7, n=9）
    #[allow(clippy::excessive_precision)]
    let lanczos_coef: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // 反射公式：x < 0.5 时利用 Γ(x)Γ(1-x) = π/sin(πx)
    if x < 0.5 {
        return std::f64::consts::PI / (std::f64::consts::PI * x).sin().ln() - log_gamma(1.0 - x);
    }

    let x = x - 1.0;
    // Lanczos 级数求和
    let a = lanczos_coef[0]
        + lanczos_coef[1..]
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, c)| acc + c / (x + i as f64 + 1.0));

    // Stirling 级数近似
    let t = x + lanczos_g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// 正则化不完全 Beta 函数 I_x(α, β)
///
/// 使用中点法数值积分近似计算。
/// I_x(α, β) = ∫_0^x f(t; α, β) dt
fn regularized_incomplete_beta(x: f64, alpha: f64, beta: f64) -> f64 {
    let n_points = 1000;
    let dx = x / n_points as f64;

    let mut integral = 0.0f64;
    for k in 0..n_points {
        let x_k = (k as f64 + 0.5) * dx; // 中点
        let pdf_val = beta_pdf(x_k, alpha, beta);
        integral += pdf_val * dx;
    }

    integral
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 测试随机正交矩阵的正交性：Q * Q^T = I
    #[test]
    fn test_random_orthogonal_orthogonality() {
        let d = 10;
        let q = random_orthogonal_matrix(d, 42);

        let identity = &q * &q.transpose();
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(identity[(i, j)], expected, epsilon = 1e-8);
            }
        }
    }

    /// 测试行列式为 ±1
    #[test]
    fn test_random_orthogonal_determinant() {
        let d = 5;
        let q = random_orthogonal_matrix(d, 42);
        let det = q.determinant();
        assert_relative_eq!(det.abs(), 1.0, epsilon = 1e-8);
    }

    /// 测试同一种子生成相同矩阵
    #[test]
    fn test_random_orthogonal_reproducibility() {
        let q1 = random_orthogonal_matrix(8, 123);
        let q2 = random_orthogonal_matrix(8, 123);
        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(q1[(i, j)], q2[(i, j)], epsilon = 1e-12);
            }
        }
    }

    /// 测试 Hadamard 旋转保持 L2 范数不变
    #[test]
    fn test_hadamard_norm_preservation() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0).sin()).collect();
        let x_rotated = hadamard_rotation(&x);
        let norm_orig = vector_norm(&x);
        let norm_rot = vector_norm(&x_rotated);
        assert_relative_eq!(norm_orig, norm_rot, epsilon = 1e-10);
    }

    /// 测试 Hadamard 旋转的线性性：H(a*x + b*y) = a*H(x) + b*H(y)
    #[test]
    fn test_hadamard_linearity() {
        let x1: Vec<f64> = (0..8).map(|i| (i as f64).cos()).collect();
        let x2: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let a = 2.0;
        let b = 3.0;

        let ax_plus_by: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&xi, &yi)| a * xi + b * yi)
            .collect();
        let result1 = hadamard_rotation(&ax_plus_by);

        let hx1 = hadamard_rotation(&x1);
        let hx2 = hadamard_rotation(&x2);
        let result2: Vec<f64> = hx1
            .iter()
            .zip(hx2.iter())
            .map(|(&hxi, &hyi)| a * hxi + b * hyi)
            .collect();

        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_relative_eq!(r1, r2, epsilon = 1e-10);
        }
    }

    /// 测试 2D 极坐标往返变换
    #[test]
    fn test_polar_roundtrip_2d() {
        let x = vec![3.0, 4.0];
        let (r, angles) = cartesian_to_polar(&x);
        let x_recon = polar_to_cartesian(r, &angles);
        for (orig, recon) in x.iter().zip(x_recon.iter()) {
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }

    /// 测试 3D 极坐标往返变换
    #[test]
    fn test_polar_roundtrip_3d() {
        let x = vec![1.0, 2.0, 2.0];
        let (r, angles) = cartesian_to_polar(&x);
        let x_recon = polar_to_cartesian(r, &angles);
        for (orig, recon) in x.iter().zip(x_recon.iter()) {
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }

    /// 测试 nD 极坐标往返变换
    #[test]
    fn test_polar_roundtrip_nd() {
        for d in [2, 4, 8, 16] {
            let x: Vec<f64> = (0..d).map(|i| ((i + 1) as f64).sin()).collect();
            let (r, angles) = cartesian_to_polar(&x);
            let x_recon = polar_to_cartesian(r, &angles);
            for (orig, recon) in x.iter().zip(x_recon.iter()) {
                assert_relative_eq!(orig, recon, epsilon = 1e-8);
            }
        }
    }

    /// 测试半径计算正确性：||[3,4]|| = 5
    #[test]
    fn test_polar_radius() {
        let x = vec![3.0, 4.0];
        let (r, _) = cartesian_to_polar(&x);
        assert_relative_eq!(r, 5.0, epsilon = 1e-10);
    }

    /// 测试零向量的极坐标变换
    #[test]
    fn test_polar_zero_vector() {
        let x = vec![0.0; 5];
        let (r, angles) = cartesian_to_polar(&x);
        assert_relative_eq!(r, 0.0, epsilon = 1e-10);
        assert_eq!(angles.len(), 4);
    }

    /// 测试 Beta 分布参数对称性：α = β = d/2
    #[test]
    fn test_beta_distribution_params() {
        for d in [2, 4, 8, 16, 32] {
            let (alpha, beta_param) = beta_distribution_params(d);
            assert_relative_eq!(alpha, beta_param);
            assert_relative_eq!(alpha, d as f64 / 2.0);
        }
    }

    /// 测试 Lloyd-Max 质心在 [0,1] 范围内
    #[test]
    fn test_lloyd_max_centroids_in_range() {
        let centroids = compute_lloyd_max_centroids(2.0, 2.0, 4, 100);
        assert_eq!(centroids.len(), 16);
        for &c in &centroids {
            assert!(c >= 0.0 && c <= 1.0, "centroid {} out of range", c);
        }
    }

    /// 测试 Lloyd-Max 量化/反量化往返
    #[test]
    fn test_lloyd_max_quantize_dequantize() {
        let centroids = compute_lloyd_max_centroids(2.0, 2.0, 3, 100);
        let x = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let indices = lloyd_max_quantize(&x, &centroids);
        let x_recon = lloyd_max_dequantize(&indices, &centroids);

        // 索引应在合法范围内
        for &idx in &indices {
            assert!((idx as usize) < centroids.len());
        }

        // 反量化值应等于某个质心
        for (_orig, recon) in x.iter().zip(x_recon.iter()) {
            assert!(centroids.iter().any(|c| (c - recon).abs() < 1e-10));
        }
    }

    /// 测试 log(Γ) 函数：Γ(1) = 1, Γ(2) = 1, Γ(5) = 24
    #[test]
    fn test_log_gamma() {
        assert_relative_eq!(log_gamma(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_gamma(2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_gamma(5.0), (24.0f64).ln(), epsilon = 1e-8);
    }

    /// 测试 Beta PDF：Beta(2,2) 在 x=0.5 处的值为 1.5
    #[test]
    fn test_beta_pdf() {
        let pdf_val = beta_pdf(0.5, 2.0, 2.0);
        assert!(pdf_val > 0.0);
        assert_relative_eq!(pdf_val, 1.5, epsilon = 0.01);
    }

    /// 测试 Beta CDF：对称分布 Beta(2,2) 在 x=0.5 处 CDF = 0.5
    #[test]
    fn test_beta_cdf() {
        let cdf_val = beta_cdf(0.5, 2.0, 2.0);
        assert_relative_eq!(cdf_val, 0.5, epsilon = 0.01);
    }
}
