//! Hadamard 旋转模块
//!
//! 实现 3 轮随机符号翻转的快速 Walsh-Hadamard 变换 (FWHT)，
//! 基于 Ailon-Chazelle 定理，将任意分布的向量旋转为近似正态分布。
//!
//! # 核心思想
//! 1. 随机符号翻转 (随机 ±1)
//! 2. Hadamard 变换 (FWHT)
//! 3. 重复 3 次
//!
//! # 效果
//! - 将任意分布归一化为近似正态分布
//! - 保持向量范数不变
//! - O(d log d) 时间复杂度

use crate::utils::next_power_of_2;
use rand::Rng;
use rand::SeedableRng;

/// Hadamard 旋转器
///
/// 存储 3 轮旋转所需的随机符号和缩放因子
pub struct HadamardRotation {
    /// 输入维度
    pub d_in: usize,
    /// 输出维度 (补齐到 2 的幂)
    pub d_out: usize,
    /// 第一轮随机符号
    pub signs1: Vec<f32>,
    /// 第二轮随机符号
    pub signs2: Vec<f32>,
    /// 第三轮随机符号
    pub signs3: Vec<f32>,
    /// 缩放因子 1 / (d * √d)
    pub scale: f32,
}

/// 原地快速 Walsh-Hadamard 变换
///
/// FWHT 是 Hadamard 矩阵乘法的快速算法，
/// 时间复杂度 O(n log n)，空间复杂度 O(1)。
///
/// # 参数
/// - `buf`: 待变换的向量 (原地修改)
///
/// # 变换过程
/// 对于 n=4 的例子:
/// ```text
/// [a, b, c, d] -> [a+b, a-b, c+d, c-d] -> [a+b+c+d, a+b-c-d, a-b+c-d, a-b-c+d]
/// ```
fn fwht_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let mut step = 1;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in i..i + step {
                let a = buf[j];
                let b = buf[j + step];
                // 蝶形运算: [a, b] -> [a+b, a-b]
                buf[j] = a + b;
                buf[j + step] = a - b;
            }
            i += step * 2;
        }
        step *= 2;
    }
}

impl HadamardRotation {
    /// 创建新的 Hadamard 旋转器
    ///
    /// # 参数
    /// - `d`: 输入向量维度
    /// - `seed`: 随机种子 (保证可复现)
    ///
    /// # 返回值
    /// 初始化的 Hadamard 旋转器
    pub fn new(d: usize, seed: u64) -> Self {
        // 补齐到 2 的幂 (FWHT 要求)
        let d_out = next_power_of_2(d);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        // 生成 3 轮随机符号
        let signs1: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        let signs3: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();

        // 缩放因子: 1 / (d * √d) = 1 / d^(3/2)
        // 这是因为 3 次 FWHT 后，每个元素是 d^(3/2) 个原始元素的线性组合
        let scale = 1.0 / (d_out as f32 * (d_out as f32).sqrt());

        Self {
            d_in: d,
            d_out,
            signs1,
            signs2,
            signs3,
            scale,
        }
    }

    /// 对单个向量应用 Hadamard 旋转
    ///
    /// # 参数
    /// - `x`: 输入向量
    ///
    /// # 返回值
    /// 旋转后的向量 (长度为 d_out)
    ///
    /// # 过程
    /// 1. 应用第一轮随机符号
    /// 2. FWHT
    /// 3. 应用第二轮随机符号
    /// 4. FWHT
    /// 5. 应用第三轮随机符号
    /// 6. FWHT
    /// 7. 缩放
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut buf = vec![0.0f32; self.d_out];
        self.apply_into(x, &mut buf);
        buf
    }

    pub fn apply_into(&self, x: &[f32], buf: &mut [f32]) {
        assert!(x.len() >= self.d_in);
        assert!(buf.len() >= self.d_out);

        for i in 0..self.d_in {
            buf[i] = x[i] * self.signs1[i];
        }
        for i in self.d_in..self.d_out {
            buf[i] = 0.0;
        }
        fwht_inplace(&mut buf[..self.d_out]);

        for i in 0..self.d_out {
            buf[i] *= self.signs2[i];
        }
        fwht_inplace(&mut buf[..self.d_out]);

        for i in 0..self.d_out {
            buf[i] *= self.signs3[i];
        }
        fwht_inplace(&mut buf[..self.d_out]);

        for i in 0..self.d_out {
            buf[i] *= self.scale;
        }
    }

    pub fn apply_batch(&self, n: usize, x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; n * self.d_out];
        let mut buf = vec![0.0f32; self.d_out];
        for i in 0..n {
            self.apply_into(&x[i * self.d_in..(i + 1) * self.d_in], &mut buf);
            result[i * self.d_out..(i + 1) * self.d_out].copy_from_slice(&buf[..self.d_out]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::l2_norm;

    /// 测试 FWHT 基本功能
    #[test]
    fn test_fwht_basic() {
        let mut buf = vec![1.0f32, 0.0, 0.0, 0.0];
        fwht_inplace(&mut buf);
        // [1,0,0,0] 经过 Hadamard 变换后应该全是 1
        assert!((buf[0] - 1.0).abs() < 1e-5);
        assert!((buf[1] - 1.0).abs() < 1e-5);
        assert!((buf[2] - 1.0).abs() < 1e-5);
        assert!((buf[3] - 1.0).abs() < 1e-5);
    }

    /// 测试 Hadamard 旋转保持范数
    #[test]
    fn test_hadamard_rotation_preserves_norm() {
        let rot = HadamardRotation::new(128, 12345);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let norm_before = l2_norm(&x);
        let rotated = rot.apply(&x);
        let norm_after = l2_norm(&rotated[..128]);
        // 范数应该基本保持不变 (允许 1% 误差)
        assert!((norm_before - norm_after).abs() / norm_before < 0.01);
    }

    /// 测试 Hadamard 旋转是确定性的
    #[test]
    fn test_hadamard_rotation_deterministic() {
        let rot = HadamardRotation::new(64, 42);
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let r1 = rot.apply(&x);
        let r2 = rot.apply(&x);
        for i in 0..64 {
            assert!((r1[i] - r2[i]).abs() < 1e-10);
        }
    }
}
