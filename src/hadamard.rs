use crate::utils::next_power_of_2;
use rand::Rng;
use rand::SeedableRng;

pub struct HadamardRotation {
    pub d_in: usize,
    pub d_out: usize,
    signs1: Vec<f32>,
    signs2: Vec<f32>,
    signs3: Vec<f32>,
    scale: f32,
}

fn fwht_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let mut step = 1;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in i..i + step {
                let a = buf[j];
                let b = buf[j + step];
                buf[j] = a + b;
                buf[j + step] = a - b;
            }
            i += step * 2;
        }
        step *= 2;
    }
}

impl HadamardRotation {
    pub fn new(d: usize, seed: u64) -> Self {
        let d_out = next_power_of_2(d);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let signs1: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        let signs3: Vec<f32> = (0..d_out).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();

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

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= self.d_in);
        let mut buf = vec![0.0f32; self.d_out];

        for i in 0..self.d_in {
            buf[i] = x[i] * self.signs1[i];
        }

        fwht_inplace(&mut buf);

        for i in 0..self.d_out {
            buf[i] *= self.signs2[i];
        }
        fwht_inplace(&mut buf);

        for i in 0..self.d_out {
            buf[i] *= self.signs3[i];
        }
        fwht_inplace(&mut buf);

        for i in 0..self.d_out {
            buf[i] *= self.scale;
        }

        buf
    }

    pub fn apply_batch(&self, n: usize, x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; n * self.d_out];
        for i in 0..n {
            let rotated = self.apply(&x[i * self.d_in..(i + 1) * self.d_in]);
            result[i * self.d_out..(i + 1) * self.d_out].copy_from_slice(&rotated);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::l2_norm;

    #[test]
    fn test_fwht_basic() {
        let mut buf = vec![1.0f32, 0.0, 0.0, 0.0];
        fwht_inplace(&mut buf);
        assert!((buf[0] - 1.0).abs() < 1e-5);
        assert!((buf[1] - 1.0).abs() < 1e-5);
        assert!((buf[2] - 1.0).abs() < 1e-5);
        assert!((buf[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hadamard_rotation_preserves_norm() {
        let rot = HadamardRotation::new(128, 12345);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let norm_before = l2_norm(&x);
        let rotated = rot.apply(&x);
        let norm_after = l2_norm(&rotated[..128]);
        assert!((norm_before - norm_after).abs() / norm_before < 0.01);
    }

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
