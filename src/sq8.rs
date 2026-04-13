pub struct SQ8Quantizer {
    pub d: usize,
    vmin: Vec<f32>,
    vmax: Vec<f32>,
}

impl SQ8Quantizer {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            vmin: vec![f32::MAX; d],
            vmax: vec![f32::MIN; d],
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.vmin.fill(f32::MAX);
        self.vmax.fill(f32::MIN);

        for i in 0..n {
            for j in 0..self.d {
                let val = data[i * self.d + j];
                self.vmin[j] = self.vmin[j].min(val);
                self.vmax[j] = self.vmax[j].max(val);
            }
        }

        for j in 0..self.d {
            if self.vmax[j] - self.vmin[j] < 1e-6 {
                self.vmax[j] = self.vmin[j] + 1e-6;
            }
        }
    }

    pub fn code_size(&self) -> usize {
        self.d
    }

    pub fn encode(&self, x: &[f32], code: &mut [u8]) {
        for j in 0..self.d {
            let normalized = ((x[j] - self.vmin[j]) / (self.vmax[j] - self.vmin[j]))
                .clamp(0.0, 1.0);
            code[j] = (normalized * 255.0) as u8;
        }
    }

    pub fn decode(&self, code: &[u8], x: &mut [f32]) {
        for j in 0..self.d {
            x[j] = self.vmin[j] + (code[j] as f32 / 255.0) * (self.vmax[j] - self.vmin[j]);
        }
    }

    pub fn compute_distance(&self, code: &[u8], query: &[f32]) -> f32 {
        let mut dist = 0.0f32;
        for j in 0..self.d {
            let decoded = self.vmin[j] + (code[j] as f32 / 255.0) * (self.vmax[j] - self.vmin[j]);
            let diff = decoded - query[j];
            dist += diff * diff;
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::l2_distance;

    #[test]
    fn test_sq8_roundtrip() {
        let d = 128;
        let mut sq8 = SQ8Quantizer::new(d);
        let data: Vec<f32> = (0..1000 * d).map(|i| (i as f32 * 0.001).sin()).collect();
        sq8.train(&data, 1000);

        let x = &data[0..d];
        let mut code = vec![0u8; sq8.code_size()];
        sq8.encode(x, &mut code);
        let mut decoded = vec![0.0f32; d];
        sq8.decode(&code, &mut decoded);

        let orig_norm = crate::utils::l2_norm(x);
        let err = l2_distance(x, &decoded).sqrt();
        assert!(err / orig_norm < 0.07, "SQ8 relative error too large: {}", err / orig_norm);
    }

    #[test]
    fn test_sq8_distance_accuracy() {
        let d = 64;
        let mut sq8 = SQ8Quantizer::new(d);
        let data: Vec<f32> = (0..500 * d).map(|i| (i as f32 * 0.01).cos()).collect();
        sq8.train(&data, 500);

        let a = &data[0..d];
        let b = &data[d..2 * d];

        let mut code_a = vec![0u8; sq8.code_size()];
        let mut code_b = vec![0u8; sq8.code_size()];
        sq8.encode(a, &mut code_a);
        sq8.encode(b, &mut code_b);

        let true_dist = l2_distance(a, b);
        let sq8_dist = sq8.compute_distance(&code_a, b);
        assert!((true_dist - sq8_dist).abs() / true_dist < 0.1);
    }
}
