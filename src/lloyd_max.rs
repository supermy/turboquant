pub struct LloydMaxQuantizer {
    pub d: usize,
    pub nbits: usize,
    pub k: usize,
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
}

impl LloydMaxQuantizer {
    pub fn new(d: usize, nbits: usize) -> Self {
        let k = 1usize << nbits;
        let mut q = Self {
            d,
            nbits,
            k,
            centroids: Vec::new(),
            boundaries: Vec::new(),
        };
        q.build_codebook();
        q
    }

    fn build_codebook(&mut self) {
        self.centroids = vec![0.0f32; self.k];
        self.boundaries = vec![0.0f32; self.k - 1];

        if self.d == 1 {
            for i in 0..self.k {
                self.centroids[i] = if i < self.k / 2 { -1.0 } else { 1.0 };
            }
        } else {
            self.lloyd_max_iteration();
        }

        for i in 0..self.k - 1 {
            self.boundaries[i] = 0.5 * (self.centroids[i] + self.centroids[i + 1]);
        }
    }

    fn lloyd_max_iteration(&mut self) {
        let ngrid = 32768usize;
        let step = 2.0 / ngrid as f64;
        let alpha = 0.5 * (self.d as f64 - 3.0);

        let mut xs = vec![0.0f64; ngrid];
        let mut prefix_w = vec![0.0f64; ngrid + 1];
        let mut prefix_wx = vec![0.0f64; ngrid + 1];

        for i in 0..ngrid {
            let x = -1.0 + (i as f64 + 0.5) * step;
            let one_minus_x2 = (1.0 - x * x).max(0.0);
            let w = if alpha == 0.0 {
                1.0
            } else {
                one_minus_x2.powf(alpha)
            };
            let w = if w.is_finite() && w >= 0.0 { w } else { 0.0 };

            xs[i] = x;
            prefix_w[i + 1] = prefix_w[i] + w;
            prefix_wx[i + 1] = prefix_wx[i] + w * x;
        }

        let range_mean = |i0: usize, i1: usize, fallback: f64| -> f64 {
            let w = prefix_w[i1] - prefix_w[i0];
            if w <= 0.0 {
                return fallback;
            }
            (prefix_wx[i1] - prefix_wx[i0]) / w
        };

        let mut cuts = vec![0usize; self.k + 1];
        cuts[self.k] = ngrid;
        let total_w = *prefix_w.last().unwrap();

        for i in 1..self.k {
            let target = total_w * i as f64 / self.k as f64;
            cuts[i] = match prefix_w.binary_search_by(|v| v.partial_cmp(&target).unwrap()) {
                Ok(idx) => idx.min(ngrid),
                Err(idx) => idx.min(ngrid),
            };
        }

        let mut centroids_d: Vec<f64> = (0..self.k)
            .map(|i| {
                let left = -1.0 + 2.0 * i as f64 / self.k as f64;
                let right = -1.0 + 2.0 * (i + 1) as f64 / self.k as f64;
                range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right))
            })
            .collect();

        let mut boundaries_d = vec![0.0f64; self.k - 1];
        for _ in 0..100 {
            for i in 0..self.k - 1 {
                boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
            }

            cuts[0] = 0;
            cuts[self.k] = ngrid;
            for i in 1..self.k {
                cuts[i] = match xs.binary_search_by(|v| v.partial_cmp(&boundaries_d[i - 1]).unwrap()) {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };
            }

            let mut max_delta = 0.0f64;
            for i in 0..self.k {
                let left = if i == 0 { -1.0 } else { boundaries_d[i - 1] };
                let right = if i + 1 == self.k { 1.0 } else { boundaries_d[i] };
                let c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
                let c = c.min(right).max(left);
                max_delta = max_delta.max((c - centroids_d[i]).abs());
                centroids_d[i] = c;
            }

            if max_delta < 1e-8 {
                break;
            }
        }

        centroids_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.centroids = centroids_d.iter().map(|&c| c as f32).collect();
    }

    pub fn code_size(&self) -> usize {
        (self.d * self.nbits + 7) / 8
    }

    pub fn select_index(&self, x: f32) -> u8 {
        match self.boundaries.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
            Ok(idx) => idx as u8,
            Err(idx) => idx as u8,
        }
    }

    pub fn encode_index(&self, idx: u8, code: &mut [u8], i: usize) {
        let bit_offset = i * self.nbits;
        let byte_offset = bit_offset >> 3;
        let bit_shift = bit_offset & 7;
        let mask = ((1u16 << self.nbits) - 1) as u16;
        let packed = ((idx as u16) & mask) << bit_shift;
        code[byte_offset] |= (packed & 0xff) as u8;
        if bit_shift + self.nbits > 8 {
            code[byte_offset + 1] |= (packed >> 8) as u8;
        }
    }

    pub fn decode_index(&self, code: &[u8], i: usize) -> u8 {
        let bit_offset = i * self.nbits;
        let byte_offset = bit_offset >> 3;
        let bit_shift = bit_offset & 7;
        let mask = ((1u16 << self.nbits) - 1) as u16;

        let mut packed = code[byte_offset] as u16;
        if bit_shift + self.nbits > 8 {
            packed |= (code[byte_offset + 1] as u16) << 8;
        }
        ((packed >> bit_shift) & mask) as u8
    }

    pub fn encode(&self, x: &[f32], code: &mut [u8]) {
        code.fill(0);
        for i in 0..self.d {
            let idx = self.select_index(x[i]);
            self.encode_index(idx, code, i);
        }
    }

    pub fn decode(&self, code: &[u8], x: &mut [f32]) {
        for i in 0..self.d {
            let idx = self.decode_index(code, i);
            x[i] = self.centroids[idx as usize];
        }
    }

    pub fn compute_distance(&self, code: &[u8], query: &[f32]) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..self.d {
            let idx = self.decode_index(code, i);
            let diff = self.centroids[idx as usize] - query[i];
            dist += diff * diff;
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lloyd_max_4bit() {
        let q = LloydMaxQuantizer::new(128, 4);
        assert_eq!(q.k, 16);
        assert_eq!(q.centroids.len(), 16);
        assert_eq!(q.boundaries.len(), 15);
        assert!(q.centroids.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let q = LloydMaxQuantizer::new(128, 4);
        let x = vec![0.5f32; 128];
        let mut code = vec![0u8; q.code_size()];
        q.encode(&x, &mut code);
        let mut decoded = vec![0.0f32; 128];
        q.decode(&code, &mut decoded);
        for i in 0..128 {
            let idx = q.select_index(x[i]);
            assert!((decoded[i] - q.centroids[idx as usize]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_code_size() {
        let q4 = LloydMaxQuantizer::new(128, 4);
        assert_eq!(q4.code_size(), 64);
        let q6 = LloydMaxQuantizer::new(128, 6);
        assert_eq!(q6.code_size(), 96);
        let q8 = LloydMaxQuantizer::new(128, 8);
        assert_eq!(q8.code_size(), 128);
    }
}
