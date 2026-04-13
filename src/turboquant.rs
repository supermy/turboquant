use std::collections::BinaryHeap;

use crate::hadamard::HadamardRotation;
use crate::lloyd_max::LloydMaxQuantizer;
use crate::sq8::SQ8Quantizer;
use crate::utils::{l2_normalize, next_power_of_2, FloatOrd};

pub struct TurboQuantFlatIndex {
    pub d: usize,
    pub nbits: usize,
    rotation: HadamardRotation,
    quantizer: LloydMaxQuantizer,
    codes: Vec<u8>,
    sq8: Option<SQ8Quantizer>,
    sq8_codes: Vec<u8>,
    ntotal: usize,
}

impl TurboQuantFlatIndex {
    pub fn new(d: usize, nbits: usize, use_sq8: bool) -> Self {
        let d_rotated = next_power_of_2(d);
        let rotation = HadamardRotation::new(d, 12345);
        let quantizer = LloydMaxQuantizer::new(d_rotated, nbits);
        let sq8 = if use_sq8 { Some(SQ8Quantizer::new(d)) } else { None };

        Self {
            d,
            nbits,
            rotation,
            quantizer,
            codes: Vec::new(),
            sq8,
            sq8_codes: Vec::new(),
            ntotal: 0,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        if let Some(ref mut sq8) = self.sq8 {
            sq8.train(data, n);
        }
    }

    pub fn add(&mut self, data: &[f32], n: usize) {
        let mut x_normalized = data.to_vec();
        for i in 0..n {
            l2_normalize(&mut x_normalized[i * self.d..(i + 1) * self.d]);
        }

        let x_rotated = self.rotation.apply_batch(n, &x_normalized);

        let code_sz = self.quantizer.code_size();
        self.codes.resize((self.ntotal + n) * code_sz, 0);

        for i in 0..n {
            let xi = &x_rotated[i * self.rotation.d_out..(i + 1) * self.rotation.d_out];
            self.quantizer.encode(xi, &mut self.codes[(self.ntotal + i) * code_sz..(self.ntotal + i + 1) * code_sz]);
        }

        if let Some(ref sq8) = self.sq8 {
            let sq8_sz = sq8.code_size();
            self.sq8_codes.resize((self.ntotal + n) * sq8_sz, 0);
            for i in 0..n {
                let xi = &data[i * self.d..(i + 1) * self.d];
                sq8.encode(xi, &mut self.sq8_codes[(self.ntotal + i) * sq8_sz..(self.ntotal + i + 1) * sq8_sz]);
            }
        }

        self.ntotal += n;
    }

    pub fn search(&self, queries: &[f32], n: usize, k: usize, refine_factor: usize) -> Vec<Vec<(usize, f32)>> {
        let mut x_normalized = queries.to_vec();
        for i in 0..n {
            l2_normalize(&mut x_normalized[i * self.d..(i + 1) * self.d]);
        }

        let x_rotated = self.rotation.apply_batch(n, &x_normalized);
        let code_sz = self.quantizer.code_size();

        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &x_rotated[q * self.rotation.d_out..(q + 1) * self.rotation.d_out];

            let k1 = if self.sq8.is_some() {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for i in 0..self.ntotal {
                let code = &self.codes[i * code_sz..(i + 1) * code_sz];
                let dist = self.quantizer.compute_distance(code, query);

                if heap.len() < k1 {
                    heap.push((FloatOrd(dist), i));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((FloatOrd(dist), i));
                }
            }

            let candidates: Vec<(f32, usize)> = heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

            let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

            if let Some(ref sq8) = self.sq8 {
                let sq8_sz = sq8.code_size();
                let orig_query = &queries[q * self.d..(q + 1) * self.d];
                for (_, idx) in &candidates {
                    let sq8_code = &self.sq8_codes[*idx * sq8_sz..(*idx + 1) * sq8_sz];
                    let refined_dist = sq8.compute_distance(sq8_code, orig_query);

                    if final_heap.len() < k {
                        final_heap.push((FloatOrd(refined_dist), *idx));
                    } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                        final_heap.pop();
                        final_heap.push((FloatOrd(refined_dist), *idx));
                    }
                }
            } else {
                for (dist, idx) in candidates {
                    if final_heap.len() < k {
                        final_heap.push((FloatOrd(dist), idx));
                    } else if dist < final_heap.peek().unwrap().0 .0 {
                        final_heap.pop();
                        final_heap.push((FloatOrd(dist), idx));
                    }
                }
            }

            let mut result: Vec<(usize, f32)> = final_heap
                .into_iter()
                .map(|(FloatOrd(d), i)| (i, d))
                .collect();
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results.push(result);
        }

        results
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    pub fn code_size(&self) -> usize {
        self.quantizer.code_size()
    }

    pub fn total_storage(&self) -> usize {
        let base = self.ntotal * self.quantizer.code_size();
        let sq8_storage = if self.sq8.is_some() {
            self.ntotal * self.d
        } else {
            0
        };
        base + sq8_storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    #[test]
    fn test_turboquant_4bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 4, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 4-bit Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.5, "TurboQuant 4-bit recall too low: {}", recall);
    }

    #[test]
    fn test_turboquant_4bit_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 4-bit + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.9, "TurboQuant 4-bit + SQ8 recall too low: {}", recall);
    }

    #[test]
    fn test_turboquant_6bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = TurboQuantFlatIndex::new(d, 6, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("TurboQuant 6-bit Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.8, "TurboQuant 6-bit recall too low: {}", recall);
    }
}
