use std::collections::BinaryHeap;

use crate::sq8::SQ8Quantizer;
use crate::utils::{l2_distance, l2_norm_sq, FloatOrd};

#[derive(Clone, Debug)]
pub struct SignBitFactors {
    pub or_minus_c_l2sqr: f32,
    pub dp_multiplier: f32,
}

#[derive(Clone, Debug)]
pub struct QueryFactorsData {
    pub c1: f32,
    pub c34: f32,
    pub qr_to_c_l2sqr: f32,
    pub rotated_q: Vec<f32>,
}

pub struct RaBitQCodec {
    pub d: usize,
    pub nb_bits: usize,
    pub is_inner_product: bool,
}

impl RaBitQCodec {
    pub fn new(d: usize, nb_bits: usize, is_inner_product: bool) -> Self {
        Self {
            d,
            nb_bits,
            is_inner_product,
        }
    }

    pub fn code_size(&self) -> usize {
        let base_size = (self.d + 7) / 8;
        let factor_size = std::mem::size_of::<SignBitFactors>();
        base_size + factor_size
    }

    fn compute_vector_intermediate_values(
        &self,
        x: &[f32],
        centroid: Option<&[f32]>,
    ) -> (f32, f32, f32) {
        let mut norm_l2sqr = 0.0f32;
        let mut or_l2sqr = 0.0f32;
        let mut dp_oo = 0.0f32;

        for j in 0..self.d {
            let x_val = x[j];
            let centroid_val = centroid.map_or(0.0, |c| c[j]);
            let or_minus_c = x_val - centroid_val;

            norm_l2sqr += or_minus_c * or_minus_c;
            or_l2sqr += x_val * x_val;

            if or_minus_c > 0.0 {
                dp_oo += or_minus_c;
            } else {
                dp_oo -= or_minus_c;
            }
        }

        (norm_l2sqr, or_l2sqr, dp_oo)
    }

    pub fn encode(&self, x: &[f32], centroid: Option<&[f32]>, code: &mut [u8]) {
        code.fill(0);

        let (norm_l2sqr, or_l2sqr, dp_oo) = self.compute_vector_intermediate_values(x, centroid);

        for i in 0..self.d {
            let or_minus_c = x[i] - centroid.map_or(0.0, |c| c[i]);
            if or_minus_c > 0.0 {
                code[i / 8] |= 1 << (i % 8);
            }
        }

        let sqrt_norm_l2 = norm_l2sqr.sqrt();
        let inv_d_sqrt = 1.0 / (self.d as f32).sqrt();
        let inv_norm_l2 = if norm_l2sqr < 1e-10 { 1.0 } else { 1.0 / sqrt_norm_l2 };

        let normalized_dp = dp_oo * inv_norm_l2 * inv_d_sqrt;
        let inv_dp_oo = if normalized_dp.abs() < 1e-10 { 1.0 } else { 1.0 / normalized_dp };

        let factors = SignBitFactors {
            or_minus_c_l2sqr: if self.is_inner_product {
                norm_l2sqr - or_l2sqr
            } else {
                norm_l2sqr
            },
            dp_multiplier: inv_dp_oo * sqrt_norm_l2,
        };

        let base_size = (self.d + 7) / 8;
        let factors_bytes = unsafe {
            std::slice::from_raw_parts(
                &factors as *const SignBitFactors as *const u8,
                std::mem::size_of::<SignBitFactors>(),
            )
        };
        code[base_size..base_size + std::mem::size_of::<SignBitFactors>()].copy_from_slice(factors_bytes);
    }

    pub fn compute_distance(
        &self,
        code: &[u8],
        query_fac: &QueryFactorsData,
    ) -> f32 {
        let base_size = (self.d + 7) / 8;
        let factors = unsafe {
            let ptr = code[base_size..].as_ptr() as *const SignBitFactors;
            &*ptr
        };

        let mut dot_qo = 0.0f32;
        for i in 0..self.d {
            let bit = (code[i / 8] >> (i % 8)) & 1;
            if bit != 0 {
                dot_qo += query_fac.rotated_q[i];
            }
        }

        let final_dot = query_fac.c1 * dot_qo - query_fac.c34;

        let dist = factors.or_minus_c_l2sqr + query_fac.qr_to_c_l2sqr
            - 2.0 * factors.dp_multiplier * final_dot;

        if self.is_inner_product {
            -0.5 * (dist - query_fac.qr_to_c_l2sqr)
        } else {
            dist.max(0.0)
        }
    }
}

pub fn compute_query_factors(
    query: &[f32],
    d: usize,
    centroid: Option<&[f32]>,
    _is_inner_product: bool,
) -> QueryFactorsData {
    let qr_to_c_l2sqr = match centroid {
        Some(c) => l2_distance(query, c),
        None => l2_norm_sq(query),
    };

    let rotated_q: Vec<f32> = match centroid {
        Some(c) => query.iter().zip(c.iter()).map(|(&q, &c)| q - c).collect(),
        None => query.to_vec(),
    };

    let inv_d = 1.0 / (d as f32).sqrt();
    let sum_q: f32 = rotated_q.iter().sum();

    QueryFactorsData {
        c1: 2.0 * inv_d,
        c34: sum_q * inv_d,
        qr_to_c_l2sqr,
        rotated_q,
    }
}

pub struct RaBitQFlatIndex {
    pub d: usize,
    pub nb_bits: usize,
    pub is_inner_product: bool,
    centroid: Vec<f32>,
    codec: RaBitQCodec,
    codes: Vec<u8>,
    sq8: Option<SQ8Quantizer>,
    sq8_codes: Vec<u8>,
    ntotal: usize,
}

impl RaBitQFlatIndex {
    pub fn new(d: usize, nb_bits: usize, is_inner_product: bool, use_sq8: bool) -> Self {
        let codec = RaBitQCodec::new(d, nb_bits, is_inner_product);
        let sq8 = if use_sq8 { Some(SQ8Quantizer::new(d)) } else { None };

        Self {
            d,
            nb_bits,
            is_inner_product,
            centroid: vec![0.0; d],
            codec,
            codes: Vec::new(),
            sq8,
            sq8_codes: Vec::new(),
            ntotal: 0,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.centroid.fill(0.0);
        for i in 0..n {
            for j in 0..self.d {
                self.centroid[j] += data[i * self.d + j];
            }
        }
        for j in 0..self.d {
            self.centroid[j] /= n as f32;
        }

        if let Some(ref mut sq8) = self.sq8 {
            sq8.train(data, n);
        }
    }

    pub fn add(&mut self, data: &[f32], n: usize) {
        let code_sz = self.codec.code_size();
        self.codes.resize((self.ntotal + n) * code_sz, 0);

        if let Some(ref sq8) = self.sq8 {
            let sq8_sz = sq8.code_size();
            self.sq8_codes.resize((self.ntotal + n) * sq8_sz, 0);
        }

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];

            self.codec.encode(xi, Some(&self.centroid), &mut self.codes[(self.ntotal + i) * code_sz..(self.ntotal + i + 1) * code_sz]);

            if let Some(ref sq8) = self.sq8 {
                let sq8_sz = sq8.code_size();
                sq8.encode(xi, &mut self.sq8_codes[(self.ntotal + i) * sq8_sz..(self.ntotal + i + 1) * sq8_sz]);
            }
        }

        self.ntotal += n;
    }

    pub fn search(&self, queries: &[f32], n: usize, k: usize, refine_factor: usize) -> Vec<Vec<(usize, f32)>> {
        let code_sz = self.codec.code_size();
        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &queries[q * self.d..(q + 1) * self.d];
            let query_fac = compute_query_factors(query, self.d, Some(&self.centroid), self.is_inner_product);

            let k1 = if self.sq8.is_some() {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for i in 0..self.ntotal {
                let code = &self.codes[i * code_sz..(i + 1) * code_sz];
                let dist = self.codec.compute_distance(code, &query_fac);

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
                for (_, idx) in &candidates {
                    let sq8_code = &self.sq8_codes[*idx * sq8_sz..(*idx + 1) * sq8_sz];
                    let refined_dist = sq8.compute_distance(sq8_code, query);

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
        self.codec.code_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    #[test]
    fn test_rabitq_flat_1bit_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQFlatIndex::new(d, 1, false, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 1);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ Flat 1-bit Recall@{}: {:.4}", k, recall);
    }

    #[test]
    fn test_rabitq_flat_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ Flat 1-bit + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.85, "RaBitQ Flat + SQ8 recall too low: {}", recall);
    }
}
