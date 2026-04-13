use std::collections::{BinaryHeap, HashMap};

use crate::kmeans::KMeans;
use crate::rabitq::{RaBitQCodec, compute_query_factors};
use crate::sq8::SQ8Quantizer;
use crate::utils::FloatOrd;

struct ClusterData {
    codes: Vec<u8>,
    ids: Vec<usize>,
    sq8_codes: Vec<u8>,
}

pub struct RaBitQIVFIndex {
    pub d: usize,
    pub nlist: usize,
    pub nb_bits: usize,
    pub is_inner_product: bool,

    kmeans: KMeans,
    codecs: Vec<RaBitQCodec>,
    cluster_centroids: Vec<Vec<f32>>,
    clusters: Vec<ClusterData>,
    sq8_quantizers: Vec<Option<SQ8Quantizer>>,
    ntotal: usize,
}

impl RaBitQIVFIndex {
    pub fn new(d: usize, nlist: usize, nb_bits: usize, is_inner_product: bool, use_sq8: bool) -> Self {
        let kmeans = KMeans::new(d, nlist, 20);
        let codecs: Vec<RaBitQCodec> = (0..nlist)
            .map(|_| RaBitQCodec::new(d, nb_bits, is_inner_product))
            .collect();
        let cluster_centroids = vec![vec![0.0; d]; nlist];
        let clusters = (0..nlist)
            .map(|_| ClusterData {
                codes: Vec::new(),
                ids: Vec::new(),
                sq8_codes: Vec::new(),
            })
            .collect();
        let sq8_quantizers: Vec<Option<SQ8Quantizer>> = if use_sq8 {
            (0..nlist).map(|_| Some(SQ8Quantizer::new(d))).collect()
        } else {
            (0..nlist).map(|_| None).collect()
        };

        Self {
            d,
            nlist,
            nb_bits,
            is_inner_product,
            kmeans,
            codecs,
            cluster_centroids,
            clusters,
            sq8_quantizers,
            ntotal: 0,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.kmeans.train(data, n, 42);

        for i in 0..self.nlist {
            self.cluster_centroids[i].copy_from_slice(
                &self.kmeans.centroids[i * self.d..(i + 1) * self.d],
            );
        }

        let use_sq8 = self.sq8_quantizers[0].is_some();
        if use_sq8 {
            let mut cluster_data: Vec<Vec<f32>> = vec![Vec::new(); self.nlist];
            for i in 0..n {
                let cluster_id = self.kmeans.assign_cluster(&data[i * self.d..(i + 1) * self.d]);
                cluster_data[cluster_id].extend_from_slice(&data[i * self.d..(i + 1) * self.d]);
            }

            for c in 0..self.nlist {
                let n_in_cluster = cluster_data[c].len() / self.d;
                if n_in_cluster > 0 {
                    if let Some(ref mut sq8) = self.sq8_quantizers[c] {
                        sq8.train(&cluster_data[c], n_in_cluster);
                    }
                }
            }
        }
    }

    pub fn add(&mut self, data: &[f32], n: usize) {
        let code_sz = self.codecs[0].code_size();

        for i in 0..n {
            let xi = &data[i * self.d..(i + 1) * self.d];
            let cluster_id = self.kmeans.assign_cluster(xi);

            let mut code = vec![0u8; code_sz];
            self.codecs[cluster_id].encode(xi, Some(&self.cluster_centroids[cluster_id]), &mut code);
            self.clusters[cluster_id].codes.extend_from_slice(&code);

            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                let mut sq8_code = vec![0u8; sq8.code_size()];
                sq8.encode(xi, &mut sq8_code);
                self.clusters[cluster_id].sq8_codes.extend_from_slice(&sq8_code);
            }

            self.clusters[cluster_id].ids.push(self.ntotal + i);
        }

        self.ntotal += n;
    }

    pub fn search(
        &self,
        queries: &[f32],
        n: usize,
        k: usize,
        nprobe: usize,
        refine_factor: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let code_sz = self.codecs[0].code_size();
        let use_sq8 = self.sq8_quantizers[0].is_some();
        let mut results = Vec::with_capacity(n);

        for q in 0..n {
            let query = &queries[q * self.d..(q + 1) * self.d];

            let nearest = self.kmeans.nearest_clusters(query, nprobe);

            let k1 = if use_sq8 {
                (k * refine_factor).min(self.ntotal)
            } else {
                k
            };

            let mut heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k1);

            for (_, cluster_id) in &nearest {
                let cluster = &self.clusters[*cluster_id];
                let n_vectors = cluster.ids.len();
                if n_vectors == 0 {
                    continue;
                }

                let query_fac = compute_query_factors(
                    query,
                    self.d,
                    Some(&self.cluster_centroids[*cluster_id]),
                    self.is_inner_product,
                );

                for v in 0..n_vectors {
                    let code = &cluster.codes[v * code_sz..(v + 1) * code_sz];
                    let dist = self.codecs[*cluster_id].compute_distance(code, &query_fac);

                    if heap.len() < k1 {
                        heap.push((FloatOrd(dist), cluster.ids[v]));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((FloatOrd(dist), cluster.ids[v]));
                    }
                }
            }

            let candidates: Vec<(f32, usize)> = heap.into_iter().map(|(FloatOrd(d), i)| (d, i)).collect();

            let mut final_heap: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(k);

            if use_sq8 {
                let id_to_cluster: HashMap<usize, usize> = self
                    .clusters
                    .iter()
                    .enumerate()
                    .flat_map(|(c, cluster)| cluster.ids.iter().map(move |&id| (id, c)))
                    .collect();

                for (_, idx) in &candidates {
                    if let Some(&cluster_id) = id_to_cluster.get(idx) {
                        let cluster = &self.clusters[cluster_id];
                        let pos = cluster.ids.iter().position(|&id| id == *idx);
                        if let Some(pos) = pos {
                            if let Some(ref sq8) = self.sq8_quantizers[cluster_id] {
                                let sq8_sz = sq8.code_size();
                                let sq8_code = &cluster.sq8_codes[pos * sq8_sz..(pos + 1) * sq8_sz];
                                let refined_dist = sq8.compute_distance(sq8_code, query);

                                if final_heap.len() < k {
                                    final_heap.push((FloatOrd(refined_dist), *idx));
                                } else if refined_dist < final_heap.peek().unwrap().0 .0 {
                                    final_heap.pop();
                                    final_heap.push((FloatOrd(refined_dist), *idx));
                                }
                            }
                        }
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
        self.codecs[0].code_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{compute_recall, compute_ground_truth, generate_clustered_data, generate_queries};

    #[test]
    fn test_rabitq_ivf_sq8_recall() {
        let d = 128;
        let nb = 10000;
        let nq = 100;
        let k = 10;
        let nlist = 64;

        let data = generate_clustered_data(nb, d, 100, 0.1, 42);
        let queries = generate_queries(&data, nb, nq, d, 0.05, 123);
        let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

        let mut index = RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let results = index.search(&queries, nq, k, nlist.min(64), 10);
        let result_ids: Vec<Vec<usize>> = results.iter().map(|r| r.iter().map(|&(i, _)| i).collect()).collect();
        let recall = compute_recall(&result_ids, &gt, nq, k);

        println!("RaBitQ IVF + SQ8 Recall@{}: {:.4}", k, recall);
        assert!(recall > 0.85, "RaBitQ IVF + SQ8 recall too low: {}", recall);
    }
}
