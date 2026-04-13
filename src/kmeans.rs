use crate::utils::l2_distance;
use rand::Rng;
use rand::SeedableRng;

pub struct KMeans {
    pub d: usize,
    pub k: usize,
    pub niter: usize,
    pub centroids: Vec<f32>,
}

impl KMeans {
    pub fn new(d: usize, k: usize, niter: usize) -> Self {
        Self {
            d,
            k,
            niter,
            centroids: vec![0.0; k * d],
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize, seed: u64) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let init_count = self.k.min(n);
        for i in 0..init_count {
            self.centroids[i * self.d..(i + 1) * self.d]
                .copy_from_slice(&data[indices[i] * self.d..(indices[i] + 1) * self.d]);
        }

        let mut assign = vec![0usize; n];
        let mut counts = vec![0usize; self.k];
        let mut new_centroids = vec![0.0f32; self.k * self.d];

        for _ in 0..self.niter {
            counts.fill(0);
            new_centroids.fill(0.0);

            for i in 0..n {
                let xi = &data[i * self.d..(i + 1) * self.d];

                let mut min_dist = f32::MAX;
                let mut min_idx = 0;

                for j in 0..self.k {
                    let dist = l2_distance(xi, &self.centroids[j * self.d..(j + 1) * self.d]);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = j;
                    }
                }

                assign[i] = min_idx;
                counts[min_idx] += 1;

                for j in 0..self.d {
                    new_centroids[min_idx * self.d + j] += xi[j];
                }
            }

            for i in 0..self.k {
                if counts[i] > 0 {
                    for j in 0..self.d {
                        self.centroids[i * self.d + j] = new_centroids[i * self.d + j] / counts[i] as f32;
                    }
                }
            }
        }
    }

    pub fn assign_cluster(&self, x: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;

        for i in 0..self.k {
            let dist = l2_distance(x, &self.centroids[i * self.d..(i + 1) * self.d]);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        min_idx
    }

    pub fn nearest_clusters(&self, x: &[f32], nprobe: usize) -> Vec<(f32, usize)> {
        let nprobe = nprobe.min(self.k);
        let mut dists: Vec<(f32, usize)> = (0..self.k)
            .map(|i| (l2_distance(x, &self.centroids[i * self.d..(i + 1) * self.d]), i))
            .collect();
        if nprobe < self.k {
            dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        dists.truncate(nprobe);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_kmeans_convergence() {
        let d = 32;
        let k = 4;
        let n = 1000;

        let mut data = vec![0.0f32; n * d];
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.1).unwrap();

        for i in 0..n {
            let cluster = i % k;
            let center_offset = cluster as f32 * 10.0;
            for j in 0..d {
                data[i * d + j] = center_offset + normal.sample(&mut rng) as f32;
            }
        }

        let mut kmeans = KMeans::new(d, k, 20);
        kmeans.train(&data, n, 42);

        let test_x = &data[0..d];
        let cluster = kmeans.assign_cluster(test_x);
        assert!(cluster < k);
    }
}
