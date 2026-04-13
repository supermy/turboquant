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

pub fn l2_norm_sq(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}

pub fn l2_norm(x: &[f32]) -> f32 {
    l2_norm_sq(x).sqrt()
}

pub fn l2_normalize(x: &mut [f32]) {
    let norm = l2_norm(x);
    if norm > 1e-10 {
        for v in x.iter_mut() {
            *v /= norm;
        }
    }
}

pub fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
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
            .map(|i| (l2_distance(query, &data[i * d..(i + 1) * d]), i))
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
