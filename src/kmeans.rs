//! KMeans 聚类算法
//!
//! 实现 KMeans 聚类用于 IVF 索引的向量分区。

use crate::utils::l2_distance;
use rand::Rng;
use rand::SeedableRng;

/// KMeans 聚类器
///
/// 将向量聚类到 k 个簇中，用于 IVF 索引。
pub struct KMeans {
    /// 向量维度
    pub d: usize,
    /// 聚类数量
    pub k: usize,
    /// 迭代次数
    pub niter: usize,
    /// 聚类中心
    pub centroids: Vec<f32>,
}

impl KMeans {
    /// 创建新的 KMeans 聚类器
    ///
    /// # 参数
    /// - `d`: 向量维度
    /// - `k`: 聚类数量
    /// - `niter`: 迭代次数
    pub fn new(d: usize, k: usize, niter: usize) -> Self {
        Self {
            d,
            k,
            niter,
            centroids: vec![0.0; k * d],
        }
    }

    /// 训练聚类器
    ///
    /// 使用 KMeans++ 初始化和迭代优化。
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `n`: 数据点数量
    /// - `seed`: 随机种子
    pub fn train(&mut self, data: &[f32], n: usize, seed: u64) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        // KMeans++ 初始化: 随机打乱后选择前 k 个点
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // 初始化聚类中心
        let init_count = self.k.min(n);
        for i in 0..init_count {
            self.centroids[i * self.d..(i + 1) * self.d]
                .copy_from_slice(&data[indices[i] * self.d..(indices[i] + 1) * self.d]);
        }

        // 迭代优化
        let mut assign = vec![0usize; n];
        let mut counts = vec![0usize; self.k];
        let mut new_centroids = vec![0.0f32; self.k * self.d];

        for _ in 0..self.niter {
            counts.fill(0);
            new_centroids.fill(0.0);

            // 分配点到最近的聚类
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

            // 更新聚类中心
            for i in 0..self.k {
                if counts[i] > 0 {
                    for j in 0..self.d {
                        self.centroids[i * self.d + j] = new_centroids[i * self.d + j] / counts[i] as f32;
                    }
                }
            }
        }
    }

    /// 分配向量到最近的聚类
    ///
    /// # 参数
    /// - `x`: 输入向量
    ///
    /// # 返回值
    /// 最近聚类的索引
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

    /// 获取最近的 nprobe 个聚类
    ///
    /// # 参数
    /// - `x`: 输入向量
    /// - `nprobe`: 返回的聚类数量
    ///
    /// # 返回值
    /// (距离, 聚类索引) 列表，按距离排序
    pub fn nearest_clusters(&self, x: &[f32], nprobe: usize) -> Vec<(f32, usize)> {
        let mut buf = Vec::with_capacity(self.k);
        self.nearest_clusters_into(x, nprobe, &mut buf);
        buf
    }

    pub fn nearest_clusters_into(&self, x: &[f32], nprobe: usize, buf: &mut Vec<(f32, usize)>) {
        let nprobe = nprobe.min(self.k);
        buf.clear();
        buf.extend((0..self.k)
            .map(|i| (l2_distance(x, &self.centroids[i * self.d..(i + 1) * self.d]), i)));
        if nprobe < self.k {
            buf.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        buf.truncate(nprobe);
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, Normal};

    /// 测试 KMeans 收敛
    #[test]
    fn test_kmeans_convergence() {
        let d = 32;
        let k = 4;
        let n = 1000;

        // 生成有明显聚类结构的数据
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

        // 测试分配
        let test_x = &data[0..d];
        let cluster = kmeans.assign_cluster(test_x);
        assert!(cluster < k);
    }
}
