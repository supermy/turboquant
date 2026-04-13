//! 工具模块: 提供向量计算、数据生成、召回率评估等基础功能

use rand::Rng;
use rand::SeedableRng;
use rand_distr::Distribution;

/// 浮点数包装类型，用于支持 BinaryHeap 排序
///
/// Rust 的 BinaryHeap 需要元素实现 Ord trait，
/// 但 f32 不实现 Ord (因为有 NaN)，
/// 所以用这个包装类型来解决。
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

/// 计算两个向量的点积
///
/// # 参数
/// - `a`: 第一个向量
/// - `b`: 第二个向量
///
/// # 返回值
/// 点积值 Σ(a[i] * b[i])
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// 计算两个向量的 L2 距离平方
///
/// # 参数
/// - `a`: 第一个向量
/// - `b`: 第二个向量
///
/// # 返回值
/// L2 距离平方 Σ(a[i] - b[i])²
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// 计算向量的 L2 范数平方
///
/// # 参数
/// - `x`: 输入向量
///
/// # 返回值
/// L2 范数平方 Σ(x[i]²)
pub fn l2_norm_sq(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}

/// 计算向量的 L2 范数
///
/// # 参数
/// - `x`: 输入向量
///
/// # 返回值
/// L2 范数 √Σ(x[i]²)
pub fn l2_norm(x: &[f32]) -> f32 {
    l2_norm_sq(x).sqrt()
}

/// 将向量 L2 归一化到单位球面
///
/// # 参数
/// - `x`: 待归一化的向量 (原地修改)
///
/// # 说明
/// 归一化后向量 L2 范数为 1，适合余弦相似度计算
pub fn l2_normalize(x: &mut [f32]) {
    let norm = l2_norm(x);
    if norm > 1e-10 {
        for v in x.iter_mut() {
            *v /= norm;
        }
    }
}

/// 计算大于等于 n 的最小 2 的幂
///
/// # 参数
/// - `n`: 输入值
///
/// # 返回值
/// 大于等于 n 的最小 2 的幂
///
/// # 示例
/// ```
/// assert_eq!(next_power_of_2(100), 128);
/// assert_eq!(next_power_of_2(128), 128);
/// assert_eq!(next_power_of_2(129), 256);
/// ```
pub fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

/// 生成聚类数据 (模拟真实数据集)
///
/// 生成具有聚类结构的数据，模拟 SIFT1M、GIST1M 等真实数据集的分布。
/// 每个聚类中心随机生成，数据点围绕中心添加高斯噪声。
///
/// # 参数
/// - `n`: 数据点数量
/// - `d`: 向量维度
/// - `n_clusters`: 聚类数量
/// - `cluster_std`: 聚类内标准差 (控制数据分散程度)
/// - `seed`: 随机种子 (保证可复现)
///
/// # 返回值
/// 生成的数据向量 (n * d 个 f32)
///
/// # 说明
/// 生成的数据会自动归一化到单位球面
pub fn generate_clustered_data(
    n: usize,
    d: usize,
    n_clusters: usize,
    cluster_std: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];

    // 生成聚类中心
    let mut centroids = vec![0.0f32; n_clusters * d];
    for c in 0..n_clusters {
        // 随机生成中心点
        for j in 0..d {
            centroids[c * d + j] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        // 归一化中心点
        let norm = l2_norm(&centroids[c * d..(c + 1) * d]);
        for j in 0..d {
            centroids[c * d + j] /= norm;
        }
    }

    // 围绕中心生成数据点
    let normal = rand_distr::Normal::new(0.0, cluster_std as f64).unwrap();
    for i in 0..n {
        let cluster_id = i % n_clusters;
        let center = &centroids[cluster_id * d..(cluster_id + 1) * d];

        // 添加高斯噪声
        for j in 0..d {
            data[i * d + j] = center[j] + normal.sample(&mut rng) as f32;
        }

        // 归一化到单位球面
        let norm = l2_norm(&data[i * d..(i + 1) * d]);
        for j in 0..d {
            data[i * d + j] /= norm;
        }
    }

    data
}

/// 生成查询向量
///
/// 从数据集中随机选择点并添加噪声，模拟真实查询场景。
///
/// # 参数
/// - `data`: 原始数据集
/// - `n_data`: 数据集大小
/// - `n_queries`: 查询数量
/// - `d`: 向量维度
/// - `noise_level`: 噪声水平
/// - `seed`: 随机种子
///
/// # 返回值
/// 生成的查询向量 (n_queries * d 个 f32)
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
        // 随机选择一个数据点作为查询基础
        let src_idx = rng.gen_range(0..n_data);

        // 添加噪声
        for j in 0..d {
            queries[i * d + j] = data[src_idx * d + j] + normal.sample(&mut rng) as f32;
        }

        // 归一化
        let norm = l2_norm(&queries[i * d..(i + 1) * d]);
        for j in 0..d {
            queries[i * d + j] /= norm;
        }
    }

    queries
}

/// 计算真实最近邻 (暴力搜索)
///
/// 使用暴力搜索计算每个查询的 k 个最近邻，作为召回率评估的基准。
///
/// # 参数
/// - `data`: 数据集
/// - `queries`: 查询向量
/// - `n_data`: 数据集大小
/// - `n_queries`: 查询数量
/// - `d`: 向量维度
/// - `k`: 返回的最近邻数量
///
/// # 返回值
/// 每个查询的 k 个最近邻索引
pub fn compute_ground_truth(data: &[f32], queries: &[f32], n_data: usize, n_queries: usize, d: usize, k: usize) -> Vec<Vec<usize>> {
    let mut gt = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let query = &queries[q * d..(q + 1) * d];
        // 计算所有距离
        let mut dists: Vec<(f32, usize)> = (0..n_data)
            .map(|i| (l2_distance(query, &data[i * d..(i + 1) * d]), i))
            .collect();
        // 快速选择前 k 个
        dists.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        gt.push(dists.iter().map(|&(_, i)| i).collect());
    }
    gt
}

/// 计算召回率
///
/// 计算搜索结果与真实最近邻的重叠比例。
///
/// # 参数
/// - `result_ids`: 搜索结果 (每个查询返回的 k 个索引)
/// - `gt_ids`: 真实最近邻
/// - `nq`: 查询数量
/// - `k`: 返回数量
///
/// # 返回值
/// 召回率 (0.0 ~ 1.0)
///
/// # 公式
/// recall = 正确命中的数量 / (nq * k)
pub fn compute_recall(result_ids: &[Vec<usize>], gt_ids: &[Vec<usize>], nq: usize, k: usize) -> f32 {
    let mut total_recall = 0usize;
    for q in 0..nq {
        // 排序后用二分查找加速
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
