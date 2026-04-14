// PolarQuant KV Cache 压缩演示
//
// 模拟 LLM 推理中的 KV Cache 压缩场景，展示 PolarQuant 的：
// 1. 压缩效率（压缩比、内存节省）
// 2. 重建质量（余弦相似度、注意力分数保持）
// 3. 不同维度/位宽下的性能对比
// 4. 长序列（10K tokens）的可扩展性

use polarquant::{CompressedVector, PolarQuant, PolarQuantConfig};
use rand::Rng;
use std::time::Instant;

/// 向量归一化到单位球面
/// 
/// # 参数
/// - `x`: 输入向量
/// 
/// # 返回值
/// 归一化后的向量（L2范数为1）
fn normalize_vector(x: &[f64]) -> Vec<f64> {
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return x.to_vec();
    }
    x.iter().map(|&v| v / norm).collect()
}

/// 向量点积
/// 
/// # 参数
/// - `a`: 第一个向量
/// - `b`: 第二个向量
/// 
/// # 返回值
/// 两个向量的点积值
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// 余弦相似度
/// 
/// # 参数
/// - `a`: 第一个向量
/// - `b`: 第二个向量
/// 
/// # 返回值
/// 余弦相似度值，范围[-1, 1]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// 模拟 KV Cache 结构
///
/// 支持：
/// - 原始模式：存储完整 f64 向量
/// - 压缩模式：使用 PolarQuant 压缩后存储
struct SimulatedKVCache {
    head_dim: usize,               // 注意力头维度
    use_compression: bool,         // 是否启用压缩
    quantizer: Option<PolarQuant>, // 量化器实例
    keys: Vec<Vec<f64>>,           // 原始 Key 向量（用于质量评估）
    values: Vec<Vec<f64>>,         // 原始 Value 向量
    compressed_keys: Vec<CompressedVector>,   // 压缩后的 Key
    compressed_values: Vec<CompressedVector>, // 压缩后的 Value
    total_original_bytes: usize,   // 原始数据总字节数
    total_compressed_bytes: usize, // 压缩后数据总字节数
}

impl SimulatedKVCache {
    /// 创建 KV Cache 实例
    /// 
    /// # 参数
    /// - `head_dim`: 注意力头维度
    /// - `use_compression`: 是否启用压缩
    fn new(head_dim: usize, use_compression: bool) -> Self {
        let quantizer = if use_compression {
            let config = PolarQuantConfig::builder(head_dim)
                .radius_bits(8)
                .angle_bits(4)
                .seed(42)
                .build()
                .unwrap();
            Some(PolarQuant::new(config).unwrap())
        } else {
            None
        };

        Self {
            head_dim,
            use_compression,
            quantizer,
            keys: Vec::new(),
            values: Vec::new(),
            compressed_keys: Vec::new(),
            compressed_values: Vec::new(),
            total_original_bytes: 0,
            total_compressed_bytes: 0,
        }
    }

    /// 追加一个 token 的 KV 对
    ///
    /// # 参数
    /// - `key`: Key 向量
    /// - `value`: Value 向量
    /// 
    /// # 返回值
    /// 压缩耗时（秒）
    fn append(&mut self, key: &[f64], value: &[f64]) -> f64 {
        let key_norm = normalize_vector(key);
        let value_norm = normalize_vector(value);

        // 保存原始向量（用于质量评估）
        self.keys.push(key_norm.clone());
        self.values.push(value_norm.clone());

        // 原始大小：2 个 head_dim 维 f64 向量
        self.total_original_bytes += 2 * self.head_dim * 8;

        if self.use_compression {
            let start = Instant::now();

            // 压缩 Key 和 Value
            let ck = self.quantizer.as_ref().unwrap().compress(&key_norm).unwrap();
            let cv = self.quantizer.as_ref().unwrap().compress(&value_norm).unwrap();

            let elapsed = start.elapsed().as_secs_f64();

            // 压缩后大小估算：radius(4B) + (d-1) * angle(4B)，每个 KV 对
            let compressed_size = 2 * (4 + (self.head_dim - 1) * 4);
            self.total_compressed_bytes += compressed_size;

            self.compressed_keys.push(ck);
            self.compressed_values.push(cv);

            elapsed
        } else {
            // 未压缩模式：与原始大小相同
            self.total_compressed_bytes += 2 * self.head_dim * 8;
            0.0
        }
    }

    /// 获取第 idx 个 Key 向量（压缩模式自动解压）
    /// 
    /// # 参数
    /// - `idx`: 索引
    fn get_key(&self, idx: usize) -> Vec<f64> {
        if self.use_compression {
            self.quantizer
                .as_ref()
                .unwrap()
                .decompress(&self.compressed_keys[idx])
        } else {
            self.keys[idx].clone()
        }
    }

    /// 获取第 idx 个 Value 向量（压缩模式自动解压）
    /// 
    /// # 参数
    /// - `idx`: 索引
    fn get_value(&self, idx: usize) -> Vec<f64> {
        if self.use_compression {
            self.quantizer
                .as_ref()
                .unwrap()
                .decompress(&self.compressed_values[idx])
        } else {
            self.values[idx].clone()
        }
    }

    /// 计算注意力分数（query 与所有 keys 的点积）
    /// 
    /// # 参数
    /// - `query`: 查询向量
    /// 
    /// # 返回值
    /// 注意力分数列表
    fn compute_attention_scores(&self, query: &[f64]) -> Vec<f64> {
        let query_norm = normalize_vector(query);
        (0..self.keys.len())
            .map(|i| {
                let key = self.get_key(i);
                dot(&query_norm, &key)
            })
            .collect()
    }

    /// 计算实际压缩比
    /// 
    /// # 返回值
    /// 压缩比 = 原始大小 / 压缩后大小
    fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            return 1.0;
        }
        self.total_original_bytes as f64 / self.total_compressed_bytes as f64
    }

    /// 计算节省空间百分比
    /// 
    /// # 返回值
    /// 节省空间百分比
    fn space_saved_pct(&self) -> f64 {
        let ratio = self.compression_ratio();
        if ratio <= 1.0 {
            return 0.0;
        }
        (1.0 - 1.0 / ratio) * 100.0
    }
}

/// 演示 1: KV Cache 基础压缩
///
/// 模拟 100 个 token 的 KV Cache，评估：
/// - 压缩耗时
/// - 内存节省
/// - 注意力分数保持度
/// - Top-K 检索准确率
fn demo_kv_cache_compression() {
    println!("{}", "=".repeat(70));
    println!("KV Cache 压缩基础演示");
    println!("{}", "=".repeat(70));

    let head_dim = 64;
    let seq_len = 100;

    println!("\n配置:");
    println!("  注意力头维度: {}", head_dim);
    println!("  序列长度: {}", seq_len);
    println!("  量化配置: 8-bit 半径, 4-bit 角度");

    let mut cache_compressed = SimulatedKVCache::new(head_dim, true);
    let mut cache_uncompressed = SimulatedKVCache::new(head_dim, false);

    let mut rng = rand::thread_rng();
    let mut total_compress_time = 0.0f64;

    println!("\n生成 {} 个 KV 对...", seq_len);

    for _ in 0..seq_len {
        let key: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let value: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        cache_uncompressed.append(&key, &value);
        total_compress_time += cache_compressed.append(&key, &value);
    }

    println!("压缩耗时: {:.2} ms", total_compress_time * 1000.0);
    println!(
        "平均每 token: {:.3} ms",
        total_compress_time / seq_len as f64 * 1000.0
    );

    println!("\n内存使用:");
    println!(
        "  原始大小: {:.2} KB",
        cache_compressed.total_original_bytes as f64 / 1024.0
    );
    println!(
        "  压缩后大小: {:.2} KB",
        cache_compressed.total_compressed_bytes as f64 / 1024.0
    );
    println!(
        "  实际压缩比: {:.2}x (u32 索引存储)",
        cache_compressed.compression_ratio()
    );
    let theoretical_ratio = cache_compressed.quantizer.as_ref().unwrap().compression_ratio();
    println!(
        "  理论压缩比 (bit-packed): {:.2}x",
        theoretical_ratio
    );
    println!("  节省空间: {:.1}%", cache_compressed.space_saved_pct());

    // 注意力分数质量评估
    println!("\n注意力分数质量:");
    let query: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

    let scores_uncompressed = cache_uncompressed.compute_attention_scores(&query);

    let start = Instant::now();
    let scores_compressed = cache_compressed.compute_attention_scores(&query);
    let decompress_time = start.elapsed().as_secs_f64();

    println!(
        "  解压耗时: {:.2} ms",
        decompress_time * 1000.0
    );

    // 计算 Pearson 相关系数
    let n = scores_uncompressed.len().min(scores_compressed.len());
    let mean_uncompressed: f64 = scores_uncompressed.iter().sum::<f64>() / n as f64;
    let mean_compressed: f64 = scores_compressed.iter().sum::<f64>() / n as f64;

    let cov: f64 = scores_uncompressed
        .iter()
        .zip(scores_compressed.iter())
        .map(|(&a, &b)| (a - mean_uncompressed) * (b - mean_compressed))
        .sum::<f64>()
        / n as f64;
    let var_u: f64 = scores_uncompressed
        .iter()
        .map(|a| (a - mean_uncompressed).powi(2))
        .sum::<f64>()
        / n as f64;
    let var_c: f64 = scores_compressed
        .iter()
        .map(|a| (a - mean_compressed).powi(2))
        .sum::<f64>()
        / n as f64;
    let correlation = cov / (var_u.sqrt() * var_c.sqrt());

    // 平均绝对误差
    let mae: f64 = scores_uncompressed
        .iter()
        .zip(scores_compressed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / n as f64;

    println!("  分数相关性 (Pearson): {:.4}", correlation);
    println!("  平均绝对误差: {:.6}", mae);

    // Top-5 检索准确率
    let mut indexed_scores: Vec<(usize, f64)> = scores_uncompressed
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let top_k_uncompressed: Vec<usize> = indexed_scores.iter().rev().take(5).map(|(i, _)| *i).collect();

    let mut indexed_scores_c: Vec<(usize, f64)> = scores_compressed
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed_scores_c.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let top_k_compressed: Vec<usize> = indexed_scores_c.iter().rev().take(5).map(|(i, _)| *i).collect();

    let top_k_overlap = top_k_uncompressed
        .iter()
        .filter(|i| top_k_compressed.contains(i))
        .count();
    let top_k_accuracy = top_k_overlap as f64 / 5.0;

    println!("  Top-5 检索准确率: {:.1}%", top_k_accuracy * 100.0);
}

/// 演示 2: 不同注意力头维度的压缩效果对比
///
/// 维度越高，Beta 分布越集中，量化精度越高
fn demo_attention_head_comparison() {
    println!("\n{}", "=".repeat(70));
    println!("注意力头维度对比");
    println!("{}", "=".repeat(70));

    let head_dims = [32, 64, 128, 256];
    let seq_len = 50;

    println!(
        "\n{:<12} {:<10} {:<10} {:<15}",
        "头维度", "压缩比", "余弦相似度", "内存 (KB)"
    );
    println!("{}", "-".repeat(50));

    let mut rng = rand::thread_rng();

    for &head_dim in &head_dims {
        let mut cache = SimulatedKVCache::new(head_dim, true);

        for _ in 0..seq_len {
            let key: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            let value: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            cache.append(&key, &value);
        }

        // 计算重建质量
        let mut cosine_sims = Vec::new();
        for i in 0..seq_len {
            let orig = &cache.keys[i];
            let recon = cache.get_key(i);
            cosine_sims.push(cosine_similarity(orig, &recon));
        }
        let mean_cosine = cosine_sims.iter().sum::<f64>() / cosine_sims.len() as f64;

        println!(
            "{:<12} {:<10.2} {:<10.4} {:<15.2}",
            head_dim,
            cache.compression_ratio(),
            mean_cosine,
            cache.total_compressed_bytes as f64 / 1024.0
        );
    }
}

/// 演示 3: 长序列压缩（10K tokens）
///
/// 测试大规模 KV Cache 的压缩性能和随机访问速度
fn demo_long_sequence() {
    println!("\n{}", "=".repeat(70));
    println!("长序列压缩 (10K tokens)");
    println!("{}", "=".repeat(70));

    let head_dim = 64;
    let seq_len = 10000;

    println!("\n模拟 {} 个 tokens，head_dim={}", seq_len, head_dim);

    let mut cache = SimulatedKVCache::new(head_dim, true);
    let mut rng = rand::thread_rng();

    for i in 0..seq_len {
        let key: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let value: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        cache.append(&key, &value);

        if (i + 1) % 2000 == 0 {
            println!("  已处理 {} tokens...", i + 1);
        }
    }

    println!("\n内存使用:");
    println!(
        "  原始: {:.2} MB",
        cache.total_original_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  压缩后: {:.2} MB",
        cache.total_compressed_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  实际压缩比: {:.2}x (u32 索引存储)",
        cache.compression_ratio()
    );
    let theoretical_ratio = cache.quantizer.as_ref().unwrap().compression_ratio();
    println!(
        "  理论压缩比 (bit-packed): {:.2}x",
        theoretical_ratio
    );
    println!("  节省空间: {:.1}%", cache.space_saved_pct());

    // 随机访问性能测试
    println!("\n随机访问测试:");
    let n_samples = 100;
    let mut sample_indices: Vec<usize> = Vec::new();
    for _ in 0..n_samples {
        sample_indices.push(rng.gen_range(0..seq_len));
    }

    let start = Instant::now();
    for &idx in &sample_indices {
        let _ = cache.get_key(idx);
        let _ = cache.get_value(idx);
    }
    let elapsed = start.elapsed().as_secs_f64();

    println!(
        "  访问 {} 个随机 KV 对耗时 {:.2} ms",
        n_samples,
        elapsed * 1000.0
    );
    println!(
        "  平均访问时间: {:.3} ms",
        elapsed / n_samples as f64 * 1000.0
    );
}

/// 演示 4: 质量 vs 压缩率权衡
///
/// 不同量化位宽下的压缩率和重建质量对比
fn demo_quality_vs_compression() {
    println!("\n{}", "=".repeat(70));
    println!("质量 vs 压缩率权衡");
    println!("{}", "=".repeat(70));

    let head_dim = 128;
    let seq_len = 100;

    let bit_configs = [
        (4, 2, "低质量"),
        (6, 3, "中低质量"),
        (8, 4, "中等质量"),
        (10, 6, "中高质量"),
        (12, 8, "高质量"),
    ];

    println!(
        "\n{:<15} {:<8} {:<8} {:<8} {:<10}",
        "配置", "半径位", "角度位", "压缩比", "余弦"
    );
    println!("{}", "-".repeat(60));

    let mut rng = rand::thread_rng();

    for (r_bits, a_bits, label) in bit_configs {
        let config = PolarQuantConfig::builder(head_dim)
            .radius_bits(r_bits)
            .angle_bits(a_bits)
            .seed(42)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let mut keys = Vec::new();
        let mut compressed_keys = Vec::new();

        for _ in 0..seq_len {
            let key: Vec<f64> = (0..head_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            let key_norm = normalize_vector(&key);
            keys.push(key_norm.clone());
            compressed_keys.push(pq.compress(&key_norm).unwrap());
        }

        // 计算平均余弦相似度
        let mut cosine_sims = Vec::new();
        for i in 0..seq_len {
            let recon = pq.decompress(&compressed_keys[i]);
            cosine_sims.push(cosine_similarity(&keys[i], &recon));
        }
        let mean_cosine = cosine_sims.iter().sum::<f64>() / cosine_sims.len() as f64;

        println!(
            "{:<15} {:<8} {:<8} {:<8.2} {:<10.4}",
            label,
            r_bits,
            a_bits,
            pq.compression_ratio(),
            mean_cosine
        );
    }
}

/// 主函数：运行所有演示
fn main() {
    println!();
    println!("{}", "*".repeat(70));
    println!("*{}*", " ".repeat(68));
    let title = "  PolarQuant: KV Cache 压缩演示 (Rust)";
    let padded: String = title
        .chars()
        .chain(std::iter::repeat(' '))
        .take(68)
        .collect();
    println!("*{}*", padded);
    println!("*{}*", " ".repeat(68));
    println!("{}", "*".repeat(70));

    demo_kv_cache_compression();
    demo_attention_head_comparison();
    demo_long_sequence();
    demo_quality_vs_compression();

    println!("\n{}", "=".repeat(70));
    println!("所有演示完成！");
    println!("{}", "=".repeat(70));
}
