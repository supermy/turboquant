use polarquant::{PolarQuant, PolarQuantBatch, PolarQuantConfig};

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

/// 生成单位向量集合
/// 
/// # 参数
/// - `n`: 向量数量
/// - `dim`: 向量维度
/// 
/// # 返回值
/// 包含n个单位向量的向量集合
fn generate_unit_vectors(n: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|seed| {
            let x: Vec<f64> = (0..dim)
                .map(|i| ((seed * dim + i + 1) as f64 * 0.7).sin())
                .collect();
            normalize_vector(&x)
        })
        .collect()
}

/// 测试：KV Cache 压缩
/// 
/// 验证压缩-解压过程能够保持较高的余弦相似度
/// 使用100个token，64维head，8-bit半径+4-bit角度配置
#[test]
fn test_kv_cache_compression() {
    let head_dim = 64;
    let config = PolarQuantConfig::builder(head_dim)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let n_tokens = 100;
    let keys = generate_unit_vectors(n_tokens, head_dim);

    let compressed_keys: Vec<_> = keys.iter().map(|k| pq.compress(k).unwrap()).collect();
    let reconstructed_keys: Vec<_> = compressed_keys.iter().map(|c| pq.decompress(c)).collect();

    let cosine_sims: Vec<f64> = keys
        .iter()
        .zip(reconstructed_keys.iter())
        .map(|(orig, recon)| {
            let norm_o: f64 = orig.iter().map(|v| v * v).sum::<f64>().sqrt();
            let norm_r: f64 = recon.iter().map(|v| v * v).sum::<f64>().sqrt();
            orig.iter()
                .zip(recon.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                / (norm_o * norm_r)
        })
        .collect();

    let mean_cosine = cosine_sims.iter().sum::<f64>() / cosine_sims.len() as f64;
    assert!(
        mean_cosine > 0.85,
        "平均余弦相似度过低: {}",
        mean_cosine
    );
}

/// 测试：注意力分数保持
/// 
/// 验证压缩后的Key向量在注意力计算中能够保持原始分数的相关性
/// 使用Pearson相关系数评估注意力分数的一致性
#[test]
fn test_attention_score_preservation() {
    let head_dim = 64;
    let config = PolarQuantConfig::builder(head_dim)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let mut rng = rand::thread_rng();
    use rand::Rng;
    let query: Vec<f64> = (0..head_dim)
        .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
        .collect();
    let query = normalize_vector(&query);

    let keys: Vec<Vec<f64>> = (0..50)
        .map(|_| {
            let k: Vec<f64> = (0..head_dim)
                .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
                .collect();
            normalize_vector(&k)
        })
        .collect();

    let original_scores: Vec<f64> = keys.iter().map(|k| dot(&query, k)).collect();

    let compressed_keys: Vec<_> = keys.iter().map(|k| pq.compress(k).unwrap()).collect();
    let reconstructed_keys: Vec<_> = compressed_keys.iter().map(|c| pq.decompress(c)).collect();
    let reconstructed_scores: Vec<f64> =
        reconstructed_keys.iter().map(|k| dot(&query, k)).collect();

    let correlation = pearson_correlation(&original_scores, &reconstructed_scores);
    assert!(
        correlation > 0.7,
        "注意力分数相关性过低: {}",
        correlation
    );
}

/// 测试：高维嵌入压缩
/// 
/// 验证在768维（BERT/GPT标准维度）下的压缩效果
/// 高维向量由于Beta分布更集中，压缩质量更高
#[test]
fn test_high_dimensional_embeddings() {
    let config = PolarQuantConfig::builder(768)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let embeddings = generate_unit_vectors(10, 768);

    let compressed: Vec<_> = embeddings.iter().map(|e| pq.compress(e).unwrap()).collect();
    let reconstructed: Vec<_> = compressed.iter().map(|c| pq.decompress(c)).collect();

    let cosine_sims: Vec<f64> = embeddings
        .iter()
        .zip(reconstructed.iter())
        .map(|(orig, recon)| cosine_similarity(orig, recon))
        .collect();

    let mean_cosine = cosine_sims.iter().sum::<f64>() / cosine_sims.len() as f64;
    assert!(mean_cosine > 0.9, "平均余弦相似度过低: {}", mean_cosine);
}

/// 测试：语义相似度保持
/// 
/// 验证压缩后相似向量保持相似，不相似向量保持不相似
/// 测试压缩对向量空间结构的保持能力
#[test]
fn test_semantic_similarity_preservation() {
    let config = PolarQuantConfig::builder(256)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let base = normalize_vector(&(0..256).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());

    let mut similar = base.clone();
    for (i, v) in similar.iter_mut().enumerate() {
        *v += 0.05 * ((i + 1) as f64 * 0.3).cos();
    }
    similar = normalize_vector(&similar);

    let dissimilar = normalize_vector(
        &(0..256)
            .map(|i| ((i + 100) as f64 * 0.7).cos())
            .collect::<Vec<_>>(),
    );

    let base_c = pq.compress(&base).unwrap();
    let similar_c = pq.compress(&similar).unwrap();
    let dissimilar_c = pq.compress(&dissimilar).unwrap();

    let base_r = pq.decompress(&base_c);
    let similar_r = pq.decompress(&similar_c);
    let dissimilar_r = pq.decompress(&dissimilar_c);

    let recon_sim = cosine_similarity(&base_r, &similar_r);
    let recon_dissim = cosine_similarity(&base_r, &dissimilar_r);

    assert!(
        recon_sim > 0.5,
        "相似向量应保持正相关: {}",
        recon_sim
    );
    assert!(
        recon_dissim.abs() < 0.6,
        "不相似向量应保持不相似: {}",
        recon_dissim
    );
}

/// 测试：压缩比计算
/// 
/// 验证不同维度下的压缩比计算是否正确
/// 压缩比 = 原始位数 / 压缩后位数
#[test]
fn test_compression_ratios() {
    let test_cases = vec![(64, 8, 4), (128, 8, 4), (256, 8, 4), (512, 8, 4)];

    for (dim, r_bits, a_bits) in test_cases {
        let config = PolarQuantConfig::builder(dim)
            .radius_bits(r_bits)
            .angle_bits(a_bits)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let ratio = pq.compression_ratio();
        let expected = (dim as f64 * 32.0) / (r_bits as f64 + (dim - 1) as f64 * a_bits as f64);

        assert!(
            (ratio - expected).abs() < 1e-5,
            "维度 {} 的压缩比不匹配: 实际 {}, 期望 {}",
            dim,
            ratio,
            expected
        );
    }
}

/// 测试：高维压缩比更优
/// 
/// 验证随着维度增加，压缩比单调递增
/// 这是因为高维时Beta分布更集中，量化效率更高
#[test]
fn test_higher_dimensions_better_ratios() {
    let dims = vec![32, 64, 128, 256, 512];
    let mut ratios = Vec::new();

    for &dim in &dims {
        let config = PolarQuantConfig::builder(dim).build().unwrap();
        let pq = PolarQuant::new(config).unwrap();
        ratios.push(pq.compression_ratio());
    }

    for i in 0..ratios.len() - 1 {
        assert!(
            ratios[i + 1] >= ratios[i],
            "维度 {} 应该有比维度 {} 更好的压缩比",
            dims[i + 1],
            dims[i]
        );
    }
}

/// 测试：小数值鲁棒性
/// 
/// 验证对极小数值（1e-6量级）的压缩稳定性
/// 确保不会出现数值溢出或下溢
#[test]
fn test_robustness_small_values() {
    let config = PolarQuantConfig::builder(16).build().unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let x: Vec<f64> = (0..16).map(|i| 1e-6 * ((i + 1) as f64).sin()).collect();
    let compressed = pq.compress(&x).unwrap();
    let x_recon = pq.decompress(&compressed);

    assert_eq!(x_recon.len(), 16);
    assert!(x_recon.iter().all(|v| v.is_finite()));
}

/// 测试：大数值鲁棒性
/// 
/// 验证对极大数值（1e6量级）的压缩稳定性
/// 确保不会出现数值溢出
#[test]
fn test_robustness_large_values() {
    let config = PolarQuantConfig::builder(16).build().unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let x: Vec<f64> = (0..16).map(|i| 1e6 * ((i + 1) as f64).sin()).collect();
    let compressed = pq.compress(&x).unwrap();
    let x_recon = pq.decompress(&compressed);

    assert_eq!(x_recon.len(), 16);
    assert!(x_recon.iter().all(|v| v.is_finite()));
}

/// 测试：混合尺度鲁棒性
/// 
/// 验证同时包含大数值（1e3）和小数值（1e-3）的向量的压缩稳定性
/// 测试算法对不同量级混合数据的处理能力
#[test]
fn test_robustness_mixed_scale() {
    let config = PolarQuantConfig::builder(16).build().unwrap();
    let pq = PolarQuant::new(config).unwrap();

    let mut x: Vec<f64> = (0..16).map(|i| ((i + 1) as f64).sin()).collect();
    x[0] = 1e3;
    x[1] = 1e-3;

    let compressed = pq.compress(&x).unwrap();
    let x_recon = pq.decompress(&compressed);

    assert_eq!(x_recon.len(), 16);
    assert!(x_recon.iter().all(|v| v.is_finite()));
}

/// 测试：不同维度支持
/// 
/// 验证算法支持多种常见维度（4, 16, 64, 256）
/// 确保压缩-解压过程在所有测试维度下都能正常工作
#[test]
fn test_different_dimensions() {
    for dim in [4, 16, 64, 256] {
        let config = PolarQuantConfig::builder(dim)
            .radius_bits(8)
            .angle_bits(4)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let x: Vec<f64> = (0..dim).map(|i| ((i + 1) as f64).sin()).collect();
        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);

        assert_eq!(x_recon.len(), dim);
    }
}

/// 测试：不同位宽配置
/// 
/// 验证算法支持多种位宽组合（4/2, 8/4, 12/8）
/// 测试不同精度配置下的压缩效果
#[test]
fn test_different_bit_widths() {
    for (r_bits, a_bits) in [(4, 2), (8, 4), (12, 8)] {
        let config = PolarQuantConfig::builder(16)
            .radius_bits(r_bits)
            .angle_bits(a_bits)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();

        let x = normalize_vector(&(0..16).map(|i| ((i + 1) as f64).sin()).collect::<Vec<_>>());
        let compressed = pq.compress(&x).unwrap();
        let x_recon = pq.decompress(&compressed);

        assert_eq!(x_recon.len(), 16);
    }
}

/// 测试：批量KV Cache处理
/// 
/// 验证 PolarQuantBatch 的批量压缩/解压功能
/// 测试批量操作的性能和准确性
#[test]
fn test_batch_kv_cache() {
    let config = PolarQuantConfig::builder(64)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();
    let batch = PolarQuantBatch::new(&pq);

    let vectors = generate_unit_vectors(100, 64);
    let compressed = batch.compress_batch(&vectors);
    let reconstructed = batch.decompress_batch(&compressed);

    assert_eq!(compressed.len(), 100);
    assert_eq!(reconstructed.len(), 100);

    let errors = batch.compute_batch_error(&vectors, &reconstructed);
    assert!(
        errors.mean_cosine > 0.8,
        "批量平均余弦相似度过低: {}",
        errors.mean_cosine
    );
}

/// 计算两个向量的点积
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// 计算两个向量的余弦相似度
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

/// 计算两组数据的Pearson相关系数
/// 
/// # 参数
/// - `x`: 第一组数据
/// - `y`: 第二组数据
/// 
/// # 返回值
/// Pearson相关系数，范围[-1, 1]
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - mean_x) * (b - mean_y))
        .sum::<f64>()
        / n;
    let var_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum::<f64>() / n;
    let var_y: f64 = y.iter().map(|b| (b - mean_y).powi(2)).sum::<f64>() / n;

    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();

    if std_x < 1e-10 || std_y < 1e-10 {
        return 0.0;
    }

    cov / (std_x * std_y)
}
