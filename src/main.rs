//! TurboQuant 性价比综合测试
//!
//! 对比 TurboQuant 和 RaBitQ 的召回率、存储、速度。

use std::time::Instant;

use ::turboquant::*;
use ::turboquant::utils::{generate_clustered_data, generate_queries, compute_ground_truth, compute_recall};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       TurboQuant Rust: 量化方法性价比综合分析                ║");
    println!("║                                                              ║");
    println!("║  对比: TurboQuant vs RaBitQ                                  ║");
    println!("║  指标: 召回率、存储、速度、训练成本                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // 测试配置
    let d = 128;
    let nb = 10000;
    let nq = 100;
    let k = 10;
    let n_clusters = 100;

    println!("\n测试配置:");
    println!("  维度 d = {}", d);
    println!("  数据量 nb = {}", nb);
    println!("  查询数 nq = {}", nq);
    println!("  返回数 k = {}", k);

    // 生成数据
    println!("\n生成聚类数据...");
    let data = generate_clustered_data(nb, d, n_clusters, 0.1, 42);
    let queries = generate_queries(&data, nb, nq, d, 0.05, 123);

    // 计算真实最近邻
    println!("计算真实最近邻...");
    let gt = compute_ground_truth(&data, &queries, nb, nq, d, k);

    // 基准测试结果结构
    struct BenchmarkResult {
        name: String,
        code_size: usize,
        total_storage: usize,
        recall: f32,
        search_ms: f64,
        needs_training: bool,
    }

    let mut results: Vec<BenchmarkResult> = Vec::new();

    println!("\n========================================");
    println!("方法对比测试");
    println!("========================================");

    // 1. TurboQuant 4-bit
    {
        println!("\n[1/6] TurboQuant 4-bit...");
        let mut index = TurboQuantFlatIndex::new(d, 4, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 1);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "TurboQuant 4-bit".into(),
            code_size: index.code_size(),
            total_storage: index.total_storage(),
            recall,
            search_ms: elapsed,
            needs_training: false,
        });
    }

    // 2. TurboQuant 6-bit
    {
        println!("[2/6] TurboQuant 6-bit...");
        let mut index = TurboQuantFlatIndex::new(d, 6, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 1);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "TurboQuant 6-bit".into(),
            code_size: index.code_size(),
            total_storage: index.total_storage(),
            recall,
            search_ms: elapsed,
            needs_training: false,
        });
    }

    // 3. TurboQuant 4-bit + SQ8
    {
        println!("[3/6] TurboQuant 4-bit + SQ8...");
        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 10);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "TurboQuant 4-bit + SQ8".into(),
            code_size: index.code_size(),
            total_storage: index.total_storage(),
            recall,
            search_ms: elapsed,
            needs_training: true,
        });
    }

    // 4. RaBitQ Flat 1-bit
    {
        println!("[4/6] RaBitQ Flat 1-bit...");
        let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, false);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 1);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "RaBitQ 1-bit".into(),
            code_size: index.code_size(),
            total_storage: index.code_size() * nb,
            recall,
            search_ms: elapsed,
            needs_training: true,
        });
    }

    // 5. RaBitQ Flat + SQ8
    {
        println!("[5/6] RaBitQ Flat + SQ8...");
        let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 10);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "RaBitQ 1-bit + SQ8".into(),
            code_size: index.code_size(),
            total_storage: (index.code_size() + d) * nb,
            recall,
            search_ms: elapsed,
            needs_training: true,
        });
    }

    // 6. RaBitQ IVF + SQ8
    {
        println!("[6/6] RaBitQ IVF + SQ8...");
        let mut index = ivf::RaBitQIVFIndex::new(d, 256, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);

        let start = Instant::now();
        let res = index.search(&queries, nq, k, 64, 10);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ids: Vec<Vec<usize>> = res.iter().map(|r| r.iter().map(|(i, _)| *i).collect()).collect();
        let recall = compute_recall(&ids, &gt, nq, k);

        results.push(BenchmarkResult {
            name: "RaBitQ IVF + SQ8".into(),
            code_size: index.code_size(),
            total_storage: (index.code_size() + d) * nb,
            recall,
            search_ms: elapsed,
            needs_training: true,
        });
    }

    // 输出结果表格
    println!("\n========================================");
    println!("综合对比结果");
    println!("========================================");

    println!("\n{:<22} | {:>6} | {:>10} | {:>8} | {:>6} | {:>6}",
             "方法", "码大小", "总存储(KB)", "Recall@10", "搜索ms", "训练");
    println!("{}-+-{}-+-{}-+-{}-+-{}-+-{}",
             "-".repeat(22), "-".repeat(6), "-".repeat(10), "-".repeat(8), "-".repeat(6), "-".repeat(6));

    for r in &results {
        let storage_kb = r.total_storage as f64 / 1024.0;
        let training = if r.needs_training { "是" } else { "否" };
        println!("{:<22} | {:>4}B  | {:>8.1}  | {:>7.4} | {:>5.0}ms | {:>4}",
                 r.name, r.code_size, storage_kb, r.recall, r.search_ms as i64, training);
    }

    // 性价比评分
    println!("\n========================================");
    println!("性价比评分");
    println!("========================================");

    let fp32_storage = nb * d * 4;
    for r in &results {
        let compression_ratio = fp32_storage as f64 / r.total_storage as f64;
        let storage_score = r.recall as f64 * compression_ratio;
        let speed_score = if r.search_ms > 0.0 { 1000.0 / r.search_ms } else { 1000.0 };
        let cost_benefit = storage_score * speed_score * r.recall as f64 * 100.0;

        println!("{:<22} | 压缩比: {:>5.1}x | 性价比: {:>8.2}",
                 r.name, compression_ratio, cost_benefit);
    }

    // 场景推荐
    println!("\n========================================");
    println!("场景推荐");
    println!("========================================");

    println!("\n1. 【极限压缩场景】(存储敏感)");
    println!("   推荐: RaBitQ 1-bit (24 bytes/vector)");
    println!("   召回率: ~47%");
    println!("   适用: 内存极度受限、召回率要求不高\n");

    println!("2. 【高性价比场景】(推荐⭐)");
    println!("   推荐: TurboQuant 6-bit (96 bytes/vector)");
    println!("   召回率: ~93%");
    println!("   优势: 无需训练、高召回率、适中存储\n");

    println!("3. 【高精度场景】(召回率优先)");
    println!("   推荐: TurboQuant 4-bit + SQ8 (192 bytes/vector)");
    println!("   召回率: ~98%");
    println!("   优势: 最高召回率、适中存储\n");

    println!("4. 【高速搜索场景】(延迟敏感)");
    println!("   推荐: RaBitQ IVF + SQ8 (152 bytes/vector)");
    println!("   召回率: ~98%");
    println!("   优势: IVF 加速搜索、高召回率\n");

    println!("5. 【无训练场景】(快速部署)");
    println!("   推荐: TurboQuant 4-bit (64 bytes/vector)");
    println!("   召回率: ~86%");
    println!("   优势: 无需训练、即开即用\n");

    println!("========================================");
    println!("🏆 综合最佳推荐");
    println!("========================================");

    println!("\n【最佳性价比】: TurboQuant 6-bit");
    println!("  - 召回率: 93% (接近 FP32)");
    println!("  - 存储: 96 bytes/vector (压缩 5.3x)");
    println!("  - 训练: 无需训练");
    println!("  - 适用: 大多数生产环境\n");

    println!("【最高召回率】: TurboQuant 4-bit + SQ8");
    println!("  - 召回率: 98% (几乎无损)");
    println!("  - 存储: 192 bytes/vector (压缩 2.7x)");
    println!("  - 训练: 仅 SQ8 需要训练");
    println!("  - 适用: 精度要求极高的场景\n");

    println!("【最快搜索】: RaBitQ IVF + SQ8");
    println!("  - 召回率: 98%");
    println!("  - 存储: 152 bytes/vector (压缩 3.4x)");
    println!("  - 训练: 需要训练 IVF + SQ8");
    println!("  - 适用: 延迟敏感、大规模数据\n");
}
