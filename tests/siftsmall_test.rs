//! SIFT Small 真实数据测试: 召回率 + QPS

use std::path::Path;
use std::time::Instant;

use ::turboquant::*;
use ::turboquant::sift::SiftSmallDataset;

fn compute_recall_sift(result: &[usize], gt: &[i32], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k {
        if result.contains(&(gt[i] as usize)) { hits += 1; }
    }
    hits as f32 / k as f32
}

#[test]
fn test_siftsmall_recall_and_qps() {
    let data_dir = std::env::var("SIFT_DATA_DIR")
        .map(|p| Path::new(&p).to_path_buf())
        .unwrap_or_else(|_| Path::new("/Users/moyong/project/ai/models/data/siftsmall").to_path_buf());
    if !data_dir.exists() { println!("跳过测试: 数据目录不存在 (设置 SIFT_DATA_DIR 环境变量)"); return; }

    let dataset = SiftSmallDataset::load(data_dir).unwrap();
    let k = 10;
    let nq = dataset.nq;
    let nb = dataset.nb;
    let d = dataset.d;
    let fp32_size = d * 4;

    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║     SIFT Small 真实数据测试 - 召回率 & QPS (Queries/Second)        ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    println!("数据集信息:");
    println!("  基础向量: {} x {}d ({} KB, {:.1} MB)", nb, d, nb * d * 4 / 1024, nb as f64 * d as f64 * 4.0 / 1024.0 / 1024.0);
    println!("  查询向量: {} x {}d", nq, d);
    println!("  返回数量: k={}", k);

    struct BenchResult {
        name: &'static str,
        storage_per_vec: usize,
        recall: f32,
        qps: f64,
        latency_ms: f64,
        train_time_ms: f64,
        add_time_ms: f64,
    }

    let mut results: Vec<BenchResult> = Vec::new();

    // ========== 1. TurboQuant 4-bit ==========
    {
        print!("[1/5] TurboQuant 4-bit ... ");
        let t0 = Instant::now();
        let mut index = TurboQuantFlatIndex::new(d, 4, false);
        index.train(&dataset.base, nb);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        index.add(&dataset.base, nb);
        let add_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let res = index.search(&dataset.query, nq, k, 1);
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut total_recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / nq as f32;
        let qps = nq as f64 / (search_ms / 1000.0);
        results.push(BenchResult {
            name: "TurboQuant 4-bit",
            storage_per_vec: index.code_size(),
            recall,
            qps,
            latency_ms: search_ms / nq as f64,
            train_time_ms: train_ms,
            add_time_ms: add_ms,
        });
        println!("Recall@{}={:.2}% QPS={:.0} lat={:.2}ms/q", k, recall * 100.0, qps, search_ms / nq as f64);
    }

    // ========== 2. TurboQuant 6-bit ==========
    {
        print!("[2/5] TurboQuant 6-bit ... ");
        let t0 = Instant::now();
        let mut index = TurboQuantFlatIndex::new(d, 6, false);
        index.train(&dataset.base, nb);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        index.add(&dataset.base, nb);
        let add_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let res = index.search(&dataset.query, nq, k, 1);
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut total_recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / nq as f32;
        let qps = nq as f64 / (search_ms / 1000.0);
        results.push(BenchResult {
            name: "TurboQuant 6-bit",
            storage_per_vec: index.code_size(),
            recall,
            qps,
            latency_ms: search_ms / nq as f64,
            train_time_ms: train_ms,
            add_time_ms: add_ms,
        });
        println!("Recall@{}={:.2}% QPS={:.0} lat={:.2}ms/q", k, recall * 100.0, qps, search_ms / nq as f64);
    }

    // ========== 3. TurboQuant 4-bit + SQ8 ==========
    {
        print!("[3/5] TurboQuant 4-bit + SQ8 ... ");
        let t0 = Instant::now();
        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&dataset.base, nb);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        index.add(&dataset.base, nb);
        let add_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let res = index.search(&dataset.query, nq, k, 10);
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut total_recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / nq as f32;
        let qps = nq as f64 / (search_ms / 1000.0);
        results.push(BenchResult {
            name: "TurboQuant 4-bit + SQ8",
            storage_per_vec: index.total_storage() / index.ntotal(),
            recall,
            qps,
            latency_ms: search_ms / nq as f64,
            train_time_ms: train_ms,
            add_time_ms: add_ms,
        });
        println!("Recall@{}={:.2}% QPS={:.0} lat={:.2}ms/q", k, recall * 100.0, qps, search_ms / nq as f64);
    }

    // ========== 4. RaBitQ Flat + SQ8 ==========
    {
        print!("[4/5] RaBitQ Flat + SQ8 ... ");
        let t0 = Instant::now();
        let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&dataset.base, nb);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        index.add(&dataset.base, nb);
        let add_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let res = index.search(&dataset.query, nq, k, 10);
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut total_recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / nq as f32;
        let qps = nq as f64 / (search_ms / 1000.0);
        results.push(BenchResult {
            name: "RaBitQ Flat + SQ8",
            storage_per_vec: index.code_size() + d,
            recall,
            qps,
            latency_ms: search_ms / nq as f64,
            train_time_ms: train_ms,
            add_time_ms: add_ms,
        });
        println!("Recall@{}={:.2}% QPS={:.0} lat={:.2}ms/q", k, recall * 100.0, qps, search_ms / nq as f64);
    }

    // ========== 5. RaBitQ IVF + SQ8 ==========
    {
        print!("[5/5] RaBitQ IVF + SQ8 ... ");
        let t0 = Instant::now();
        let mut index = ivf::RaBitQIVFIndex::new(d, 64, 1, false, true);
        index.train(&dataset.base, nb);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        index.add(&dataset.base, nb);
        let add_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let res = index.search(&dataset.query, nq, k, 32, 10);
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let mut total_recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / nq as f32;
        let qps = nq as f64 / (search_ms / 1000.0);
        results.push(BenchResult {
            name: "RaBitQ IVF + SQ8",
            storage_per_vec: index.code_size() + d,
            recall,
            qps,
            latency_ms: search_ms / nq as f64,
            train_time_ms: train_ms,
            add_time_ms: add_ms,
        });
        println!("Recall@{}={:.2}% QPS={:.0} lat={:.2}ms/q", k, recall * 100.0, qps, search_ms / nq as f64);
    }

    // ========== 输出汇总表格 ==========
    println!("\n{:=^90}", "");
    println!("综合对比报告");
    println!("{:=^90}", "");

    println!("\n{:<25} | {:>7} | {:>9} | {:>8} | {:>8} | {:>8} | {:>8}",
             "方法", "存储/B", "Recall@K", "QPS", "lat(ms)", "train(ms)", "add(ms)");
    println!("{:-<25}-+-{:>-7}-+-{:>-9}-+-{:>-8}-+-{:>-8}-+-{:>-8}-+-{:>-8}",
             "", "", "", "", "", "", "");

    for r in &results {
        println!("{:<25} | {:>6}B | {:>8.2}% | {:>7.0} | {:>7.2} | {:>7.1} | {:>7.1}",
                 r.name, r.storage_per_vec, r.recall * 100.0, r.qps,
                 r.latency_ms, r.train_time_ms, r.add_time_ms);
    }
    println!("{:-<25}-+-{:>-7}-+-{:>-9}-+-{:>-8}-+-{:>-8}-+-{:>-8}-+-{:>-8}",
             "", "", "", "", "", "", "");
    println!("{:<25} | {:>6}B |           |         |         |         |         |",
             "FP32基准", fp32_size);

    // ========== 性价比评分 ==========
    println!("\n{:=^90}", "");
    println!("性价比分析");
    println!("{:=^90}", "");

    for r in &results {
        let compression = fp32_size as f64 / r.storage_per_vec as f64;
        let score_recall = r.recall as f64 * 100.0;
        let score_speed = r.qps.log2() * 10.0;
        let score_storage = compression.log2() * 10.0;
        let cost_benefit = score_recall * score_speed * score_storage / 100.0;

        println!("{:<25}: Recall={:>5.1}分  Speed={:>5.1}分  Storage={:>5.1}分  综合={:>6.1}",
                 r.name, score_recall, score_speed, score_storage, cost_benefit);
    }

    // ========== 推荐结论 ==========
    println!("\n{:=^90}", "");
    println!("推荐结论");
    println!("{:=^90}", "");

    let best_recall = results.iter().max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap()).unwrap();
    let best_speed = results.iter().max_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap()).unwrap();

    println!("\n最高召回率: {} (Recall@{}={:.2}%)",
             best_recall.name, k, best_recall.recall * 100.0);
    println!("最高速度:   {} (QPS={}, 延迟={:.2}ms/query)",
             best_speed.name, best_speed.qps as i64, best_speed.latency_ms);

    println!("\n推荐场景:");
    println!("  - 高精度需求 -> {} ({:.1}% 召回率)", best_recall.name, best_recall.recall * 100.0);
    println!("  - 低延迟需求 -> {} ({} QPS)", best_speed.name, best_speed.qps as i64);
}
