//! SIFT Small 真实数据测试

use std::path::Path;

use ::turboquant::*;
use ::turboquant::sift::SiftSmallDataset;

/// 计算 SIFT 数据集的召回率
fn compute_recall_sift(result: &[usize], gt: &[i32], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k {
        if result.contains(&(gt[i] as usize)) {
            hits += 1;
        }
    }
    hits as f32 / k as f32
}

#[test]
fn test_siftsmall_turboquant_4bit() {
    let data_dir = Path::new("/Users/moyong/project/ai/models/data/siftsmall");
    if !data_dir.exists() {
        println!("跳过测试: 数据目录不存在");
        return;
    }

    println!("\n========================================");
    println!("SIFT Small 数据集测试");
    println!("========================================\n");

    let dataset = SiftSmallDataset::load(data_dir).unwrap();
    println!("数据集加载完成:");
    println!("  基础向量: {} x {}", dataset.nb, dataset.d);
    println!("  查询向量: {} x {}", dataset.nq, dataset.d);
    println!("  真实最近邻: {} x {}", dataset.nq, dataset.k);

    let k = 10;

    // TurboQuant 4-bit
    println!("\n[1] TurboQuant 4-bit...");
    let mut index = TurboQuantFlatIndex::new(dataset.d, 4, false);
    index.train(&dataset.base, dataset.nb);
    index.add(&dataset.base, dataset.nb);

    let results = index.search(&dataset.query, dataset.nq, k, 1);
    let mut total_recall = 0.0;
    for q in 0..dataset.nq {
        let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
        total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
    }
    let recall = total_recall / dataset.nq as f32;
    println!("  TurboQuant 4-bit Recall@{}: {:.4}", k, recall);

    // TurboQuant 6-bit
    println!("\n[2] TurboQuant 6-bit...");
    let mut index = TurboQuantFlatIndex::new(dataset.d, 6, false);
    index.train(&dataset.base, dataset.nb);
    index.add(&dataset.base, dataset.nb);

    let results = index.search(&dataset.query, dataset.nq, k, 1);
    let mut total_recall = 0.0;
    for q in 0..dataset.nq {
        let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
        total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
    }
    let recall = total_recall / dataset.nq as f32;
    println!("  TurboQuant 6-bit Recall@{}: {:.4}", k, recall);

    // TurboQuant 4-bit + SQ8
    println!("\n[3] TurboQuant 4-bit + SQ8...");
    let mut index = TurboQuantFlatIndex::new(dataset.d, 4, true);
    index.train(&dataset.base, dataset.nb);
    index.add(&dataset.base, dataset.nb);

    let results = index.search(&dataset.query, dataset.nq, k, 10);
    let mut total_recall = 0.0;
    for q in 0..dataset.nq {
        let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
        total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
    }
    let recall = total_recall / dataset.nq as f32;
    println!("  TurboQuant 4-bit + SQ8 Recall@{}: {:.4}", k, recall);

    // RaBitQ Flat + SQ8
    println!("\n[4] RaBitQ Flat + SQ8...");
    let mut index = rabitq::RaBitQFlatIndex::new(dataset.d, 1, false, true);
    index.train(&dataset.base, dataset.nb);
    index.add(&dataset.base, dataset.nb);

    let results = index.search(&dataset.query, dataset.nq, k, 10);
    let mut total_recall = 0.0;
    for q in 0..dataset.nq {
        let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
        total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
    }
    let recall = total_recall / dataset.nq as f32;
    println!("  RaBitQ Flat + SQ8 Recall@{}: {:.4}", k, recall);

    // RaBitQ IVF + SQ8
    println!("\n[5] RaBitQ IVF + SQ8...");
    let mut index = ivf::RaBitQIVFIndex::new(dataset.d, 64, 1, false, true);
    index.train(&dataset.base, dataset.nb);
    index.add(&dataset.base, dataset.nb);

    let results = index.search(&dataset.query, dataset.nq, k, 32, 10);
    let mut total_recall = 0.0;
    for q in 0..dataset.nq {
        let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
        total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
    }
    let recall = total_recall / dataset.nq as f32;
    println!("  RaBitQ IVF + SQ8 Recall@{}: {:.4}", k, recall);

    println!("\n========================================");
    println!("测试完成");
    println!("========================================");
}

#[test]
fn test_siftsmall_detailed() {
    let data_dir = Path::new("/Users/moyong/project/ai/models/data/siftsmall");
    if !data_dir.exists() {
        println!("跳过测试: 数据目录不存在");
        return;
    }

    let dataset = SiftSmallDataset::load(data_dir).unwrap();
    let k = 10;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          SIFT Small 真实数据测试 - 详细报告                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("数据集信息:");
    println!("  文件: siftsmall_base.fvecs, siftsmall_query.fvecs");
    println!("  基础向量: {} x {} ({} KB)", 
             dataset.nb, dataset.d, 
             dataset.nb * dataset.d * 4 / 1024);
    println!("  查询向量: {} x {} ({} KB)", 
             dataset.nq, dataset.d, 
             dataset.nq * dataset.d * 4 / 1024);

    println!("\n{:-^70}", "");
    println!("{:<25} | {:>10} | {:>12} | {:>12}", 
             "方法", "存储/向量", "Recall@10", "压缩比");
    println!("{:-^70}", "");

    let fp32_size = dataset.d * 4;

    // TurboQuant 4-bit
    {
        let mut index = TurboQuantFlatIndex::new(dataset.d, 4, false);
        index.train(&dataset.base, dataset.nb);
        index.add(&dataset.base, dataset.nb);

        let results = index.search(&dataset.query, dataset.nq, k, 1);
        let mut total_recall = 0.0;
        for q in 0..dataset.nq {
            let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / dataset.nq as f32;
        let storage = index.code_size();
        let ratio = fp32_size as f64 / storage as f64;

        println!("{:<25} | {:>8}B  | {:>11.2}% | {:>10.1}x", 
                 "TurboQuant 4-bit", storage, recall * 100.0, ratio);
    }

    // TurboQuant 6-bit
    {
        let mut index = TurboQuantFlatIndex::new(dataset.d, 6, false);
        index.train(&dataset.base, dataset.nb);
        index.add(&dataset.base, dataset.nb);

        let results = index.search(&dataset.query, dataset.nq, k, 1);
        let mut total_recall = 0.0;
        for q in 0..dataset.nq {
            let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / dataset.nq as f32;
        let storage = index.code_size();
        let ratio = fp32_size as f64 / storage as f64;

        println!("{:<25} | {:>8}B  | {:>11.2}% | {:>10.1}x", 
                 "TurboQuant 6-bit", storage, recall * 100.0, ratio);
    }

    // TurboQuant 4-bit + SQ8
    {
        let mut index = TurboQuantFlatIndex::new(dataset.d, 4, true);
        index.train(&dataset.base, dataset.nb);
        index.add(&dataset.base, dataset.nb);

        let results = index.search(&dataset.query, dataset.nq, k, 10);
        let mut total_recall = 0.0;
        for q in 0..dataset.nq {
            let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / dataset.nq as f32;
        let storage = index.total_storage() / index.ntotal();
        let ratio = fp32_size as f64 / storage as f64;

        println!("{:<25} | {:>8}B  | {:>11.2}% | {:>10.1}x", 
                 "TurboQuant 4-bit + SQ8", storage, recall * 100.0, ratio);
    }

    // RaBitQ Flat + SQ8
    {
        let mut index = rabitq::RaBitQFlatIndex::new(dataset.d, 1, false, true);
        index.train(&dataset.base, dataset.nb);
        index.add(&dataset.base, dataset.nb);

        let results = index.search(&dataset.query, dataset.nq, k, 10);
        let mut total_recall = 0.0;
        for q in 0..dataset.nq {
            let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / dataset.nq as f32;
        let storage = (index.code_size() + dataset.d) as f64;
        let ratio = fp32_size as f64 / storage;

        println!("{:<25} | {:>8.0}B | {:>11.2}% | {:>10.1}x", 
                 "RaBitQ Flat + SQ8", storage, recall * 100.0, ratio);
    }

    // RaBitQ IVF + SQ8
    {
        let mut index = ivf::RaBitQIVFIndex::new(dataset.d, 64, 1, false, true);
        index.train(&dataset.base, dataset.nb);
        index.add(&dataset.base, dataset.nb);

        let results = index.search(&dataset.query, dataset.nq, k, 32, 10);
        let mut total_recall = 0.0;
        for q in 0..dataset.nq {
            let ids: Vec<usize> = results[q].iter().map(|(i, _)| *i).collect();
            total_recall += compute_recall_sift(&ids, dataset.get_groundtruth(q), k);
        }
        let recall = total_recall / dataset.nq as f32;
        let storage = (index.code_size() + dataset.d) as f64;
        let ratio = fp32_size as f64 / storage;

        println!("{:<25} | {:>8.0}B | {:>11.2}% | {:>10.1}x", 
                 "RaBitQ IVF + SQ8", storage, recall * 100.0, ratio);
    }

    println!("{:-^70}", "");
}
