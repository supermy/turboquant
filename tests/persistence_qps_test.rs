//! 持久化性能测试: 插入与查询 QPS

use std::path::Path;
use std::time::Instant;

use ::turboquant::*;
use ::turboquant::store::VectorStore;

fn compute_recall_test(result: &[usize], gt: &[usize], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k {
        if result.contains(&gt[i]) { hits += 1; }
    }
    hits as f32 / k as f32
}

#[test]
fn test_persistence_qps_benchmark() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("qps_test");

    let d = 128;
    let nb = 10000;
    let nq = 100;
    let k = 10;

    println!("\n{}", "=".repeat(80));
    println!("持久化性能测试: 插入与查询 QPS");
    println!("{}", "=".repeat(80));
    println!("数据集: {} 向量 x {} 维度", nb, d);
    println!("查询数: {}", nq);

    let data = utils::generate_clustered_data(nb, d, 100, 0.1, 42);
    let queries = utils::generate_queries(&data, nb, nq, d, 0.05, 123);
    let gt = utils::compute_ground_truth(&data, &queries, nb, nq, d, k);

    struct BenchResult {
        name: String,
        insert_qps: f64,
        search_qps: f64,
        latency_ms: f64,
        recall: f32,
        save_time_ms: f64,
        load_time_ms: f64,
        storage_bytes: usize,
    }

    let mut results: Vec<BenchResult> = Vec::new();

    // ========== 1. TurboQuant 4-bit + SQ8 ==========
    {
        println!("\n[1/3] TurboQuant 4-bit + SQ8 ...");

        // 内存插入
        let t0 = Instant::now();
        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&data, nb);
        index.add(&data, nb);
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let insert_qps = nb as f64 / (insert_ms / 1000.0);

        // 内存查询
        let t1 = Instant::now();
        let res = index.search(&queries, nq, k, 10);
        let search_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let search_qps = nq as f64 / (search_ms / 1000.0);

        // 召回率
        let mut recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
            recall += compute_recall_test(&ids, &gt_q, k);
        }
        recall /= nq as f32;

        // 持久化
        let store = VectorStore::open(&db_path).unwrap();
        let t2 = Instant::now();
        store.save_turboquant(&index).unwrap();
        let save_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 加载
        let t3 = Instant::now();
        let loaded = store.load_turboquant().unwrap();
        let load_ms = t3.elapsed().as_secs_f64() * 1000.0;

        // 加载后查询
        let t4 = Instant::now();
        let _res2 = loaded.search(&queries, nq, k, 10);
        let search2_ms = t4.elapsed().as_secs_f64() * 1000.0;
        let search2_qps = nq as f64 / (search2_ms / 1000.0);

        let stats = store.stats().unwrap();
        let storage_bytes = stats.code_count * index.code_size() + stats.sq8_count * d;

        println!("  内存插入: {:.0} QPS ({:.1}ms)", insert_qps, insert_ms);
        println!("  内存查询: {:.0} QPS ({:.2}ms/query)", search_qps, search_ms / nq as f64);
        println!("  持久化:   {:.1}ms", save_ms);
        println!("  加载:     {:.1}ms", load_ms);
        println!("  加载后查: {:.0} QPS", search2_qps);
        println!("  召回率:   {:.2}%", recall * 100.0);

        results.push(BenchResult {
            name: "TurboQuant 4-bit+SQ8".to_string(),
            insert_qps,
            search_qps,
            latency_ms: search_ms / nq as f64,
            recall,
            save_time_ms: save_ms,
            load_time_ms: load_ms,
            storage_bytes,
        });
    }

    // ========== 2. RaBitQ Flat + SQ8 ==========
    {
        println!("\n[2/3] RaBitQ Flat + SQ8 ...");

        let db_path2 = tmp_dir.path().join("rabitq_flat_qps");

        let t0 = Instant::now();
        let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let insert_qps = nb as f64 / (insert_ms / 1000.0);

        let t1 = Instant::now();
        let res = index.search(&queries, nq, k, 10);
        let search_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let search_qps = nq as f64 / (search_ms / 1000.0);

        let mut recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
            recall += compute_recall_test(&ids, &gt_q, k);
        }
        recall /= nq as f32;

        let store = VectorStore::open(&db_path2).unwrap();
        let t2 = Instant::now();
        store.save_rabitq_flat(&index).unwrap();
        let save_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        let loaded = store.load_rabitq_flat().unwrap();
        let load_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let t4 = Instant::now();
        let _res2 = loaded.search(&queries, nq, k, 10);
        let search2_ms = t4.elapsed().as_secs_f64() * 1000.0;
        let search2_qps = nq as f64 / (search2_ms / 1000.0);

        let stats = store.stats().unwrap();
        let storage_bytes = stats.code_count * index.code_size() + stats.sq8_count * d;

        println!("  内存插入: {:.0} QPS ({:.1}ms)", insert_qps, insert_ms);
        println!("  内存查询: {:.0} QPS ({:.2}ms/query)", search_qps, search_ms / nq as f64);
        println!("  持久化:   {:.1}ms", save_ms);
        println!("  加载:     {:.1}ms", load_ms);
        println!("  加载后查: {:.0} QPS", search2_qps);
        println!("  召回率:   {:.2}%", recall * 100.0);

        results.push(BenchResult {
            name: "RaBitQ Flat+SQ8".to_string(),
            insert_qps,
            search_qps,
            latency_ms: search_ms / nq as f64,
            recall,
            save_time_ms: save_ms,
            load_time_ms: load_ms,
            storage_bytes,
        });
    }

    // ========== 3. RaBitQ IVF + SQ8 ==========
    {
        println!("\n[3/3] RaBitQ IVF + SQ8 ...");

        let db_path3 = tmp_dir.path().join("rabitq_ivf_qps");
        let nlist = 64;

        let t0 = Instant::now();
        let mut index = ivf::RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&data, nb);
        index.add(&data, nb);
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let insert_qps = nb as f64 / (insert_ms / 1000.0);

        let t1 = Instant::now();
        let res = index.search(&queries, nq, k, 32, 10);
        let search_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let search_qps = nq as f64 / (search_ms / 1000.0);

        let mut recall = 0.0_f32;
        for q in 0..nq {
            let ids: Vec<usize> = res[q].iter().map(|(i, _)| *i).collect();
            let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
            recall += compute_recall_test(&ids, &gt_q, k);
        }
        recall /= nq as f32;

        let store = VectorStore::open(&db_path3).unwrap();
        let t2 = Instant::now();
        store.save_rabitq_ivf(&index).unwrap();
        let save_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        let loaded = store.load_rabitq_ivf().unwrap();
        let load_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let t4 = Instant::now();
        let _res2 = loaded.search(&queries, nq, k, 32, 10);
        let search2_ms = t4.elapsed().as_secs_f64() * 1000.0;
        let search2_qps = nq as f64 / (search2_ms / 1000.0);

        let stats = store.stats().unwrap();
        let storage_bytes = stats.code_count * index.code_size() + stats.sq8_count * d;

        println!("  内存插入: {:.0} QPS ({:.1}ms)", insert_qps, insert_ms);
        println!("  内存查询: {:.0} QPS ({:.2}ms/query)", search_qps, search_ms / nq as f64);
        println!("  持久化:   {:.1}ms", save_ms);
        println!("  加载:     {:.1}ms", load_ms);
        println!("  加载后查: {:.0} QPS", search2_qps);
        println!("  召回率:   {:.2}%", recall * 100.0);

        results.push(BenchResult {
            name: "RaBitQ IVF+SQ8".to_string(),
            insert_qps,
            search_qps,
            latency_ms: search_ms / nq as f64,
            recall,
            save_time_ms: save_ms,
            load_time_ms: load_ms,
            storage_bytes,
        });
    }

    // ========== 汇总表格 ==========
    println!("\n{}", "=".repeat(80));
    println!("性能汇总");
    println!("{}", "=".repeat(80));

    println!("\n{:<22} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} | {:>10}",
             "方法", "插入QPS", "查询QPS", "延迟(ms)", "保存(ms)", "加载(ms)", "存储(KB)");
    println!("{}", "-".repeat(90));

    for r in &results {
        println!("{:<22} | {:>10.0} | {:>10.0} | {:>10.2} | {:>8.1} | {:>8.1} | {:>10.1}",
                 r.name, r.insert_qps, r.search_qps, r.latency_ms,
                 r.save_time_ms, r.load_time_ms, r.storage_bytes as f64 / 1024.0);
    }

    println!("\n{}", "=".repeat(80));
    println!("关键发现");
    println!("{}", "=".repeat(80));

    let best_insert = results.iter().max_by(|a, b| a.insert_qps.partial_cmp(&b.insert_qps).unwrap()).unwrap();
    let best_search = results.iter().max_by(|a, b| a.search_qps.partial_cmp(&b.search_qps).unwrap()).unwrap();
    let best_recall = results.iter().max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap()).unwrap();

    println!("- 最高插入速度: {} ({:.0} QPS)", best_insert.name, best_insert.insert_qps);
    println!("- 最高查询速度: {} ({:.0} QPS)", best_search.name, best_search.search_qps);
    println!("- 最高召回率:   {} ({:.1}%)", best_recall.name, best_recall.recall * 100.0);
}

#[test]
fn test_incremental_insert_qps() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("incremental_qps");

    let d = 128;
    let nb = 1000;

    println!("\n{}", "=".repeat(80));
    println!("增量插入 QPS 测试");
    println!("{}", "=".repeat(80));

    let data = utils::generate_clustered_data(nb, d, 10, 0.1, 42);

    let mut index = TurboQuantFlatIndex::new(d, 4, true);
    index.train(&data, nb);

    let store = VectorStore::open(&db_path).unwrap();

    // 增量插入测试
    let code_sz = index.code_size();
    let t0 = Instant::now();
    for i in 0..nb {
        let xi = &data[i * d..(i + 1) * d];

        let mut code = vec![0u8; code_sz];
        index.quantizer.encode(&index.rotation.apply(xi), &mut code);

        let mut sq8_code = vec![0u8; d];
        if let Some(ref sq8) = index.sq8 {
            sq8.encode(xi, &mut sq8_code);
        }

        store.insert_turboquant_vector(i as u64, &code, Some(&sq8_code)).unwrap();
    }
    let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let insert_qps = nb as f64 / (insert_ms / 1000.0);

    println!("增量插入 {} 向量:", nb);
    println!("  耗时: {:.1}ms", insert_ms);
    println!("  QPS:  {:.0}", insert_qps);

    // 读取测试
    let t1 = Instant::now();
    for i in 0..nb {
        let _ = store.get_code(i as u64).unwrap();
    }
    let read_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let read_qps = nb as f64 / (read_ms / 1000.0);

    println!("\n随机读取 {} 向量:", nb);
    println!("  耗时: {:.1}ms", read_ms);
    println!("  QPS:  {:.0}", read_qps);
}
