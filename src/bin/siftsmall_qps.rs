use std::path::Path;
use std::time::Instant;

use turboquant::sift::SiftSmallDataset;
use turboquant::store::VectorStore;
use turboquant::rabitq;
use turboquant::ivf;
use turboquant::ivf::TurboQuantIVFIndex;
use turboquant::TurboQuantFlatIndex;

fn compute_recall(result: &[(usize, f32)], gt: &[i32], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k.min(gt.len()) {
        if result.iter().take(k).any(|(id, _)| *id == gt[i] as usize) {
            hits += 1;
        }
    }
    hits as f32 / k as f32
}

fn main() {
    let data_dir = Path::new("/Users/moyong/project/ai/models/data/siftsmall");
    if !data_dir.exists() {
        eprintln!("siftsmall data not found at {:?}", data_dir);
        std::process::exit(1);
    }

    let ds = SiftSmallDataset::load(data_dir).unwrap();
    println!("SIFT Small: {} base x {}D, {} queries", ds.nb, ds.d, ds.nq);

    let d = ds.d;
    let nb = ds.nb;
    let nq = ds.nq;
    let k = 10;

    let warmup_rounds = 3;
    let bench_rounds = 10;

    struct BenchResult {
        name: String,
        search_qps: f64,
        latency_us: f64,
        recall: f32,
        p99_us: f64,
        p50_us: f64,
    }

    let mut all_results: Vec<BenchResult> = Vec::new();

    // ========== 1. TurboQuant 4-bit + SQ8 ==========
    {
        println!("\n[1/7] TurboQuant 4-bit + SQ8 ...");
        let mut index = TurboQuantFlatIndex::new(d, 4, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for _ in 0..warmup_rounds {
            let _ = index.search(&ds.query, nq, k, 10);
        }

        let mut latencies = Vec::with_capacity(bench_rounds * nq);
        let mut total_ms = 0.0f64;
        for _ in 0..bench_rounds {
            let mut round_lat = Vec::with_capacity(nq);
            let t0 = Instant::now();
            for q in 0..nq {
                let tq = Instant::now();
                let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, 10);
                round_lat.push(tq.elapsed().as_micros() as f64);
            }
            total_ms += t0.elapsed().as_secs_f64() * 1000.0;
            latencies.extend(round_lat);
        }

        let res = index.search(&ds.query, nq, k, 10);
        let mut recall = 0.0f32;
        for q in 0..nq {
            recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
        }
        recall /= nq as f32;

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
        let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

        println!("  QPS: {:.0}, Recall@{}: {:.2}%, P50: {:.0}us, P99: {:.0}us",
                 search_qps, k, recall * 100.0, p50, p99);

        all_results.push(BenchResult {
            name: "TQ 4bit+SQ8".into(), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
        });
    }

    // ========== 2. RaBitQ Flat + SQ8 ==========
    {
        println!("[2/7] RaBitQ Flat + SQ8 ...");
        let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for _ in 0..warmup_rounds {
            let _ = index.search(&ds.query, nq, k, 10);
        }

        let mut latencies = Vec::with_capacity(bench_rounds * nq);
        let mut total_ms = 0.0f64;
        for _ in 0..bench_rounds {
            let mut round_lat = Vec::with_capacity(nq);
            let t0 = Instant::now();
            for q in 0..nq {
                let tq = Instant::now();
                let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, 10);
                round_lat.push(tq.elapsed().as_micros() as f64);
            }
            total_ms += t0.elapsed().as_secs_f64() * 1000.0;
            latencies.extend(round_lat);
        }

        let res = index.search(&ds.query, nq, k, 10);
        let mut recall = 0.0f32;
        for q in 0..nq {
            recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
        }
        recall /= nq as f32;

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
        let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

        println!("  QPS: {:.0}, Recall@{}: {:.2}%, P50: {:.0}us, P99: {:.0}us",
                 search_qps, k, recall * 100.0, p50, p99);

        all_results.push(BenchResult {
            name: "RaBitQ Flat+SQ8".into(), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
        });
    }

    // ========== 3. RaBitQ IVF + SQ8 (nlist=64) ==========
    {
        println!("[3/7] RaBitQ IVF nlist=64 + SQ8 ...");
        let nlist = 64;
        let mut index = ivf::RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for nprobe in [8, 16, 32] {
            for _ in 0..warmup_rounds {
                let _ = index.search(&ds.query, nq, k, nprobe, 10);
            }

            let mut latencies = Vec::with_capacity(bench_rounds * nq);
            let mut total_ms = 0.0f64;
            for _ in 0..bench_rounds {
                let mut round_lat = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for q in 0..nq {
                    let tq = Instant::now();
                    let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, nprobe, 10);
                    round_lat.push(tq.elapsed().as_micros() as f64);
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                latencies.extend(round_lat);
            }

            let res = index.search(&ds.query, nq, k, nprobe, 10);
            let mut recall = 0.0f32;
            for q in 0..nq {
                recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
            }
            recall /= nq as f32;

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = latencies[latencies.len() / 2];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
            let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

            println!("  nprobe={}: QPS={:.0}, Recall@{}={:.2}%, P50={:.0}us, P99={:.0}us",
                     nprobe, search_qps, k, recall * 100.0, p50, p99);

            all_results.push(BenchResult {
                name: format!("IVF-64 np={}", nprobe), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
            });
        }
    }

    // ========== 4. RaBitQ IVF + SQ8 (nlist=256) ==========
    {
        println!("[4/7] RaBitQ IVF nlist=256 + SQ8 ...");
        let nlist = 256;
        let mut index = ivf::RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for nprobe in [8, 16, 32, 64] {
            for _ in 0..warmup_rounds {
                let _ = index.search(&ds.query, nq, k, nprobe, 10);
            }

            let mut latencies = Vec::with_capacity(bench_rounds * nq);
            let mut total_ms = 0.0f64;
            for _ in 0..bench_rounds {
                let mut round_lat = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for q in 0..nq {
                    let tq = Instant::now();
                    let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, nprobe, 10);
                    round_lat.push(tq.elapsed().as_micros() as f64);
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                latencies.extend(round_lat);
            }

            let res = index.search(&ds.query, nq, k, nprobe, 10);
            let mut recall = 0.0f32;
            for q in 0..nq {
                recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
            }
            recall /= nq as f32;

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = latencies[latencies.len() / 2];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
            let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

            println!("  nprobe={}: QPS={:.0}, Recall@{}={:.2}%, P50={:.0}us, P99={:.0}us",
                     nprobe, search_qps, k, recall * 100.0, p50, p99);

            all_results.push(BenchResult {
                name: format!("IVF-256 np={}", nprobe), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
            });
        }
    }

    // ========== 5. TurboQuant IVF + SQ8 (nlist=64) ==========
    {
        println!("[5/7] TurboQuant IVF nlist=64 + SQ8 ...");
        let nlist = 64;
        let mut index = TurboQuantIVFIndex::new(d, nlist, 4, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for nprobe in [8, 16, 32] {
            for _ in 0..warmup_rounds {
                let _ = index.search(&ds.query, nq, k, nprobe, 10);
            }

            let mut latencies = Vec::with_capacity(bench_rounds * nq);
            let mut total_ms = 0.0f64;
            for _ in 0..bench_rounds {
                let mut round_lat = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for q in 0..nq {
                    let tq = Instant::now();
                    let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, nprobe, 10);
                    round_lat.push(tq.elapsed().as_micros() as f64);
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                latencies.extend(round_lat);
            }

            let res = index.search(&ds.query, nq, k, nprobe, 10);
            let mut recall = 0.0f32;
            for q in 0..nq {
                recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
            }
            recall /= nq as f32;

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = latencies[latencies.len() / 2];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
            let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

            println!("  nprobe={}: QPS={:.0}, Recall@{}={:.2}%, P50={:.0}us, P99={:.0}us",
                     nprobe, search_qps, k, recall * 100.0, p50, p99);

            all_results.push(BenchResult {
                name: format!("TQ-IVF-64 np={}", nprobe), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
            });
        }
    }

    // ========== 6. TurboQuant IVF + SQ8 (nlist=256) ==========
    {
        println!("[6/7] TurboQuant IVF nlist=256 + SQ8 ...");
        let nlist = 256;
        let mut index = TurboQuantIVFIndex::new(d, nlist, 4, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        for nprobe in [8, 16, 32] {
            for _ in 0..warmup_rounds {
                let _ = index.search(&ds.query, nq, k, nprobe, 10);
            }

            let mut latencies = Vec::with_capacity(bench_rounds * nq);
            let mut total_ms = 0.0f64;
            for _ in 0..bench_rounds {
                let mut round_lat = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for q in 0..nq {
                    let tq = Instant::now();
                    let _ = index.search(&ds.query[q * d..(q + 1) * d], 1, k, nprobe, 10);
                    round_lat.push(tq.elapsed().as_micros() as f64);
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                latencies.extend(round_lat);
            }

            let res = index.search(&ds.query, nq, k, nprobe, 10);
            let mut recall = 0.0f32;
            for q in 0..nq {
                recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
            }
            recall /= nq as f32;

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = latencies[latencies.len() / 2];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
            let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

            println!("  nprobe={}: QPS={:.0}, Recall@{}={:.2}%, P50={:.0}us, P99={:.0}us",
                     nprobe, search_qps, k, recall * 100.0, p50, p99);

            all_results.push(BenchResult {
                name: format!("TQ-IVF-256 np={}", nprobe), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
            });
        }
    }

    // ========== 7. RaBitQ IVF 持久化查询 ==========
    {
        println!("[7/7] RaBitQ IVF 持久化 + RocksDB 查询 ...");
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("ivf_qps");
        let nlist = 64;

        let mut index = ivf::RaBitQIVFIndex::new(d, nlist, 1, false, true);
        index.train(&ds.base, nb);
        index.add(&ds.base, nb);

        let store = VectorStore::open(&db_path).unwrap();
        store.save_rabitq_ivf(&index).unwrap();

        let loaded = store.load_rabitq_ivf().unwrap();

        for nprobe in [8, 16, 32] {
            for _ in 0..warmup_rounds {
                let _ = loaded.search(&ds.query, nq, k, nprobe, 10);
            }

            let mut latencies = Vec::with_capacity(bench_rounds * nq);
            let mut total_ms = 0.0f64;
            for _ in 0..bench_rounds {
                let mut round_lat = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for q in 0..nq {
                    let tq = Instant::now();
                    let _ = loaded.search(&ds.query[q * d..(q + 1) * d], 1, k, nprobe, 10);
                    round_lat.push(tq.elapsed().as_micros() as f64);
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                latencies.extend(round_lat);
            }

            let res = loaded.search(&ds.query, nq, k, nprobe, 10);
            let mut recall = 0.0f32;
            for q in 0..nq {
                recall += compute_recall(&res[q], ds.get_groundtruth(q), k);
            }
            recall /= nq as f32;

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = latencies[latencies.len() / 2];
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
            let search_qps = (nq * bench_rounds) as f64 / (total_ms / 1000.0);

            println!("  nprobe={}: QPS={:.0}, Recall@{}={:.2}%, P50={:.0}us, P99={:.0}us",
                     nprobe, search_qps, k, recall * 100.0, p50, p99);

            all_results.push(BenchResult {
                name: format!("Persisted np={}", nprobe), search_qps, latency_us: p50, recall, p99_us: p99, p50_us: p50,
            });
        }
    }

    // ========== 汇总 ==========
    println!("\n{}", "=".repeat(90));
    println!("SIFT Small QPS 基准测试结果 (d={}, nb={}, nq={}, k={})", d, nb, nq, k);
    println!("{}", "=".repeat(90));
    println!("{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
             "方法", "QPS", "P50(us)", "P99(us)", "Recall@10", "延迟比");
    println!("{}", "-".repeat(80));

    let baseline_p50 = all_results.first().map(|r| r.p50_us).unwrap_or(1.0);
    for r in &all_results {
        let ratio = r.p50_us / baseline_p50;
        println!("{:<20} | {:>10.0} | {:>10.0} | {:>10.0} | {:>9.2}% | {:>7.2}x",
                 r.name, r.search_qps, r.p50_us, r.p99_us, r.recall * 100.0, ratio);
    }

    let best_qps = all_results.iter().max_by(|a, b| a.search_qps.partial_cmp(&b.search_qps).unwrap()).unwrap();
    let best_recall = all_results.iter().max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap()).unwrap();
    println!("\n最高 QPS: {} ({:.0} QPS)", best_qps.name, best_qps.search_qps);
    println!("最高 Recall: {} ({:.2}%)", best_recall.name, best_recall.recall * 100.0);
}
