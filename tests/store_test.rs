//! RocksDB 持久化测试

use std::path::Path;

use ::turboquant::store::VectorStore;
use ::turboquant::*;

fn compute_recall_test(result: &[usize], gt: &[usize], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k {
        if result.contains(&gt[i]) {
            hits += 1;
        }
    }
    hits as f32 / k as f32
}

#[test]
fn test_turboquant_persistence() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("tq_test");

    let d = 64;
    let nb = 500;
    let nq = 10;
    let k = 5;

    let data = utils::generate_clustered_data(nb, d, 10, 0.1, 42);
    let queries = utils::generate_queries(&data, nb, nq, d, 0.05, 123);
    let gt = utils::compute_ground_truth(&data, &queries, nb, nq, d, k);

    let mut index = TurboQuantFlatIndex::new(d, 4, true);
    index.train(&data, nb);
    index.add(&data, nb);

    let results_before = index.search(&queries, nq, k, 10);

    let store = VectorStore::open(&db_path).unwrap();
    store.save_turboquant(&index).unwrap();

    let stats = store.stats().unwrap();
    println!(
        "TurboQuant 存储: {:?} (codes={}, sq8={})",
        stats.index_type, stats.code_count, stats.sq8_count
    );
    assert_eq!(stats.ntotal, nb);
    assert_eq!(stats.code_count, nb);
    assert_eq!(stats.sq8_count, nb);

    let loaded = store.load_turboquant().unwrap();
    let results_after = loaded.search(&queries, nq, k, 10);

    let mut recall_before = 0.0_f32;
    let mut recall_after = 0.0_f32;
    for q in 0..nq {
        let ids_before: Vec<usize> = results_before[q].iter().map(|(i, _)| *i).collect();
        let ids_after: Vec<usize> = results_after[q].iter().map(|(i, _)| *i).collect();
        let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
        recall_before += compute_recall_test(&ids_before, &gt_q, k);
        recall_after += compute_recall_test(&ids_after, &gt_q, k);
    }
    recall_before /= nq as f32;
    recall_after /= nq as f32;

    println!(
        "TurboQuant 4-bit+SQ8 持久化前后召回率: {:.4} vs {:.4}",
        recall_before, recall_after
    );
    assert!(
        (recall_before - recall_after).abs() < 0.01,
        "持久化前后召回率不一致"
    );
}

#[test]
fn test_rabitq_flat_persistence() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("rabitq_flat_test");

    let d = 64;
    let nb = 500;
    let nq = 10;
    let k = 5;

    let data = utils::generate_clustered_data(nb, d, 10, 0.1, 42);
    let queries = utils::generate_queries(&data, nb, nq, d, 0.05, 123);
    let gt = utils::compute_ground_truth(&data, &queries, nb, nq, d, k);

    let mut index = rabitq::RaBitQFlatIndex::new(d, 1, false, true);
    index.train(&data, nb);
    index.add(&data, nb);

    let results_before = index.search(&queries, nq, k, 10);

    let store = VectorStore::open(&db_path).unwrap();
    store.save_rabitq_flat(&index).unwrap();

    let stats = store.stats().unwrap();
    println!(
        "RaBitQ Flat 存储: {:?} (codes={}, sq8={})",
        stats.index_type, stats.code_count, stats.sq8_count
    );
    assert_eq!(stats.ntotal, nb);

    let loaded = store.load_rabitq_flat().unwrap();
    let results_after = loaded.search(&queries, nq, k, 10);

    let mut recall_before = 0.0_f32;
    let mut recall_after = 0.0_f32;
    for q in 0..nq {
        let ids_before: Vec<usize> = results_before[q].iter().map(|(i, _)| *i).collect();
        let ids_after: Vec<usize> = results_after[q].iter().map(|(i, _)| *i).collect();
        let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
        recall_before += compute_recall_test(&ids_before, &gt_q, k);
        recall_after += compute_recall_test(&ids_after, &gt_q, k);
    }
    recall_before /= nq as f32;
    recall_after /= nq as f32;

    println!(
        "RaBitQ Flat+SQ8 持久化前后召回率: {:.4} vs {:.4}",
        recall_before, recall_after
    );
    assert!(
        (recall_before - recall_after).abs() < 0.01,
        "持久化前后召回率不一致"
    );
}

#[test]
fn test_rabitq_ivf_persistence() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("rabitq_ivf_test");

    let d = 64;
    let nb = 1000;
    let nq = 10;
    let k = 5;
    let nlist = 8;

    let data = utils::generate_clustered_data(nb, d, 10, 0.1, 42);
    let queries = utils::generate_queries(&data, nb, nq, d, 0.05, 123);
    let gt = utils::compute_ground_truth(&data, &queries, nb, nq, d, k);

    let mut index = ivf::RaBitQIVFIndex::new(d, nlist, 1, false, true);
    index.train(&data, nb);
    index.add(&data, nb);

    let results_before = index.search(&queries, nq, k, nlist, 10);

    let store = VectorStore::open(&db_path).unwrap();
    store.save_rabitq_ivf(&index).unwrap();

    let stats = store.stats().unwrap();
    println!(
        "RaBitQ IVF 存储: {:?} (codes={}, sq8={})",
        stats.index_type, stats.code_count, stats.sq8_count
    );
    assert_eq!(stats.ntotal, nb);

    let loaded = store.load_rabitq_ivf().unwrap();
    let results_after = loaded.search(&queries, nq, k, nlist, 10);

    let mut recall_before = 0.0_f32;
    let mut recall_after = 0.0_f32;
    for q in 0..nq {
        let ids_before: Vec<usize> = results_before[q].iter().map(|(i, _)| *i).collect();
        let ids_after: Vec<usize> = results_after[q].iter().map(|(i, _)| *i).collect();
        let gt_q: Vec<usize> = gt[q].iter().map(|&i| i).collect();
        recall_before += compute_recall_test(&ids_before, &gt_q, k);
        recall_after += compute_recall_test(&ids_after, &gt_q, k);
    }
    recall_before /= nq as f32;
    recall_after /= nq as f32;

    println!(
        "RaBitQ IVF+SQ8 持久化前后召回率: {:.4} vs {:.4}",
        recall_before, recall_after
    );
    assert!(
        (recall_before - recall_after).abs() < 0.01,
        "持久化前后召回率不一致"
    );
}

#[test]
fn test_incremental_insert_and_delete() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("incremental_test");

    let store = VectorStore::open(&db_path).unwrap();

    let code = vec![0xABu8, 0xCD, 0xEF, 0x01];
    let sq8 = vec![0x12u8, 0x34, 0x56, 0x78];

    store
        .insert_turboquant_vector(42, &code, Some(&sq8))
        .unwrap();

    let loaded_code = store.get_code(42).unwrap().unwrap();
    assert_eq!(loaded_code, code);

    let loaded_sq8 = store.get_sq8_code(42).unwrap().unwrap();
    assert_eq!(loaded_sq8, sq8);

    store.delete_vector(42).unwrap();

    assert!(store.get_code(42).unwrap().is_none());
    assert!(store.get_sq8_code(42).unwrap().is_none());

    println!("增量插入/删除测试通过");
}

#[test]
fn test_rabitq_ivf_incremental_insert() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("ivf_incremental_test");

    let store = VectorStore::open(&db_path).unwrap();

    let code = vec![0xFFu8; 24];
    let sq8 = vec![0xAAu8; 64];

    store
        .insert_rabitq_ivf_vector(100, 3, &code, Some(&sq8))
        .unwrap();

    let loaded_code = store.get_code(100).unwrap().unwrap();
    assert_eq!(loaded_code, code);

    let loaded_sq8 = store.get_sq8_code(100).unwrap().unwrap();
    assert_eq!(loaded_sq8, sq8);

    println!("RaBitQ IVF 增量插入测试通过");
}
