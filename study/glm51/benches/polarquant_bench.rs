use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use polarquant::{PolarQuant, PolarQuantBatch, PolarQuantConfig};

fn normalize_vector(x: &[f64]) -> Vec<f64> {
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return x.to_vec();
    }
    x.iter().map(|&v| v / norm).collect()
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress");

    for dim in [64, 128, 256, 512] {
        let config = PolarQuantConfig::builder(dim)
            .radius_bits(8)
            .angle_bits(4)
            .seed(42)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();
        let x: Vec<f64> = (0..dim).map(|i| ((i + 1) as f64).sin()).collect();
        let x = normalize_vector(&x);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| pq.compress(black_box(&x)).unwrap());
        });
    }

    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress");

    for dim in [64, 128, 256, 512] {
        let config = PolarQuantConfig::builder(dim)
            .radius_bits(8)
            .angle_bits(4)
            .seed(42)
            .build()
            .unwrap();
        let pq = PolarQuant::new(config).unwrap();
        let x: Vec<f64> = (0..dim).map(|i| ((i + 1) as f64).sin()).collect();
        let x = normalize_vector(&x);
        let compressed = pq.compress(&x).unwrap();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| pq.decompress(black_box(&compressed)));
        });
    }

    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    let config = PolarQuantConfig::builder(128)
        .radius_bits(8)
        .angle_bits(4)
        .seed(42)
        .build()
        .unwrap();
    let pq = PolarQuant::new(config).unwrap();
    let batch = PolarQuantBatch::new(&pq);

    let vectors: Vec<Vec<f64>> = (0..100)
        .map(|seed| {
            let x: Vec<f64> = (0..128)
                .map(|i| ((seed * 128 + i + 1) as f64 * 0.7).sin())
                .collect();
            normalize_vector(&x)
        })
        .collect();

    c.bench_function("batch_compress_100x128", |b| {
        b.iter(|| batch.compress_batch(black_box(&vectors)));
    });

    let compressed = batch.compress_batch(&vectors);

    c.bench_function("batch_decompress_100x128", |b| {
        b.iter(|| batch.decompress_batch(black_box(&compressed)));
    });
}

fn bench_lloyd_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("lloyd_max");

    for bits in [2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("bits", bits), &bits, |b, &bits| {
            b.iter(|| polarquant::compute_lloyd_max_centroids(32.0, 32.0, bits, 100));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_batch,
    bench_lloyd_max
);
criterion_main!(benches);
