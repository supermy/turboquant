use criterion::{criterion_group, criterion_main, Criterion};
use turboquant::*;

fn bench_turboquant_search(c: &mut Criterion) {
    let d = 128;
    let nb = 10000;
    let data = utils::generate_clustered_data(nb, d, 100, 0.1, 42);

    let mut index = TurboQuantFlatIndex::new(d, 4, false);
    index.train(&data, nb);
    index.add(&data, nb);

    let queries = utils::generate_queries(&data, nb, 100, d, 0.05, 123);

    c.bench_function("turboquant_4bit_search_100q", |b| {
        b.iter(|| index.search(&queries, 100, 10, 1))
    });
}

fn bench_rabitq_ivf_search(c: &mut Criterion) {
    let d = 128;
    let nb = 10000;
    let data = utils::generate_clustered_data(nb, d, 100, 0.1, 42);

    let mut index = RaBitQIVFIndex::new(d, 64, 1, false, true);
    index.train(&data, nb);
    index.add(&data, nb);

    let queries = utils::generate_queries(&data, nb, 100, d, 0.05, 123);

    c.bench_function("rabitq_ivf_sq8_search_100q", |b| {
        b.iter(|| index.search(&queries, 100, 10, 32, 10))
    });
}

criterion_group!(benches, bench_turboquant_search, bench_rabitq_ivf_search);
criterion_main!(benches);
