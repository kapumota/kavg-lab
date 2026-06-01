use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kavg_lab::optimization::projections::project_to_simplex;

fn bench_simplex_projection(c: &mut Criterion) {
    let values: Vec<f64> = (0..1024).map(|index| (index as f64).sin()).collect();
    c.bench_function("simplex_projection_1024", |b| {
        b.iter(|| project_to_simplex(black_box(&values)))
    });
}

criterion_group!(benches, bench_simplex_projection);
criterion_main!(benches);
