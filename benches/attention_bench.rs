use criterion::{criterion_group, criterion_main, Criterion};
use kavg_lab::attention::run_attention_demo;
use kavg_lab::config::AttentionConfig;
use std::path::Path;

fn bench_attention(c: &mut Criterion) {
    let config =
        AttentionConfig::from_yaml_file(Path::new("examples/attention_demo.yaml")).unwrap();
    c.bench_function("attention_demo", |b| {
        b.iter(|| run_attention_demo(&config).unwrap())
    });
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
