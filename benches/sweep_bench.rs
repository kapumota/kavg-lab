use criterion::{criterion_group, criterion_main, Criterion};
use kavg_lab::attention::run_agent_sweep;
use kavg_lab::config::AgentSweepConfig;
use std::path::Path;

fn bench_sweep(c: &mut Criterion) {
    let config =
        AgentSweepConfig::from_yaml_file(Path::new("examples/attention_sweep.yaml")).unwrap();
    c.bench_function("agent_sweep", |b| {
        b.iter(|| run_agent_sweep(&config).unwrap())
    });
}

criterion_group!(benches, bench_sweep);
criterion_main!(benches);
