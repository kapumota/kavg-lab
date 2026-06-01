use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kavg_lab::config::ExperimentConfig;
use kavg_lab::functions::build_function;
use kavg_lab::kernels::build_kernel;
use kavg_lab::optimization::averages::AverageKind;
use kavg_lab::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use std::path::Path;

fn bench_kernel_average(c: &mut Criterion) {
    let config = ExperimentConfig::from_yaml_file(Path::new("examples/quadratic_l1.yaml")).unwrap();
    let f1 = build_function(&config.f1).unwrap();
    let f2 = build_function(&config.f2).unwrap();
    let kernel = build_kernel(&config.kernel).unwrap();
    let point = config.points[0].clone();

    c.bench_function("kernel_average_single_point", |b| {
        b.iter(|| {
            solve_kernel_average(KernelAverageInput {
                f1: f1.as_ref(),
                f2: f2.as_ref(),
                kernel: kernel.as_ref(),
                lambda1: config.lambda1,
                x: black_box(&point),
                solver: &config.solver,
                average_kind: AverageKind::Kernel,
            })
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_kernel_average);
criterion_main!(benches);
