use kavg_lab::attention::{run_agent_sweep, run_attention_demo, run_multihead_attention_demo};
use kavg_lab::config::{
    AgentSweepConfig, AttentionConfig, ExperimentConfig, MultiHeadAttentionConfig,
};
use kavg_lab::fenchel::{verify_fenchel_identity, FenchelIdentityInput};
use kavg_lab::functions::{build_conjugate_function, build_function};
use kavg_lab::kernels::build_kernel;
use kavg_lab::optimization::averages::AverageKind;
use kavg_lab::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use std::fs;
use std::path::{Path, PathBuf};

fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn assert_probability_vector(values: &[f64]) {
    let sum: f64 = values.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1.0e-6,
        "la distribución debe sumar 1, pero sumó {sum}"
    );
    assert!(
        values.iter().all(|v| v.is_finite() && *v >= -1.0e-10),
        "la distribución contiene valores inválidos: {values:?}"
    );
}

fn temporary_yaml_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "kavg_lab_{name}_{}_{}.yaml",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

#[test]
fn config_validation_accepts_real_yaml_and_rejects_invalid_causal_shape() {
    let experiment = ExperimentConfig::from_yaml_file(&example("quadratic_l1.yaml"))
        .expect("quadratic_l1.yaml debe ser válido");
    assert_eq!(experiment.dimension, 2);
    assert_eq!(experiment.points.len(), 3);

    let attention = AttentionConfig::from_yaml_file(&example("attention_demo.yaml"))
        .expect("attention_demo.yaml debe ser válido");
    assert_eq!(attention.dimension, 3);
    assert_eq!(attention.keys.len(), 4);

    let causal_attention = AttentionConfig::from_yaml_file(&example("attention_causal.yaml"))
        .expect("attention_causal.yaml debe ser válido con queries.len() == keys.len()");
    assert_eq!(causal_attention.queries.len(), causal_attention.keys.len());

    let invalid_causal = r#"
dimension: 2
temperature: 1.0
kernel_gamma: 0.5
attention_solver:
  method: projected-gradient
  initial_step: 0.1
  tolerance: 1e-8
  min_step: 1e-10
  max_iterations: 100
mask:
  type: causal
queries:
  - [1.0, 0.0]
keys:
  - [1.0, 0.0]
  - [0.0, 1.0]
values:
  - [1.0]
  - [0.0]
"#;
    let path = temporary_yaml_path("invalid_causal");
    fs::write(&path, invalid_causal).expect("no se pudo escribir YAML temporal");
    let error = AttentionConfig::from_yaml_file(&path)
        .expect_err("la máscara causal debe rechazar configs con queries.len() != keys.len()");
    let message = format!("{error:#}");
    assert!(
        message.contains("mask.type=causal") && message.contains("queries.len() == keys.len()"),
        "mensaje de error inesperado: {message}"
    );
    let _ = fs::remove_file(path);
}

#[test]
fn attention_demo_runs_with_real_yaml() {
    let config = AttentionConfig::from_yaml_file(&example("attention_demo.yaml"))
        .expect("attention_demo.yaml debe parsear");
    let results = run_attention_demo(&config).expect("attention-demo debe ejecutarse");

    assert_eq!(results.len(), config.queries.len());
    for result in results {
        assert_eq!(result.standard_weights.len(), config.keys.len());
        assert_eq!(result.regularized_weights.len(), config.keys.len());
        assert_probability_vector(&result.standard_weights);
        assert_probability_vector(&result.regularized_weights);
        assert!(result.weight_l1_distance.is_finite());
        assert!(result.weight_l2_distance.is_finite());
        assert!(result.kl_regularized_to_softmax.is_finite());
        assert!(result.js_softmax_regularized.is_finite());
        assert!(result.effective_tokens_regularized >= 1.0 - 1.0e-8);
    }
}

#[test]
fn multihead_attention_runs_with_real_yaml() {
    let config = MultiHeadAttentionConfig::from_yaml_file(&example("multihead_attention.yaml"))
        .expect("multihead_attention.yaml debe parsear");
    let results =
        run_multihead_attention_demo(&config).expect("multihead-attention-demo debe ejecutarse");

    assert_eq!(results.len(), config.queries.len());
    for result in results {
        assert_eq!(result.heads.len(), config.heads.len());
        assert!(result.average_entropy.is_finite());
        assert!(result.mean_pairwise_l1.is_finite());
        assert!(result.mean_pairwise_l2.is_finite());
        assert!(result.mean_pairwise_js.is_finite());
        for head in result.heads {
            assert_probability_vector(&head.regularized_weights);
            assert!(head.entropy.is_finite());
            assert!(head.effective_tokens >= 1.0 - 1.0e-8);
        }
    }
}

#[test]
fn agent_sweep_runs_with_real_yaml() {
    let config = AgentSweepConfig::from_yaml_file(&example("attention_sweep.yaml"))
        .expect("attention_sweep.yaml debe parsear");
    let results = run_agent_sweep(&config).expect("agent-sweep debe ejecutarse");

    assert!(!results.is_empty());
    if let Some(limit) = config.output_limit {
        assert!(results.len() <= limit);
    }
    for (index, result) in results.iter().enumerate() {
        assert_eq!(result.rank, index + 1);
        assert!(result.score.is_finite());
        assert!(result.mean_regularized_entropy.is_finite());
        assert!(result.mean_distance_to_prior.is_finite());
        assert!(result.mean_difference_from_softmax.is_finite());
        assert!(result.mean_output_shift.is_finite());
        assert!(result.mean_js_softmax_regularized.is_finite());
    }
}

#[test]
fn compute_quadratic_l1_runs_with_real_yaml() {
    let experiment = ExperimentConfig::from_yaml_file(&example("quadratic_l1.yaml"))
        .expect("quadratic_l1.yaml debe parsear");
    let f1 = build_function(&experiment.f1).expect("f1 debe construirse");
    let f2 = build_function(&experiment.f2).expect("f2 debe construirse");
    let kernel = build_kernel(&experiment.kernel).expect("kernel debe construirse");
    let lambda1 = experiment.lambda1;

    for point in &experiment.points {
        let result = solve_kernel_average(KernelAverageInput {
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            kernel: kernel.as_ref(),
            lambda1,
            x: point,
            solver: &experiment.solver,
            average_kind: AverageKind::Kernel,
        })
        .expect("compute quadratic_l1 debe resolver cada punto");

        assert!(result.value.is_finite());
        assert_eq!(result.y1.len(), experiment.dimension);
        assert_eq!(result.y2.len(), experiment.dimension);
        assert!(result.solver_metric.is_finite());
    }
}

#[test]
fn verify_fenchel_quadratic_l2_runs_with_real_yaml() {
    let experiment = ExperimentConfig::from_yaml_file(&example("fenchel_quadratic_l2.yaml"))
        .expect("fenchel_quadratic_l2.yaml debe parsear");
    let f1 = build_function(&experiment.f1).expect("f1 debe construirse");
    let f2 = build_function(&experiment.f2).expect("f2 debe construirse");
    let f1_star = build_conjugate_function(&experiment.f1).expect("f1* debe construirse");
    let f2_star = build_conjugate_function(&experiment.f2).expect("f2* debe construirse");
    let kernel = build_kernel(&experiment.kernel).expect("kernel debe construirse");

    for (index, dual_point) in experiment.points.iter().enumerate() {
        let result = verify_fenchel_identity(FenchelIdentityInput {
            index,
            dual_point,
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            f1_star: f1_star.as_ref(),
            f2_star: f2_star.as_ref(),
            kernel: kernel.as_ref(),
            lambda1: experiment.lambda1,
            solver: &experiment.solver,
        })
        .expect("verify-fenchel quadratic_l2 debe ejecutarse");

        assert!(result.left_approx.is_finite());
        assert!(result.right_value.is_finite());
        assert!(result.absolute_error.is_finite());
        assert!(
            result.passed || result.absolute_error < 1.0e-4,
            "Fenchel check no pasó para {:?}: error absoluto {}",
            dual_point,
            result.absolute_error
        );
    }
}
