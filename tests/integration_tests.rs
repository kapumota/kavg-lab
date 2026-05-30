use kavg_lab::attention::{run_agent_sweep, run_attention_demo, run_multihead_attention_demo};
use kavg_lab::config::{
    AgentSweepConfig, AttentionConfig, ExperimentConfig, MultiHeadAttentionConfig,
};
use kavg_lab::fenchel::{verify_fenchel_identity, FenchelIdentityInput};
use kavg_lab::functions::{build_conjugate_function, build_function, ConvexFunction};
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

#[test]
fn compute_results_can_be_exported_as_json_and_manifest() {
    use kavg_lab::io::json_export::{
        export_compute_results_json, export_execution_manifest, ExecutionManifest,
    };
    use std::time::SystemTime;

    let experiment = ExperimentConfig::from_yaml_file(&example("quadratic_l1.yaml"))
        .expect("quadratic_l1.yaml debe parsear");
    let f1 = build_function(&experiment.f1).expect("f1 debe construirse");
    let f2 = build_function(&experiment.f2).expect("f2 debe construirse");
    let kernel = build_kernel(&experiment.kernel).expect("kernel debe construirse");

    let mut results = Vec::new();
    for (index, point) in experiment.points.iter().enumerate() {
        let result = solve_kernel_average(KernelAverageInput {
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            kernel: kernel.as_ref(),
            lambda1: experiment.lambda1,
            x: point,
            solver: &experiment.solver,
            average_kind: AverageKind::Kernel,
        })
        .expect("compute quadratic_l1 debe resolver cada punto")
        .with_index_and_point(index, point.clone());
        results.push(result);
    }

    let json_path = std::env::temp_dir().join(format!(
        "kavg_lab_compute_results_{}.json",
        std::process::id()
    ));
    let manifest_path = std::env::temp_dir().join(format!(
        "kavg_lab_compute_manifest_{}.json",
        std::process::id()
    ));

    export_compute_results_json(&json_path, &results).expect("debe exportar JSON");
    export_execution_manifest(
        &manifest_path,
        &ExecutionManifest {
            command: "compute",
            config_path: &example("quadratic_l1.yaml"),
            csv_output: None,
            json_output: Some(&json_path),
            started_at: SystemTime::now(),
            finished_at: SystemTime::now(),
            result_count: results.len(),
            status: "passed",
        },
    )
    .expect("debe exportar manifiesto");

    let json_text = fs::read_to_string(&json_path).expect("debe leer JSON");
    let manifest_text = fs::read_to_string(&manifest_path).expect("debe leer manifiesto");

    assert!(json_text.contains("\"command\": \"compute\""));
    assert!(json_text.contains("\"results\""));
    assert!(manifest_text.contains("\"config_hash_fnv1a64\""));
    assert!(manifest_text.contains("\"result_count\""));

    let _ = fs::remove_file(json_path);
    let _ = fs::remove_file(manifest_path);
}


#[test]
fn phase2_convex_functions_and_kernels_parse_and_run() {
    let experiment = ExperimentConfig::from_yaml_file(&example("fase2_elastic_box_mahalanobis.yaml"))
        .expect("fase2_elastic_box_mahalanobis.yaml debe parsear");
    let f1 = build_function(&experiment.f1).expect("f1 elastic-net debe construirse");
    let f2 = build_function(&experiment.f2).expect("f2 indicator-box debe construirse");
    let kernel = build_kernel(&experiment.kernel).expect("kernel mahalanobis debe construirse");

    assert_eq!(f1.name(), "elastic-net");
    assert_eq!(f2.name(), "indicator-box");
    assert_eq!(kernel.name(), "mahalanobis");

    for point in &experiment.points {
        let result = solve_kernel_average(KernelAverageInput {
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            kernel: kernel.as_ref(),
            lambda1: experiment.lambda1,
            x: point,
            solver: &experiment.solver,
            average_kind: AverageKind::Kernel,
        })
        .expect("el caso Fase 2 con OSQP debe resolver cada punto");

        assert!(result.value.is_finite());
        assert_eq!(result.y1.len(), experiment.dimension);
        assert_eq!(result.y2.len(), experiment.dimension);
        assert!(result.y2.iter().all(|v| *v >= -1.0 - 1.0e-6 && *v <= 1.0 + 1.0e-6));
    }
}

#[test]
fn phase2_supports_extra_convex_configs_without_osqp() {
    let text = r#"
dimension: 2
lambda1: 0.4
f1:
  type: logistic-loss
  samples:
    - [1.0, 0.0]
    - [0.0, 1.0]
  labels: [1.0, -1.0]
  l2_alpha: 0.01
f2:
  type: max-affine
  pieces:
    - slope: [1.0, 0.0]
      intercept: 0.0
    - slope: [0.0, -1.0]
      intercept: 0.2
kernel:
  type: huber
  delta: 1.0
solver:
  method: subgradient
  initial_step: 0.05
  tolerance: 1.0e-6
  min_step: 1.0e-10
  max_iterations: 200
points:
  - [0.1, -0.2]
"#;
    let path = temporary_yaml_path("phase2_extra");
    fs::write(&path, text).expect("no se pudo escribir YAML temporal Fase 2");
    let experiment = ExperimentConfig::from_yaml_file(&path)
        .expect("config Fase 2 con logistic, max-affine y huber debe parsear");
    let f1 = build_function(&experiment.f1).expect("logistic-loss debe construirse");
    let f2 = build_function(&experiment.f2).expect("max-affine debe construirse");
    let kernel = build_kernel(&experiment.kernel).expect("huber kernel debe construirse");

    let result = solve_kernel_average(KernelAverageInput {
        f1: f1.as_ref(),
        f2: f2.as_ref(),
        kernel: kernel.as_ref(),
        lambda1: experiment.lambda1,
        x: &experiment.points[0],
        solver: &experiment.solver,
        average_kind: AverageKind::Kernel,
    })
    .expect("subgradient debe resolver el caso Fase 2 no cuadrático");

    assert!(result.value.is_finite());
    assert_eq!(result.y1.len(), experiment.dimension);
    let _ = fs::remove_file(path);
}

#[test]
fn l1_conjugate_is_linf_ball_indicator() {
    let function = kavg_lab::functions::L1ConjugateFunction::new(0.5)
        .expect("conjugado L1 debe construirse");
    assert_eq!(function.value(&[0.2, -0.5]), 0.0);
    assert!(function.value(&[0.2, -0.7]).is_infinite());
}
