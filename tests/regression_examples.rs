use kavg_lab::attention::run_attention_demo;
use kavg_lab::config::AttentionConfig;
use kavg_lab::parallel::ExecutionMode;
use kavg_lab::profile::profile_agent_sweep;
use std::path::Path;

#[test]
fn phase6_sparsemax_example_still_runs() {
    let config =
        AttentionConfig::from_yaml_file(Path::new("examples/fase6_attention_sparsemax.yaml"))
            .expect("fase6 sparsemax debe parsear");
    let results = run_attention_demo(&config).expect("fase6 sparsemax debe ejecutar");
    assert!(!results.is_empty());
    assert!(results.iter().all(|row| row.attention_rule == "sparsemax"));
}

#[test]
fn phase6_topk_example_still_runs() {
    let config = AttentionConfig::from_yaml_file(Path::new("examples/fase6_attention_topk.yaml"))
        .expect("fase6 top-k debe parsear");
    let results = run_attention_demo(&config).expect("fase6 top-k debe ejecutar");
    assert!(!results.is_empty());
    for result in results {
        let positive = result
            .standard_weights
            .iter()
            .filter(|value| **value > 1.0e-10)
            .count();
        assert!(
            positive
                <= config
                    .attention_top_k
                    .unwrap_or(config.top_k.unwrap_or(positive))
        );
    }
}

#[test]
fn profile_agent_sweep_one_repeat_runs() {
    let record = profile_agent_sweep(
        Path::new("examples/attention_sweep.yaml"),
        1,
        ExecutionMode::Sequential,
        "sequential",
    )
    .expect("profile debe ejecutar una repetición");
    assert_eq!(record.repeat, 1);
    assert!(record.mean_ms >= 0.0);
    assert!(record.max_ms >= record.min_ms);
}
