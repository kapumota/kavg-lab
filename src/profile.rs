use crate::attention::run_agent_sweep_with_mode;
use crate::config::AgentSweepConfig;
use crate::parallel::ExecutionMode;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

/// Registro CSV compacto de una medición CLI.
#[derive(Debug, Clone)]
pub struct ProfileRecord {
    pub experiment: String,
    pub dimension: usize,
    pub n_queries: usize,
    pub n_keys: usize,
    pub solver: String,
    pub parallel: bool,
    pub jobs: String,
    pub repeat: usize,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_ms: f64,
}

/// Perfilado reproducible de `agent-sweep` usando `std::time::Instant`.
pub fn profile_agent_sweep(
    config_path: &Path,
    repeat: usize,
    mode: ExecutionMode,
    jobs_label: &str,
) -> Result<ProfileRecord> {
    anyhow::ensure!(repeat > 0, "--repeat debe ser mayor que cero.");
    let config = AgentSweepConfig::from_yaml_file(config_path)
        .with_context(|| format!("No se pudo cargar perfil: {}", config_path.display()))?;

    let mut elapsed_ms = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let started = Instant::now();
        let results = run_agent_sweep_with_mode(&config, mode)?;
        anyhow::ensure!(
            !results.is_empty(),
            "agent-sweep no produjo resultados durante profile."
        );
        elapsed_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let mean_ms = mean(&elapsed_ms);
    let min_ms = elapsed_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = elapsed_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let std_ms = stddev(&elapsed_ms, mean_ms);

    Ok(ProfileRecord {
        experiment: config_path.display().to_string(),
        dimension: config.base_attention.dimension,
        n_queries: config.base_attention.queries.len(),
        n_keys: config.base_attention.keys.len(),
        solver: config
            .base_attention
            .attention_solver
            .method()
            .as_str()
            .to_string(),
        parallel: mode.is_parallel(),
        jobs: if mode.is_parallel() {
            jobs_label.to_string()
        } else {
            "sequential".to_string()
        },
        repeat,
        mean_ms,
        min_ms,
        max_ms,
        std_ms,
    })
}

pub fn export_profile_records(path: &Path, records: &[ProfileRecord]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "experiment",
        "dimension",
        "n_queries",
        "n_keys",
        "solver",
        "parallel",
        "jobs",
        "repeat",
        "mean_ms",
        "min_ms",
        "max_ms",
        "std_ms",
    ])?;

    for record in records {
        writer.write_record([
            record.experiment.clone(),
            record.dimension.to_string(),
            record.n_queries.to_string(),
            record.n_keys.to_string(),
            record.solver.clone(),
            record.parallel.to_string(),
            record.jobs.clone(),
            record.repeat.to_string(),
            format!("{:.6}", record.mean_ms),
            format!("{:.6}", record.min_ms),
            format!("{:.6}", record.max_ms),
            format!("{:.6}", record.std_ms),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    variance.sqrt()
}
