use crate::attention::{run_attention_demo_with_mode, run_multihead_attention_demo_with_mode};
use crate::config::{
    AttentionConfig, AttentionSolverMethod, ExperimentConfig, MultiHeadAttentionConfig,
    SolverMethod,
};
use crate::fenchel::{verify_fenchel_identity, FenchelIdentityInput};
use crate::functions::{build_conjugate_function, build_function};
use crate::io::csv_export::{
    export_attention_results, export_fenchel_checks, export_multihead_results, export_results,
    export_solver_comparison,
};
use crate::kernels::build_kernel;
use crate::optimization::averages::AverageKind;
use crate::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use crate::optimization::solver_comparison::compare_solvers_for_points_with_mode;
use crate::parallel::{self, ExecutionMode};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuración declarativa de una suite reproducible.
///
/// La suite no ejecuta un dashboard ni un servidor. Solo orquesta comandos CLI
/// equivalentes y deja artefactos verificables en un directorio de evidencia.
#[derive(Debug, Deserialize)]
pub struct SuiteConfig {
    pub name: Option<String>,
    pub compute: Option<SuiteComputeStep>,
    pub fenchel: Option<SuiteFenchelStep>,
    pub attention: Option<SuiteAttentionStep>,
    pub multihead_attention: Option<SuiteMultiheadAttentionStep>,
    pub compare_solvers: Option<SuiteCompareSolversStep>,
}

#[derive(Debug, Deserialize)]
pub struct SuiteComputeStep {
    pub config: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct SuiteFenchelStep {
    pub config: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct SuiteAttentionStep {
    pub config: PathBuf,
    pub solver: Option<AttentionSolverMethod>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct SuiteMultiheadAttentionStep {
    pub config: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct SuiteCompareSolversStep {
    pub config: PathBuf,
    pub solvers: Vec<SolverMethod>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct SuiteStepReport {
    name: &'static str,
    config_path: PathBuf,
    output_path: PathBuf,
    result_count: usize,
    status: &'static str,
}

/// Ejecuta una suite reproducible y genera un paquete de evidencia.
pub fn run_suite(suite_path: &Path, out_dir: &Path) -> Result<()> {
    run_suite_with_mode(suite_path, out_dir, ExecutionMode::Sequential)
}

/// Ejecuta una suite reproducible con paralelismo opcional en pasos internos independientes.
pub fn run_suite_with_mode(suite_path: &Path, out_dir: &Path, mode: ExecutionMode) -> Result<()> {
    let started_at = SystemTime::now();
    let suite_text = fs::read_to_string(suite_path)
        .with_context(|| format!("No se pudo leer la suite: {}", suite_path.display()))?;
    let suite: SuiteConfig = serde_yaml::from_str(&suite_text)
        .with_context(|| format!("No se pudo parsear suite YAML: {}", suite_path.display()))?;

    validate_suite(&suite)?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("No se pudo crear evidencia: {}", out_dir.display()))?;
    fs::copy(suite_path, out_dir.join("suite.yaml")).with_context(|| {
        format!(
            "No se pudo copiar la suite al paquete de evidencia: {}",
            suite_path.display()
        )
    })?;

    let suite_dir = suite_path.parent().unwrap_or_else(|| Path::new("."));
    let mut reports = Vec::new();
    let mut commands = Vec::new();

    if let Some(step) = suite.compute.as_ref() {
        let config_path = resolve_config_path(suite_dir, &step.config);
        let output_path =
            resolve_output_path(out_dir, step.output.as_deref(), "compute_results.csv");
        commands.push(format!(
            "kavg-lab compute --config {} --output {}{}",
            config_path.display(),
            output_path.display(),
            mode.cli_suffix()
        ));
        let result_count = run_compute_step(&config_path, &output_path, mode)?;
        reports.push(SuiteStepReport {
            name: "compute",
            config_path,
            output_path,
            result_count,
            status: "passed",
        });
    }

    if let Some(step) = suite.fenchel.as_ref() {
        let config_path = resolve_config_path(suite_dir, &step.config);
        let output_path =
            resolve_output_path(out_dir, step.output.as_deref(), "fenchel_results.csv");
        commands.push(format!(
            "kavg-lab verify-fenchel --config {} --output {}{}",
            config_path.display(),
            output_path.display(),
            mode.cli_suffix()
        ));
        let result_count = run_fenchel_step(&config_path, &output_path, mode)?;
        reports.push(SuiteStepReport {
            name: "verify-fenchel",
            config_path,
            output_path,
            result_count,
            status: "passed",
        });
    }

    if let Some(step) = suite.attention.as_ref() {
        let config_path = resolve_config_path(suite_dir, &step.config);
        let output_path =
            resolve_output_path(out_dir, step.output.as_deref(), "attention_results.csv");
        let mut command = format!("kavg-lab attention-demo --config {}", config_path.display());
        if let Some(solver) = step.solver.as_ref() {
            command.push_str(&format!(" --solver {}", solver.as_str()));
        }
        command.push_str(&format!(
            " --output {}{}",
            output_path.display(),
            mode.cli_suffix()
        ));
        commands.push(command);
        let result_count =
            run_attention_step(&config_path, step.solver.clone(), &output_path, mode)?;
        reports.push(SuiteStepReport {
            name: "attention-demo",
            config_path,
            output_path,
            result_count,
            status: "passed",
        });
    }

    if let Some(step) = suite.multihead_attention.as_ref() {
        let config_path = resolve_config_path(suite_dir, &step.config);
        let output_path = resolve_output_path(
            out_dir,
            step.output.as_deref(),
            "multihead_attention_results.csv",
        );
        commands.push(format!(
            "kavg-lab multihead-attention-demo --config {} --output {}{}",
            config_path.display(),
            output_path.display(),
            mode.cli_suffix()
        ));
        let result_count = run_multihead_attention_step(&config_path, &output_path, mode)?;
        reports.push(SuiteStepReport {
            name: "multihead-attention-demo",
            config_path,
            output_path,
            result_count,
            status: "passed",
        });
    }

    if let Some(step) = suite.compare_solvers.as_ref() {
        let config_path = resolve_config_path(suite_dir, &step.config);
        let output_path =
            resolve_output_path(out_dir, step.output.as_deref(), "solver_comparison.csv");
        let solver_list = step
            .solvers
            .iter()
            .map(|solver| solver.as_str())
            .collect::<Vec<_>>()
            .join(",");
        commands.push(format!(
            "kavg-lab compare-solvers --config {} --solvers {} --output {}{}",
            config_path.display(),
            solver_list,
            output_path.display(),
            mode.cli_suffix()
        ));
        let result_count =
            run_compare_solvers_step(&config_path, &step.solvers, &output_path, mode)?;
        reports.push(SuiteStepReport {
            name: "compare-solvers",
            config_path,
            output_path,
            result_count,
            status: "passed",
        });
    }

    let finished_at = SystemTime::now();
    write_commands_log(&out_dir.join("commands.log"), &commands)?;
    write_summary_json(&out_dir.join("summary.json"), &suite, &reports)?;
    write_manifest_json(
        &out_dir.join("manifest.json"),
        suite_path,
        &reports,
        started_at,
        finished_at,
    )?;
    write_readme(&out_dir.join("README.md"), &suite, &reports)?;

    println!("Suite reproducible ejecutada correctamente.");
    println!("Evidencia generada en: {}", out_dir.display());
    println!("Pasos ejecutados: {}", reports.len());
    println!("Manifiesto: {}", out_dir.join("manifest.json").display());
    println!("Resumen: {}", out_dir.join("summary.json").display());

    Ok(())
}

fn validate_suite(suite: &SuiteConfig) -> Result<()> {
    let count = usize::from(suite.compute.is_some())
        + usize::from(suite.fenchel.is_some())
        + usize::from(suite.attention.is_some())
        + usize::from(suite.multihead_attention.is_some())
        + usize::from(suite.compare_solvers.is_some());
    anyhow::ensure!(count > 0, "La suite debe declarar al menos un paso.");
    if let Some(step) = suite.compare_solvers.as_ref() {
        anyhow::ensure!(
            !step.solvers.is_empty(),
            "compare_solvers.solvers no puede estar vacío."
        );
    }
    Ok(())
}

fn run_compute_step(config_path: &Path, output_path: &Path, mode: ExecutionMode) -> Result<usize> {
    let experiment = ExperimentConfig::from_yaml_file(config_path)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;
    let results = parallel::map_indexed(&experiment.points, mode, |index, point| {
        let result = solve_kernel_average(KernelAverageInput {
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            kernel: kernel.as_ref(),
            lambda1: experiment.lambda1,
            x: point,
            solver: &experiment.solver,
            average_kind: AverageKind::Kernel,
        })?;
        Ok(result.with_index_and_point(index, point.clone()))
    })?;

    export_results(output_path, &results)?;
    Ok(results.len())
}

fn run_fenchel_step(config_path: &Path, output_path: &Path, mode: ExecutionMode) -> Result<usize> {
    let experiment = ExperimentConfig::from_yaml_file(config_path)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let f1_star = build_conjugate_function(&experiment.f1)?;
    let f2_star = build_conjugate_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;
    let results = parallel::map_indexed(&experiment.points, mode, |index, dual_point| {
        verify_fenchel_identity(FenchelIdentityInput {
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
    })?;

    export_fenchel_checks(output_path, &results)?;
    Ok(results.len())
}

fn run_attention_step(
    config_path: &Path,
    solver_override: Option<AttentionSolverMethod>,
    output_path: &Path,
    mode: ExecutionMode,
) -> Result<usize> {
    let mut config = AttentionConfig::from_yaml_file(config_path)?;
    if let Some(solver) = solver_override {
        config.attention_solver.method = Some(solver);
    }
    let results = run_attention_demo_with_mode(&config, mode)?;
    export_attention_results(output_path, &results)?;
    Ok(results.len())
}

fn run_multihead_attention_step(
    config_path: &Path,
    output_path: &Path,
    mode: ExecutionMode,
) -> Result<usize> {
    let config = MultiHeadAttentionConfig::from_yaml_file(config_path)?;
    let results = run_multihead_attention_demo_with_mode(&config, mode)?;
    export_multihead_results(output_path, &results)?;
    Ok(results.len())
}

fn run_compare_solvers_step(
    config_path: &Path,
    solvers: &[SolverMethod],
    output_path: &Path,
    mode: ExecutionMode,
) -> Result<usize> {
    let experiment = ExperimentConfig::from_yaml_file(config_path)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;
    let results = compare_solvers_for_points_with_mode(
        &experiment.points,
        f1.as_ref(),
        f2.as_ref(),
        kernel.as_ref(),
        experiment.lambda1,
        &experiment.solver,
        solvers,
        mode,
    )?;
    export_solver_comparison(output_path, &results)?;
    Ok(results.len())
}

fn resolve_config_path(suite_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.exists() {
        path.to_path_buf()
    } else {
        suite_dir.join(path)
    }
}

fn resolve_output_path(out_dir: &Path, configured: Option<&Path>, default_name: &str) -> PathBuf {
    match configured {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => out_dir.join(path),
        None => out_dir.join(default_name),
    }
}

fn write_commands_log(path: &Path, commands: &[String]) -> Result<()> {
    let mut text = String::new();
    for command in commands {
        text.push_str(command);
        text.push('\n');
    }
    fs::write(path, text)
        .with_context(|| format!("No se pudo escribir commands.log: {}", path.display()))
}

fn write_summary_json(path: &Path, suite: &SuiteConfig, reports: &[SuiteStepReport]) -> Result<()> {
    let mut text = String::new();
    text.push_str("{\n");
    text.push_str(&format!(
        "  \"schema_version\": \"1.0\",\n  \"suite_name\": {},\n  \"status\": \"passed\",\n  \"step_count\": {},\n  \"steps\": [\n",
        json_string(suite.name.as_deref().unwrap_or("unnamed-suite")),
        reports.len()
    ));
    for (index, report) in reports.iter().enumerate() {
        text.push_str("    {\n");
        text.push_str(&format!(
            "      \"name\": {},\n      \"config_path\": {},\n      \"output_path\": {},\n      \"result_count\": {},\n      \"status\": {}\n",
            json_string(report.name),
            json_string(&report.config_path.display().to_string()),
            json_string(&report.output_path.display().to_string()),
            report.result_count,
            json_string(report.status)
        ));
        text.push_str("    }");
        if index + 1 != reports.len() {
            text.push(',');
        }
        text.push('\n');
    }
    text.push_str("  ]\n}\n");
    fs::write(path, text)
        .with_context(|| format!("No se pudo escribir summary.json: {}", path.display()))
}

fn write_manifest_json(
    path: &Path,
    suite_path: &Path,
    reports: &[SuiteStepReport],
    started_at: SystemTime,
    finished_at: SystemTime,
) -> Result<()> {
    let elapsed_ms = finished_at
        .duration_since(started_at)
        .unwrap_or(Duration::from_millis(0))
        .as_millis();
    let rustc = command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_string());
    let git_commit = command_output("git", &["rev-parse", "--short", "HEAD"])
        .unwrap_or_else(|| option_env!("GIT_COMMIT").unwrap_or("unknown").to_string());
    let config_hash = fnv1a64_file(suite_path)?;

    let started_ms = unix_millis(started_at);
    let finished_ms = unix_millis(finished_at);
    let hash_hex = format!("{config_hash:016x}");
    let text = format!(
        "{{\n  \"schema_version\": \"1.0\",\n  \"tool\": {{\n    \"name\": \"kavg-lab\",\n    \"version\": {}\n  }},\n  \"command\": \"run-suite\",\n  \"rustc\": {},\n  \"started_at\": {},\n  \"finished_at\": {},\n  \"started_at_unix_ms\": {},\n  \"finished_at_unix_ms\": {},\n  \"elapsed_ms\": {},\n  \"suite_path\": {},\n  \"config_hash\": {},\n  \"config_hash_fnv1a64\": {},\n  \"git_commit\": {},\n  \"step_count\": {},\n  \"status\": \"passed\"\n}}\n",
        json_string(env!("CARGO_PKG_VERSION")),
        json_string(&rustc),
        json_string(&started_ms.to_string()),
        json_string(&finished_ms.to_string()),
        started_ms,
        finished_ms,
        elapsed_ms,
        json_string(&suite_path.display().to_string()),
        json_string(&hash_hex),
        json_string(&hash_hex),
        json_string(&git_commit),
        reports.len()
    );
    fs::write(path, text)
        .with_context(|| format!("No se pudo escribir manifest.json: {}", path.display()))
}

fn write_readme(path: &Path, suite: &SuiteConfig, reports: &[SuiteStepReport]) -> Result<()> {
    let suite_name = suite.name.as_deref().unwrap_or("unnamed-suite");
    let mut text = String::new();
    text.push_str("# KAvgLab Evidence Pack\n\n");
    text.push_str(&format!("Suite: `{suite_name}`\n\n"));
    text.push_str("Este directorio fue generado por `kavg-lab run-suite`. Contiene entradas, comandos equivalentes, resultados tabulares y metadatos de trazabilidad.\n\n");
    text.push_str("## Archivos\n\n");
    text.push_str(
        "- `manifest.json`: metadatos de ejecución, versión, hash de suite, rustc y commit.\n",
    );
    text.push_str("- `suite.yaml`: copia exacta de la suite usada.\n");
    text.push_str("- `commands.log`: comandos CLI equivalentes.\n");
    text.push_str("- `summary.json`: resumen estructurado de pasos y conteos.\n");
    for report in reports {
        text.push_str(&format!(
            "- `{}`: resultados del paso `{}`.\n",
            report
                .output_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("resultado.csv"),
            report.name
        ));
    }
    fs::write(path, text).with_context(|| {
        format!(
            "No se pudo escribir README de evidencia: {}",
            path.display()
        )
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn fnv1a64_file(path: &Path) -> Result<u64> {
    let bytes = fs::read(path)
        .with_context(|| format!("No se pudo leer archivo para hash: {}", path.display()))?;
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

fn unix_millis(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_millis(0))
        .as_millis()
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}
