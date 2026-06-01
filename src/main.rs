mod cli;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use kavg_lab::attention::{run_agent_sweep, run_attention_demo, run_multihead_attention_demo};
use kavg_lab::config::{
    AgentSweepConfig, AttentionConfig, AttentionSolverMethod, ExperimentConfig,
    MultiHeadAttentionConfig, SolverMethod,
};
use kavg_lab::fenchel::{verify_fenchel_identity, FenchelIdentityInput};
use kavg_lab::functions::{build_conjugate_function, build_function};
use kavg_lab::io::csv_export::{
    export_agent_sweep_results, export_attention_results, export_comparison, export_fenchel_checks,
    export_multihead_results, export_results, export_solver_comparison,
};
use kavg_lab::io::json_export::{
    export_compute_results_json, export_execution_manifest, ExecutionManifest,
};
use kavg_lab::kernels::build_kernel;
use kavg_lab::optimization::averages::AverageKind;
use kavg_lab::optimization::comparison::compare_averages;
use kavg_lab::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use kavg_lab::optimization::solver_comparison::compare_solvers_for_points;
use kavg_lab::suite::run_suite;
use std::time::SystemTime;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compute {
            config,
            output,
            json_output,
            manifest,
        } => run_compute(&config, output, json_output, manifest)?,
        Commands::Compare { config, output } => run_compare(&config, output)?,
        Commands::CompareSolvers {
            config,
            solvers,
            output,
        } => run_compare_solvers(&config, solvers, output)?,
        Commands::VerifyFenchel { config, output } => run_verify_fenchel(&config, output)?,
        Commands::AttentionDemo {
            config,
            solver,
            output,
        } => run_attention_demo_command(&config, solver, output)?,
        Commands::MultiheadAttentionDemo { config, output } => {
            run_multihead_attention_demo_command(&config, output)?
        }
        Commands::RunSuite { suite, out } => run_suite(&suite, &out)?,
        Commands::AgentSweep { config, output } => run_agent_sweep_command(&config, output)?,
    }

    Ok(())
}

fn run_compute(
    config: &std::path::Path,
    output: Option<std::path::PathBuf>,
    json_output: Option<std::path::PathBuf>,
    manifest: Option<std::path::PathBuf>,
) -> Result<()> {
    let started_at = SystemTime::now();
    let experiment = ExperimentConfig::from_yaml_file(config)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;

    let mut results = Vec::new();
    let lambda1 = experiment.lambda1;
    let lambda2 = 1.0 - lambda1;

    println!("Experimento KAvgLab - Calcula");
    println!("f1: {}", f1.name());
    println!("f2: {}", f2.name());
    println!("kernel: {}", kernel.name());
    println!("solver: {}", experiment.solver.method().as_str());
    println!("lambda1: {:.6}, lambda2: {:.6}", lambda1, lambda2);
    println!();

    for (index, point) in experiment.points.iter().enumerate() {
        let input = KernelAverageInput {
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            kernel: kernel.as_ref(),
            lambda1,
            x: point,
            solver: &experiment.solver,
            average_kind: AverageKind::Kernel,
        };

        let result = solve_kernel_average(input)?;

        println!("Punto #{index}");
        println!("  x                 = {:?}", point);
        println!("  P(x)              = {:.10}", result.value);
        println!("  y1                = {:?}", result.y1);
        println!("  y2                = {:?}", result.y2);
        println!("  penalización base = {:.10}", result.raw_penalty);
        println!("  penalización peso = {:.10}", result.weighted_penalty);
        println!("  iteraciones       = {}", result.iterations);
        println!("  métrica solver    = {:.6e}", result.solver_metric);
        println!();

        results.push(result.with_index_and_point(index, point.clone()));
    }

    if let Some(path) = output.as_ref() {
        export_results(path, &results)?;
        println!("Resultados CSV exportados a: {}", path.display());
    }

    if let Some(path) = json_output.as_ref() {
        export_compute_results_json(path, &results)?;
        println!("Resultados JSON exportados a: {}", path.display());
    }

    if let Some(path) = manifest.as_ref() {
        let finished_at = SystemTime::now();
        export_execution_manifest(
            path,
            &ExecutionManifest {
                command: "compute",
                config_path: config,
                csv_output: output.as_deref(),
                json_output: json_output.as_deref(),
                started_at,
                finished_at,
                result_count: results.len(),
                status: "passed",
            },
        )?;
        println!("Manifiesto exportado a: {}", path.display());
    }

    Ok(())
}

fn run_compare(config: &std::path::Path, output: Option<std::path::PathBuf>) -> Result<()> {
    let experiment = ExperimentConfig::from_yaml_file(config)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;

    let mut results = Vec::new();
    let lambda1 = experiment.lambda1;
    let lambda2 = 1.0 - lambda1;

    println!("Experimento KAvgLab - Compara");
    println!("f1: {}", f1.name());
    println!("f2: {}", f2.name());
    println!("kernel para proximal: {}", kernel.name());
    println!("solver: {}", experiment.solver.method().as_str());
    println!("lambda1: {:.6}, lambda2: {:.6}", lambda1, lambda2);
    println!();

    for (index, point) in experiment.points.iter().enumerate() {
        let result = compare_averages(
            index,
            point,
            f1.as_ref(),
            f2.as_ref(),
            kernel.as_ref(),
            lambda1,
            &experiment.solver,
        )?;

        println!("Punto #{index}");
        println!("  x                         = {:?}", point);
        println!(
            "  promedio aritmético       = {:.10}",
            result.arithmetic_value
        );
        println!(
            "  promedio epigráfico       = {:.10}",
            result.epigraphical.value
        );
        println!(
            "  proximal average          = {:.10}",
            result.proximal.value
        );
        println!(
            "  proximal - epigráfico     = {:.10}",
            result.proximal.value - result.epigraphical.value
        );
        println!(
            "  aritmético - proximal     = {:.10}",
            result.arithmetic_value - result.proximal.value
        );
        println!("  y1 proximal               = {:?}", result.proximal.y1);
        println!("  y2 proximal               = {:?}", result.proximal.y2);
        println!();

        results.push(result);
    }

    if let Some(path) = output {
        export_comparison(&path, &results)?;
        println!("Comparación exportada a: {}", path.display());
    }

    Ok(())
}

fn run_compare_solvers(
    config: &std::path::Path,
    solvers: Vec<String>,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let experiment = ExperimentConfig::from_yaml_file(config)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;
    let methods = parse_solver_methods(&solvers)?;

    println!("Experimento KAvgLab - Comparación de solvers");
    println!("f1: {}", f1.name());
    println!("f2: {}", f2.name());
    println!("kernel: {}", kernel.name());
    println!("puntos: {}", experiment.points.len());
    println!(
        "solvers: {}",
        methods
            .iter()
            .map(|method| method.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    let results = compare_solvers_for_points(
        &experiment.points,
        f1.as_ref(),
        f2.as_ref(),
        kernel.as_ref(),
        experiment.lambda1,
        &experiment.solver,
        &methods,
    );

    for result in &results {
        if result.status == "ok" {
            println!(
                "solver={} punto={} valor={:.10} iteraciones={} métrica={:.6e}",
                result.solver_method,
                result.index,
                result.value.unwrap_or_default(),
                result.iterations.unwrap_or_default(),
                result.solver_metric.unwrap_or_default()
            );
        } else {
            println!(
                "solver={} punto={} ERROR: {}",
                result.solver_method,
                result.index,
                result.error.clone().unwrap_or_default()
            );
        }
    }

    if let Some(path) = output {
        export_solver_comparison(&path, &results)?;
        println!("Comparación de solvers exportada a: {}", path.display());
    }

    Ok(())
}

fn parse_solver_methods(values: &[String]) -> Result<Vec<SolverMethod>> {
    anyhow::ensure!(
        !values.is_empty(),
        "Debe indicar al menos un solver con --solvers."
    );
    values.iter().map(|value| value.parse()).collect()
}

fn run_verify_fenchel(config: &std::path::Path, output: Option<std::path::PathBuf>) -> Result<()> {
    let experiment = ExperimentConfig::from_yaml_file(config)?;
    let f1 = build_function(&experiment.f1)?;
    let f2 = build_function(&experiment.f2)?;
    let f1_star = build_conjugate_function(&experiment.f1)?;
    let f2_star = build_conjugate_function(&experiment.f2)?;
    let kernel = build_kernel(&experiment.kernel)?;

    let mut results = Vec::new();
    let lambda1 = experiment.lambda1;
    let lambda2 = 1.0 - lambda1;

    println!("Experimento KAvgLab - Verifica Fenchel");
    println!("f1: {}", f1.name());
    println!("f2: {}", f2.name());
    println!("f1*: {}", f1_star.name());
    println!("f2*: {}", f2_star.name());
    println!("kernel: {}", kernel.name());
    println!("solver: {}", experiment.solver.method().as_str());
    println!("lambda1: {:.6}, lambda2: {:.6}", lambda1, lambda2);
    println!("Nota: en verify-fenchel, points se interpreta como lista de puntos duales s.");
    println!();

    for (index, dual_point) in experiment.points.iter().enumerate() {
        let result = verify_fenchel_identity(FenchelIdentityInput {
            index,
            dual_point,
            f1: f1.as_ref(),
            f2: f2.as_ref(),
            f1_star: f1_star.as_ref(),
            f2_star: f2_star.as_ref(),
            kernel: kernel.as_ref(),
            lambda1,
            solver: &experiment.solver,
        })?;

        println!("Punto dual #{index}");
        println!("  s                                = {:?}", dual_point);
        println!(
            "  lado izquierdo aproximado         = {:.10}",
            result.left_approx
        );
        println!(
            "  lado derecho por conjugados       = {:.10}",
            result.right_value
        );
        println!(
            "  error absoluto                    = {:.6e}",
            result.absolute_error
        );
        println!(
            "  error relativo                    = {:.6e}",
            result.relative_error
        );
        println!(
            "  estado                            = {}",
            if result.passed { "PASSED" } else { "FAILED" }
        );
        println!(
            "  argmax primal aproximado          = {:?}",
            result.primal_argmax
        );
        println!(
            "  y1 lado derecho                   = {:?}",
            result.right_y1
        );
        println!(
            "  y2 lado derecho                   = {:?}",
            result.right_y2
        );
        println!();

        results.push(result);
    }

    if let Some(path) = output {
        export_fenchel_checks(&path, &results)?;
        println!("Verificación Fenchel exportada a: {}", path.display());
    }

    Ok(())
}

fn run_attention_demo_command(
    config: &std::path::Path,
    solver_override: Option<AttentionSolverMethod>,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let mut experiment = AttentionConfig::from_yaml_file(config)?;
    if let Some(solver) = solver_override {
        experiment.attention_solver.method = Some(solver);
    }
    let results = run_attention_demo(&experiment)?;

    println!("Experimento KAvgLab - Demostracion de atencion");
    println!("dimension de queries/keys: {}", experiment.dimension);
    println!("temperature: {:.6}", experiment.temperature);
    println!("kernel_gamma: {:.6}", experiment.kernel_gamma);
    println!(
        "attention_solver: {}",
        experiment.attention_solver.method().as_str()
    );
    println!(
        "queries: {}, keys/values: {}",
        experiment.queries.len(),
        experiment.keys.len()
    );
    println!();

    for result in &results {
        println!("Query #{}", result.index);
        println!("  q                          = {:?}", result.query);
        println!("  scores                     = {:?}", result.scores);
        println!(
            "  atención softmax           = {:?}",
            result.standard_weights
        );
        println!(
            "  atención regularizada      = {:?}",
            result.regularized_weights
        );
        println!(
            "  salida softmax             = {:?}",
            result.standard_output
        );
        println!(
            "  salida regularizada        = {:?}",
            result.regularized_output
        );
        println!(
            "  distancia L1 pesos         = {:.10}",
            result.weight_l1_distance
        );
        println!(
            "  distancia L2 pesos         = {:.10}",
            result.weight_l2_distance
        );
        println!(
            "  distancia L2 salidas       = {:.10}",
            result.output_l2_distance
        );
        println!(
            "  entropía softmax           = {:.10}",
            result.standard_entropy
        );
        println!(
            "  entropía regularizada      = {:.10}",
            result.regularized_entropy
        );
        println!(
            "  KL(reg || softmax)         = {:.10}",
            result.kl_regularized_to_softmax
        );
        println!(
            "  KL(reg || prior)           = {:.10}",
            result.kl_regularized_to_prior
        );
        println!(
            "  JS(softmax, reg)           = {:.10}",
            result.js_softmax_regularized
        );
        println!(
            "  tokens efectivos reg       = {:.10}",
            result.effective_tokens_regularized
        );
        println!(
            "  masa top-1 reg             = {:.10}",
            result.regularized_top1_mass
        );
        println!(
            "  masa top-k reg             = {:.10}",
            result.regularized_topk_mass
        );
        println!("  iteraciones                = {}", result.iterations);
        println!(
            "  métrica solver             = {:.6e}",
            result.solver_metric
        );
        println!();
    }

    if let Some(path) = output {
        export_attention_results(&path, &results)?;
        println!("Resultados de atención exportados a: {}", path.display());
    }

    Ok(())
}

fn run_multihead_attention_demo_command(
    config: &std::path::Path,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let experiment = MultiHeadAttentionConfig::from_yaml_file(config)?;
    let results = run_multihead_attention_demo(&experiment)?;

    println!("Experimento KAvgLab - Demostracion Multi-head");
    println!("dimension de queries/keys: {}", experiment.dimension);
    println!("cabeceras: {}", experiment.heads.len());
    println!(
        "queries: {}, keys/values: {}",
        experiment.queries.len(),
        experiment.keys.len()
    );
    println!(
        "solver por defecto: {}",
        experiment.default_attention_solver.method().as_str()
    );
    println!();

    for result in &results {
        println!("Query #{}", result.query_index);
        println!("  q                          = {:?}", result.query);
        println!(
            "  salida agregada            = {:?}",
            result.aggregated_output
        );
        println!(
            "  entropía promedio          = {:.10}",
            result.average_entropy
        );
        println!(
            "  diversidad L1 media        = {:.10}",
            result.mean_pairwise_l1
        );
        println!(
            "  diversidad L2 media        = {:.10}",
            result.mean_pairwise_l2
        );
        println!(
            "  diversidad JS media        = {:.10}",
            result.mean_pairwise_js
        );
        for head in &result.heads {
            println!(
                "    {}: gamma={:.4}, temp={:.4}, entropía={:.6}, pesos={:?}",
                head.head_name,
                head.kernel_gamma,
                head.temperature,
                head.entropy,
                head.regularized_weights
            );
        }
        println!();
    }

    if let Some(path) = output {
        export_multihead_results(&path, &results)?;
        println!("Resultados multi-head exportados a: {}", path.display());
    }

    Ok(())
}

fn run_agent_sweep_command(
    config: &std::path::Path,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let experiment = AgentSweepConfig::from_yaml_file(config)?;
    let results = run_agent_sweep(&experiment)?;

    println!("Experimento KAvgLab - Agent Sweep");
    println!("objetivo: {}", experiment.objective.as_str());
    println!("candidatos devueltos: {}", results.len());
    println!();

    for result in &results {
        println!("Rank #{}", result.rank);
        println!("  gamma                         = {:.6}", result.gamma);
        println!(
            "  temperature                   = {:.6}",
            result.temperature
        );
        println!("  prior                         = {}", result.prior_name);
        println!("  score                         = {:.10}", result.score);
        println!(
            "  entropía media reg            = {:.10}",
            result.mean_regularized_entropy
        );
        println!(
            "  distancia media al prior      = {:.10}",
            result.mean_distance_to_prior
        );
        println!(
            "  diferencia media vs softmax   = {:.10}",
            result.mean_difference_from_softmax
        );
        println!(
            "  desplazamiento medio salida   = {:.10}",
            result.mean_output_shift
        );
        println!(
            "  JS medio softmax/reg          = {:.10}",
            result.mean_js_softmax_regularized
        );
        println!();
    }

    if let Some(path) = output {
        export_agent_sweep_results(&path, &results)?;
        println!("Barrido agente exportado a: {}", path.display());
    }

    Ok(())
}
