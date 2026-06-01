use crate::math::format_vector;
use crate::optimization::comparison::ComparisonResult;
use crate::optimization::kernel_average::KernelAverageResult;
use anyhow::Result;
use std::path::Path;

/// Exporta los resultados de `compute` a CSV.
pub fn export_results(path: &Path, results: &[KernelAverageResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "index",
        "point",
        "average_kind",
        "value",
        "raw_penalty",
        "weighted_penalty",
        "iterations",
        "solver_method",
        "solver_metric",
        "y1",
        "y2",
    ])?;

    for result in results {
        writer.write_record([
            result.index.unwrap_or_default().to_string(),
            result.point_csv(),
            result.average_kind.clone(),
            format!("{:.12}", result.value),
            format!("{:.12}", result.raw_penalty),
            format!("{:.12}", result.weighted_penalty),
            result.iterations.to_string(),
            result.solver_method.clone(),
            format!("{:.12e}", result.solver_metric),
            result.y1_csv(),
            result.y2_csv(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Exporta la comparación entre promedio aritmético, epigráfico y proximal average.
pub fn export_comparison(path: &Path, results: &[ComparisonResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "index",
        "point",
        "arithmetic_value",
        "epigraphical_value",
        "proximal_value",
        "proximal_minus_epigraphical",
        "arithmetic_minus_proximal",
        "epigraphical_iterations",
        "proximal_iterations",
        "solver_method",
        "epigraphical_y1",
        "epigraphical_y2",
        "proximal_y1",
        "proximal_y2",
    ])?;

    for result in results {
        writer.write_record([
            result.index.to_string(),
            format_vector(&result.point),
            format!("{:.12}", result.arithmetic_value),
            format!("{:.12}", result.epigraphical.value),
            format!("{:.12}", result.proximal.value),
            format!("{:.12}", result.proximal.value - result.epigraphical.value),
            format!("{:.12}", result.arithmetic_value - result.proximal.value),
            result.epigraphical.iterations.to_string(),
            result.proximal.iterations.to_string(),
            result.proximal.solver_method.clone(),
            result.epigraphical.y1_csv(),
            result.epigraphical.y2_csv(),
            result.proximal.y1_csv(),
            result.proximal.y2_csv(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

use crate::fenchel::FenchelCheckResult;

/// Exporta la verificación numérica de la identidad de Fenchel.
pub fn export_fenchel_checks(path: &Path, results: &[FenchelCheckResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "index",
        "dual_point",
        "left_approx",
        "right_value",
        "absolute_error",
        "relative_error",
        "passed",
        "outer_iterations",
        "outer_metric",
        "primal_argmax",
        "right_y1",
        "right_y2",
    ])?;

    for result in results {
        writer.write_record([
            result.index.to_string(),
            result.dual_point_csv(),
            format!("{:.12}", result.left_approx),
            format!("{:.12}", result.right_value),
            format!("{:.12e}", result.absolute_error),
            format!("{:.12e}", result.relative_error),
            result.passed.to_string(),
            result.outer_iterations.to_string(),
            format!("{:.12e}", result.outer_metric),
            result.primal_argmax_csv(),
            result.right_y1_csv(),
            result.right_y2_csv(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

use crate::attention::{AgentSweepResult, AttentionResult, MultiHeadAttentionResult};

/// Exporta la comparación entre atención softmax y atención regularizada por kernel.
pub fn export_attention_results(path: &Path, results: &[AttentionResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "index",
        "solver_method",
        "attention_rule",
        "query",
        "scores",
        "masked_scores",
        "prior",
        "standard_weights",
        "regularized_weights",
        "standard_output",
        "regularized_output",
        "weight_l1_distance",
        "weight_l2_distance",
        "output_l2_distance",
        "standard_entropy",
        "regularized_entropy",
        "kl_regularized_to_softmax",
        "kl_regularized_to_prior",
        "js_softmax_regularized",
        "effective_tokens_standard",
        "effective_tokens_regularized",
        "standard_top1_mass",
        "regularized_top1_mass",
        "standard_topk_mass",
        "regularized_topk_mass",
        "iterations",
        "solver_metric",
    ])?;

    for result in results {
        writer.write_record([
            result.index.to_string(),
            result.solver_method.clone(),
            result.attention_rule.clone(),
            result.query_csv(),
            result.scores_csv(),
            result.masked_scores_csv(),
            result.prior_csv(),
            result.standard_weights_csv(),
            result.regularized_weights_csv(),
            result.standard_output_csv(),
            result.regularized_output_csv(),
            format!("{:.12}", result.weight_l1_distance),
            format!("{:.12}", result.weight_l2_distance),
            format!("{:.12}", result.output_l2_distance),
            format!("{:.12}", result.standard_entropy),
            format!("{:.12}", result.regularized_entropy),
            format!("{:.12}", result.kl_regularized_to_softmax),
            format!("{:.12}", result.kl_regularized_to_prior),
            format!("{:.12}", result.js_softmax_regularized),
            format!("{:.12}", result.effective_tokens_standard),
            format!("{:.12}", result.effective_tokens_regularized),
            format!("{:.12}", result.standard_top1_mass),
            format!("{:.12}", result.regularized_top1_mass),
            format!("{:.12}", result.standard_topk_mass),
            format!("{:.12}", result.regularized_topk_mass),
            result.iterations.to_string(),
            format!("{:.12e}", result.solver_metric),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Exporta la demostración multi-head en una fila por query, con pesos/salidas por cabecera compactados.
pub fn export_multihead_results(path: &Path, results: &[MultiHeadAttentionResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "query_index",
        "query",
        "aggregated_output",
        "average_entropy",
        "mean_pairwise_l1",
        "mean_pairwise_l2",
        "mean_pairwise_js",
        "head_names",
        "head_weights",
        "head_outputs",
        "head_entropies",
    ])?;

    for result in results {
        writer.write_record([
            result.query_index.to_string(),
            result.query_csv(),
            result.aggregated_output_csv(),
            format!("{:.12}", result.average_entropy),
            format!("{:.12}", result.mean_pairwise_l1),
            format!("{:.12}", result.mean_pairwise_l2),
            format!("{:.12}", result.mean_pairwise_js),
            result.head_names_csv(),
            result.head_weights_csv(),
            result.head_outputs_csv(),
            result.head_entropies_csv(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Exporta el barrido tipo agente experimental.
pub fn export_agent_sweep_results(path: &Path, results: &[AgentSweepResult]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "rank",
        "gamma",
        "temperature",
        "prior_name",
        "objective",
        "score",
        "mean_regularized_entropy",
        "mean_distance_to_prior",
        "mean_difference_from_softmax",
        "mean_output_shift",
        "mean_js_softmax_regularized",
        "mean_effective_tokens",
    ])?;

    for result in results {
        writer.write_record([
            result.rank.to_string(),
            format!("{:.12}", result.gamma),
            format!("{:.12}", result.temperature),
            result.prior_name.clone(),
            result.objective.clone(),
            format!("{:.12}", result.score),
            format!("{:.12}", result.mean_regularized_entropy),
            format!("{:.12}", result.mean_distance_to_prior),
            format!("{:.12}", result.mean_difference_from_softmax),
            format!("{:.12}", result.mean_output_shift),
            format!("{:.12}", result.mean_js_softmax_regularized),
            format!("{:.12}", result.mean_effective_tokens),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

use crate::optimization::solver_comparison::SolverComparisonRow;

/// Exporta la comparación CLI entre varios solvers sobre los mismos puntos.
pub fn export_solver_comparison(path: &Path, results: &[SolverComparisonRow]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "solver_method",
        "index",
        "point",
        "status",
        "value",
        "iterations",
        "solver_metric",
        "raw_penalty",
        "weighted_penalty",
        "y1",
        "y2",
        "error",
    ])?;

    for result in results {
        writer.write_record([
            result.solver_method.clone(),
            result.index.to_string(),
            result.point_csv(),
            result.status.clone(),
            result
                .value
                .map(|v| format!("{:.12}", v))
                .unwrap_or_default(),
            result.iterations.map(|v| v.to_string()).unwrap_or_default(),
            result
                .solver_metric
                .map(|v| format!("{:.12e}", v))
                .unwrap_or_default(),
            result
                .raw_penalty
                .map(|v| format!("{:.12}", v))
                .unwrap_or_default(),
            result
                .weighted_penalty
                .map(|v| format!("{:.12}", v))
                .unwrap_or_default(),
            result.y1_csv(),
            result.y2_csv(),
            result.error.clone().unwrap_or_default(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
